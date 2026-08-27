"""Readable Python lifecycle around the compiled numerical step."""

import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from tqdm import tqdm

from Common.trainer.training_result import TrainingResult
from NCA.trainer.checkpointing import BestCheckpoint
from NCA.trainer.instrumentation import (
    call_and_time,
    compile_and_time,
    maybe_save_device_memory_profile,
    start_trace,
    stop_trace,
)
from NCA.trainer.pool import PoolAdmissionController, TimePoolAdmissionController
from NCA.trainer.state import TrainState


class RuntimeMetrics:
    def __init__(self, compile_seconds):
        self.values = {
            "jit_compile_seconds": compile_seconds,
            "first_execution_seconds": None,
            "step_compute_seconds": None,
            "step_compute_per_second": None,
            "steady_step_mean_seconds": None,
            "steady_step_mean_per_second": None,
            "steady_step_count": 0,
            "iteration_excluding_logging_seconds": None,
            "steady_iteration_mean_seconds": None,
            "steady_iteration_mean_per_second": None,
            "steady_iteration_count": 0,
        }
        self._step_total = 0.0
        self._iteration_total = 0.0

    def record_step(self, seconds):
        values = self.values
        if values["first_execution_seconds"] is None:
            values["first_execution_seconds"] = seconds
        else:
            self._step_total += seconds
            values["steady_step_count"] += 1
            mean = self._step_total / values["steady_step_count"]
            values["steady_step_mean_seconds"] = mean
            values["steady_step_mean_per_second"] = 1.0 / max(mean, 1e-12)
        values["step_compute_seconds"] = seconds
        values["step_compute_per_second"] = 1.0 / max(seconds, 1e-12)

    def record_iteration(self, iteration, seconds):
        self.values["iteration_excluding_logging_seconds"] = seconds
        if iteration == 0:
            return
        self._iteration_total += seconds
        self.values["steady_iteration_count"] += 1
        mean = self._iteration_total / self.values["steady_iteration_count"]
        self.values["steady_iteration_mean_seconds"] = mean
        self.values["steady_iteration_mean_per_second"] = 1.0 / max(mean, 1e-12)

    def as_log_dict(self):
        return {
            f"runtime/{name}": value
            for name, value in self.values.items()
            if value is not None
        }


def _divergence_code(loss, states):
    """Classify divergence without confusing a bad state for a bad loss.

    The objective is evaluated from ``states``, so state finiteness must be
    checked first. Otherwise a NaN produced by the rollout is always reported
    as a loss NaN and the useful distinction is lost.
    """
    if any(
        bool(jax.device_get(jnp.any(~jnp.isfinite(value))))
        for value in jtu.tree_leaves(states)
    ):
        return 2
    if bool(jax.device_get(~jnp.isfinite(loss))):
        return 1
    if abs(float(jax.device_get(loss))) > 1e16:
        return 3
    return 0


def _array_diagnostic(value):
    value = jnp.asarray(value)
    finite = jnp.isfinite(value)
    finite_count = int(jax.device_get(jnp.sum(finite)))
    total_count = value.size
    if finite_count:
        finite_values = jnp.where(finite, value, 0)
        max_abs = float(jax.device_get(jnp.max(jnp.abs(finite_values))))
    else:
        max_abs = float("nan")
    return {
        "shape": tuple(value.shape),
        "finite": f"{finite_count}/{total_count}",
        "nan": int(jax.device_get(jnp.sum(jnp.isnan(value)))),
        "inf": int(jax.device_get(jnp.sum(jnp.isinf(value)))),
        "finite_max_abs": max_abs,
    }


def _merge_advanced_states(advanced, previous, source_admitted):
    """Keep a shifted prediction only when its source transition was admitted."""
    source_admitted = jnp.asarray(source_admitted, dtype=jnp.bool_)
    destination_admitted = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.bool_), source_admitted]
    )

    def merge(advanced_value, previous_value):
        time_axis = advanced_value.ndim - 4
        if (
            time_axis < 0
            or advanced_value.shape[time_axis] != len(destination_admitted)
        ):
            raise ValueError(
                "Pool state time dimension is incompatible with admission mask: "
                f"shape {advanced_value.shape}, mask length {len(destination_admitted)}"
            )
        mask_shape = (
            (1,) * time_axis
            + (len(destination_admitted),)
            + (1,) * (advanced_value.ndim - time_axis - 1)
        )
        return jnp.where(
            destination_admitted.reshape(mask_shape),
            advanced_value,
            previous_value,
        )

    return jtu.tree_map(merge, advanced, previous)


def _update_training_pool(
    state,
    states_before_step,
    targets_before_step,
    source_admitted,
    iteration,
    execution,
):
    """Advance admitted transitions and discard every rejected rollout."""
    admission_mask = None
    if isinstance(source_admitted, bool):
        admitted = source_admitted
    else:
        admission_mask = tuple(bool(value) for value in source_admitted)
        admitted = any(admission_mask)
    if not admitted:
        return state._replace(
            states=states_before_step,
            targets=targets_before_step,
        )

    next_key, augment_key = execution.split_key(state.key)
    states, targets = execution.apply_advance_pool(
        state.states, state.targets, iteration, augment_key
    )
    if admission_mask is not None and not all(admission_mask):
        states = _merge_advanced_states(
            states, states_before_step, admission_mask
        )
    states = jtu.tree_map(state.model.prepare_pool_state, states)
    return state._replace(states=states, targets=targets, key=next_key)


def _report_divergence(output, state, iteration):
    """Print enough numerical context to locate the first failing subsystem."""
    print(f"Divergence diagnostics at step {iteration}:")
    print(f"  loss: {_array_diagnostic(output.loss)}")

    level_channels = getattr(state.model, "LEVEL_CHANNELS", None)
    for batch_index, batch_states in enumerate(jtu.tree_leaves(state.states)):
        if level_channels is None:
            print(f"  state[{batch_index}]: {_array_diagnostic(batch_states)}")
            continue
        child = batch_states[..., :level_channels, :, :]
        parent = batch_states[..., level_channels:, :, :]
        print(f"  state[{batch_index}].child: {_array_diagnostic(child)}")
        print(f"  state[{batch_index}].parent: {_array_diagnostic(parent)}")

    for name, value in output.metrics.items():
        if name == "states":
            continue
        leaves = jtu.tree_leaves(value)
        if leaves and all(hasattr(leaf, "shape") for leaf in leaves):
            bad = any(
                bool(jax.device_get(jnp.any(~jnp.isfinite(leaf))))
                for leaf in leaves
            )
            if bad or name.startswith(("loss", "boundary")):
                summaries = [_array_diagnostic(leaf) for leaf in leaves]
                print(f"  metric.{name}: {summaries}")

    bad_parameter_leaves = []
    for path, value in jtu.tree_leaves_with_path(state.model):
        if eqx.is_array(value) and not bool(
            jax.device_get(jnp.all(jnp.isfinite(value)))
        ):
            bad_parameter_leaves.append((jtu.keystr(path), _array_diagnostic(value)))
    if bad_parameter_leaves:
        for name, summary in bad_parameter_leaves:
            print(f"  parameter.{name}: {summary}")
    else:
        print("  model parameters: all finite")


def run_training(
    trainer,
    setup,
    train_step,
    *,
    progress_callback=None,
    validation_evaluator=None,
):
    """Compile once, then coordinate state, pool, logging and checkpoints.

    ``progress_callback``, when provided, is called after each successful
    iteration with ``(iteration, loss, metrics)``. It is intended for runtime
    interfaces such as notebook visualisations and is kept outside compiled
    numerical code.
    """

    state = TrainState(
        trainer.model,
        setup.initial_states,
        setup.targets,
        setup.optimizer_state,
        setup.key,
        setup.initial_loss_weights,
    )
    compiled_step, compile_seconds = compile_and_time(train_step, state)
    runtime = RuntimeMetrics(compile_seconds)
    checkpoint = BestCheckpoint(
        Path(trainer.model_path).with_suffix(".eqx"), setup.checkpoint_warmup
    )
    admission_config = trainer.config.training.trainer.pool_admission
    admission = (
        TimePoolAdmissionController(admission_config, setup.warmup)
        if setup.is_multi_target
        else PoolAdmissionController(admission_config, setup.warmup)
    )
    trace_start = min(5, max(0, setup.iterations - 1))
    trace_stop = min(trace_start + 4, setup.iterations - 1)
    trace_active = False
    trace_directory = os.getenv("PROFILE_GPU_DIR", "output/jax-training-trace")
    error_code = 0
    error_iteration = None
    saved = False
    previous_loss_stage = setup.loss_weight_schedule.stage_signature(0)

    progress = tqdm(range(setup.iterations))
    for iteration in progress:
        iteration_start = time.perf_counter()
        loss_stage = setup.loss_weight_schedule.stage_signature(iteration)
        loss_stage_changed = loss_stage != previous_loss_stage
        if loss_stage_changed:
            admission.reset_references()
        previous_loss_stage = loss_stage
        states_before_step = state.states
        targets_before_step = state.targets
        state = state._replace(
            key=setup.execution.fold_in_key(state.key, iteration),
            loss_weights=setup.loss_weight_schedule(iteration),
        )
        if setup.trace_enabled and iteration == trace_start:
            start_trace(trace_directory)
            trace_active = True
        if trace_active:
            with jax.profiler.StepTraceAnnotation("train", step_num=iteration):
                output, step_seconds = call_and_time(compiled_step, state)
        else:
            output, step_seconds = call_and_time(compiled_step, state)
        if trace_active and iteration == trace_stop:
            stop_trace(output)
            trace_active = False

        maybe_save_device_memory_profile(iteration)
        state = output.state
        loss_value = float(jax.device_get(output.loss))
        metrics = setup.execution.prepare_log_dict(output.metrics)
        runtime.record_step(step_seconds)
        if setup.learning_rate_schedule is not None:
            metrics["learning_rate"] = float(
                jax.device_get(setup.learning_rate_schedule(iteration))
            )
        metrics["best_loss"] = checkpoint.best_loss

        error_code = _divergence_code(output.loss, state.states)
        if error_code:
            error_iteration = iteration
            _report_divergence(output, state, iteration)
            break

        if setup.is_multi_target:
            per_time_losses = jax.device_get(
                setup.execution.prepare_admission_losses(output.metrics["losses"])
            )
            # The final transition has no subsequent biological input slot.
            propagation_losses = tuple(
                float(value) for value in per_time_losses[:-1]
            )
            decisions = admission.decide(propagation_losses, iteration)
            source_admitted = tuple(decision.admit for decision in decisions)
        else:
            decision = admission.decide(loss_value, iteration)
            source_admitted = decision.admit
        state = _update_training_pool(
            state,
            states_before_step,
            targets_before_step,
            source_admitted,
            iteration,
            setup.execution,
        )
        if setup.is_multi_target:
            admission.update(decisions, propagation_losses)
            metrics.update(admission.metrics(decisions))
        else:
            admission.update(decision, loss_value)
            metrics.update(admission.metrics(decision))
        metrics["loss_schedule/stage"] = max(loss_stage, default=0)
        metrics["loss_schedule/stage_changed"] = int(loss_stage_changed)
        validation_every = getattr(
            trainer.config.training.trainer, "validation_every", None
        )
        should_validate = (
            validation_evaluator is not None
            and validation_every is not None
            and (
                iteration % validation_every == 0
                or iteration == setup.iterations - 1
            )
        )
        if should_validate:
            validation_metrics = validation_evaluator(
                state.model, state.loss_weights
            )
            metrics.update(
                jax.device_get(setup.execution.prepare_log_dict(validation_metrics))
            )
        runtime.record_iteration(iteration, time.perf_counter() - iteration_start)
        metrics.update(runtime.as_log_dict())

        if progress_callback is not None:
            progress_callback(iteration, loss_value, metrics)

        progress.set_postfix(
            {
                name: value
                for name, value in metrics.items()
                if name != "states"
                and not name.startswith(("pool/", "runtime/", "loss_detail/"))
            }
        )
        if checkpoint.should_save(iteration, loss_value):
            trainer.model = state.model
            trainer.model.save(checkpoint.path, overwrite=True)
            checkpoint.record(iteration, loss_value)
            saved = True
        if trainer.is_logging:
            trainer.logger.tb_training_loop_log_sequence(
                metrics,
                iteration,
                state.model,
                write_images=setup.write_images,
                LOG_EVERY=setup.log_interval,
            )

    if error_code:
        print(f"Training stopped with error {error_code} at step {error_iteration}")
    else:
        print("Training completed successfully")
    if trainer.is_logging and saved and setup.write_images:
        trainer.logger.tb_training_end_log(
            trainer.model,
            trainer.data_augmenter,
            t=setup.timesteps,
            boundary_callback=trainer.boundary_callbacks,
            SAVE_TRAJECTORY=False,
        )
    wandb_run_id = None
    if trainer.is_logging:
        wandb_run_id = getattr(getattr(trainer.logger, "run", None), "id", None)
        trainer.logger.finish()
    return TrainingResult(
        checkpoint_path=checkpoint.path if saved else None,
        best_iteration=checkpoint.best_iteration,
        best_loss=checkpoint.best_loss,
        completed=error_code == 0 and saved,
        error_code=error_code,
        wandb_run_id=wandb_run_id,
    )
