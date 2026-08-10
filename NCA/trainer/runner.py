"""Readable Python lifecycle around the compiled numerical step."""

import os
import time
from pathlib import Path

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
from NCA.trainer.pool import PoolAdmissionController
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
    if bool(jax.device_get(jnp.isnan(loss))):
        return 1
    if any(
        bool(jax.device_get(jnp.any(jnp.isnan(value))))
        for value in jtu.tree_leaves(states)
    ):
        return 2
    if float(jax.device_get(loss)) > 1e16:
        return 3
    return 0


def run_training(trainer, setup, train_step, *, progress_callback=None):
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
    admission = PoolAdmissionController(
        trainer.config.training.trainer.pool_admission, setup.warmup
    )
    trace_start = min(5, max(0, setup.iterations - 1))
    trace_stop = min(trace_start + 4, setup.iterations - 1)
    trace_active = False
    trace_directory = os.getenv("PROFILE_GPU_DIR", "output/jax-training-trace")
    error_code = 0
    error_iteration = None
    saved = False

    progress = tqdm(range(setup.iterations))
    for iteration in progress:
        iteration_start = time.perf_counter()
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
            break

        decision = admission.decide(loss_value, iteration)
        if decision.admit:
            next_key, augment_key = setup.execution.split_key(state.key)
            states, targets = setup.execution.apply_advance_pool(
                state.states, state.targets, iteration, augment_key
            )
            state = state._replace(
                states=states, targets=targets, key=next_key
            )
        admission.update(decision, loss_value)
        metrics.update(admission.metrics(decision))
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
            trainer.model.save(trainer.model_path, overwrite=True)
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
