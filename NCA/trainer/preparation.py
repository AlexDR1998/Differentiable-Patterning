"""Resolve immutable experiment configuration into one executable training setup."""

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

from Common.trainer.loss import build_loss_functions, build_loss_initialiser
from Common.trainer.loss_multi_target import init_texture_params
import NCA.trainer.NCA_regulariser as regularisers
from NCA.trainer.objective import resolve_loss_component_weights, resolve_objective
from NCA.trainer.loss_schedule import (
    build_loss_weight_schedule,
    final_transition_iteration,
)
from NCA.trainer.interval_schedule import IntervalSchedule, build_interval_schedule
from NCA.trainer.optimizer import build_optimizer


@dataclass(frozen=True)
class PreparedTraining:
    interval_schedule: IntervalSchedule
    iterations: int
    warmup: int
    checkpoint_warmup: int
    log_interval: int
    write_images: bool
    write_videos: bool
    trace_enabled: bool
    learning_rate_schedule: Callable | None
    optimiser: Any
    execution: Any
    loss_names: tuple[str, ...]
    loss_arguments: dict[str, Any]
    loss_functions: tuple[Callable, ...]
    loss_component_weights: Any
    loss_weight_schedule: Callable
    initial_loss_weights: Any
    loss_channels: Any
    loss_cache: Any
    is_multi_target: bool
    multi_target_params: Any
    regulariser_functions: dict[str, Callable]
    regulariser_coefficients: dict[str, float]
    initial_states: Any
    targets: Any
    optimizer_state: Any
    key: Any

    @property
    def timesteps(self):
        """Scan length of the training rollout (``t`` for uniform schedules)."""
        return self.interval_schedule.scan_length


def _interval_schedule(loop, context, n_slots, timesteps=None):
    """Resolve per-slot timing; ``timesteps`` overrides ``loop.t`` (e.g. emoji fire-rate t)."""
    if loop.interval_mode in ("fire_rate", "dt"):
        raise NotImplementedError(
            f"training.loop.interval_mode={loop.interval_mode!r} is not implemented yet; "
            "use 'uniform' or 'steps'"
        )
    return build_interval_schedule(
        loop.t if timesteps is None else timesteps,
        n_slots,
        mode=loop.interval_mode,
        times=getattr(context, "observation_times", None),
        reference_interval=loop.reference_interval,
        explicit_steps=loop.interval_steps,
    )


def _singular_value_settings(config):
    settings = config.logging.singular_values
    return {
        "enabled": bool(settings.enabled),
        "plot_spectra": bool(settings.plot_spectra),
        "epsilon": float(settings.epsilon),
    }


def _regularisers(coefficients):
    available = {
        "intermediate_state": regularisers.intermediate_reg,
        "boundary": regularisers.boundary_regulariser,
        "contiguous_growth": regularisers.contiguous_growth_regulariser,
        "localised_hidden": regularisers.localised_hidden_regulariser,
        "update_sensitivity": regularisers.update_sensitivity_regulariser,
        "perturbation_conservation": regularisers.perturbation_conservation_regulariser,
        "hidden_state_size": regularisers.hidden_state_size_regulariser,
    }
    active = {name: float(value) for name, value in coefficients.items() if value}
    unknown = set(active) - set(available)
    if unknown:
        raise ValueError(f"Unknown regularisers: {sorted(unknown)}")
    return {name: available[name] for name in active}, active


def _prepare_loss_cache(trainer, names, arguments, targets, key, is_multi_target):
    if is_multi_target:
        return arguments, [None] * len(targets)
    initialiser = build_loss_initialiser(names, arguments)
    if initialiser is None:
        return arguments, [None] * len(targets)
    observed_targets = jtu.tree_map(
        lambda target: target[:, : trainer.data_channels], targets
    )
    target_cache = initialiser(
        observed_targets, key, trainer.loss_time_channel_mask
    )
    arguments = {**arguments, "vgg_params": target_cache["vgg_params"]}
    cache_targets = not arguments.get("random_crop", False) and not arguments.get(
        "random_channel_shuffle", False
    )
    if not cache_targets:
        return arguments, [None] * len(targets)
    return arguments, [
        target_cache["target_feats"][index] for index in range(len(targets))
    ]


def prepare_training(trainer, *, key, timesteps=None, loss_overrides=None):
    config = trainer.config
    loop = config.run
    trainer_config = config.trainer
    objective = resolve_objective(config.loss, loss_overrides)
    names = tuple(objective.names)
    arguments = dict(objective.arguments)
    is_multi_target = names == ("multi_target",)
    if is_multi_target and trainer.channel_schema is None:
        raise ValueError("multi_target requires a channel schema")
    if is_multi_target and trainer.grad_loss:
        raise ValueError("multi_target requires trainer.grad_loss=False")

    optimiser, _, schedule = build_optimizer(
        config.optimiser, loop.iterations, return_schedule=True
    )
    loss_channels = arguments.get("channels")
    if loss_channels is None:
        loss_channels = jnp.full((trainer.observed_channels,), -1, dtype=jnp.int32)
    elif len(loss_channels) != trainer.observed_channels:
        raise ValueError(
            "loss channels must have one entry per observable state channel"
        )

    states, targets = trainer.data_augmenter.initialize_pool(key)
    states = jtu.tree_map(trainer.model.prepare_pool_state, states)
    interval_schedule = _interval_schedule(
        loop, trainer.context, jtu.tree_leaves(states)[0].shape[0], timesteps
    )
    # Rollout helpers (interventions, logging) read the schedule from the trainer.
    trainer.interval_schedule = interval_schedule
    if interval_schedule.mode != "uniform":
        print(f"Interval schedule ({interval_schedule.mode}): steps per transition = {interval_schedule.steps}")
    loss_weight_schedule = build_loss_weight_schedule(
        config.loss, loop.iterations
    )
    initial_loss_weights = loss_weight_schedule(0)
    multi_target_params = None
    texture_enabled = bool(
        arguments.get("multi_target_weights", {}).get("texture", 1.0)
    )
    if is_multi_target:
        arguments["texture_enabled"] = texture_enabled
    if is_multi_target and texture_enabled:
        multi_target_params = init_texture_params(
            key,
            targets[0].shape[-2:],
            arguments.get("metric", "l2"),
            arguments.get("samples", 128),
        )
    arguments, loss_cache = _prepare_loss_cache(
        trainer, names, arguments, targets, key, is_multi_target
    )
    trainer.loss_cache = loss_cache
    loss_functions = () if is_multi_target else tuple(
        build_loss_functions(names, arguments)
    )
    regulariser_functions, coefficients = _regularisers(
        objective.regulariser_coefficients
    )
    execution = trainer._training_execution()
    model, states, targets, optimizer_state, key = execution.prepare_inputs(
        trainer.model,
        states,
        targets,
        optimiser.init(trainer.model.partition()[0]),
        key,
    )

    trainer.setup_logging(
        config.logging.backend,
        wandb_args={
            "project": config.logging.wandb.project,
            "group": config.logging.wandb.group,
            "tags": list(trainer.context.wandb_tags),
            "name": trainer.context.run_name,
        },
        knockout={
            "time": config.data.knockout.time,
            "channel": config.data.knockout.channel,
        },
        singular_value_settings=_singular_value_settings(config),
    )
    trainer.model = model
    return PreparedTraining(
        interval_schedule=interval_schedule,
        iterations=loop.iterations,
        warmup=config.run.checkpoint_warmup,
        checkpoint_warmup=max(
            config.run.checkpoint_warmup,
            max(
                0,
                final_transition_iteration(
                    config.loss, loop.iterations
                )
                - 1,
            ),
        ),
        log_interval=trainer_config.log_every,
        write_images=loop.write_images,
        write_videos=loop.write_videos,
        trace_enabled=trainer_config.jax_trace,
        learning_rate_schedule=schedule,
        optimiser=optimiser,
        execution=execution,
        loss_names=names,
        loss_arguments=arguments,
        loss_functions=loss_functions,
        loss_component_weights=resolve_loss_component_weights(
            arguments.get("component_weights"), len(names)
        ),
        loss_weight_schedule=loss_weight_schedule,
        initial_loss_weights=initial_loss_weights,
        loss_channels=loss_channels,
        loss_cache=loss_cache,
        is_multi_target=is_multi_target,
        multi_target_params=multi_target_params,
        regulariser_functions=regulariser_functions,
        regulariser_coefficients=coefficients,
        initial_states=states,
        targets=targets,
        optimizer_state=optimizer_state,
        key=key,
    )
