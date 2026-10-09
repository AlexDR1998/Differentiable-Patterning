import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from Experiments.emoji.config_helpers import build_hidden_seeds
from Experiments.emoji.config_helpers import load_data as load_emoji_data
from NCA.trainer.impulse.perturbation import perturbation
from NCA.trainer.impulse import (
    ExternalTargetPairSource,
    MaximalPreservativeObjective,
    MinimalDestructiveObjective,
    ModelFuturePairSource,
    StableAttractorPairSource,
    TargetedObjective,
    TrajectoryStatePairSource,
)


def _as_dict(value):
    """Convert a mapping-like config section to a plain dictionary."""

    if value is None:
        return {}
    return {key: item for key, item in value.items()}


def load_impulse_data(data_config, model, impath=None):
    """Load emoji trajectories as [pair, time, model channels, H, W] NCA states.

    Hidden channels are zero apart from pixel seeds, as in training.
    """

    if data_config.dataset != "emojis":
        raise ValueError("The initial impulse entrypoint currently supports data.dataset=emojis")
    data, _ = load_emoji_data(data_config, impath=impath)
    data = jnp.asarray(data)

    model_channels = model.N_CHANNELS
    data_channels = data.shape[2]
    if data_channels > model_channels:
        raise ValueError(
            f"Loaded data has {data_channels} channels but the model has {model_channels}"
        )
    hidden_seeds = build_hidden_seeds(data_config, data, model_channels)
    data = jnp.pad(data, ((0, 0), (0, 0), (0, model_channels - data_channels), (0, 0), (0, 0)))
    if hidden_seeds is not None:
        data = data.at[:, 0, data_channels:].set(hidden_seeds)

    pad = data_config.emoji.pad
    if pad is not None:
        if isinstance(pad, int):
            pad = [pad, pad, pad, pad]
        data = jnp.pad(
            data,
            ((0, 0), (0, 0), (0, 0), (pad[0], pad[1]), (pad[2], pad[3])),
        )
    return data


def build_pair_source(impulse_config, model, trajectories):
    """Construct the configured initial-state/target-state provider."""

    pair_cfg = impulse_config.pair_source
    pair_type = pair_cfg.type
    source, target = pair_cfg.source_index, pair_cfg.target_index
    for name, index in (("source_index", source), ("target_index", target)):
        if not 0 <= index < len(trajectories):
            raise ValueError(
                f"impulse.pair_source.{name}={index} is outside the {len(trajectories)} data pairs"
            )
    # [1, time, C, H, W]: the source pattern's trajectory
    source_trajectory = trajectories[source : source + 1]
    if pair_type == "external_target":
        # Source initial condition -> target pattern's data image
        return ExternalTargetPairSource(source_trajectory[:, 0], trajectories[target : target + 1, -1])
    if pair_type == "model_future":
        return ModelFuturePairSource(
            source_trajectory[:, 0],
            target_steps=pair_cfg.target_steps,
            scan_kind=impulse_config.rollout.scan_kind,
        )
    if pair_type == "trajectory_state":
        return TrajectoryStatePairSource(
            source_trajectory,
            initial_index=pair_cfg.initial_time,
            target_index=pair_cfg.target_time,
        )
    if pair_type == "stable_attractor":
        return StableAttractorPairSource(
            trajectories[:, 0],
            source_index=pair_cfg.source_index,
            target_index=pair_cfg.target_index,
            stabilisation_steps=tuple(pair_cfg.stabilisation_steps),
            scan_kind=impulse_config.rollout.scan_kind,
        )
    raise ValueError(f"Unknown impulse pair source {pair_type!r}")


def build_objective(objective_cfg):
    """Construct the configured targeted, destructive, or preservative objective."""

    if objective_cfg.type == "targeted":
        return TargetedObjective(target_weight=objective_cfg.target_weight)
    if objective_cfg.type == "minimal_destructive":
        return MinimalDestructiveObjective(target_weight=objective_cfg.target_weight)
    if objective_cfg.type == "maximal_preservative":
        return MaximalPreservativeObjective(
            tolerance=objective_cfg.tolerance,
            constraint_weight=objective_cfg.constraint_weight,
            magnitude=objective_cfg.magnitude,
            reward_weight=objective_cfg.reward_weight,
        )
    raise ValueError(f"Unknown impulse objective {objective_cfg.type!r}")


def build_intervention(intervention_cfg, observed_channels, model, trajectories, key):
    """Initialise the legacy Equinox perturbation module from config."""

    return perturbation(
        mode={
            "channel": intervention_cfg.channels,
            "spatial": intervention_cfg.spatial,
        },
        CHANNELS=model.N_CHANNELS,
        OBS_CHANNELS=observed_channels,
        x=trajectories[:1, 0],
        WIDTH=intervention_cfg.width,
        key=key,
    )


def build_impulse_optimiser(optimiser_cfg):
    """Build the lightweight Optax optimiser used for intervention parameters."""

    constructors = {
        "adam": optax.adam,
        "nadam": optax.nadam,
        "adamw": optax.adamw,
        "sgd": optax.sgd,
    }
    if optimiser_cfg.type not in constructors:
        raise ValueError(f"Unknown impulse optimiser {optimiser_cfg.type!r}")
    optimiser = constructors[optimiser_cfg.type](optimiser_cfg.learn_rate)
    clip_norm = optimiser_cfg.gradient_clip_norm
    if clip_norm is not None:
        optimiser = optax.chain(optax.clip_by_global_norm(clip_norm), optimiser)
    return optimiser


def resolve_output_directory(output_cfg, env=None):
    """Resolve and create the directory used for intervention outputs."""

    directory = Path(str(output_cfg.directory)).expanduser()
    environment = os.environ if env is None else env
    if not directory.is_absolute():
        base_env = output_cfg.base_env
        base = environment.get(str(base_env)) if base_env else None
        if base:
            directory = Path(base).expanduser() / directory
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def loss_args_from_config(loss_config):
    """Return plain loss arguments accepted by ``build_loss_functions``."""
    from NCA.trainer.objective import resolve_objective

    return resolve_objective(loss_config).arguments
