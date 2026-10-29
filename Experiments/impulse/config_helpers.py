import jax
import jax.numpy as jnp
import optax

from Experiments.emoji.config_helpers import build_hidden_seeds
from Experiments.emoji.config_helpers import load_data as load_emoji_data
from NCA.trainer.impulse.outcomes import target_gap
from NCA.trainer.impulse.perturbation import perturbation
from NCA.trainer.impulse import (
    AttractorContrastObjective,
    ExternalTargetPairSource,
    MaximalPreservativeObjective,
    MinimalDestructiveObjective,
    ModelFuturePairSource,
    StableAttractorPairSource,
    TargetedObjective,
    TrajectoryStatePairSource,
)


INTERVENTION_CHANNELS = ("all", "obs", "hidden", "single")
INTERVENTION_SPATIAL = ("local", "patch", "global", "flat")
INITIAL_LOCATIONS = ("random", "centre")


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


def build_objective(objective_cfg, attractors=None, target_attractor=None):
    """Construct the configured targeted, destructive, preservative or contrast objective.

    ``attractors`` [K, observed channels, H, W] and the index of the target
    among them are needed for attractor_contrast only.
    """

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
    if objective_cfg.type == "attractor_contrast":
        if attractors is None:
            raise ValueError("impulse.objective.type=attractor_contrast needs a multi_attractor model")
        temperature = objective_cfg.temperature
        if temperature is None:
            temperature = target_gap(attractors)
        if not temperature > 0 or temperature == float("inf"):
            raise ValueError(f"The contrast temperature must be positive and finite, got {temperature}")
        return AttractorContrastObjective(
            attractors=jnp.asarray(attractors),
            target_index=target_attractor,
            temperature=float(temperature),
            target_weight=objective_cfg.target_weight,
            contrast_weight=objective_cfg.contrast_weight,
        )
    raise ValueError(f"Unknown impulse objective {objective_cfg.type!r}")


def build_intervention(intervention_cfg, observed_channels, model, trajectories, key):
    """Build the perturbation module (``impulse/perturbation.py``) for a model and its data."""

    state_shape = (model.N_CHANNELS, *trajectories.shape[-2:])
    return intervention_from_config(intervention_cfg, observed_channels, state_shape, key)


def build_restarts(intervention_cfg, iterations, observed_channels, model, trajectories, key):
    """The other starting interventions and the iterations each start is scouted for.

    Start i > 0 uses ``fold_in(key, i)``; the first start uses ``key`` itself
    (see ``build_intervention``), so one start matches a run without restarts.
    """

    restarts = intervention_cfg.restarts
    if restarts < 1:
        raise ValueError(f"impulse.intervention.restarts must be at least 1, got {restarts}")
    if restarts == 1:
        return (), 0
    if not 0 < intervention_cfg.scout_fraction < 1:
        raise ValueError(
            f"impulse.intervention.scout_fraction must be between 0 and 1, got {intervention_cfg.scout_fraction}"
        )
    if intervention_cfg.initial_location == "centre" and intervention_cfg.init_scale == 0:
        raise ValueError(
            "Restarts with initial_location=centre and init_scale=0 would all be identical; "
            "set init_scale > 0 or initial_location=random"
        )
    others = tuple(
        build_intervention(intervention_cfg, observed_channels, model, trajectories, jax.random.fold_in(key, index))
        for index in range(1, restarts)
    )
    return others, max(int(round(intervention_cfg.scout_fraction * iterations)), 1)


def intervention_from_config(intervention_cfg, observed_channels, state_shape, key):
    """Build the perturbation module for NCA states of shape ``state_shape`` (C, H, W).

    Saved impulse records rebuild their intervention with this, from the
    shapes in their manifest, so no model or data is needed.
    """

    # "observed" matches the loss channel names; the perturbation calls it "obs".
    channels = {"observed": "obs"}.get(intervention_cfg.channels, intervention_cfg.channels)
    if channels not in INTERVENTION_CHANNELS:
        raise ValueError(
            f"Unknown impulse.intervention.channels {intervention_cfg.channels!r}; "
            f"expected one of {['observed', *INTERVENTION_CHANNELS]}"
        )
    if intervention_cfg.spatial not in INTERVENTION_SPATIAL:
        raise ValueError(
            f"Unknown impulse.intervention.spatial {intervention_cfg.spatial!r}; "
            f"expected one of {list(INTERVENTION_SPATIAL)}"
        )
    lattice_size = min(state_shape[-2:])
    if intervention_cfg.spatial == "patch" and not 1 <= intervention_cfg.patch_size <= lattice_size:
        raise ValueError(
            f"impulse.intervention.patch_size must be between 1 and the lattice size "
            f"({lattice_size}), got {intervention_cfg.patch_size}"
        )
    if intervention_cfg.duration < 1:
        raise ValueError(
            f"impulse.intervention.duration must be at least 1, got {intervention_cfg.duration}"
        )
    if intervention_cfg.initial_location not in INITIAL_LOCATIONS:
        raise ValueError(
            f"Unknown impulse.intervention.initial_location {intervention_cfg.initial_location!r}; "
            f"expected one of {list(INITIAL_LOCATIONS)}"
        )
    if intervention_cfg.init_scale < 0:
        raise ValueError(
            f"impulse.intervention.init_scale must be non-negative, got {intervention_cfg.init_scale}"
        )
    return perturbation(
        mode={
            "channel": channels,
            "spatial": intervention_cfg.spatial,
        },
        CHANNELS=state_shape[0],
        OBS_CHANNELS=observed_channels,
        x=jnp.zeros((1, *state_shape)),
        WIDTH=intervention_cfg.width,
        key=key,
        PATCH_SIZE=intervention_cfg.patch_size,
        duration=intervention_cfg.duration,
        init_scale=intervention_cfg.init_scale,
        initial_location=intervention_cfg.initial_location,
    )


def pixel_steps(intervention_cfg):
    """Number of perturbed (pixel, step) pairs of a patch impulse, k * k * duration (else None)."""

    if intervention_cfg.spatial != "patch":
        return None
    return intervention_cfg.patch_size**2 * intervention_cfg.duration


def build_impulse_optimiser(optimiser_cfg):
    """Optax optimiser for the intervention parameters."""

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


def loss_args_from_config(loss_config):
    """Return plain loss arguments accepted by ``build_loss_functions``."""
    from NCA.trainer.objective import resolve_objective

    return resolve_objective(loss_config).arguments
