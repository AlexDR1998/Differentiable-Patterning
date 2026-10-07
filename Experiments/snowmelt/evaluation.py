"""Inference evaluation of trained snowmelt NCA model bundles.

Each bundle's input is rebuilt from its saved config exactly as training built
it (same target and static channels, downsampling, border and catchment
threshold), and checked against the fingerprint stored in the bundle. The model
is then rolled out from an observed acquisition and compared with the later
acquisitions:

``free``
    One rollout from the first acquisition through all later ones. This is the
    forecasting test: nothing observed after the first date is used.
``interval``
    Every interval restarts from the observed image at its start, as in
    training. This isolates the error made within a single interval.

Scores use physical units (reflectance, NDSI/NDVI in [-1, 1], snow cover
fraction), catchment pixels only, and are compared with persistence (the first
image repeated; in ``interval`` mode the image at the start of each interval),
the usual no-skill reference for snow-cover forecasts.

Dates held out of training (``data.snowmelt.hold_out_dates``) are put back for
evaluation and flagged in the scores, so they test whether the model
interpolates (or, for the last date, extrapolates) to dates it never saw.
"""

import numpy as np

from Common.dataloader.snowmelt import build_snowmelt_sequence
from NCA.trainer.interval_schedule import interval_schedule_from_config

ROLLOUT_MODES = ("free", "interval")
# Snow / no-snow threshold for each channel, in physical units.
SNOW_THRESHOLDS = {"SCA": 0.5, "NDSI": 0.4}


def input_recipe(cfg):
    """The arguments of ``build_snowmelt_sequence`` that a training config used."""
    snowmelt = cfg.data.snowmelt
    return {
        "version": snowmelt.version,
        "target_channels": tuple(snowmelt.target_channels),
        "static_channels": tuple(snowmelt.static_channels),
        "downsample": int(cfg.data.downsample),
        "pad": int(snowmelt.pad),
        "mask_threshold": float(snowmelt.mask_threshold),
        "exclude_dates": tuple(snowmelt.exclude_dates),
        "hold_out_dates": tuple(snowmelt.hold_out_dates),
    }


def model_label(cfg):
    """Short readable name, e.g. ``gNCA · NDSI+B3+B11 · 40 m · t128 steps · rep0``."""
    loop = cfg.run
    label = (
        f"{cfg.model.family} · {'+'.join(cfg.data.snowmelt.target_channels)} · "
        f"{10 * cfg.data.downsample} m · t{loop.t} {loop.interval_mode}"
    )
    return label if loop.repeat is None else f"{label} · rep{loop.repeat}"


def load_bundle_sequence(bundle, raw, cache=None, verify=True):
    """Rebuild a bundle's evaluation sequence from ``raw`` (a ``load_snowmelt`` result).

    This is the training sequence plus any held-out dates (``sequence.held_out``
    marks them); excluded dates stay out. ``raw`` must be the dataset version
    the bundle was trained on.

    With ``verify``, the training sequence's initial state and boundary channels
    must match the fingerprint recorded at training time. ``cache`` (a dict)
    shares sequences between bundles that used the same recipe.
    """
    from Experiments.model_registry import verify_evaluation_input

    recipe = input_recipe(bundle.config)
    if raw.get("version") not in (None, recipe["version"]):
        raise ValueError(
            f"{bundle.id} was trained on snowmelt dataset {recipe['version']}, "
            f"but the loaded data is {raw['version']}"
        )
    cache = {} if cache is None else cache

    def build(include_held_out):
        key = (*recipe.items(), include_held_out)
        if key not in cache:
            cache[key] = build_snowmelt_sequence(raw=raw, include_held_out=include_held_out, **recipe)
        return cache[key]

    if verify:
        if "evaluation_input" not in bundle.manifest:
            raise ValueError(f"{bundle.id} has no evaluation-input fingerprint to verify against")
        training = build(include_held_out=False)
        verify_evaluation_input(
            training.data, bundle.manifest.evaluation_input, boundary_mask=training.boundary_mask
        )
    return build(include_held_out=True)


def initial_state(observed, boundary, n_channels):
    """Full NCA state from an observed ``[C_obs, H, W]`` image.

    Observed channels first, hidden channels zero, and the fixed boundary
    channels (catchment, then static terrain) last, as in the training pool.
    """
    state = np.zeros((n_channels, *observed.shape[-2:]), dtype=np.float32)
    state[: observed.shape[0]] = observed
    state[-boundary.shape[0]:] = boundary
    return state


def rollout_schedule(cfg, sequence):
    """NCA steps per interval, resolved from the training config and acquisition times."""
    return interval_schedule_from_config(
        cfg, len(sequence.dates) - 1, sequence.observation_times
    )


def predict(model, cfg, sequence, key, n_rollouts=1, mode="free"):
    """Predicted observed channels at every acquisition, ``[n_rollouts, T, C_obs, H, W]``.

    Index 0 is the observed first image. The rollouts differ only in their
    random key (stochastic cell updates).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from NCA.trainer.intervention import rollout_model_sampled

    if mode not in ROLLOUT_MODES:
        raise ValueError(f"mode must be one of {ROLLOUT_MODES}, got {mode!r}")
    observed = np.asarray(sequence.data)[0]
    boundary = jnp.asarray(sequence.boundary_mask[0])
    n_obs = observed.shape[1]
    schedule = rollout_schedule(cfg, sequence)
    boundary_mode = cfg.trainer.boundary_mode

    def run(start, steps, run_key):
        state = jnp.asarray(initial_state(start, sequence.boundary_mask[0], model.N_CHANNELS))
        observation_steps = jnp.asarray(steps, dtype=jnp.int32)
        keys = jr.split(run_key, n_rollouts)
        frames = jax.vmap(
            lambda k: rollout_model_sampled(
                model, state, boundary, boundary_mode, k, int(steps[-1]), observation_steps
            )
        )(keys)
        return np.asarray(frames[:, :, :n_obs])

    if mode == "free":
        return run(observed[0], schedule.observation_steps, key)

    first = np.broadcast_to(observed[0], (n_rollouts, 1, *observed.shape[1:]))
    later = [
        run(observed[slot], (0, schedule.steps[slot]), jr.fold_in(key, slot))[:, -1:]
        for slot in range(schedule.n_slots)
    ]
    return np.concatenate([first, *later], axis=1)


def trajectory(model, cfg, sequence, key, stride=1, channels=None):
    """Free run from the first acquisition, keeping ``channels`` every ``stride`` NCA steps.

    Returns ``(frames [F, len(channels), H, W], steps)`` where ``steps[i]`` is
    the NCA step of frame ``i`` (frame 0 is the initial state). Only the kept
    channels are stored, so long high-resolution rollouts stay small.
    """
    import jax.numpy as jnp

    if cfg.trainer.boundary_mode != "soft":
        raise ValueError("Snowmelt trajectories expect trainer.boundary_mode='soft'")
    stride = max(1, int(stride))
    n_frames = rollout_schedule(cfg, sequence).total_steps // stride
    channels = tuple(range(model.N_CHANNELS)) if channels is None else tuple(channels)
    state = jnp.asarray(initial_state(np.asarray(sequence.data)[0, 0], sequence.boundary_mask[0], model.N_CHANNELS))
    frames = _strided_rollout(
        model, state, jnp.asarray(sequence.boundary_mask[0]), key, n_frames, stride, jnp.asarray(channels)
    )
    return np.asarray(frames), tuple(range(0, (n_frames + 1) * stride, stride))


def _strided_rollout(model, state, boundary, key, n_frames, stride, channels):
    import equinox as eqx
    import jax
    import jax.random as jr

    @eqx.filter_jit
    def run(model, state, boundary, key, channels):
        def write_boundary(x):
            return x.at[-boundary.shape[0]:].set(boundary)

        def advance(x, frame):
            def step(i, x):
                return model(x, write_boundary, key=jr.fold_in(key, frame * stride + i))

            x = jax.lax.fori_loop(0, stride, step, x)
            return x, x[channels]

        _, frames = jax.lax.scan(advance, state, jax.numpy.arange(n_frames))
        return jax.numpy.concatenate([state[channels][None], frames])

    return run(model, state, boundary, key, channels)


def to_physical(values, channel_names):
    """Undo the training scaling along the channel axis (``-3``): NDSI/NDVI back to [-1, 1]."""
    values = np.array(values, dtype=np.float32, copy=True)
    for index, name in enumerate(channel_names):
        if name in ("NDSI", "NDVI"):
            values[..., index, :, :] = 2.0 * values[..., index, :, :] - 1.0
    return values


def strip_border(values, pad):
    """Remove the zero border that training adds around the grid."""
    return values if pad == 0 else values[..., pad:-pad, pad:-pad]


def upsample_blocks(values, factor):
    """Blow a downsampled grid back up to 10 m by repeating each block."""
    return np.repeat(np.repeat(values, factor, axis=-2), factor, axis=-1)


def full_resolution_reference(raw, channel_names, factor, exclude_dates=()):
    """Observed channels and catchment on the 10 m grid, cropped to the model's footprint.

    Returns ``(observed [T, C, H, W], catchment [H, W])`` with the same scaling,
    gap-filling and dates as the evaluation sequence (all but ``exclude_dates``),
    but no downsampling.
    """
    sequence = build_snowmelt_sequence(
        raw=raw, target_channels=channel_names, static_channels=(), downsample=1, pad=0,
        exclude_dates=exclude_dates,
    )
    height, width = raw["mask"].shape
    height, width = (height // factor) * factor, (width // factor) * factor
    observed = sequence.data[0][..., :height, :width]
    return observed, raw["mask"][:height, :width]


def score(prediction, observed, catchment, channel_names, dates, days, mode="free", held_out=None):
    """Per-date, per-channel scores of a prediction over catchment pixels.

    ``prediction`` is ``[R, T, C, H, W]`` and ``observed`` ``[T, C, H, W]``,
    both in physical units. The ensemble mean over the ``R`` rollouts is scored
    against the observation and against persistence: ``observed[0]`` in
    ``free`` mode, the previous acquisition in ``interval`` mode.
    ``skill`` is the MSE skill score ``1 - MSE / MSE_persistence``: 1 is
    perfect, 0 is no better than persistence, negative is worse.

    Channels in :data:`SNOW_THRESHOLDS` are also classified as snow / no snow
    and compared with the observed snow map: the snow-covered fractions and
    the critical success index (hits / (hits + misses + false alarms)).
    ``held_out`` (one flag per date, e.g. ``sequence.held_out``) marks the
    dates the model was not trained on.
    """
    held_out = (False,) * observed.shape[0] if held_out is None else tuple(held_out)
    rows = []
    for t in range(1, observed.shape[0]):
        for c, name in enumerate(channel_names):
            rollouts = prediction[:, t, c][:, catchment]
            mean = rollouts.mean(axis=0)
            truth = observed[t, c][catchment]
            persistence = observed[0 if mode == "free" else t - 1, c][catchment]
            error = mean - truth
            mse = float(np.mean(error ** 2))
            mse_persistence = float(np.mean((persistence - truth) ** 2))
            row = {
                "date": dates[t],
                "days": float(days[t]),
                "held_out": bool(held_out[t]),
                "channel": name,
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "persistence_rmse": float(np.sqrt(mse_persistence)),
                "skill": 1.0 - mse / mse_persistence if mse_persistence > 0 else np.nan,
                "rollout_spread": float(np.mean(rollouts.std(axis=0))),
            }
            if name in SNOW_THRESHOLDS:
                threshold = SNOW_THRESHOLDS[name]
                true_snow = truth > threshold
                predicted_snow = mean > threshold
                hits = np.sum(predicted_snow & true_snow)
                misses = np.sum(~predicted_snow & true_snow)
                false_alarms = np.sum(predicted_snow & ~true_snow)
                row.update({
                    "observed_snow_fraction": float(true_snow.mean()),
                    "predicted_snow_fraction": float(predicted_snow.mean()),
                    "snow_fraction_spread": float((rollouts > threshold).mean(axis=1).std()),
                    "snow_csi": float(hits / max(hits + misses + false_alarms, 1)),
                    "snow_accuracy": float(np.mean(predicted_snow == true_snow)),
                })
            rows.append(row)
    return rows


__all__ = [
    "ROLLOUT_MODES",
    "SNOW_THRESHOLDS",
    "full_resolution_reference",
    "initial_state",
    "input_recipe",
    "load_bundle_sequence",
    "model_label",
    "predict",
    "rollout_schedule",
    "score",
    "strip_border",
    "to_physical",
    "trajectory",
    "upsample_blocks",
]
