"""Evaluation of multi-attractor emoji NCAs: which target does a seed grow into?

A multi-attractor model (``data.emoji.task: multi_attractor``) is trained to
grow a different emoji from each seed in ``data.emoji.pairs``. The helpers here
rebuild those seeds and targets from a bundle's saved config, write other
colours into the seed, roll many seeds out at once, and label each final state
with the target it is closest to.

Distances are the mean squared error of the observed channels (RGBA) over the
whole padded grid, as in the training loss. A final state counts as a target
when its distance is below ``tolerance`` times the smallest distance between
two different targets, so the threshold does not depend on the emoji used.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from Experiments.emoji.config_helpers import _pad_tuple, build_hidden_seeds, load_data
from NCA.model.factory import portable_family

# Outcome labels and the nearest-target rule are shared with the impulse validation
from NCA.trainer.impulse.outcomes import ALIVE_ALPHA, DIED, DIVERGED, OTHER, OUTCOME_NAMES, classify, mse, target_gap  # noqa: F401


def is_multi_attractor(cfg):
    return cfg.data.dataset == "emojis" and cfg.data.emoji.task == "multi_attractor"


def emoji_name(filename):
    return Path(filename).stem


def seed_type(initial):
    """Input space of a pair's ``initial`` entry: ``full``, ``patch`` or ``pixel``."""
    return "full" if isinstance(initial, str) else initial.get("mode", "full")


def seed_description(initial):
    """Short text for a pair's ``initial`` entry, e.g. ``patch 1`` or ``pixel [3, 4]``."""
    if isinstance(initial, str):
        return "full"
    mode = initial.get("mode", "full")
    if mode == "patch":
        return f"patch {initial.get('size', 12)}"
    if mode == "pixel":
        return f"pixel {list(initial.get('channels', [initial.get('channel', 0)]))}"
    return mode


def target_names(cfg):
    """Distinct target emoji names, in the order of ``data.emoji.pairs``."""
    return list(dict.fromkeys(emoji_name(pair.target) for pair in cfg.data.emoji.pairs))


def summary_row(bundle):
    """One table row of the settings that differ between multi-attractor runs."""
    cfg = bundle.config
    emoji = cfg.data.emoji
    names = target_names(cfg)
    training = bundle.manifest.get("training", {})
    return {
        "model_id": bundle.id,
        "created_at": str(bundle.manifest.created_at),
        "collection": str(bundle.manifest.collection),
        "experiment": bundle.manifest.get("experiment"),
        "wandb_group": cfg.logging.wandb.group,
        "attractors": len(names),
        "targets": " + ".join(names),
        "seed_type": " / ".join(dict.fromkeys(seed_type(pair.initial) for pair in emoji.pairs)),
        "seed": " / ".join(dict.fromkeys(seed_description(pair.initial) for pair in emoji.pairs)),
        "family": cfg.model.family,
        "channels": cfg.model.channels,
        "padding": cfg.model.padding,
        "fire_rate": cfg.model.fire_rate,
        # Only the nNCA and gnNCA families use model.parameter_noise_level
        "parameter_noise": (cfg.model.parameter_noise_level
                            if portable_family(cfg.model.family) in {"nNCA", "gnNCA"} else 0.0),
        # Only aNCA and gaNCA learn their firing rate
        "firing": (cfg.model.firing_estimator
                   if cfg.model.family in {"aNCA", "gaNCA"} else "fixed"),
        "t": cfg.run.t,
        "iterations": cfg.run.iterations,
        "batches": cfg.data.batches,
        "stability_mode": cfg.experiment.stability_mode,
        "terminal_carry": emoji.terminal_carry.enabled,
        "regeneration": emoji.regeneration.enabled,
        "repeat": cfg.run.repeat,
        "best_loss": training.get("best_loss"),
        "best_iteration": training.get("best_iteration"),
    }


def load_seeds_and_targets(cfg, model_channels, impath):
    """Seeds and targets of a multi-attractor config, as training builds them.

    Returns ``(seeds, targets, pair_target, data)``: ``seeds`` [pair, model
    channels, H, W] and ``targets`` [target, data channels, H, W] are padded
    like the training pool; ``pair_target`` gives each pair's index into
    ``targets`` (pairs may share a target); ``data`` is the unpadded
    [pair, time, data channels, H, W] array, for ``verify_evaluation_input``.
    """
    data, _ = load_data(cfg.data, impath=impath)
    data = np.asarray(data, dtype=np.float32)
    pairs, _, data_channels, height, width = data.shape
    hidden = build_hidden_seeds(cfg.data, data, model_channels)
    if hidden is None:
        hidden = np.zeros((pairs, model_channels - data_channels, height, width), np.float32)
    seeds = np.concatenate([data[:, 0], hidden], axis=1)

    names = target_names(cfg)
    pair_target = np.array([names.index(emoji_name(pair.target)) for pair in cfg.data.emoji.pairs])
    first_pair = [list(pair_target).index(index) for index in range(len(names))]
    targets = data[first_pair, -1]

    pad = _pad_tuple(cfg.data.emoji.pad)
    if pad is not None:
        top, bottom, left, right = pad
        widths = ((0, 0), (0, 0), (top, bottom), (left, right))
        seeds, targets = np.pad(seeds, widths), np.pad(targets, widths)
    return seeds, targets, pair_target, data


def seed_footprint(seeds):
    """[H, W] mask of the pixels any training seed sets: one pixel for pixel and
    1x1 patch seeds, a square for larger patches."""
    footprint = np.any(seeds != 0, axis=(0, 1))
    if not footprint.any():
        footprint[seeds.shape[-2] // 2, seeds.shape[-1] // 2] = True
    return footprint


def seed_colours(seeds, footprint, data_channels=4):
    """Mean RGBA of each training seed over the footprint, [pair, 4]."""
    return seeds[:, :data_channels, footprint].mean(axis=-1)


def hsva_to_rgba(hsva):
    """[..., 4] hue, saturation, value, alpha (all in [0, 1]) to RGBA."""
    hsva = np.asarray(hsva, dtype=np.float32)
    rgb = hsv_to_rgb(np.clip(hsva[..., :3], 0.0, 1.0))
    return np.concatenate([rgb, hsva[..., 3:4]], axis=-1).astype(np.float32)


def rgba_to_hsva(rgba):
    rgba = np.asarray(rgba, dtype=np.float32)
    hsv = rgb_to_hsv(np.clip(rgba[..., :3], 0.0, 1.0))
    return np.concatenate([hsv, rgba[..., 3:4]], axis=-1)


def coloured_seeds(base, footprint, rgba):
    """Copies of the ``base`` seed [C, H, W] with the footprint set to each colour in
    ``rgba`` [N, 4]; the hidden channels keep the base seed's values."""
    rgba = np.asarray(rgba, dtype=np.float32)
    states = np.repeat(np.asarray(base, dtype=np.float32)[None], len(rgba), axis=0)
    states[:, : rgba.shape[1], footprint] = rgba[:, :, None]
    return states


def _rollout(model, state, key, targets, steps, stride):
    """One trajectory of ``steps`` updates: RGBA every ``stride`` steps and the
    distance to each target at every step, both including the seed."""
    data_channels = targets.shape[1]

    def distance(state):
        return jnp.mean((state[None, :data_channels] - targets) ** 2, axis=(1, 2, 3))

    def step(state, index):
        state = model(state, lambda x: x, key=jax.random.fold_in(key, index))
        return state, distance(state)

    def frame(state, start):
        state, distances = jax.lax.scan(step, state, start + jnp.arange(stride))
        return state, (state[:data_channels], distances)

    _, (frames, distances) = jax.lax.scan(frame, state, stride * jnp.arange(steps // stride))
    frames = jnp.concatenate([state[None, :data_channels], frames])
    distances = jnp.concatenate([distance(state)[None], distances.reshape(-1, len(targets))])
    return frames, distances


@eqx.filter_jit
def _rollout_batch(model, states, keys, targets, steps, stride):
    return jax.vmap(lambda state, key: _rollout(model, state, key, targets, steps, stride))(states, keys)


def chunk_size(model, height, width, memory_gb):
    """How many seeds can be rolled out at once in about ``memory_gb`` of device memory.

    The largest arrays are the hidden layer's feature maps (``N_FEATURES`` x H x W
    floats per seed). A compiled rollout holds about three of them per seed;
    one more is allowed for the GPU convolution workspace.
    """
    features = getattr(model, "N_FEATURES", 4 * model.N_CHANNELS)
    per_seed = 4 * 4 * features * height * width
    return max(1, int(memory_gb * 2**30 // per_seed))


def rollout(model, states, targets, steps, key, frames=1, memory_gb=1.0):
    """Roll out each seed in ``states`` [N, C, H, W] for ``steps`` updates.

    Returns the distance to each target at every step, [N, steps + 1, target],
    and the RGBA every ``steps / frames`` steps, [N, frames + 1, 4, H, W]
    (``frames=1``: the seed and the final state). Only these are kept, and the
    seeds are run in chunks that fit in about ``memory_gb`` (see ``chunk_size``).
    """
    if steps % frames:
        raise ValueError(f"steps ({steps}) must be a multiple of frames ({frames})")
    states = np.asarray(states, dtype=np.float32)
    count = len(states)
    chunk = min(count, chunk_size(model, *states.shape[-2:], memory_gb))
    keys = jax.random.split(key, count)
    # Pad to whole chunks, so every chunk has the same shape and compiles once
    padding = -count % chunk
    states = np.concatenate([states, np.zeros((padding, *states.shape[1:]), np.float32)])
    keys = jnp.concatenate([keys, jnp.repeat(keys[:1], padding, axis=0)])
    targets = jnp.asarray(targets)
    distances, rgba = [], []
    for start in range(0, len(states), chunk):
        chunk_rgba, chunk_distances = _rollout_batch(
            model, jnp.asarray(states[start : start + chunk]), keys[start : start + chunk],
            targets, int(steps), int(steps // frames),
        )
        rgba.append(np.asarray(chunk_rgba))
        distances.append(np.asarray(chunk_distances))
    return np.concatenate(distances)[:count], np.concatenate(rgba)[:count]


def outcome_name(label, names):
    return names[label] if label >= 0 else OUTCOME_NAMES[label]


def evaluate_seeds(model, seeds, targets, pair_target, steps, horizon, repeats, key, tolerance, memory_gb=1.0):
    """Roll out every training seed ``repeats`` times and score the outcomes.

    Returns ``(rows, final, labels)``. ``rows`` has one row per pair with the
    fraction of rollouts that grew the right target (``correct``), grew a
    different target (``wrong_target``), grew something else (``other``) or
    died/diverged (``failed``), and the mean distance to the right target at
    ``horizon`` (the training rollout length) and at the end. ``final``
    [pair, repeat, 4, H, W] is the final RGBA of every rollout and ``labels``
    [pair, repeat] its outcome (see ``classify``).
    """
    states = np.repeat(seeds, repeats, axis=0)
    distances, rgba = rollout(model, states, targets, steps, key, memory_gb=memory_gb)
    labels = classify(distances[:, -1], rgba[:, -1], tolerance * target_gap(targets))
    rows = []
    for pair, target in enumerate(pair_target):
        own = slice(pair * repeats, (pair + 1) * repeats)
        pair_labels = labels[own]
        rows.append({
            "pair": pair,
            "target": int(target),
            "correct": float(np.mean(pair_labels == target)),
            "wrong_target": float(np.mean((pair_labels >= 0) & (pair_labels != target))),
            "other": float(np.mean(pair_labels == OTHER)),
            "failed": float(np.mean(pair_labels <= DIED)),
            "mse_at_t": float(np.mean(distances[own, min(horizon, steps), target])),
            "mse_final": float(np.mean(distances[own, -1, target])),
        })
    shape = (len(pair_target), repeats)
    return rows, rgba[:, -1].reshape(*shape, *rgba.shape[-3:]), labels.reshape(shape)


def to_image(rgba):
    """RGBA [..., 4, H, W] drawn over white, as [..., H, W, 3] in [0, 1]."""
    rgba = np.clip(np.nan_to_num(np.asarray(rgba)), 0.0, 1.0)
    rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:4, :, :]
    return np.moveaxis(rgb * alpha + (1.0 - alpha), -3, -1)
