"""Evaluation of emoji NCAs with a learned per-cell firing rate (aNCA, gaNCA).

These models write each cell's firing probability into their last state
channel. Fixed-rate models (NCA, gNCA) are treated as firing at ``FIRE_RATE``
everywhere, so both can be compared on the same measures:

- activity: the mean firing probability over the whole grid at each step
- updates per cell: the sum of the activity over the rollout (expected number of updates)
- quiet step: the first step after which the activity stays below a threshold,
  i.e. when the NCA signals that it has finished

Distances are the RGBA mean squared error to a target, as in ``multi_attractor``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Experiments.emoji import multi_attractor as ma


ADAPTIVE_FAMILIES = {"aNCA", "gaNCA"}
FAMILIES = {"NCA", "gNCA"} | ADAPTIVE_FAMILIES


def summary_row(bundle):
    """``multi_attractor.summary_row`` plus the firing energy settings."""
    cfg = bundle.config
    energy = float(cfg.loss.regularisers.get("firing_energy", 0.0))
    return {
        **ma.summary_row(bundle),
        "firing_energy": energy,
        # Cost of one update of every cell; the training loss averages over run.t steps
        "energy_per_update": energy / cfg.run.t,
    }


def has_learned_firing(model):
    """True for aNCA/gaNCA models, which store their firing probability in the last channel."""
    return getattr(model, "FIRING", "fixed") != "fixed"


def firing_map(model, state):
    """Firing probability of each cell [H, W]: learned, or the fixed ``FIRE_RATE``."""
    if has_learned_firing(model):
        return model.firing_probability(state)
    return jnp.full(state.shape[-2:], getattr(model, "FIRE_RATE", 1.0), dtype=state.dtype)


def damage_mask(height, width, radius):
    """[H, W] disc of ``radius`` pixels at the grid centre."""
    rows, columns = jnp.mgrid[:height, :width]
    return ((rows - (height - 1) / 2) ** 2 + (columns - (width - 1) / 2) ** 2) <= radius**2


def _trajectory(model, state, key, steps, damage_step, damage_radius, record):
    """Roll ``state`` out for ``steps`` updates, zeroing the damage disc at
    ``damage_step`` (-1: never); returns ``record`` of the seed and every step."""
    damage = damage_mask(*state.shape[-2:], damage_radius)

    def step(state, index):
        state = jnp.where((index == damage_step) & damage, 0.0, state)
        state = model(state, lambda x: x, key=jax.random.fold_in(key, index))
        return state, record(state)

    final, history = jax.lax.scan(step, state, jnp.arange(steps))
    first = record(state)
    return final, jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b]), first, history)


@eqx.filter_jit
def activity_rollout(model, state, targets, steps, key, damage_step=-1, damage_radius=0.0):
    """One trajectory, keeping every frame.

    Returns, for the seed and every step, the RGBA [steps + 1, 4, H, W], the
    firing probability [steps + 1, H, W] and the distance to each target
    [steps + 1, target].
    """
    data_channels = targets.shape[1]

    def record(state):
        distance = jnp.mean((state[None, :data_channels] - targets) ** 2, axis=(1, 2, 3))
        return state[:data_channels], firing_map(model, state), distance

    _, history = _trajectory(model, state, key, steps, damage_step, damage_radius, record)
    return history


@eqx.filter_jit
def _activity_traces_batch(model, states, keys, targets, steps, damage_step, damage_radius):
    data_channels = targets.shape[1]

    def record(state):
        distance = jnp.mean((state[None, :data_channels] - targets) ** 2, axis=(1, 2, 3))
        return jnp.mean(firing_map(model, state)), distance

    def one(state, key):
        final, (activity, distances) = _trajectory(
            model, state, key, steps, damage_step, damage_radius, record
        )
        return activity, distances, final[:data_channels]

    return jax.vmap(one)(states, keys)


def activity_traces(model, states, targets, steps, key, damage_step=-1, damage_radius=0.0, memory_gb=1.0):
    """Roll out each seed in ``states`` [N, C, H, W], keeping only summaries.

    Returns the activity [N, steps + 1], the distance to each target
    [N, steps + 1, target] and the final RGBA [N, 4, H, W]. Seeds are run in
    chunks that fit in about ``memory_gb``, as in ``multi_attractor.rollout``.
    """
    states = np.asarray(states, dtype=np.float32)
    count = len(states)
    chunk = min(count, ma.chunk_size(model, *states.shape[-2:], memory_gb))
    keys = jax.random.split(key, count)
    # Pad to whole chunks, so every chunk has the same shape and compiles once
    padding = -count % chunk
    states = np.concatenate([states, np.zeros((padding, *states.shape[1:]), np.float32)])
    keys = jnp.concatenate([keys, jnp.repeat(keys[:1], padding, axis=0)])
    targets = jnp.asarray(targets)
    outputs = []
    for start in range(0, len(states), chunk):
        outputs.append(_activity_traces_batch(
            model, jnp.asarray(states[start : start + chunk]), keys[start : start + chunk],
            targets, int(steps), int(damage_step), float(damage_radius),
        ))
    return tuple(np.concatenate([np.asarray(o[i]) for o in outputs])[:count] for i in range(3))


def quiet_step(activity, threshold):
    """First step from which ``activity`` [steps + 1] stays below ``threshold``; NaN if never."""
    above = np.nonzero(np.asarray(activity) >= threshold)[0]
    if len(above) == 0:
        return 0.0
    last = above[-1] + 1
    return float(last) if last < len(activity) else np.nan


def trace_metrics(activity, distance, t, threshold, damage_step=-1):
    """Measures of one rollout from its activity and distance to its own target, both [steps + 1].

    ``t`` is the training rollout length (``run.t``).
    """
    activity, distance = np.asarray(activity), np.asarray(distance)
    steps = len(activity) - 1
    quiet = quiet_step(activity, threshold)
    row = {
        "updates_per_cell": float(activity[1:].sum()),
        "updates_to_t": float(activity[1 : min(t, steps) + 1].sum()),
        "activity_at_t": float(activity[min(t, steps)]),
        "activity_final": float(activity[-1]),
        "quiet_step": quiet,
        "quiet_step_over_t": quiet / t,
        "mse_at_t": float(distance[min(t, steps)]),
        "mse_final": float(distance[-1]),
        "mse_at_quiet": float(distance[int(quiet)]) if np.isfinite(quiet) else np.nan,
        "mse_min": float(distance.min()),
        "step_of_mse_min": int(distance.argmin()),
    }
    if damage_step >= 0:
        after = slice(damage_step + 1, None)
        row["damage_peak_activity"] = float(activity[after].max())
        row["damage_updates_per_cell"] = float(activity[after].sum())
        row["mse_after_damage"] = float(distance[damage_step + 1])
    return row


def evaluate_seeds(model, seeds, targets, pair_target, t, steps, repeats, key, threshold,
                   damage_step=-1, damage_radius=0.0, memory_gb=1.0):
    """Roll out every training seed ``repeats`` times and measure each rollout.

    Returns ``(rows, activity, distance)``: one row of ``trace_metrics`` per
    rollout (with ``pair``, ``target`` and ``rollout``), and the activity and
    distance to the own target, both [pair, repeat, steps + 1].
    """
    states = np.repeat(seeds, repeats, axis=0)
    activity, distances, _ = activity_traces(
        model, states, targets, steps, key, damage_step, damage_radius, memory_gb
    )
    own = np.repeat(np.asarray(pair_target), repeats)
    distance = distances[np.arange(len(states)), :, own]
    rows = []
    for index in range(len(states)):
        pair, rollout = divmod(index, repeats)
        rows.append({
            "pair": pair, "target": int(own[index]), "rollout": rollout,
            **trace_metrics(activity[index], distance[index], t, threshold, damage_step),
        })
    shape = (len(seeds), repeats, steps + 1)
    return rows, activity.reshape(shape), distance.reshape(shape)
