"""Measures of a trained NCA on held-out signalling task instances.

Each held-out instance is rolled out several times, for several times the
trained rollout length. From each rollout:

- accuracy: fraction of cells (tissue in the target or the prediction) whose
  RGBA is within ``tolerance`` of the target, at each observed frame and at
  each evaluation horizon (1x, 2x, 4x the trained rollout)
- arrival step: first step at which 90% of the probe cells (e.g. the far end
  of a bar) match the final target. Its spread over repeated rollouts of the
  same instance shows how precise the timing is
- updates per cell and quiet step (see Experiments/emoji/adaptive_firing.py):
  for aNCA/gaNCA, whether the model does less work on easy instances and
  stops once it is done

Each is also plotted against the instance difficulty (bar length, path
length, ...), which shows the speed limit of a local NCA.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Experiments.emoji.adaptive_firing import firing_map, quiet_step
from Experiments.signalling.tasks import sample_batch

# Seed of the held-out instances; training pools draw from other seeds
EVALUATION_SEED = 12345
ARRIVAL_ACCURACY = 0.9


def cell_accuracy(prediction, target, tolerance=0.2, mask=None):
    """Fraction of cells in ``mask`` where every RGBA channel of ``prediction`` [4, H, W] is within ``tolerance`` of ``target``.

    By default ``mask`` is the cells with tissue (alpha > 0.1) in either image.
    """
    if mask is None:
        mask = (target[3] > 0.1) | (prediction[3] > 0.1)
    correct = jnp.all(jnp.abs(prediction - target) < tolerance, axis=0)
    return jnp.sum(correct & mask) / jnp.maximum(jnp.sum(mask), 1)


@eqx.filter_jit
def _rollouts(model, states, keys, targets, probes, steps, tolerance):
    """Roll out ``states`` [N, C, H, W] for ``steps``, keeping only per-step summaries.

    Returns the mean firing probability [N, steps + 1], the accuracy against
    each target frame [N, steps + 1, frames] and the probe accuracy against
    the last frame [N, steps + 1].
    """

    def record(state, targets, probe):
        rgba = state[:4]
        frame_accuracy = jax.vmap(lambda target: cell_accuracy(rgba, target, tolerance))(targets)
        probe_accuracy = cell_accuracy(rgba, targets[-1], tolerance, mask=probe)
        return jnp.mean(firing_map(model, state)), frame_accuracy, probe_accuracy

    def one(state, key, targets, probe):
        def step(state, index):
            state = model(state, lambda x: x, key=jax.random.fold_in(key, index))
            return state, record(state, targets, probe)

        _, history = jax.lax.scan(step, state, jnp.arange(steps))
        first = record(state, targets, probe)
        return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b]), first, history)

    return jax.vmap(one)(states, keys, targets, probes)


def first_step_above(trace, level):
    """First index at which ``trace`` reaches ``level``; NaN if never."""
    above = np.nonzero(np.asarray(trace) >= level)[0]
    return float(above[0]) if len(above) else np.nan


def _correlation(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.std(a[finite]) == 0 or np.std(b[finite]) == 0:
        return np.nan
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def evaluate_task(model, key, *, task, size, params, target_repeats, t, samples, repeats, horizons,
                  tolerance=0.2, quiet_threshold=0.05):
    """Measure ``model`` on held-out instances of ``task``; returns (scalars, images) for W&B.

    ``t`` is the number of NCA steps between observed frames (run.t).
    """
    data, difficulty, probe = sample_batch(task, EVALUATION_SEED, samples, size, target_repeats, **params)
    frames = data.shape[1] - 1
    trained_steps = t * frames
    steps = max(horizons) * trained_steps
    states = np.zeros((samples, model.N_CHANNELS, size, size), dtype=np.float32)
    states[:, :4] = data[:, 0]
    states = jax.vmap(model.prepare_pool_state)(jnp.repeat(jnp.asarray(states), repeats, axis=0))
    targets = jnp.repeat(jnp.asarray(data[:, 1:]), repeats, axis=0)
    probes = jnp.repeat(jnp.asarray(probe), repeats, axis=0)
    keys = jax.random.split(key, samples * repeats)
    activity, frame_accuracy, probe_accuracy = (
        np.asarray(a) for a in _rollouts(model, states, keys, targets, probes, int(steps), float(tolerance))
    )
    final_accuracy = frame_accuracy[..., -1]
    difficulty = np.repeat(difficulty, repeats)

    arrival = np.array([first_step_above(trace, ARRIVAL_ACCURACY) for trace in probe_accuracy])
    quiet = np.array([quiet_step(trace, quiet_threshold) for trace in activity])
    accuracy_at = {h: final_accuracy[:, h * trained_steps] for h in horizons}
    scalars = {
        f"accuracy_frame_{k}": float(frame_accuracy[:, t * k, k - 1].mean()) for k in range(1, frames + 1)
    }
    scalars.update({f"accuracy_x{h}": float(a.mean()) for h, a in accuracy_at.items()})
    scalars.update({
        "arrival_step": float(np.nanmean(arrival)) if np.isfinite(arrival).any() else np.nan,
        "arrival_fraction": float(np.isfinite(arrival).mean()),
        # Mean over instances of the spread of arrival over repeated rollouts
        "arrival_spread": float(np.nanmean(np.nanstd(arrival.reshape(samples, repeats), axis=1)))
        if np.isfinite(arrival).any() else np.nan,
        "updates_per_cell": float(activity[:, 1:trained_steps + 1].sum(axis=1).mean()),
        "updates_per_cell_after": float(activity[:, trained_steps + 1:].sum(axis=1).mean()),
        "quiet_step": float(np.nanmean(quiet)) if np.isfinite(quiet).any() else np.nan,
        "quiet_fraction": float(np.isfinite(quiet).mean()),
        "corr_difficulty_accuracy": _correlation(difficulty, accuracy_at[horizons[0]]),
        "corr_difficulty_arrival": _correlation(difficulty, arrival),
        "corr_difficulty_quiet_step": _correlation(difficulty, quiet),
    })
    figure = plot_evaluation(
        difficulty, accuracy_at[horizons[0]], arrival, quiet,
        final_accuracy.mean(axis=0), activity.mean(axis=0), trained_steps, t * np.arange(1, frames + 1),
    )
    return scalars, {"difficulty": figure}


def plot_evaluation(difficulty, accuracy, arrival, quiet, mean_accuracy, mean_activity, trained_steps,
                    observation_steps):
    """Accuracy, arrival step and quiet step against difficulty, and the mean accuracy and activity against step."""
    import matplotlib.pyplot as plt
    from NCA.trainer.logging.diagnostics import plot_to_image

    figure, axes = plt.subplots(1, 4, figsize=(16, 3.6), dpi=120)
    panels = (
        (accuracy, "Accuracy after trained rollout"),
        (arrival, "Arrival step at probe"),
        (quiet, "Quiet step"),
    )
    for ax, (values, label) in zip(axes, panels):
        ax.scatter(difficulty, values, s=8, alpha=0.6)
        ax.set_xlabel("Difficulty (distance, cells)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    for ax in axes[1:3]:
        ax.axhline(trained_steps, color="grey", linewidth=0.5)
    steps = np.arange(len(mean_accuracy))
    ax = axes[3]
    ax.plot(steps, mean_accuracy, label="accuracy (final target)")
    ax.plot(steps, mean_activity, label="firing probability")
    for step in observation_steps:
        ax.axvline(step, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Step")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.25)
    figure.tight_layout()
    return plot_to_image(figure)
