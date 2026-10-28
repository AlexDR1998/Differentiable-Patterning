"""Which attractor does a final NCA state end up at?

A final state is labelled with the attractor it is closest to (mean squared
error of the observed channels over the whole grid) when that distance is below
``tolerance`` times the smallest distance between two attractors, so the
threshold does not depend on the patterns used. Otherwise it is ``OTHER``,
``DIED`` (no living cell) or ``DIVERGED`` (non-finite). Used by the impulse
validation and by ``Experiments/emoji/multi_attractor.py``. ``switch_steps``
times when an impulse rollout switches from its source to its target.
"""

import numpy as np

# Labels of final states that are not one of the attractors
OTHER, DIED, DIVERGED = -1, -2, -3
OUTCOME_NAMES = {OTHER: "other", DIED: "died", DIVERGED: "diverged"}
# A cell is alive when its alpha is above this, as in the usual NCA convention
ALIVE_ALPHA = 0.1


def mse(a, b):
    """Mean squared error over the last three axes."""
    return np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=(-3, -2, -1))


def target_gap(targets):
    """Smallest distance between two different targets (inf for one target)."""
    gaps = [mse(targets[i], targets[j]) for i in range(len(targets)) for j in range(i + 1, len(targets))]
    return float(min(gaps)) if gaps else float("inf")


def classify(distances, final_rgba, threshold):
    """Label of each final state: the index of its target when the distance is
    below ``threshold``, else ``OTHER``; ``DIED`` when no cell is alive and
    ``DIVERGED`` for non-finite states.

    ``distances`` [..., target] are the final distances, ``final_rgba``
    [..., 4, H, W] the final states.
    """
    distances = np.asarray(distances)
    best = distances.min(axis=-1)
    labels = np.where(best < threshold, distances.argmin(axis=-1), OTHER)
    alive = np.nan_to_num(np.asarray(final_rgba)[..., 3, :, :], nan=1.0).max(axis=(-2, -1)) > ALIVE_ALPHA
    labels = np.where(alive, labels, DIED)
    return np.where(np.isfinite(best), labels, DIVERGED)


def switch_steps(trajectory, sources, targets, frame_every):
    """NCA step from which each rollout stays nearer its target than its source.

    ``trajectory`` [batch, frames, C, H, W] holds the state after every
    ``frame_every`` steps; ``sources`` and ``targets`` [batch, >= C, H, W] are
    compared on the trajectory's channels, as for the switch rate. Rollouts
    that end nearer the source are NaN.
    """
    trajectory = np.asarray(trajectory)
    channels = trajectory.shape[2]
    to_target = mse(trajectory, np.asarray(targets)[:, None, :channels])
    to_source = mse(trajectory, np.asarray(sources)[:, None, :channels])
    nearer = to_target < to_source
    # stays[:, f]: nearer the target in frame f and every later frame
    stays = np.flip(np.cumprod(np.flip(nearer, axis=1), axis=1), axis=1).astype(bool)
    steps = (np.argmax(stays, axis=1) + 1) * frame_every
    return np.where(stays[:, -1], steps, np.nan)


def outcome_fractions(final_rgba, attractors, tolerance):
    """Fraction of final states [batch, 4, H, W] at each attractor [K, 4, H, W],
    then other, died and diverged: an array of length K + 3."""
    final_rgba = np.asarray(final_rgba)
    distances = mse(final_rgba[:, None], np.asarray(attractors)[None])
    labels = classify(distances, final_rgba, tolerance * target_gap(attractors))
    order = [*range(len(attractors)), OTHER, DIED, DIVERGED]
    return np.array([np.mean(labels == label) for label in order])
