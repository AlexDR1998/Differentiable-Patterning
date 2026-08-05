"""Pure stochastic NCA augmentation transforms.

All functions return new PyTrees and accept an explicit PRNG key. They do not
modify augmenter instances or derive randomness from wall-clock time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def add_noise(data, amount: float, key, mode: str = "full", observable_channels: int | None = None):
    """Blend Gaussian noise into trajectory leaves."""

    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be between zero and one")
    leaves, treedef = jtu.tree_flatten(data)
    keys = jax.random.split(key, len(leaves))
    noisy = [
        (1.0 - amount) * value + amount * jax.random.normal(subkey, value.shape)
        for value, subkey in zip(leaves, keys)
    ]
    result = jtu.tree_unflatten(treedef, noisy)
    if mode == "full":
        return result
    if observable_channels is None:
        raise ValueError("observable_channels is required for partial noise")
    if mode not in {"observable", "hidden"}:
        raise ValueError("mode must be 'full', 'observable', or 'hidden'")

    def restore(original, modified):
        if mode == "observable":
            return modified.at[..., observable_channels:, :, :].set(original[..., observable_channels:, :, :])
        return modified.at[..., :observable_channels, :, :].set(original[..., :observable_channels, :, :])

    return jtu.tree_map(restore, data, result)


def propagate_pool(x):
    """Move each pool trajectory one step forward, preserving its shape."""

    if hasattr(x, "ndim"):
        if x.ndim < 2:
            raise ValueError("A stacked pool must have batch and time axes")
        return x.at[:, 1:].set(x[:, :-1])
    return jtu.tree_map(lambda value: value.at[1:].set(value[:-1]), x)


def reinject_observations(x, x_true, observable_channels: int, key, fraction: float = 0.5):
    """Reproduce the legacy pool propagation and observation reinjection.

    Exactly ``floor(fraction * eligible_slots)`` batch/time slots are selected
    globally. A stacked array uses axes ``(batch, time, ...)``; PyTree leaves
    each represent one trajectory with a leading time axis.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    x = propagate_pool(x)

    if hasattr(x, "ndim"):
        if x.shape[:2] != x_true.shape[:2]:
            raise ValueError("x and x_true must have matching batch/time axes")
        batch_count, time_count = x.shape[:2]
        x = x.at[:, 0].set(x_true[:, 0])
        eligible = batch_count * max(time_count - 1, 0)
        inject_count = int(eligible * fraction)
        if inject_count == 0:
            return x
        scores = jax.random.uniform(key, shape=(eligible,))
        mask = jnp.zeros((eligible,), dtype=bool).at[
            jnp.argsort(scores)[:inject_count]
        ].set(True)
        mask = mask.reshape((batch_count, time_count - 1, 1, 1, 1))
        observed = jnp.where(
            mask,
            x_true[:, 1:, :observable_channels],
            x[:, 1:, :observable_channels],
        )
        return x.at[:, 1:, :observable_channels].set(observed)

    leaves, xdef = jtu.tree_flatten(x)
    true_leaves, truedef = jtu.tree_flatten(x_true)
    if xdef != truedef or len(leaves) != len(true_leaves):
        raise ValueError("x and x_true must have matching PyTree structures")
    eligible_counts = [max(value.shape[0] - 1, 0) for value in leaves]
    eligible = sum(eligible_counts)
    inject_count = int(eligible * fraction)
    flat_mask = jnp.zeros((eligible,), dtype=bool)
    if inject_count > 0:
        scores = jax.random.uniform(key, shape=(eligible,))
        flat_mask = flat_mask.at[jnp.argsort(scores)[:inject_count]].set(True)

    result = []
    offset = 0
    for value, truth, count in zip(leaves, true_leaves, eligible_counts):
        if value.shape != truth.shape:
            raise ValueError("x and x_true leaves must have matching shapes")
        value = value.at[0].set(truth[0])
        if count:
            mask = flat_mask[offset : offset + count, None, None, None]
            observed = jnp.where(
                mask,
                truth[1:, :observable_channels],
                value[1:, :observable_channels],
            )
            value = value.at[1:, :observable_channels].set(observed)
        result.append(value)
        offset += count
    return jtu.tree_unflatten(xdef, result)


def bernoulli_reinject_observations(
    x,
    x_true,
    observable_channels: int,
    key,
    probability,
    global_batch_indices=None,
    global_batch_count=None,
):
    """Propagate a pool and independently reinject each eligible slot."""

    x = propagate_pool(x)
    if hasattr(x, "ndim"):
        batch_count, time_count = x.shape[:2]
        x = x.at[:, 0].set(x_true[:, 0])
        global_count = batch_count if global_batch_count is None else global_batch_count
        indices = (
            jnp.arange(batch_count)
            if global_batch_indices is None
            else global_batch_indices
        )
        mask = jax.random.bernoulli(
            key, probability, (global_count, time_count - 1)
        )[indices, :, None, None, None]
        observed = jnp.where(
            mask,
            x_true[:, 1:, :observable_channels],
            x[:, 1:, :observable_channels],
        )
        return x.at[:, 1:, :observable_channels].set(observed)

    leaves, treedef = jtu.tree_flatten(x)
    true_leaves, true_def = jtu.tree_flatten(x_true)
    if treedef != true_def or len(leaves) != len(true_leaves):
        raise ValueError("x and x_true must have matching PyTree structures")
    batch_count = len(leaves)
    time_count = leaves[0].shape[0]
    if any(value.shape[0] != time_count for value in leaves):
        raise ValueError("Bernoulli reinjection requires a common trajectory length")
    global_count = batch_count if global_batch_count is None else global_batch_count
    indices = (
        jnp.arange(batch_count)
        if global_batch_indices is None
        else global_batch_indices
    )
    masks = jax.random.bernoulli(
        key, probability, (global_count, time_count - 1)
    )[indices]
    result = []
    for value, truth, mask in zip(leaves, true_leaves, masks):
        value = value.at[0].set(truth[0])
        observed = jnp.where(
            mask[:, None, None, None],
            truth[1:, :observable_channels],
            value[1:, :observable_channels],
        )
        result.append(value.at[1:, :observable_channels].set(observed))
    return jtu.tree_unflatten(treedef, result)


def terminal_carry(x, previous_terminal, probability: float, key):
    """Keep each trajectory's previous terminal state with given probability."""

    if hasattr(x, "ndim"):
        carry = jax.random.bernoulli(key, probability, (x.shape[0],))
        replacement = jnp.where(
            carry.reshape((x.shape[0],) + (1,) * (x.ndim - 2)),
            previous_terminal,
            x[:, -1],
        )
        return x.at[:, -1].set(replacement)
    leaves, treedef = jtu.tree_flatten(x)
    carry = jax.random.bernoulli(key, probability, (len(leaves),))
    previous_leaves, previous_def = jtu.tree_flatten(previous_terminal)
    if previous_def != treedef:
        raise ValueError("x and previous_terminal must have matching PyTree structures")
    result = [
        value.at[-1].set(jnp.where(keep, previous, value[-1]))
        for value, previous, keep in zip(leaves, previous_leaves, carry)
    ]
    return jtu.tree_unflatten(treedef, result)


def scheduled_probability(i, start, schedule, initial, final):
    """Linearly interpolate a probability after ``start``."""

    progress = jnp.clip((i - start) / max(schedule, 1), 0.0, 1.0)
    return jnp.where(i < start, 0.0, initial + progress * (final - initial))
