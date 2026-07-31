"""Pure trajectory and pool operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def split_trajectory(data, n_steps: int = 1, real_to_latent=lambda x: x):
    """Return latent initial states and observed target states.

    ``data`` is a PyTree whose leaves have shape ``(time, channels, height,
    width)``. The input is never mutated.
    """

    if n_steps < 1:
        raise ValueError("n_steps must be at least one")
    x = jtu.tree_map(lambda value: real_to_latent(value[:-n_steps]), data)
    y = jtu.tree_map(lambda value: value[n_steps:], data)
    return x, y


def duplicate_batches(data, repetitions: int):
    """Duplicate a PyTree of trajectories along its leading batch structure."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if isinstance(data, list):
        # The NCA augmenters represent variable-size batches as a list of
        # trajectory leaves, so duplication extends that outer batch list.
        return [jnp.asarray(leaf) for _ in range(repetitions) for leaf in data]
    if isinstance(data, tuple):
        return tuple(jnp.asarray(leaf) for _ in range(repetitions) for leaf in data)
    return jnp.repeat(jnp.asarray(data), repetitions, axis=0)


def pad_spatial(data, amount: int | tuple[int, int, int, int]):
    """Pad trajectory leaves spatially with zeros."""

    if isinstance(amount, int):
        amount = (amount, amount, amount, amount)
    top, bottom, left, right = amount
    padding = ((0, 0), (0, 0), (top, bottom), (left, right))
    return jtu.tree_map(lambda value: jnp.pad(value, padding), data)
