"""Portable tree batching helpers used by the SYCL trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def shape_probe_rollout(state, rollout_steps):
    """Return pmap-global placeholders without executing an accelerator call."""
    trajectory = jnp.broadcast_to(
        state[:, None],
        (state.shape[0], rollout_steps, *state.shape[1:]),
    )
    return state, trajectory


def shape_probe_losses(state):
    """Return the per-tile, per-timestep loss shape used by filter_pmap."""
    return jnp.zeros(state.shape[:2], dtype=state.dtype)


def apply_flat_batched_nca(nca, x, callbacks, key_array, fallback):
    """Batch each compatible leaf over N while keeping outer B leaves separate."""
    leaves, tree_definition = jtu.tree_flatten(x)
    if not leaves:
        return x
    key_leaves = tree_definition.flatten_up_to(key_array)
    callback_leaves = tree_definition.flatten_up_to(callbacks)

    compatible = all(leaf.ndim == 4 for leaf in leaves)
    if not compatible:
        return fallback(x, callbacks, key_array)

    updated_leaves = []
    for state, keys, callback in zip(leaves, key_leaves, callback_leaves):
        updated = nca.batched_call(state, keys)
        updated = jax.vmap(callback)(updated)
        updated_leaves.append(updated)
    return jtu.tree_unflatten(tree_definition, updated_leaves)


__all__ = [
    "apply_flat_batched_nca",
    "shape_probe_losses",
    "shape_probe_rollout",
]
