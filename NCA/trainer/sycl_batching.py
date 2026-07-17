"""Portable tree batching helpers used by the SYCL trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def apply_flat_batched_nca(nca, x, callbacks, key_array, fallback):
    """Flatten compatible tree leaves, update them once, and restore the tree."""
    leaves, tree_definition = jtu.tree_flatten(x)
    if not leaves:
        return x
    key_leaves = tree_definition.flatten_up_to(key_array)
    callback_leaves = tree_definition.flatten_up_to(callbacks)

    compatible = all(
        leaf.ndim == 4
        and leaf.shape[1:] == leaves[0].shape[1:]
        and leaf.dtype == leaves[0].dtype
        for leaf in leaves
    )
    if not compatible:
        return fallback(x, callbacks, key_array)

    batch_sizes = [leaf.shape[0] for leaf in leaves]
    flat_state = jnp.concatenate(leaves, axis=0)
    flat_keys = jnp.concatenate(key_leaves, axis=0)
    flat_updated = nca.batched_call(flat_state, flat_keys)

    updated_leaves = []
    offset = 0
    for size, callback in zip(batch_sizes, callback_leaves):
        updated = flat_updated[offset : offset + size]
        updated = jax.vmap(callback)(updated)
        updated_leaves.append(updated)
        offset += size
    return jtu.tree_unflatten(tree_definition, updated_leaves)


__all__ = ["apply_flat_batched_nca"]
