"""Portable tree batching helpers used by the SYCL trainer."""

from __future__ import annotations

import jax
import jax.tree_util as jtu


def apply_flat_batched_nca(nca, x, callbacks, key_array, fallback):
    """Update each compatible outer-B leaf with one native batched call.

    State leaves have shape ``[N,C,H,W]`` and key leaves ``[N,2]``. Leaves of
    another rank use ``fallback`` so nonstandard model variants retain the
    reference trainer's behaviour. The returned PyTree matches ``x``.
    """
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


__all__ = ["apply_flat_batched_nca"]
