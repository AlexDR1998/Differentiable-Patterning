"""Portable tree batching helpers used by the SYCL trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.intervention import nodal_read_block_mask


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


def apply_flat_batched_nca_interventions(
    nca,
    x,
    callbacks,
    key_array,
    intervention_times,
    nodal_channel,
    *,
    time_offset=0,
):
    """Apply mixed NODAL read blocks without vmapping model parameters."""
    leaves, tree_definition = jtu.tree_flatten(x)
    key_leaves = tree_definition.flatten_up_to(key_array)
    callback_leaves = tree_definition.flatten_up_to(callbacks)
    if len(leaves) != len(intervention_times):
        raise ValueError("Intervention times must match the outer training batch")

    updated_leaves = []
    for states, keys, callback, intervention_time in zip(
        leaves, key_leaves, callback_leaves, intervention_times
    ):
        blocked = nodal_read_block_mask(
            intervention_time, states.shape[0], time_offset=time_offset
        )
        if (
            intervention_time < 0
            or intervention_time // 12 - time_offset >= states.shape[0]
        ):
            updated_leaves.append(
                jax.vmap(callback)(nca.batched_call(states, keys))
            )
            continue

        read_states = states.at[:, nodal_channel].set(0.0)
        blocked_updates = nca.batched_call(read_states, keys)
        blocked_updates = jax.vmap(callback)(
            states + (blocked_updates - read_states)
        )
        if intervention_time // 12 <= time_offset:
            updated_leaves.append(blocked_updates)
            continue

        ordinary = jax.vmap(callback)(nca.batched_call(states, keys))
        selection = jnp.reshape(
            blocked, (blocked.shape[0],) + (1,) * (states.ndim - 1)
        )
        updated_leaves.append(jnp.where(selection, blocked_updates, ordinary))
    return jtu.tree_unflatten(tree_definition, updated_leaves)


__all__ = [
    "apply_flat_batched_nca",
    "apply_flat_batched_nca_interventions",
]
