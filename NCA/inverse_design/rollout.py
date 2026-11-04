"""Differentiable NCA rollouts that keep only the final state."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from NCA.trainer.intervention import _apply_boundary


def rollout_final(
    model,
    initial_state: Array,
    boundary_mask: Array,
    boundary_mode: str,
    key: Array,
    total_steps: int,
    scan_kind: str = "lax",
    checkpoints: int | None = None,
) -> Array:
    """Return only the final state while differentiating through all updates.

    With ``scan_kind="checkpointed"`` the backward pass reruns steps from a
    few saved states (``checkpoints``; None lets equinox choose) instead of
    keeping every state, so long rollouts fit in memory.
    """

    if total_steps == 0:
        return initial_state

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    def step(state, step_index):
        step_key = jax.random.fold_in(key, step_index)
        return model(state, boundary_callback, key=step_key), None

    final_state, _ = eqx.internal.scan(
        step,
        initial_state,
        jnp.arange(total_steps),
        kind=scan_kind,
        checkpoints=checkpoints,
    )
    return final_state
