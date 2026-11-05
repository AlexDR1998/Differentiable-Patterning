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


def rollout_trajectory(
    model,
    initial_state: Array,
    boundary_mask: Array,
    boundary_mode: str,
    key: Array,
    total_steps: int,
) -> Array:
    """Every state ``[total_steps + 1, C, H, W]``, starting with ``initial_state``.

    Uses the same per-step keys as ``rollout_final``, so the last state is
    the one ``rollout_final`` returns. Not meant for differentiation.
    """

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    def step(state, step_index):
        step_key = jax.random.fold_in(key, step_index)
        state = model(state, boundary_callback, key=step_key)
        return state, state

    _, states = jax.lax.scan(step, initial_state, jnp.arange(total_steps))
    return jnp.concatenate([initial_state[None], states], axis=0)
