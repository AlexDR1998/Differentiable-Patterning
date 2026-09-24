"""Memory-conscious differentiable NCA rollouts."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from NCA.trainer.intervention import _apply_boundary


@eqx.filter_jit
def rollout_final(
    model,
    initial_state: Array,
    boundary_mask: Array,
    boundary_mode: str,
    key: Array,
    total_steps: int,
) -> Array:
    """Return only the final state while differentiating through all updates."""

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    def step(state, step_index):
        step_key = jax.random.fold_in(key, step_index)
        return model(state, boundary_callback, key=step_key), None

    final_state, _ = jax.lax.scan(step, initial_state, jnp.arange(total_steps))
    return final_state
