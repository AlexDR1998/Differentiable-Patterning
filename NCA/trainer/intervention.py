"""State-preserving interventions applied at the NCA update boundary."""

import jax
import jax.numpy as jnp

from Common.model.boundary import no_boundary


def nodal_read_block_mask(intervention_time, state_count, *, time_offset=0):
    """Select 12-hour developmental slots at or after an intervention."""

    time_indices = jnp.arange(state_count) + time_offset
    return (intervention_time >= 0) & (time_indices >= intervention_time // 12)


def apply_model_with_blocked_channel(
    nca, state, boundary_callback, key, channel, blocked
):
    """Apply an NCA update without exposing one recurrent channel to perception."""

    def blocked_update(_):
        read_state = state.at[channel].set(0.0)
        updated_read_state = nca(read_state, no_boundary(), key)
        return boundary_callback(state + (updated_read_state - read_state))

    def ordinary_update(_):
        return nca(state, boundary_callback, key)

    return jax.lax.cond(blocked, blocked_update, ordinary_update, operand=None)
