"""State-preserving interventions applied at the NCA update boundary."""

import equinox as eqx
import jax
import jax.numpy as jnp

from Common.model.boundary import no_boundary


def intervention_slot(intervention_time, observation_times=None):
    """Index of the transition slot in which an intervention time falls.

    Without ``observation_times`` this is the historical 12-hour convention
    ``time // 12``. With them, it is the last slot starting at or before
    ``time`` (identical on a uniform 12 h grid). Works on Python scalars and
    traced arrays; negative times (no intervention) are handled by callers.
    """
    if observation_times is None:
        return intervention_time // 12
    times = jnp.asarray(observation_times, dtype=jnp.float32)
    return jnp.searchsorted(times, intervention_time, side="right") - 1


def nodal_read_block_mask(
    intervention_time, state_count, *, time_offset=0, observation_times=None
):
    """Select developmental slots at or after an intervention."""

    time_indices = jnp.arange(state_count) + time_offset
    return (intervention_time >= 0) & (
        time_indices >= intervention_slot(intervention_time, observation_times)
    )


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


@eqx.filter_jit
def rollout_model(nca, initial_state, boundary_callback, key, total_steps):
    """Compile and run a complete ordinary NCA trajectory.

    The returned trajectory includes ``initial_state`` at index zero, matching
    the existing ``NCA.run`` convention.
    """

    def step(carry, step_index):
        state, previous_key = carry
        step_key = jax.random.fold_in(previous_key, step_index)
        state = nca(state, boundary_callback, key=step_key)
        return (state, step_key), state

    _, states = jax.lax.scan(
        step,
        (initial_state, key),
        jnp.arange(total_steps),
    )
    return jnp.concatenate((initial_state[None], states), axis=0)


@eqx.filter_jit
def rollout_model_with_blocked_channel(
    nca,
    initial_state,
    boundary_callback,
    key,
    total_steps,
    channel,
    knockout_step,
):
    """Compile an NCA trajectory with recurrent channel reads blocked on time."""

    def step(carry, step_index):
        state, previous_key = carry
        step_key = jax.random.fold_in(previous_key, step_index)
        state = apply_model_with_blocked_channel(
            nca,
            state,
            boundary_callback,
            step_key,
            channel,
            step_index >= knockout_step,
        )
        return (state, step_key), state

    _, states = jax.lax.scan(
        step,
        (initial_state, key),
        jnp.arange(total_steps),
    )
    return jnp.concatenate((initial_state[None], states), axis=0)


def _apply_boundary(state, boundary_mask, boundary_mode):
    if boundary_mode == "none":
        return state
    if boundary_mode == "hard":
        return state * boundary_mask.reshape(
            (1,) * (state.ndim - boundary_mask.ndim) + boundary_mask.shape
        )
    if boundary_mode == "soft":
        return state.at[-boundary_mask.shape[0]:].set(boundary_mask)
    raise ValueError(f"Unknown boundary mode {boundary_mode!r}")


def _store_observation(frames, state, step, observation_steps):
    matches = observation_steps == step
    slot = jnp.argmax(matches)
    return jax.lax.cond(
        jnp.any(matches),
        lambda values: values.at[slot].set(state),
        lambda values: values,
        frames,
    )


@eqx.filter_jit
def rollout_model_sampled(
    nca,
    initial_state,
    boundary_mask,
    boundary_mode,
    key,
    total_steps,
    observation_steps,
):
    """Compile a rollout while retaining only requested trajectory states."""

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    frames = jnp.zeros(
        (observation_steps.shape[0], *initial_state.shape),
        dtype=initial_state.dtype,
    )
    frames = _store_observation(frames, initial_state, 0, observation_steps)

    def step(carry, step_index):
        state, previous_key, stored_frames = carry
        step_key = jax.random.fold_in(previous_key, step_index)
        state = nca(state, boundary_callback, key=step_key)
        stored_frames = _store_observation(
            stored_frames,
            state,
            step_index + 1,
            observation_steps,
        )
        return (state, step_key, stored_frames), None

    (state, _, frames), _ = jax.lax.scan(
        step,
        (initial_state, key, frames),
        jnp.arange(total_steps),
    )
    del state
    return frames


@eqx.filter_jit
def rollout_model_with_blocked_channel_sampled(
    nca,
    initial_state,
    boundary_mask,
    boundary_mode,
    key,
    total_steps,
    channel,
    knockout_step,
    observation_steps,
):
    """Compile a blocked-channel rollout and retain only requested states."""

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    frames = jnp.zeros(
        (observation_steps.shape[0], *initial_state.shape),
        dtype=initial_state.dtype,
    )
    frames = _store_observation(frames, initial_state, 0, observation_steps)

    def step(carry, step_index):
        state, previous_key, stored_frames = carry
        step_key = jax.random.fold_in(previous_key, step_index)
        state = apply_model_with_blocked_channel(
            nca,
            state,
            boundary_callback,
            step_key,
            channel,
            step_index >= knockout_step,
        )
        stored_frames = _store_observation(
            stored_frames,
            state,
            step_index + 1,
            observation_steps,
        )
        return (state, step_key, stored_frames), None

    (state, _, frames), _ = jax.lax.scan(
        step,
        (initial_state, key, frames),
        jnp.arange(total_steps),
    )
    del state
    return frames


@eqx.filter_jit
def rollout_model_with_blocked_channel_pattern_counts(
    nca,
    initial_state,
    boundary_mask,
    boundary_mode,
    key,
    total_steps,
    channel,
    knockout_step,
    fate_channels,
    fate_thresholds,
    pixel_groups,
    n_groups,
):
    """Roll out an intervention while retaining only marker-pattern counts.

    At every step, each pixel's high/low pattern of the ``fate_channels``
    (high = above ``fate_thresholds``; bit ``i`` = ``fate_channels[i]``) is
    counted within its group in ``pixel_groups`` (``[H, W]``, e.g. radial
    rings; -1 = left out). Returns ``[total_steps + 1, n_groups,
    2 ** len(fate_channels)]``; cell-type shares follow from these counts
    (see ``Common.dataloader.cell_type_shares``).
    """

    def boundary_callback(state):
        return _apply_boundary(state, boundary_mask, boundary_mode)

    n_patterns = 2 ** fate_channels.shape[0]
    bits = 2 ** jnp.arange(fate_channels.shape[0])
    groups = pixel_groups.astype(jnp.int32)

    def measure(state):
        high = state[fate_channels] > fate_thresholds[:, None, None]
        codes = jnp.sum(high * bits[:, None, None], axis=0)
        # Left-out pixels go to one extra bin that is dropped.
        index = jnp.where(groups >= 0, groups * n_patterns + codes, n_groups * n_patterns)
        counts = jnp.bincount(index.ravel(), length=n_groups * n_patterns + 1)
        return counts[:-1].reshape(n_groups, n_patterns)

    initial_counts = measure(initial_state)

    def step(carry, step_index):
        state, previous_key = carry
        step_key = jax.random.fold_in(previous_key, step_index)
        state = apply_model_with_blocked_channel(
            nca,
            state,
            boundary_callback,
            step_key,
            channel,
            step_index >= knockout_step,
        )
        return (state, step_key), measure(state)

    (_, _), counts = jax.lax.scan(
        step,
        (initial_state, key),
        jnp.arange(total_steps),
    )
    return jnp.concatenate((initial_counts[None], counts), axis=0)
