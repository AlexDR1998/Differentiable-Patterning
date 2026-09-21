import equinox as eqx
import jax
import jax.numpy as jnp

from Common.model.boundary import model_boundary, no_boundary
from NCA.trainer.intervention import (
    apply_model_with_blocked_channel,
    rollout_model,
    rollout_model_with_blocked_channel,
    rollout_model_sampled,
    rollout_model_with_blocked_channel_sampled,
)


class AdditiveModel(eqx.Module):
    def __call__(self, state, boundary_callback, key=None):
        update = jax.random.uniform(key, state.shape)
        return boundary_callback(state + update)


def test_compiled_rollout_matches_existing_loop():
    model = AdditiveModel()
    initial = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    key = jax.random.PRNGKey(4)
    callback = no_boundary()
    expected = [initial]
    state = initial
    rolling_key = key
    for step in range(4):
        rolling_key = jax.random.fold_in(rolling_key, step)
        state = model(state, callback, key=rolling_key)
        expected.append(state)

    actual = rollout_model(model, initial, callback, key, 4)

    assert jnp.allclose(actual, jnp.stack(expected))


def test_compiled_blocked_rollout_matches_existing_loop():
    model = AdditiveModel()
    initial = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    key = jax.random.PRNGKey(7)
    callback = no_boundary()
    expected = [initial]
    state = initial
    rolling_key = key
    for step in range(4):
        rolling_key = jax.random.fold_in(rolling_key, step)
        state = apply_model_with_blocked_channel(
            model,
            state,
            callback,
            rolling_key,
            channel=1,
            blocked=step >= 2,
        )
        expected.append(state)

    actual = rollout_model_with_blocked_channel(
        model,
        initial,
        callback,
        key,
        4,
        channel=1,
        knockout_step=jnp.asarray(2, dtype=jnp.int32),
    )

    assert jnp.allclose(actual, jnp.stack(expected))


def test_sampled_rollouts_return_only_requested_boundary_applied_states():
    model = AdditiveModel()
    initial = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    key = jax.random.PRNGKey(11)
    boundary_mask = jnp.full((1, 2, 2), 3.0)
    callback = model_boundary(boundary_mask)
    observation_steps = jnp.asarray((0, 2, 4), dtype=jnp.int32)

    full = rollout_model(model, initial, callback, key, 4)
    sampled = rollout_model_sampled(
        model,
        initial,
        boundary_mask,
        "soft",
        key,
        4,
        observation_steps,
    )
    full_blocked = rollout_model_with_blocked_channel(
        model,
        initial,
        callback,
        key,
        4,
        channel=1,
        knockout_step=jnp.asarray(2, dtype=jnp.int32),
    )
    sampled_blocked = rollout_model_with_blocked_channel_sampled(
        model,
        initial,
        boundary_mask,
        "soft",
        key,
        4,
        channel=1,
        knockout_step=jnp.asarray(2, dtype=jnp.int32),
        observation_steps=observation_steps,
    )

    assert sampled.shape == (3, *initial.shape)
    assert jnp.allclose(sampled, full[observation_steps])
    assert jnp.allclose(sampled_blocked, full_blocked[observation_steps])
