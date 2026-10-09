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


def test_pattern_count_rollout_counts_marker_patterns_per_group():
    from NCA.trainer.intervention import rollout_model_with_blocked_channel_pattern_counts

    model = AdditiveModel()
    initial = jax.random.uniform(jax.random.PRNGKey(0), (3, 4, 4))
    key = jax.random.PRNGKey(5)
    boundary_mask = jnp.ones((1, 4, 4))
    groups = jnp.asarray([[0, 0, 1, 1]] * 3 + [[-1, -1, -1, -1]])
    thresholds = jnp.asarray([1.0, 1.5])
    fate_channels = jnp.asarray([2, 0])

    counts = rollout_model_with_blocked_channel_pattern_counts(
        model, initial, boundary_mask, "soft", key, 3,
        channel=1,
        knockout_step=jnp.asarray(1, dtype=jnp.int32),
        fate_channels=fate_channels,
        fate_thresholds=thresholds,
        pixel_groups=groups,
        n_groups=2,
    )
    states = rollout_model_with_blocked_channel(
        model, initial, model_boundary(boundary_mask), key, 3,
        channel=1, knockout_step=jnp.asarray(1, dtype=jnp.int32),
    )

    assert counts.shape == (4, 2, 4)
    for state, step_counts in zip(states, counts):
        high = state[fate_channels] > thresholds[:, None, None]
        codes = high[0] * 1 + high[1] * 2
        for group in range(2):
            expected = jnp.bincount(codes[groups == group], length=4)
            assert jnp.array_equal(step_counts[group], expected)
