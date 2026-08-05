import jax
import jax.numpy as jnp

from NCA.trainer.data_augmenter import (
    add_noise,
    bernoulli_reinject_observations,
    propagate_pool,
    reinject_observations,
    scheduled_probability,
    split_trajectory,
    terminal_carry,
)


def _data():
    return [jnp.arange(4 * 2 * 2 * 2, dtype=jnp.float32).reshape(4, 2, 2, 2)]


def test_split_trajectory_returns_input_states_and_observable_targets():
    x, y = split_trajectory(_data())

    assert x[0].shape == (3, 2, 2, 2)
    assert y[0].shape == (3, 2, 2, 2)
    assert jnp.array_equal(x[0], _data()[0][:-1])
    assert jnp.array_equal(y[0], _data()[0][1:])


def test_propagate_and_reinject_preserve_shape_and_reset_initial_state():
    data = [jnp.ones((3, 2, 2, 2))]
    truth = [jnp.full((3, 2, 2, 2), 7.0)]

    propagated = propagate_pool(data)
    result = reinject_observations(propagated, truth, 1, jax.random.PRNGKey(0), fraction=1.0)

    assert result[0].shape == data[0].shape
    assert jnp.array_equal(result[0][0], truth[0][0])
    assert jnp.array_equal(result[0][1:, 0], truth[0][1:, 0])
    assert jnp.array_equal(result[0][1:, 1], propagated[0][1:, 1])


def test_stacked_pool_propagates_time_axis_and_reinjects_exact_global_fraction():
    batch, time = 2, 4
    x = jnp.arange(batch * time, dtype=jnp.float32).reshape(batch, time, 1, 1, 1)
    truth = jnp.full_like(x, 99.0)

    result = reinject_observations(
        x, truth, 1, jax.random.PRNGKey(8), fraction=0.5
    )

    assert jnp.all(result[:, 0] == 99.0)
    # floor(2 batches * 3 eligible times * 0.5) = exactly three resets.
    assert jnp.sum(result[:, 1:, 0, 0, 0] == 99.0) == 3
    propagated = jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)
    retained = result[:, 1:, 0, 0, 0] != 99.0
    assert jnp.array_equal(
        result[:, 1:, 0, 0, 0][retained],
        propagated[:, 1:, 0, 0, 0][retained],
    )


def test_bernoulli_reinjection_constant_schedule_supports_zero_and_one():
    x = [jnp.zeros((4, 2, 1, 1)) for _ in range(2)]
    truth = [jnp.full_like(value, 7.0) for value in x]

    none = bernoulli_reinject_observations(
        x, truth, 2, jax.random.PRNGKey(1), 0.0
    )
    all_slots = bernoulli_reinject_observations(
        x, truth, 2, jax.random.PRNGKey(1), 1.0
    )

    assert jnp.all(jnp.stack(none)[:, 0] == 7.0)
    assert jnp.all(jnp.stack(none)[:, 1:] == 0.0)
    assert jnp.all(jnp.stack(all_slots) == 7.0)


def test_add_noise_is_reproducible_for_a_fixed_key():
    data = [jnp.zeros((2, 2, 2, 2))]
    first = add_noise(data, 1.0, jax.random.PRNGKey(4))
    second = add_noise(data, 1.0, jax.random.PRNGKey(4))

    assert jnp.array_equal(first[0], second[0])
    assert not jnp.array_equal(first[0], data[0])


def test_terminal_carry_uses_previous_terminal_values():
    current = [jnp.zeros((2, 1, 1, 1))]
    previous = [jnp.ones((1, 1, 1))]

    result = terminal_carry(current, previous, 1.0, jax.random.PRNGKey(0))

    assert result[0][-1, 0, 0, 0] == 1.0


def test_scheduled_probability_is_zero_before_start_and_reaches_final_value():
    assert scheduled_probability(2, 5, 10, 0.0, 1.0) == 0.0
    assert scheduled_probability(15, 5, 10, 0.0, 1.0) == 1.0
