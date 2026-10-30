from types import SimpleNamespace

import jax
import jax.numpy as jnp

from NCA.trainer.data_augmenter.emoji import EmojiAugmenter, schedule_probability
from NCA.trainer.data_augmenter.transforms import scheduled_probability


def _schedule(enabled, probability=1.0):
    return SimpleNamespace(
        enabled=enabled,
        start_fraction=0.0,
        end_fraction=0.0,
        initial_probability=probability,
        final_probability=probability,
    )


def _advance(terminal_carry):
    # One trajectory with 3 time slots; reinjection is the only other change
    data = jnp.zeros((1, 3, 1, 2, 2))
    augmenter = EmojiAugmenter(
        data, hidden_channels=0, terminal_carry=terminal_carry, total_iterations=10
    )
    x = [jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 9.0)])]
    x, _ = augmenter.advance_pool(x, [jnp.zeros_like(x[0])], 0, jax.random.PRNGKey(0))
    return x


def test_terminal_probability_starts_at_zero_then_follows_linear_schedule():
    probability = scheduled_probability

    assert probability(99, 100, 100, 0.5, 0.9) == 0.0
    assert probability(100, 100, 100, 0.5, 0.9) == 0.5
    assert jnp.allclose(probability(150, 100, 100, 0.5, 0.9), 0.7)
    assert probability(200, 100, 100, 0.5, 0.9) == 0.9
    assert schedule_probability(_schedule(False), 150, 1000) == 0.0


def test_schedule_fractions_scale_with_the_run_length():
    schedule = SimpleNamespace(
        enabled=True,
        start_fraction=0.1,
        end_fraction=0.3,
        initial_probability=0.5,
        final_probability=0.9,
    )

    # 0.1 of 10,000 iterations is the first 1,000; the ramp then runs to 3,000.
    assert schedule_probability(schedule, 999, 10_000) == 0.0
    assert schedule_probability(schedule, 1000, 10_000) == 0.5
    assert jnp.allclose(schedule_probability(schedule, 2000, 10_000), 0.7)
    assert schedule_probability(schedule, 3000, 10_000) == 0.9
    # Half the run length halves the iterations.
    assert schedule_probability(schedule, 499, 5000) == 0.0
    assert schedule_probability(schedule, 1500, 5000) == 0.9


def test_terminal_carry_preserves_the_previous_terminal_prediction():
    assert jnp.allclose(_advance(_schedule(True))[0][-1], 9.0)


def test_disabled_terminal_carry_retains_basic_pool_propagation():
    assert jnp.allclose(_advance(_schedule(False))[0][-1], 1.0)
