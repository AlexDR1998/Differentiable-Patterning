from types import SimpleNamespace

import jax
import jax.numpy as jnp

from NCA.trainer.data_augmenter.emoji import EmojiAugmenter, schedule_probability
from NCA.trainer.data_augmenter.transforms import scheduled_probability


def _schedule(enabled, probability=1.0):
    return SimpleNamespace(
        enabled=enabled,
        start_iteration=0,
        schedule_iterations=0,
        initial_probability=probability,
        final_probability=probability,
    )


def _advance(terminal_carry):
    # One trajectory with 3 time slots; reinjection is the only other change
    data = jnp.zeros((1, 3, 1, 2, 2))
    augmenter = EmojiAugmenter(data, hidden_channels=0, terminal_carry=terminal_carry)
    x = [jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 9.0)])]
    x, _ = augmenter.advance_pool(x, [jnp.zeros_like(x[0])], 0, jax.random.PRNGKey(0))
    return x


def test_terminal_probability_starts_at_zero_then_follows_linear_schedule():
    probability = scheduled_probability

    assert probability(99, 100, 100, 0.5, 0.9) == 0.0
    assert probability(100, 100, 100, 0.5, 0.9) == 0.5
    assert jnp.allclose(probability(150, 100, 100, 0.5, 0.9), 0.7)
    assert probability(200, 100, 100, 0.5, 0.9) == 0.9
    assert schedule_probability(_schedule(False), 150) == 0.0


def test_terminal_carry_preserves_the_previous_terminal_prediction():
    assert jnp.allclose(_advance(_schedule(True))[0][-1], 9.0)


def test_disabled_terminal_carry_retains_basic_pool_propagation():
    assert jnp.allclose(_advance(_schedule(False))[0][-1], 1.0)
