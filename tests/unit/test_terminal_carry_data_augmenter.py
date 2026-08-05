import jax
import jax.numpy as jnp

from NCA.trainer.data_augmenter.nca_terminal import TerminalCarryDataAugmenter


class AlwaysCarryAugmenter(TerminalCarryDataAugmenter):
    TERMINAL_CARRY_ENABLED = True
    TERMINAL_CARRY_INITIAL = 1.0
    TERMINAL_CARRY_FINAL = 1.0


class NeverCarryAugmenter(TerminalCarryDataAugmenter):
    TERMINAL_CARRY_ENABLED = False


def _augmenter(augmenter_type):
    data = jnp.zeros((1, 3, 1, 2, 2))
    return augmenter_type(data, hidden_channels=0)


def test_terminal_probability_starts_at_zero_then_follows_linear_schedule():
    probability = TerminalCarryDataAugmenter.scheduled_probability

    assert probability(99, 100, 100, 0.5, 0.9) == 0.0
    assert probability(100, 100, 100, 0.5, 0.9) == 0.5
    assert jnp.allclose(probability(150, 100, 100, 0.5, 0.9), 0.7)
    assert probability(200, 100, 100, 0.5, 0.9) == 0.9


def test_terminal_carry_preserves_the_previous_terminal_prediction():
    augmenter = _augmenter(AlwaysCarryAugmenter)
    x = [jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 9.0)])]
    x_true = [jnp.zeros_like(x[0])]

    carried = augmenter.propagate_with_terminal_carry(
        x,
        x_true,
        i=0,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(carried[0][-1], 9.0)


def test_disabled_terminal_carry_retains_basic_pool_propagation():
    augmenter = _augmenter(NeverCarryAugmenter)
    x = [jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 9.0)])]
    x_true = [jnp.zeros_like(x[0])]

    propagated = augmenter.propagate_with_terminal_carry(
        x,
        x_true,
        i=0,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(propagated[0][-1], 1.0)

