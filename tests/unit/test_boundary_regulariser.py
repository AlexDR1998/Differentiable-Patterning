import jax
import jax.numpy as jnp

from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from NCA.trainer.NCA_regulariser import boundary_regulariser


def _evaluate(states, callbacks):
    """Evaluate the shared regulariser with unused trainer arguments omitted."""
    return boundary_regulariser(
        None,
        states,
        None,
        None,
        None,
        {"BOUNDARY_CALLBACK": callbacks},
        None,
    )


def test_model_boundary_penalises_other_channels_only_outside_mask():
    mask = jnp.asarray([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    state = jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 2, 3) / 10.0
    state = state.at[:, -1].set(mask[0])

    actual = _evaluate([state], [model_boundary(mask)])
    expected = jnp.mean(jnp.abs(state[:, :-1]) * (1.0 - mask[0]))

    assert actual.shape == (1,)
    assert jnp.allclose(actual[0], expected)


def test_model_boundary_gradient_ignores_inside_and_mask_channel():
    mask = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    state = jnp.ones((1, 3, 2, 2), dtype=jnp.float32)

    gradient = jax.grad(
        lambda value: jnp.sum(_evaluate([value], [model_boundary(mask)]))
    )(state)

    assert jnp.all(gradient[:, :-1, mask[0].astype(bool)] == 0.0)
    assert jnp.all(gradient[:, :-1, ~mask[0].astype(bool)] > 0.0)
    assert jnp.all(gradient[:, -1] == 0.0)


def test_hard_boundary_penalises_all_channels_outside_mask():
    mask = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    state = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 2, 2)

    actual = _evaluate([state], [hard_boundary(mask)])
    expected = jnp.mean(jnp.abs(state) * (1.0 - mask[0]))

    assert jnp.allclose(actual[0], expected)


def test_no_boundary_has_zero_penalty():
    state = jnp.ones((2, 3, 4, 5), dtype=jnp.float32)
    actual = _evaluate([state], [no_boundary()])
    assert jnp.array_equal(actual, jnp.zeros((1,), dtype=state.dtype))
