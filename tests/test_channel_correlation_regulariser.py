import jax
import jax.numpy as jnp

from NCA.trainer.NCA_regulariser import (
    CORRELATION_PAIR_I,
    CORRELATION_PAIR_J,
    CORRELATION_PAIR_WEIGHTS,
    channel_correlation_regulariser,
)


def _target_layout(unique_channels):
    return jnp.concatenate(
        [
            unique_channels[:, 0:4],
            unique_channels[:, 0:3],
            unique_channels[:, 4:8],
            unique_channels[:, 8:9],
        ],
        axis=1,
    )


def _example_prediction():
    patterns = jnp.array(
        [
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-2.0, 0.0, 0.0, 2.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [-1.0, -0.5, 0.5, 1.0],
            [1.0, 0.5, -0.5, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    return patterns.reshape(1, 9, 2, 2)


def test_matching_grouped_correlations_have_zero_regulariser():
    prediction = _example_prediction()
    target = _target_layout(prediction)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)
    boundary = jnp.ones((1, 2, 2), dtype=jnp.float32)

    loss = channel_correlation_regulariser(
        [prediction], [target], [channel_mask], [boundary]
    )

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_sox2_correlation_mismatch_is_penalised():
    prediction = _example_prediction()
    target = _target_layout(prediction)
    target = target.at[:, 3].set(target[:, 0])
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)

    loss = channel_correlation_regulariser(
        [prediction], [target], [channel_mask]
    )

    assert loss[0] > 0.1


def test_correlation_regulariser_has_finite_nonzero_gradients():
    prediction = _example_prediction()
    target = _target_layout(prediction).at[:, 3].set(prediction[:, 0])
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)

    def loss_fn(values):
        return channel_correlation_regulariser(
            [values], [target], [channel_mask]
        ).sum()

    gradient = jax.grad(loss_fn)(prediction)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0)


def test_pixels_outside_boundary_do_not_affect_correlation_loss():
    prediction = jnp.pad(_example_prediction(), ((0, 0), (0, 0), (0, 1), (0, 1)))
    target = _target_layout(prediction)
    target = target.at[:, :, 2, :].set(100.0)
    target = target.at[:, :, :, 2].set(-100.0)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)
    boundary = jnp.array(
        [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
    )

    loss = channel_correlation_regulariser(
        [prediction], [target], [channel_mask], [boundary]
    )

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_duplicate_marker_pairs_receive_half_weight_per_experiment():
    pair_weights = {
        (int(i), int(j)): float(weight)
        for i, j, weight in zip(
            CORRELATION_PAIR_I,
            CORRELATION_PAIR_J,
            CORRELATION_PAIR_WEIGHTS,
        )
    }

    assert pair_weights[(0, 1)] == 0.5
    assert pair_weights[(4, 5)] == 0.5
    assert pair_weights[(0, 3)] == 1.0
    assert pair_weights[(4, 7)] == 1.0
