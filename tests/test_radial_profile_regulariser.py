import jax
import jax.numpy as jnp

from NCA.trainer.NCA_regulariser import (
    RADIAL_CHANNEL_WEIGHTS,
    radial_profile_regulariser,
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


def _central_prediction():
    prediction = jnp.zeros((1, 9, 5, 5), dtype=jnp.float32)
    return prediction.at[:, :, 2, 2].set(1.0)


def test_matching_radial_profiles_have_zero_regulariser():
    prediction = _central_prediction()
    target = _target_layout(prediction)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)

    loss = radial_profile_regulariser(
        [prediction], [target], [channel_mask], radial_bins=4
    )

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_sox2_radial_profile_mismatch_is_penalised():
    prediction = _central_prediction()
    target = _target_layout(prediction)
    target = target.at[:, 3].set(0.0)
    target = target.at[:, 3, 0, :].set(1.0)
    target = target.at[:, 3, 4, :].set(1.0)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)

    loss = radial_profile_regulariser(
        [prediction], [target], [channel_mask], radial_bins=4
    )

    assert loss[0] > 0.0


def test_unmeasured_channel_does_not_contribute():
    prediction = _central_prediction()
    target = _target_layout(prediction).at[:, 3].set(10.0)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)
    channel_mask = channel_mask.at[:, 3].set(False)

    loss = radial_profile_regulariser(
        [prediction], [target], [channel_mask], radial_bins=4
    )

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_pixels_outside_boundary_do_not_affect_radial_loss():
    prediction = _central_prediction()
    target = _target_layout(prediction)
    target = target.at[:, :, 0, :].set(100.0)
    target = target.at[:, :, 4, :].set(-100.0)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)
    boundary = jnp.zeros((1, 5, 5), dtype=jnp.float32)
    boundary = boundary.at[:, 1:4, 1:4].set(1.0)

    loss = radial_profile_regulariser(
        [prediction], [target], [channel_mask], [boundary], radial_bins=4
    )

    assert jnp.allclose(loss, 0.0, atol=1e-6)


def test_radial_regulariser_has_finite_nonzero_gradients():
    prediction = _central_prediction()
    target = _target_layout(prediction).at[:, 3].set(0.5)
    channel_mask = jnp.ones((1, 12, 1, 1), dtype=bool)

    def loss_fn(values):
        return radial_profile_regulariser(
            [values], [target], [channel_mask], radial_bins=4
        ).sum()

    gradient = jax.grad(loss_fn)(prediction)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0)


def test_duplicate_measurements_share_one_channel_weight():
    assert jnp.allclose(RADIAL_CHANNEL_WEIGHTS[:3], 0.5)
    assert jnp.allclose(RADIAL_CHANNEL_WEIGHTS[4:7], 0.5)
    unique_indices = jnp.array([3, 7, 8, 9, 10, 11])
    assert jnp.allclose(RADIAL_CHANNEL_WEIGHTS[unique_indices], 1.0)
    assert jnp.isclose(jnp.sum(RADIAL_CHANNEL_WEIGHTS), 9.0)
