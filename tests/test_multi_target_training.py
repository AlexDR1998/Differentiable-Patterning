import jax
import jax.numpy as jnp

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from Common.trainer.loss_multi_target import multi_target_loss
from NCA.trainer.data_augmenter_260726 import DataAugmenter

COMPONENT_NAMES = ("texture", "channel_mean", "radial", "correlation")


def test_multi_target_loss_is_invariant_to_groupwise_batch_order():
    schema = MICROPATTERN_260726_SCHEMA
    prediction = jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 10, 8, 8))
    target = jnp.take(prediction, jnp.asarray(schema.target_to_state), axis=2)
    for time in range(2):
        for group, channels in enumerate(schema.group_measurement_indices):
            order = jnp.roll(jnp.arange(3), time + group)
            target = target.at[:, time, channels].set(target[order, time][:, channels])

    loss, components = multi_target_loss(
        prediction,
        target,
        jnp.ones((8, 8), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(1),
        {"multi_target_weights": {"texture": 0.0}},
    )

    assert jnp.allclose(loss, 0.0, atol=1e-5)
    component_total = sum(components[name] for name in COMPONENT_NAMES)
    assert jnp.allclose(loss, component_total + components["assignment_regularisation"])


def test_soft_assignment_components_reconstruct_loss():
    schema = MICROPATTERN_260726_SCHEMA
    key = jax.random.PRNGKey(4)
    prediction = jax.random.uniform(key, (3, 2, 10, 8, 8))
    target = jax.random.uniform(jax.random.fold_in(key, 1), (3, 2, 14, 8, 8))

    loss, components = multi_target_loss(
        prediction,
        target,
        jnp.ones((8, 8), dtype=bool),
        schema,
        None,
        key,
        {
            "assignment": "softmin",
            "assignment_tau": 0.05,
            "multi_target_weights": {"texture": 0.0},
        },
    )

    component_total = sum(components[name] for name in COMPONENT_NAMES)
    assert jnp.allclose(loss, component_total + components["assignment_regularisation"])
    assert jnp.all(components["assignment_entropy"] >= 0)


def test_snapshot_augmenter_outputs_unique_state_and_measurement_targets():
    class DownsamplingModel:
        N_CHANNELS = 16

        @staticmethod
        def real_to_latent(x):
            return jax.image.resize(x, (*x.shape[:-2], 4, 4), "linear")

    data = jax.random.uniform(jax.random.PRNGKey(2), (3, 5, 14, 8, 8))
    augmenter = DataAugmenter(data, hidden_channels=2, nca_model=DownsamplingModel())
    augmenter.noise_strength = 0.0

    x, y = augmenter.data_load(jax.random.PRNGKey(3))

    assert len(x) == len(y) == 3
    assert x[0].shape == (4, 16, 4, 4)
    assert y[0].shape == (4, 16, 8, 8)
    assert augmenter.OBS_CHANNELS == 10
    assert DataAugmenter.schema is MICROPATTERN_260726_SCHEMA
