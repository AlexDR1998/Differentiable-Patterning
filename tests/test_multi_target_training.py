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
    for group in schema.group_names:
        group_total = sum(components[f"group/{group}/{name}"] for name in COMPONENT_NAMES)
        assert jnp.allclose(components[f"group/{group}/total"], group_total)


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
    assert components["group/rna_expression/texture"].shape == loss.shape


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


def test_reinjection_preserves_group_specific_duplicate_measurements():
    schema = MICROPATTERN_260726_SCHEMA
    batch, time, channel = jnp.meshgrid(
        jnp.arange(3), jnp.arange(5), jnp.arange(14), indexing="ij"
    )
    data = (1000 * batch + 100 * time + channel)[..., None, None].astype(jnp.float32)
    data = jnp.broadcast_to(data, (3, 5, 14, 2, 2))
    augmenter = DataAugmenter(data)
    augmenter.noise_strength = 0.0
    zeros = [jnp.zeros((4, 10, 2, 2)) for _ in range(3)]
    targets = [value[1:] for value in data]
    observed = {"cell_fate_s2": False, "protein_response": False}

    for seed in range(20):
        result, _ = augmenter.data_callback(zeros, targets, 0, jax.random.PRNGKey(seed))
        result = jnp.stack(result)
        for batch_index in range(3):
            for time_index in range(1, 4):
                values = result[batch_index, time_index, :, 0, 0]
                if values[4] != 0:
                    donor = jnp.rint((values[4] - 100 * time_index - 5) / 1000).astype(int)
                    assert values[2] == 1000 * donor + 100 * time_index + 4
                    assert values[1] == 1000 * donor + 100 * time_index + 6
                    assert values[0] == 1000 * donor + 100 * time_index + 7
                    observed["cell_fate_s2"] = True
                if values[8] != 0:
                    donor = jnp.rint((values[8] - 100 * time_index - 11) / 1000).astype(int)
                    assert values[0] == 1000 * donor + 100 * time_index + 12
                    assert values[9] == 1000 * donor + 100 * time_index + 13
                    observed["protein_response"] = True

    assert all(observed.values())
