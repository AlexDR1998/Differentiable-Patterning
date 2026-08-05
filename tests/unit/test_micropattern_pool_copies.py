from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from NCA.trainer.data_augmenter.micropattern import DataAugmenter


def test_260726_augmenter_preserves_loader_batch_count():
    # These four batches represent the output of the loader, which applies the
    # configured pool copies together with its masks and metadata.
    data = jnp.zeros(
        (4, 2, MICROPATTERN_260726_SCHEMA.n_measurement_channels, 2, 2)
    )

    augmenter = DataAugmenter(data_true=data)
    augmenter.data_init()

    assert len(augmenter.return_saved_data()) == 4


def _time_coded_data(batch_count=3, time_count=5, size=2):
    channels = MICROPATTERN_260726_SCHEMA.n_measurement_channels
    values = jnp.arange(time_count, dtype=jnp.float32)[None, :, None, None, None]
    return jnp.broadcast_to(
        values, (batch_count, time_count, channels, size, size)
    )


def test_initialize_pool_preserves_raw_timestep_alignment():
    data = _time_coded_data()
    augmenter = DataAugmenter(data_true=data)
    augmenter.noise_strength = 0.0

    initialized_x, initialized_y = augmenter.initialize_pool(jr.PRNGKey(0))
    split_x, split_y = augmenter.split_x_y(1)

    assert jnp.array_equal(jnp.stack(initialized_x), jnp.stack(split_x))
    assert jnp.array_equal(jnp.stack(initialized_y), jnp.stack(split_y))


def test_initialize_pool_pads_schema_state_to_model_channels():
    data = _time_coded_data(batch_count=1)
    model = SimpleNamespace(N_CHANNELS=48)
    augmenter = DataAugmenter(data_true=data, nca_model=model)
    augmenter.noise_strength = 0.0

    initialized_x, _ = augmenter.initialize_pool(jr.PRNGKey(0))

    assert initialized_x[0].shape[1] == model.N_CHANNELS


def test_advance_pool_shifts_perfect_rollout_into_next_transition_slots():
    data = _time_coded_data()
    augmenter = DataAugmenter(
        data_true=data,
        intermediate_reinjection_probability=0.0,
    )
    augmenter.noise_strength = 0.0
    _, targets = augmenter.split_x_y(1)
    perfect_rollout = [augmenter._to_state(trajectory)[1:] for trajectory in data]

    advanced_x, advanced_y = augmenter.advance_pool(
        perfect_rollout, targets, 0, jr.PRNGKey(1)
    )

    expected_times = jnp.arange(data.shape[1] - 1, dtype=jnp.float32)
    observed_times = jnp.stack(advanced_x)[:, :, 0, 0, 0]
    assert jnp.array_equal(
        observed_times,
        jnp.broadcast_to(expected_times, observed_times.shape),
    )
    assert jnp.array_equal(jnp.stack(advanced_y), jnp.stack(targets))
