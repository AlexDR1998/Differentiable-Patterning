"""Pool augmentation for unordered 260726 micropattern snapshots."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as BasicAugmenter
from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA


class DataAugmenter(BasicAugmenter):
    noise_strength = 0.005
    schema = MICROPATTERN_260726_SCHEMA

    def __init__(self, *args, **kwargs):
        model = kwargs.get("nca_model")
        self.channels = (
            model.N_CHANNELS if model is not None
            else self.schema.n_state_channels + kwargs.get("hidden_channels", 0)
        )
        if self.channels < self.schema.n_state_channels:
            raise ValueError("NCA has fewer channels than the measurement schema")
        super().__init__(*args, **kwargs)
        self.OBS_CHANNELS = self.schema.n_state_channels
        self.state_groups = tuple(
            tuple(self.schema.target_to_state[channel] for channel in group)
            for group in self.schema.group_measurement_indices
        )

    def _to_state(self, data):
        state = data[:, self.schema.primary_measurements]
        return jnp.pad(state, ((0, 0), (0, self.channels - state.shape[1]), (0, 0), (0, 0)))

    def split_x_y(self, N_steps=1):
        if self.batch_mode == "array":
            x = self.map_batches(
                lambda data: self.real_to_latent(self._to_state(data[:-N_steps])),
                self.data_saved,
            )
            y = self.data_saved[:, N_steps:]
        else:
            x = jtu.tree_map(lambda data: self.real_to_latent(self._to_state(data[:-N_steps])), self.data_saved)
            y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        return x, y

    def data_callback(self, x, y, i, key):
        """Propagate the pool and reinject randomly permuted experiment groups."""
        x = self.as_array(x)
        saved = self.as_array(self.data_saved)
        measurements = saved[:, :, :self.schema.n_measurement_channels]
        truth = jax.vmap(lambda data: self.real_to_latent(self._to_state(data)))(saved)
        batch_count, time_count = x.shape[:2]
        x = x.at[:, 1:].set(x[:, :-1])

        key, reset_key = jax.random.split(key)
        reset = truth[jax.random.permutation(reset_key, batch_count), 0]
        x = x.at[:, 0].set(reset)

        if time_count > 1:
            key, mask_key, group_key = jax.random.split(key, 3)
            inject = jax.random.bernoulli(mask_key, 0.5, (batch_count, time_count - 1))
            choices = jax.random.randint(group_key, inject.shape, 0, len(self.schema.experiment_groups))
            for t in range(1, time_count):
                for g, (targets, states) in enumerate(
                    zip(self.schema.group_measurement_indices, self.state_groups)
                ):
                    key, donor_key = jax.random.split(key)
                    donors = jax.random.permutation(donor_key, batch_count)
                    values = self.real_to_latent(measurements[donors, t][:, targets])
                    keep = (inject[:, t - 1] & (choices[:, t - 1] == g))[:, None, None, None]
                    x = x.at[:, t, states].set(jnp.where(keep, values, x[:, t, states]))

        x = self.restore_batch_mode(x)
        x = self.noise(x, self.noise_strength, key=key)
        self.PREVIOUS_KEY = key
        return x, y
