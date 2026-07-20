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
        x = jtu.tree_map(lambda data: self.real_to_latent(self._to_state(data[:-N_steps])), self.data_saved)
        y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        return x, y

    def data_callback(self, x, y, i, key):
        """Propagate the pool and reinject randomly permuted experiment groups."""
        x = jnp.stack(x)
        truth = jnp.stack([
            self.real_to_latent(self._to_state(data)) for data in self.data_saved
        ])
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
                for g, states in enumerate(self.state_groups):
                    key, donor_key = jax.random.split(key)
                    donors = jax.random.permutation(donor_key, batch_count)
                    values = truth[donors, t][:, states]
                    keep = (inject[:, t - 1] & (choices[:, t - 1] == g))[:, None, None, None]
                    x = x.at[:, t, states].set(jnp.where(keep, values, x[:, t, states]))

        x = self.noise(list(x), self.noise_strength, key=key)
        self.PREVIOUS_KEY = key
        return x, y
