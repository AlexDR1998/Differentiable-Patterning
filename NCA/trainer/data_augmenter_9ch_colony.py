"""Augmentation for the grouped 12-measurement/9-state micropattern task."""

import jax.numpy as jnp
import jax.tree_util as jtu

from Common.dataloader.micropattern_schemas import MICROPATTERN_GROUPED_12CH_SCHEMA
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch


class DataAugmenter(DataAugmenter4Ch):
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA

    def __init__(self, *args, **kwargs):
        model = kwargs.get("nca_model")
        self.channels = (
            model.N_CHANNELS
            if model is not None
            else self.schema.n_state_channels + kwargs.get("hidden_channels", 0)
        )
        if self.channels < self.schema.n_state_channels:
            raise ValueError("NCA has fewer channels than the grouped channel schema")
        super().__init__(*args, **kwargs)
        self.OBS_CHANNELS = self.schema.n_state_channels

    def _to_state(self, data):
        state = data[:, self.schema.primary_measurements]
        return jnp.pad(
            state,
            ((0, 0), (0, self.channels - state.shape[1]), (0, 0), (0, 0)),
        )

    def split_x_y(self, N_steps=1):
        x = jtu.tree_map(
            lambda data: self.real_to_latent(self._to_state(data[:-N_steps])),
            self.data_saved,
        )
        y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        return x, y


__all__ = ["DataAugmenter"]
