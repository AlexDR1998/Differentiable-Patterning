"""The data and training pool that every NCA augmenter holds."""

import jax.numpy as jnp

from NCA.trainer.data_augmenter.trajectory import split_trajectory


def as_trajectory_list(data):
    """Return ``data`` as a list of ``[time, channels, H, W]`` trajectories.

    Accepts a stacked ``[batch, time, channels, H, W]`` array or a list, so
    batches may have different spatial sizes.
    """
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data[index] for index in range(data.shape[0])]


class PoolAugmenter:
    """Hold the training data and move the pool of states on between steps.

    ``data`` is padded with ``hidden_channels`` zero channels. The trainer
    calls ``initialize_pool`` once, then ``advance_pool`` after every training
    step with the states the NCA reached. Subclasses decide how the pool is
    refreshed (reinjecting true data, noise, damage, ...); all randomness
    comes from the key passed in.
    """

    def __init__(self, data, hidden_channels=0):
        data = as_trajectory_list(data)
        self.OBS_CHANNELS = data[0].shape[1]
        self.hidden_channels = hidden_channels
        # data_true is never changed; data_saved may be padded or duplicated
        # by a subclass before training starts.
        self.data_true = [
            jnp.pad(trajectory, ((0, 0), (0, hidden_channels), (0, 0), (0, 0)))
            for trajectory in data
        ]
        self.data_saved = self.data_true

    def split_x_y(self, n_steps=1):
        """Initial states ``x[t]`` and the targets ``y[t] = data[t + n_steps]``."""
        return split_trajectory(self.data_saved, n_steps)

    def initialize_pool(self, key):
        x, y = self.split_x_y(1)
        return self.advance_pool(x, y, 0, key)

    def advance_pool(self, x, y, i, key):
        """Return the next pool from the states ``x`` reached at iteration ``i``."""
        return x, y

    def return_saved_data(self):
        return self.data_saved

    def return_observed_data(self):
        """Current targets without the hidden (latent-only) channels."""
        schema = getattr(self, "schema", None)
        channel_count = schema.n_measurement_channels if schema is not None else self.OBS_CHANNELS
        return [trajectory[:, :channel_count] for trajectory in self.data_saved]
