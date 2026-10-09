"""Pool augmentation for simulated PDE trajectories.

Pool slot k holds the transition from frame k to frame k+1. After each step the
pool moves one slot forward, so predictions become the next start states, and
each slot is reset to the true frame with probability ``reinjection_probability``.
This is the snowmelt scheme without its fixed boundary channels.
"""

import equinox as eqx
import jax

from NCA.trainer.data_augmenter.base import PoolAugmenter
from NCA.trainer.data_augmenter.transforms import add_noise, bernoulli_reinject_observations


@eqx.filter_jit
def pde_advance(x, x_true, observable_channels, key, probability, noise_strength):
    """Propagate the pool, reinject observations and add noise to observable channels."""
    reinject_key, noise_key = jax.random.split(key)
    x = bernoulli_reinject_observations(x, x_true, observable_channels, reinject_key, probability)
    if noise_strength > 0:
        x = add_noise(x, noise_strength, noise_key, mode="observable", observable_channels=observable_channels)
    return x


class PdeAugmenter(PoolAugmenter):
    """Pool for PDE trajectories of shape [B, T, C, H, W]."""

    def __init__(self, data, hidden_channels, reinjection_probability=0.5, noise_strength=0.001):
        super().__init__(data, hidden_channels)
        self.reinjection_probability = reinjection_probability
        self.noise_strength = noise_strength

    def initialize_pool(self, key):
        # Slot k starts from frame k; the base implementation would shift the pool first.
        x, y = self.split_x_y(1)
        if self.noise_strength > 0:
            x = add_noise(x, self.noise_strength, key, mode="observable", observable_channels=self.OBS_CHANNELS)
        return x, y

    def advance_pool(self, x, y, i, key):
        x_true, _ = self.split_x_y(1)
        x = pde_advance(x, x_true, self.OBS_CHANNELS, key, self.reinjection_probability, self.noise_strength)
        return x, y


__all__ = ["PdeAugmenter", "pde_advance"]
