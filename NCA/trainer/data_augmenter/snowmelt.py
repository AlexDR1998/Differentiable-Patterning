"""Pool augmentation for snowmelt sequences with fixed boundary channels.

The final ``m`` state channels hold the catchment mask and static terrain
covariates (the same array the soft ``model_boundary`` writes after every NCA
step). This augmenter writes them into the pool too, so the very first update
of every slot already perceives the terrain, and keeps observable channels
zero outside the catchment.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.data_augmenter.base import PoolAugmenter
from NCA.trainer.data_augmenter.transforms import add_noise, bernoulli_reinject_observations


@eqx.filter_jit
def snowmelt_advance(x, x_true, boundary, observable_channels, key, probability, noise_strength):
    """Propagate the pool, reinject observations, add noise, restore boundary channels."""
    reinject_key, noise_key = jax.random.split(key)
    x = bernoulli_reinject_observations(x, x_true, observable_channels, reinject_key, probability)
    if noise_strength > 0:
        x = add_noise(x, noise_strength, noise_key, mode="observable", observable_channels=observable_channels)
    return apply_boundary_channels(x, boundary, observable_channels)


@eqx.filter_jit
def snowmelt_initial_pool(x, boundary, observable_channels, key, noise_strength):
    """Noise the observed start states in place; no propagation or reinjection."""
    if noise_strength > 0:
        x = add_noise(x, noise_strength, key, mode="observable", observable_channels=observable_channels)
    return apply_boundary_channels(x, boundary, observable_channels)


def apply_boundary_channels(x, boundary, observable_channels):
    """Zero observables outside the catchment and write the fixed final channels.

    ``boundary`` is a list of ``[m, H, W]`` arrays, one per batch; channel 0 is
    the catchment mask.
    """

    def _apply(state, mask):
        m_channels = mask.shape[0]
        catchment = mask[0]
        state = state.at[:, :observable_channels].multiply(catchment)
        return state.at[:, -m_channels:].set(jnp.broadcast_to(mask, state[:, -m_channels:].shape))

    return jtu.tree_map(_apply, x, list(boundary))


class SnowmeltAugmenter(PoolAugmenter):
    """Pool for snowmelt training, bound to one ``[B, m, H, W]`` boundary array."""

    def __init__(self, data, hidden_channels, boundary, reinjection_probability=0.5, noise_strength=0.005):
        super().__init__(data, hidden_channels)
        self.boundary = [jnp.asarray(mask, dtype=jnp.float32) for mask in boundary]
        self.reinjection_probability = reinjection_probability
        self.noise_strength = noise_strength
        self.data_saved = apply_boundary_channels(self.data_saved, self.boundary, self.OBS_CHANNELS)

    def initialize_pool(self, key):
        # Slot k must start from image k. The base implementation calls
        # advance_pool, which shifts the pool by one slot first.
        x, y = self.split_x_y(1)
        x = snowmelt_initial_pool(x, self.boundary, self.OBS_CHANNELS, key, self.noise_strength)
        return x, y

    def advance_pool(self, x, y, i, key):
        x_true, _ = self.split_x_y(1)
        x = snowmelt_advance(
            x, x_true, self.boundary, self.OBS_CHANNELS, key,
            self.reinjection_probability, self.noise_strength,
        )
        return x, y


__all__ = [
    "SnowmeltAugmenter",
    "apply_boundary_channels",
    "snowmelt_advance",
    "snowmelt_initial_pool",
]
