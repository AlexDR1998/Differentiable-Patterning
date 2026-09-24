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

from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from NCA.trainer.data_augmenter.transforms import add_noise, bernoulli_reinject_observations


@eqx.filter_jit
def snowmelt_advance(x, x_true, boundary, observable_channels, key, probability, noise_strength):
    """Propagate the pool, reinject observations, add noise, restore boundary channels."""
    reinject_key, noise_key = jax.random.split(key)
    x = bernoulli_reinject_observations(x, x_true, observable_channels, reinject_key, probability)
    if noise_strength > 0:
        x = add_noise(x, noise_strength, noise_key, mode="observable", observable_channels=observable_channels)
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


def build_snowmelt_augmenter(boundary_mask, reinjection_probability=0.5, noise_strength=0.005):
    """Return an augmenter class bound to one ``[B, m, H, W]`` boundary array."""
    boundary = [jnp.asarray(mask, dtype=jnp.float32) for mask in boundary_mask]

    class SnowmeltDataAugmenter(DataAugmenterAbstract):
        def data_init(self, SHARDING=None):
            if SHARDING not in (None, 1):
                raise ValueError("The snowmelt augmenter does not support sharding")
            data = apply_boundary_channels(self.return_saved_data(), boundary, self.OBS_CHANNELS)
            self.save_data(data)
            return None

        def advance_pool(self, x, y, i, key):
            x_true, _ = self.split_x_y(1)
            x = snowmelt_advance(
                x, x_true, boundary, self.OBS_CHANNELS, key,
                reinjection_probability, noise_strength,
            )
            self.PREVIOUS_KEY = key
            return x, y

    return SnowmeltDataAugmenter


__all__ = ["apply_boundary_channels", "build_snowmelt_augmenter", "snowmelt_advance"]
