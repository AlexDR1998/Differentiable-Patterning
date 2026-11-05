"""Cell density of the initial texture: how much of the colony cells cover.

A texture ``T`` ``[C, H, W]`` holds what cells look like, with the gaps
between them filled in (``Experiments/micropatterns/reference.py:
fill_reference_gaps``). Where the cells are is a cut of a random field ``F``
with the cell/gap spectrum of the training colonies:

    O = sigmoid((F - ndtri(1 - density)) / edge_width)
    x = O * T + (1 - O) * gap_values

``F`` has zero mean and unit variance and is close to Gaussian, so
``mean(O)`` is close to ``density``. Changing the density changes how much
empty space there is between cells, not how bright the cells are.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jaxtyping import Array


def coverage_field(amplitude: Array, shape: tuple[int, int], key: Array) -> Array:
    """Random field ``[H, W]`` with amplitude spectrum ``amplitude`` ``[H, W // 2 + 1]``.

    White noise filtered by the spectrum, then set to zero mean and unit
    variance.
    """

    noise = jax.random.normal(key, shape, dtype=amplitude.dtype)
    field = jnp.fft.irfft2(jnp.fft.rfft2(noise) * amplitude, s=shape)
    field = field - jnp.mean(field)
    return field / jnp.maximum(jnp.std(field), 1.0e-6)


def cell_coverage(field: Array, density: Array, edge_width: float) -> Array:
    """Share of each pixel covered by cells, with ``mean`` about ``density``."""

    level = ndtri(1.0 - density)
    return jax.nn.sigmoid((field - level) / edge_width)


def apply_density(texture: Array, gap_values: Array, coverage: Array) -> Array:
    """Cells (``texture`` ``[C, H, W]``) where covered, ``gap_values`` ``[C]`` in the gaps."""

    gaps = jnp.asarray(gap_values, dtype=texture.dtype)[:, None, None]
    return coverage[None] * texture + (1.0 - coverage[None]) * gaps


class CellDensity(eqx.Module):
    """The covered fraction of the colony, ``sigmoid(raw)``, so always in (0, 1)."""

    raw: Array

    def __init__(self, density: float):
        if not 0.0 < density < 1.0:
            raise ValueError("density must be between 0 and 1")
        self.raw = jnp.log(jnp.asarray(density) / (1.0 - jnp.asarray(density)))

    @property
    def value(self) -> Array:
        return jax.nn.sigmoid(self.raw)


class CellGaps(eqx.Module):
    """What the gaps look like: their value in each channel and their spectrum.

    ``edge_width`` (in standard deviations of the field) sets how sharp the
    edge between cells and gaps is. It is part of the texture, so it is the
    same during optimisation and evaluation.
    """

    gap_values: Array
    amplitude: Array
    edge_width: float = eqx.field(static=True, default=0.1)

    def apply(self, texture: Array, density: CellDensity, key: Array) -> Array:
        """``texture`` ``[C, H, W]`` with gaps for ``density``, placed by ``key``."""

        field = coverage_field(self.amplitude, texture.shape[-2:], key)
        coverage = cell_coverage(field, density.value, self.edge_width)
        return apply_density(texture, self.gap_values, coverage)
