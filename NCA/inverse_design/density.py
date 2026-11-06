"""Cell density of the initial texture: how much of the colony cells cover.

A density texture is made of whole measured cells, placed at random over
the grid without overlapping (``Experiments/micropatterns/texture_synthesis.py:
place_cells``). It has a few layers, each filled with cells, and every cell
has a random rank ``u`` in [0, 1). For a density, cells are kept layer by
layer in rank order until that share of the grid is covered; the rest of
the grid gets the gap values. So the density sets how many cells there are
and how far apart they lie, while each kept cell stays whole and keeps its
measured values. Layer 0 is drawn on top, so cells of later layers fill the
space between its cells and can lie partly under them.

Each rollout shifts every rank by the same random amount (modulo 1), so it
keeps a different random set of cells. To give the density a gradient, the
cut is a sigmoid of width ``softness`` in rank: the few cells at the cut are
faded rather than kept or dropped.

The texture of one colony is stored as one array ``[L * (C + P), H, W]``
(``pack_cells``): for each of the ``L`` layers, the ``C`` state channels and
then the rank maps of the ``P`` staining panels (cells of different panels
come from different colonies, so their shapes differ; -1 where no cell is).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def pack_cells(textures: Array, ranks: Array) -> Array:
    """One array ``[L * (C + P), H, W]`` from ``textures`` ``[L, C, H, W]`` and ``ranks`` ``[L, P, H, W]``."""

    layers = textures.shape[0]
    packed = jnp.concatenate([jnp.asarray(textures), jnp.asarray(ranks, dtype=textures.dtype)], axis=1)
    return packed.reshape(layers * packed.shape[1], *packed.shape[-2:])


def fill_level(density: Array, layer_coverage: Array) -> Array:
    """How far to fill the layers (0 to ``L``) to cover ``density`` of the grid.

    Filling layer ``k`` with share ``f`` of its cells covers ``f *
    layer_coverage[k]`` of the grid. The layers are independent, so with
    layers ``0 .. k-1`` full the grid is covered ``1 - prod_j (1 - c_j)
    (1 - f c_k)``. The result is ``k + f``; densities above what all layers
    cover give ``L``.
    """

    uncovered = jnp.cumprod(jnp.concatenate([jnp.ones(1), 1.0 - layer_coverage[:-1]]))
    share = (1.0 - (1.0 - density) / uncovered) / layer_coverage
    return jnp.sum(jnp.clip(share, 0.0, 1.0))


def kept_cells(ranks: Array, level: Array, shift: Array, softness: float) -> Array:
    """How much each cell pixel is kept (0 to 1), for rank maps ``[L, P, H, W]``."""

    layer = jnp.arange(ranks.shape[0], dtype=ranks.dtype)[:, None, None, None]
    order = layer + jnp.mod(ranks + shift, 1.0)
    return jnp.where(ranks >= 0, jax.nn.sigmoid((level - order) / softness), 0.0)


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
    """How to read a packed density texture and fill the gaps between its cells.

    ``gap_values`` ``[C]`` is each state channel's value in the gaps,
    ``channel_panels`` ``[C]`` the staining panel each state channel's cells
    come from, and ``layer_coverage`` ``[L]`` the share of the grid each
    full layer covers (averaged over the textures).
    """

    gap_values: Array
    channel_panels: Array
    layer_coverage: Array
    panels: int = eqx.field(static=True)
    softness: float = eqx.field(static=True, default=0.02)

    @property
    def max_density(self) -> Array:
        return 1.0 - jnp.prod(1.0 - self.layer_coverage)

    def unpack(self, packed: Array) -> tuple[Array, Array]:
        """``(textures [L, C, H, W], ranks [L, P, H, W])`` of a packed texture."""

        layers, channels = self.layer_coverage.shape[0], self.gap_values.shape[0]
        packed = packed.reshape(layers, channels + self.panels, *packed.shape[-2:])
        return packed[:, :channels], packed[:, channels:]

    def coverage(self, packed: Array, density: CellDensity, key: Array) -> Array:
        """How much each pixel is covered by cells of each panel ``[P, H, W]``."""

        _, ranks = self.unpack(packed)
        kept = self._kept(ranks, density, key)
        return 1.0 - jnp.prod(1.0 - kept, axis=0)

    def apply(self, packed: Array, density: CellDensity, key: Array) -> Array:
        """The state channels ``[C, H, W]`` of a packed texture with cells for ``density``, chosen by ``key``."""

        textures, ranks = self.unpack(packed)
        kept = self._kept(ranks, density, key)[:, self.channel_panels]
        image = jnp.broadcast_to(
            jnp.asarray(self.gap_values, dtype=textures.dtype)[:, None, None], textures.shape[1:]
        )
        for layer in reversed(range(textures.shape[0])):
            image = kept[layer] * textures[layer] + (1.0 - kept[layer]) * image
        return image

    def _kept(self, ranks, density, key):
        shift = jax.random.uniform(key, dtype=ranks.dtype)
        level = fill_level(density.value, self.layer_coverage)
        return kept_cells(ranks, level, shift, self.softness)
