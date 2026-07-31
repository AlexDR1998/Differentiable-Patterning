"""Protocols and shared type aliases for NCA augmenters."""

from __future__ import annotations

from typing import Any, Protocol, Tuple, TypeAlias

import jax
from jaxtyping import PyTree

AugmenterBatch: TypeAlias = Tuple[PyTree[jax.Array], PyTree[jax.Array]]


class NCAAugmenterProtocol(Protocol):
    """Minimum interface consumed by the NCA training loop.

    Implementations may store a data pool, but each stochastic callback must
    derive its result from the supplied key. This keeps the numerical part of
    augmentation reproducible and compatible with JAX transformations.
    """

    OBS_CHANNELS: int

    def data_init(self, SHARDING: Any = None) -> None:
        ...

    def data_load(self, key: jax.Array) -> AugmenterBatch:
        ...

    def data_callback(
        self,
        x: PyTree[jax.Array],
        y: PyTree[jax.Array],
        i: int,
        key: jax.Array,
    ) -> AugmenterBatch:
        ...

    def return_saved_data(self) -> PyTree[jax.Array]:
        ...
