"""Protocols and shared type aliases for NCA augmenters."""

from __future__ import annotations

from typing import Protocol, Tuple, TypeAlias

import jax
from jaxtyping import PyTree

AugmenterBatch: TypeAlias = Tuple[PyTree[jax.Array], PyTree[jax.Array]]


class NCAAugmenterProtocol(Protocol):
    """Methods the NCA training loop calls on an augmenter.

    All randomness must come from the key passed in, so runs are reproducible.
    """

    OBS_CHANNELS: int

    def initialize_pool(self, key: jax.Array) -> AugmenterBatch:
        ...

    def advance_pool(
        self,
        x: PyTree[jax.Array],
        y: PyTree[jax.Array],
        i: int,
        key: jax.Array,
    ) -> AugmenterBatch:
        ...

    def return_saved_data(self) -> PyTree[jax.Array]:
        ...

    def return_observed_data(self) -> PyTree[jax.Array]:
        ...
