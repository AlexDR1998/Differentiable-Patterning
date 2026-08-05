"""JAX-compatible building blocks and active NCA data augmenters.

The low-level transforms remain exported here, while task-specific augmenters
live in this package as separate modules.
"""

from .protocols import AugmenterBatch, NCAAugmenterProtocol
from .transforms import (
    add_noise,
    bernoulli_reinject_observations,
    propagate_pool,
    reinject_observations,
    scheduled_probability,
    terminal_carry,
)
from .trajectory import split_trajectory

__all__ = [
    "AugmenterBatch",
    "NCAAugmenterProtocol",
    "add_noise",
    "bernoulli_reinject_observations",
    "propagate_pool",
    "reinject_observations",
    "scheduled_probability",
    "split_trajectory",
    "terminal_carry",
]
