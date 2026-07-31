"""Pure, JAX-compatible building blocks for NCA data augmentation.

The legacy augmenter classes remain in ``NCA.trainer`` for compatibility. New
augmenters should use the functions and protocols exported here so that data
representation, trajectory sampling, and stochastic transforms stay separate.
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
