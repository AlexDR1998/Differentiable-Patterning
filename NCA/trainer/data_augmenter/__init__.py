"""Data augmenters: they hold the training data and update the pool of states.

* ``base.PoolAugmenter`` holds the data; the domain augmenters build on it:
  ``emoji.EmojiAugmenter``, ``micropattern.MicropatternAugmenter``,
  ``snowmelt.SnowmeltAugmenter`` and ``pde.PdeAugmenter``.
* ``transforms.py`` and ``trajectory.py`` hold the pure functions they use
  (reinjection, noise, shifts, damage, padding). Each takes an explicit key.
"""

from .base import PoolAugmenter
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
    "PoolAugmenter",
    "add_noise",
    "bernoulli_reinject_observations",
    "propagate_pool",
    "reinject_observations",
    "scheduled_probability",
    "split_trajectory",
    "terminal_carry",
]
