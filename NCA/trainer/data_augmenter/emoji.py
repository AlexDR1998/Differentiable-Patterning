"""Pool augmentation for emoji trajectories."""

import equinox as eqx
import jax
import jax.numpy as jnp

from NCA.trainer.data_augmenter.base import PoolAugmenter
from NCA.trainer.data_augmenter.trajectory import duplicate_batches, pad_spatial
from NCA.trainer.data_augmenter.transforms import (
    add_noise,
    reinject_observations,
    scheduled_probability,
    shift_trajectories,
    terminal_carry,
    zero_random_circles,
)


def schedule_probability(schedule, i):
    """Probability at iteration ``i`` from a schedule with ``enabled``,
    ``start_iteration``, ``schedule_iterations``, ``initial_probability`` and
    ``final_probability`` (e.g. ``data.emoji.regeneration``)."""
    if not schedule.enabled:
        return 0.0
    return scheduled_probability(
        i,
        schedule.start_iteration,
        schedule.schedule_iterations,
        schedule.initial_probability,
        schedule.final_probability,
    )


# Put the true state back into half of the pool slots after every step
_reinject_half = eqx.filter_jit(
    lambda x, x_true, observable_channels, key: reinject_observations(
        x, x_true, observable_channels, key, fraction=0.5
    )
)


class EmojiAugmenter(PoolAugmenter):
    """Pool for emoji training.

    After every step the pool moves one slot along each trajectory, half the
    slots get the true images back, and optionally:

    * ``terminal_carry``: each trajectory keeps its last predicted state, so
      the NCA learns to stay stable over longer rollouts;
    * ``shift_amount``: the images are rolled by a random offset;
    * ``regeneration``: a random circle is set to zero, so the NCA learns to
      repair damage;
    * ``noise_strength``: Gaussian noise is blended in (``noise_mode`` "full",
      "observable" or "hidden" channels).

    ``terminal_carry`` and ``regeneration`` are probability schedules (see
    ``schedule_probability``). The data is repeated ``batches`` times and
    padded spatially by ``pad`` = (top, bottom, left, right), or not if None.
    """

    def __init__(
        self,
        data,
        hidden_channels,
        *,
        batches=1,
        pad=None,
        shift_amount=0,
        noise_strength=0.0,
        noise_mode="full",
        terminal_carry=None,
        regeneration=None,
    ):
        super().__init__(data, hidden_channels)
        self.shift_amount = shift_amount
        self.noise_strength = noise_strength
        self.noise_mode = noise_mode
        self.terminal_carry = terminal_carry
        self.regeneration = regeneration
        # Key of the last shift, so it can be undone before the next step
        self.previous_key = None
        data = duplicate_batches(self.data_saved, batches)
        if pad is not None:
            data = pad_spatial(data, pad)
        self.data_saved = data

    def advance_pool(self, x, y, i, key):
        if self.shift_amount and self.previous_key is not None:
            x = shift_trajectories(x, self.shift_amount, self.previous_key, undo=True)
            y = shift_trajectories(y, self.shift_amount, self.previous_key, undo=True)

        x_true, _ = self.split_x_y(1)
        final_states = [trajectory[-1] for trajectory in x]
        x = _reinject_half(x, x_true, self.OBS_CHANNELS, key)
        if self.terminal_carry is not None:
            x = terminal_carry(
                x,
                final_states,
                schedule_probability(self.terminal_carry, i),
                jax.random.fold_in(key, 1),
            )

        if self.shift_amount:
            x = shift_trajectories(x, self.shift_amount, key)
            y = shift_trajectories(y, self.shift_amount, key)
        if self.regeneration is not None and self.regeneration.enabled:
            damaged = zero_random_circles(x, key)
            damage = jax.random.bernoulli(
                jax.random.fold_in(key, 2),
                schedule_probability(self.regeneration, i),
                (len(x),),
            )
            x = [jnp.where(damage[b], damaged[b], x[b]) for b in range(len(x))]
        if self.noise_strength:
            x = add_noise(
                x,
                self.noise_strength,
                jax.random.fold_in(key, 3),
                mode=self.noise_mode,
                observable_channels=self.OBS_CHANNELS,
            )

        self.previous_key = key
        return x, y
