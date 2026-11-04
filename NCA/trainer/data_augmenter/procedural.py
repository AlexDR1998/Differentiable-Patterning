"""Pool augmentation for tasks whose trajectories are drawn at random every step.

Used for the signalling tasks (Experiments/signalling/tasks.py): every
instance has new random cues, so the NCA can't memorise the answers.
"""

import jax
import jax.numpy as jnp
import numpy as np

from NCA.trainer.data_augmenter.base import PoolAugmenter, as_trajectory_list
from NCA.trainer.data_augmenter.transforms import add_noise


class ProceduralTaskAugmenter(PoolAugmenter):
    """Pool of random task instances.

    After every step, ``resample_fraction`` of the trajectories are replaced
    by fresh random instances. The others move one slot along, as in the
    other augmenters: each prediction becomes the start state of the next
    transition, and slot 0 is reset to the instance's initial state.

    Parameters
    ----------
    sample : callable (seed: int, count: int) -> array [count, T, C, H, W]
        Draws ``count`` random trajectories.
    batches : int
        Number of trajectories in the pool.
    hidden_channels : int
        Zero channels added after the observed ones.
    resample_fraction : float
        Fraction of the pool replaced after every step.
    noise_strength : float
        Gaussian noise blended into the observed channels of the start states.
    seed : int
        Seed of the first pool.
    evaluate : callable (model, key) -> (dict of scalars, dict of images), optional
        Task measures, logged by the W&B logger at the end of training.
    """

    def __init__(self, sample, batches, hidden_channels=0, resample_fraction=0.5,
                 noise_strength=0.0, seed=0, evaluate=None):
        super().__init__(jnp.asarray(sample(seed, batches)), hidden_channels)
        self.sample = sample
        self.resample_fraction = resample_fraction
        self.noise_strength = noise_strength
        self.evaluate = evaluate

    def _noisy(self, x, key):
        if not self.noise_strength:
            return x
        return add_noise(x, self.noise_strength, key, mode="observable", observable_channels=self.OBS_CHANNELS)

    def initialize_pool(self, key):
        x, y = self.split_x_y(1)
        return self._noisy(x, key), y

    def advance_pool(self, x, y, i, key):
        count = len(x)
        choice_key, sample_key, noise_key = jax.random.split(key, 3)
        replaced = np.asarray(jax.random.permutation(choice_key, count))[: round(self.resample_fraction * count)]
        saved = list(self.data_saved)
        if len(replaced):
            seed = int(jax.random.randint(sample_key, (), 0, 2**31 - 1))
            fresh = jnp.pad(
                jnp.asarray(self.sample(seed, len(replaced))),
                ((0, 0), (0, 0), (0, self.hidden_channels), (0, 0), (0, 0)),
            )
            for index, trajectory in zip(replaced, as_trajectory_list(fresh)):
                saved[index] = trajectory
        x_next, y_next = [], []
        for b in range(count):
            if b in replaced:
                x_next.append(saved[b][:-1])
                y_next.append(saved[b][1:])
            else:
                x_next.append(jnp.concatenate([saved[b][:1], x[b][:-1]]))
                y_next.append(y[b])
        self.data_saved = saved
        return self._noisy(x_next, noise_key), y_next


__all__ = ["ProceduralTaskAugmenter"]
