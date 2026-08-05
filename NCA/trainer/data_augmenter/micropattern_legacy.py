"""Unified schema-driven augmenter for NCA micropattern datasets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from Common.dataloader.micropattern_schemas import (
    MICROPATTERN_4CH_SCHEMA,
    MICROPATTERN_260726_SCHEMA,
)
from NCA.trainer.data_augmenter import (
    bernoulli_reinject_observations,
    split_trajectory,
)
from NCA.trainer.data_augmenter.nca_basic import DataAugmenter as BasicAugmenter


class MicropatternDataAugmenter(BasicAugmenter):
    """Handle channel conversion, groups, and replicate pools from a schema.

    The input is a list/array of raw measurement trajectories. The schema
    chooses one primary measurement per biological state for model inputs while
    preserving all measurements as targets. Batch cardinality is fixed by the
    loader and remains unchanged during augmentation.
    """

    schema = MICROPATTERN_4CH_SCHEMA
    # The legacy 4- and 9-state micropattern augmenters used 1% noise. The
    # snapshot augmenter below overrides this with its historical 0.5% value.
    noise_strength = 0.01
    group_reinjection = False
    supports_global_donor_pool = False
    supports_global_reinjection_mask = True

    def __init__(self, *args, **kwargs):
        self.schema = kwargs.pop("schema", None) or type(self).schema
        self.group_reinjection = kwargs.pop(
            "group_reinjection", type(self).group_reinjection
        )
        self.intermediate_reinjection_probability = kwargs.pop(
            "intermediate_reinjection_probability", 0.5
        )
        self.intermediate_reinjection_probability_end = kwargs.pop(
            "intermediate_reinjection_probability_end",
            self.intermediate_reinjection_probability,
        )
        self.intermediate_reinjection_decay_start_fraction = kwargs.pop(
            "intermediate_reinjection_decay_start_fraction", 0.25
        )
        self.intermediate_reinjection_total_iterations = kwargs.pop(
            "intermediate_reinjection_total_iterations", None
        )
        if not all(
            0.0 <= probability <= 1.0
            for probability in (
                self.intermediate_reinjection_probability,
                self.intermediate_reinjection_probability_end,
            )
        ):
            raise ValueError(
                "intermediate reinjection probabilities must be between 0 and 1"
            )
        if not 0.0 <= self.intermediate_reinjection_decay_start_fraction < 1.0:
            raise ValueError(
                "intermediate_reinjection_decay_start_fraction must be in [0, 1)"
            )
        model = kwargs.pop("nca_model", None)
        self.channels = (
            model.N_CHANNELS
            if model is not None
            else self.schema.n_state_channels + kwargs.get("hidden_channels", 0)
        )
        if self.channels < self.schema.n_state_channels:
            raise ValueError(
                f"NCA has fewer channels than schema {self.schema.name!r} requires"
            )
        super().__init__(*args, **kwargs)
        self.OBS_CHANNELS = self.schema.n_state_channels
        self.state_groups = tuple(
            tuple(self.schema.target_to_state[index] for index in group)
            for group in self.schema.group_measurement_indices
        )

    def _to_state(self, data):
        state = data[:, self.schema.primary_measurements]
        return jnp.pad(
            state,
            ((0, 0), (0, self.channels - state.shape[1]), (0, 0), (0, 0)),
        )

    def split_x_y(self, N_steps=1):
        state_data = jtu.tree_map(self._to_state, self.data_saved)
        x, _ = split_trajectory(state_data, N_steps)
        y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        return x, y

    def initialize_pool(self, key):
        """Create time-aligned initial states without advancing the pool.

        ``advance_pool`` expects completed NCA rollouts and shifts them into
        the following transition slots. Raw snapshots are already aligned to
        those slots, so initialization only applies input noise.
        """

        x, y = self.split_x_y(1)
        x = self.noise(x, self.noise_strength, key=key)
        self.PREVIOUS_KEY = key
        return x, y

    def reinjection_probability(self, i):
        """Return the configured piecewise-linear reinjection probability."""

        probability = self.intermediate_reinjection_probability
        if self.intermediate_reinjection_total_iterations is None:
            return probability
        decay_start = (
            self.intermediate_reinjection_decay_start_fraction
            * self.intermediate_reinjection_total_iterations
        )
        decay_duration = max(
            self.intermediate_reinjection_total_iterations - decay_start, 1
        )
        progress = jnp.clip((i - decay_start) / decay_duration, 0.0, 1.0)
        return probability + progress * (
            self.intermediate_reinjection_probability_end - probability
        )

    def _group_reinject(self, x, i, key):
        """Apply the legacy snapshot pool update with coherent group donors.

        One experiment group is selected for each admitted batch/time slot.
        All measurements in that group come from the same donor, preventing
        overlapping state channels from being assembled from incompatible
        experiments. Random choices are generated over the global donor pool
        before selecting local shard indices.
        """

        saved = jnp.stack(self.data_saved)
        x = jnp.stack(x)
        measurements = saved[:, :, : self.schema.n_measurement_channels]
        truth = jax.vmap(self._to_state)(saved)
        batch_count, time_count = x.shape[:2]
        donor_count = saved.shape[0]
        global_indices = jnp.asarray(
            getattr(self, "_global_batch_indices", jnp.arange(batch_count))
        )
        global_key = getattr(self, "_sharded_global_key", key)

        x = x.at[:, 1:].set(x[:, :-1])

        global_key, reset_key = jax.random.split(global_key)
        reset_donors = jax.random.permutation(reset_key, donor_count)[global_indices]
        x = x.at[:, 0].set(truth[reset_donors, 0])

        if time_count > 1:
            global_key, mask_key, group_key = jax.random.split(global_key, 3)
            global_shape = (donor_count, time_count - 1)
            probability = self.reinjection_probability(i)
            inject = jax.random.bernoulli(
                mask_key, probability, global_shape
            )[global_indices]
            choices = jax.random.randint(
                group_key,
                global_shape,
                0,
                len(self.schema.experiment_groups),
            )[global_indices]

            for time_index in range(1, time_count):
                for group_index, (measurement_indices, state_indices) in enumerate(
                    zip(self.schema.group_measurement_indices, self.state_groups)
                ):
                    global_key, donor_key = jax.random.split(global_key)
                    donors = jax.random.permutation(
                        donor_key, donor_count
                    )[global_indices]
                    values = measurements[donors, time_index][:, measurement_indices]
                    keep = (
                        inject[:, time_index - 1]
                        & (choices[:, time_index - 1] == group_index)
                    )[:, None, None, None]
                    x = x.at[:, time_index, state_indices].set(
                        jnp.where(keep, values, x[:, time_index, state_indices])
                    )

        noise_key = key if hasattr(self, "_sharded_global_key") else global_key
        result = self.noise(list(x), self.noise_strength, key=noise_key)
        self.PREVIOUS_KEY = noise_key
        return result

    def advance_pool(self, x, y, i, key):
        if self.group_reinjection:
            x = self._group_reinject(x, i, key)
        else:
            x_true, _ = self.split_x_y(1)
            global_key = getattr(self, "_sharded_global_key", key)
            x = bernoulli_reinject_observations(
                x,
                x_true,
                self.OBS_CHANNELS,
                global_key,
                self.reinjection_probability(i),
                getattr(self, "_global_batch_indices", None),
                getattr(self, "_global_batch_count", None),
            )
            x = self.noise(x, self.noise_strength, key=key)
            self.PREVIOUS_KEY = key
        return x, y


class MicropatternSnapshotAugmenter(MicropatternDataAugmenter):
    schema = MICROPATTERN_260726_SCHEMA
    noise_strength = 0.005
    group_reinjection = True
    supports_global_donor_pool = True


__all__ = ["MicropatternDataAugmenter", "MicropatternSnapshotAugmenter"]
