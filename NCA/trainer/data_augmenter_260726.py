"""Pool augmentation for unordered 260726 micropattern snapshots."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as BasicAugmenter
from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA


class DataAugmenter(BasicAugmenter):
    """Augmenter for the selected, possibly multiplied 260726 batch pool."""

    noise_strength = 0.005
    schema = MICROPATTERN_260726_SCHEMA
    supports_global_donor_pool = True

    def __init__(self, *args, **kwargs):
        self.schema = kwargs.pop("schema", None) or type(self).schema
        self.batch_multiplier = kwargs.pop("batch_multiplier", 1)
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
        model = kwargs.get("nca_model")
        self.channels = (
            model.N_CHANNELS if model is not None
            else self.schema.n_state_channels + kwargs.get("hidden_channels", 0)
        )
        if self.channels < self.schema.n_state_channels:
            raise ValueError("NCA has fewer channels than the measurement schema")
        super().__init__(*args, **kwargs)
        self.OBS_CHANNELS = self.schema.n_state_channels
        self.state_groups = tuple(
            tuple(self.schema.target_to_state[channel] for channel in group)
            for group in self.schema.group_measurement_indices
        )

    def _to_state(self, data):
        state = data[:, self.schema.primary_measurements]
        return jnp.pad(state, ((0, 0), (0, self.channels - state.shape[1]), (0, 0), (0, 0)))

    def split_x_y(self, N_steps=1):
        if self.batch_mode == "array":
            x = self.map_batches(
                lambda data: self.real_to_latent(self._to_state(data[:-N_steps])),
                self.data_saved,
            )
            y = self.data_saved[:, N_steps:]
        else:
            x = jtu.tree_map(lambda data: self.real_to_latent(self._to_state(data[:-N_steps])), self.data_saved)
            y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        return x, y

    def data_callback(self, x, y, i, key):
        """Propagate the pool and reinject randomly permuted experiment groups."""
        x = self.as_array(x)
        saved = self.as_array(self.data_saved)
        measurements = saved[:, :, :self.schema.n_measurement_channels]
        truth = jax.vmap(lambda data: self.real_to_latent(self._to_state(data)))(saved)
        batch_count, time_count = x.shape[:2]
        donor_count = saved.shape[0]
        global_indices = jnp.asarray(
            getattr(self, "_global_batch_indices", jnp.arange(batch_count))
        )
        global_key = getattr(self, "_sharded_global_key", key)
        x = x.at[:, 1:].set(x[:, :-1])

        global_key, reset_key = jax.random.split(global_key)
        reset_donors = jax.random.permutation(reset_key, donor_count)[global_indices]
        reset = truth[reset_donors, 0]
        x = x.at[:, 0].set(reset)

        if time_count > 1:
            global_key, mask_key, group_key = jax.random.split(global_key, 3)
            global_shape = (donor_count, time_count - 1)
            if self.intermediate_reinjection_total_iterations is None:
                reinjection_probability = self.intermediate_reinjection_probability
            else:
                decay_start = (
                    self.intermediate_reinjection_decay_start_fraction
                    * self.intermediate_reinjection_total_iterations
                )
                decay_progress = jnp.clip(
                    (i - decay_start)
                    / (self.intermediate_reinjection_total_iterations - decay_start),
                    0.0,
                    1.0,
                )
                reinjection_probability = (
                    self.intermediate_reinjection_probability
                    + decay_progress
                    * (
                        self.intermediate_reinjection_probability_end
                        - self.intermediate_reinjection_probability
                    )
                )
            inject = jax.random.bernoulli(
                mask_key,
                reinjection_probability,
                global_shape,
            )[global_indices]
            choices = jax.random.randint(
                group_key, global_shape, 0, len(self.schema.experiment_groups)
            )[global_indices]
            for t in range(1, time_count):
                for g, (targets, states) in enumerate(
                    zip(self.schema.group_measurement_indices, self.state_groups)
                ):
                    global_key, donor_key = jax.random.split(global_key)
                    donors = jax.random.permutation(
                        donor_key, donor_count
                    )[global_indices]
                    values = self.real_to_latent(measurements[donors, t][:, targets])
                    keep = (inject[:, t - 1] & (choices[:, t - 1] == g))[:, None, None, None]
                    x = x.at[:, t, states].set(jnp.where(keep, values, x[:, t, states]))

        x = self.restore_batch_mode(x)
        noise_key = key if hasattr(self, "_sharded_global_key") else global_key
        x = self.noise(x, self.noise_strength, key=noise_key)
        self.PREVIOUS_KEY = noise_key
        return x, y
