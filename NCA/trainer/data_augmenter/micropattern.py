"""Pool augmentation for micropattern data, driven by a channel schema.

The data holds raw measurement channels. The schema chooses one measurement
per biological state channel as the model input, while all measurements stay
as targets (some markers were measured in more than one experiment).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.data_augmenter.base import PoolAugmenter
from NCA.trainer.data_augmenter.trajectory import split_trajectory
from NCA.trainer.data_augmenter.transforms import add_noise
from NCA.trainer.intervention import intervention_slot

# NODAL state channel of the older colony datasets (4- and 12-measurement)
COLONY_NODAL_CHANNEL = 7


class MicropatternAugmenter(PoolAugmenter):
    """Pool for micropattern training.

    After every step the pool moves one slot along each trajectory, the first
    slot is reset to a true initial state, and some later slots get true
    measurements back (with probability ``reinjection_probability``, which can
    change linearly to ``reinjection_probability_end`` after
    ``reinjection_decay_start_fraction * total_iterations``). Then Gaussian
    noise of strength ``noise_strength`` is blended in.

    ``reinjection`` chooses how true measurements are put back:

    * ``"group"`` (260726 snapshot data): for each slot one experiment group
      is chosen, and all its channels come from the same randomly chosen
      replicate. Replicates are only mixed within the same knockout condition
      (``intervention_times``). ``measurement_mask`` [batch, time - 1,
      measurement] marks which measurements exist.
    * ``"masked"`` (older colony data): true state channels are put back where
      ``measurement_mask`` [batch, time - 1, state] is set, and NODAL is
      zeroed from each batch's ``knockout_times`` entry onwards (None or -1
      means no knockout). ``observation_times`` maps knockout times to slots
      for non-uniform time intervals.

    ``channels`` is the number of NCA channels; hidden channels start at zero.
    """

    def __init__(
        self,
        data,
        schema,
        channels,
        *,
        reinjection="group",
        noise_strength=0.005,
        reinjection_probability=0.5,
        reinjection_probability_end=None,
        reinjection_decay_start_fraction=0.25,
        total_iterations=None,
        measurement_mask=None,
        intervention_times=None,
        knockout_times=None,
        observation_times=None,
    ):
        if reinjection not in {"group", "masked"}:
            raise ValueError("reinjection must be 'group' or 'masked'")
        if reinjection_probability_end is None:
            reinjection_probability_end = reinjection_probability
        if not all(0.0 <= p <= 1.0 for p in (reinjection_probability, reinjection_probability_end)):
            raise ValueError("intermediate reinjection probabilities must be between 0 and 1")
        if not 0.0 <= reinjection_decay_start_fraction < 1.0:
            raise ValueError("intermediate_reinjection_decay_start_fraction must be in [0, 1)")
        if channels < schema.n_state_channels:
            raise ValueError(f"NCA has fewer channels than schema {schema.name!r} requires")
        super().__init__(data, hidden_channels=0)

        self.schema = schema
        self.channels = channels
        self.reinjection = reinjection
        self.noise_strength = noise_strength
        self.reinjection_probability_start = reinjection_probability
        self.reinjection_probability_end = reinjection_probability_end
        self.reinjection_decay_start_fraction = reinjection_decay_start_fraction
        self.total_iterations = total_iterations
        self.observation_times = None if observation_times is None else tuple(observation_times)
        self.OBS_CHANNELS = schema.n_state_channels
        batch_count = len(self.data_saved)
        target_times = self.data_saved[0].shape[0] - 1

        # Read by the trainer and logger for knockout runs
        self.intervention_times = intervention_times
        self.nodal_channel = (
            schema.state_channels.index("NODAL") if intervention_times is not None else None
        )
        if intervention_times is not None and len(intervention_times) != batch_count:
            raise ValueError("Intervention times must match the global reinjection donor pool")

        if reinjection == "group":
            expected_shape = (batch_count, target_times, schema.n_measurement_channels)
            if measurement_mask is not None and measurement_mask.shape != expected_shape:
                raise ValueError(
                    "Reinjection measurement mask must have shape "
                    f"[batch, target_time, measurement]={expected_shape}, got "
                    f"{measurement_mask.shape}"
                )
            self.measurement_mask = measurement_mask
            self.state_groups = tuple(
                tuple(schema.target_to_state[index] for index in group)
                for group in schema.group_measurement_indices
            )
        else:
            if measurement_mask is None:
                measurement_mask = jnp.ones((batch_count, target_times, self.OBS_CHANNELS))
            self.measurement_mask = jnp.asarray(measurement_mask, dtype=jnp.float32)
            if knockout_times is None:
                knockout_times = [None] * batch_count
            self.knockout_times = jnp.array(
                [-1 if time is None else time for time in knockout_times], dtype=jnp.int32
            )

    def _to_state(self, data):
        """Measurements -> state channels (plus zero hidden channels)."""
        state = data[:, self.schema.primary_measurements]
        return jnp.pad(state, ((0, 0), (0, self.channels - state.shape[1]), (0, 0), (0, 0)))

    def split_x_y(self, n_steps=1):
        """States built from the measurements, and the raw measurements as targets."""
        x, _ = split_trajectory(jtu.tree_map(self._to_state, self.data_saved), n_steps)
        y = jtu.tree_map(lambda data: data[n_steps:], self.data_saved)
        return x, y

    def initialize_pool(self, key):
        # The raw snapshots are already in the right slots, so only add noise
        x, y = self.split_x_y(1)
        return add_noise(x, self.noise_strength, key), y

    def reinjection_probability(self, i):
        """Reinjection probability at iteration ``i``."""
        probability = self.reinjection_probability_start
        if self.total_iterations is None:
            return probability
        decay_start = self.reinjection_decay_start_fraction * self.total_iterations
        decay_duration = max(self.total_iterations - decay_start, 1)
        progress = jnp.clip((i - decay_start) / decay_duration, 0.0, 1.0)
        return probability + progress * (self.reinjection_probability_end - probability)

    def advance_pool(self, x, y, i, key):
        if self.reinjection == "group":
            x, noise_key = self._group_reinject(x, i, key)
        else:
            x_true, _ = self.split_x_y(1)
            x = masked_reinject(
                x,
                x_true,
                self.OBS_CHANNELS,
                jax.random.fold_in(key, 0),
                self.measurement_mask,
                self.knockout_times,
                self.reinjection_probability(i),
                self.observation_times,
            )
            noise_key = key
        return add_noise(x, self.noise_strength, noise_key), y

    def _matched_donors(self, key, donor_count):
        """Random replicate for each batch, only swapping within one knockout condition."""
        if self.intervention_times is None:
            return jax.random.permutation(key, donor_count)
        intervention_times = tuple(int(value) for value in self.intervention_times)
        donors = jnp.arange(donor_count)
        for group_index, intervention_time in enumerate(dict.fromkeys(intervention_times)):
            indices = jnp.asarray(
                [index for index, value in enumerate(intervention_times) if value == intervention_time]
            )
            donors = donors.at[indices].set(
                jax.random.permutation(jax.random.fold_in(key, group_index), indices)
            )
        return donors

    def _group_reinject(self, x, i, key):
        """Pool update where each reinjected experiment group comes from one replicate.

        Returns the new pool and the key to use for noise.
        """
        saved = jnp.stack(self.data_saved)
        x = jnp.stack(x)
        measurements = saved[:, :, : self.schema.n_measurement_channels]
        truth = jax.vmap(self._to_state)(saved)
        time_count = x.shape[1]
        donor_count = saved.shape[0]

        x = x.at[:, 1:].set(x[:, :-1])

        key, reset_key = jax.random.split(key)
        x = x.at[:, 0].set(truth[self._matched_donors(reset_key, donor_count), 0])

        if time_count > 1:
            key, mask_key, group_key = jax.random.split(key, 3)
            shape = (donor_count, time_count - 1)
            inject = jax.random.bernoulli(mask_key, self.reinjection_probability(i), shape)
            choices = jax.random.randint(group_key, shape, 0, len(self.schema.experiment_groups))

            for time_index in range(1, time_count):
                for group_index, (measurement_indices, state_indices) in enumerate(
                    zip(self.schema.group_measurement_indices, self.state_groups)
                ):
                    key, donor_key = jax.random.split(key)
                    donors = self._matched_donors(donor_key, donor_count)
                    values = measurements[donors, time_index][:, measurement_indices]
                    if self.measurement_mask is None:
                        measured = jnp.ones(values.shape[:2], dtype=bool)
                    else:
                        measured = jnp.asarray(self.measurement_mask)[
                            donors, time_index - 1
                        ][:, measurement_indices].astype(bool)
                    # Only measured channels are replaced. Hidden channels keep
                    # the propagated state as trajectory memory.
                    keep = (
                        inject[:, time_index - 1] & (choices[:, time_index - 1] == group_index)
                    )[:, None, None, None] & measured[:, :, None, None]
                    x = x.at[:, time_index, state_indices].set(
                        jnp.where(keep, values, x[:, time_index, state_indices])
                    )
        return list(x), key


@eqx.filter_jit
def masked_reinject(
    x,
    x_true,
    obs_channels,
    key,
    channel_timestep_mask,
    knockout_times,
    probability,
    observation_times=None,
):
    """Move the pool on one slot, reinject measured state channels, and apply knockouts."""
    x = jax.tree_util.tree_map(lambda xi: xi.at[1:].set(xi[:-1]), x)
    x = jax.tree_util.tree_map(lambda xi, xi_true: xi.at[0].set(xi_true[0]), x, x_true)

    B = len(x)
    T = x[0].shape[0]
    inject_mask = jax.random.bernoulli(key, probability, shape=(B, T - 1))

    if T > 1:
        for b in range(B):
            measured = channel_timestep_mask[b, : T - 1]
            if measured.shape[1] < obs_channels:
                measured = jnp.pad(
                    measured,
                    ((0, 0), (0, obs_channels - measured.shape[1])),
                    constant_values=0,
                )
            measured = measured[:, :obs_channels]
            mask = (inject_mask[b, :, None] & measured.astype(bool))[:, :, None, None]
            x_obs = jnp.where(mask, x_true[b][1:, :obs_channels], x[b][1:, :obs_channels])
            x[b] = x[b].at[1:, :obs_channels].set(x_obs)

    for b in range(B):
        knockout_time = knockout_times[b]
        knockout_index = intervention_slot(knockout_time, observation_times)
        zero_mask = (knockout_time >= 0) & (jnp.arange(T) >= knockout_index)
        nodal = jnp.where(zero_mask[:, None, None], 0.0, x[b][:, COLONY_NODAL_CHANNEL])
        x[b] = x[b].at[:, COLONY_NODAL_CHANNEL].set(nodal)
    return x


__all__ = ["COLONY_NODAL_CHANNEL", "MicropatternAugmenter", "masked_reinject"]
