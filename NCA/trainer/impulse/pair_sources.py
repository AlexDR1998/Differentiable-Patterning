from dataclasses import dataclass

import jax
import jax.numpy as jnp

from NCA.trainer.impulse.rollout import (
    identity_boundary,
    run_nca_batch,
    run_nca_batch_to_steps,
)
from NCA.trainer.impulse.types import ImpulseBatch


def _sample_rows(array, batch_size, key):
    """Sample rows with replacement, broadcasting a single row when possible."""

    array = jnp.asarray(array)
    if array.ndim != 4:
        raise ValueError("State arrays must have shape [batch, channels, height, width]")
    if len(array) == 1:
        return jnp.repeat(array, batch_size, axis=0)
    indices = jax.random.choice(key, len(array), shape=(batch_size,), replace=True)
    return array[indices]


@dataclass
class ExternalTargetPairSource:
    """Sample explicit initial and external target states."""

    initial_states: object
    target_states: object

    def sample(self, batch_size, model, key):
        """Return aligned samples from the supplied initial and target arrays."""

        initial = jnp.asarray(self.initial_states)
        target = jnp.asarray(self.target_states)
        if initial.shape[1:] != target.shape[1:]:
            raise ValueError("Initial and target state shapes must match")
        if len(initial) == len(target) and len(initial) > 1:
            indices = jax.random.choice(key, len(initial), shape=(batch_size,), replace=True)
            return ImpulseBatch(initial[indices], target[indices])
        key_initial, key_target = jax.random.split(key)
        return ImpulseBatch(
            _sample_rows(initial, batch_size, key_initial),
            _sample_rows(target, batch_size, key_target),
        )


@dataclass
class ModelFuturePairSource:
    """Use an unperturbed model future as the target state."""

    initial_states: object
    target_steps: int
    boundary_callback: object = identity_boundary
    scan_kind: str = "lax"

    def sample(self, batch_size, model, key):
        """Sample initial states and generate their unperturbed futures."""

        sample_key, rollout_key = jax.random.split(key)
        initial = _sample_rows(self.initial_states, batch_size, sample_key)
        target = run_nca_batch(
            model,
            initial,
            self.target_steps,
            rollout_key,
            boundary_callback=self.boundary_callback,
            scan_kind=self.scan_kind,
        )
        return ImpulseBatch(initial, target, {"target_steps": self.target_steps})


@dataclass
class TrajectoryStatePairSource:
    """Select initial and target times from one or more stored trajectories."""

    trajectories: object
    initial_index: int = 0
    target_index: int = -1

    def sample(self, batch_size, model, key):
        """Sample trajectories and return the configured timepoint pairs."""

        trajectories = jnp.asarray(self.trajectories)
        if trajectories.ndim == 4:
            trajectories = trajectories[None]
        if trajectories.ndim != 5:
            raise ValueError("Trajectories must have shape [batch, time, channels, height, width]")
        indices = jax.random.choice(key, len(trajectories), shape=(batch_size,), replace=True)
        sampled = trajectories[indices]
        return ImpulseBatch(
            sampled[:, self.initial_index],
            sampled[:, self.target_index],
            {"initial_index": self.initial_index, "target_index": self.target_index},
        )


@dataclass
class StableAttractorPairSource:
    """Generate source and target attractors from condition-specific initial states.

    With ``impulse_time``, the impulse acts on the source grown for exactly
    that many steps, while the target and the source it is scored against
    (``ImpulseBatch.source_states``) are still the mature attractors grown for
    ``stabilisation_steps``.
    """

    condition_states: object
    source_index: int
    target_index: int
    stabilisation_steps: tuple[int, int]
    boundary_callback: object = identity_boundary
    scan_kind: str = "lax"
    impulse_time: int | None = None

    def sample(self, batch_size, model, key):
        """Generate a stochastic pool and select source and target conditions."""

        conditions = jnp.asarray(self.condition_states)
        if conditions.ndim != 4:
            raise ValueError("condition_states must have shape [condition, channels, height, width]")
        if not 0 <= self.source_index < len(conditions):
            raise ValueError("source_index is outside condition_states")
        if not 0 <= self.target_index < len(conditions):
            raise ValueError("target_index is outside condition_states")
        minimum, maximum = self.stabilisation_steps
        if minimum < 0 or maximum <= minimum:
            raise ValueError("stabilisation_steps must be an increasing [minimum, maximum) range")
        if self.impulse_time is not None and self.impulse_time < 0:
            raise ValueError("impulse_time must be non-negative")

        # One step count per batch item, shared by its source and target, as
        # if both conditions had been grown for the same time.
        steps_key, rollout_key = jax.random.split(key)
        steps = jax.random.randint(steps_key, (batch_size,), minimum, maximum)
        if self.impulse_time is not None:
            return self._sample_developing(model, conditions, steps, maximum, rollout_key)
        pair = conditions[jnp.asarray([self.source_index, self.target_index])]
        # [batch, 2, C, H, W] -> [batch * 2, C, H, W], to run as one batch
        initial = jnp.broadcast_to(pair, (batch_size, *pair.shape)).reshape((-1, *pair.shape[1:]))
        stable = run_nca_batch_to_steps(
            model,
            initial,
            jnp.repeat(steps, 2),
            maximum - 1,
            rollout_key,
            boundary_callback=self.boundary_callback,
            scan_kind=self.scan_kind,
        ).reshape((batch_size, *pair.shape))
        return ImpulseBatch(
            stable[:, 0],
            stable[:, 1],
            {"stabilisation_steps": steps},
        )

    def _sample_developing(self, model, conditions, steps, maximum, key):
        """Source grown for impulse_time steps; mature source and target grown for ``steps``."""

        batch_size = len(steps)
        # Per batch item: source at the impulse time, mature source, mature target
        triple = conditions[jnp.asarray([self.source_index, self.source_index, self.target_index])]
        initial = jnp.broadcast_to(triple, (batch_size, *triple.shape)).reshape((-1, *triple.shape[1:]))
        impulse_steps = jnp.full((batch_size,), self.impulse_time)
        grown = run_nca_batch_to_steps(
            model,
            initial,
            jnp.stack([impulse_steps, steps, steps], axis=1).reshape(-1),
            max(maximum - 1, self.impulse_time),
            key,
            boundary_callback=self.boundary_callback,
            scan_kind=self.scan_kind,
        ).reshape((batch_size, *triple.shape))
        return ImpulseBatch(
            grown[:, 0],
            grown[:, 2],
            {"stabilisation_steps": steps, "impulse_time": self.impulse_time},
            source_states=grown[:, 1],
        )
