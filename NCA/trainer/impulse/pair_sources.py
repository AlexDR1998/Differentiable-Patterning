from dataclasses import dataclass

import jax
import jax.numpy as jnp

from NCA.trainer.impulse.rollout import identity_boundary, run_nca_batch
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
    """Generate source and target attractors from condition-specific initial states."""

    condition_states: object
    source_index: int
    target_index: int
    stabilisation_steps: tuple[int, int]
    boundary_callback: object = identity_boundary
    scan_kind: str = "lax"

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

        source_states = []
        target_states = []
        sampled_steps = []
        for batch_index in range(batch_size):
            item_key = jax.random.fold_in(key, batch_index)
            steps = int(jax.random.randint(item_key, (), minimum, maximum))
            stable = run_nca_batch(
                model,
                conditions,
                steps,
                item_key,
                boundary_callback=self.boundary_callback,
                scan_kind=self.scan_kind,
            )
            source_states.append(stable[self.source_index])
            target_states.append(stable[self.target_index])
            sampled_steps.append(steps)
        return ImpulseBatch(
            jnp.stack(source_states),
            jnp.stack(target_states),
            {"stabilisation_steps": sampled_steps},
        )

