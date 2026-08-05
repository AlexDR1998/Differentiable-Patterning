from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TargetedObjective:
    """Minimise target error and the weighted intervention penalty."""

    target_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics):
        """Return the scalar targeted intervention objective."""

        return self.target_weight * target_loss + intervention_penalty


@dataclass(frozen=True)
class MinimalDestructiveObjective:
    """Maximise target error while penalising intervention size."""

    target_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics):
        """Return the scalar destructive intervention objective."""

        return -self.target_weight * target_loss + intervention_penalty


@dataclass(frozen=True)
class MaximalPreservativeObjective:
    """Increase intervention size while keeping target error below a tolerance."""

    tolerance: float = 0.01
    constraint_weight: float = 100.0
    magnitude: str = "l2"
    reward_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics):
        """Return a soft constrained preservation objective."""

        violation = jnp.maximum(target_loss - self.tolerance, 0.0)
        magnitude = intervention_metrics[self.magnitude]
        return (
            self.constraint_weight * violation**2
            - self.reward_weight * magnitude
            + intervention_penalty
        )

