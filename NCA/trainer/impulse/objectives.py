from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Every objective is called as objective(target_loss, intervention_penalty,
# intervention_metrics, predictions), where predictions are the rollout states
# the target loss was computed on, [samples, >= observed channels, H, W].


@dataclass(frozen=True)
class TargetedObjective:
    """Minimise target error and the weighted intervention penalty."""

    target_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics, predictions=None):
        """Return the scalar targeted intervention objective."""

        return self.target_weight * target_loss + intervention_penalty


@dataclass(frozen=True)
class MinimalDestructiveObjective:
    """Maximise target error while penalising intervention size."""

    target_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics, predictions=None):
        """Return the scalar destructive intervention objective."""

        return -self.target_weight * target_loss + intervention_penalty


@dataclass(frozen=True)
class MaximalPreservativeObjective:
    """Increase intervention size while keeping target error below a tolerance."""

    tolerance: float = 0.01
    constraint_weight: float = 100.0
    magnitude: str = "l2"
    reward_weight: float = 1.0

    def __call__(self, target_loss, intervention_penalty, intervention_metrics, predictions=None):
        """Return a soft constrained preservation objective."""

        violation = jnp.maximum(target_loss - self.tolerance, 0.0)
        magnitude = intervention_metrics[self.magnitude]
        return (
            self.constraint_weight * violation**2
            - self.reward_weight * magnitude
            + intervention_penalty
        )


@dataclass(frozen=True)
class AttractorContrastObjective:
    """Targeted objective plus a push away from the other attractors of the model.

    The contrast term is a cross-entropy over -distance / temperature to each
    attractor, with the target as the label. On its own it is about
    (d_target - min_k d_k) / temperature, which is zero for any state nearer
    the target than every other attractor, even one far from all of them.
    The target loss keeps the state on the target; the contrast term only
    pushes it away from the source and the other attractors.
    """

    # [K, observed channels, H, W] patterns of a multi-attractor model
    attractors: object
    # Index of the target pattern in attractors
    target_index: int
    temperature: float
    target_weight: float = 1.0
    contrast_weight: float = 0.1

    def __call__(self, target_loss, intervention_penalty, intervention_metrics, predictions=None):
        """Return the targeted objective plus the weighted mean cross-entropy."""

        if predictions is None:
            raise ValueError("AttractorContrastObjective needs the rollout states")
        attractors = jnp.asarray(self.attractors)
        states = predictions[:, : attractors.shape[1]]
        # [samples, K] mean squared error to each attractor
        distances = jnp.mean((states[:, None] - attractors[None]) ** 2, axis=(-3, -2, -1))
        cross_entropy = -jax.nn.log_softmax(-distances / self.temperature, axis=-1)[:, self.target_index]
        return (
            self.target_weight * target_loss
            + self.contrast_weight * jnp.mean(cross_entropy)
            + intervention_penalty
        )
