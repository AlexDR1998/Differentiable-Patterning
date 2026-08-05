from dataclasses import dataclass, field
from typing import Any

from jaxtyping import Array


@dataclass
class ImpulseBatch:
    """Initial and target states used for one intervention optimisation batch."""

    initial_states: Array
    target_states: Array
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpulseResult:
    """Best intervention and evaluation arrays returned by an optimisation run."""

    best_intervention: Any
    best_step: int
    best_loss: float
    metrics: dict[str, Any]
    initial_states: Array
    target_states: Array
    perturbed_initial_states: Array
    final_states: Array
    baseline_trajectory: Array
    perturbed_trajectory: Array

