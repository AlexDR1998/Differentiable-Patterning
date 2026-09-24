"""Result containers for geometry optimisation."""

from dataclasses import dataclass
from typing import Any

from jaxtyping import Array


@dataclass(frozen=True)
class OptimisationResult:
    geometry: Any
    loss_history: Array
    component_history: dict[str, Array]
    final_occupancy: Array
    final_state: Array
    sampled_biology: Array
