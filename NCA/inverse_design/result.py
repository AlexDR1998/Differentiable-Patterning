"""Result containers for geometry optimisation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Array


@dataclass(frozen=True)
class BinaryEvaluation:
    """Rollouts of one geometry with its binary mask (``level_set < 0``).

    ``scores`` holds the hard target share of each rollout, ``final_states``
    the states ``[seeds, C, H, W]`` and ``border_fraction`` the share of the
    outermost two pixels covered by the mask (above 0, the colony reaches the
    edge of the grid, which wraps round with circular padding).
    """

    mask: Array
    final_states: Array
    scores: Array
    border_fraction: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores))


@dataclass(frozen=True)
class OptimisationResult:
    geometry: Any
    best_geometry: Any
    best_iteration: int
    loss_history: Array
    component_history: dict[str, Array]
    # Binary evaluations at iterations 0, log_every, 2 log_every, ... and the end
    evaluation_iterations: tuple[int, ...]
    evaluations: tuple[BinaryEvaluation, ...]

    @property
    def best_evaluation(self) -> BinaryEvaluation:
        return self.evaluations[self.evaluation_iterations.index(self.best_iteration)]
