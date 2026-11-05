"""Differentiable inverse design of NCA micropattern boundaries."""

from NCA.inverse_design.geometry import FourierRadialGeometry, coordinate_grid
from NCA.inverse_design.initial_state import (
    PatchSamplingConfig,
    build_initial_state,
    sample_circular_patches,
)
from NCA.inverse_design.objectives import (
    CellTypeShareObjective,
    GeometryPenalties,
    geometry_loss,
)
from NCA.inverse_design.optimise import (
    OptimisationConfig,
    binary_trajectories,
    evaluate_binary,
    optimise_geometry,
)
from NCA.inverse_design.result import BinaryEvaluation, OptimisationResult
from NCA.inverse_design.rollout import rollout_final, rollout_trajectory

__all__ = (
    "BinaryEvaluation",
    "CellTypeShareObjective",
    "FourierRadialGeometry",
    "GeometryPenalties",
    "OptimisationConfig",
    "OptimisationResult",
    "PatchSamplingConfig",
    "binary_trajectories",
    "build_initial_state",
    "coordinate_grid",
    "evaluate_binary",
    "geometry_loss",
    "optimise_geometry",
    "rollout_final",
    "rollout_trajectory",
    "sample_circular_patches",
)
