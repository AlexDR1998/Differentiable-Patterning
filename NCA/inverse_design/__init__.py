"""Differentiable inverse design of NCA micropattern boundaries."""

from NCA.inverse_design.geometry import FourierRadialGeometry, coordinate_grid
from NCA.inverse_design.initial_state import (
    PatchSamplingConfig,
    build_initial_state,
    sample_circular_patches,
)
from NCA.inverse_design.objectives import CellTypeObjective, GeometryPenalties
from NCA.inverse_design.optimise import OptimisationConfig, optimise_geometry
from NCA.inverse_design.result import OptimisationResult
from NCA.inverse_design.rollout import rollout_final

__all__ = (
    "CellTypeObjective",
    "FourierRadialGeometry",
    "GeometryPenalties",
    "OptimisationConfig",
    "OptimisationResult",
    "PatchSamplingConfig",
    "build_initial_state",
    "coordinate_grid",
    "optimise_geometry",
    "rollout_final",
    "sample_circular_patches",
)
