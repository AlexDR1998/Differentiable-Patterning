"""Pattern-forming PDEs that can be simulated as NCA training data.

This module does not import JAX, so typed configs can validate against it.
Default parameter values live in each model's constructor.
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PdeSpec:
    module: str
    channel_names: tuple[str, ...]
    parameter_names: tuple[str, ...]


PDE_MODELS = {
    "gray_scott": PdeSpec(
        "PDE.model.fixed_models.update_gray_scott",
        ("A", "B"),
        ("DA", "DB", "alpha", "gamma"),
    ),
    "chhabra": PdeSpec(
        "PDE.model.fixed_models.update_chhabra",
        ("activator", "inhibitor"),
        ("KI", "KdA", "KdI", "SA", "SI", "DA", "DI"),
    ),
    "cahn_hilliard": PdeSpec(
        "PDE.model.fixed_models.update_cahn_hilliard",
        ("phase",),
        ("gamma", "D"),
    ),
    "heat": PdeSpec(
        "PDE.model.fixed_models.update_heat_equation",
        ("heat",),
        ("D",),
    ),
    "hillen_painter": PdeSpec(
        "PDE.model.fixed_models.update_hillen_painter",
        ("cells", "signal"),
        ("logistic_growth_rate", "gamma", "alpha", "chi", "phi", "D"),
    ),
    "keller_segel": PdeSpec(
        "PDE.model.fixed_models.update_keller_segel",
        ("cells", "signal"),
        ("c", "alpha", "D", "epsilon"),
    ),
}

INITIAL_CONDITIONS = ("uniform", "square", "gray_scott_shapes", "gray_scott_spots")
NORMALISATIONS = ("minmax", "channel_minmax", "none")
SOLVERS = ("heun", "euler", "tsit5", "kencarp3", "dopri5", "kvaerno3", "dopri8")


def build_pde(name, padding, dx, parameters=None, kernel_scale=1):
    """Return the right-hand side F(t, X, args) of the named PDE."""
    spec = PDE_MODELS[name]
    unknown = set(parameters or {}) - set(spec.parameter_names)
    if unknown:
        raise ValueError(f"Unknown parameters for PDE {name!r}: {sorted(unknown)}")
    F = import_module(spec.module).F
    return F(PADDING=padding, dx=dx, KERNEL_SCALE=kernel_scale, **dict(parameters or {}))
