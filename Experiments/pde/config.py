"""Configuration owned by the PDE experiment workflow."""

from dataclasses import dataclass, field

from Common.config import ConfigValue
from PDE.catalogue import INITIAL_CONDITIONS, NORMALISATIONS, PDE_MODELS, SOLVERS


@dataclass(frozen=True)
class PdeDataConfig(ConfigValue):
    # A model from PDE/catalogue.py: gray_scott, chhabra, cahn_hilliard, heat,
    # hillen_painter or keller_segel.
    model: str = "gray_scott"
    # Overrides of the model's default parameters, e.g. {alpha: 0.0623, gamma: 0.06268}.
    parameters: dict[str, float] = field(default_factory=dict)
    # PDE channels the NCA must reproduce, by name; empty means all of them.
    observed_channels: tuple[str, ...] = ()
    size: int = 64
    padding: str = "CIRCULAR"
    dx: float = 1.0
    # Solver step. Explicit solvers need dt well below dx^2 / (4 * largest diffusion rate).
    dt: float = 0.5
    solver: str = "heun"
    # The trajectory is sampled at `frames` evenly spaced times in [0, t_end].
    t_end: float = 5000.0
    frames: int = 12
    initial_condition: str = "gray_scott_shapes"
    # uniform and square initial conditions are value * scale + offset.
    initial_condition_scale: float = 1.0
    initial_condition_offset: float = 0.0
    # Number of smoothing passes applied to the initial condition.
    initial_condition_blur: int = 0
    # minmax: rescale all data to [0, 1]; channel_minmax: each channel separately.
    normalisation: str = "minmax"
    # Seed for the initial conditions, separate from the model seed so that every
    # model in a sweep trains on the same trajectories.
    seed: int = 0
    noise_strength: float = 0.001
    reinjection_probability: float = 0.5

    def __post_init__(self):
        if self.model not in PDE_MODELS:
            raise ValueError(f"data.pde.model must be one of {tuple(PDE_MODELS)}")
        spec = PDE_MODELS[self.model]
        unknown = set(self.parameters) - set(spec.parameter_names)
        if unknown:
            raise ValueError(
                f"data.pde.parameters for {self.model} must be drawn from {spec.parameter_names}, "
                f"got {sorted(unknown)}"
            )
        if set(self.observed_channels) - set(spec.channel_names):
            raise ValueError(f"data.pde.observed_channels for {self.model} must be drawn from {spec.channel_names}")
        if self.initial_condition not in INITIAL_CONDITIONS:
            raise ValueError(f"data.pde.initial_condition must be one of {INITIAL_CONDITIONS}")
        if self.normalisation not in NORMALISATIONS:
            raise ValueError(f"data.pde.normalisation must be one of {NORMALISATIONS}")
        if self.solver not in SOLVERS:
            raise ValueError(f"data.pde.solver must be one of {SOLVERS}")
        if self.size <= 0 or self.dx <= 0 or self.dt <= 0 or self.t_end <= 0:
            raise ValueError("data.pde.size, dx, dt and t_end must be positive")
        if self.frames < 2:
            raise ValueError("data.pde.frames must be at least 2")
        if self.initial_condition_blur < 0:
            raise ValueError("data.pde.initial_condition_blur must be >= 0")
        if not 0.0 <= self.reinjection_probability <= 1.0:
            raise ValueError("data.pde.reinjection_probability must be in [0, 1]")

    @property
    def channel_names(self):
        """Names of the channels the NCA is trained on."""
        return self.observed_channels or PDE_MODELS[self.model].channel_names
