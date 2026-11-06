"""Configuration owned by the micropattern geometry-optimisation workflow."""

from dataclasses import dataclass, field

from Common.config import ConfigValue


@dataclass(frozen=True)
class InverseDesignGeometryConfig(ConfigValue):
    # Fourier-radial shape: r(theta) = R exp(sum_k a_k cos k theta + b_k sin k theta)
    harmonics: int = 3
    # Each |a_k|, |b_k| stays below this
    coefficient_limit: float = 0.3
    # Std of the random initial (unconstrained) coefficients; 0 starts from the circle
    initial_scale: float = 0.05
    # Keep R, and so the colony area, fixed
    fixed_radius: bool = True
    # R as a multiple of the training colonies' radius
    radius_scale: float = 1.0
    # Grid side as a multiple of the training grid's. Training colonies
    # nearly fill their grid, so a larger one leaves room for new shapes.
    grid_scale: float = 1.5
    # Sweep over this to repeat a config from other initial shapes (with seed_mode: index)
    repeat: int = 0
    # false keeps the initial shape (a density-only control)
    optimise: bool = True


@dataclass(frozen=True)
class InverseDesignObjectiveConfig(ConfigValue):
    # A cell type (or Several / Other) of the cell type file
    target: str = "Notochord"
    # Width of the sigmoid that replaces each marker threshold
    temperature: float = 0.03
    # false minimises the target share (a control)
    maximise: bool = True
    # Weight of the target's amount (share x area fraction) added to its share
    amount_weight: float = 0.0


@dataclass(frozen=True)
class InverseDesignPenaltyConfig(ConfigValue):
    perimeter_weight: float = 0.0
    coefficient_weight: float = 0.0
    # Mask in the outermost two grid pixels (it would wrap round)
    border_weight: float = 1.0


@dataclass(frozen=True)
class MaskSoftnessConfig(ConfigValue):
    # Edge width of the optimised mask, as a fraction of the boundary radius
    # (0.02 is about one pixel for a 50-pixel radius). It shrinks
    # geometrically from start to end over the run.
    start: float = 0.08
    end: float = 0.02


@dataclass(frozen=True)
class InverseDesignOptimiserConfig(ConfigValue):
    learn_rate: float = 0.01
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class InverseDesignRolloutConfig(ConfigValue):
    # Developmental time scored; one model run.t is 12 h
    hours: int = 48
    # lax or checkpointed (less memory, slower backward pass)
    scan_kind: str = "checkpointed"
    checkpoints: int | None = None


@dataclass(frozen=True)
class TextureConfig(ConfigValue):
    """Synthetic 0 h textures, as in ``boundary_mask_explorer.py``.

    With ``file`` set, every setting except ``bank_size`` comes from that
    file's section for the model's projection
    (``Experiments/micropatterns/texture_settings.py``), and the other fields
    here must keep their defaults.
    """

    # Number of textures; each batch draws from them
    bank_size: int = 16
    # Texture settings file (relative to the repository root); null: the fields below
    file: str | None = None
    # patch (multichannel patch resampling) or iaaft
    mode: str = "patch"
    # Patch sizes in pixels at patch_downsample, rescaled to the model's
    # data.downsample; null: in the model's pixels
    patch_size: int = 11
    patch_overlap: int = 7
    patch_jitter: int = 3
    patch_downsample: int | None = None
    interior_fraction: float = 0.8
    candidates: int = 24
    iaaft_iterations: int = 25
    # Match the reference histograms and panel covariances afterwards
    normalise: bool = False
    normalise_iterations: int = 4

    def __post_init__(self):
        if self.mode not in ("patch", "iaaft"):
            raise ValueError(f"Unknown texture mode {self.mode!r}; use patch or iaaft")
        if self.file is not None:
            defaults = TextureConfig()
            changed = [
                name for name in self.__dataclass_fields__
                if name not in ("bank_size", "file") and getattr(self, name) != getattr(defaults, name)
            ]
            if changed:
                raise ValueError(
                    f"inverse_design.texture.file is set, so texture.{changed} would be ignored; "
                    "change them in the file, or set texture.file: null"
                )


@dataclass(frozen=True)
class InverseDesignDensityConfig(ConfigValue):
    """Cell density: the share of the colony covered by cells (NCA/inverse_design/density.py)."""

    # none: textures keep the reference colonies' own gaps (as before);
    # fixed: whole cells kept for the density `initial`; optimise: learn the density
    mode: str = "none"
    # Starting density in (0, 1); null: the mean of the training colonies
    initial: float | None = None
    # Generic cell markers that tell cells from gaps
    markers: tuple[str, ...] = ("LMBR", "SOX2")
    # Covered pixels are split into cells at marker maxima at least this many
    # pixels apart (at the model's resolution); 0: connected patches are not split
    split_distance: int = 2
    # Smaller cells (pixels) are dropped
    min_cell_area: int = 2
    # Cells are only taken from within this fraction of the colony radius
    interior_fraction: float = 0.8
    # Layers of non-overlapping cells. One layer can only cover part of the
    # grid; later layers fill the space between cells, partly under them.
    layers: int = 2
    # Random places tried per cell that would fit in the grid
    attempts: int = 10
    # Width of the cut in cell rank (a share of a layer's cells): the cells
    # at the cut are faded, which gives the density a gradient
    softness: float = 0.02
    # Circles also scored at these multiples of the starting density
    baselines: tuple[float, ...] = (0.5, 0.75, 1.25)

    def __post_init__(self):
        if self.mode not in ("none", "fixed", "optimise"):
            raise ValueError(f"Unknown density mode {self.mode!r}; use none, fixed or optimise")
        if self.initial is not None and not 0.0 < self.initial < 1.0:
            raise ValueError("density.initial must be between 0 and 1")
        if self.split_distance < 0 or self.min_cell_area < 1:
            raise ValueError("density.split_distance must be nonnegative and min_cell_area positive")
        if not 0.0 < self.interior_fraction <= 1.0:
            raise ValueError("density.interior_fraction must be in (0, 1]")
        if self.layers < 1 or self.attempts < 1:
            raise ValueError("density.layers and density.attempts must be positive")
        if self.softness <= 0:
            raise ValueError("density.softness must be positive")


@dataclass(frozen=True)
class InverseDesignConfig(ConfigValue):
    # Cell type file (relative to the repository root); its projection must
    # match the model's training data
    cell_types_file: str
    iterations: int = 400
    batch_size: int = 4
    # Binary-mask evaluation and image logging every log_every iterations
    log_every: int = 25
    evaluation_seeds: int = 16
    # Rollouts of the best geometry shown in the final W&B video (0: no video)
    video_seeds: int = 2
    # Fixed-area ellipses scored before optimising: harmonic-2 amplitudes,
    # each at 0 and 45 degrees (the circle is always scored)
    baseline_ellipses: tuple[float, ...] = (0.15, 0.3)
    geometry: InverseDesignGeometryConfig = field(default_factory=InverseDesignGeometryConfig)
    objective: InverseDesignObjectiveConfig = field(default_factory=InverseDesignObjectiveConfig)
    penalties: InverseDesignPenaltyConfig = field(default_factory=InverseDesignPenaltyConfig)
    mask_softness: MaskSoftnessConfig = field(default_factory=MaskSoftnessConfig)
    optimiser: InverseDesignOptimiserConfig = field(default_factory=InverseDesignOptimiserConfig)
    rollout: InverseDesignRolloutConfig = field(default_factory=InverseDesignRolloutConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    density: InverseDesignDensityConfig = field(default_factory=InverseDesignDensityConfig)
