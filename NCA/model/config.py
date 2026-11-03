"""Typed configuration for NCA model architectures."""

from dataclasses import dataclass, field

from Common.config import ConfigValue


@dataclass(frozen=True)
class KANConfig(ConfigValue):
    basis: str = "rbf"
    hidden_features: int | None = None
    num_basis: int = 8
    grid_min: float = -2.0
    grid_max: float = 2.0
    rbf_width: float | None = None
    trainable_width: bool = True
    extrapolation: str = "constant"
    use_base_branch: bool = True
    base_activation: str = "identity"
    use_layernorm: bool = True
    spline_init_scale: float = 0.1
    base_init_scale: float = 0.1
    final_zero_init: bool = True


@dataclass(frozen=True)
class ModelConfig(ConfigValue):
    family: str
    channels: int
    kernel_str: tuple[str, ...]
    fire_rate: float
    padding: str
    activation: str = "relu"
    # Discrete spatial-kernel radius used to construct a
    # (2 * kernel_scale + 1)-wide convolution kernel.
    kernel_scale: int = 1
    parameter_noise_level: float = 0.01
    # aNCA/gaNCA only: "straight_through" (cells fire at random with their
    # learned probability) or "expected" (deterministic, update scaled by it).
    firing_estimator: str = "straight_through"

    def __post_init__(self):
        # NCA_fast, NCA_sycl and gNCA_sycl are retired, and are kept only so
        # that old bundle configs still load (see PORTABLE_MODEL_FAMILIES).
        supported = {
            "NCA", "gNCA", "nNCA", "gnNCA", "aNCA", "gaNCA", "FastKaNCA", "HNCA",
            "sNCA", "gsNCA",
            "NCA_fast", "NCA_sycl", "gNCA_sycl",
        }
        if self.family not in supported:
            raise ValueError(f"Unsupported model family {self.family!r}")
        if self.channels <= 0 or not 0 <= self.fire_rate <= 1:
            raise ValueError(
                "model channels must be positive and fire_rate must be in [0, 1]"
            )
        if self.firing_estimator not in {"straight_through", "expected"}:
            raise ValueError(
                "model.firing_estimator must be 'straight_through' or 'expected'"
            )


@dataclass(frozen=True)
class KANModelConfig(ModelConfig):
    """Configuration for NCA families whose update network uses KAN layers."""

    kan: KANConfig = field(default_factory=KANConfig)


@dataclass(frozen=True)
class HNCAModelConfig(ModelConfig):
    """Configuration for the two-level hierarchical NCA."""

    scale: int = 4
    obs_channels: int = 4
    parent_learnable_kernels: bool = False
    child_gated: bool = False
    parent_gated: bool = False
    actuator_gated: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.scale < 1:
            raise ValueError("model.scale must be a positive integer")
        if not 0 <= self.obs_channels < self.channels:
            raise ValueError("model.obs_channels must be in [0, model.channels)")


@dataclass(frozen=True)
class SignalModelConfig(ModelConfig):
    """Configuration for NCAs with grown signalling structures (sNCA, gsNCA)."""

    # cable, axon, steady, excitable, broadcast or pointer (see NCA/model/NCA_signalling.py)
    signal_mode: str = "cable"
    # Must be even for excitable (voltage and recovery halves)
    signal_channels: int = 4
    # Propagation sub-steps per NCA step (solver iterations for steady)
    signal_substeps: int = 8
    # Diffusion rate of each cable or excitable sub-step, in (0, 0.25]
    signal_diffusion: float = 0.25
    # steady only: screening rate; smaller values spread signals further
    signal_screening: float = 0.01
    # broadcast: widest kernel; pointer: furthest reach (cells)
    signal_range: int = 16
    # If true, gates are open where their channel is 0, so signals can cross
    # empty tissue and the NCA has to grow insulation instead of wires
    signal_open_at_rest: bool = False

    def __post_init__(self):
        super().__post_init__()
        structure_channels = {
            "cable": 1, "axon": 3, "steady": 1, "excitable": 1, "broadcast": 2, "pointer": 3,
        }
        if self.signal_mode not in structure_channels:
            raise ValueError(f"model.signal_mode must be one of {tuple(structure_channels)}")
        if not 1 <= self.signal_channels < self.channels - structure_channels[self.signal_mode]:
            raise ValueError(
                "model.signal_channels must be at least 1 and leave room for the "
                "structure channels within model.channels"
            )
        if self.signal_mode == "excitable" and self.signal_channels % 2:
            raise ValueError("model.signal_channels must be even for signal_mode 'excitable'")
        if self.signal_screening <= 0 or self.signal_range < 1:
            raise ValueError("model.signal_screening must be positive and model.signal_range at least 1")
        if self.signal_substeps < 1:
            raise ValueError("model.signal_substeps must be a positive integer")
        if not 0 < self.signal_diffusion <= 0.25:
            raise ValueError("model.signal_diffusion must be in (0, 0.25]")
