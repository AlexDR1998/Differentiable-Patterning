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

    def __post_init__(self):
        supported = {
            "NCA", "NCA_fast", "NCA_sycl", "gNCA", "nNCA", "gnNCA",
            "FastKaNCA",
        }
        if self.family not in supported:
            raise ValueError(f"Unsupported model family {self.family!r}")
        if self.channels <= 0 or not 0 <= self.fire_rate <= 1:
            raise ValueError(
                "model channels must be positive and fire_rate must be in [0, 1]"
            )


@dataclass(frozen=True)
class KANModelConfig(ModelConfig):
    """Configuration for NCA families whose update network uses KAN layers."""

    kan: KANConfig = field(default_factory=KANConfig)
