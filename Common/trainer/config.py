"""Configuration shared by trainer implementations across domains."""

from dataclasses import dataclass, field
from typing import Mapping

from Common.config import ConfigValue


@dataclass(frozen=True)
class ScheduleConfig(ConfigValue):
    type: str = "exponential"
    warmup_init_lr: float = 1e-6
    final_factor: float = 0.1
    transition_fraction: float = 0.75
    decay_rate: float | None = None


@dataclass(frozen=True)
class OptimizerConfig(ConfigValue):
    type: str = "nadam"
    learn_rate: float = 1e-3
    warmup_steps: int = 64
    decay_rate: float = 0.99
    blocknorm: bool = True
    sam: bool = False
    sam_rho: float = 0.05
    sam_sync_period: int = 2
    gradient_clip_norm: float | None = None
    apply_if_finite: bool = False
    max_consecutive_errors: int = 8
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    def __post_init__(self):
        if self.learn_rate <= 0:
            raise ValueError("training.optimizer.learn_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("training.optimizer.warmup_steps cannot be negative")


@dataclass(frozen=True)
class LossTermConfig(ConfigValue):
    """One independently configured component of the training objective."""

    type: str = "l2"
    weight: float = 1.0
    channels: tuple[int, ...] | str | None = None
    experiment_groups: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("loss term weights cannot be negative")


@dataclass(frozen=True)
class PointwiseLossConfig(LossTermConfig):
    """Parameter-free pointwise, spectral, and distribution losses."""


@dataclass(frozen=True)
class GroupedPointwiseLossConfig(LossTermConfig):
    channel_importance: tuple[float, ...] | None = None


@dataclass(frozen=True)
class VggLossConfig(LossTermConfig):
    metric: str = "l2"
    random_crop: bool = False
    random_channel_shuffle: bool = False
    channel_importance: tuple[float, ...] | None = None
    samples: int = 128
    epsilon: float = 0.1
    normalize: bool | None = None
    tau: float | None = None


@dataclass(frozen=True)
class OttLossConfig(LossTermConfig):
    S: int = 1024
    K: int = 5
    D: int = 3
    sharpen: bool = True
    epsilon: float = 0.1
    metric: str = "l2"


@dataclass(frozen=True)
class WassersteinLossConfig(LossTermConfig):
    samples: int = 128
    epsilon: float = 0.1
    metric: str = "l2"
    normalize: bool | None = None
    tau: float | None = None
    amplitude_penalty: float | None = None


@dataclass(frozen=True)
class SummaryLossConfig(LossTermConfig):
    radial_bins: int = 16
    channel_importance: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MultiTargetLossConfig(LossTermConfig):
    multi_target_weights: Mapping[str, float] | None = None
    assignment: str = "hard"
    assignment_tau: float = 0.05
    radial_bins: int = 16
    texture_size: int = 128
    metric: str = "l2"
    random_crop: bool = False
    random_channel_shuffle: bool = False


@dataclass(frozen=True)
class LossConfig(ConfigValue):
    terms: tuple[LossTermConfig, ...] = field(
        default_factory=lambda: (PointwiseLossConfig(),)
    )
    regularisers: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.terms:
            raise ValueError("training.loss.terms must contain at least one term")
        if not any(term.weight > 0 for term in self.terms):
            raise ValueError("training.loss.terms must contain a positive weight")
