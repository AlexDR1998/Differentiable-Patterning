"""Configuration owned by the impulse-optimisation workflow."""

from dataclasses import dataclass

from Common.config import ConfigValue
from Common.trainer.config import LossConfig


@dataclass(frozen=True)
class CheckpointLoadConfig(ConfigValue):
    path: str
    base_directory: str | None = None
    base_env: str | None = "MODEL_SAVE_PATH"


@dataclass(frozen=True)
class ImpulsePairSourceConfig(ConfigValue):
    type: str
    source_index: int = 0
    target_index: int = 1
    stabilisation_steps: tuple[int, ...] = (128, 256)
    target_steps: int = 192
    initial_index: int = 0


@dataclass(frozen=True)
class ImpulseRolloutConfig(ConfigValue):
    steps: int = 64
    evaluation_steps: int = 256
    scan_kind: str = "lax"


@dataclass(frozen=True)
class ImpulseInterventionConfig(ConfigValue):
    channels: str = "hidden"
    spatial: str = "local"
    width: float = 0.2


@dataclass(frozen=True)
class ImpulseObjectiveConfig(ConfigValue):
    type: str = "targeted"
    target_weight: float = 1.0
    tolerance: float = 0.01
    constraint_weight: float = 100.0
    magnitude: str = "l2"
    reward_weight: float = 1.0


@dataclass(frozen=True)
class ImpulseOptimizerConfig(ConfigValue):
    type: str = "adam"
    learn_rate: float = 1e-3
    gradient_clip_norm: float | None = 1.0


@dataclass(frozen=True)
class OutputConfig(ConfigValue):
    directory: str = "impulse_runs"
    base_env: str | None = "IMPULSE_OUTPUT_PATH"


@dataclass(frozen=True)
class ImpulseConfig(ConfigValue):
    iterations: int
    batch_size: int
    resample_every: int
    log_every: int
    pair_source: ImpulsePairSourceConfig
    rollout: ImpulseRolloutConfig
    intervention: ImpulseInterventionConfig
    objective: ImpulseObjectiveConfig
    loss: LossConfig
    optimiser: ImpulseOptimizerConfig
    output: OutputConfig
