"""Configuration owned by the impulse-optimisation workflow."""

from dataclasses import dataclass

from Common.config import ConfigValue
from Common.trainer.config import LossConfig


@dataclass(frozen=True)
class CheckpointLoadConfig(ConfigValue):
    """Trained model to perturb, looked up by ID (or alias) in the model registry."""

    model_id: str
    # Defaults to MODEL_STORE_ROOT, then ./models, as for the registry CLI.
    store_root: str | None = None


@dataclass(frozen=True)
class ImpulsePairSourceConfig(ConfigValue):
    type: str
    # Indices into data.emoji.pairs: the run switches the source pattern to
    # the target pattern. Every pair source only uses these two patterns.
    source_index: int = 0
    target_index: int = 1
    stabilisation_steps: tuple[int, ...] = (128, 256)
    target_steps: int = 192
    # Time indices into the source trajectory, for trajectory_state
    initial_time: int = 0
    target_time: int = -1


@dataclass(frozen=True)
class ImpulseRolloutConfig(ConfigValue):
    steps: int = 64
    # Number of final rollout states the target loss is averaged over.
    loss_window: int = 1
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
class ImpulseOptimiserConfig(ConfigValue):
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
    optimiser: ImpulseOptimiserConfig
    output: OutputConfig
