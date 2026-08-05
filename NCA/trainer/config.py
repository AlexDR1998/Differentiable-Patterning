"""Typed configuration for NCA training behavior and backends."""

from dataclasses import dataclass, field
from typing import Any

from Common.config import ConfigValue


@dataclass(frozen=True)
class PoolAdmissionConfig(ConfigValue):
    enabled: bool = True
    relative_threshold: float = 1.25
    previous_relative_threshold: float = 1.10
    absolute_threshold: float | None = None
    ema_decay: float = 0.95
    warmup: int | None = None

    def __post_init__(self):
        if self.relative_threshold <= 0 or self.previous_relative_threshold <= 0:
            raise ValueError("pool admission thresholds must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("pool_admission.ema_decay must be in [0, 1)")
        if self.warmup is not None and self.warmup < 0:
            raise ValueError("pool_admission.warmup cannot be negative")


@dataclass(frozen=True)
class TrainerBackendConfig(ConfigValue):
    type: str = "none"


@dataclass(frozen=True)
class NvidiaTrainerBackendConfig(TrainerBackendConfig):
    type: str = "nvidia"


@dataclass(frozen=True)
class SyclTrainerBackendConfig(TrainerBackendConfig):
    type: str = "sycl"
    fused_steps: int = 2
    synchronize_custom_calls: bool = False
    strict_stage_synchronization: bool = False
    regulariser_reduction: str = "atomic"
    pmean_loss: bool = True
    pmean_regularisers: bool = True
    serialize_custom_calls: bool = False
    serialize_onemkl: bool = False
    serialize_backward_custom_calls: bool = False

    def __post_init__(self):
        if self.fused_steps < 1:
            raise ValueError("trainer.backend.fused_steps must be positive")
        if self.regulariser_reduction not in {"atomic", "two_stage"}:
            raise ValueError(
                "trainer.backend.regulariser_reduction must be 'atomic' or 'two_stage'"
            )


@dataclass(frozen=True)
class TrainerConfig(ConfigValue):
    boundary_mode: str = "soft"
    grad_loss: bool = False
    sharding: int | None = None
    log_directory: str = "logs/"
    loss_time_channel_mask: Any = None
    loop_autodiff: str = "checkpointed"
    log_every: int = 100
    jax_trace: bool = False
    pool_admission: PoolAdmissionConfig = field(default_factory=PoolAdmissionConfig)
    backend: TrainerBackendConfig = field(default_factory=TrainerBackendConfig)

    def __post_init__(self):
        if self.boundary_mode not in {"soft", "hard"}:
            raise ValueError("trainer.boundary_mode must be 'soft' or 'hard'")
        if self.loop_autodiff not in {"checkpointed", "lax"}:
            raise ValueError("trainer.loop_autodiff must be 'checkpointed' or 'lax'")
        if self.sharding is not None and self.sharding < 1:
            raise ValueError("trainer.sharding must be positive or None")
        if self.log_every < 1:
            raise ValueError("trainer.log_every must be positive")
