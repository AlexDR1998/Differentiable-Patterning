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


@dataclass(frozen=True)
class TrainerConfig(ConfigValue):
    boundary_mode: str = "soft"
    grad_loss: bool = False
    sharding: int | None = None
    log_directory: str = "logs/"
    loss_time_channel_mask: Any = None
    loop_autodiff: str = "checkpointed"
    clear_cache_every: int | None = None
    log_every: int = 100
    jax_trace: bool = False
    pool_admission: PoolAdmissionConfig = field(default_factory=PoolAdmissionConfig)
    backend: TrainerBackendConfig = field(default_factory=TrainerBackendConfig)
