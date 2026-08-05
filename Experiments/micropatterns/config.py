"""Configuration owned by the micropattern experiment workflow."""

from dataclasses import dataclass

from Common.config import ConfigValue


@dataclass(frozen=True)
class MicropatternDataConfig(ConfigValue):
    task: str | None = None
    pool_copies: int = 1
    experiment_groups: tuple[str, ...] | None = None
    timesteps: tuple[int, ...] = (0, 12, 24, 36, 48)
    data_channels: int | None = 12
    pad_multiple: int | None = None
    noise_strength: float = 0.005
    intermediate_reinjection_probability: float = 0.5
    intermediate_reinjection_probability_end: float = 0.5
    intermediate_reinjection_decay_start_fraction: float = 0.25
    duplicate_final_timestep: bool = False


@dataclass(frozen=True)
class KnockoutConfig(ConfigValue):
    mode: str | None = None
    time: int | None = None
    channel: str | None = None
