"""Configuration owned by the emoji experiment workflow."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from Common.config import ConfigValue


@dataclass(frozen=True)
class ProbabilityScheduleConfig(ConfigValue):
    enabled: bool = False
    start_iteration: int = 0
    schedule_iterations: int = 0
    initial_probability: float = 0.0
    final_probability: float = 0.0

    def __post_init__(self):
        if not 0 <= self.initial_probability <= 1 or not 0 <= self.final_probability <= 1:
            raise ValueError("augmentation probabilities must be in [0, 1]")


@dataclass(frozen=True)
class EmojiPairConfig(ConfigValue):
    initial: str | Mapping[str, Any]
    target: str


@dataclass(frozen=True)
class EmojiDataConfig(ConfigValue):
    task: str = "sequence"
    sequence: tuple[str, ...] = ()
    pairs: tuple[EmojiPairConfig, ...] = ()
    target_repeats: int = 2
    timesteps: tuple[int, ...] | None = None
    observed_channels: int = 4
    data_channels: int = 4
    crop_square: bool = False
    pad: tuple[int, int, int, int] = (10, 10, 10, 10)
    shift_amount: int = 2
    noise_strength: float = 0.005
    noise_mode: str = "full"
    regenerate: bool = True
    terminal_carry: ProbabilityScheduleConfig = field(default_factory=ProbabilityScheduleConfig)
    regeneration: ProbabilityScheduleConfig = field(default_factory=ProbabilityScheduleConfig)
