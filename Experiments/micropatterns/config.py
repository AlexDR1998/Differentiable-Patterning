"""Configuration owned by the micropattern experiment workflow."""

from dataclasses import dataclass, field
from typing import Mapping

from Common.config import ConfigValue
from Common.dataloader.micropattern_schemas import (
    DEFAULT_260726_HISTOGRAM_PERCENTILES,
    DEFAULT_260726_INITIAL_INTENSITY_SCALES,
)


@dataclass(frozen=True)
class MicropatternDataConfig(ConfigValue):
    task: str | None = None
    pool_copies: int = 1
    train_replicates: tuple[int, ...] | None = None
    validation_replicates: tuple[int, ...] | None = None
    experiment_groups: tuple[str, ...] | None = None
    timesteps: tuple[int, ...] = (0, 12, 24, 36, 48)
    data_channels: int | None = 12
    pad_multiple: int | None = None
    noise_strength: float = 0.005
    intermediate_reinjection_probability: float = 0.5
    intermediate_reinjection_probability_end: float = 0.5
    intermediate_reinjection_decay_start_fraction: float = 0.25
    duplicate_final_timestep: bool = False
    # 260726 only: low/high percentiles each channel is clipped to before
    # rescaling to [0, 1], and factors for raw 0h intensities keyed by
    # measurement name (e.g. "cell_fate_s2/FOXA2"). Unlisted channels are
    # unchanged. See Common/dataloader/micropattern_260726.py.
    histogram_percentiles: tuple[float, float] = DEFAULT_260726_HISTOGRAM_PERCENTILES
    initial_intensity_scales: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_260726_INITIAL_INTENSITY_SCALES)
    )

    def __post_init__(self):
        low, high = self.histogram_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError(
                "data.micropattern.histogram_percentiles must increase within [0, 100]"
            )
        if any(value < 0 for value in self.initial_intensity_scales.values()):
            raise ValueError(
                "data.micropattern.initial_intensity_scales must be non-negative"
            )


@dataclass(frozen=True)
class KnockoutConfig(ConfigValue):
    mode: str | None = None
    time: int | None = None
    channel: str | None = None
    curriculum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelInitializationConfig(ConfigValue):
    model_id: str | None = None
