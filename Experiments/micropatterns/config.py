"""Configuration owned by the micropattern experiment workflow."""

from dataclasses import dataclass, field
from typing import Mapping

from Common.config import ConfigValue
from Common.dataloader.micropattern_cleaning import MicropatternCleaningConfig
from Common.dataloader.micropattern_schemas import (
    DEFAULT_260726_HISTOGRAM_PERCENTILES,
    DEFAULT_260726_INTENSITY_FACTORS,
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
    # rescaling to [0, 1], and factors for raw intensities keyed by
    # measurement name and timestep (e.g. {"cell_fate_s2/FOXA2": {0: 0.02}}).
    # Unlisted channels and timesteps are unchanged. With cleaning enabled,
    # the images are cleaned, centred and normalised as described in
    # Common/dataloader/micropattern_cleaning.py.
    histogram_percentiles: tuple[float, float] = DEFAULT_260726_HISTOGRAM_PERCENTILES
    intensity_factors: Mapping[str, Mapping[int, float]] = field(
        default_factory=lambda: {
            name: dict(by_hour) for name, by_hour in DEFAULT_260726_INTENSITY_FACTORS.items()
        }
    )
    cleaning: MicropatternCleaningConfig = field(default_factory=MicropatternCleaningConfig)
    # 260726 only: images flagged as low quality in the data cleaning notebook
    # (a file from Common/dataloader/quality_flags.py; a relative path is
    # relative to the repository root). Each is treated as not measured.
    quality_flags_file: str | None = None

    def __post_init__(self):
        low, high = self.histogram_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError(
                "data.micropattern.histogram_percentiles must increase within [0, 100]"
            )
        if any(
            value < 0 for by_hour in self.intensity_factors.values() for value in by_hour.values()
        ):
            raise ValueError("data.micropattern.intensity_factors must be non-negative")


@dataclass(frozen=True)
class KnockoutConfig(ConfigValue):
    mode: str | None = None
    time: int | None = None
    channel: str | None = None
    curriculum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelInitializationConfig(ConfigValue):
    model_id: str | None = None
