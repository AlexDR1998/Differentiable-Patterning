"""Configuration and validation for image preprocessing pipelines."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ProcessingStep(str, Enum):
    """Supported legacy micropattern preprocessing operations."""

    HISTOGRAM_EQUALISE = "hist_eq"
    REMOVE_BACKGROUND = "remove_background"
    THRESHOLD = "threshold"
    BATCH_AVERAGE = "batch_average"
    STANDARDISE = "mean_0_std_1"
    SCALE_ZERO_ONE = "map_to_0_1"
    PAD_TO_FULL_WIDTH = "pad_to_full_width"
    DOWNSAMPLE = "downsample"
    ALIGN_CENTRE_OF_MASS = "align"


@dataclass(frozen=True)
class PreprocessingConfig:
    """An ordered, reproducible preprocessing specification."""

    steps: tuple[ProcessingStep, ...] = ()
    downsample: int = 1
    histogram_percentiles: tuple[float, float] = (5.0, 95.0)
    histogram_bins: object | None = None
    background_radius: int = 50
    batch_average: bool = False

    def __post_init__(self):
        if self.downsample <= 0:
            raise ValueError("downsample must be positive")
        low, high = self.histogram_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("histogram_percentiles must increase within [0, 100]")
        if self.background_radius <= 0:
            raise ValueError("background_radius must be positive")

    @classmethod
    def from_legacy(
        cls,
        steps: Iterable[str | ProcessingStep],
        **kwargs,
    ):
        if isinstance(steps, (set, frozenset)):
            raise TypeError(
                "preprocessing steps must be ordered; use a list or tuple, not a set"
            )
        return cls(steps=tuple(ProcessingStep(step) for step in steps), **kwargs)
