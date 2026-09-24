"""Configuration owned by the snowmelt experiment workflow."""

from dataclasses import dataclass

from Common.config import ConfigValue
from Common.dataloader.snowmelt import DYNAMIC_CHANNELS, STATIC_CHANNELS


@dataclass(frozen=True)
class SnowmeltDataConfig(ConfigValue):
    # Dataset root; null falls back to $SNOWMELT_DATA_ROOT, then $DATA_PATH_BASE/snowmelt.
    root: str | None = None
    # Dynamic channels the NCA must reproduce, in state-channel order.
    target_channels: tuple[str, ...] = ("SCA",)
    # Fixed per-pixel inputs written into the final state channels every step,
    # after the catchment mask.
    static_channels: tuple[str, ...] = STATIC_CHANNELS
    # Zero border (pixels) so circular padding cannot couple opposite edges.
    pad: int = 4
    # Fraction of a downsampled block that must lie in the catchment.
    mask_threshold: float = 0.5
    noise_strength: float = 0.005
    reinjection_probability: float = 0.5

    def __post_init__(self):
        unknown = set(self.target_channels) - set(DYNAMIC_CHANNELS)
        if not self.target_channels or unknown:
            raise ValueError(
                f"data.snowmelt.target_channels must be non-empty and drawn from {DYNAMIC_CHANNELS}"
            )
        if set(self.static_channels) - set(STATIC_CHANNELS):
            raise ValueError(f"data.snowmelt.static_channels must be drawn from {STATIC_CHANNELS}")
        if self.pad < 0 or not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("data.snowmelt.pad must be >= 0 and mask_threshold in (0, 1]")
        if not 0.0 <= self.reinjection_probability <= 1.0:
            raise ValueError("data.snowmelt.reinjection_probability must be in [0, 1]")
