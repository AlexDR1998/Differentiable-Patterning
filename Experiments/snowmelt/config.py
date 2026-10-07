"""Configuration owned by the snowmelt experiment workflow."""

from dataclasses import dataclass
from datetime import date

from Common.config import ConfigValue
from Common.dataloader.snowmelt import DEFAULT_VERSION, DYNAMIC_CHANNELS, STATIC_CHANNELS_BY_VERSION


@dataclass(frozen=True)
class SnowmeltDataConfig(ConfigValue):
    # Folder holding the dataset versions; null falls back to $SNOWMELT_DATA_ROOT,
    # then $DATA_PATH_BASE/snowmelt. The data is read from <root>/<version>.
    root: str | None = None
    # Dataset version folder: v1 (5 dates, May-July) or v2 (10 dates, April-September).
    version: str = DEFAULT_VERSION
    # Dynamic channels the NCA must reproduce, in state-channel order.
    target_channels: tuple[str, ...] = ("SCA",)
    # Fixed per-pixel inputs written into the final state channels every step,
    # after the catchment mask. v1: DEM, INCIDENCE. v2: DEM, SLOPE, NORTHNESS, EASTNESS.
    static_channels: tuple[str, ...] = ("DEM",)
    # Zero border (pixels) so circular padding cannot couple opposite edges.
    pad: int = 4
    # Fraction of a downsampled block that must lie in the catchment.
    mask_threshold: float = 0.5
    # Acquisitions never used (e.g. cloud or brightness artefacts), as indices into
    # the dataset's dates (negative counts from the end) or ISO dates.
    exclude_dates: tuple[int | str, ...] = ()
    # Acquisitions left out of training but scored at evaluation, given the same way.
    # Needs run.interval_mode=steps and an explicit run.reference_interval.
    hold_out_dates: tuple[int | str, ...] = ()
    noise_strength: float = 0.005
    reinjection_probability: float = 0.5

    def __post_init__(self):
        unknown = set(self.target_channels) - set(DYNAMIC_CHANNELS)
        if not self.target_channels or unknown:
            raise ValueError(
                f"data.snowmelt.target_channels must be non-empty and drawn from {DYNAMIC_CHANNELS}"
            )
        if self.version not in STATIC_CHANNELS_BY_VERSION:
            raise ValueError(f"data.snowmelt.version must be one of {tuple(STATIC_CHANNELS_BY_VERSION)}")
        allowed = STATIC_CHANNELS_BY_VERSION[self.version]
        if set(self.static_channels) - set(allowed):
            raise ValueError(
                f"data.snowmelt.static_channels for dataset {self.version} must be drawn from {allowed}"
            )
        if self.pad < 0 or not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("data.snowmelt.pad must be >= 0 and mask_threshold in (0, 1]")
        for name in ("exclude_dates", "hold_out_dates"):
            for entry in getattr(self, name):
                if isinstance(entry, str):
                    date.fromisoformat(entry)  # raises on anything but an ISO date
                elif isinstance(entry, bool) or not isinstance(entry, int):
                    raise ValueError(f"data.snowmelt.{name} entries must be indices or ISO dates, got {entry!r}")
        if not 0.0 <= self.reinjection_probability <= 1.0:
            raise ValueError("data.snowmelt.reinjection_probability must be in [0, 1]")
