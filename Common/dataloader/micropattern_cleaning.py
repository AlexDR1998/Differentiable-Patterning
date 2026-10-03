"""Settings for cleaning the raw 260726 micropattern images before training.

``data.micropattern.cleaning`` in the experiment config. With ``enabled``,
``load_micropattern_260726`` processes every image in this order, as tuned in
``Experiments/micropatterns/data_cleaning.py``:

1. intensity factors (``data.micropattern.intensity_factors``);
2. hot pixel replacement (``hot_pixel_thresholds``, ``hot_pixel_window``),
   see ``Common/dataloader/hot_pixels.py``;
3. rolling-ball background removal (``background_radii``,
   ``background_shrink``), see ``Common/dataloader/background.py``;
4. block-averaging to the normalisation resolution (downsampling 4, the
   notebook's, when the training downsampling is a multiple of 4; otherwise
   the training downsampling itself);
5. centring with the colony centres in ``alignment_file``, see
   ``Common/dataloader/alignment.py``, and a circular mask of the pattern
   radius times ``mask_radius_scale``;
6. linear rescaling of each channel to [0, 1] between a low and a high
   percentile (``data.micropattern.histogram_percentiles``, or
   ``channel_percentiles`` per channel), taken over all timesteps of a
   trajectory, shared between trajectories as set by ``normalisation``, see
   ``Common/dataloader/normalisation.py``;
7. zeroing outside the mask and block-averaging to the training resolution.

Without ``enabled`` the loader keeps its earlier behaviour, which configs
saved before schema version 6 rely on.
"""

from dataclasses import dataclass, field
from typing import Mapping

from Common.config import ConfigValue
from Common.dataloader.micropattern_schemas import (
    DEFAULT_260726_BACKGROUND_RADII,
    DEFAULT_260726_HOT_PIXEL_THRESHOLDS,
    MICROPATTERN_260726_SCHEMA,
)
from Common.dataloader.normalisation import NORMALISATION_MODES


@dataclass(frozen=True)
class MicropatternCleaningConfig(ConfigValue):
    enabled: bool = False
    # Hot pixels: threshold in noise levels above the local median per
    # channel (0 = off), and the odd window size in full-resolution pixels.
    hot_pixel_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_260726_HOT_PIXEL_THRESHOLDS)
    )
    hot_pixel_window: int = 5
    # Rolling-ball radius per channel in full-resolution pixels (0 = off),
    # and the shrink factor used to estimate the background quickly.
    background_radii: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_260726_BACKGROUND_RADII)
    )
    background_shrink: int = 4
    # Alignment file exported from the data cleaning notebook. A relative
    # path is resolved against the repository root.
    alignment_file: str | None = None
    mask_radius_scale: float = 1.0
    # One of NORMALISATION_MODES: "replicate_mean", "per_replicate", "pooled".
    normalisation: str = "replicate_mean"
    # Percentile overrides per channel, e.g. {"cell_fate_s1/SOX2": (1.0, 99.0)}.
    channel_percentiles: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    percentiles_inside_mask: bool = True
    # Knockout trajectories take their bounds from the control replicates.
    knockouts_use_control_bounds: bool = False

    def __post_init__(self):
        names = set(MICROPATTERN_260726_SCHEMA.measurement_names)
        for label, mapping in (
            ("hot_pixel_thresholds", self.hot_pixel_thresholds),
            ("background_radii", self.background_radii),
            ("channel_percentiles", self.channel_percentiles),
        ):
            unknown = sorted(set(mapping) - names)
            if unknown:
                raise ValueError(
                    f"Unknown channels in data.micropattern.cleaning.{label}: "
                    + ", ".join(unknown)
                )
        if any(value < 0 for value in self.hot_pixel_thresholds.values()):
            raise ValueError("cleaning.hot_pixel_thresholds must be non-negative")
        if any(value < 0 for value in self.background_radii.values()):
            raise ValueError("cleaning.background_radii must be non-negative")
        if self.hot_pixel_window < 3 or self.hot_pixel_window % 2 == 0:
            raise ValueError("cleaning.hot_pixel_window must be an odd integer of at least 3")
        if self.background_shrink < 1:
            raise ValueError("cleaning.background_shrink must be a positive integer")
        if self.mask_radius_scale <= 0:
            raise ValueError("cleaning.mask_radius_scale must be positive")
        if self.normalisation not in NORMALISATION_MODES:
            raise ValueError(
                f"cleaning.normalisation must be one of {tuple(NORMALISATION_MODES)}"
            )
        for name, (low, high) in self.channel_percentiles.items():
            if not 0 <= low < high <= 100:
                raise ValueError(
                    f"cleaning.channel_percentiles[{name!r}] must increase within [0, 100]"
                )
        if self.enabled and not self.alignment_file:
            raise ValueError(
                "cleaning.alignment_file is required when cleaning is enabled; export "
                "it from Experiments/micropatterns/data_cleaning.py"
            )

    @property
    def shares_bounds(self):
        """Whether all trajectories of a channel use the same clipping bounds."""
        return self.normalisation != "per_replicate"
