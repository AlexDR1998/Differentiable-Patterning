"""Nivolet (Gran Paradiso) Sentinel-2 snowmelt rasters as NCA target sequences.

The dataset comes in versions (``<data>/snowmelt/v1``, ``v2``, ...); the root is
one version folder. Layout of v1::

    S2_rawbands/DoraNivolet_<band>_<date>.tif      12 L2A bands x 5 dates (LZW, float64)
    S2_derived_indexes/{NDSI,NDVI}_Nivolet_<date>.tif
    S2_derived_indexes/SCA_Nivolet_NDSIgt04_<date>.tif   binary snow cover (NDSI > 0.4)
    S2_topographic_attributes/{DEM,INCIDENCEANGLE}_10mTinitaly_NivoletMask.tif

v2 adds ``_v1.0`` to the folder names, has 10 dates (April to September 2018),
a gap-filled DEM, slope and aspect maps, and replaces the single incidence-angle
map with an hourly one for every day of the season (so INCIDENCE is a static
channel in v1 only; v2 offers SLOPE, NORTHNESS and EASTNESS instead)::

    S2_topographic_attributes_v1.0/{DEM,SLOPE,ASPECT}_Nivolet_10m_filled/<name>_10mDTMUnicoTinitaly_NivoletMask.tif
    S2_topographic_attributes_v1.0/INCIDENCEANGLE_Nivolet_10m_filled/INCIDENCEANGLE_DOY<ddd>_H<hh>.tif

All rasters share one 10 m grid in ED50 / UTM 32N. The catchment is where the
DEM has a value (v1 stores 0 outside it, v2 float32-min); each layer
additionally carries sparse no-data of its own.

:func:`load_snowmelt` returns the raw rasters for exploration (NaN outside the
catchment and at no-data pixels). :func:`build_snowmelt_sequence` turns them
into a training sequence: selected dynamic channels scaled to ``[0, 1]``,
missing values filled, block-downsampled, zero-padded, plus fixed per-pixel
boundary channels (catchment mask and static terrain covariates) and the
acquisition times in days.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import logging
import os
from pathlib import Path
import re
import warnings

import numpy as np
import scipy.ndimage as ndi
import tifffile

from Common.dataloader.tiff_lzw import read_tiff

# Sentinel-2 MSI: central wavelength (nm), native resolution (m), name.
# B10 (cirrus) is absent, as in L2A products.
BAND_INFO = {
    "B1": (443, 60, "Coastal aerosol"),
    "B2": (490, 10, "Blue"),
    "B3": (560, 10, "Green"),
    "B4": (665, 10, "Red"),
    "B5": (705, 20, "Red edge 1"),
    "B6": (740, 20, "Red edge 2"),
    "B7": (783, 20, "Red edge 3"),
    "B8": (842, 10, "NIR"),
    "B8A": (865, 20, "Narrow NIR"),
    "B9": (945, 60, "Water vapour"),
    "B11": (1610, 20, "SWIR 1"),
    "B12": (2190, 20, "SWIR 2"),
}
BANDS = tuple(BAND_INFO)
DYNAMIC_CHANNELS = BANDS + ("NDSI", "NDVI", "SCA")
# Static (per-pixel, time-independent) channels each dataset version provides.
STATIC_CHANNELS_BY_VERSION = {
    "v1": ("DEM", "INCIDENCE"),
    "v2": ("DEM", "SLOPE", "NORTHNESS", "EASTNESS"),
}
STATIC_CHANNELS = tuple(dict.fromkeys(c for names in STATIC_CHANNELS_BY_VERSION.values() for c in names))
DEFAULT_VERSION = "v2"
# Hour of the hourly incidence-angle files used for the acquisition dates.
# Sentinel-2 passes over the Alps at about 10:20 UTC; the hour convention of the
# files (UTC or local time, start or end of the hour) is not documented.
S2_OVERPASS_HOUR = 11


def resolve_snowmelt_root(root=None, version=DEFAULT_VERSION):
    """Folder of one dataset version: ``<base>/<version>``.

    ``<base>`` is the folder holding the version folders: the explicit argument,
    else ``$SNOWMELT_DATA_ROOT``, else ``$DATA_PATH_BASE/snowmelt``.
    """
    base = root or os.environ.get("SNOWMELT_DATA_ROOT")
    if not base and os.environ.get("DATA_PATH_BASE"):
        base = Path(os.environ["DATA_PATH_BASE"]) / "snowmelt"
    if not base:
        raise ValueError(
            "Snowmelt data root not set: pass data.snowmelt.root, or set "
            "SNOWMELT_DATA_ROOT or DATA_PATH_BASE"
        )
    folder = Path(base).expanduser() / version
    if not any(path.is_dir() for path in folder.glob("S2_rawbands*")):
        found = sorted(path.name for path in Path(base).expanduser().glob("v*") if path.is_dir())
        raise FileNotFoundError(
            f"No snowmelt dataset {version!r} under {base} (expected {folder}/S2_rawbands*; "
            f"versions found: {', '.join(found) or 'none'})"
        )
    return folder


def dataset_folder(root, prefix):
    """The folder ``prefix`` (v1) or ``prefix_v<x>`` (v2) under a dataset root."""
    matches = sorted(path for path in Path(root).glob(f"{prefix}*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No {prefix} directory under snowmelt root {root}")
    return matches[0]


def topography_file(root, name):
    """Path of the ``<name>_*NivoletMask.tif`` topography raster, or None if absent."""
    matches = sorted(dataset_folder(root, "S2_topographic_attributes").rglob(f"{name}_*NivoletMask.tif"))
    return matches[0] if matches else None


def incidence_files(root):
    """Hourly incidence-angle rasters as ``{(day_of_year, hour): path}``; empty for v1."""
    files = {}
    for path in dataset_folder(root, "S2_topographic_attributes").rglob("INCIDENCEANGLE_DOY*_H*.tif"):
        match = re.fullmatch(r"INCIDENCEANGLE_DOY(\d+)_H(\d+)", path.stem)
        if match:
            files[int(match.group(1)), int(match.group(2))] = path
    return dict(sorted(files.items()))


def day_of_year(value):
    """Day of year (1 = 1 January) of an ISO date string."""
    return date.fromisoformat(value).timetuple().tm_yday


@contextmanager
def _quiet_tifffile():
    """Hide tifffile's log line about the float32 GDAL_NODATA tag.

    The topography files store float32-min as a decimal string that tifffile
    cannot cast back exactly, so it logs a parse error for the tag. The tag is
    unused: no-data is identified by value in :func:`load_snowmelt`.
    """
    tifffile_logger = logging.getLogger("tifffile")
    previous = tifffile_logger.level
    tifffile_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        tifffile_logger.setLevel(previous)


def read_tif(path):
    """Read a single-band GeoTIFF as float32.

    The raw bands are LZW-compressed; without ``imagecodecs`` installed, tifffile
    cannot decode them and :func:`Common.dataloader.tiff_lzw.read_tiff` falls
    back to a built-in decoder.
    """
    with _quiet_tifffile():
        return np.array(read_tiff(path), dtype=np.float32)  # a writable copy


def read_georef(path):
    """Return (x0, y0, dx, dy, crs_name) from the GeoTIFF tags."""
    with _quiet_tifffile():
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            dx, dy, _ = page.tags["ModelPixelScaleTag"].value
            _, _, _, x0, y0, _ = page.tags["ModelTiepointTag"].value
            crs = (tif.geotiff_metadata or {}).get("GTCitationGeoKey", "unknown CRS")
    return x0, y0, dx, dy, crs


def load_snowmelt(root, incidence_hour=S2_OVERPASS_HOUR):
    """Load every raster into stacked arrays, NaN outside the catchment and at no-data pixels.

    The catchment mask comes from the DEM (0 outside the catchment in v1, float32-min in v2).
    Each layer additionally carries its own sparse no-data (-9999 in the raw bands, NaN in
    NDVI, slope and aspect, -3.4e38 in the v1 incidence angle), which is kept per layer
    rather than merged into the mask.

    Returns a dict with ``raw`` (T, B, H, W), ``ndsi``/``ndvi``/``sca`` (T, H, W),
    ``dem``/``mask`` (H, W), ``slope``/``aspect`` (H, W, or None when the version has none),
    ``incidence``, ``dates``, ``bands`` and georeferencing. ``incidence`` is the static (H, W)
    map in v1; with hourly incidence files (v2) it is (T, H, W), the angle at
    ``incidence_hour`` on each acquisition date, and ``incidence_hour`` is recorded.
    """
    root = Path(root)
    raw_dir = dataset_folder(root, "S2_rawbands")
    index_dir = dataset_folder(root, "S2_derived_indexes")
    raw_files = sorted(raw_dir.glob("DoraNivolet_*_*.tif"))
    dates = sorted({re.match(r"DoraNivolet_(\w+?)_(\d{4}-\d{2}-\d{2})", f.stem).group(2) for f in raw_files})

    dem = read_tif(topography_file(root, "DEM"))
    mask = np.isfinite(dem) & (dem > -9000) & (dem != 0)

    def _clean(a):
        a[(a < -9000) | ~mask] = np.nan  # covers both -9999 and float32-min no-data values
        return a

    raw = _clean(np.stack([
        np.stack([read_tif(raw_dir / f"DoraNivolet_{b}_{d}.tif") for b in BANDS])
        for d in dates
    ]))

    def _stack(prefix):
        return _clean(np.stack([read_tif(index_dir / f"{prefix}_{d}.tif") for d in dates]))

    def _topography(name):
        path = topography_file(root, name)
        return None if path is None else _clean(read_tif(path))

    dem = _clean(dem)
    hourly = incidence_files(root)
    if hourly:
        missing = [d for d in dates if (day_of_year(d), incidence_hour) not in hourly]
        if missing:
            raise FileNotFoundError(f"No incidence-angle file at hour {incidence_hour} for {missing}")
        incidence = _clean(np.stack([read_tif(hourly[day_of_year(d), incidence_hour]) for d in dates]))
    else:
        incidence = _topography("INCIDENCEANGLE")

    x0, y0, dx, dy, crs = read_georef(raw_files[0])
    H, W = mask.shape
    return {
        "dates": dates,
        "bands": BANDS,
        "raw": raw,
        "ndsi": _stack("NDSI_Nivolet"),
        "ndvi": _stack("NDVI_Nivolet"),
        "sca": _stack("SCA_Nivolet_NDSIgt04"),
        "dem": dem,
        "slope": _topography("SLOPE"),
        "aspect": _topography("ASPECT"),
        "incidence": incidence,
        "incidence_hour": incidence_hour if hourly else None,
        "version": root.name,
        "mask": mask,
        "crs": crs,
        "pixel_size": (dx, dy),
        # imshow extent in km: (left, right, bottom, top)
        "extent_km": (x0 / 1e3, (x0 + W * dx) / 1e3, (y0 - H * dy) / 1e3, y0 / 1e3),
    }


def block_mean(arr, factor):
    """Downsample the trailing two axes by nan-aware block averaging."""
    if factor == 1:
        return arr
    H, W = arr.shape[-2:]
    h, w = H // factor, W // factor
    a = arr[..., : h * factor, : w * factor]
    a = a.reshape(*arr.shape[:-2], h, factor, w, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks outside the mask
        return np.nanmean(a, axis=(-3, -1))


def days_since_first(dates):
    """Acquisition times in days relative to the first date."""
    parsed = [date.fromisoformat(value) for value in dates]
    return tuple(float((value - parsed[0]).days) for value in parsed)


def _dynamic_channel(raw, name):
    """(T, H, W) array for a dynamic channel, scaled to roughly [0, 1]."""
    if name in raw["bands"]:
        # Surface reflectance; fresh snow can exceed 1, so clip.
        return np.clip(raw["raw"][:, raw["bands"].index(name)], 0.0, 1.0)
    if name in ("NDSI", "NDVI"):
        return (raw[name.lower()] + 1.0) / 2.0
    if name == "SCA":
        return raw["sca"]
    raise ValueError(f"Unknown snowmelt dynamic channel {name!r}; expected one of {DYNAMIC_CHANNELS}")


def _static_channel(raw, name):
    """(H, W) static covariate scaled to [0, 1] within the catchment.

    Aspect is cyclic (0 and 360 degrees are both north), so it enters as two
    channels: NORTHNESS = cos(aspect) and EASTNESS = sin(aspect), mapped from
    [-1, 1] to [0, 1].
    """
    if name == "DEM":
        dem = raw["dem"]
        low, high = np.nanmin(dem), np.nanmax(dem)
        return (dem - low) / (high - low)
    if name == "INCIDENCE":
        if raw["incidence"].ndim != 2:
            raise ValueError(
                "This snowmelt dataset has hourly incidence angles (v2), not one static map; "
                "INCIDENCE is a static channel in v1 only (use SLOPE, NORTHNESS, EASTNESS)"
            )
        return raw["incidence"] / 90.0
    if name in ("SLOPE", "NORTHNESS", "EASTNESS"):
        if raw.get("slope") is None or raw.get("aspect") is None:
            raise ValueError(f"Static channel {name} needs the slope and aspect maps of dataset v2")
        if name == "SLOPE":
            return raw["slope"] / 90.0
        aspect = np.deg2rad(raw["aspect"])
        return ((np.cos(aspect) if name == "NORTHNESS" else np.sin(aspect)) + 1.0) / 2.0
    raise ValueError(f"Unknown snowmelt static channel {name!r}; expected one of {STATIC_CHANNELS}")


def fill_missing(values, mask):
    """Fill NaNs inside ``mask`` from the nearest valid pixel; zero outside ``mask``.

    ``values`` is ``(..., H, W)``; each leading slice is filled independently.
    """
    out = np.array(values, dtype=np.float32, copy=True)
    flat = out.reshape(-1, *out.shape[-2:])
    for image in flat:
        missing = np.isnan(image) & mask
        if missing.any():
            valid = ~np.isnan(image) & mask
            if not valid.any():
                raise ValueError("Cannot fill a channel with no valid catchment pixels")
            _, (rows, cols) = ndi.distance_transform_edt(~valid, return_indices=True)
            image[missing] = image[rows[missing], cols[missing]]
        image[~mask] = 0.0
    return np.nan_to_num(out, nan=0.0)


def _date_position(dates, entry):
    """Position in ``dates`` of an entry given as an index (negative counts from the end) or ISO date."""
    if isinstance(entry, str):
        if entry not in dates:
            raise ValueError(f"Snowmelt date {entry!r} is not one of the acquisitions {dates}")
        return dates.index(entry)
    if not -len(dates) <= int(entry) < len(dates):
        raise ValueError(f"Snowmelt date index {entry} is out of range for {len(dates)} acquisitions")
    return int(entry) % len(dates)


@dataclass(frozen=True)
class DateSplit:
    """Which acquisitions a run uses, as positions in the dataset's full date list.

    ``used`` are the dates that are not excluded (e.g. for bad data), in time
    order: the sequence a model is evaluated on. ``training`` are the used dates
    minus the held-out ones: the sequence it is trained on. A held-out date is
    skipped in training, so its two neighbours form one longer interval.
    """

    dates: tuple[str, ...]
    used: tuple[int, ...]
    training: tuple[int, ...]

    @classmethod
    def resolve(cls, dates, exclude=(), hold_out=()):
        """Split ``dates``; ``exclude`` and ``hold_out`` hold indices or ISO dates."""
        dates = tuple(dates)
        excluded = {_date_position(dates, entry) for entry in exclude}
        held_out = {_date_position(dates, entry) for entry in hold_out}
        if excluded & held_out:
            raise ValueError("A snowmelt date cannot be both excluded and held out")
        used = tuple(i for i in range(len(dates)) if i not in excluded)
        training = tuple(i for i in used if i not in held_out)
        if len(training) < 2:
            raise ValueError(f"Training needs at least 2 snowmelt dates, but only {len(training)} are left")
        if used[0] in held_out:
            raise ValueError("The first used snowmelt date is the initial condition and cannot be held out")
        return cls(dates, used, training)

    @property
    def held_out(self):
        return tuple(i for i in self.used if i not in self.training)


@dataclass(frozen=True)
class SnowmeltSequence:
    data: np.ndarray            # [1, T, C, H, W] dynamic target channels, zero outside the catchment
    boundary_mask: np.ndarray   # [1, 1 + S, H, W] catchment mask then static covariates
    observation_times: tuple[float, ...]  # days since the first acquisition
    dates: tuple[str, ...]
    held_out: tuple[bool, ...]  # per date: True if it was held out of training
    channel_names: tuple[str, ...]
    boundary_channel_names: tuple[str, ...]
    catchment_fraction: np.ndarray  # [H, W] fraction of each block inside the catchment


def build_snowmelt_sequence(
    root=None,
    version=DEFAULT_VERSION,
    target_channels=("SCA",),
    static_channels=("DEM",),
    downsample=1,
    pad=4,
    mask_threshold=0.5,
    exclude_dates=(),
    hold_out_dates=(),
    include_held_out=False,
    raw=None,
):
    """Assemble one snowmelt training sequence.

    Dynamic channels are scaled to ``[0, 1]`` (reflectance clipped, NDSI/NDVI
    mapped from ``[-1, 1]``), NaN-filled from the nearest valid pixel at full
    resolution, then block-averaged by ``downsample``. A block is inside the
    catchment when at least ``mask_threshold`` of it is. ``pad`` zero pixels
    are added on every side so circular padding cannot couple opposite edges.
    ``exclude_dates`` are never used and ``hold_out_dates`` are left out of
    training (see :class:`DateSplit`); ``include_held_out`` keeps the held-out
    dates, giving the evaluation sequence. Observation times count from the
    first used date.
    ``root`` and ``version`` locate the data (see :func:`resolve_snowmelt_root`);
    ``raw`` may instead be a precomputed :func:`load_snowmelt` result.
    """
    if not target_channels:
        raise ValueError("At least one snowmelt target channel is required")
    raw = load_snowmelt(resolve_snowmelt_root(root, version)) if raw is None else raw
    mask = raw["mask"]
    split = DateSplit.resolve(raw["dates"], exclude_dates, hold_out_dates)
    keep = list(split.used if include_held_out else split.training)
    dates = tuple(split.dates[i] for i in keep)

    dynamic = np.stack(
        [fill_missing(_dynamic_channel(raw, name)[keep], mask) for name in target_channels], axis=1
    )
    static = [fill_missing(_static_channel(raw, name), mask) for name in static_channels]

    fraction = block_mean(mask.astype(np.float32), downsample)
    inside = fraction >= mask_threshold

    def _downsample(values):
        # Average over catchment pixels only, so edge blocks are not diluted by zeros.
        weighted = block_mean(values * mask, downsample)
        with np.errstate(invalid="ignore", divide="ignore"):
            averaged = np.where(fraction > 0, weighted / np.maximum(fraction, 1e-12), 0.0)
        return np.where(inside, averaged, 0.0).astype(np.float32)

    dynamic = _downsample(dynamic)
    boundary = np.stack([inside.astype(np.float32)] + [_downsample(channel) for channel in static])

    pad_width = [(0, 0)] * (dynamic.ndim - 2) + [(pad, pad), (pad, pad)]
    dynamic = np.pad(dynamic, pad_width)
    boundary = np.pad(boundary, [(0, 0), (pad, pad), (pad, pad)])
    fraction = np.pad(fraction, pad)

    return SnowmeltSequence(
        data=dynamic[None],
        boundary_mask=boundary[None],
        observation_times=days_since_first(dates),
        dates=dates,
        held_out=tuple(i in split.held_out for i in keep),
        channel_names=tuple(target_channels),
        boundary_channel_names=("CATCHMENT",) + tuple(static_channels),
        catchment_fraction=fraction,
    )


__all__ = [
    "BANDS",
    "BAND_INFO",
    "DEFAULT_VERSION",
    "DYNAMIC_CHANNELS",
    "DateSplit",
    "S2_OVERPASS_HOUR",
    "STATIC_CHANNELS",
    "STATIC_CHANNELS_BY_VERSION",
    "SnowmeltSequence",
    "block_mean",
    "build_snowmelt_sequence",
    "dataset_folder",
    "day_of_year",
    "days_since_first",
    "fill_missing",
    "incidence_files",
    "load_snowmelt",
    "read_georef",
    "read_tif",
    "resolve_snowmelt_root",
    "topography_file",
]
