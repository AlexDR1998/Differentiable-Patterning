"""Nivolet (Gran Paradiso) Sentinel-2 snowmelt rasters as NCA target sequences.

Data layout under the dataset root::

    S2_rawbands/DoraNivolet_<band>_<date>.tif      12 L2A bands x T dates (LZW, float64)
    S2_derived_indexes/{NDSI,NDVI}_Nivolet_<date>.tif
    S2_derived_indexes/SCA_Nivolet_NDSIgt04_<date>.tif   binary snow cover (NDSI > 0.4)
    S2_topographic_attributes/{DEM,INCIDENCEANGLE}_10mTinitaly_NivoletMask.tif

All rasters share one 10 m grid in ED50 / UTM 32N. The catchment is where the
DEM is non-zero; each layer additionally carries sparse no-data of its own.

:func:`load_snowmelt` returns the raw rasters for exploration (NaN outside the
catchment and at no-data pixels). :func:`build_snowmelt_sequence` turns them
into a training sequence: selected dynamic channels scaled to ``[0, 1]``,
missing values filled, block-downsampled, zero-padded, plus fixed per-pixel
boundary channels (catchment mask and static terrain covariates) and the
acquisition times in days.
"""

from dataclasses import dataclass
from datetime import date
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
STATIC_CHANNELS = ("DEM", "INCIDENCE")


def resolve_snowmelt_root(root=None):
    """Dataset root: explicit argument, else ``$SNOWMELT_DATA_ROOT``, else ``$DATA_PATH_BASE/snowmelt``."""
    root = root or os.environ.get("SNOWMELT_DATA_ROOT")
    if not root and os.environ.get("DATA_PATH_BASE"):
        root = Path(os.environ["DATA_PATH_BASE"]) / "snowmelt"
    if not root:
        raise ValueError(
            "Snowmelt data root not set: pass data.snowmelt.root, or set "
            "SNOWMELT_DATA_ROOT or DATA_PATH_BASE"
        )
    root = Path(root).expanduser()
    if not (root / "S2_rawbands").is_dir():
        raise FileNotFoundError(f"No S2_rawbands directory under snowmelt root {root}")
    return root


def read_tif(path):
    """Read a single-band GeoTIFF as float32.

    The raw bands are LZW-compressed; without ``imagecodecs`` installed, tifffile
    cannot decode them and :func:`Common.dataloader.tiff_lzw.read_tiff` falls
    back to a built-in decoder.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # float32 files carry an uncastable GDAL_NODATA tag
        return np.asarray(read_tiff(path), dtype=np.float32)


def read_georef(path):
    """Return (x0, y0, dx, dy, crs_name) from the GeoTIFF tags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # float32 files carry an uncastable GDAL_NODATA tag
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            dx, dy, _ = page.tags["ModelPixelScaleTag"].value
            _, _, _, x0, y0, _ = page.tags["ModelTiepointTag"].value
            crs = (tif.geotiff_metadata or {}).get("GTCitationGeoKey", "unknown CRS")
    return x0, y0, dx, dy, crs


def load_snowmelt(root):
    """Load every raster into stacked arrays, NaN outside the catchment and at no-data pixels.

    The catchment mask comes from the DEM (exactly 0 outside the catchment). Each layer
    additionally carries its own sparse no-data (-9999 in the raw bands, NaN in NDVI,
    -3.4e38 in the incidence angle), which is kept per layer rather than merged into the mask.

    Returns a dict with ``raw`` (T, B, H, W), ``ndsi``/``ndvi``/``sca`` (T, H, W),
    ``dem``/``incidence``/``mask`` (H, W), ``dates``, ``bands`` and georeferencing.
    """
    root = Path(root)
    raw_files = sorted((root / "S2_rawbands").glob("DoraNivolet_*_*.tif"))
    dates = sorted({re.match(r"DoraNivolet_(\w+?)_(\d{4}-\d{2}-\d{2})", f.stem).group(2) for f in raw_files})

    topo = root / "S2_topographic_attributes"
    dem = read_tif(topo / "DEM_10mTinitaly_NivoletMask.tif")
    mask = dem != 0

    def _clean(a):
        a[(a < -9000) | ~mask] = np.nan  # covers both -9999 and float32-min no-data values
        return a

    raw = _clean(np.stack([
        np.stack([read_tif(root / "S2_rawbands" / f"DoraNivolet_{b}_{d}.tif") for b in BANDS])
        for d in dates
    ]))

    def _stack(prefix):
        return _clean(np.stack([read_tif(root / "S2_derived_indexes" / f"{prefix}_{d}.tif") for d in dates]))

    dem = _clean(dem)
    incidence = _clean(read_tif(topo / "INCIDENCEANGLE_10mTinitaly_NivoletMask.tif"))

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
        "incidence": incidence,
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
    """(H, W) static covariate scaled to [0, 1] within the catchment."""
    if name == "DEM":
        dem = raw["dem"]
        low, high = np.nanmin(dem), np.nanmax(dem)
        return (dem - low) / (high - low)
    if name == "INCIDENCE":
        return raw["incidence"] / 90.0
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


@dataclass(frozen=True)
class SnowmeltSequence:
    data: np.ndarray            # [1, T, C, H, W] dynamic target channels, zero outside the catchment
    boundary_mask: np.ndarray   # [1, 1 + S, H, W] catchment mask then static covariates
    observation_times: tuple[float, ...]  # days since the first acquisition
    dates: tuple[str, ...]
    channel_names: tuple[str, ...]
    boundary_channel_names: tuple[str, ...]
    catchment_fraction: np.ndarray  # [H, W] fraction of each block inside the catchment


def build_snowmelt_sequence(
    root=None,
    target_channels=("SCA",),
    static_channels=STATIC_CHANNELS,
    downsample=1,
    pad=4,
    mask_threshold=0.5,
    raw=None,
):
    """Assemble one snowmelt training sequence.

    Dynamic channels are scaled to ``[0, 1]`` (reflectance clipped, NDSI/NDVI
    mapped from ``[-1, 1]``), NaN-filled from the nearest valid pixel at full
    resolution, then block-averaged by ``downsample``. A block is inside the
    catchment when at least ``mask_threshold`` of it is. ``pad`` zero pixels
    are added on every side so circular padding cannot couple opposite edges.
    ``raw`` may be a precomputed :func:`load_snowmelt` result.
    """
    if not target_channels:
        raise ValueError("At least one snowmelt target channel is required")
    raw = load_snowmelt(resolve_snowmelt_root(root)) if raw is None else raw
    mask = raw["mask"]

    dynamic = np.stack([fill_missing(_dynamic_channel(raw, name), mask) for name in target_channels], axis=1)
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
        observation_times=days_since_first(raw["dates"]),
        dates=tuple(raw["dates"]),
        channel_names=tuple(target_channels),
        boundary_channel_names=("CATCHMENT",) + tuple(static_channels),
        catchment_fraction=fraction,
    )


__all__ = [
    "BANDS",
    "BAND_INFO",
    "DYNAMIC_CHANNELS",
    "STATIC_CHANNELS",
    "SnowmeltSequence",
    "block_mean",
    "build_snowmelt_sequence",
    "days_since_first",
    "fill_missing",
    "load_snowmelt",
    "read_georef",
    "read_tif",
    "resolve_snowmelt_root",
]
