# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "tifffile",
#   "opencv-python",
# ]
# ///

"""Interactive exploration of the Nivolet (Gran Paradiso) Sentinel-2 snowmelt dataset.

Run from the repository root with:

    marimo edit Experiments/snowmelt/snowmelt_explorer.py

Data layout (``SNOWMELT_DATA_ROOT``, default ``~/PhD/Data/snowmelt``):

    S2_rawbands/DoraNivolet_<band>_<date>.tif      12 L2A bands x 5 dates (LZW, float64)
    S2_derived_indexes/{NDSI,NDVI}_Nivolet_<date>.tif
    S2_derived_indexes/SCA_Nivolet_NDSIgt04_<date>.tif   binary snow cover (NDSI > 0.4)
    S2_topographic_attributes/{DEM,INCIDENCEANGLE}_10mTinitaly_NivoletMask.tif

All rasters share one 517 x 514 grid at 10 m in ED50 / UTM 32N (EPSG:23032).
The raw bands are LZW-compressed; without ``imagecodecs`` installed tifffile cannot
decode them, so reading falls back to OpenCV's bundled libtiff.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    import re
    import warnings
    from pathlib import Path

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    import matplotlib
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.ticker import MaxNLocator

    # Light figures regardless of the marimo theme, so annotation ink stays legible.
    plt.style.use("default")

    DATA_ROOT = Path(
        os.environ.get("SNOWMELT_DATA_ROOT", Path.home() / "PhD" / "Data" / "snowmelt")
    )

    # Sentinel-2 MSI: central wavelength (nm) and native resolution (m).
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

    RGB_PRESETS = {
        "True colour (B4, B3, B2)": ("B4", "B3", "B2"),
        "NIR false colour (B8, B4, B3)": ("B8", "B4", "B3"),
        "SWIR (B12, B8A, B4): snow appears cyan": ("B12", "B8A", "B4"),
        "Agriculture (B11, B8, B2)": ("B11", "B8", "B2"),
    }

    # Snow is white: dark rock (no snow / low NDSI) -> white (snow / high NDSI)
    SNOW_CMAP = LinearSegmentedColormap.from_list("snow", ["#2b2622", "#7d7266", "#c9c3bb", "#ffffff"])
    if "snow" not in matplotlib.colormaps:
        matplotlib.colormaps.register(SNOW_CMAP)

    LAYERS = ("Raw band", "NDSI", "NDVI", "SCA (NDSI > 0.4)", "DEM", "Incidence angle")
    # Per-layer defaults: (colormap, diverging about zero, fixed limits or None, units)
    LAYER_STYLE = {
        "Raw band": ("Greys_r", False, None, "reflectance"),
        "NDSI": ("snow", False, None, "NDSI"),
        "NDVI": ("BrBG", True, None, "NDVI"),
        "SCA (NDSI > 0.4)": ("snow", False, (0.0, 1.0), "snow cover"),
        "DEM": ("Oranges", False, None, "elevation (m)"),
        "Incidence angle": ("Purples", False, None, "incidence angle (°)"),
    }
    CMAPS = ("auto", "snow", "Greys_r", "Blues", "Oranges", "Purples", "RdBu", "BrBG", "cividis", "viridis")

    def read_tif(path):
        """Read a single-band GeoTIFF as float32, falling back to OpenCV for LZW."""
        try:
            arr = tifffile.imread(path)
        except ValueError:  # compression codec needs imagecodecs
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise IOError(f"Could not decode {path}")
        return np.asarray(arr, dtype=np.float32)

    def read_georef(path):
        """Return (x0, y0, dx, dy, crs_name) from the GeoTIFF tags."""
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

    def get_layer(data, layer, band, t):
        """Return the (H, W) array for a layer at date index t (static layers ignore t)."""
        if layer == "Raw band":
            return data["raw"][t, data["bands"].index(band)]
        return {
            "NDSI": lambda: data["ndsi"][t],
            "NDVI": lambda: data["ndvi"][t],
            "SCA (NDSI > 0.4)": lambda: data["sca"][t],
            "DEM": lambda: data["dem"],
            "Incidence angle": lambda: data["incidence"],
        }[layer]()

    # Every candidate NCA channel: raw bands, derived indexes, then static topography
    CHANNELS = BANDS + ("NDSI", "NDVI", "SCA", "DEM", "Incidence angle")

    def channel_stack(data, name):
        """Return (array, is_static, LAYER_STYLE key, units); array is (T, H, W), or (H, W) if static."""
        if name in data["bands"]:
            return data["raw"][:, data["bands"].index(name)], False, "Raw band", f"{name} reflectance"
        return {
            "NDSI": (data["ndsi"], False, "NDSI", "NDSI"),
            "NDVI": (data["ndvi"], False, "NDVI", "NDVI"),
            "SCA": (data["sca"], False, "SCA (NDSI > 0.4)", "snow cover"),
            "DEM": (data["dem"], True, "DEM", "elevation (m)"),
            "Incidence angle": (data["incidence"], True, "Incidence angle", "incidence angle (°)"),
        }[name]

    def color_limits(arr, layer, pct):
        """Colour limits: fixed for SCA, symmetric about zero for diverging layers, else percentiles."""
        _, diverging, fixed, _ = LAYER_STYLE[layer]
        if fixed is not None:
            return fixed
        lo, hi = np.nanpercentile(arr, pct)
        if diverging:
            m = max(abs(lo), abs(hi))
            return -m, m
        return lo, hi

    def hillshade(dem, azimuth=315.0, altitude=45.0, pixel_size=10.0):
        """Standard Lambertian hillshade in [0, 1]."""
        z = np.nan_to_num(dem, nan=np.nanmean(dem))
        gy, gx = np.gradient(z, pixel_size)
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
        shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
        return np.clip(shade, 0, 1)

    def rgb_composite(data, bands, t, pct, gamma):
        """Per-channel percentile stretch, with limits computed over all dates so time steps are comparable."""
        idx = [data["bands"].index(b) for b in bands]
        chans = []
        for i in idx:
            lo, hi = np.nanpercentile(data["raw"][:, i], pct)
            c = np.clip((data["raw"][t, i] - lo) / (hi - lo + 1e-12), 0, 1) ** (1 / gamma)
            chans.append(c)
        rgb = np.stack(chans, axis=-1)
        # Transparent outside the catchment and wherever any channel is no-data
        alpha = np.all(np.isfinite(rgb), axis=-1, keepdims=True).astype(np.float32)
        return np.concatenate([np.nan_to_num(rgb), alpha], axis=-1)

    def style_map_axes(ax, title=None):
        ax.set_facecolor("#e6e6e6")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.tick_params(labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        if title:
            ax.set_title(title, fontsize=10)

    def draw_outline(ax, mask, extent=None):
        """Catchment outline, so white (snow) pixels stay distinguishable from the background."""
        # With an extent, match imshow's upper origin; without one, pixel-index coords already align
        ax.contour(mask.astype(np.float32), levels=[0.5], colors="#555555", linewidths=0.6,
                   extent=extent, origin="upper" if extent is not None else None)

    def date_colors(n):
        """Ordered dates get an ordered (sequential, single-hue) ramp."""
        return plt.cm.Blues(np.linspace(0.35, 1.0, n))

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Nivolet snowmelt: Sentinel-2 dataset explorer

    Visual exploration of five Sentinel-2 L2A acquisitions (May–July 2018) over the
    Dora/Nivolet catchment, together with derived snow/vegetation indexes and topography.
    The aim is to understand the spatiotemporal structure of snowmelt before framing it as an
    NCA target sequence.
    """)
    return


@app.cell
def _():
    data = load_snowmelt(DATA_ROOT)
    return (data,)


@app.cell(hide_code=True)
def _(data):
    _T, _B, _H, _W = data["raw"].shape
    _n_in = int(data["mask"].sum())
    _dx, _dy = data["pixel_size"]
    _ext = data["extent_km"]
    _band_rows = "\n".join(
        f"| {b} | {wl} | {res} | {name} |" for b, (wl, res, name) in BAND_INFO.items()
    )
    mo.md(f"""
    ## Dataset summary

    | | |
    |---|---|
    | Grid | {_H} × {_W} pixels at {_dx:g} × {_dy:g} m ({_H * _dy / 1e3:.2f} × {_W * _dx / 1e3:.2f} km) |
    | CRS | {data["crs"]} |
    | Extent (km) | E {_ext[0]:.2f}–{_ext[1]:.2f}, N {_ext[2]:.2f}–{_ext[3]:.2f} |
    | Catchment pixels | {_n_in:,} of {_H * _W:,} ({100 * _n_in / (_H * _W):.1f}%) |
    | Dates | {", ".join(data["dates"])} |
    | Raw bands | {_B} (all resampled to 10 m) |

    <details><summary>Sentinel-2 band reference</summary>

    | Band | Centre (nm) | Native res. (m) | Name |
    |---|---|---|---|
    {_band_rows}

    </details>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Missing data

    Pixels **inside the catchment** that are no-data in a given file (outside-catchment pixels
    are excluded). Missing values are sparse and date-specific. They appear only in
    some raw bands on the two July acquisitions, in the NDVI of those dates (which uses B8), and
    in the static incidence-angle map. NDSI and SCA (built from B3 and B11) are complete.
    These pixels are NaN in the arrays loaded here and need filling or masking before NCA training.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    _mask = data["mask"]
    _dates = data["dates"]
    missing_row_labels = list(data["bands"]) + ["NDSI", "NDVI", "SCA"]
    _dynamic = [data["raw"][:, _b] for _b in range(len(data["bands"]))] + [data["ndsi"], data["ndvi"], data["sca"]]

    # (layer, date) -> boolean (H, W) map of missing catchment pixels
    missing_maps = {}
    missing_matrix = np.zeros((len(missing_row_labels), len(_dates)), dtype=int)
    for _i, (_name, _stack) in enumerate(zip(missing_row_labels, _dynamic)):
        for _t, _d in enumerate(_dates):
            _m = np.isnan(_stack[_t]) & _mask
            missing_matrix[_i, _t] = _m.sum()
            if _m.any():
                missing_maps[f"{_name} · {_d}"] = _m
    for _name, _key in (("DEM", "dem"), ("Incidence angle", "incidence")):
        _m = np.isnan(data[_key]) & _mask
        if _m.any():
            missing_maps[f"{_name} · static"] = _m

    _n_catch = int(_mask.sum())
    missing_rows = [
        {"layer · date": _k, "missing pixels": int(_v.sum()), "% of catchment": round(100 * _v.sum() / _n_catch, 4)}
        for _k, _v in missing_maps.items()
    ]
    missing_select = mo.ui.dropdown(["All layers"] + list(missing_maps), value="All layers", label="Highlight")
    return (
        missing_maps,
        missing_matrix,
        missing_row_labels,
        missing_rows,
        missing_select,
    )


@app.cell(hide_code=True)
def _(
    data,
    missing_maps,
    missing_matrix,
    missing_row_labels,
    missing_rows,
    missing_select,
):
    _mask = data["mask"]
    _dates = data["dates"]
    _any = np.any(list(missing_maps.values()), axis=0) if missing_maps else np.zeros_like(_mask)
    _count = np.sum(list(missing_maps.values()), axis=0) if missing_maps else np.zeros(_mask.shape, int)

    _fig, (_ax_mat, _ax_map) = plt.subplots(
        1, 2, figsize=(18, 8.8), constrained_layout=True, gridspec_kw={"width_ratios": [0.8, 1.2]}
    )

    # Layer x date matrix of missing-pixel counts; zero cells are left blank
    _shown = np.where(missing_matrix > 0, missing_matrix, np.nan)
    _ax_mat.imshow(_shown, cmap="Oranges", vmin=0, vmax=max(1, missing_matrix.max()), aspect="auto")
    for (_i, _t), _v in np.ndenumerate(missing_matrix):
        if _v:
            _ax_mat.text(_t, _i, str(_v), ha="center", va="center", fontsize=8,
                         color="white" if _v > 0.6 * missing_matrix.max() else "#222")
    _ax_mat.set_xticks(range(len(_dates)), _dates, rotation=30, fontsize=8)
    _ax_mat.set_yticks(range(len(missing_row_labels)), missing_row_labels, fontsize=8)
    _ax_mat.set_xticks(np.arange(-0.5, len(_dates)), minor=True)
    _ax_mat.set_yticks(np.arange(-0.5, len(missing_row_labels)), minor=True)
    _ax_mat.grid(which="minor", color="#e0e0e0", lw=1)
    _ax_mat.tick_params(which="minor", length=0)
    for _s in _ax_mat.spines.values():
        _s.set_visible(False)
    _ax_mat.set_title("Missing catchment pixels per layer and date", fontsize=10)
    _static = [f"{_k.split(' · ')[0]}: {int(_v.sum())}" for _k, _v in missing_maps.items() if _k.endswith("static")]
    _ax_mat.set_xlabel("Static layers: " + (", ".join(_static) if _static else "none missing"), fontsize=8)

    # Map: hillshade + missing pixels as markers (single pixels are invisible at full-map scale)
    _hs = np.where(_mask, hillshade(np.where(_mask, data["dem"], np.nan)), np.nan)
    _ax_map.imshow(_hs, cmap="Greys_r", vmin=0, vmax=1, extent=data["extent_km"], alpha=0.6)
    _x0, _, _, _y1 = data["extent_km"]
    _dx, _dy = data["pixel_size"]
    if missing_select.value == "All layers":
        _rr, _cc = np.nonzero(_any)
        _sc = _ax_map.scatter(_x0 + (_cc + 0.5) * _dx / 1e3, _y1 - (_rr + 0.5) * _dy / 1e3, c=_count[_rr, _cc],
                              cmap="Oranges", vmin=0, s=14, edgecolors="#7a2e00", linewidths=0.4)
        _cb = _fig.colorbar(_sc, ax=_ax_map, shrink=0.8, label="number of layers missing")
        _cb.locator = MaxNLocator(integer=True)
        _cb.update_ticks()
        _title = f"All layers: {_any.sum()} pixels missing in ≥ 1 layer ({100 * _any.sum() / _mask.sum():.3f}% of catchment)"
    else:
        _rr, _cc = np.nonzero(missing_maps[missing_select.value])
        _ax_map.scatter(_x0 + (_cc + 0.5) * _dx / 1e3, _y1 - (_rr + 0.5) * _dy / 1e3, s=14,
                        color="#d9480f", edgecolors="#7a2e00", linewidths=0.4)
        _title = f"{missing_select.value}: {len(_rr)} pixels"
    style_map_axes(_ax_map, _title)

    mo.vstack([
        missing_select,
        _fig,
        mo.accordion({"Table of missing pixels": mo.ui.table(missing_rows, selection=None)}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Layer explorer
    """)
    return


@app.cell(hide_code=True)
def _(data):
    layer_select = mo.ui.dropdown(LAYERS, value="NDSI", label="Layer")
    band_select = mo.ui.dropdown(BANDS, value="B3", label="Band (raw only)")
    date_select = mo.ui.dropdown(
        {d: i for i, d in enumerate(data["dates"])}, value=data["dates"][0], label="Date"
    )
    cmap_select = mo.ui.dropdown(CMAPS, value="auto", label="Colormap")
    pct_range = mo.ui.range_slider(0, 100, step=0.5, value=[2, 98], label="Stretch percentiles", show_value=True)
    shade_toggle = mo.ui.checkbox(value=True, label="Hillshade underlay")
    mo.hstack([layer_select, band_select, date_select, cmap_select, pct_range, shade_toggle], justify="start", gap=1.5, wrap=True)
    return (
        band_select,
        cmap_select,
        date_select,
        layer_select,
        pct_range,
        shade_toggle,
    )


@app.cell(hide_code=True)
def _(
    band_select,
    cmap_select,
    data,
    date_select,
    layer_select,
    pct_range,
    shade_toggle,
):
    _layer = layer_select.value
    _arr = get_layer(data, _layer, band_select.value, date_select.value)
    _cmap_default, _, _, _units = LAYER_STYLE[_layer]
    _cmap = _cmap_default if cmap_select.value == "auto" else cmap_select.value
    _vmin, _vmax = color_limits(_arr, _layer, pct_range.value)

    _fig, (_ax_map, _ax_hist) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True
    )
    if shade_toggle.value:
        _hs = np.where(data["mask"], hillshade(data["dem"]), np.nan)
        _ax_map.imshow(_hs, cmap="Greys_r", extent=data["extent_km"], vmin=0, vmax=1)
    _im = _ax_map.imshow(
        _arr, cmap=_cmap, vmin=_vmin, vmax=_vmax, extent=data["extent_km"],
        alpha=0.75 if shade_toggle.value else 1.0, interpolation="nearest",
    )
    draw_outline(_ax_map, data["mask"], data["extent_km"])
    _fig.colorbar(_im, ax=_ax_map, shrink=0.8, label=_units)
    _when ="" if _layer in ("DEM", "Incidence angle") else f" — {date_select.selected_key}"
    _what = f"{band_select.value} ({BAND_INFO[band_select.value][2]})" if _layer == "Raw band" else _layer
    style_map_axes(_ax_map, f"{_what}{_when}")

    _vals = _arr[np.isfinite(_arr)]
    _ax_hist.hist(_vals, bins=100, color="#4a78b5", edgecolor="none")
    _ax_hist.axvline(_vmin, color="#555", lw=1, ls="--")
    _ax_hist.axvline(_vmax, color="#555", lw=1, ls="--")
    _ax_hist.set_xlabel(_units)
    _ax_hist.set_ylabel("pixels")
    _ax_hist.set_title(
        f"mean {_vals.mean():.3g} · median {np.median(_vals):.3g} · std {_vals.std():.3g}", fontsize=9
    )
    _ax_hist.spines[["top", "right"]].set_visible(False)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## RGB composites

    Channel stretch limits are computed over **all dates jointly**, so brightness changes
    between dates are real rather than artefacts of per-image normalisation.
    """)
    return


@app.cell(hide_code=True)
def _():
    rgb_preset = mo.ui.dropdown(RGB_PRESETS, value="True colour (B4, B3, B2)", label="Composite")
    rgb_pct = mo.ui.range_slider(0, 100, step=0.5, value=[1, 99], label="Stretch percentiles", show_value=True)
    rgb_gamma = mo.ui.slider(0.5, 3.0, step=0.1, value=1.4, label="Gamma", show_value=True)
    mo.hstack([rgb_preset, rgb_pct, rgb_gamma], justify="start", gap=1.5, wrap=True)
    return rgb_gamma, rgb_pct, rgb_preset


@app.cell(hide_code=True)
def _(data, rgb_gamma, rgb_pct, rgb_preset):
    _T = len(data["dates"])
    _fig, _axes = plt.subplots(1, _T, figsize=(3.6 * _T, 4), constrained_layout=True, sharex=True, sharey=True)
    for _t, _ax in enumerate(np.atleast_1d(_axes)):
        _ax.imshow(rgb_composite(data, rgb_preset.value, _t, rgb_pct.value, rgb_gamma.value), extent=data["extent_km"])
        style_map_axes(_ax, data["dates"][_t])
        if _t > 0:
            _ax.set_ylabel("")
    _fig.suptitle(rgb_preset.selected_key, fontsize=11)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time series: small multiples

    One layer across all dates on a shared colour scale. *Change from first date* shows
    where each quantity has moved since the late-May acquisition.
    """)
    return


@app.cell(hide_code=True)
def _():
    ts_layer = mo.ui.dropdown(LAYERS[:4], value="NDSI", label="Layer")
    ts_band = mo.ui.dropdown(BANDS, value="B11", label="Band (raw only)")
    ts_mode = mo.ui.radio(["Absolute", "Change from first date"], value="Absolute", label="Mode", inline=True)
    mo.hstack([ts_layer, ts_band, ts_mode], justify="start", gap=1.5, wrap=True)
    return ts_band, ts_layer, ts_mode


@app.cell(hide_code=True)
def _(data, ts_band, ts_layer, ts_mode):
    _dates = data["dates"]
    _stack = np.stack([get_layer(data, ts_layer.value, ts_band.value, _t) for _t in range(len(_dates))])
    _cmap, _, _, _units = LAYER_STYLE[ts_layer.value]
    if ts_mode.value == "Change from first date":
        _stack = _stack - _stack[0]
        _m = np.nanpercentile(np.abs(_stack[1:]), 99) if len(_dates) > 1 else 1.0
        _vmin, _vmax, _cmap, _units = -_m, _m, "RdBu", f"Δ {_units}"
    else:
        _vmin, _vmax = color_limits(_stack, ts_layer.value, (2, 98))

    _fig, _axes = plt.subplots(1, len(_dates), figsize=(3.6 * len(_dates), 4), constrained_layout=True, sharey=True)
    for _t, _ax in enumerate(_axes):
        _im = _ax.imshow(_stack[_t], cmap=_cmap, vmin=_vmin, vmax=_vmax, extent=data["extent_km"], interpolation="nearest")
        draw_outline(_ax, data["mask"], data["extent_km"])
        style_map_axes(_ax, _dates[_t])
        if _t > 0:
            _ax.set_ylabel("")
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label=_units)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Snow-cover dynamics

    Left: the last date on which each pixel is classified as snow (SCA = 1), a discrete
    proxy for melt-out timing. Pixels that are snow-free and then snow-covered again are
    **non-monotonic**, caused by either fresh snowfall or misclassification (cloud, shadow).
    A developmental model that assumes monotonic melt will not reproduce these pixels.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    _sca = data["sca"]
    _mask = data["mask"]
    _dates = data["dates"]
    _T = len(_dates)
    _snow = np.nan_to_num(_sca) > 0.5

    # -1 = never snow-covered in the record, t = last date with snow
    _last = np.where(_snow.any(0), _T - 1 - np.argmax(_snow[::-1], axis=0), -1).astype(float)
    _last[~_mask] = np.nan
    # Non-monotonic: snow appears at some date after an earlier snow-free date
    _gained = (_snow[1:] & ~_snow[:-1]) & _mask
    _nonmono = _gained.any(0)

    _fig, (_ax_map, _ax_nm, _ax_frac) = plt.subplots(
        1, 3, figsize=(15, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 1]}
    )
    _cols = np.vstack([[[0.85, 0.72, 0.55, 1.0]], date_colors(_T)])  # tan = never snow
    _cm = ListedColormap(_cols)
    _norm = BoundaryNorm(np.arange(-1.5, _T), _cm.N)
    _im = _ax_map.imshow(_last, cmap=_cm, norm=_norm, extent=data["extent_km"], interpolation="nearest")
    _cb = _fig.colorbar(_im, ax=_ax_map, ticks=np.arange(-1, _T), shrink=0.85)
    _cb.ax.set_yticklabels(["never"] + list(_dates), fontsize=8)
    style_map_axes(_ax_map, "Last snow-covered date")

    _nm_img = np.where(_mask, _nonmono.astype(float), np.nan)
    _ax_nm.imshow(_nm_img, cmap=ListedColormap(["#d9d9d9", "#c0392b"]), vmin=0, vmax=1, extent=data["extent_km"], interpolation="nearest")
    style_map_axes(_ax_nm, f"Non-monotonic pixels (red): {100 * _nonmono.sum() / _mask.sum():.1f}%")
    _ax_nm.set_ylabel("")

    _frac = [_snow[_t][_mask].mean() for _t in range(_T)]
    _gain_frac = [0.0] + [_gained[_t][_mask].mean() for _t in range(_T - 1)]
    _x = np.arange(_T)
    _ax_frac.plot(_x, _frac, marker="o", ms=7, lw=2, color="#2f5d95", label="snow-covered")
    _ax_frac.plot(_x, _gain_frac, marker="o", ms=7, lw=2, color="#c0392b", label="newly snow-covered")
    for _xi, _f in zip(_x, _frac):
        _ax_frac.annotate(f"{100 * _f:.0f}%", (_xi, _f), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#333")
    _ax_frac.set_xticks(_x, _dates, rotation=30, fontsize=8)
    _ax_frac.set_ylabel("fraction of catchment")
    _ax_frac.set_ylim(0, 1.05)
    _ax_frac.grid(axis="y", color="#e5e5e5")
    _ax_frac.spines[["top", "right"]].set_visible(False)
    _ax_frac.legend(frameon=False, fontsize=8)
    _ax_frac.set_title("Snow-covered area over time", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    elev_bin = mo.ui.slider(25, 300, step=25, value=100, label="Elevation bin (m)", show_value=True)
    elev_bin
    return (elev_bin,)


@app.cell(hide_code=True)
def _(data, elev_bin):
    _mask = data["mask"]
    _dem = data["dem"][_mask]
    _inc = data["incidence"][_mask]
    _snow = np.nan_to_num(data["sca"][:, _mask]) > 0.5
    _dates = data["dates"]
    _cols = date_colors(len(_dates))

    def _binned(values, edges):
        _idx = np.digitize(values, edges) - 1
        _ok = (_idx >= 0) & (_idx < len(edges) - 1)
        _counts = np.bincount(_idx[_ok], minlength=len(edges) - 1)
        _fracs = np.array([
            np.bincount(_idx[_ok], weights=_s[_ok].astype(float), minlength=len(edges) - 1) / np.maximum(_counts, 1)
            for _s in _snow
        ])
        _fracs[:, _counts < 20] = np.nan  # hide sparsely populated bins
        return 0.5 * (edges[1:] + edges[:-1]), _fracs, _counts

    _e_edges = np.arange(np.floor(np.nanmin(_dem) / elev_bin.value) * elev_bin.value, np.nanmax(_dem) + elev_bin.value, elev_bin.value)
    _e_mid, _e_frac, _e_cnt = _binned(_dem, _e_edges)
    _finite_inc = np.isfinite(_inc)
    _i_edges = np.arange(0, 95, 5.0)
    _i_mid, _i_frac, _ = _binned(np.where(_finite_inc, _inc, -1), _i_edges)

    _fig, (_ax_e, _ax_h, _ax_i) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 0.6, 1.1]}
    )
    for _t, _d in enumerate(_dates):
        _ax_e.plot(_e_frac[_t], _e_mid, lw=2, color=_cols[_t], label=_d)
        _ax_i.plot(_i_mid, _i_frac[_t], lw=2, color=_cols[_t], label=_d)
    _ax_e.set_xlabel("snow-covered fraction")
    _ax_e.set_ylabel("elevation (m)")
    _ax_e.set_title("Snow cover by elevation", fontsize=10)
    _ax_e.legend(frameon=False, fontsize=8, loc="lower right")

    _ax_h.barh(_e_mid, _e_cnt, height=elev_bin.value * 0.9, color="#bdbdbd")
    _ax_h.set_ylim(_ax_e.get_ylim())
    _ax_h.set_yticklabels([])
    _ax_h.set_xlabel("pixels")
    _ax_h.set_title("Hypsometry", fontsize=10)

    _ax_i.set_xlabel("incidence angle (°)")
    _ax_i.set_ylabel("snow-covered fraction")
    _ax_i.set_title("Snow cover by incidence angle", fontsize=10)
    _ax_i.legend(frameon=False, fontsize=8)
    for _ax in (_ax_e, _ax_h, _ax_i):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(color="#eeeeee")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pixel inspector

    Spectral signature and index trajectory at a single pixel. Snow is bright in the visible
    range and dark in the SWIR (B11/B12), which is what the NDSI measures.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    _H, _W = data["mask"].shape
    _rr, _cc = np.nonzero(data["mask"])
    _start = int(np.nanargmax(np.where(data["mask"], data["dem"], -np.inf)))  # highest pixel
    px_row = mo.ui.slider(0, _H - 1, value=_start // _W, label="Row", show_value=True, full_width=True)
    px_col = mo.ui.slider(0, _W - 1, value=_start % _W, label="Column", show_value=True, full_width=True)
    mo.hstack([px_row, px_col], widths="equal", gap=2)
    return px_col, px_row


@app.cell(hide_code=True)
def _(data, px_col, px_row):
    _r, _c = px_row.value, px_col.value
    _dates = data["dates"]
    _cols = date_colors(len(_dates))
    _x0, _, _, _y1 = data["extent_km"]
    _dx, _dy = data["pixel_size"]
    _xk, _yk = _x0 + (_c + 0.5) * _dx / 1e3, _y1 - (_r + 0.5) * _dy / 1e3
    _inside = bool(data["mask"][_r, _c])

    _fig, (_ax_map, _ax_spec, _ax_idx) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.3, 1]}
    )
    _ax_map.imshow(rgb_composite(data, RGB_PRESETS["True colour (B4, B3, B2)"], 0, (1, 99), 1.4), extent=data["extent_km"])
    _ax_map.plot(_xk, _yk, marker="o", ms=10, mfc="none", mec="#c0392b", mew=2)
    style_map_axes(_ax_map, f"({_r}, {_c})  ·  {data['dem'][_r, _c]:.0f} m" if _inside else f"({_r}, {_c})  ·  outside catchment")

    if _inside:
        _wl = np.array([BAND_INFO[b][0] for b in data["bands"]])
        _order = np.argsort(_wl)
        for _t, _d in enumerate(_dates):
            _ax_spec.plot(_wl[_order], data["raw"][_t, _order, _r, _c], marker="o", ms=5, lw=2, color=_cols[_t], label=_d)
        for _b, _w in zip(np.array(data["bands"])[_order], _wl[_order]):
            _ax_spec.annotate(_b, (_w, 0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color="#777")
        _ax_spec.set_xlabel("wavelength (nm)")
        _ax_spec.set_ylabel("reflectance")
        _ax_spec.legend(frameon=False, fontsize=8)

        _xt = np.arange(len(_dates))
        _ax_idx.plot(_xt, data["ndsi"][:, _r, _c], marker="o", ms=7, lw=2, color="#2f5d95", label="NDSI")
        _ax_idx.plot(_xt, data["ndvi"][:, _r, _c], marker="o", ms=7, lw=2, color="#5a8f3c", label="NDVI")
        _ax_idx.axhline(0.4, color="#2f5d95", lw=1, ls="--")
        _ax_idx.annotate("snow threshold", (0, 0.4), xytext=(2, 3), textcoords="offset points", fontsize=7, color="#2f5d95")
        _ax_idx.set_xticks(_xt, _dates, rotation=30, fontsize=8)
        _ax_idx.set_ylim(-1, 1)
        _ax_idx.legend(frameon=False, fontsize=8)
    for _ax in (_ax_spec, _ax_idx):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(color="#eeeeee")
    _ax_spec.set_title("Spectral signature", fontsize=10)
    _ax_idx.set_title("Index trajectory", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Band distributions

    Per-band reflectance histograms across dates. The shift of visible-band mass from
    high (snow) to low reflectance tracks melt.
    """)
    return


@app.cell(hide_code=True)
def _():
    hist_log = mo.ui.checkbox(value=False, label="Log counts")
    hist_log
    return (hist_log,)


@app.cell(hide_code=True)
def _(data, hist_log):
    _dates = data["dates"]
    _cols = date_colors(len(_dates))
    _mask = data["mask"]
    _edges = np.linspace(0, np.nanpercentile(data["raw"], 99.9), 80)
    _fig, _axes = plt.subplots(2, 6, figsize=(16, 5.5), constrained_layout=True, sharex=True)
    for _b, _ax in enumerate(_axes.ravel()):
        for _t, _d in enumerate(_dates):
            _vals = data["raw"][_t, _b][_mask]
            _ax.hist(_vals[np.isfinite(_vals)], bins=_edges, histtype="step", lw=1.5, color=_cols[_t], label=_d, log=hist_log.value)
        _name = data["bands"][_b]
        _ax.set_title(f"{_name} · {BAND_INFO[_name][0]} nm", fontsize=9)
        _ax.tick_params(labelsize=7)
        _ax.spines[["top", "right"]].set_visible(False)
    _axes[0, 0].legend(frameon=False, fontsize=7)
    _fig.supxlabel("reflectance", fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Consistency checks

    Recompute NDSI = (B3 − B11) / (B3 + B11) from the raw bands and compare with the
    provided index. Then check that SCA equals NDSI > 0.4. Large discrepancies would mean the
    derived products were made from differently processed bands.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    _b3 = data["raw"][:, data["bands"].index("B3")]
    _b11 = data["raw"][:, data["bands"].index("B11")]
    with np.errstate(invalid="ignore", divide="ignore"):
        _ndsi_calc = (_b3 - _b11) / (_b3 + _b11)
    _mask = data["mask"]
    _rows = []
    for _t, _d in enumerate(data["dates"]):
        _p, _c = data["ndsi"][_t][_mask], _ndsi_calc[_t][_mask]
        _ok = np.isfinite(_p) & np.isfinite(_c)
        _sca = data["sca"][_t][_mask] > 0.5
        _rows.append({
            "date": _d,
            "NDSI RMSE": float(np.sqrt(np.mean((_p[_ok] - _c[_ok]) ** 2))),
            "NDSI max |Δ|": float(np.max(np.abs(_p[_ok] - _c[_ok]))),
            "SCA == (NDSI>0.4) agreement": f"{100 * np.mean(_sca == (np.nan_to_num(_p) > 0.4)):.3f}%",
            "SCA == (recomputed NDSI>0.4)": f"{100 * np.mean(_sca == (np.nan_to_num(_c) > 0.4)):.3f}%",
        })
    ndsi_recomputed = _ndsi_calc
    mo.ui.table(_rows, selection=None)
    return (ndsi_recomputed,)


@app.cell(hide_code=True)
def _(data, ndsi_recomputed):
    _fig, _axes = plt.subplots(1, len(data["dates"]), figsize=(16, 3.4), constrained_layout=True, sharey=True)
    for _t, _ax in enumerate(_axes):
        _p = data["ndsi"][_t][data["mask"]]
        _c = ndsi_recomputed[_t][data["mask"]]
        _ok = np.isfinite(_p) & np.isfinite(_c)
        _ax.hexbin(_c[_ok], _p[_ok], gridsize=60, bins="log", cmap="Blues", mincnt=1, extent=(-1, 1, -1, 1))
        _ax.plot([-1, 1], [-1, 1], color="#c0392b", lw=1)
        _ax.set_title(data["dates"][_t], fontsize=9)
        _ax.set_xlabel("NDSI from B3/B11")
        _ax.set_aspect("equal")
    _axes[0].set_ylabel("NDSI provided")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NCA-resolution preview

    NCA training in this repository runs on grids much smaller than 517 × 514. This
    block-averages the selected channels (NaN-aware; the mask is carried as a fraction) to show
    how much spatial structure survives at a given training resolution. Each row is one channel
    on a colour scale shared across dates. Static channels (DEM, incidence angle) are shown once.
    The catchment mask (from the DEM) is a single connected domain with no interior holes.
    """)
    return


@app.cell(hide_code=True)
def _():
    nca_factor = mo.ui.dropdown({"1× (517×514)": 1, "2×": 2, "4×": 4, "8×": 8, "16×": 16}, value="4×", label="Downsample")
    nca_channels = mo.ui.multiselect(CHANNELS, value=["SCA", "NDSI", "B3", "B11"], label="Channels")
    mo.hstack([nca_factor, nca_channels], justify="start", gap=1.5, wrap=True)
    return nca_channels, nca_factor


@app.cell(hide_code=True)
def _(data, nca_channels, nca_factor):
    _dates = data["dates"]
    _T = len(_dates)
    _names = [c for c in CHANNELS if c in nca_channels.value]  # keep a stable channel order
    _mask_small = block_mean(data["mask"].astype(np.float32), nca_factor.value)
    _inside = _mask_small > 0.5

    if not _names:
        _out = mo.callout(mo.md("Select at least one channel."), kind="neutral")
    else:
        _fig, _axes = plt.subplots(
            len(_names), _T + 1, figsize=(2.7 * (_T + 1), 2.6 * len(_names)),
            constrained_layout=True, squeeze=False,
        )
        for _r, _name in enumerate(_names):
            _stack, _static, _style, _units = channel_stack(data, _name)
            _cmap = LAYER_STYLE[_style][0]
            _vmin, _vmax = color_limits(_stack, _style, (2, 98))
            _small = block_mean(_stack, nca_factor.value)
            for _t in range(_T):
                _ax = _axes[_r, _t]
                _ax.set_xticks([]); _ax.set_yticks([])
                for _s in _ax.spines.values():
                    _s.set_visible(False)
                if _static and _t > 0:
                    _ax.axis("off")
                    continue
                _im = _ax.imshow(_small if _static else _small[_t], cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation="nearest")
                draw_outline(_ax, _inside)
                _ax.set_facecolor("#e6e6e6")
                if _r == 0 or _static:
                    _ax.set_title("static" if _static else _dates[_t], fontsize=9)
            _axes[_r, 0].set_ylabel(_name, fontsize=10)
            _fig.colorbar(_im, ax=list(_axes[_r, :_T]), shrink=0.9, pad=0.01, label=_units)

            _ax_last = _axes[_r, _T]
            if _r == 0:
                _ax_last.imshow(_mask_small, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
                _ax_last.set_title("mask fraction", fontsize=9)
                _ax_last.set_xticks([]); _ax_last.set_yticks([])
            else:
                _ax_last.axis("off")
        _fig.suptitle(f"{len(_names)} channel(s) at {_mask_small.shape[0]}×{_mask_small.shape[1]}", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Temporal sampling

    The acquisitions are **not evenly spaced in time**. The NCA trainer runs a fixed number of
    steps `t` between consecutive target images (`_run_nca_steps` in `NCA/trainer/trainer.py`).
    If every interval gets the same `t`, the model has to reproduce a 5-day change and a
    20-day change in the same number of steps, i.e. run at different "speeds" in different
    intervals. The panels below show the true spacing, how the melt curve distorts if the
    images are treated as evenly spaced, and how much changes per interval and per day.
    """)
    return


@app.cell(hide_code=True)
def _(data):
    _d64 = np.array(data["dates"], dtype="datetime64[D]")
    days = (_d64 - _d64[0]).astype(int)  # days since first acquisition
    dt_days = np.diff(days)
    steps_per_day = mo.ui.slider(1, 20, step=1, value=4, label="NCA steps per day (for steps ∝ Δt)", show_value=True)
    steps_per_day
    return days, dt_days, steps_per_day


@app.cell(hide_code=True)
def _(data, days, dt_days):
    _mask = data["mask"]
    _dates = data["dates"]
    _snow = np.nan_to_num(data["sca"][:, _mask]) > 0.5
    _frac = _snow.mean(axis=1)
    _melted = (_snow[:-1] & ~_snow[1:]).mean(axis=1)  # fraction of catchment melting per interval
    _mid = 0.5 * (days[1:] + days[:-1])

    _fig, (_ax_tl, _ax_curve, _ax_rate) = plt.subplots(
        3, 1, figsize=(11, 9), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [0.55, 1, 1]}
    )
    _cols = date_colors(len(_dates))

    # (a) Timeline with interval lengths
    for _i in range(len(dt_days)):
        _ax_tl.axvspan(days[_i], days[_i + 1], color="#f2f2f2" if _i % 2 == 0 else "#e4e4e4", lw=0)
        _ax_tl.annotate(f"Δt = {dt_days[_i]} d", (_mid[_i], 0.62), ha="center", fontsize=9, color="#333")
    _ax_tl.scatter(days, np.zeros_like(days), s=80, color=_cols, edgecolors="white", linewidths=2, zorder=3)
    for _x, _d in zip(days, _dates):
        _ax_tl.annotate(_d, (_x, 0), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color="#333")
    _ax_tl.set_ylim(-1, 1)
    _ax_tl.set_yticks([])
    _ax_tl.set_title("Acquisition timeline", fontsize=10)

    # (b) Melt curve in true time vs. the curve implied by uniform spacing (both in days)
    _uniform = np.linspace(days[0], days[-1], len(days))
    _ax_curve.plot(days, 100 * _frac, marker="o", ms=8, lw=2, color="#2f5d95", label="true acquisition times")
    _ax_curve.plot(_uniform, 100 * _frac, marker="o", ms=8, lw=1.5, ls="--", color="#999999",
                   label="if treated as evenly spaced")
    _ax_curve.set_ylabel("snow-covered area (%)")
    _ax_curve.set_ylim(0, 105)
    _ax_curve.legend(frameon=False, fontsize=9)
    _ax_curve.set_title("Snow-covered area: true time vs. uniform spacing", fontsize=10)

    # (c) Melt per interval, normalised by interval length; bar width = interval length
    _rate = 100 * _melted / dt_days
    _ax_rate.bar(days[:-1], _rate, width=dt_days, align="edge", color="#6b8fbf", edgecolor="white", linewidth=2)
    for _i in range(len(dt_days)):
        _ax_rate.annotate(f"{_rate[_i]:.2f} %/d\n({100 * _melted[_i]:.0f}% total)", (_mid[_i], _rate[_i]),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color="#333")
    _ax_rate.set_ylabel("catchment melting (% per day)")
    _ax_rate.set_ylim(0, _rate.max() * 1.3)
    _ax_rate.set_xlabel(f"days since {_dates[0]}")
    _ax_rate.set_title("Melt rate per interval (snow → no snow, normalised by Δt)", fontsize=10)

    for _ax in (_ax_tl, _ax_curve, _ax_rate):
        _ax.spines[["top", "right", "left"] if _ax is _ax_tl else ["top", "right"]].set_visible(False)
    for _ax in (_ax_curve, _ax_rate):
        _ax.set_axisbelow(True)
        _ax.grid(axis="y", color="#eeeeee")
    _fig
    return


@app.cell(hide_code=True)
def _(data, days, dt_days, steps_per_day):
    _mask = data["mask"]
    _dates = data["dates"]
    _snow = np.nan_to_num(data["sca"][:, _mask]) > 0.5
    _ndsi = data["ndsi"][:, _mask]
    _uniform_steps = int(round(steps_per_day.value * dt_days.mean()))
    _rows = []
    for _i, _dt in enumerate(dt_days):
        _melt = (_snow[_i] & ~_snow[_i + 1]).mean()
        _dndsi = np.nanmean(np.abs(_ndsi[_i + 1] - _ndsi[_i]))
        _rows.append({
            "interval": f"{_dates[_i]} → {_dates[_i + 1]}",
            "Δt (days)": int(_dt),
            "melted (% catchment)": round(100 * _melt, 2),
            "melt rate (%/day)": round(100 * _melt / _dt, 3),
            "mean |ΔNDSI|": round(float(_dndsi), 3),
            "mean |ΔNDSI| per day": round(float(_dndsi / _dt), 4),
            "NCA steps (∝ Δt)": int(round(steps_per_day.value * _dt)),
            f"NCA steps (uniform, mean Δt)": _uniform_steps,
        })
    mo.vstack([
        mo.md(f"""
    **Interval summary.** With steps ∝ Δt at {steps_per_day.value} steps/day the rollout covers
    {int(steps_per_day.value * days[-1])} NCA steps in total. The 5-day interval gets a quarter of the
    steps of the 20-day intervals. Under uniform spacing every interval gets {_uniform_steps} steps,
    so the per-step dynamics in the 5-day interval would need to be ~{dt_days.max() / dt_days.min():.0f}× slower.
    """),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _():
    change_norm = mo.ui.radio(["Per interval", "Per day (÷ Δt)"], value="Per day (÷ Δt)", label="ΔNDSI normalisation", inline=True)
    change_norm
    return (change_norm,)


@app.cell(hide_code=True)
def _(change_norm, data, dt_days):
    _dates = data["dates"]
    _delta = np.diff(data["ndsi"], axis=0)
    _per_day = change_norm.value.startswith("Per day")
    if _per_day:
        _delta = _delta / dt_days[:, None, None]
    _m = np.nanpercentile(np.abs(_delta), 99)
    _fig, _axes = plt.subplots(1, len(dt_days), figsize=(3.8 * len(dt_days), 4.2), constrained_layout=True, sharey=True)
    for _i, _ax in enumerate(_axes):
        _im = _ax.imshow(_delta[_i], cmap="RdBu", vmin=-_m, vmax=_m, extent=data["extent_km"], interpolation="nearest")
        draw_outline(_ax, data["mask"], data["extent_km"])
        style_map_axes(_ax, f"{_dates[_i][5:]} → {_dates[_i + 1][5:]}  (Δt = {dt_days[_i]} d)")
        if _i > 0:
            _ax.set_ylabel("")
    _fig.colorbar(_im, ax=list(_axes), shrink=0.8, label="ΔNDSI per day" if _per_day else "ΔNDSI")
    _fig.suptitle("NDSI change between consecutive acquisitions (red = loss of snow signal)", fontsize=10)
    _fig
    return


if __name__ == "__main__":
    app.run()
