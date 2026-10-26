"""Interactive pre-processing experiments for the 260726 micropattern dataset.

Run with:
    marimo edit Experiments/micropatterns/data_cleaning.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns", app_title="data_cleaning")


@app.cell(column=0, hide_code=True)
def _():
    import base64
    import io
    from functools import lru_cache
    from pathlib import Path

    import anywidget
    import ml_dtypes
    import tifffile
    import traitlets
    import yaml

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage as ndi
    from matplotlib.patches import Wedge

    from Common.dataloader.alignment import (
        ColonyAlignment,
        block_centre,
        coverage_map,
        fit_colony_centre,
        fit_score,
        grid_position,
        load_colony_alignment,
        save_colony_alignment,
    )
    from Common.dataloader.background import rolling_ball_background, subtract_background
    from Common.dataloader.cell_type_fit import (
        BANDS,
        band_margins,
        cell_type_score,
        colony_share,
        coordinate_search,
        radial_bands,
        radial_rings,
        target_score,
        threshold_grid,
    )
    from Common.dataloader.cell_type_shares import share_estimator
    from Common.dataloader.cell_types import (
        CellTypeRules,
        classify_within_stain,
        label_names,
        load_cell_type_rules,
        save_cell_type_rules,
    )
    from Common.dataloader.hot_pixels import replace_hot_pixels_per_channel
    from Common.dataloader.micropattern_260726 import (
        build_micropattern_260726_manifest,
        circular_colony_mask,
        load_micropattern_260726,
        read_micropattern_260726_image,
        source_condition,
    )
    from Common.dataloader.micropattern_cleaning import MicropatternCleaningConfig
    from Common.dataloader.micropattern_schemas import (
        DEFAULT_260726_HISTOGRAM_PERCENTILES,
        DEFAULT_260726_INTENSITY_FACTORS,
        MICROPATTERN_260726_SCHEMA,
    )
    from Common.dataloader.normalisation import (
        KNOCKOUT_BOUNDS,
        NORMALISATION_MODES,
        condition_bins,
        rescale,
        trajectory_bins,
    )
    from Common.dataloader.quality_flags import load_quality_flags, save_quality_flags
    from Common.dataloader.stripes import remove_stripes, remove_stripes_per_channel

    return (
        BANDS,
        CellTypeRules,
        ColonyAlignment,
        DEFAULT_260726_HISTOGRAM_PERCENTILES,
        DEFAULT_260726_INTENSITY_FACTORS,
        KNOCKOUT_BOUNDS,
        MICROPATTERN_260726_SCHEMA,
        MicropatternCleaningConfig,
        NORMALISATION_MODES,
        Path,
        Wedge,
        anywidget,
        band_margins,
        base64,
        block_centre,
        build_micropattern_260726_manifest,
        cell_type_score,
        circular_colony_mask,
        classify_within_stain,
        colony_share,
        condition_bins,
        coordinate_search,
        coverage_map,
        fit_colony_centre,
        fit_score,
        grid_position,
        io,
        label_names,
        load_cell_type_rules,
        load_colony_alignment,
        load_micropattern_260726,
        load_quality_flags,
        lru_cache,
        ml_dtypes,
        mo,
        ndi,
        np,
        plt,
        radial_bands,
        radial_rings,
        read_micropattern_260726_image,
        remove_stripes,
        remove_stripes_per_channel,
        replace_hot_pixels_per_channel,
        rescale,
        rolling_ball_background,
        save_cell_type_rules,
        save_colony_alignment,
        save_quality_flags,
        share_estimator,
        source_condition,
        subtract_background,
        target_score,
        threshold_grid,
        tifffile,
        traitlets,
        trajectory_bins,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data_cleaning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    A place to try out pre-processing of the raw 260726 micropattern images.
    Each image goes through these steps, in order:

    1. **Intensity factors**: multiply each channel at each timestep by a
       factor, to correct imaging artifacts (the 0h column holds the existing
       `initial_intensity_scales`).
    2. **Hot pixels**: replace isolated bright specks (smaller than a
       nucleus) by the mean of their neighbours, at full resolution.
    3. **Background removal**: subtract a rolling-ball background, with a
       separate ball radius for each channel (radius 0 = off).
    4. **Circular mask**: fit a circle to the colony (or draw one by hand),
       optionally centre the colony, and zero everything outside the circle.
    5. **Normalisation**: rescale each channel linearly to [0, 1] between a
       low and a high percentile, clipping values outside. The percentiles
       are taken over all timesteps of a trajectory at once, so every
       timestep shares one scale. They can be found per replicate, or per
       channel by averaging the replicates' values.

    The middle column shows one image at full resolution and updates live.
    The right column processes the whole selection at the training
    resolution (downsampling 4).

    **To train with these settings**: copy the **Training settings** block
    (left column) into the micropattern config, and export the colony
    centres under **Export and load alignment** (middle column). The
    training loader then processes the images exactly as shown here;
    **Compare with the training loader** checks this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Select raw images
    """)
    return


@app.cell(hide_code=True)
def _(MICROPATTERN_260726_SCHEMA, mo):
    root = mo.ui.text(
        value="../Data/260726_nca_dataset", label="Dataset root", full_width=True
    )
    conditions = mo.ui.multiselect(
        options={
            "Control": "ctrl",
            "Nodal knockout at 0h": "sl0",
            "Nodal knockout at 24h": "sl24",
        },
        value=["Control"],
        label="Conditions",
    )
    # Version of the confocal cell-fate and protein response stacks; RNA
    # images are the same for all three.
    projection = mo.ui.dropdown(
        options={"Max projection": "max", "Sum projection": "sum", "Single slice": "slice"},
        value="Max projection",
        label="Projection",
    )
    experiment_groups = mo.ui.multiselect(
        options={
            "Cell fate (stain 1)": "cell_fate_s1",
            "Cell fate (stain 2)": "cell_fate_s2",
            "RNA expression": "rna_expression",
            "Protein response": "protein_response",
        },
        value=[
            "Cell fate (stain 1)",
            "Cell fate (stain 2)",
            "RNA expression",
            "Protein response",
        ],
        label="Experiment groups",
    )
    timesteps = mo.ui.text(value="0,12,24,36,48", label="Timesteps (hours)")
    replicate_count = mo.ui.slider(1, 6, value=4, label="Replicates", show_value=True)
    # Left out of the views, the bounds, the cell types and the exported
    # training settings (data.micropattern.excluded_measurements).
    excluded_channels = mo.ui.multiselect(
        options=list(MICROPATTERN_260726_SCHEMA.measurement_names),
        value=["cell_fate_s2/SOX17"],
        label="Excluded channels",
    )
    load_button = mo.ui.run_button(label="Load raw images")
    mo.vstack(
        [
            root,
            mo.hstack([conditions, timesteps, replicate_count]),
            mo.hstack([projection, experiment_groups, excluded_channels], justify="start"),
            load_button,
        ]
    )
    return (
        conditions,
        excluded_channels,
        experiment_groups,
        load_button,
        projection,
        replicate_count,
        root,
        timesteps,
    )


@app.cell(hide_code=True)
def _(
    Path,
    build_micropattern_260726_manifest,
    conditions,
    excluded_channels,
    experiment_groups,
    load_button,
    mo,
    np,
    projection,
    read_micropattern_260726_image,
    replicate_count,
    root,
    tifffile,
    timesteps,
):
    mo.stop(
        not load_button.value,
        mo.md("Click **Load raw images** to read the selection."),
    )
    _conditions = tuple(conditions.value) or ("ctrl",)
    _timesteps = tuple(
        int(_value) for _value in timesteps.value.split(",") if _value.strip()
    )
    _inventory = build_micropattern_260726_manifest(
        root.value,
        conditions=_conditions,
        projection=projection.value,
        timesteps=_timesteps,
        replicate_count=replicate_count.value,
        experiment_groups=tuple(experiment_groups.value) or None,
    )
    selection = {
        "root": Path(root.value).expanduser().resolve(),
        "conditions": _conditions,
        "projection": projection.value,
        "timesteps": _timesteps,
        "replicates": _inventory["replicate_indices"],
        "groups": tuple(experiment_groups.value) or None,
        "excluded": tuple(excluded_channels.value),
    }
    # Raw images are kept as uint16 when that is exact (whole numbers in
    # [0, 65535]), to halve memory use, and as float32 otherwise (e.g. sum
    # projections, which exceed the uint16 range).
    raw_images = []
    for _record in mo.status.progress_bar(
        _inventory["records"], title="Reading raw images", remove_on_exit=True
    ):
        _channels, _foreground = read_micropattern_260726_image(_record)
        _exact = (
            _channels.min() >= 0
            and _channels.max() <= 65535
            and np.array_equal(_channels, np.rint(_channels))
        )
        with tifffile.TiffFile(_record.path) as _tiff:
            _stored = str(_tiff.series[0].dtype)
        raw_images.append(
            {
                "record": _record,
                "channels": _channels.astype(np.uint16 if _exact else np.float32),
                "foreground": _foreground,
                "stored_dtype": _stored,
            }
        )
    mo.md(f"Loaded **{len(raw_images)}** raw images.")
    return raw_images, selection


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bit depth
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bit_depth_run = mo.ui.run_button(label="Check bit depth")
    mo.vstack(
        [
            mo.md(
                "- Error from casting to 16 bits, divided by the image's own "
                "pixel noise (robust estimate from the residual after a 3 x 3 "
                "median). Below 0.1 (✓) the cast adds under 1% to the noise "
                "variance.\n"
                "- **uint16**: raw images at full resolution; images above "
                "65535 are divided by max / 65535 first.\n"
                "- **bfloat16 / float16**: cleaned images (section 2 settings) "
                "at the training resolution, rescaled to [0, 1] between the "
                "channel's low and high % of each image.\n"
                "- Worst case over the loaded images. Cleans every image, so "
                "takes a few minutes the first time.\n"
                "- `system.precision: bfloat16` only sets the matmul "
                "precision; training data stays float32."
            ),
            bit_depth_run,
        ]
    )
    return (bit_depth_run,)


@app.cell(hide_code=True)
def _(
    GROUP_CHANNELS,
    bit_depth_run,
    clean_channel_with,
    cleaning_settings,
    high_percentiles,
    low_percentiles,
    ml_dtypes,
    mo,
    ndi,
    np,
    raw_images,
):
    mo.stop(not bit_depth_run.value, mo.md("Press **Check bit depth** to run this check."))

    def bit_depth_noise(image):
        """Robust pixel noise level: scaled median absolute deviation of image - 3 x 3 median."""
        residual = image - ndi.median_filter(image, size=3, mode="reflect")
        level = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        return level if level > 0 else float(np.std(residual))

    def bit_depth_normalised(index, name, channel):
        """Cleaned channel at the training resolution, rescaled to [0, 1] with the section 2 percentiles."""
        _, cleaned = clean_channel_with(cleaning_settings, index, name, channel)
        low, high = np.percentile(
            cleaned, (low_percentiles.value[name], high_percentiles.value[name])
        )
        return np.clip((cleaned - low) / max(high - low, 1e-12), 0.0, 1.0).astype(np.float32)

    def _error_ratio(original, cast):
        _error = float(np.std(cast.astype(np.float32) - original))
        _level = bit_depth_noise(original)
        if _error == 0:
            return 0.0
        return _error / _level if _level > 0 else float("inf")

    bit_depth_rows = []
    for _index, _entry in enumerate(
        mo.status.progress_bar(raw_images, title="Checking bit depth", remove_on_exit=True)
    ):
        for _channel, _name in enumerate(GROUP_CHANNELS[_entry["record"].group]):
            _raw = _entry["channels"][..., _channel].astype(np.float32)
            _step = max(1.0, float(_raw.max()) / 65535)
            _as_uint16 = np.rint(_raw / _step).astype(np.uint16) * np.float32(_step)
            _normalised = bit_depth_normalised(_index, _name, _channel)
            bit_depth_rows.append(
                {
                    "index": _index,
                    "channel": _name,
                    "stored": _entry["stored_dtype"],
                    "max": float(_raw.max()),
                    "integer": bool(np.array_equal(_raw, np.rint(_raw))),
                    "above": float(np.mean(_raw > 65535)),
                    "uint16": _error_ratio(_raw, _as_uint16),
                    "bfloat16": _error_ratio(_normalised, _normalised.astype(ml_dtypes.bfloat16)),
                    "float16": _error_ratio(_normalised, _normalised.astype(np.float16)),
                }
            )

    def _mark(ratio):
        return f"{ratio:.3f} {'✓' if ratio < 0.1 else '✗'}"

    _summary = []
    for _name in dict.fromkeys(_row["channel"] for _row in bit_depth_rows):
        _rows = [_row for _row in bit_depth_rows if _row["channel"] == _name]
        _summary.append(
            {
                "channel": _name,
                "stored as": ", ".join(sorted({_row["stored"] for _row in _rows})),
                "max": f"{max(_row['max'] for _row in _rows):.0f}",
                "whole numbers": all(_row["integer"] for _row in _rows),
                "% above 65535": f"{100 * max(_row['above'] for _row in _rows):.2f}",
                "uint16": _mark(max(_row["uint16"] for _row in _rows)),
                "bfloat16": _mark(max(_row["bfloat16"] for _row in _rows)),
                "float16": _mark(max(_row["float16"] for _row in _rows)),
            }
        )
    mo.ui.table(_summary, selection=None)
    return bit_depth_noise, bit_depth_normalised, bit_depth_rows


@app.cell(hide_code=True)
def _(bit_depth_rows, mo):
    _names = list(dict.fromkeys(_row["channel"] for _row in bit_depth_rows))
    bit_depth_channel = mo.ui.dropdown(options=_names, value=_names[0], label="Channel")
    mo.vstack(
        [
            mo.md("- The image of this channel with the largest bfloat16 error."),
            bit_depth_channel,
        ]
    )
    return (bit_depth_channel,)


@app.cell(hide_code=True)
def _(
    GROUP_CHANNELS,
    bit_depth_channel,
    bit_depth_noise,
    bit_depth_normalised,
    bit_depth_rows,
    ml_dtypes,
    np,
    plt,
    raw_images,
    record_label,
    show_figure,
):
    _worst = max(
        (_row for _row in bit_depth_rows if _row["channel"] == bit_depth_channel.value),
        key=lambda _row: _row["bfloat16"],
    )
    _record = raw_images[_worst["index"]]["record"]
    _normalised = bit_depth_normalised(
        _worst["index"], _worst["channel"], GROUP_CHANNELS[_record.group].index(_worst["channel"])
    )
    _error = _normalised.astype(ml_dtypes.bfloat16).astype(np.float32) - _normalised
    _noise = bit_depth_noise(_normalised)
    _limit = max(float(np.abs(_error).max()), 1e-9)

    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _axes[0].imshow(_normalised, cmap="magma", vmin=0, vmax=1)
    _axes[0].set_title("normalised (float32)", fontsize=9)
    _axes[1].imshow(_error, cmap="RdBu_r", vmin=-_limit, vmax=_limit)
    _axes[1].set_title("bfloat16 - float32", fontsize=9)
    for _axis in _axes[:2]:
        _axis.set_xticks([])
        _axis.set_yticks([])
    _axes[2].hist(_error.ravel(), bins=101, color="grey")
    for _sign in (-1, 1):
        _axes[2].axvline(_sign * _noise, color="C3", linestyle="--", linewidth=0.8)
    _axes[2].set_yscale("log")
    _axes[2].set_title("bfloat16 error (red: ± pixel noise)", fontsize=9)
    _figure.suptitle(
        f"{_worst['channel']}, {record_label(_record)}: error / noise = {_worst['bfloat16']:.3f}",
        fontsize=10,
    )
    _figure.tight_layout()
    show_figure(_figure, dpi=120)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Per-channel controls
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_260726_HISTOGRAM_PERCENTILES, MICROPATTERN_260726_SCHEMA, mo):
    _names = MICROPATTERN_260726_SCHEMA.measurement_names
    shrink = mo.ui.dropdown(
        options={"1 (exact, slow)": 1, "2": 2, "4": 4, "8": 8},
        value="4",
        label="Background shrink factor",
    )
    hot_pixel_size = mo.ui.dropdown(
        options={"3": 3, "5": 5, "7": 7, "9": 9},
        value="5",
        label="Hot pixel window (pixels)",
    )
    stripe_block = mo.ui.dropdown(
        options={"whole rows": 0, "540": 540, "270": 270, "128": 128, "64": 64},
        value="whole rows",
        label="Stripe block (pixels)",
    )
    # LMBR is the structural stain: by default it gets no hot pixel
    # replacement or background removal. SMAD23 gets hot pixel replacement
    # but no background removal. Every other channel gets both.
    _marker = {
        _channel.name: _channel.marker
        for _channel in MICROPATTERN_260726_SCHEMA.measurement_channels
    }
    _is_lmbr = {_name: _marker[_name] == "LMBR" for _name in _names}
    _no_background = {_name: _marker[_name] in ("LMBR", "SMAD23", "LEFTY", "CER1", "NODAL") for _name in _names}
    hot_pixel_thresholds = mo.ui.dictionary(
        {
            _name: mo.ui.number(0.0, 100.0, value=0.0 if _is_lmbr[_name] else 30.0, step=0.5)
            for _name in _names
        }
    )
    background_radii = mo.ui.dictionary(
        {
            _name: mo.ui.slider(
                0, 400, step=10, value=0 if _no_background[_name] else 10, show_value=True
            )
            for _name in _names
        }
    )
    # Stitching band removal, off by default (largest band width, full-res pixels).
    h_stripe_widths = mo.ui.dictionary(
        {_name: mo.ui.number(0, 200, value=0, step=5) for _name in _names}
    )
    v_stripe_widths = mo.ui.dictionary(
        {_name: mo.ui.number(0, 200, value=0, step=5) for _name in _names}
    )
    low_percentiles = mo.ui.dictionary(
        {
            _name: mo.ui.number(
                0.0, 99.0, value=DEFAULT_260726_HISTOGRAM_PERCENTILES[0], step=0.5
            )
            for _name in _names
        }
    )
    high_percentiles = mo.ui.dictionary(
        {
            _name: mo.ui.number(
                1.0, 100.0, value=DEFAULT_260726_HISTOGRAM_PERCENTILES[1], step=0.05
            )
            for _name in _names
        }
    )
    _widths = [3, 1.3, 1.1, 1.1, 3, 1.1, 1.1]
    _table = mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**channel**"),
                    mo.md("**hot pixel threshold** (0 = off)"),
                    mo.md("**h stripe width** (0 = off)"),
                    mo.md("**v stripe width** (0 = off)"),
                    mo.md("**ball radius** (0 = off)"),
                    mo.md("**low %**"),
                    mo.md("**high %**"),
                ],
                widths=_widths,
            )
        ]
        + [
            mo.hstack(
                [
                    mo.md(f"`{_name}`"),
                    hot_pixel_thresholds[_name],
                    h_stripe_widths[_name],
                    v_stripe_widths[_name],
                    background_radii[_name],
                    low_percentiles[_name],
                    high_percentiles[_name],
                ],
                widths=_widths,
                align="center",
            )
            for _name in _names
        ]
    )
    mo.vstack(
        [
            mo.md(
                "- **Hot pixel threshold**: a pixel is replaced by the mean of "
                "its other neighbours when it is more than this many noise "
                "levels above the median of its **window**. The noise level "
                "is a robust estimate of the pixel-to-pixel spread in that "
                "image. Make the window wider than the specks but narrower "
                "than a nucleus. Lower thresholds catch more pixels. Runs at "
                "full resolution (about 0.4 s per channel).\n"
                "- **Defaults**: LMBR channels (the structural stain) get no hot "
                "pixel replacement and no background removal. SMAD23 gets hot "
                "pixel replacement but no background removal. Every other "
                "channel starts with hot pixel threshold 30 and ball radius 50.\n"
                "- **Ball radius** is in full-resolution pixels (images are "
                "1080 x 1080). Anything narrower than the ball is kept, and "
                "slowly varying light underneath is removed. Pick a radius a "
                "little larger than the widest feature to keep; if it is too "
                "small, the middle of broad bright regions is removed too. "
                "**Shrink** estimates the background on a block-averaged "
                "copy, which is much faster and usually looks the same.\n"
                "- **Intensity factors** for each channel and timestep are set "
                "in the table below, once images are loaded.\n"
                "- **low % / high %** are the percentiles mapped to 0 and 1 in "
                "the normalisation (section 5). The defaults are the loader's, "
                "which takes them over whole images; inside the mask a lower "
                "low % is often enough."
            ),
            mo.hstack([hot_pixel_size, shrink, stripe_block], justify="start"),
            _table,
            mo.md(
                "- **h / v stripe width**: removes horizontal / vertical bands of "
                "raised background (e.g. stitching seams in sum projections) up "
                "to this width, in full-resolution pixels. Runs after hot pixels, "
                "before the ball. See **Stripe removal comparison**.\n"
                "- **Stripe block**: for seams covering only part of a row (one "
                "tile), the background is estimated in pieces of this length. "
                "Smaller pieces follow seams more closely but are more likely "
                "to take cell texture for a stripe."
            ),
        ]
    )
    return (
        background_radii,
        h_stripe_widths,
        high_percentiles,
        hot_pixel_size,
        hot_pixel_thresholds,
        low_percentiles,
        shrink,
        stripe_block,
        v_stripe_widths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Intensity factors
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_260726_INTENSITY_FACTORS, loaded_names, mo, selection):
    _hours = selection["timesteps"]
    intensity_scales = mo.ui.dictionary(
        {
            _name: mo.ui.dictionary(
                {
                    f"{_hour}h": mo.ui.number(
                        0.0,
                        5.0,
                        value=DEFAULT_260726_INTENSITY_FACTORS.get(_name, {}).get(_hour, 1.0),
                        step=0.005,
                    )
                    for _hour in _hours
                }
            )
            for _name in loaded_names
        }
    )
    _widths = [3] + [1] * len(_hours)
    mo.vstack(
        [
            mo.md(
                "Each factor multiplies the raw intensities of one channel at "
                "one timestep, for every condition, before anything else. Use "
                "it to correct timesteps whose overall brightness is off "
                "because of imaging artifacts. A factor of 1 changes nothing. "
                "The factors start from the defaults "
                "(`DEFAULT_260726_INTENSITY_FACTORS`, the same as "
                "`data.micropattern.intensity_factors` in the base config). The corrected "
                "images also feed the normalisation bounds."
            ),
            mo.hstack(
                [mo.md("**channel**")] + [mo.md(f"**{_hour}h**") for _hour in _hours],
                widths=_widths,
            ),
            *[
                mo.hstack(
                    [mo.md(f"`{_name}`")]
                    + [intensity_scales[_name][f"{_hour}h"] for _hour in _hours],
                    widths=_widths,
                    align="center",
                )
                for _name in loaded_names
            ],
        ]
    )
    return (intensity_scales,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Circular mask
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mask_mode = mo.ui.dropdown(
        options={
            "Fixed-radius template fit (recommended)": "fixed",
            "Fitted to each image": "per_image",
            "One common circle (as the loader)": "common",
            "Manual radius, centred": "manual",
        },
        value="Fixed-radius template fit (recommended)",
        label="Mask",
    )
    pattern_radius_override = mo.ui.number(
        0, 540, value=0, step=5, label="Pattern radius (full-res pixels, 0 = estimate)"
    )
    coverage_blur = mo.ui.slider(
        0, 40, step=2, value=0, label="Coverage blur (full-res pixels)", show_value=True
    )
    edge_width = mo.ui.slider(
        0.02, 0.2, step=0.01, value=0.08, label="Edge width (fraction of radius)", show_value=True
    )
    radius_quantile = mo.ui.slider(
        0.80, 1.0, step=0.005, value=0.98, label="Radius quantile", show_value=True
    )
    radius_scale = mo.ui.slider(
        0.5, 1.2, step=0.01, value=1.0, label="Radius scale", show_value=True
    )
    manual_radius = mo.ui.slider(
        100, 540, step=5, value=400, label="Manual radius (full-res pixels)", show_value=True
    )
    align = mo.ui.checkbox(value=True, label="Centre each colony (as the loader)")
    mo.vstack(
        [
            mo.md(
                "**Fixed-radius template fit** (default). The micropattern has "
                "the same size in every image, so all images share one "
                "**pattern radius**. It is estimated as the median circle "
                "radius (see below) over the latest loaded timesteps (36h and "
                "later), where the colony fills the pattern, unless you set it. "
                "Only the centre is fitted per image. The colony pixels (at "
                "the training resolution) are matched against a disk of the "
                "pattern radius, with a penalty for colony pixels in a ring "
                "just outside it (**edge width**). **Coverage blur** "
                "optionally smooths the colony pixels first; it is off by "
                "default, because blur spreads the dense side of a colony "
                "outwards and pulls the centre towards it. The best match puts "
                "the disk edge at the outer edge of the cells, so a denser "
                "side or a sparse early timepoint does not pull the centre or "
                "shrink the circle. Each image (one experiment group, timestep "
                "and replicate) is fitted from its own structural stain, and "
                "all its channels move together. With centring on, every "
                "image's mask is then the same centred disk, of the pattern "
                "radius times the **radius scale**. Check the fits under "
                "**Alignment check** in the middle column.\n\n"
                "The other modes fit a circle to the colony pixels of each image: its "
                "centre is their median position, and its radius is the "
                "**radius quantile** of their distances from the centre (this "
                "ignores stray bright pixels far away), times the **radius "
                "scale**. A scale below 1 trims the colony edge.\n\n"
                "- *Fitted to each image*: every image uses its own circle.\n"
                "- *One common circle*: as in training. Each trajectory "
                "contributes the circle of its last cell-fate (stain 1) image; "
                "pixels inside at least half of these circles form the mask.\n"
                "- *Manual radius*: one circle of the chosen radius, centred "
                "on the image (use with centring on).\n\n"
                "Centring shifts each image so its fitted circle sits in the "
                "middle. Pixels outside the mask can be set to zero in section 5."
            ),
            mask_mode,
            mo.hstack([pattern_radius_override, coverage_blur, edge_width], justify="start"),
            mo.hstack([radius_quantile, radius_scale], justify="start"),
            mo.hstack([manual_radius, align], justify="start"),
        ]
    )
    return (
        align,
        coverage_blur,
        edge_width,
        manual_radius,
        mask_mode,
        pattern_radius_override,
        radius_quantile,
        radius_scale,
    )


@app.cell(hide_code=True)
def _(
    MICROPATTERN_260726_SCHEMA,
    io,
    mo,
    ndi,
    np,
    plt,
    remove_stripes_per_channel,
    replace_hot_pixels_per_channel,
    subtract_background,
):
    # Downsampling used for the whole-selection images, matching training.
    DISPLAY_DOWNSAMPLE = 4
    # Measurement names of each experiment group, in the loader's channel order.
    GROUP_CHANNELS = {
        _group.name: tuple(_channel.name for _channel in _group.channels)
        for _group in MICROPATTERN_260726_SCHEMA.experiment_groups
    }
    # Measurement name -> (experiment group, channel index within that group).
    CHANNEL_SOURCE = {
        _name: (_group, _index)
        for _group, _names in GROUP_CHANNELS.items()
        for _index, _name in enumerate(_names)
    }
    CONDITION_COLOURS = {"ctrl": "C0", "sl0": "C1", "sl24": "C2"}

    def clean_image(entry, settings):
        """Run the full-resolution cleaning steps on one raw image.

        ``settings`` holds the per-channel dictionaries ``scales`` (intensity
        factors per timestep), ``hot_thresholds``, ``h_stripes``, ``v_stripes``
        and ``radii``, plus ``hot_size``, ``stripe_block`` and ``shrink``. Returns float32
        ``[X, Y, C]`` arrays after each step: ``scaled`` (intensity factor),
        ``despeckled`` (hot pixels replaced), ``destriped``, ``background`` and
        ``cleaned``, plus the boolean ``hot`` map.
        """
        record = entry["record"]
        names = GROUP_CHANNELS[record.group]
        scaled = entry["channels"].astype(np.float32) * np.array(
            [intensity_factor(settings["scales"], name, record.timestep) for name in names],
            dtype=np.float32,
        )
        despeckled, hot = replace_hot_pixels_per_channel(
            scaled,
            [settings["hot_thresholds"].get(name, 0) for name in names],
            settings["hot_size"],
        )
        destriped, _ = remove_stripes_per_channel(
            despeckled,
            [settings["h_stripes"].get(name, 0) for name in names],
            [settings["v_stripes"].get(name, 0) for name in names],
            settings["stripe_block"],
        )
        cleaned, background = subtract_background(
            destriped, [settings["radii"].get(name, 0) for name in names], settings["shrink"]
        )
        return {
            "scaled": scaled,
            "hot": hot,
            "despeckled": despeckled,
            "destriped": destriped,
            "background": background,
            "cleaned": cleaned,
        }

    def colony_statistics(image, foreground):
        """Per-channel 99th percentile on the colony and median off it."""
        return (
            np.percentile(image[foreground], 99, axis=0),
            np.median(image[~foreground], axis=0),
        )

    def block_mean(image, factor=DISPLAY_DOWNSAMPLE):
        """Average ``factor x factor`` blocks of an ``[X, Y, ...]`` array."""
        height = image.shape[0] // factor
        width = image.shape[1] // factor
        image = image[: height * factor, : width * factor]
        return image.reshape(height, factor, width, factor, *image.shape[2:]).mean(
            axis=(1, 3)
        )

    def intensity_factor(scales, name, timestep):
        """Factor for one channel at one timestep from the intensity factor table."""
        return scales.get(name, {}).get(f"{timestep}h", 1.0)

    def show_figure(figure, dpi=90):
        """Display a figure as a PNG rendered once at ``dpi``, and close it.

        marimo otherwise renders figures at screen resolution (four times
        the pixels on high-DPI screens). Grids of fluorescence images then
        become tens of megabytes, too large to display reliably.
        """
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi)
        plt.close(figure)
        return mo.image(buffer.getvalue())

    def outline_flagged(axis):
        """Draw a red border around a panel showing a low-quality image."""
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("red")
            spine.set_linewidth(3)

    def place_image(values, shift):
        """Shift ``[X, Y, ...]`` values by ``(rows, columns)``; ``None`` leaves them in place."""
        if shift is None:
            return values
        return ndi.shift(
            values,
            (*shift, *([0] * (values.ndim - 2))),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    def centred_disk(shape, radius):
        rows, columns = np.ogrid[: shape[0], : shape[1]]
        centre = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
        return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius**2

    def record_label(record):
        return (
            f"{record.group} / {record.condition} / {record.timestep}h / "
            f"replicate {record.replicate + 1}"
        )

    def trajectory_label(trajectory):
        condition, replicate = trajectory
        return f"{condition}, replicate {replicate + 1}"

    return (
        CHANNEL_SOURCE,
        CONDITION_COLOURS,
        DISPLAY_DOWNSAMPLE,
        GROUP_CHANNELS,
        block_mean,
        centred_disk,
        clean_image,
        colony_statistics,
        intensity_factor,
        outline_flagged,
        place_image,
        record_label,
        show_figure,
        trajectory_label,
    )


@app.cell(hide_code=True)
def _(
    background_radii,
    h_stripe_widths,
    hot_pixel_size,
    hot_pixel_thresholds,
    intensity_scales,
    shrink,
    stripe_block,
    v_stripe_widths,
):
    cleaning_settings = {
        "scales": intensity_scales.value,
        "hot_thresholds": hot_pixel_thresholds.value,
        "hot_size": hot_pixel_size.value,
        "h_stripes": h_stripe_widths.value,
        "v_stripes": v_stripe_widths.value,
        "stripe_block": stripe_block.value,
        "radii": background_radii.value,
        "shrink": shrink.value,
    }
    return (cleaning_settings,)


@app.cell(hide_code=True)
def _(
    block_mean,
    intensity_factor,
    lru_cache,
    np,
    raw_images,
    remove_stripes_per_channel,
    replace_hot_pixels_per_channel,
    rolling_ball_background,
    subtract_background,
):
    # Cleaning of single channels for the live views. Results are cached per
    # image, channel and setting, so changing one channel's controls only
    # recomputes that channel. The cache is rebuilt when images are reloaded.
    # It holds every channel of every loaded image twice over: the cell-type
    # view cleans several markers across all trajectories in turn, and a
    # smaller cache drops each marker's images before they are reused.
    _cache_size = 2 * sum(_entry["channels"].shape[-1] for _entry in raw_images)

    @lru_cache(maxsize=max(_cache_size, 128))
    def clean_channel(
        index, channel, scale, hot_threshold, hot_size, h_stripe, v_stripe, block, radius, shrink_factor
    ):
        """Return ``(scaled, cleaned)`` for one channel at the training resolution."""
        scaled = raw_images[index]["channels"][..., channel : channel + 1].astype(np.float32)
        scaled = scaled * np.float32(scale)
        despeckled, _ = replace_hot_pixels_per_channel(scaled, [hot_threshold], hot_size)
        destriped, _ = remove_stripes_per_channel(despeckled, [h_stripe], [v_stripe], block)
        cleaned, _ = subtract_background(destriped, [radius], shrink_factor)
        return block_mean(scaled[..., 0]), block_mean(cleaned[..., 0])

    def despeckle_settings(settings, index, name):
        """``(scale, hot_threshold, hot_size, h_stripe, v_stripe, block)`` for one channel of one image."""
        timestep = raw_images[index]["record"].timestep
        return (
            intensity_factor(settings["scales"], name, timestep),
            settings["hot_thresholds"].get(name, 0),
            settings["hot_size"],
            settings["h_stripes"].get(name, 0),
            settings["v_stripes"].get(name, 0),
            settings["stripe_block"],
        )

    def clean_channel_with(settings, index, name, channel):
        """``clean_channel`` with the per-channel values taken from ``settings``."""
        return clean_channel(
            index,
            channel,
            *despeckle_settings(settings, index, name),
            settings["radii"].get(name, 0),
            settings["shrink"],
        )

    # Full-resolution versions for comparing ball sizes on one image.
    @lru_cache(maxsize=12)
    def despeckled_channel(
        index, channel, scale, hot_threshold, hot_size, h_stripe=0, v_stripe=0, block=0
    ):
        """One channel after the intensity factor, hot pixel and stripe removal, full resolution."""
        image = raw_images[index]["channels"][..., channel : channel + 1].astype(np.float32)
        image, _ = replace_hot_pixels_per_channel(image * np.float32(scale), [hot_threshold], hot_size)
        image, _ = remove_stripes_per_channel(image, [h_stripe], [v_stripe], block)
        return image[..., 0]

    @lru_cache(maxsize=16)
    def ball_background(
        index, channel, scale, hot_threshold, hot_size, h_stripe, v_stripe, block, radius, shrink_factor
    ):
        """Rolling-ball background of ``despeckled_channel``, full resolution."""
        return rolling_ball_background(
            despeckled_channel(
                index, channel, scale, hot_threshold, hot_size, h_stripe, v_stripe, block
            ),
            radius,
            shrink_factor,
        )

    return (
        ball_background,
        clean_channel_with,
        despeckle_settings,
        despeckled_channel,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training settings
    """)
    return


@app.cell(hide_code=True)
def _(
    DEFAULT_260726_HISTOGRAM_PERCENTILES,
    MICROPATTERN_260726_SCHEMA,
    align,
    alignment_path,
    background_radii,
    flags_path,
    h_stripe_widths,
    high_percentiles,
    hot_pixel_size,
    hot_pixel_thresholds,
    image_source,
    inside_only,
    intensity_scales,
    knockout_bounds,
    low_percentiles,
    mask_mode,
    mo,
    normalise_mode,
    radius_scale,
    selection,
    shrink,
    stripe_block,
    v_stripe_widths,
    yaml,
    zero_outside,
):
    # The data.micropattern settings that make the training loader process
    # the images as this notebook does (see Common/dataloader/micropattern_cleaning.py).
    _default = tuple(DEFAULT_260726_HISTOGRAM_PERCENTILES)
    _names = MICROPATTERN_260726_SCHEMA.measurement_names
    training_settings = {
        "projection": selection["projection"],
        "excluded_measurements": list(selection["excluded"]),
        "histogram_percentiles": list(_default),
        "quality_flags_file": flags_path.value,
        "intensity_factors": {
            _name: {int(_hour[:-1]): _value for _hour, _value in _row.items() if _value != 1.0}
            for _name, _row in intensity_scales.value.items()
            if any(_value != 1.0 for _value in _row.values())
        },
        "cleaning": {
            "enabled": True,
            "hot_pixel_window": hot_pixel_size.value,
            "hot_pixel_thresholds": {_n: float(hot_pixel_thresholds.value[_n]) for _n in _names},
            "background_shrink": shrink.value,
            "background_radii": {_n: int(background_radii.value[_n]) for _n in _names},
            "horizontal_stripe_widths": {
                _n: int(h_stripe_widths.value[_n]) for _n in _names if h_stripe_widths.value[_n]
            },
            "vertical_stripe_widths": {
                _n: int(v_stripe_widths.value[_n]) for _n in _names if v_stripe_widths.value[_n]
            },
            "stripe_block": int(stripe_block.value),
            "alignment_file": alignment_path.value,
            "mask_radius_scale": float(radius_scale.value),
            "normalisation": normalise_mode.value,
            "channel_percentiles": {
                _n: [float(low_percentiles.value[_n]), float(high_percentiles.value[_n])]
                for _n in _names
                if (low_percentiles.value[_n], high_percentiles.value[_n]) != _default
            },
            "percentiles_inside_mask": bool(inside_only.value),
            "knockout_bounds": knockout_bounds.value,
        },
    }
    # Notebook-only choices that training does not offer.
    training_mismatches = [
        _text
        for _text, _differs in [
            ("the mask mode is not the fixed-radius template fit", mask_mode.value != "fixed"),
            ("colony centring is off", not align.value),
            ("the images shown are raw, not background removed", image_source.value != "cleaned"),
            ("pixels outside the mask are not zeroed", not zero_outside.value),
        ]
        if _differs
    ]
    _block = yaml.safe_dump({"micropattern": training_settings}, sort_keys=False)
    mo.md(
        "The `data.micropattern` settings that make the training loader "
        "process the images as this notebook does. Paste them into "
        "`Experiments/micropatterns/conf/base_config.yaml` (or a sweep), and "
        "export the alignment file under **Export and load alignment**."
        + (
            "\n\n**Training will differ from the notebook view because "
            + "; ".join(training_mismatches)
            + ".**"
            if training_mismatches
            else ""
        )
        + f"\n\n```yaml\n{_block}```"
    )
    return training_mismatches, training_settings


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect one image
    """)
    return


@app.cell(hide_code=True)
def _(mo, raw_images, record_label):
    record_picker = mo.ui.dropdown(
        options={
            record_label(_entry["record"]): _index
            for _index, _entry in enumerate(raw_images)
        },
        value=record_label(raw_images[0]["record"]),
        label="Image",
        full_width=True,
    )
    inspect_run = mo.ui.run_button(label="Show image")
    mo.vstack([mo.hstack([record_picker, inspect_run], justify="start")])
    return inspect_run, record_picker


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    GROUP_CHANNELS,
    background_radii,
    circular_colony_mask,
    clean_image,
    cleaning_settings,
    colony_fits,
    colony_statistics,
    inspect_run,
    manual_radius,
    mask_mode,
    mo,
    ndi,
    np,
    pattern_radius,
    plt,
    radius_quantile,
    radius_scale,
    raw_images,
    record_picker,
    show_figure,
):
    mo.stop(not inspect_run.value, mo.md("Press **Show image** to draw this view. It does not redraw by itself when the controls change."))
    _entry = raw_images[record_picker.value]
    _group = _entry["record"].group
    _names = GROUP_CHANNELS[_group]
    _steps = clean_image(_entry, cleaning_settings)
    _scaled, _hot = _steps["scaled"], _steps["hot"]
    _background, _cleaned = _steps["background"], _steps["cleaned"]
    _foreground = _entry["foreground"]
    if mask_mode.value == "fixed":
        # Fitted at the training resolution; block i covers full-resolution
        # pixels 4i to 4i + 3, so its centre is at 4i + 1.5.
        _centre = tuple(
            DISPLAY_DOWNSAMPLE * _c + (DISPLAY_DOWNSAMPLE - 1) / 2.0
            for _c in colony_fits[record_picker.value]["centre"]
        )
        _rows, _columns = np.ogrid[: _foreground.shape[0], : _foreground.shape[1]]
        _circle = (_rows - _centre[0]) ** 2 + (_columns - _centre[1]) ** 2 <= (
            pattern_radius * radius_scale.value
        ) ** 2
    else:
        _circle = circular_colony_mask(
            _foreground, _group, radius_quantile.value, radius_scale.value
        )
        _centre = ndi.center_of_mass(_circle)
    if mask_mode.value == "manual":
        # The manual circle is centred on the colony once images are centred.
        _rows, _columns = np.ogrid[: _circle.shape[0], : _circle.shape[1]]
        _circle = (_rows - _centre[0]) ** 2 + (_columns - _centre[1]) ** 2 <= (
            manual_radius.value**2
        )
    # Horizontal profile through the circle centre.
    _row = int(round(_centre[0]))
    _circle_columns = np.nonzero(_circle[_row])[0]

    _figure, _axes = plt.subplots(
        len(_names), 4, figsize=(14, 3.2 * len(_names)), squeeze=False
    )
    for _channel, _name in enumerate(_names):
        _axis_row = _axes[_channel]
        _low, _high = np.percentile(_scaled[..., _channel], (0.5, 99.5))
        _axis_row[0].imshow(_scaled[..., _channel], cmap="magma", vmin=_low, vmax=_high)
        _axis_row[0].contour(_circle, levels=[0.5], colors="cyan", linewidths=0.8)
        _hot_rows, _hot_columns = np.nonzero(_hot[..., _channel])
        _axis_row[0].scatter(
            _hot_columns, _hot_rows, s=12, facecolors="none", edgecolors="lime", linewidths=0.6
        )
        _axis_row[0].set_title(
            f"{_name}\nraw (cyan: mask, green: {_hot_rows.size} hot pixels)", fontsize=8
        )
        # Same colour scale as the raw image, so the two can be compared.
        _axis_row[1].imshow(
            _background[..., _channel], cmap="magma", vmin=_low, vmax=_high
        )
        _axis_row[1].set_title(
            f"background (radius {background_radii.value[_name]})", fontsize=8
        )
        _axis_row[2].imshow(
            _cleaned[..., _channel] * _circle,
            cmap="magma",
            vmin=0,
            vmax=max(np.percentile(_cleaned[..., _channel][_circle], 99.5), 1e-6),
        )
        _axis_row[2].set_title("cleaned, zero outside mask", fontsize=8)
        for _axis in _axis_row[:3]:
            _axis.axhline(_row, color="cyan", linewidth=0.5, alpha=0.5, linestyle=":")
            _axis.set_xticks([])
            _axis.set_yticks([])
        _axis_row[3].plot(_scaled[_row, :, _channel], color="grey", linewidth=0.6, label="raw")
        _axis_row[3].plot(_background[_row, :, _channel], color="C0", label="background")
        _axis_row[3].plot(_cleaned[_row, :, _channel], color="C3", linewidth=0.6, label="cleaned")
        if _circle_columns.size:
            _axis_row[3].axvspan(
                _circle_columns.min(), _circle_columns.max(), color="black", alpha=0.06
            )
        _axis_row[3].set_title("profile along dotted line (shaded = mask)", fontsize=8)
        _axis_row[3].legend(fontsize=7)
    _figure.tight_layout()

    _raw_inside, _raw_outside = colony_statistics(_scaled, _foreground)
    _clean_inside, _clean_outside = colony_statistics(_cleaned, _foreground)
    _rows_table = [
        {
            "channel": _name,
            "hot pixels replaced": int(_hot[..., _channel].sum()),
            "raw: colony p99": round(float(_raw_inside[_channel]), 1),
            "raw: off-colony median": round(float(_raw_outside[_channel]), 1),
            "cleaned: colony p99": round(float(_clean_inside[_channel]), 1),
            "cleaned: off-colony median": round(float(_clean_outside[_channel]), 1),
        }
        for _channel, _name in enumerate(_names)
    ]
    mo.vstack(
        [
            mo.md(
                "Raw is before hot pixel replacement; replaced pixels are "
                "circled in green. Raw and background share a colour scale. "
                "Cleaned (hot pixels replaced, then background removed) has its own "
                "scale starting at zero. The background curve in the profile "
                "should follow the floor of the raw curve, not cut into the "
                "colony signal. Values are raw intensity units after the 0h "
                "factor. The statistics split pixels into colony and off-colony "
                "by brightness of the structural stain, independent of the mask."
            ),
            mo.ui.table(_rows_table, selection=None),
            show_figure(_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alignment check
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fit_flag = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.8, label="Flag fits scoring below", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                "How well the fixed-radius template fit (section 3) found each "
                "colony. The **score** is 1 minus the colony coverage just "
                "outside the fitted disk relative to the coverage inside it, "
                "so it does not depend on cell density. It is near 1 when no "
                "cells lie outside the fitted pattern, and low when the disk "
                "is misplaced, the pattern radius is too small, or the colony "
                "is cut off by the image edge. The averaged images "
                "show the colony coverage after centring, for each experiment "
                "group and timestep, with the mask outline. A sharp circular "
                "edge means the images line up; a blurred or doubled edge "
                "means some do not. Correct single images under **Manual "
                "centring** below."
            ),
            fit_flag,
        ]
    )
    return (fit_flag,)


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    MICROPATTERN_260726_SCHEMA,
    colony_fits,
    coverage_maps,
    estimated_radius,
    fit_flag,
    image_geometry,
    mask_mode,
    mo,
    np,
    pattern_radius,
    place_image,
    plt,
    raw_images,
    selection,
    show_figure,
):
    _shape = coverage_maps[0].shape
    _middle = ((_shape[0] - 1) / 2.0, (_shape[1] - 1) / 2.0)
    _rows = []
    for _entry, _fit in zip(raw_images, colony_fits):
        _record = _entry["record"]
        _rows.append(
            {
                "group": _record.group,
                "condition": _record.condition,
                "timestep": _record.timestep,
                "replicate": _record.replicate + 1,
                "centre offset down": round(DISPLAY_DOWNSAMPLE * (_fit["centre"][0] - _middle[0]), 1),
                "centre offset right": round(DISPLAY_DOWNSAMPLE * (_fit["centre"][1] - _middle[1]), 1),
                "manual offset": _fit["offset"] if _fit["offset"] != (0.0, 0.0) else "",
                "score": round(_fit["score"], 3),
                "flagged": _fit["score"] < fit_flag.value,
            }
        )
    _flagged = sum(_row["flagged"] for _row in _rows)

    # Fitted centres relative to the image centre, coloured by experiment group.
    _groups = [
        _group.name
        for _group in MICROPATTERN_260726_SCHEMA.experiment_groups
        if any(_row["group"] == _group.name for _row in _rows)
    ]
    _offsets, _axis = plt.subplots(figsize=(4.8, 4.2))
    for _number, _group in enumerate(_groups):
        _members = [_row for _row in _rows if _row["group"] == _group]
        _axis.scatter(
            [_row["centre offset right"] for _row in _members],
            [_row["centre offset down"] for _row in _members],
            color=f"C{_number}",
            s=18,
            label=_group,
        )
        _bad = [_row for _row in _members if _row["flagged"]]
        _axis.scatter(
            [_row["centre offset right"] for _row in _bad],
            [_row["centre offset down"] for _row in _bad],
            facecolors="none",
            edgecolors="red",
            s=70,
        )
    _axis.axhline(0, color="0.8", linewidth=0.6)
    _axis.axvline(0, color="0.8", linewidth=0.6)
    _axis.invert_yaxis()
    _axis.set_xlabel("right of image centre (full-res pixels)", fontsize=8)
    _axis.set_ylabel("below image centre (full-res pixels)", fontsize=8)
    _axis.set_title("Fitted colony centres (red ring = flagged)", fontsize=9)
    _axis.legend(fontsize=7)
    _offsets.tight_layout()

    # Coverage after centring, averaged per group and timestep, with the mask.
    _hours = selection["timesteps"]
    _average, _axes = plt.subplots(
        len(_groups), len(_hours), figsize=(2.1 * len(_hours), 2.2 * len(_groups)), squeeze=False
    )
    for _row_number, _group in enumerate(_groups):
        for _column, _hour in enumerate(_hours):
            _axis = _axes[_row_number, _column]
            _axis.set_xticks([])
            _axis.set_yticks([])
            if _row_number == 0:
                _axis.set_title(f"{_hour}h", fontsize=8)
            if _column == 0:
                _axis.set_ylabel(_group, fontsize=7)
            _members = [
                _index
                for _index, _entry in enumerate(raw_images)
                if _entry["record"].group == _group and _entry["record"].timestep == _hour
            ]
            if not _members:
                _axis.set_facecolor("0.9")
                continue
            _mean_coverage = np.mean(
                [place_image(coverage_maps[_index], image_geometry[_index][0]) for _index in _members],
                axis=0,
            )
            _mean_mask = np.mean([image_geometry[_index][1] for _index in _members], axis=0)
            _axis.imshow(_mean_coverage, cmap="gray", vmin=0, vmax=1)
            _axis.contour(_mean_mask, levels=[0.5], colors="cyan", linewidths=0.6)
            _axis.set_xlabel(f"{len(_members)} images", fontsize=6)
    _average.tight_layout()

    _note = (
        ""
        if mask_mode.value == "fixed"
        else "\n\nThe mask mode is not the template fit, so the averages use "
        "the selected mode while the table and centres describe the template fit."
    )
    mo.vstack(
        [
            mo.md(
                f"Pattern radius: estimated **{estimated_radius:.0f}** full-res "
                f"pixels, using **{pattern_radius:.0f}**. **{_flagged}** of "
                f"{len(_rows)} images scored below {fit_flag.value:.2f}." + _note
            ),
            mo.ui.tabs(
                {
                    "Averaged after centring": show_figure(_average),
                    "Fitted centres": _offsets,
                    "Per-image fits": mo.ui.table(_rows, selection=None),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class DragAlign(anywidget.AnyWidget):
        """An image that can be dragged under a fixed pattern circle.

        The image is drawn so that its colony centre, the automatic fit
        (``centre_row``, ``centre_column``) plus the correction (``x`` right,
        ``y`` down), sits in the middle of the training frame. Dragging the
        image changes the correction; the value is sent when you release.
        Arrow keys move the image by 1 pixel (10 with Shift), and a
        double-click returns to the automatic fit. All values are in
        full-resolution pixels.
        """

        _esm = """
        function render({ model, el }) {
          const view = 600;
          const canvas = document.createElement("canvas");
          canvas.width = view;
          canvas.height = view;
          canvas.tabIndex = 0;
          canvas.style.cssText = "cursor: grab; touch-action: none; border-radius: 4px; background: black; max-width: 100%;";
          const label = document.createElement("div");
          label.style.cssText = "font: 12px sans-serif; margin-top: 4px;";
          el.append(canvas, label);
          const context = canvas.getContext("2d");
          const picture = new Image();
          let correction = { x: model.get("x"), y: model.get("y") };
          const size = () => model.get("image_size");
          const margin = () => model.get("margin");
          const scale = () => view / (size() + 2 * margin());
          const toView = (value) => (value + margin()) * scale();
          const clamp = (value) => Math.max(-model.get("limit"), Math.min(model.get("limit"), value));
          function circle(radius, dashes, width) {
            const middle = toView((size() - 1) / 2);
            context.setLineDash(dashes);
            context.lineWidth = width;
            context.beginPath();
            context.arc(middle, middle, radius * scale(), 0, 2 * Math.PI);
            context.stroke();
          }
          function draw() {
            const s = scale();
            const n = size();
            const middle = (n - 1) / 2;
            context.fillStyle = "black";
            context.fillRect(0, 0, view, view);
            const shiftRow = middle - (model.get("centre_row") + correction.y);
            const shiftColumn = middle - (model.get("centre_column") + correction.x);
            if (picture.complete && picture.naturalWidth) {
              context.drawImage(picture, toView(-0.5 + shiftColumn), toView(-0.5 + shiftRow), n * s, n * s);
            }
            context.strokeStyle = "white";
            context.setLineDash([6, 4]);
            context.lineWidth = 1;
            context.strokeRect(toView(-0.5), toView(-0.5), n * s, n * s);
            context.strokeStyle = "#1f77b4";
            circle(model.get("radius"), [], 2);
            if (Math.abs(model.get("mask_radius") - model.get("radius")) > 0.5) circle(model.get("mask_radius"), [2, 3], 1);
            context.setLineDash([]);
            context.beginPath();
            context.moveTo(toView(middle) - 8, toView(middle));
            context.lineTo(toView(middle) + 8, toView(middle));
            context.moveTo(toView(middle), toView(middle) - 8);
            context.lineTo(toView(middle), toView(middle) + 8);
            context.stroke();
            label.textContent = `correction: right ${correction.x} px, down ${correction.y} px. Drag the image; arrow keys 1 px (Shift 10 px); double-click for the automatic fit.`;
          }
          function send() {
            model.set("x", correction.x);
            model.set("y", correction.y);
            model.save_changes();
          }
          let start = null;
          canvas.addEventListener("pointerdown", (event) => {
            start = { left: event.clientX, top: event.clientY, x: correction.x, y: correction.y };
            canvas.setPointerCapture(event.pointerId);
            canvas.style.cursor = "grabbing";
            canvas.focus();
          });
          canvas.addEventListener("pointermove", (event) => {
            if (!start) return;
            // Screen pixels to full-resolution pixels, allowing for CSS scaling.
            const perPixel = 1 / (scale() * canvas.clientWidth / view);
            correction = {
              x: clamp(Math.round(start.x - (event.clientX - start.left) * perPixel)),
              y: clamp(Math.round(start.y - (event.clientY - start.top) * perPixel)),
            };
            draw();
          });
          canvas.addEventListener("pointerup", () => {
            if (!start) return;
            start = null;
            canvas.style.cursor = "grab";
            send();
          });
          canvas.addEventListener("dblclick", () => { correction = { x: 0, y: 0 }; draw(); send(); });
          canvas.addEventListener("keydown", (event) => {
            const step = event.shiftKey ? 10 : 1;
            // Arrow keys move the image, so the correction moves the other way.
            const moves = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
            if (!(event.key in moves)) return;
            event.preventDefault();
            correction = { x: clamp(correction.x + moves[event.key][0]), y: clamp(correction.y + moves[event.key][1]) };
            draw();
            send();
          });
          picture.onload = draw;
          picture.src = model.get("image");
          model.on("change:image", () => { picture.src = model.get("image"); });
          model.on("change:x", () => { correction.x = model.get("x"); draw(); });
          model.on("change:y", () => { correction.y = model.get("y"); draw(); });
          for (const name of ["centre_row", "centre_column", "radius", "mask_radius"]) model.on(`change:${name}`, draw);
          draw();
        }
        export default { render };
        """
        image = traitlets.Unicode("").tag(sync=True)
        image_size = traitlets.Int(1080).tag(sync=True)
        centre_row = traitlets.Float(0.0).tag(sync=True)
        centre_column = traitlets.Float(0.0).tag(sync=True)
        radius = traitlets.Float(400.0).tag(sync=True)
        mask_radius = traitlets.Float(400.0).tag(sync=True)
        margin = traitlets.Int(200).tag(sync=True)
        limit = traitlets.Int(300).tag(sync=True)
        x = traitlets.Int(0).tag(sync=True)
        y = traitlets.Int(0).tag(sync=True)

    return (DragAlign,)


@app.cell(hide_code=True)
def _(mo, raw_images):
    # The pickers are chained: each offers only what was loaded for the
    # choices before it (knockout images exist only at some timesteps).
    centring_records = [_entry["record"] for _entry in raw_images]
    centring_group = mo.ui.dropdown(
        options=sorted({_r.group for _r in centring_records}),
        value=centring_records[0].group,
        label="Experiment group",
    )
    return centring_group, centring_records


@app.cell(hide_code=True)
def _(centring_group, centring_records, mo):
    _conditions = sorted({_r.condition for _r in centring_records if _r.group == centring_group.value})
    centring_condition = mo.ui.dropdown(options=_conditions, value=_conditions[0], label="Condition")
    return (centring_condition,)


@app.cell(hide_code=True)
def _(centring_condition, centring_group, centring_records, mo):
    _hours = sorted(
        {
            _r.timestep
            for _r in centring_records
            if (_r.group, _r.condition) == (centring_group.value, centring_condition.value)
        }
    )
    centring_hour = mo.ui.dropdown(
        options={f"{_h}h": _h for _h in _hours}, value=f"{_hours[0]}h", label="Timestep"
    )
    return (centring_hour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manual centring
    """)
    return


@app.cell(hide_code=True)
def _(centring_condition, centring_group, centring_hour, centring_records, mo):
    _replicates = sorted(
        {
            _r.replicate
            for _r in centring_records
            if (_r.group, _r.condition, _r.timestep)
            == (centring_group.value, centring_condition.value, centring_hour.value)
        }
    )
    centring_replicate = mo.ui.dropdown(
        options={f"{_n + 1}": _n for _n in _replicates},
        value=f"{_replicates[0] + 1}",
        label="Replicate",
    )
    mo.vstack(
        [
            mo.md(
                "One image, uncropped and unmasked, placed so that its current "
                "centre (automatic fit plus any manual correction) sits in the "
                "middle of the dashed training frame. The blue circle is the "
                "pattern circle (dotted: the mask, if the radius scale is not "
                "1). **Drag the image** until the pattern edge lies on the "
                "circle, or click it and use the arrow keys (Shift for 10 "
                "pixels); double-click to go back to the automatic fit. "
                "**Save offset** keeps the correction for this image, and every "
                "other view then uses it.\n\n"
                "Each picker lists only the loaded images that match the "
                "choices to its left. Knockout conditions only have their own "
                "images after the knockout (e.g. sl24 from 36h); their earlier "
                "timepoints are control images, listed under ctrl."
            ),
            mo.hstack(
                [centring_group, centring_condition, centring_hour, centring_replicate],
                justify="start",
            ),
        ]
    )
    return (centring_replicate,)


@app.cell(hide_code=True)
def _(
    GROUP_CHANNELS,
    centring_condition,
    centring_group,
    centring_hour,
    centring_replicate,
    mo,
    raw_images,
):
    centring_index = next(
        (
            _index
            for _index, _entry in enumerate(raw_images)
            if (
                _entry["record"].group,
                _entry["record"].condition,
                _entry["record"].timestep,
                _entry["record"].replicate,
            )
            == (
                centring_group.value,
                centring_condition.value,
                centring_hour.value,
                centring_replicate.value,
            )
        ),
        None,
    )
    mo.stop(centring_index is None, mo.md("This image was not loaded."))
    centring_channels = mo.ui.multiselect(
        options=list(GROUP_CHANNELS[centring_group.value]),
        value=list(GROUP_CHANNELS[centring_group.value]),
        label="Channels shown",
    )
    centring_channels
    return centring_channels, centring_index


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    DragAlign,
    centring_index,
    colony_fits,
    get_offsets,
    image_key,
    mo,
    pattern_radius,
    radius_scale,
    raw_images,
):
    _entry = raw_images[centring_index]
    # Automatic fit at full resolution: block i covers pixels 4i to 4i + 3.
    centring_automatic = tuple(
        DISPLAY_DOWNSAMPLE * _c + (DISPLAY_DOWNSAMPLE - 1) / 2.0
        for _c in colony_fits[centring_index]["fitted"]
    )
    # The view starts at the saved correction for this image, if any.
    _saved = get_offsets().get(image_key(_entry["record"]), (0.0, 0.0))
    # The picture is filled in by the composite cell below, so changing the
    # channels shown does not rebuild the view or lose an unsaved drag.
    centring_widget = DragAlign(
        image_size=_entry["channels"].shape[0],
        centre_row=centring_automatic[0],
        centre_column=centring_automatic[1],
        radius=pattern_radius,
        mask_radius=pattern_radius * radius_scale.value,
        x=int(round(_saved[1])),
        y=int(round(_saved[0])),
    )
    centring_view = mo.ui.anywidget(centring_widget)
    centring_save = mo.ui.run_button(label="Save offset")
    centring_reset = mo.ui.run_button(label="Remove offset for this image")
    centring_clear = mo.ui.run_button(label="Clear all offsets")
    mo.vstack(
        [
            centring_view,
            mo.hstack([centring_save, centring_reset, centring_clear], justify="start"),
        ]
    )
    return (
        centring_automatic,
        centring_clear,
        centring_reset,
        centring_save,
        centring_view,
        centring_widget,
    )


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    MICROPATTERN_260726_SCHEMA,
    base64,
    block_mean,
    centring_channels,
    centring_index,
    centring_widget,
    io,
    mo,
    np,
    plt,
    raw_images,
):
    # Composite of the selected channels on black, sent to the view at half
    # resolution (it is drawn at full size). LMBR, the membrane marker, is
    # white; the other channels of the group take cyan, magenta and yellow in
    # schema order, so a channel keeps its colour when others are toggled.
    _entry = raw_images[centring_index]
    _group = _entry["record"].group
    _channels = next(
        _g.channels for _g in MICROPATTERN_260726_SCHEMA.experiment_groups if _g.name == _group
    )
    _palette = iter([(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)])
    _colours = {
        _channel.name: (1.0, 1.0, 1.0) if _channel.marker == "LMBR" else next(_palette)
        for _channel in _channels
    }
    _composite = None
    for _name in centring_channels.value:
        _values = _entry["channels"][..., CHANNEL_SOURCE[_name][1]].astype(np.float32)
        _low, _high = np.percentile(_values, (0.5, 99.5))
        _values = np.clip((block_mean(_values, 2) - _low) / max(_high - _low, 1e-6), 0.0, 1.0)
        _layer = _values[..., None] * np.array(_colours[_name], dtype=np.float32)
        _composite = _layer if _composite is None else _composite + _layer
    if _composite is None:
        _composite = np.zeros((*block_mean(_entry["channels"][..., 0], 2).shape, 3), dtype=np.float32)
    _buffer = io.BytesIO()
    plt.imsave(_buffer, np.clip(_composite, 0.0, 1.0), format="png")
    centring_widget.image = "data:image/png;base64," + base64.b64encode(_buffer.getvalue()).decode()

    _swatches = " &nbsp; ".join(
        f"<span style='display:inline-block; width:0.9em; height:0.9em; margin-right:0.3em; "
        f"vertical-align:middle; border:1px solid #888; "
        f"background: rgb({int(255 * _c[0])}, {int(255 * _c[1])}, {int(255 * _c[2])})'></span>"
        + (f"`{_name}`" if _name in centring_channels.value else f"~~`{_name}`~~ (hidden)")
        for _name, _c in _colours.items()
    )
    mo.md(
        "**Channels:** " + _swatches + "<br>Each channel is scaled to its own "
        "0.5–99.5 percentiles; overlapping colours add up (e.g. cyan + magenta "
        "+ yellow = white)."
    )
    return


@app.cell(hide_code=True)
def _(centring_index, get_flags, image_key, mo, raw_images):
    _current = get_flags().get(image_key(raw_images[centring_index]["record"]))
    flag_reason = mo.ui.text(value=_current or "", label="Reason (optional)", full_width=True)
    flag_set = mo.ui.run_button(label="Flag as low quality")
    flag_remove = mo.ui.run_button(label="Remove flag")
    mo.vstack(
        [
            mo.md(
                (
                    "**Low quality: flagged**"
                    + (f" ({_current})" if _current else "")
                    + ". Training treats this image as not measured."
                )
                if _current is not None
                else "**Low quality: not flagged.** Flag images that should not be "
                "trained on (e.g. out of focus or damaged). Training then treats "
                "the image (this experiment group at this timestep and replicate) "
                "as not measured; the replicate's other groups and timesteps are kept."
            ),
            mo.hstack([flag_reason, flag_set, flag_remove], justify="start"),
        ]
    )
    return flag_reason, flag_remove, flag_set


@app.cell(hide_code=True)
def _(
    centring_index,
    flag_reason,
    flag_remove,
    flag_set,
    image_key,
    raw_images,
    set_flags,
):
    _key = image_key(raw_images[centring_index]["record"])
    if flag_set.value:
        set_flags(lambda _current: {**_current, _key: flag_reason.value})
    if flag_remove.value:
        set_flags(lambda _current: {_k: _v for _k, _v in _current.items() if _k != _key})
    return


@app.cell(hide_code=True)
def _(
    centring_clear,
    centring_index,
    centring_reset,
    centring_save,
    centring_view,
    image_key,
    raw_images,
    set_offsets,
):
    _label = image_key(raw_images[centring_index]["record"])
    if centring_save.value:
        _offset = (float(centring_view.value["y"]), float(centring_view.value["x"]))
        set_offsets(lambda _current: {**_current, _label: _offset})
    if centring_reset.value:
        set_offsets(lambda _current: {_k: _v for _k, _v in _current.items() if _k != _label})
    if centring_clear.value:
        set_offsets({})
    return


@app.cell(hide_code=True)
def _(
    centring_automatic,
    centring_index,
    centring_view,
    get_offsets,
    image_key,
    mo,
    raw_images,
):
    _entry = raw_images[centring_index]
    _size = _entry["channels"].shape[:2]
    _middle = ((_size[0] - 1) / 2.0, (_size[1] - 1) / 2.0)
    _manual = (float(centring_view.value["y"]), float(centring_view.value["x"]))
    _centre = (centring_automatic[0] + _manual[0], centring_automatic[1] + _manual[1])
    _shift = (_middle[0] - _centre[0], _middle[1] - _centre[1])
    _saved = get_offsets().get(image_key(_entry["record"]))
    _rows = [
        {"": "correction to automatic centring", "down": _manual[0], "right": _manual[1]},
        {"": "colony centre in raw image (row, column)", "down": round(_centre[0], 1), "right": round(_centre[1], 1)},
        {
            "": "colony centre relative to raw image centre",
            "down": round(_centre[0] - _middle[0], 1),
            "right": round(_centre[1] - _middle[1], 1),
        },
        {"": "shift applied to the raw image", "down": round(_shift[0], 1), "right": round(_shift[1], 1)},
        {
            "": "automatic fit: colony centre in raw image",
            "down": round(centring_automatic[0], 1),
            "right": round(centring_automatic[1], 1),
        },
    ]
    mo.vstack(
        [
            mo.md(
                "All values in full-resolution pixels; they update when you "
                "release the image. Saved correction for this image: "
                + (f"**down {_saved[0]:g}, right {_saved[1]:g}**" if _saved else "**none**")
                + ("" if _saved is None or _saved == _manual else " (the view differs; click **Save offset** to keep it)")
            ),
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Export and load alignment
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    # Relative alignment paths are relative to the repository root, as in
    # data.micropattern.cleaning.alignment_file.
    _notebook_dir = mo.notebook_dir()
    _repository = _notebook_dir.parents[1] if _notebook_dir is not None else Path.cwd()

    def resolve_alignment_path(text):
        path = Path(text).expanduser()
        return path if path.is_absolute() else _repository / path

    alignment_path = mo.ui.text(
        value="Experiments/micropatterns/conf/alignment/260726_alignment.yaml",
        label="Alignment file (relative to the repository root)",
        full_width=True,
    )
    export_alignment = mo.ui.run_button(label="Export alignment")
    load_alignment = mo.ui.run_button(label="Load alignment")
    use_loaded_alignment = mo.ui.checkbox(
        value=True, label="Use the loaded file's automatic centres and pattern radius"
    )
    check_loader = mo.ui.run_button(label="Compare with the training loader")
    mo.vstack(
        [
            mo.md(
                "The dataset is fixed, so the colony centres are computed once "
                "and saved in a file that training reads "
                "(`data.micropattern.cleaning.alignment_file`).\n\n"
                "- **Export alignment** fits every image in the dataset (all "
                "conditions, groups, timesteps, replicates and projections, not "
                "only the loaded ones), adds the manual corrections from **Manual "
                "centring**, and writes the centres with the pattern radius. "
                "Images not loaded here are read and fitted with the current "
                "settings, which takes a little while. Click **Load "
                "alignment** afterwards to view the exported file.\n"
                "- **Load alignment** reads a file. With the checkbox on, its "
                "automatic centres and pattern radius replace the notebook's "
                "own fit, so every view shows the alignment training would "
                "use. Its manual corrections are copied into **Manual "
                "centring** (replacing the current ones), so you can keep "
                "correcting and export again.\n"
                "- **Compare with the training loader** runs the training "
                "loader with the **Training settings** and the loaded file on "
                "the loaded selection, and compares the result with the "
                "notebook's normalised images (run **Clean all loaded images** "
                "first). The differences should be close to zero."
            ),
            alignment_path,
            mo.hstack([export_alignment, load_alignment, use_loaded_alignment], justify="start"),
            check_loader,
        ]
    )
    return (
        alignment_path,
        check_loader,
        export_alignment,
        load_alignment,
        resolve_alignment_path,
        use_loaded_alignment,
    )


@app.cell(hide_code=True)
def _(
    alignment_path,
    load_alignment,
    load_colony_alignment,
    mo,
    resolve_alignment_path,
    set_loaded_alignment,
    set_offsets,
):
    mo.stop(not load_alignment.value)
    _alignment = load_colony_alignment(resolve_alignment_path(alignment_path.value))
    set_loaded_alignment({"path": alignment_path.value, "alignment": _alignment})
    set_offsets(
        {
            _key: (float(_details["manual"][0]), float(_details["manual"][1]))
            for _key, _details in _alignment.details.items()
            if "manual" in _details and any(float(_v) != 0.0 for _v in _details["manual"])
        }
    )
    return


@app.cell(hide_code=True)
def _(
    ColonyAlignment,
    DISPLAY_DOWNSAMPLE,
    Path,
    alignment_path,
    block_centre,
    build_micropattern_260726_manifest,
    colony_fits,
    coverage_blur,
    coverage_map,
    edge_width,
    export_alignment,
    fit_colony_centre,
    fit_flag,
    fit_score,
    get_loaded_alignment,
    get_offsets,
    grid_position,
    image_key,
    mo,
    pattern_radius,
    pattern_radius_source,
    radius_quantile,
    read_micropattern_260726_image,
    resolve_alignment_path,
    save_colony_alignment,
    selection,
    use_loaded_alignment,
):
    mo.stop(not export_alignment.value)
    _path = resolve_alignment_path(alignment_path.value)
    from Common.dataloader.micropattern_projections import PROJECTIONS as _PROJECTIONS
    from Common.dataloader.micropattern_projections import image_key as _path_key

    # Every image of the dataset, each in its own condition (no substitution),
    # in every projection. The centres are keyed without the projection, so
    # the file serves all three; a few images exist in only some projections.
    # Each image is fitted on the selected projection where it has one.
    _records, _all_keys = {}, set()
    for _projection in sorted(_PROJECTIONS, key=lambda _name: _name != selection["projection"]):
        _inventory = build_micropattern_260726_manifest(
            selection["root"],
            conditions=("ctrl", "sl0", "sl24"),
            projection=_projection,
            timesteps=(0, 12, 24, 36, 48),
            replicate_count=64,
            substitute_preperturbation=False,
        )
        for _record in _inventory["records"]:
            _records.setdefault(image_key(_record), _record)
        _all_keys |= {
            _path_key(Path(_file).relative_to(selection["root"]).as_posix())
            for _file in _inventory["unselected_files"]
        }
    _records = list(_records.values())
    _tiffs = len(_all_keys | {image_key(_record) for _record in _records})
    _known = {_fit["key"]: _fit for _fit in colony_fits}
    _loaded = get_loaded_alignment()
    _file = _loaded["alignment"] if _loaded is not None and use_loaded_alignment.value else None
    _offsets = get_offsets()
    _radius = pattern_radius / DISPLAY_DOWNSAMPLE
    _centres, _details = {}, {}
    for _record in mo.status.progress_bar(_records, title="Fitting colonies", remove_on_exit=True):
        _key = image_key(_record)
        _manual = _offsets.get(_key, (0.0, 0.0))
        if _key in _known:
            _fitted = _known[_key]["fitted"]
            _score = _known[_key]["score"]
        else:
            _, _foreground = read_micropattern_260726_image(_record)
            _coverage = coverage_map(_foreground, DISPLAY_DOWNSAMPLE, coverage_blur.value)
            if _file is not None and _key in _file.centres:
                _automatic = _file.details.get(_key, {}).get("automatic", _file.centres[_key])
                _fitted = tuple(float(_v) for _v in grid_position(_automatic, DISPLAY_DOWNSAMPLE))
            else:
                _fitted, _ = fit_colony_centre(_coverage, _radius, edge_width.value)
            _corrected = (
                _fitted[0] + _manual[0] / DISPLAY_DOWNSAMPLE,
                _fitted[1] + _manual[1] / DISPLAY_DOWNSAMPLE,
            )
            _score = fit_score(_coverage, _corrected, _radius, edge_width.value)
        _automatic_full = tuple(float(_v) for _v in block_centre(_fitted, DISPLAY_DOWNSAMPLE))
        _centres[_key] = (_automatic_full[0] + _manual[0], _automatic_full[1] + _manual[1])
        _details[_key] = {"automatic": _automatic_full, "manual": _manual, "score": _score}
    save_colony_alignment(
        _path,
        ColonyAlignment(
            pattern_radius=pattern_radius,
            centres=_centres,
            details=_details,
            settings={
                "method": "fixed_radius_template",
                "fit_downsample": DISPLAY_DOWNSAMPLE,
                "edge_width": float(edge_width.value),
                "coverage_blur": float(coverage_blur.value),
                "radius_quantile": float(radius_quantile.value),
                "pattern_radius": pattern_radius_source,
            },
        ),
    )
    # The file is not loaded back here: that would change the centres this
    # cell reads, re-run it (the button still counts as clicked) and export
    # again, endlessly. Click "Load alignment" to view the exported file.
    _flagged = sorted(_key for _key, _d in _details.items() if _d["score"] < fit_flag.value)
    mo.vstack(
        [
            mo.md(
                f"Exported **{len(_centres)}** images to `{_path}`; click **Load "
                f"alignment** to view them (pattern radius "
                f"{pattern_radius:.1f}, {pattern_radius_source}), "
                f"{sum(1 for _d in _details.values() if _d['manual'] != (0.0, 0.0))} with a "
                f"manual correction. There are {_tiffs} distinct images over "
                "the max, sum and slice projections"
                + (
                    "."
                    if _tiffs == len(_centres)
                    else "; files that the loader does not use (e.g. extra RNA replicates "
                    "without a replicate number) are not in the alignment file."
                )
                + f" **{len(_flagged)}** images scored below {fit_flag.value:.2f}"
                + (": check them under **Manual centring**." if _flagged else ".")
            ),
            mo.ui.table(
                [{"image": _key, "score": round(_details[_key]["score"], 3)} for _key in _flagged],
                selection=None,
            )
            if _flagged
            else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(colony_fits, get_loaded_alignment, mo, use_loaded_alignment):
    _loaded = get_loaded_alignment()
    if _loaded is None:
        _status = "No alignment file loaded; the views use the notebook's own fit."
    else:
        _alignment = _loaded["alignment"]
        _covered = sum(1 for _fit in colony_fits if _fit["source"] == "file")
        _status = (
            f"Loaded `{_loaded['path']}`: {len(_alignment.centres)} images, pattern "
            f"radius {_alignment.pattern_radius:.1f}. "
            + (
                f"In use: {_covered} of the {len(colony_fits)} loaded images take "
                "their automatic centre from the file"
                + (
                    "."
                    if _covered == len(colony_fits)
                    else "; the others are fitted here and missing from the file, "
                    "so export again before training."
                )
                if use_loaded_alignment.value
                else "Not in use (checkbox off)."
            )
        )
    mo.md(_status)
    return


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    MicropatternCleaningConfig,
    check_loader,
    get_flags,
    get_loaded_alignment,
    load_micropattern_260726,
    mo,
    normalised_image,
    np,
    resolve_alignment_path,
    selection,
    training_mismatches,
    training_settings,
    use_loaded_alignment,
):
    mo.stop(not check_loader.value)
    _loaded = get_loaded_alignment()
    mo.stop(
        _loaded is None or not use_loaded_alignment.value,
        mo.md("Export or load an alignment file, with **Use the loaded file** on, first."),
    )
    mo.stop(
        bool(training_mismatches),
        mo.md("Not comparable, because " + "; ".join(training_mismatches) + "."),
    )
    _settings = dict(training_settings["cleaning"])
    _settings["alignment_file"] = str(resolve_alignment_path(_loaded["path"]))
    _settings["channel_percentiles"] = {
        _name: tuple(_pair) for _name, _pair in _settings["channel_percentiles"].items()
    }
    with mo.status.spinner(title="Running the training loader"):
        _dataset = load_micropattern_260726(
            selection["root"],
            conditions=selection["conditions"],
            projection=selection["projection"],
            timesteps=selection["timesteps"],
            downsample=DISPLAY_DOWNSAMPLE,
            replicate_indices=selection["replicates"],
            experiment_groups=selection["groups"],
            excluded_measurements=selection["excluded"],
            hist_eqs=tuple(training_settings["histogram_percentiles"]),
            intensity_factors=training_settings["intensity_factors"],
            cleaning=MicropatternCleaningConfig(**_settings),
            excluded_images=tuple(get_flags()),
        )
    _data = np.asarray(_dataset.data)
    _measured = np.asarray(_dataset.measurement_mask)
    _aux = _dataset.aux
    _differences = {}
    for _batch, (_condition, _replicate) in enumerate(
        zip(_aux["batch_conditions"], _aux["batch_replicates"])
    ):
        for _time, _hour in enumerate(_aux["timesteps"]):
            for _channel, _name in enumerate(_dataset.channel_names):
                if not _measured[_batch, _time, _channel]:
                    continue
                _notebook = normalised_image(_name, (_condition, _replicate - 1), _hour)
                if _notebook is None:
                    continue
                _differences.setdefault(_name, []).append(
                    np.abs(_data[_batch, _time, _channel] - _notebook[0])
                )
    _rows = [
        {
            "channel": _name,
            "images": len(_values),
            "max difference": float(np.max([_v.max() for _v in _values])),
            "mean difference": float(np.mean([_v.mean() for _v in _values])),
        }
        for _name, _values in _differences.items()
    ]
    _worst = max((_row["max difference"] for _row in _rows), default=float("nan"))
    mo.vstack(
        [
            mo.md(
                f"Training loader output: shape {_data.shape} (batch, time, channel, "
                f"x, y). Largest difference from the notebook, on the 0–1 scale: "
                f"**{_worst:.2g}**."
            ),
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quality flags
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    flags_path = mo.ui.text(
        value="Experiments/micropatterns/conf/quality/260726_quality_flags.yaml",
        label="Quality flags file (relative to the repository root)",
        full_width=True,
    )
    export_flags = mo.ui.run_button(label="Export flags")
    load_flags = mo.ui.run_button(label="Load flags")
    mo.vstack(
        [
            mo.md(
                "Images flagged as low quality under **Manual centring**. "
                "**Export flags** writes them to the file that training reads "
                "(`data.micropattern.quality_flags_file`), which then treats each "
                "flagged image as not measured: it is left out of the loss, "
                "reinjection and the normalisation bounds, and its slot is filled "
                "with the mean of the same group and timestep over the other "
                "replicates of the condition. **Load flags** replaces the current "
                "flags with a file's. Flagged images have a red border in the "
                "processed-image figures."
            ),
            flags_path,
            mo.hstack([export_flags, load_flags], justify="start"),
        ]
    )
    return export_flags, flags_path, load_flags


@app.cell(hide_code=True)
def _(
    export_flags,
    flags_path,
    get_flags,
    mo,
    resolve_alignment_path,
    save_quality_flags,
):
    mo.stop(not export_flags.value)
    _path = resolve_alignment_path(flags_path.value)
    save_quality_flags(_path, get_flags())
    mo.md(f"Exported **{len(get_flags())}** flagged images to `{_path}`.")
    return


@app.cell(hide_code=True)
def _(
    flags_path,
    load_flags,
    load_quality_flags,
    mo,
    resolve_alignment_path,
    set_flags,
):
    mo.stop(not load_flags.value)
    set_flags(load_quality_flags(resolve_alignment_path(flags_path.value)))
    return


@app.cell(hide_code=True)
def _(get_flags, image_key, mo, raw_images, record_label):
    _flags = get_flags()
    _records = {image_key(_entry["record"]): _entry["record"] for _entry in raw_images}
    _rows = [
        {
            "image": record_label(_records[_key]) if _key in _records else f"{_key} (not loaded)",
            "reason": _reason,
        }
        for _key, _reason in sorted(_flags.items())
    ]
    # Flagged groups per replicate (condition and replicate), for the loaded images.
    _groups = {}
    for _key in _flags:
        if _key in _records:
            _record = _records[_key]
            _groups.setdefault((_record.condition, _record.replicate), set()).add(_record.group)
    _crowded = [
        f"{_condition}, replicate {_replicate + 1}: " + ", ".join(sorted(_names))
        for (_condition, _replicate), _names in sorted(_groups.items())
        if len(_names) > 1
    ]
    mo.vstack(
        [
            mo.md(f"**{len(_flags)}** images flagged."),
            mo.callout(
                mo.md(
                    "More than one experiment group is flagged for: "
                    + "; ".join(_crowded)
                    + ". Expected is at most one per replicate."
                ),
                kind="warn",
            )
            if _crowded
            else mo.md(""),
            mo.ui.table(_rows, selection=None) if _rows else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Local cache of cleaned images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from Common.dataloader.disk_cache import cache_dir_from_environment as _from_environment

    cache_dir = mo.ui.text(
        value=_from_environment() or "",
        label="Cache directory (DATA_CACHE_DIR)",
        full_width=True,
    )
    cache_downsample = mo.ui.number(start=1, stop=16, value=4, label="Cleaning downsampling")
    fill_cache = mo.ui.run_button(label="Fill cache")
    mo.vstack(
        [
            mo.md(
                "Cleaning (intensity factors, hot pixels, background removal "
                "and block-averaging) is the slow part of loading. With "
                "`DATA_CACHE_DIR` set (e.g. in `.env`), the training loader "
                "keeps each cleaned image in that directory and reads it back "
                "next time, so evaluation notebooks load quickly. Centring, "
                "normalisation and the final downsampling are quick and are "
                "always recomputed, so one cache folder serves every "
                "selection of conditions and replicates and every "
                "normalisation setting.\n\n"
                "- **Fill cache** cleans every image of the dataset (for the "
                "selected projection) with the "
                "**Training settings** and saves them in a folder of their "
                "own, with a `settings.yaml`. Images already in it are "
                "skipped.\n"
                "- The cleaning downsampling is 4 for any training "
                "downsampling that is a multiple of 4 (4, 8, 16, ...); "
                "otherwise it is the training downsampling itself.\n"
                "- Changing the cleaning settings, or the code that does the "
                "cleaning, starts a new folder. Old folders are listed below "
                "and can be deleted by hand."
            ),
            cache_dir,
            mo.hstack([cache_downsample, fill_cache], justify="start"),
        ]
    )
    return cache_dir, cache_downsample, fill_cache


@app.cell(hide_code=True)
def _(
    MicropatternCleaningConfig,
    Path,
    build_micropattern_260726_manifest,
    cache_dir,
    cache_downsample,
    fill_cache,
    mo,
    selection,
    training_settings,
):
    from Common.dataloader.disk_cache import list_cache_folders as _list_folders
    from Common.dataloader.micropattern_260726 import (
        cleaned_image as _cleaned_image,
        cleaning_cache_folder as _cleaning_cache_folder,
    )

    mo.stop(not cache_dir.value.strip(), mo.md("Set a cache directory to use the cache."))
    _message = mo.md("")
    if fill_cache.value:
        _settings = dict(training_settings["cleaning"])
        _settings["channel_percentiles"] = {
            _name: tuple(_pair) for _name, _pair in _settings["channel_percentiles"].items()
        }
        _cleaning = MicropatternCleaningConfig(**_settings)
        _folder = _cleaning_cache_folder(
            cache_dir.value,
            selection["root"],
            training_settings["intensity_factors"],
            _cleaning,
            int(cache_downsample.value),
        )
        # Every image of the dataset, each in its own condition (no substitution).
        _records = build_micropattern_260726_manifest(
            selection["root"],
            conditions=("ctrl", "sl0", "sl24"),
            projection=selection["projection"],
            timesteps=(0, 12, 24, 36, 48),
            replicate_count=64,
            substitute_preperturbation=False,
        )["records"]
        # One folder holds every projection; count only this one's images.
        _before = sum(
            (_folder / f"{Path(_record.path).relative_to(selection['root']).as_posix()}.npy").is_file()
            for _record in _records
        )
        for _record in mo.status.progress_bar(_records, title="Cleaning images", remove_on_exit=True):
            _cleaned_image(
                _record,
                training_settings["intensity_factors"],
                _cleaning,
                int(cache_downsample.value),
                selection["root"],
                _folder,
            )
        _message = mo.md(
            f"Cleaned **{len(_records) - _before}** images into `{_folder}` "
            f"({_before} were already there)."
        )
    _rows = [
        {_key: _value for _key, _value in _row.items() if _key != "settings"}
        | {"downsampling": _row["settings"].get("downsample")}
        for _row in _list_folders(cache_dir.value)
    ]
    mo.vstack([_message, mo.ui.table(_rows, selection=None) if _rows else mo.md("The cache is empty.")])
    return


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    bound_sources,
    clean_channel_with,
    cleaning_settings,
    flagged_indices,
    high_percentiles,
    image_geometry,
    image_source,
    inside_only,
    knockout_bounds,
    low_percentiles,
    mo,
    normalise_mode,
    np,
    place_image,
    rescale,
    trajectories,
    trajectory_bins,
):
    def live_channel(name, trajectory, normalisation=None, knockout_mode=None):
        """Process one channel with the current controls, for the live views.

        ``normalisation`` and ``knockout_mode`` replace the section 5 choices
        of how bounds are shared, to compare them side by side.

        Returns ``(bounds, images)``: the clipping bounds used for
        ``trajectory`` and ``{hour: (normalised, mask)}`` for its measured
        timesteps, at the training resolution and not yet zeroed outside the
        mask. Shared bounds (averaged or pooled) need every trajectory of the
        channel, and control or joint knockout bounds need the other
        conditions of the replicate, so those images are processed too.
        Results are cached.
        """
        group, channel = CHANNEL_SOURCE[name]
        low, high = low_percentiles.value[name], high_percentiles.value[name]
        if low >= high:
            raise ValueError(f"Low % must be below high % for {name}")
        normalisation = normalisation or normalise_mode.value
        knockout_mode = knockout_mode or knockout_bounds.value
        sources = bound_sources(knockout_mode)

        def sets_bounds(other):
            """Whether ``other`` can change the bounds of ``trajectory``."""
            if normalisation != "per_replicate" or other == trajectory:
                return True
            if other[1] != trajectory[1]:
                return False
            if knockout_mode == "control":
                return other[0] == "ctrl" and trajectory[0] != "ctrl"
            return knockout_mode == "joint"

        needed = [
            other
            for other in sources
            if sets_bounds(other)
            and any(member_group == group for member_group, _ in sources[other])
        ]
        indices = sorted(
            {
                index
                for other in needed
                for (member_group, _), index in sources[other].items()
                if member_group == group
            }
        )
        placed = {}
        for index in mo.status.progress_bar(
            indices, title=f"Cleaning {name}", remove_on_exit=True
        ):
            scaled, cleaned = clean_channel_with(cleaning_settings, index, name, channel)
            shift, mask = image_geometry[index]
            values = cleaned if image_source.value == "cleaned" else scaled
            placed[index] = (place_image(values, shift), mask)

        samples = {}
        for other in needed:
            values = [
                placed[index][0][placed[index][1]] if inside_only.value else placed[index][0].ravel()
                for (member_group, _), index in sources[other].items()
                if member_group == group and index not in flagged_indices
            ]
            if values:
                samples[other] = np.concatenate(values)
        bounds = trajectory_bins(samples, low, high, normalisation, knockout_mode)[trajectory]
        images = {
            hour: (rescale(placed[index][0], *bounds), placed[index][1])
            for (member_group, hour), index in trajectories[trajectory].items()
            if member_group == group
        }
        return bounds, images

    return (live_channel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One channel over time
    """)
    return


@app.cell(hide_code=True)
def _(
    image_source,
    inside_only,
    knockout_bounds,
    loaded_names,
    mo,
    normalise_mode,
    trajectories,
    trajectory_label,
    zero_outside,
):
    strip_channel = mo.ui.dropdown(
        options=loaded_names, value=loaded_names[0], label="Channel"
    )
    strip_trajectory = mo.ui.dropdown(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        value=trajectory_label(next(iter(trajectories))),
        label="Trajectory",
    )
    strip_histograms = mo.ui.checkbox(value=False, label="Show intensity histograms")
    strip_run = mo.ui.run_button(label="Show channel")
    mo.vstack(
        [
            mo.md(
                "Raw (top) and processed (bottom) images of one channel for "
                "one trajectory. Raw images are as stored, before the intensity "
                "factor, background removal or centring, and share one colour "
                "scale across timesteps (0.5–99.5 percentiles). Processed "
                "images are on the normalised 0–1 scale. Images flagged as low "
                "quality have a red border.\n\n"
                "Press **Show channel** to draw it (it does not need "
                "**Clean all loaded images**). Only the chosen channel is "
                "processed. With bounds shared across replicates (averaged or "
                "pooled), all trajectories of that channel are processed, so "
                "the first view of a channel, or a change to its section 2 "
                "controls, takes a few seconds. Results are cached after that.\n\n"
                "**Intensity histograms** add one histogram per timestep: raw "
                "pixels of the whole image at full resolution (top), and "
                "processed values inside the mask (bottom). Each row shares "
                "its bins across timesteps, and counts are on a log scale."
            ),
            mo.hstack([strip_channel, strip_trajectory, strip_histograms], justify="start"),
            mo.md(
                "**Intensity rescaling.** The same controls as in section 5 "
                "(changing them here changes them there too). Pick a knockout "
                "trajectory and switch the knockout bounds to see how much of "
                "its intensity change is kept."
            ),
            mo.hstack([normalise_mode, knockout_bounds], justify="start"),
            mo.hstack([image_source, inside_only, zero_outside], justify="start"),
            strip_run,
        ]
    )
    return strip_channel, strip_histograms, strip_run, strip_trajectory


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    block_mean,
    flagged_indices,
    high_percentiles,
    knockout_bounds,
    live_channel,
    low_percentiles,
    mo,
    normalise_mode,
    np,
    outline_flagged,
    plt,
    raw_images,
    selection,
    show_figure,
    strip_channel,
    strip_histograms,
    strip_run,
    strip_trajectory,
    trajectories,
    trajectory_label,
    zero_outside,
):
    mo.stop(not strip_run.value, mo.md("Press **Show channel** to draw this view. It does not redraw by itself when the controls change."))
    _name = strip_channel.value
    _group, _channel = CHANNEL_SOURCE[_name]
    _trajectory = strip_trajectory.value
    _hours = selection["timesteps"]
    mo.stop(
        low_percentiles.value[_name] >= high_percentiles.value[_name],
        mo.md(f"Low % must be below high % for `{_name}`."),
    )
    _bounds, _images = live_channel(_name, _trajectory)

    _raw = {}
    for _hour in _hours:
        _index = trajectories[_trajectory].get((_group, _hour))
        if _index is not None:
            _raw[_hour] = block_mean(
                raw_images[_index]["channels"][..., _channel].astype(np.float32)
            )
    _figure, _axes = plt.subplots(
        2, len(_hours), figsize=(2.6 * len(_hours), 5.6), squeeze=False
    )
    if _raw:
        _low, _high = np.percentile(
            np.concatenate([_v.ravel() for _v in _raw.values()]), (0.5, 99.5)
        )
    for _column, _hour in enumerate(_hours):
        _axes[0, _column].set_title(f"{_hour}h", fontsize=9)
        _index = trajectories[_trajectory].get((_group, _hour))
        for _row in range(2):
            _axis = _axes[_row, _column]
            _axis.set_xticks([])
            _axis.set_yticks([])
            if _index is None:
                _axis.set_facecolor("0.9")
                continue
            if _index in flagged_indices:
                outline_flagged(_axis)
            if _row == 0:
                _axis.imshow(_raw[_hour], cmap="magma", vmin=_low, vmax=_high)
                continue
            _processed, _mask = _images[_hour]
            if zero_outside.value:
                _processed = _processed * _mask
            _axis.imshow(_processed, cmap="magma", vmin=0, vmax=1)
            if not zero_outside.value:
                _axis.contour(_mask, levels=[0.5], colors="cyan", linewidths=0.4)
    _axes[0, 0].set_ylabel("raw", fontsize=9)
    _axes[1, 0].set_ylabel("processed", fontsize=9)
    # Where the clipping bounds come from, for the title.
    _source = {
        "per_replicate": "own bounds",
        "replicate_mean": "bounds averaged over replicates",
        "pooled": "pooled bounds",
    }[normalise_mode.value]
    if knockout_bounds.value == "joint":
        _source += ", widest over conditions"
    elif knockout_bounds.value == "control" and _trajectory[0] != "ctrl":
        _source = (
            f"bounds of {trajectory_label(('ctrl', _trajectory[1]))}"
            if normalise_mode.value == "per_replicate"
            else "bounds from control replicates"
        )
    _figure.suptitle(
        f"{_name}, {trajectory_label(_trajectory)}: clipped to "
        f"{_bounds[0]:.1f}–{_bounds[1]:.1f} ({_source})",
        fontsize=10,
    )
    _figure.tight_layout()

    _views = [show_figure(_figure)]
    if strip_histograms.value and _raw:
        _full_raw = {
            _hour: raw_images[_index]["channels"][..., _channel].ravel()
            for (_member_group, _hour), _index in trajectories[_trajectory].items()
            if _member_group == _group
        }
        # Shared raw bins up to the pooled 99.9th percentile; brighter pixels
        # are counted in the last bin.
        _raw_top = max(
            float(np.percentile(np.concatenate(list(_full_raw.values())), 99.9)), 1.0
        )
        _raw_bins = np.linspace(0.0, _raw_top, 81)
        _processed_bins = np.linspace(0.0, 1.0, 51)
        _histograms, _hist_axes = plt.subplots(
            2, len(_hours), figsize=(2.6 * len(_hours), 4.6), squeeze=False, sharey="row"
        )
        for _column, _hour in enumerate(_hours):
            _raw_axis, _processed_axis = _hist_axes[:, _column]
            _raw_axis.set_title(f"{_hour}h", fontsize=9)
            if _hour not in _full_raw:
                _raw_axis.set_facecolor("0.9")
                _processed_axis.set_facecolor("0.9")
                continue
            _raw_axis.hist(
                np.minimum(_full_raw[_hour], _raw_top), bins=_raw_bins, color="grey"
            )
            _raw_axis.set_yscale("log")
            _processed, _mask = _images[_hour]
            _inside = _processed[_mask]
            _processed_axis.hist(_inside, bins=_processed_bins, color="C3")
            _processed_axis.set_yscale("log")
            _processed_axis.set_title(
                f"{100 * np.mean(_inside <= 0):.0f}% at 0, {100 * np.mean(_inside >= 1):.0f}% at 1",
                fontsize=7,
            )
            for _axis in (_raw_axis, _processed_axis):
                _axis.tick_params(labelsize=6)
        _hist_axes[0, 0].set_ylabel("raw: pixel count", fontsize=8)
        _hist_axes[1, 0].set_ylabel("processed: pixel count", fontsize=8)
        _histograms.tight_layout()
        _views.append(_histograms)
    mo.vstack(_views)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Duplicate channels
    """)
    return


@app.cell(hide_code=True)
def _(MICROPATTERN_260726_SCHEMA, loaded_names, mo, selection):
    _marker = {
        _channel.name: _channel.marker
        for _channel in MICROPATTERN_260726_SCHEMA.measurement_channels
    }
    # Loaded measurement names of each marker: {marker: (name, ...)}.
    MARKER_CHANNELS = {
        _m: tuple(_name for _name in loaded_names if _marker[_name] == _m)
        for _m in dict.fromkeys(_marker[_name] for _name in loaded_names)
    }
    # Start on a marker measured in several groups, if there is one.
    _duplicated = [_m for _m, _n in MARKER_CHANNELS.items() if len(_n) > 1]
    duplicate_marker = mo.ui.dropdown(
        options=list(MARKER_CHANNELS),
        value=(_duplicated or list(MARKER_CHANNELS))[0],
        label="Marker",
    )
    duplicate_conditions = mo.ui.multiselect(
        options=list(selection["conditions"]),
        value=list(selection["conditions"]),
        label="Conditions",
    )
    duplicate_replicates = mo.ui.multiselect(
        options={f"replicate {_r + 1}": _r for _r in selection["replicates"]},
        value=[f"replicate {selection['replicates'][0] + 1}"],
        label="Replicates",
    )
    duplicate_hours = mo.ui.multiselect(
        options={f"{_hour}h": _hour for _hour in selection["timesteps"]},
        value=[f"{_hour}h" for _hour in selection["timesteps"]],
        label="Timesteps",
    )
    duplicate_images = mo.ui.multiselect(
        options={"Raw": "raw", "Processed": "processed"},
        value=["Raw", "Processed"],
        label="Images",
    )
    duplicate_layout = mo.ui.radio(
        options={"Full rows": "rows", "Circle segments": "segments"},
        value="Full rows",
        label="Show as",
        inline=True,
    )
    duplicate_raw_scale = mo.ui.checkbox(value=False, label="Raw: one colour scale per row / segment")
    duplicate_histograms = mo.ui.checkbox(value=False, label="Show intensity histograms")
    duplicate_run = mo.ui.run_button(label="Compare duplicates")
    mo.vstack(
        [
            mo.md(
                "- One marker across the experiment groups that measure it "
                "(e.g. SOX17 in stains 1 and 2) and across conditions "
                "(control vs knockouts). Each condition and group pair is "
                "compared with every other.\n"
                "- Pick replicates and timesteps: every pair is one column.\n"
                "- **Full rows**: one row per condition and group. **Circle "
                "segments**: one image per column, each condition and group "
                "filling one segment of the colony (needs centring, section 3).\n"
                "- Raw images are before the intensity factor and background "
                "removal, centred like the processed ones, and on one colour "
                "scale (0.5–99.5 percentiles) unless set per row. Processed "
                "images use the section 2 and 5 settings (see **Knockout "
                "bounds** for knockout intensity changes).\n"
                "- Histograms and the table use pixels inside the mask. "
                "Correlation and difference compare centred colonies pixel by pixel."
            ),
            mo.hstack([duplicate_marker, duplicate_conditions, duplicate_images], justify="start"),
            mo.hstack([duplicate_replicates, duplicate_hours], justify="start"),
            mo.hstack(
                [duplicate_layout, duplicate_raw_scale, duplicate_histograms], justify="start"
            ),
            duplicate_run,
        ]
    )
    return (
        MARKER_CHANNELS,
        duplicate_conditions,
        duplicate_histograms,
        duplicate_hours,
        duplicate_images,
        duplicate_layout,
        duplicate_marker,
        duplicate_raw_scale,
        duplicate_replicates,
        duplicate_run,
    )


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    MARKER_CHANNELS,
    block_mean,
    duplicate_conditions,
    duplicate_histograms,
    duplicate_hours,
    duplicate_images,
    duplicate_layout,
    duplicate_marker,
    duplicate_raw_scale,
    duplicate_replicates,
    duplicate_run,
    flagged_indices,
    high_percentiles,
    image_geometry,
    live_channel,
    low_percentiles,
    mo,
    ndi,
    np,
    outline_flagged,
    place_image,
    plt,
    raw_images,
    selection,
    show_figure,
    trajectories,
    zero_outside,
):
    mo.stop(not duplicate_run.value, mo.md("Press **Compare duplicates** to draw this view."))
    _names = MARKER_CHANNELS[duplicate_marker.value]
    _conditions = [_c for _c in selection["conditions"] if _c in duplicate_conditions.value]
    _kinds = [_kind for _kind in ("raw", "processed") if _kind in duplicate_images.value]
    _replicates = [
        _r
        for _r in sorted(duplicate_replicates.value)
        if any((_c, _r) in trajectories for _c in _conditions)
    ]
    _hours = sorted(duplicate_hours.value)
    _columns = [(_r, _h) for _r in _replicates for _h in _hours]
    mo.stop(
        not (_columns and _kinds),
        mo.md("Pick at least one image type, condition, timestep and replicate with images."),
    )
    _invalid = [_n for _n in _names if low_percentiles.value[_n] >= high_percentiles.value[_n]]
    mo.stop(_invalid, mo.md(f"Low % must be below high % for `{', '.join(_invalid)}`."))

    # Each compared unit is a (condition, name) pair. Labels leave out the
    # part that is the same for all of them: "ctrl s1", "s2", "sl0".
    _units = [(_c, _n) for _c in _conditions for _n in _names]

    def _label(unit):
        condition, name = unit
        group = name.split("/")[0].replace("cell_fate_", "").replace("_", " ")
        parts = ([condition] if len(_conditions) > 1 else []) + ([group] if len(_names) > 1 else [])
        return " ".join(parts) or group

    # {(unit, replicate, hour): {"index", "raw", "processed", "mask"}} at the
    # training resolution, centred as in section 3.
    _panels = {}
    for _unit in _units:
        _condition, _name = _unit
        _group, _channel = CHANNEL_SOURCE[_name]
        for _replicate in _replicates:
            _trajectory = (_condition, _replicate)
            if _trajectory not in trajectories:
                continue
            _processed = live_channel(_name, _trajectory)[1] if "processed" in _kinds else {}
            for _hour in _hours:
                _index = trajectories[_trajectory].get((_group, _hour))
                if _index is None:
                    continue
                _shift, _mask = image_geometry[_index]
                _raw = block_mean(raw_images[_index]["channels"][..., _channel].astype(np.float32))
                _panels[(_unit, _replicate, _hour)] = {
                    "index": _index,
                    "raw": place_image(_raw, _shift),
                    "processed": _processed[_hour][0] if _hour in _processed else None,
                    "mask": _mask,
                }
    mo.stop(not _panels, mo.md("None of the chosen images are loaded."))

    # Raw colour scale: shared by all units, or one per unit.
    _raw_limits = {}
    for _unit in _units:
        _scale_units = (_unit,) if duplicate_raw_scale.value else _units
        _values = [_p["raw"].ravel() for _k, _p in _panels.items() if _k[0] in _scale_units]
        if _values:
            _raw_limits[_unit] = np.percentile(np.concatenate(_values), (0.5, 99.5))

    def _display(key, kind):
        """One panel on a 0-1 colour scale."""
        panel = _panels[key]
        if kind == "processed":
            return panel["processed"] * panel["mask"] if zero_outside.value else panel["processed"]
        low, high = _raw_limits[key[0]]
        return np.clip((panel["raw"] - low) / max(high - low, 1e-12), 0.0, 1.0)

    def _clean_axis(axis):
        axis.set_xticks([])
        axis.set_yticks([])

    _cmap = plt.get_cmap("magma").copy()
    _cmap.set_bad("0.85")
    _column_titles = [f"replicate {_r + 1}, {_h}h" for _r, _h in _columns]

    if duplicate_layout.value == "rows":
        _rows = [(_kind, _unit) for _kind in _kinds for _unit in _units]
        _figure, _axes = plt.subplots(
            len(_rows), len(_columns), figsize=(2.4 * len(_columns), 2.4 * len(_rows)), squeeze=False
        )
        for _row, (_kind, _unit) in enumerate(_rows):
            _axes[_row, 0].set_ylabel(f"{_kind}\n{_label(_unit)}", fontsize=9)
            for _column, (_replicate, _hour) in enumerate(_columns):
                _axis = _axes[_row, _column]
                _clean_axis(_axis)
                if _row == 0:
                    _axis.set_title(_column_titles[_column], fontsize=9)
                _key = (_unit, _replicate, _hour)
                if _key not in _panels:
                    _axis.set_facecolor("0.85")
                    continue
                if _panels[_key]["index"] in flagged_indices:
                    outline_flagged(_axis)
                _axis.imshow(_display(_key, _kind), cmap=_cmap, vmin=0, vmax=1)
                if _kind == "raw" or not zero_outside.value:
                    _axis.contour(_panels[_key]["mask"], levels=[0.5], colors="cyan", linewidths=0.4)
    else:
        _figure, _axes = plt.subplots(
            len(_kinds), len(_columns), figsize=(3.0 * len(_columns), 3.0 * len(_kinds)), squeeze=False
        )
        _shape = next(iter(_panels.values()))["raw"].shape
        _centre = ndi.center_of_mass(next(iter(_panels.values()))["mask"])
        _rows_grid, _columns_grid = np.indices(_shape)
        # Clockwise angle from 12 o'clock, split into one segment per unit.
        _angle = np.mod(
            np.arctan2(_columns_grid - _centre[1], _centre[0] - _rows_grid), 2 * np.pi
        )
        _segment = np.minimum((_angle / (2 * np.pi) * len(_units)).astype(int), len(_units) - 1)
        _reach = max(_shape)
        for _row, _kind in enumerate(_kinds):
            _axes[_row, 0].set_ylabel(_kind, fontsize=9)
            for _column, (_replicate, _hour) in enumerate(_columns):
                _axis = _axes[_row, _column]
                _clean_axis(_axis)
                _composite = np.full(_shape, np.nan, dtype=np.float32)
                _flagged = []
                for _k, _unit in enumerate(_units):
                    _key = (_unit, _replicate, _hour)
                    if _key not in _panels:
                        continue
                    _inside = _segment == _k
                    _composite[_inside] = _display(_key, _kind)[_inside]
                    if _panels[_key]["index"] in flagged_indices:
                        _flagged.append(_label(_unit))
                _axis.imshow(_composite, cmap=_cmap, vmin=0, vmax=1)
                for _k, _unit in enumerate(_units):
                    _start = 2 * np.pi * _k / len(_units)
                    _middle = 2 * np.pi * (_k + 0.5) / len(_units)
                    if len(_units) > 1:
                        _axis.plot(
                            [_centre[1], _centre[1] + _reach * np.sin(_start)],
                            [_centre[0], _centre[0] - _reach * np.cos(_start)],
                            color="white",
                            linewidth=0.8,
                        )
                    _axis.text(
                        _centre[1] + 0.3 * _shape[1] * np.sin(_middle),
                        _centre[0] - 0.3 * _shape[0] * np.cos(_middle),
                        _label(_unit),
                        color="white",
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
                _axis.set_xlim(-0.5, _shape[1] - 0.5)
                _axis.set_ylim(_shape[0] - 0.5, -0.5)
                _title = _column_titles[_column] if _row == 0 else ""
                if _flagged:
                    _title += f"\nflagged: {', '.join(_flagged)}"
                    outline_flagged(_axis)
                _axis.set_title(_title, fontsize=8)
    _scale_text = "per row / segment" if duplicate_raw_scale.value else "shared"
    _figure.suptitle(
        f"{duplicate_marker.value}, {', '.join(_conditions)} (raw colour scale {_scale_text})",
        fontsize=10,
    )
    _figure.tight_layout()
    _views = [show_figure(_figure)]

    if duplicate_histograms.value:
        _inside_values = {
            (_kind, _key): _panels[_key][_kind][_panels[_key]["mask"]]
            for _kind in _kinds
            for _key in _panels
        }
        _bins = {"processed": np.linspace(0.0, 1.0, 51)}
        if "raw" in _kinds:
            # Shared raw bins up to the pooled 99.9th percentile; brighter
            # pixels are counted in the last bin.
            _raw_top = max(
                float(
                    np.percentile(
                        np.concatenate([_v for (_kind, _), _v in _inside_values.items() if _kind == "raw"]),
                        99.9,
                    )
                ),
                1.0,
            )
            _bins["raw"] = np.linspace(0.0, _raw_top, 81)
        _histograms, _hist_axes = plt.subplots(
            len(_kinds),
            len(_columns),
            figsize=(2.6 * len(_columns), 2.3 * len(_kinds)),
            squeeze=False,
            sharey="row",
        )
        for _row, _kind in enumerate(_kinds):
            _hist_axes[_row, 0].set_ylabel(f"{_kind}: pixel count", fontsize=8)
            for _column, (_replicate, _hour) in enumerate(_columns):
                _axis = _hist_axes[_row, _column]
                if _row == 0:
                    _axis.set_title(_column_titles[_column], fontsize=8)
                for _k, _unit in enumerate(_units):
                    _key = (_unit, _replicate, _hour)
                    if (_kind, _key) not in _inside_values:
                        continue
                    _axis.hist(
                        np.minimum(_inside_values[(_kind, _key)], _bins[_kind][-1]),
                        bins=_bins[_kind],
                        histtype="step",
                        color=f"C{_k}",
                        label=_label(_unit),
                    )
                _axis.set_yscale("log")
                _axis.tick_params(labelsize=6)
        _hist_axes[0, 0].legend(fontsize=7)
        _histograms.tight_layout()
        _views.append(show_figure(_histograms))

    # Differences between each pair of units, per column.
    _table = []
    for _replicate, _hour in _columns:
        for _i, _first in enumerate(_units):
            for _second in _units[_i + 1 :]:
                _a = _panels.get((_first, _replicate, _hour))
                _b = _panels.get((_second, _replicate, _hour))
                if _a is None or _b is None:
                    continue
                _difference = {
                    "replicate": _replicate + 1,
                    "timestep": f"{_hour}h",
                    "pair": f"{_label(_second)} vs {_label(_first)}",
                    "raw mean ratio": round(
                        float(_b["raw"][_b["mask"]].mean() / max(_a["raw"][_a["mask"]].mean(), 1e-12)), 3
                    ),
                }
                if _a["processed"] is not None and _b["processed"] is not None:
                    _both = _a["mask"] & _b["mask"]
                    _x, _y = _a["processed"][_both], _b["processed"][_both]
                    _difference["processed mean difference"] = round(float(_y.mean() - _x.mean()), 3)
                    _difference["processed mean |difference|"] = round(float(np.abs(_y - _x).mean()), 3)
                    _difference["processed correlation"] = (
                        round(float(np.corrcoef(_x, _y)[0, 1]), 3)
                        if _x.std() > 0 and _y.std() > 0
                        else None
                    )
                _table.append(_difference)
    if _table:
        _views.append(mo.ui.table(_table, selection=None))
    mo.vstack(_views)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalisation comparison
    """)
    return


@app.cell(hide_code=True)
def _(
    KNOCKOUT_BOUNDS,
    NORMALISATION_MODES,
    joint_missing,
    knockout_bounds,
    loaded_names,
    mo,
    normalise_mode,
    selection,
):
    bounds_channel = mo.ui.dropdown(options=loaded_names, value=loaded_names[0], label="Channel")
    bounds_conditions = mo.ui.multiselect(
        options=list(selection["conditions"]),
        value=list(selection["conditions"]),
        label="Conditions",
    )
    bounds_normalisations = mo.ui.multiselect(
        options=list(NORMALISATION_MODES),
        value=[normalise_mode.value],
        label="Normalisation",
    )
    bounds_knockout_modes = mo.ui.multiselect(
        options=[_mode for _mode in KNOCKOUT_BOUNDS if not (_mode == "joint" and joint_missing)],
        value=list(
            dict.fromkeys(
                [knockout_bounds.value]
                + [_mode for _mode in ("selected", "joint") if not (_mode == "joint" and joint_missing)]
            )
        ),
        label="Knockout bounds",
    )
    bounds_replicates = mo.ui.multiselect(
        options={f"replicate {_r + 1}": _r for _r in selection["replicates"]},
        value=[f"replicate {selection['replicates'][0] + 1}"],
        label="Replicates",
    )
    bounds_hours = mo.ui.multiselect(
        options={f"{_hour}h": _hour for _hour in selection["timesteps"]},
        value=[f"{_hour}h" for _hour in selection["timesteps"]],
        label="Timesteps",
    )
    bounds_layout = mo.ui.radio(
        options={"Circle segments": "segments", "Full rows": "rows"},
        value="Circle segments",
        label="Show as",
        inline=True,
    )
    bounds_show_raw = mo.ui.checkbox(value=False, label="Add raw images")
    bounds_run = mo.ui.run_button(label="Compare bounds")
    mo.vstack(
        [
            mo.md(
                "- One channel, normalised with each chosen way of setting "
                "the bounds: one row per normalisation and knockout bounds pair.\n"
                "- **Circle segments**: each condition fills one segment of "
                "the colony (needs centring, section 3). **Full rows**: one "
                "row per condition as well.\n"
                "- **Add raw images**: the images as stored (no intensity "
                "factor, cleaning, centring or mask), only averaged to the "
                "display resolution, on one colour scale (0.5–99.5 percentiles).\n"
                "- Other settings as in sections 2 and 5. The table lists the "
                "bounds of each trajectory."
                + ("\n- *joint* needs sl0 and sl24 loaded." if joint_missing else "")
            ),
            mo.hstack([bounds_channel, bounds_conditions, bounds_replicates, bounds_hours], justify="start"),
            mo.hstack(
                [bounds_normalisations, bounds_knockout_modes, bounds_layout, bounds_show_raw],
                justify="start",
            ),
            bounds_run,
        ]
    )
    return (
        bounds_channel,
        bounds_conditions,
        bounds_hours,
        bounds_knockout_modes,
        bounds_layout,
        bounds_normalisations,
        bounds_replicates,
        bounds_run,
        bounds_show_raw,
    )


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    KNOCKOUT_BOUNDS,
    NORMALISATION_MODES,
    block_mean,
    bounds_channel,
    bounds_conditions,
    bounds_hours,
    bounds_knockout_modes,
    bounds_layout,
    bounds_normalisations,
    bounds_replicates,
    bounds_run,
    bounds_show_raw,
    flagged_indices,
    high_percentiles,
    live_channel,
    low_percentiles,
    mo,
    ndi,
    np,
    outline_flagged,
    plt,
    raw_images,
    selection,
    show_figure,
    trajectories,
    trajectory_label,
    zero_outside,
):
    mo.stop(not bounds_run.value, mo.md("Press **Compare bounds** to draw this view."))
    _name = bounds_channel.value
    _group = CHANNEL_SOURCE[_name][0]
    _conditions = [_c for _c in selection["conditions"] if _c in bounds_conditions.value]
    # Each row of the figure is one (normalisation, knockout bounds) pair.
    _settings = [
        (_normalisation, _mode)
        for _normalisation in NORMALISATION_MODES
        if _normalisation in bounds_normalisations.value
        for _mode in KNOCKOUT_BOUNDS
        if _mode in bounds_knockout_modes.value
    ]
    _replicates = [
        _r
        for _r in sorted(bounds_replicates.value)
        if any((_c, _r) in trajectories for _c in _conditions)
    ]
    _columns = [(_r, _h) for _r in _replicates for _h in sorted(bounds_hours.value)]
    mo.stop(
        not (_columns and _settings and _conditions),
        mo.md("Pick at least one condition, normalisation, knockout bounds, timestep and replicate."),
    )
    mo.stop(
        low_percentiles.value[_name] >= high_percentiles.value[_name],
        mo.md(f"Low % must be below high % for `{_name}`."),
    )

    # {(setting, condition, replicate, hour): (image, mask, index)} and the
    # bounds of each (setting, trajectory).
    _panels, _bounds = {}, {}
    for _setting in _settings:
        for _condition in _conditions:
            for _replicate in _replicates:
                _trajectory = (_condition, _replicate)
                if _trajectory not in trajectories:
                    continue
                _bounds[(_setting, _trajectory)], _images = live_channel(_name, _trajectory, *_setting)
                for _replicate_hour in _columns:
                    _hour = _replicate_hour[1]
                    if _replicate_hour[0] != _replicate or _hour not in _images:
                        continue
                    _image, _mask = _images[_hour]
                    _panels[(_setting, _condition, _replicate, _hour)] = (
                        _image * _mask if zero_outside.value else _image,
                        _mask,
                        trajectories[_trajectory][(_group, _hour)],
                    )
    mo.stop(not _panels, mo.md("None of the chosen images are loaded."))
    # Geometry of the circle segments, from a processed (centred) image.
    _first_image, _first_mask, _ = next(iter(_panels.values()))

    # Raw images as stored, only averaged to the display resolution, on one
    # shared 0-1 colour scale. Their row has the setting None.
    if bounds_show_raw.value:
        _channel = CHANNEL_SOURCE[_name][1]
        _raw = {}
        for _condition in _conditions:
            for _replicate, _hour in _columns:
                _index = trajectories.get((_condition, _replicate), {}).get((_group, _hour))
                if _index is not None:
                    _raw[(_condition, _replicate, _hour)] = (
                        block_mean(raw_images[_index]["channels"][..., _channel].astype(np.float32)),
                        _index,
                    )
        if _raw:
            _low, _high = np.percentile(
                np.concatenate([_values.ravel() for _values, _ in _raw.values()]), (0.5, 99.5)
            )
            for (_condition, _replicate, _hour), (_values, _index) in _raw.items():
                _panels[(None, _condition, _replicate, _hour)] = (
                    np.clip((_values - _low) / max(_high - _low, 1e-12), 0.0, 1.0),
                    None,
                    _index,
                )
    _row_settings = ([None] if bounds_show_raw.value else []) + _settings

    def _setting_label(setting):
        return "raw" if setting is None else f"{setting[0]}\n{setting[1]}"

    def _clean_axis(axis):
        axis.set_xticks([])
        axis.set_yticks([])

    _cmap = plt.get_cmap("magma").copy()
    _cmap.set_bad("0.85")
    _column_titles = [f"replicate {_r + 1}, {_h}h" for _r, _h in _columns]

    if bounds_layout.value == "rows":
        _rows = [(_setting, _condition) for _setting in _row_settings for _condition in _conditions]
        _figure, _axes = plt.subplots(
            len(_rows), len(_columns), figsize=(2.4 * len(_columns), 2.4 * len(_rows)), squeeze=False
        )
        for _row, (_setting, _condition) in enumerate(_rows):
            _axes[_row, 0].set_ylabel(f"{_setting_label(_setting)}\n{_condition}", fontsize=8)
            for _column, (_replicate, _hour) in enumerate(_columns):
                _axis = _axes[_row, _column]
                _clean_axis(_axis)
                if _row == 0:
                    _axis.set_title(_column_titles[_column], fontsize=9)
                _key = (_setting, _condition, _replicate, _hour)
                if _key not in _panels:
                    _axis.set_facecolor("0.85")
                    continue
                _image, _mask, _index = _panels[_key]
                if _index in flagged_indices:
                    outline_flagged(_axis)
                _axis.imshow(_image, cmap=_cmap, vmin=0, vmax=1)
                if _mask is not None and not zero_outside.value:
                    _axis.contour(_mask, levels=[0.5], colors="cyan", linewidths=0.4)
    else:
        _figure, _axes = plt.subplots(
            len(_row_settings), len(_columns), figsize=(3.0 * len(_columns), 3.0 * len(_row_settings)), squeeze=False
        )
        _shape = _first_image.shape
        _centre = ndi.center_of_mass(_first_mask)
        _rows_grid, _columns_grid = np.indices(_shape)
        # Clockwise angle from 12 o'clock, split into one segment per condition.
        _angle = np.mod(np.arctan2(_columns_grid - _centre[1], _centre[0] - _rows_grid), 2 * np.pi)
        _segment = np.minimum((_angle / (2 * np.pi) * len(_conditions)).astype(int), len(_conditions) - 1)
        _reach = max(_shape)
        for _row, _setting in enumerate(_row_settings):
            _axes[_row, 0].set_ylabel(_setting_label(_setting), fontsize=8)
            for _column, (_replicate, _hour) in enumerate(_columns):
                _axis = _axes[_row, _column]
                _clean_axis(_axis)
                _composite = np.full(_shape, np.nan, dtype=np.float32)
                _flagged = []
                for _k, _condition in enumerate(_conditions):
                    _key = (_setting, _condition, _replicate, _hour)
                    if _key not in _panels:
                        continue
                    _inside = _segment == _k
                    _composite[_inside] = _panels[_key][0][_inside]
                    if _panels[_key][2] in flagged_indices:
                        _flagged.append(_condition)
                _axis.imshow(_composite, cmap=_cmap, vmin=0, vmax=1)
                for _k, _condition in enumerate(_conditions):
                    _start = 2 * np.pi * _k / len(_conditions)
                    _middle = 2 * np.pi * (_k + 0.5) / len(_conditions)
                    if len(_conditions) > 1:
                        _axis.plot(
                            [_centre[1], _centre[1] + _reach * np.sin(_start)],
                            [_centre[0], _centre[0] - _reach * np.cos(_start)],
                            color="white",
                            linewidth=0.8,
                        )
                    _axis.text(
                        _centre[1] + 0.3 * _shape[1] * np.sin(_middle),
                        _centre[0] - 0.3 * _shape[0] * np.cos(_middle),
                        _condition,
                        color="white",
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
                _axis.set_xlim(-0.5, _shape[1] - 0.5)
                _axis.set_ylim(_shape[0] - 0.5, -0.5)
                _title = _column_titles[_column] if _row == 0 else ""
                if _flagged:
                    _title += f"\nflagged: {', '.join(_flagged)}"
                    outline_flagged(_axis)
                _axis.set_title(_title, fontsize=8)
    _figure.suptitle(f"{_name}, normalised (rows: normalisation / knockout bounds)", fontsize=10)
    _figure.tight_layout()

    _table = [
        {
            "normalisation": _setting[0],
            "knockout bounds": _setting[1],
            "trajectory": trajectory_label(_trajectory),
            "low": round(_low, 2),
            "high": round(_high, 2),
        }
        for (_setting, _trajectory), (_low, _high) in _bounds.items()
    ]
    mo.vstack([show_figure(_figure), mo.ui.table(_table, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ball size comparison
    """)
    return


@app.cell(hide_code=True)
def _(loaded_names, mo, selection, trajectories, trajectory_label):
    ball_channel = mo.ui.dropdown(options=loaded_names, value=loaded_names[0], label="Channel")
    ball_trajectory = mo.ui.dropdown(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        value=trajectory_label(next(iter(trajectories))),
        label="Trajectory",
    )
    ball_hour = mo.ui.dropdown(
        options={f"{_hour}h": _hour for _hour in selection["timesteps"]},
        value=f"{selection['timesteps'][-1]}h",
        label="Timestep",
    )
    ball_radii = mo.ui.text(
        value="0, 25, 50, 100, 200, 400",
        label="Ball radii (full-resolution pixels, comma separated)",
        full_width=True,
    )
    ball_run = mo.ui.run_button(label="Compare ball sizes")
    mo.vstack(
        [
            mo.md(
                "The same image with the background removed using several "
                "ball radii, at full resolution. The intensity factor, hot pixel "
                "replacement and shrink factor from section 2 are applied; "
                "the channel's own ball radius is ignored here. Radius 0 means "
                "no background removal. Top row: background removed, all on "
                "one colour scale, so you can see how much signal each radius "
                "takes away. Bottom row: the estimated background, on the "
                "colour scale of the image before removal. Large radii take a "
                "few seconds each the first time."
            ),
            mo.hstack([ball_channel, ball_trajectory, ball_hour], justify="start"),
            ball_radii,
            ball_run,
        ]
    )
    return ball_channel, ball_hour, ball_radii, ball_run, ball_trajectory


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    ball_background,
    ball_channel,
    ball_hour,
    ball_radii,
    ball_run,
    ball_trajectory,
    cleaning_settings,
    despeckle_settings,
    despeckled_channel,
    mo,
    np,
    plt,
    raw_images,
    show_figure,
    trajectories,
    trajectory_label,
):
    mo.stop(not ball_run.value, mo.md("Press **Compare ball sizes** to draw this view. It does not redraw by itself when the controls change."))
    _name = ball_channel.value
    _group, _channel = CHANNEL_SOURCE[_name]
    _index = trajectories[ball_trajectory.value].get((_group, ball_hour.value))
    mo.stop(
        _index is None,
        mo.md(f"`{_name}` was not measured at {ball_hour.value}h for this trajectory."),
    )
    try:
        _radii = [int(float(_value)) for _value in ball_radii.value.split(",") if _value.strip()]
    except ValueError:
        _radii = []
    mo.stop(
        not _radii or min(_radii) < 0,
        mo.md("Enter one or more non-negative radii, separated by commas."),
    )

    _settings = despeckle_settings(cleaning_settings, _index, _name)
    _image = despeckled_channel(_index, _channel, *_settings)
    _backgrounds = {}
    for _radius in mo.status.progress_bar(_radii, title="Rolling balls", remove_on_exit=True):
        _backgrounds[_radius] = (
            np.zeros_like(_image)
            if _radius == 0
            else ball_background(_index, _channel, *_settings, _radius, cleaning_settings["shrink"])
        )
    _cleaned = {_radius: np.clip(_image - _bg, 0.0, None) for _radius, _bg in _backgrounds.items()}

    _low, _high = np.percentile(_image, (0.5, 99.5))
    _clean_high = max(max(np.percentile(_c, 99.5) for _c in _cleaned.values()), 1e-6)
    _figure, _axes = plt.subplots(
        2, len(_radii), figsize=(2.8 * len(_radii), 5.9), squeeze=False
    )
    for _column, _radius in enumerate(_radii):
        _axes[0, _column].imshow(_cleaned[_radius], cmap="magma", vmin=0, vmax=_clean_high)
        _axes[0, _column].set_title(
            "no removal" if _radius == 0 else f"radius {_radius}", fontsize=9
        )
        _axes[1, _column].imshow(_backgrounds[_radius], cmap="magma", vmin=_low, vmax=_high)
        for _axis in _axes[:, _column]:
            _axis.set_xticks([])
            _axis.set_yticks([])
    _axes[0, 0].set_ylabel("background removed", fontsize=9)
    _axes[1, 0].set_ylabel("background", fontsize=9)
    _figure.suptitle(
        f"{_name}, {trajectory_label(ball_trajectory.value)}, {ball_hour.value}h", fontsize=10
    )
    _figure.tight_layout()

    # Profile through the middle of the colony.
    _row = int(round(np.mean(np.nonzero(raw_images[_index]["foreground"])[0])))
    _profile, _profile_axis = plt.subplots(figsize=(10, 3))
    _profile_axis.plot(_image[_row], color="grey", linewidth=0.5, label="image")
    _nonzero = [_radius for _radius in _radii if _radius > 0]
    for _number, _radius in enumerate(_nonzero):
        _profile_axis.plot(
            _image[_row] - _backgrounds[_radius][_row],
            color=plt.cm.viridis(_number / max(len(_nonzero) - 1, 1)),
            label=f"background, radius {_radius}",
        )
    _profile_axis.set_xlabel("column (full-resolution pixels)", fontsize=8)
    _profile_axis.set_title(f"Profile along row {_row}", fontsize=9)
    _profile_axis.legend(fontsize=7)
    _profile.tight_layout()
    mo.vstack([show_figure(_figure,dpi=200), _profile])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hot pixel threshold comparison
    """)
    return


@app.cell(hide_code=True)
def _(loaded_names, mo, selection, trajectories, trajectory_label):
    hot_channel = mo.ui.dropdown(options=loaded_names, value=loaded_names[0], label="Channel")
    hot_trajectory = mo.ui.dropdown(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        value=trajectory_label(next(iter(trajectories))),
        label="Trajectory",
    )
    hot_hour = mo.ui.dropdown(
        options={f"{_hour}h": _hour for _hour in selection["timesteps"]},
        value=f"{selection['timesteps'][-1]}h",
        label="Timestep",
    )
    hot_values = mo.ui.text(
        value="0, 5, 10, 20, 40",
        label="Hot pixel thresholds (noise levels, comma separated)",
        full_width=True,
    )
    hot_zoom = mo.ui.slider(
        40, 400, step=20, value=120, label="Zoom window (full-resolution pixels)", show_value=True
    )
    hot_run = mo.ui.run_button(label="Compare thresholds")
    mo.vstack(
        [
            mo.md(
                "The same image with hot pixels replaced at several "
                "thresholds, at full resolution. The intensity factor and hot pixel "
                "window from section 2 are applied; the channel's own "
                "threshold is ignored here, and no background is removed. "
                "Threshold 0 means no replacement. Top row: the whole image, "
                "all on one colour scale, with the number of replaced pixels. "
                "Bottom row: a close-up around the brightest pixel of the "
                "image (often a speck), with replaced pixels circled in green. "
                "Each threshold takes about 0.4 s the first time."
            ),
            mo.hstack([hot_channel, hot_trajectory, hot_hour], justify="start"),
            hot_values,
            hot_zoom,
            hot_run,
        ]
    )
    return hot_channel, hot_hour, hot_run, hot_trajectory, hot_values, hot_zoom


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    cleaning_settings,
    despeckle_settings,
    despeckled_channel,
    hot_channel,
    hot_hour,
    hot_run,
    hot_trajectory,
    hot_values,
    hot_zoom,
    mo,
    np,
    plt,
    show_figure,
    trajectories,
    trajectory_label,
):
    mo.stop(not hot_run.value, mo.md("Press **Compare thresholds** to draw this view. It does not redraw by itself when the controls change."))
    _name = hot_channel.value
    _group, _channel = CHANNEL_SOURCE[_name]
    _index = trajectories[hot_trajectory.value].get((_group, hot_hour.value))
    mo.stop(
        _index is None,
        mo.md(f"`{_name}` was not measured at {hot_hour.value}h for this trajectory."),
    )
    try:
        _thresholds = [float(_value) for _value in hot_values.value.split(",") if _value.strip()]
    except ValueError:
        _thresholds = []
    mo.stop(
        not _thresholds or min(_thresholds) < 0,
        mo.md("Enter one or more non-negative thresholds, separated by commas."),
    )

    _scale, _, _size, *_ = despeckle_settings(cleaning_settings, _index, _name)
    _original = despeckled_channel(_index, _channel, _scale, 0, _size)
    _results = {}
    for _threshold in mo.status.progress_bar(
        _thresholds, title="Replacing hot pixels", remove_on_exit=True
    ):
        _results[_threshold] = despeckled_channel(_index, _channel, _scale, _threshold, _size)

    # Close-up window around the brightest raw pixel.
    _centre = np.unravel_index(np.argmax(_original), _original.shape)
    _half = hot_zoom.value // 2
    _top = int(np.clip(_centre[0] - _half, 0, _original.shape[0] - 2 * _half))
    _left = int(np.clip(_centre[1] - _half, 0, _original.shape[1] - 2 * _half))
    _rows = slice(_top, _top + 2 * _half)
    _columns = slice(_left, _left + 2 * _half)

    _low, _high = np.percentile(_original, (0.5, 99.5))
    _zoom_high = max(float(np.max(_original[_rows, _columns])), _high)
    _figure, _axes = plt.subplots(
        2, len(_thresholds), figsize=(2.8 * len(_thresholds), 6.0), squeeze=False
    )
    for _column, _threshold in enumerate(_thresholds):
        _image = _results[_threshold]
        _replaced = _image != _original
        _whole, _close = _axes[:, _column]
        _whole.imshow(_image, cmap="magma", vmin=_low, vmax=_high)
        _whole.add_patch(
            plt.Rectangle(
                (_left, _top),
                2 * _half,
                2 * _half,
                fill=False,
                edgecolor="cyan",
                linewidth=0.8,
            )
        )
        _whole.set_title(
            ("no replacement" if _threshold == 0 else f"threshold {_threshold:g}")
            + f"\n{int(_replaced.sum())} pixels replaced",
            fontsize=8,
        )
        # The close-up uses the full range of the window, so specks stand out.
        _close.imshow(_image[_rows, _columns], cmap="magma", vmin=_low, vmax=_zoom_high)
        _spot_rows, _spot_columns = np.nonzero(_replaced[_rows, _columns])
        _close.scatter(
            _spot_columns, _spot_rows, s=30, facecolors="none", edgecolors="lime", linewidths=0.7
        )
        _close.set_xlim(-0.5, 2 * _half - 0.5)
        _close.set_ylim(2 * _half - 0.5, -0.5)
        for _axis in (_whole, _close):
            _axis.set_xticks([])
            _axis.set_yticks([])
    _axes[0, 0].set_ylabel("whole image", fontsize=9)
    _axes[1, 0].set_ylabel("close-up (cyan box)", fontsize=9)
    _figure.suptitle(
        f"{_name}, {trajectory_label(hot_trajectory.value)}, {hot_hour.value}h, "
        f"window {_size} px",
        fontsize=10,
    )
    _figure.tight_layout()
    show_figure(_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stripe removal comparison
    """)
    return


@app.cell(hide_code=True)
def _(loaded_names, mo, selection, trajectories, trajectory_label):
    stripe_channel = mo.ui.dropdown(options=loaded_names, value=loaded_names[0], label="Channel")
    stripe_trajectory = mo.ui.dropdown(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        value=trajectory_label(next(iter(trajectories))),
        label="Trajectory",
    )
    stripe_hour = mo.ui.dropdown(
        options={f"{_hour}h": _hour for _hour in selection["timesteps"]},
        value=f"{selection['timesteps'][-1]}h",
        label="Timestep",
    )
    stripe_direction = mo.ui.dropdown(
        options=["horizontal", "vertical"], value="horizontal", label="Bands"
    )
    stripe_values = mo.ui.text(
        value="0, 10, 20, 40, 80",
        label="Stripe widths (full-resolution pixels, comma separated)",
        full_width=True,
    )
    stripe_run = mo.ui.run_button(label="Compare stripe widths")
    mo.vstack(
        [
            mo.md(
                "- Removes bands of raised background narrower than the width "
                "(e.g. stitching seams in sum projections). Width 0 = off.\n"
                "- Intensity factor, hot pixels and stripe block from section 2 "
                "are applied; the channel's own stripe widths and ball radius "
                "are not.\n"
                "- Top: after removal, colour scale capped at the 90th "
                "percentile so the background shows. Bottom: what was removed.\n"
                "- Line plot: background level of each row (10th percentile) "
                "before and after; with a stripe block, only over the piece "
                "of the row around the strongest stripe."
            ),
            mo.hstack(
                [stripe_channel, stripe_trajectory, stripe_hour, stripe_direction],
                justify="start",
            ),
            stripe_values,
            stripe_run,
        ]
    )
    return (
        stripe_channel,
        stripe_direction,
        stripe_hour,
        stripe_run,
        stripe_trajectory,
        stripe_values,
    )


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    cleaning_settings,
    despeckle_settings,
    despeckled_channel,
    mo,
    np,
    plt,
    remove_stripes,
    show_figure,
    stripe_channel,
    stripe_direction,
    stripe_hour,
    stripe_run,
    stripe_trajectory,
    stripe_values,
    trajectories,
    trajectory_label,
):
    mo.stop(not stripe_run.value, mo.md("Press **Compare stripe widths** to draw this view. It does not redraw by itself when the controls change."))
    _name = stripe_channel.value
    _group, _channel = CHANNEL_SOURCE[_name]
    _index = trajectories[stripe_trajectory.value].get((_group, stripe_hour.value))
    mo.stop(
        _index is None,
        mo.md(f"`{_name}` was not measured at {stripe_hour.value}h for this trajectory."),
    )
    try:
        _widths = [int(float(_value)) for _value in stripe_values.value.split(",") if _value.strip()]
    except ValueError:
        _widths = []
    mo.stop(
        not _widths or min(_widths) < 0,
        mo.md("Enter one or more non-negative widths, separated by commas."),
    )

    _scale, _hot, _size, _, _, _block = despeckle_settings(cleaning_settings, _index, _name)
    _image = despeckled_channel(_index, _channel, _scale, _hot, _size)
    _results = {
        _width: (_image, np.zeros_like(_image))
        if _width == 0
        else remove_stripes(_image, _width, stripe_direction.value, _block)
        for _width in _widths
    }

    _low, _high = np.percentile(_image, (1, 90))
    _limit = max(max(float(np.abs(_s).max()) for _, _s in _results.values()), 1e-6)
    _figure, _axes = plt.subplots(
        2, len(_widths), figsize=(2.8 * len(_widths), 5.9), squeeze=False
    )
    for _column, _width in enumerate(_widths):
        _cleaned, _stripes = _results[_width]
        _axes[0, _column].imshow(_cleaned, cmap="magma", vmin=_low, vmax=_high)
        _axes[0, _column].set_title("no removal" if _width == 0 else f"width {_width}", fontsize=9)
        _axes[1, _column].imshow(_stripes, cmap="RdBu_r", vmin=-_limit, vmax=_limit)
        for _axis in _axes[:, _column]:
            _axis.set_xticks([])
            _axis.set_yticks([])
    _axes[0, 0].set_ylabel("stripes removed", fontsize=9)
    _axes[1, 0].set_ylabel("removed", fontsize=9)
    _figure.suptitle(
        f"{_name}, {trajectory_label(stripe_trajectory.value)}, {stripe_hour.value}h, "
        f"{stripe_direction.value} bands",
        fontsize=10,
    )
    _figure.tight_layout()

    # Profile over the whole row, or with a stripe block, over the piece of the
    # row (or column) around the strongest stripe removed.
    _horizontal = stripe_direction.value == "horizontal"
    _along = 1 if _horizontal else 0
    _nonzero = [_width for _width in _widths if _width > 0]
    _span = slice(None)
    if _block and _nonzero:
        _strongest = np.unravel_index(
            np.argmax(np.abs(_results[max(_nonzero)][1])), _image.shape
        )[_along]
        _start = int(np.clip(_strongest - _block // 2, 0, _image.shape[_along] - _block))
        _span = slice(_start, _start + _block)

    def _background_profile(values):
        values = values[:, _span] if _horizontal else values[_span, :]
        return np.percentile(values, 10, axis=_along)

    _profile, _profile_axis = plt.subplots(figsize=(10, 3))
    _profile_axis.plot(_background_profile(_image), color="grey", linewidth=0.8, label="before")
    for _number, _width in enumerate(_nonzero):
        _profile_axis.plot(
            _background_profile(_results[_width][0]),
            color=plt.cm.viridis(_number / max(len(_nonzero) - 1, 1)),
            linewidth=0.8,
            label=f"width {_width}",
        )
    _profile_axis.set_xlabel(
        ("row" if _along == 1 else "column") + " (full-resolution pixels)", fontsize=8
    )
    _profile_axis.set_title(
        "Background level (10th percentile)"
        + ("" if _span == slice(None) else f", {'columns' if _horizontal else 'rows'} {_span.start}-{_span.stop}"),
        fontsize=9,
    )
    _profile_axis.legend(fontsize=7)
    _profile.tight_layout()
    mo.vstack([show_figure(_figure, dpi=200), _profile])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell type prevalence
    """)
    return


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    MICROPATTERN_260726_SCHEMA,
    get_loaded_cell_types,
    loaded_names,
    mo,
    trajectories,
    trajectory_label,
):
    FATE_MARKERS = ("SOX17", "SOX2", "TBXT", "FOXA2")
    # Each stain (cell-fate group) images some of the markers together, in the
    # same colonies: {stain: {marker: channel}} for the loaded groups.
    _marker_of = {
        _channel.name: _channel.marker
        for _channel in MICROPATTERN_260726_SCHEMA.measurement_channels
    }
    _stains = {}
    for _name in loaded_names:
        if _marker_of[_name] in FATE_MARKERS:
            _stains.setdefault(CHANNEL_SOURCE[_name][0], {})[_marker_of[_name]] = _name
    FATE_STAINS = {
        _stain: {_m: _channels[_m] for _m in FATE_MARKERS if _m in _channels}
        for _stain, _channels in _stains.items()
    }
    # One channel per marker, recorded as `channels` in the cell type file for
    # labelling a single image that holds every marker (same defaults as the
    # cell-fate explorer in Experiments/dataset_explorer.py).
    _preferred = {
        "SOX17": "cell_fate_s2/SOX17",
        "SOX2": "cell_fate_s1/SOX2",
        "TBXT": "cell_fate_s2/TBXT",
        "FOXA2": "cell_fate_s2/FOXA2",
    }
    FATE_CHANNELS = {}
    for _marker in FATE_MARKERS:
        _found = [_channels[_marker] for _channels in FATE_STAINS.values() if _marker in _channels]
        if _found:
            FATE_CHANNELS[_marker] = _preferred[_marker] if _preferred[_marker] in _found else _found[0]

    # Each cell type has one or more clauses, joined by "or".
    _default_rules = {
        "Notochord": [{"SOX17": "low", "SOX2": "any", "TBXT": "high", "FOXA2": "high"}],
        "Endoderm": [{"SOX17": "high", "SOX2": "any", "TBXT": "any", "FOXA2": "any"}],
        # TBXT+/FOXA2-/SOX17- (checked in stain 2) or TBXT+/SOX2+ (stain 1).
        "Mesoderm": [
            {"SOX17": "low", "SOX2": "any", "TBXT": "high", "FOXA2": "low"},
            {"SOX17": "any", "SOX2": "high", "TBXT": "high", "FOXA2": "any"},
        ],
    }
    # A loaded cell type file replaces the default rules.
    _loaded = get_loaded_cell_types()
    if _loaded is not None:
        _default_rules = {
            _cell_type: [{_m: _clause.get(_m, "any") for _m in FATE_MARKERS} for _clause in _clauses]
            for _cell_type, _clauses in _loaded.rules.items()
        }
    _states = {"High": "high", "Low": "low", "Indifferent": "any"}
    _state_labels = {_value: _label for _label, _value in _states.items()}
    # {cell type: {clause number: {marker: dropdown}}}
    fate_rules = mo.ui.dictionary(
        {
            _cell_type: mo.ui.dictionary(
                {
                    str(_i): mo.ui.dictionary(
                        {
                            _marker: mo.ui.dropdown(
                                options=_states, value=_state_labels[_clause[_marker]]
                            )
                            for _marker in FATE_MARKERS
                        }
                    )
                    for _i, _clause in enumerate(_clauses)
                }
            )
            for _cell_type, _clauses in _default_rules.items()
        }
    )

    def rule_clauses(rules_value):
        """``{cell type: [clause, ...]}`` from the value of ``fate_rules``."""
        return {
            _cell_type: [_clauses[_key] for _key in sorted(_clauses, key=int)]
            for _cell_type, _clauses in rules_value.items()
        }

    fate_trajectories = mo.ui.multiselect(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        # value=[trajectory_label(_trajectory) for _trajectory in trajectories],
        value = [],
        label="Trajectories (one row each)",
        full_width=True,
    )
    fate_show_channels = mo.ui.checkbox(
        value=False, label="Also show each stain's channels after thresholding"
    )
    # The cell types the rules define, and how many rows (clauses) each has
    # (fixed until a cell type file is loaded).
    FATE_CELL_TYPES = tuple(_default_rules)
    FATE_CLAUSE_COUNTS = {_cell_type: len(_clauses) for _cell_type, _clauses in _default_rules.items()}
    return (
        FATE_CELL_TYPES,
        FATE_CHANNELS,
        FATE_CLAUSE_COUNTS,
        FATE_MARKERS,
        FATE_STAINS,
        fate_rules,
        fate_show_channels,
        fate_trajectories,
        rule_clauses,
    )


@app.cell(hide_code=True)
def _(FATE_MARKERS, get_threshold_defaults, mo):
    # Threshold sliders, in a cell of their own so that a fit or a loaded file
    # can set them without resetting the other cell-type controls.
    _defaults = {_marker: 0.3 for _marker in FATE_MARKERS}
    _defaults.update({_m: _t for _m, _t in get_threshold_defaults().items() if _m in _defaults})
    fate_thresholds = mo.ui.dictionary(
        {
            _marker: mo.ui.slider(
                0.0,
                1.0,
                step=0.01,
                value=round(min(max(_defaults[_marker], 0.0), 1.0), 2),
                show_value=True,
            )
            for _marker in FATE_MARKERS
        }
    )
    return (fate_thresholds,)


@app.cell(hide_code=True)
def _(FATE_CLAUSE_COUNTS, get_weight_defaults, mo):
    # Row weights of the cell types with several rows (1 = the row fully
    # counts, 0 = ignored), in a cell of their own so that a fit or a loaded
    # file can set them without resetting the rules.
    _saved = get_weight_defaults()

    def _start(cell_type, row, rows):
        weights = _saved.get(cell_type, ())
        return round(20 * float(weights[row])) / 20 if len(weights) == rows else 1.0

    fate_weights = mo.ui.dictionary(
        {
            _cell_type: mo.ui.dictionary(
                {
                    str(_row): mo.ui.slider(
                        0.0, 1.0, step=0.05, value=_start(_cell_type, _row, _rows), show_value=True
                    )
                    for _row in range(_rows)
                }
            )
            for _cell_type, _rows in FATE_CLAUSE_COUNTS.items()
            if _rows > 1
        }
    )
    return (fate_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cell types at 48h
    """)
    return


@app.cell(hide_code=True)
def _(
    FATE_CELL_TYPES,
    FATE_MARKERS,
    FATE_STAINS,
    fate_rules,
    fate_show_channels,
    fate_thresholds,
    fate_trajectories,
    fate_weights,
    mo,
):
    _marker_table = mo.vstack(
        [mo.hstack([mo.md("**marker**"), mo.md("**imaged in**"), mo.md("**high above**")], widths=[1, 3, 3])]
        + [
            mo.hstack(
                [
                    mo.md(f"`{_marker}`"),
                    mo.md(
                        ", ".join(
                            f"`{_channels[_marker]}`"
                            for _channels in FATE_STAINS.values()
                            if _marker in _channels
                        )
                        or "*not loaded*"
                    ),
                    fate_thresholds[_marker],
                ],
                widths=[1, 3, 3],
                align="center",
            )
            for _marker in FATE_MARKERS
        ]
    )
    _rule_rows = [
        mo.hstack(
            [mo.md("**cell type**")] + [mo.md(f"**{_m}**") for _m in FATE_MARKERS] + [mo.md("**row weight**")],
            widths="equal",
        )
    ]
    for _cell_type in FATE_CELL_TYPES:
        for _i in range(len(fate_rules.value[_cell_type])):
            _rule_rows.append(
                mo.hstack(
                    [mo.md(_cell_type if _i == 0 else f"*or* {_cell_type}")]
                    + [fate_rules[_cell_type][str(_i)][_m] for _m in FATE_MARKERS]
                    + [fate_weights[_cell_type][str(_i)] if _cell_type in fate_weights else mo.md("")],
                    widths="equal",
                    align="center",
                )
            )
    mo.vstack(
        [
            mo.md(
                "Pixels are labelled from the processed (normalised 0–1) "
                "marker images at 48h, so the thresholds follow every control "
                "above, including normalisation. A marker is **high** where its "
                "value is above its threshold. A cell type is assigned where "
                "all marker states of one of its rows hold (*Indifferent* = "
                "not checked); a cell type with several rows matches if any "
                "row holds (*or*). Pixels matching no cell type are *Other*, "
                "and pixels matching more than one are *Several*.\n\n"
                "**Row weights** (cell types with several rows) say how much "
                "each row counts in the estimated shares: a cell matching "
                "rows with weights w1, w2 counts as that cell type with "
                "probability 1 − (1 − w1)(1 − w2), so weights of 1 give the "
                "plain *or* and 0 switches a row off.\n\n"
                "**Two stains.** The markers were imaged in two stains, in "
                "different colonies: SOX2 only in stain 1 and FOXA2 only in "
                "stain 2, while SOX17 and TBXT are in both (one threshold "
                "each, used for both stains). Pixels of different colonies "
                "are not the same cells, so a rule cannot be checked by "
                "putting the two stains' images on top of each other. The "
                "shares are estimated instead, ring by ring: within a ring, the share of "
                "every high/low combination of all four markers is estimated "
                "by assuming that SOX2 and FOXA2 are unrelated once SOX17 and "
                "TBXT are known (e.g. among TBXT+/SOX17- pixels, being SOX2+ "
                "says nothing about being FOXA2-). Each cell type then gets "
                "the share of the combinations its rows match, so rules that "
                "mix the stains (like the two mesoderm rows) are counted "
                "once. Model output, which has every marker in every pixel, "
                "should be scored with the same estimate (see "
                "`Common/dataloader/cell_type_shares.py`).\n\n"
                "Below, each selected trajectory is a column with one row "
                "per stain, every pixel labelled from that stain alone, "
                "using only the rows it can check (rows with weight 0 left "
                "out). A row that needs a marker the stain lacks is not "
                "checked there, so e.g. notochord (needs FOXA2) only appears "
                "in stain 2, and a pixel can be *Other* in one stain while a "
                "row checked in the other stain would match. The bar chart "
                "underneath gives the whole-colony shares estimated from both "
                "stains together. Tick the box to also see each stain's "
                "channels after thresholding.\n\n"
                "Select several trajectories to compare replicates, and "
                "control against knockout, side by side (load the knockout "
                "conditions in section 1). **Fit thresholds to radial "
                "targets** (below) sets the sliders' starting values, which "
                "you can then tweak. Export the thresholds and rules under "
                "**Export and load cell types**."
            ),
            fate_trajectories,
            fate_show_channels,
            _marker_table,
            mo.md("**Rules**"),
            mo.vstack(_rule_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    CELL_TYPE_HOUR,
    CellTypeRules,
    FATE_CHANNELS,
    FATE_MARKERS,
    FATE_STAINS,
    fate_rules,
    fate_thresholds,
    fate_weights,
    rule_clauses,
    training_settings,
):
    # The current thresholds and rules, with the stains and pre-processing
    # they hold for (None until every marker is loaded).
    cell_type_rules = (
        CellTypeRules(
            channels=dict(FATE_CHANNELS),
            thresholds={_m: float(fate_thresholds.value[_m]) for _m in FATE_MARKERS},
            rules=rule_clauses(fate_rules.value),
            hour=CELL_TYPE_HOUR,
            preprocessing=training_settings,
            stains=FATE_STAINS,
            clause_weights={
                _cell_type: [_weights[_key] for _key in sorted(_weights, key=int)]
                for _cell_type, _weights in fate_weights.value.items()
            },
        )
        if all(_m in FATE_CHANNELS for _m in FATE_MARKERS)
        else None
    )
    return (cell_type_rules,)


@app.cell(hide_code=True)
def _(
    FATE_CHANNELS,
    FATE_MARKERS,
    FATE_STAINS,
    high_percentiles,
    live_channel,
    low_percentiles,
    np,
    radial_rings,
    selection,
    share_estimator,
):
    # Nothing in this cell depends on the thresholds, so the threshold fit
    # can use it without rerunning whenever it moves the sliders.
    CELL_TYPE_HOUR = 48 if 48 in selection["timesteps"] else selection["timesteps"][-1]
    # Colours of the cell types in the views below.
    # CELL_TYPE_COLOURS = {
    #     "Notochord": (0.84, 0.15, 0.63),
    #     "Endoderm": (0.09, 0.75, 0.81),
    #     "Mesoderm": (0.17, 0.63, 0.17),
    #     "Several": (1.0, 0.6, 0.0),
    #     "Other": (0.5, 0.5, 0.5),
    # }

    CELL_TYPE_COLOURS = {
        "Notochord": (0.5, 0.8, 0.5),
        "Endoderm": (0.8, 0.8, 0.4),
        "Mesoderm": (1.0, 0.0, 0.0),
        "Several": (0.0, 0.0, 1.0),
        "Other": (0.5, 0.5, 0.5),
    }
    # Why the cell-type views cannot be drawn yet (None when they can).
    _missing = [_marker for _marker in FATE_MARKERS if _marker not in FATE_CHANNELS]
    _invalid = [
        _name
        for _channels in FATE_STAINS.values()
        for _name in _channels.values()
        if low_percentiles.value[_name] >= high_percentiles.value[_name]
    ]
    if _missing:
        cell_type_problem = "Load the cell-fate groups that contain " + ", ".join(_missing) + "."
    elif _invalid:
        cell_type_problem = "Low % must be below high % for: " + ", ".join(f"`{_n}`" for _n in _invalid)
    else:
        cell_type_problem = None

    def marker_images(trajectory):
        """``{stain: ({marker: image}, mask)}`` of one trajectory at the
        cell-type hour, or None if a stain was not imaged then."""
        stains = {}
        for stain, channels in FATE_STAINS.items():
            values = {}
            masks = []
            for marker, channel in channels.items():
                _, images = live_channel(channel, trajectory)
                if CELL_TYPE_HOUR not in images:
                    return None
                values[marker], mask = images[CELL_TYPE_HOUR]
                masks.append(mask)
            stains[stain] = (values, np.logical_and.reduce(masks))
        return stains

    def stain_samples(images, groups_of):
        """Pixels of each stain, pooled over trajectories, for ``share_estimator``.

        ``images`` lists ``marker_images`` results, one per trajectory, and
        ``groups_of(position, mask)`` gives the group of every pixel of the
        trajectory at that position in the list (-1 = left out).
        """
        samples = []
        for stain, channels in FATE_STAINS.items():
            markers = tuple(channels)
            values = [
                np.stack([stains[stain][0][m] for m in markers]).reshape(len(markers), -1)
                for stains in images
            ]
            groups = [
                np.asarray(groups_of(position, stains[stain][1])).ravel()
                for position, stains in enumerate(images)
            ]
            samples.append((markers, np.concatenate(values, axis=1), np.concatenate(groups)))
        return samples

    def ring_shares(images, n_rings, cell_type_rules):
        """Estimated label shares ``[trajectory, ring, label]`` and ring areas
        ``[trajectory, ring]`` (mean pixel count over the stains)."""

        def ring_of(position, mask):
            ring = radial_rings(mask, n_rings)
            return np.where(ring >= 0, position * n_rings + ring, -1)

        samples = stain_samples(images, ring_of)
        n_groups = len(images) * n_rings
        shares = share_estimator(samples, n_groups, cell_type_rules)(cell_type_rules.thresholds)
        area = np.mean(
            [np.bincount(groups[groups >= 0], minlength=n_groups) for _, _, groups in samples], axis=0
        )
        return shares.reshape(len(images), n_rings, -1), area.reshape(len(images), n_rings)

    return (
        CELL_TYPE_COLOURS,
        CELL_TYPE_HOUR,
        cell_type_problem,
        marker_images,
        ring_shares,
        stain_samples,
    )


@app.cell(hide_code=True)
def _(
    CELL_TYPE_COLOURS,
    CONDITION_COLOURS,
    FATE_MARKERS,
    FATE_STAINS,
    cell_type_problem,
    cell_type_rules,
    classify_within_stain,
    fate_show_channels,
    fate_trajectories,
    label_names,
    marker_images,
    mo,
    np,
    plt,
    radial_ring_count,
    ring_shares,
    show_figure,
    trajectories,
    trajectory_label,
):
    mo.stop(cell_type_problem is not None, mo.callout(cell_type_problem, kind="warn"))
    # Controls first, then knockouts, each in replicate order.
    _selected = [_t for _t in trajectories if _t in fate_trajectories.value]
    mo.stop(not _selected, mo.md("Select at least one trajectory."))
    _hour = cell_type_rules.hour
    _names = label_names(cell_type_rules)
    _colours = np.array([CELL_TYPE_COLOURS.get(_name, (1.0, 1.0, 1.0)) for _name in _names])

    _images = {}
    _not_measured = []
    for _trajectory in _selected:
        _result = marker_images(_trajectory)
        if _result is None:
            _not_measured.append(trajectory_label(_trajectory))
        else:
            _images[_trajectory] = _result
    mo.stop(
        not _images,
        mo.callout(f"None of the selected trajectories has every stain at {_hour}h.", kind="warn"),
    )

    # Cell types per pixel, each stain on its own (rows it can check): one
    # row per stain, one column per trajectory.
    _stain_names = list(FATE_STAINS)
    _figure, _axes = plt.subplots(
        len(_stain_names), len(_images), figsize=(2.8 * len(_images), 3.0 * len(_stain_names)), squeeze=False
    )
    for _column, (_trajectory, _stains) in enumerate(_images.items()):
        for _row, _stain in enumerate(_stain_names):
            _axis = _axes[_row, _column]
            _values, _inside = _stains[_stain]
            _labels = classify_within_stain(_values, cell_type_rules, _inside)
            _map = np.ones((*_inside.shape, 3))
            for _label, _cell_type in enumerate(_names):
                _map[_labels[_cell_type]] = _colours[_label]
            _axis.imshow(_map)
            _axis.set_xticks([])
            _axis.set_yticks([])
        _axes[0, _column].set_title(
            trajectory_label(_trajectory), fontsize=8, color=CONDITION_COLOURS[_trajectory[0]]
        )
    for _row, _stain in enumerate(_stain_names):
        _axes[_row, 0].set_ylabel(f"{_stain}\n({', '.join(FATE_STAINS[_stain])})", fontsize=8)
    _figure.suptitle(f"Cell types at {_hour}h, each stain on its own", fontsize=9)
    _figure.tight_layout()

    # Estimated shares per ring, combining both stains (see "Two stains").
    _n_rings = radial_ring_count.value
    _shares, _area = ring_shares(list(_images.values()), _n_rings, cell_type_rules)

    # Whole-colony prevalence: the ring estimates, weighted by area.
    _area = np.where(np.isnan(_shares[..., 0]), 0.0, _area)
    _colony = np.nansum(_shares * _area[..., None], axis=1) / np.maximum(_area.sum(axis=1), 1e-9)[:, None]
    _bars, _bar_axis = plt.subplots(figsize=(max(4.0, 1.1 * len(_images) + 2.5), 3.4))
    _positions = np.arange(len(_images))
    _bottom = np.zeros(len(_images))
    for _label, _cell_type in enumerate(_names):
        _heights = 100 * _colony[:, _label]
        _bar_axis.bar(_positions, _heights, bottom=_bottom, color=_colours[_label], label=_cell_type)
        _bottom += _heights
    _bar_axis.set_xticks(_positions)
    _bar_axis.set_xticklabels([trajectory_label(_t) for _t in _images], rotation=30, ha="right", fontsize=8)
    for _tick, _trajectory in zip(_bar_axis.get_xticklabels(), _images):
        _tick.set_color(CONDITION_COLOURS[_trajectory[0]])
    _bar_axis.set_ylabel("% of colony (estimated)", fontsize=8)
    _bar_axis.set_ylim(0, 100)
    _bar_axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    _bar_axis.set_title(f"Cell-type prevalence at {_hour}h", fontsize=9)
    _bars.tight_layout()

    _views = [show_figure(_figure), show_figure(_bars)]
    if fate_show_channels.value:
        # Each stain's channels after thresholding (only the high pixels shown).
        _panels = [(_stain, _marker) for _stain, _channels in FATE_STAINS.items() for _marker in _channels]
        _channels, _channel_axes = plt.subplots(
            len(_images), len(_panels), figsize=(2.3 * len(_panels), 2.5 * len(_images)), squeeze=False
        )
        for _row, (_trajectory, _stains) in enumerate(_images.items()):
            for _axis, (_stain, _marker) in zip(_channel_axes[_row], _panels):
                _values, _inside = _stains[_stain]
                _high = (_values[_marker] > cell_type_rules.thresholds[_marker]) & _inside
                _axis.imshow(np.where(_high, _values[_marker], 0.0), cmap="gray", vmin=0, vmax=1)
                _axis.set_title(
                    f"{_stain.rsplit('_', 1)[-1]} {_marker} high: "
                    f"{100 * _high.sum() / max(int(_inside.sum()), 1):.1f}%",
                    fontsize=8,
                )
                _axis.set_xticks([])
                _axis.set_yticks([])
            _channel_axes[_row, 0].set_ylabel(trajectory_label(_trajectory), fontsize=9)
        _channels.suptitle(
            f"{_hour}h. Thresholds: "
            + ", ".join(f"{_m} {cell_type_rules.thresholds[_m]:.2f}" for _m in FATE_MARKERS),
            fontsize=8,
        )
        _channels.tight_layout()
        _views.append(show_figure(_channels))
    mo.vstack(
        _views
        + (
            [mo.md(f"Skipped (a stain is missing at {_hour}h): " + ", ".join(_not_measured))]
            if _not_measured
            else []
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Radial cell-type prevalence
    """)
    return


@app.cell(hide_code=True)
def _(mo, trajectories):
    _conditions = list(dict.fromkeys(_condition for _condition, _ in trajectories))
    _replicates = sorted({_replicate for _, _replicate in trajectories})
    radial_conditions = mo.ui.multiselect(
        options=_conditions, value=_conditions, label="Conditions"
    )
    radial_replicates = mo.ui.multiselect(
        options={f"replicate {_r + 1}": _r for _r in _replicates},
        # value=[f"replicate {_r + 1}" for _r in _replicates],
        value = None,
        label="Replicates",
    )
    radial_ring_count = mo.ui.slider(4, 40, value=12, label="Rings", show_value=True)
    radial_spread = mo.ui.dropdown(
        options={"standard deviation": "sd", "standard error": "sem"},
        value="standard deviation",
        label="Error band",
    )
    mo.vstack(
        [
            mo.md(
                "The estimated share of each cell type in rings around the "
                "colony centre (see **Two stains** above), using the rules "
                "and thresholds above, so it follows every threshold and "
                "rule change. Rings split each colony's radius (its furthest "
                "mask pixel) evenly. Each panel is one condition, with one "
                "line per cell type: the mean over the selected replicates, "
                "and the band is their standard deviation or standard "
                "error. All labels add up to 100% at every radius. The ring "
                "count is also used for the bar chart above. The colonies "
                "must be centred for the rings to mean anything. Dashed "
                "lines mark the band edges set under **Fit thresholds to "
                "radial targets**."
            ),
            mo.hstack(
                [radial_conditions, radial_replicates, radial_ring_count, radial_spread],
                justify="start",
            ),
        ]
    )
    return (
        radial_conditions,
        radial_replicates,
        radial_ring_count,
        radial_spread,
    )


@app.cell(hide_code=True)
def _(
    CELL_TYPE_COLOURS,
    CONDITION_COLOURS,
    band_edges,
    cell_type_problem,
    cell_type_rules,
    label_names,
    marker_images,
    mo,
    np,
    plt,
    radial_conditions,
    radial_replicates,
    radial_ring_count,
    radial_spread,
    ring_shares,
    show_figure,
    trajectories,
    trajectory_label,
):
    mo.stop(cell_type_problem is not None, mo.callout(cell_type_problem, kind="warn"))
    _hour = cell_type_rules.hour
    _selected = [
        _t
        for _t in trajectories
        if _t[0] in radial_conditions.value and _t[1] in radial_replicates.value
    ]
    mo.stop(not _selected, mo.md("Select at least one condition and replicate."))

    _images = {}
    _not_measured = []
    for _trajectory in _selected:
        _result = marker_images(_trajectory)
        if _result is None:
            _not_measured.append(trajectory_label(_trajectory))
        else:
            _images[_trajectory] = _result
    mo.stop(
        not _images,
        mo.callout(f"None of the selected trajectories has every stain at {_hour}h.", kind="warn"),
    )

    _n_rings = radial_ring_count.value
    _centres = (np.arange(_n_rings) + 0.5) / _n_rings
    # [trajectory, ring, label] in percent.
    _shares = 100 * ring_shares(list(_images.values()), _n_rings, cell_type_rules)[0]
    _names = label_names(cell_type_rules)
    _used = list(_images)
    _conditions = [_c for _c in radial_conditions.value if any(_t[0] == _c for _t in _used)]

    def _mean_and_spread(stack):
        """Mean and error band over replicates (rows), ignoring empty rings."""
        count = np.sum(~np.isnan(stack), axis=0)
        mean = np.nansum(stack, axis=0) / np.where(count > 0, count, np.nan)
        squared = np.nansum((stack - mean) ** 2, axis=0)
        sd = np.sqrt(squared / np.where(count > 1, count - 1, np.nan))
        spread = sd / np.sqrt(count) if radial_spread.value == "sem" else sd
        return mean, np.nan_to_num(spread)

    _figure, _axes = plt.subplots(
        1, len(_conditions), figsize=(3.6 * len(_conditions), 3.0), sharey=True, squeeze=False
    )
    for _axis, _condition in zip(_axes[0], _conditions):
        _members = [_i for _i, _t in enumerate(_used) if _t[0] == _condition]
        for _label, _cell_type in enumerate(_names):
            _mean, _spread = _mean_and_spread(_shares[_members, :, _label])
            _colour = CELL_TYPE_COLOURS.get(_cell_type, (0.0, 0.0, 0.0))
            _axis.plot(_centres, _mean, color=_colour, label=_cell_type)
            _axis.fill_between(_centres, _mean - _spread, _mean + _spread, color=_colour, alpha=0.25, linewidth=0)
        _axis.set_title(
            f"{_condition} (n={len(_members)})",
            fontsize=9,
            color=CONDITION_COLOURS.get(_condition, "black"),
        )
        # Band edges of "Fit thresholds to radial targets" below.
        for _edge in band_edges.value:
            _axis.axvline(_edge, color="0.4", linestyle="--", linewidth=0.8)
        _axis.set_xlabel("distance from centre (fraction of radius)", fontsize=8)
        _axis.set_xlim(0, 1)
        _axis.set_ylim(0, 100)
        _axis.tick_params(labelsize=7)
    _axes[0, 0].set_ylabel("% of ring (estimated)", fontsize=8)
    _axes[0, -1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    _figure.suptitle(
        f"Radial cell-type prevalence at {_hour}h (band: "
        + ("standard error" if radial_spread.value == "sem" else "standard deviation")
        + " over replicates). Thresholds: "
        + ", ".join(f"{_m} {_t:.2f}" for _m, _t in cell_type_rules.thresholds.items()),
        fontsize=8,
    )
    _figure.tight_layout()

    mo.vstack(
        [show_figure(_figure)]
        + (
            [mo.md(f"Skipped (a stain is missing at {_hour}h): " + ", ".join(_not_measured))]
            if _not_measured
            else []
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    band_edges = mo.ui.range_slider(
        0.0,
        1.0,
        step=0.01,
        value=[0.35, 0.7],
        show_value=True,
        label="Band edges (fraction of colony radius)",
    )
    band_gap = mo.ui.slider(
        0.0, 0.3, step=0.01, value=0.0, show_value=True, label="Gap left out around each edge"
    )
    fit_restarts = mo.ui.slider(0, 20, value=5, show_value=True, label="Random restarts")
    run_fit = mo.ui.run_button(label="Fit thresholds")
    return band_edges, band_gap, fit_restarts, run_fit


@app.cell(hide_code=True)
def _(FATE_CELL_TYPES, FATE_CLAUSE_COUNTS, mo, trajectories):
    # Up to three colony-wide constraints, each off until a condition is set.
    _off = "—"
    _conditions = [_off, *dict.fromkeys(_condition for _condition, _ in trajectories)]
    _labels = [*FATE_CELL_TYPES, "Several", "Other"]
    fit_constraints = mo.ui.array(
        [
            mo.ui.dictionary(
                {
                    "condition": mo.ui.dropdown(options=_conditions, value=_off),
                    "cell type": mo.ui.dropdown(options=_labels, value=_labels[0]),
                    "direction": mo.ui.dropdown(options=["minimise", "maximise"], value="minimise"),
                    "weight": mo.ui.slider(0.0, 2.0, step=0.05, value=0.5, show_value=True),
                }
            )
            for _ in range(3)
        ]
    )
    _several_rows = [_c for _c, _rows in FATE_CLAUSE_COUNTS.items() if _rows > 1]
    fit_weight_types = mo.ui.multiselect(
        options=_several_rows, value=_several_rows, label="Also fit the row weights of"
    )
    return fit_constraints, fit_weight_types


@app.cell(hide_code=True)
def _(
    BANDS,
    FATE_CELL_TYPES,
    get_band_targets,
    mo,
    set_band_targets,
    trajectories,
):
    # One dropdown per condition and band. Choices are kept in a state, so
    # they survive this cell rerunning when the cell-type controls change.
    _no_target = "—"
    _choices = [_no_target, *FATE_CELL_TYPES, "Other"]
    _saved = get_band_targets()

    def _remember(condition, band):
        return lambda value: set_band_targets(
            lambda current: {**current, condition: {**current.get(condition, {}), band: value}}
        )

    band_targets = mo.ui.dictionary(
        {
            _condition: mo.ui.dictionary(
                {
                    _band: mo.ui.dropdown(
                        options=_choices,
                        value=_saved.get(_condition, {}).get(_band, _no_target)
                        if _saved.get(_condition, {}).get(_band, _no_target) in _choices
                        else _no_target,
                        on_change=_remember(_condition, _band),
                    )
                    for _band in BANDS
                }
            )
            for _condition in dict.fromkeys(_condition for _condition, _ in trajectories)
        }
    )
    return (band_targets,)


@app.cell(hide_code=True)
def _(BANDS, CELL_TYPE_COLOURS, Wedge):
    def draw_band_schematic(axis, band_types, edges, gap, title, title_colour="black"):
        """Colony drawn as a centre disc, a ring and a periphery, each coloured
        by its cell type in ``band_types`` (hatched where there is none)."""
        inner, outer = edges
        half = gap / 2.0
        spans = [(0.0, inner - half), (inner + half, outer - half), (outer + half, 1.0)]
        for (start, stop), band in zip(spans, BANDS):
            if stop <= max(start, 0.0):
                continue
            colour = CELL_TYPE_COLOURS.get(band_types.get(band))
            axis.add_patch(
                Wedge(
                    (0.0, 0.0),
                    stop,
                    0,
                    360,
                    width=stop - max(start, 0.0),
                    facecolor=colour if colour is not None else "white",
                    edgecolor="none" if colour is not None else "0.6",
                    hatch=None if colour is not None else "//",
                )
            )
        axis.add_patch(Wedge((0.0, 0.0), 1.0, 0, 360, fill=False, edgecolor="0.3"))
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title, fontsize=9, color=title_colour)

    return (draw_band_schematic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit thresholds to radial targets
    """)
    return


@app.cell(hide_code=True)
def _(
    BANDS,
    CELL_TYPE_COLOURS,
    CONDITION_COLOURS,
    band_edges,
    band_gap,
    band_targets,
    draw_band_schematic,
    fit_constraints,
    fit_restarts,
    fit_weight_types,
    mo,
    plt,
    run_fit,
    show_figure,
):
    _conditions = list(band_targets.value)
    _figure, _axes = plt.subplots(
        1, len(_conditions), figsize=(max(2.2 * len(_conditions), 5.0), 2.4), squeeze=False
    )
    for _axis, _condition in zip(_axes[0], _conditions):
        draw_band_schematic(
            _axis,
            band_targets.value[_condition],
            band_edges.value,
            band_gap.value,
            _condition,
            CONDITION_COLOURS.get(_condition, "black"),
        )
    _figure.suptitle("Target cell types", fontsize=9)
    _figure.tight_layout()

    _legend = " · ".join(
        f"<span style='color: rgb({int(255 * _c[0])}, {int(255 * _c[1])}, {int(255 * _c[2])})'>■</span> {_t}"
        for _t, _c in CELL_TYPE_COLOURS.items()
        if _t != "Several"
    )
    _target_table = mo.vstack(
        [mo.hstack([mo.md("**condition**")] + [mo.md(f"**{_b}**") for _b in BANDS], widths="equal")]
        + [
            mo.hstack(
                [mo.md(_condition)] + [band_targets[_condition][_b] for _b in BANDS],
                widths="equal",
                align="center",
            )
            for _condition in _conditions
        ]
    )
    mo.vstack(
        [
            mo.md(
                "Pick the cell type that should be the most common in the "
                "centre, ring and periphery of each condition (*—* = no "
                "target). Band edges are fractions of the colony radius, "
                "and the gap leaves out a strip around each edge, so pixels "
                "on a fuzzy boundary do not count.\n\n"
                "**Fit thresholds** searches the marker thresholds (in steps "
                "of 0.01, with the channels and rules above held fixed) for "
                "the ones that best meet the targets across all loaded "
                "replicates. For every band with a target and every "
                "replicate it takes the *margin*: the share of the target "
                "cell type minus the share of the most common other label "
                "(including *Several* and *Other*). A positive margin means "
                "the target wins. The fit maximises the mean margin, with "
                "each capped at 10 percentage points so that a band that is "
                "already won cannot make up for one that is lost (ties go to "
                "the larger uncapped margin).\n\n"
                "**Colony-wide constraints** ask for a cell type's share of "
                "the whole colony, in one condition, to be as small or as "
                "large as possible. The colony share is the area-weighted "
                "mean of its bands (gaps left out), averaged over "
                "replicates, and each constraint adds its weight times that "
                "share to the score (or subtracts it, to minimise). With "
                "weight 1, one percentage point of colony share counts as "
                "much as one point of mean band margin. Constraints work "
                "with or without band targets.\n\n"
                "**Row weights**: the fit can also choose the row weights "
                "(in steps of 0.05) of the chosen cell types, e.g. how much "
                "each mesoderm row counts. They are set like the thresholds "
                "afterwards.\n\n"
                "The search starts from the "
                "sliders' starting values (the last fit or loaded file, "
                "else 0.3; later tweaks are not used) and from a few random "
                "points. The "
                "fitted thresholds and row weights then become the starting "
                "values of their sliders under **Cell types at 48h**, so every "
                "cell-type view follows them and you can tweak them by hand."
            ),
            mo.hstack([band_edges, band_gap], justify="start"),
            _target_table,
            mo.md(_legend),
            show_figure(_figure),
            mo.md("**Colony-wide constraints** (optional; *—* = off)"),
            mo.vstack(
                [
                    mo.hstack(
                        [mo.md(f"**{_key}**") for _key in ("condition", "cell type", "direction", "weight")],
                        widths="equal",
                    )
                ]
                + [
                    mo.hstack(
                        [_row[_key] for _key in ("condition", "cell type", "direction", "weight")],
                        widths="equal",
                        align="center",
                    )
                    for _row in fit_constraints
                ]
            ),
            fit_weight_types,
            mo.hstack([fit_restarts, run_fit], justify="start"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    BANDS,
    CELL_TYPE_HOUR,
    CellTypeRules,
    FATE_CHANNELS,
    FATE_MARKERS,
    FATE_STAINS,
    band_edges,
    band_gap,
    band_margins,
    band_targets,
    cell_type_problem,
    cell_type_score,
    colony_share,
    coordinate_search,
    fate_rules,
    fit_constraints,
    fit_restarts,
    fit_weight_types,
    get_threshold_defaults,
    get_weight_defaults,
    label_names,
    marker_images,
    mo,
    np,
    radial_bands,
    rule_clauses,
    run_fit,
    set_fit_result,
    set_threshold_defaults,
    set_weight_defaults,
    share_estimator,
    stain_samples,
    target_score,
    threshold_grid,
    trajectories,
    trajectory_label,
):
    # This cell sets the threshold and row weight sliders, so it must not
    # depend on them (or on anything built from them, like cell_type_rules):
    # it would rerun straight away and fit again, forever. It reads the
    # marker images and rules directly, and starts from the sliders' stored
    # starting values (marimo does not rerun a cell for a state it set itself).
    mo.stop(not run_fit.value)
    mo.stop(cell_type_problem is not None, mo.callout(cell_type_problem, kind="warn"))
    _start = {_marker: 0.3 for _marker in FATE_MARKERS}
    _start.update({_m: _t for _m, _t in get_threshold_defaults().items() if _m in _start})
    _rules = CellTypeRules(
        channels=dict(FATE_CHANNELS),
        thresholds=_start,
        rules=rule_clauses(fate_rules.value),
        stains=FATE_STAINS,
    )
    _names = label_names(_rules)
    _targets = {
        _condition: {_b: _t for _b, _t in _bands.items() if _t in _names}
        for _condition, _bands in band_targets.value.items()
    }
    _active_constraints = [_row for _row in fit_constraints.value if _row["condition"] in _targets]
    _conditions_used = {_c for _c, _goal in _targets.items() if _goal} | {
        _row["condition"] for _row in _active_constraints
    }
    _wanted = [_t for _t in trajectories if _t[0] in _conditions_used]
    mo.stop(
        not _wanted,
        mo.callout("Choose a band target or a colony-wide constraint first.", kind="warn"),
    )

    _images, _used, _skipped = [], [], []
    for _trajectory in _wanted:
        _result = marker_images(_trajectory)
        if _result is None:
            _skipped.append(trajectory_label(_trajectory))
        else:
            _images.append(_result)
            _used.append(_trajectory)
    mo.stop(
        not _used,
        mo.callout(f"No chosen trajectory has every stain at {CELL_TYPE_HOUR}h.", kind="warn"),
    )

    # Group 3 * i + band holds band `band` of the i-th used trajectory, in
    # each stain; the shares of each group are estimated across the stains.
    def _band_of(position, mask):
        band = radial_bands(mask, band_edges.value, band_gap.value)
        return np.where(band >= 0, len(BANDS) * position + band, -1)

    _n_groups = len(BANDS) * len(_used)
    _samples = stain_samples(_images, _band_of)
    _shares = share_estimator(_samples, _n_groups, _rules)
    _area = np.mean(
        [np.bincount(_groups[_groups >= 0], minlength=_n_groups) for _, _, _groups in _samples], axis=0
    )
    _group_targets = []
    for _trajectory in _used:
        _goal = _targets.get(_trajectory[0], {})
        _group_targets += [_names.index(_goal[_b]) if _b in _goal else -1 for _b in BANDS]

    def _colonies(condition):
        """The band groups of each used colony of a condition."""
        return [
            [len(BANDS) * _i + _b for _b in range(len(BANDS))]
            for _i, _t in enumerate(_used)
            if _t[0] == condition
        ]

    # (colonies, label, signed weight) for cell_type_score.
    _constraints = [
        (
            _colonies(_row["condition"]),
            _names.index(_row["cell type"]),
            _row["weight"] if _row["direction"] == "maximise" else -_row["weight"],
        )
        for _row in _active_constraints
    ]

    # Parameters: one threshold per marker, then one weight per row of the
    # cell types whose row weights are fitted. The other weights stay at
    # their slider starting values.
    _saved_weights = get_weight_defaults()
    _start_weights = {
        _c: [float(_w) for _w in _saved_weights[_c]]
        if len(_saved_weights.get(_c, ())) == len(_clauses)
        else [1.0] * len(_clauses)
        for _c, _clauses in _rules.rules.items()
    }
    _free = [(_c, _row) for _c in fit_weight_types.value for _row in range(len(_rules.rules[_c]))]
    _n_markers = len(FATE_MARKERS)

    def _weights(parameters):
        weights = {_c: list(_w) for _c, _w in _start_weights.items()}
        for (cell_type, row), value in zip(_free, parameters[_n_markers:]):
            weights[cell_type][row] = float(value)
        return weights

    def _shares_at(parameters):
        return _shares(parameters[:_n_markers], _weights(parameters))

    def _objective(parameters):
        return cell_type_score(
            _shares_at(parameters), _group_targets, area=_area, constraints=_constraints
        )

    _grids = [threshold_grid(0.01)] * _n_markers + [threshold_grid(0.05)] * len(_free)
    _current = np.array(
        [_start[_m] for _m in FATE_MARKERS] + [_start_weights[_c][_row] for _c, _row in _free]
    )
    _starts = [_current] + list(
        np.random.default_rng(0).uniform(0.0, 1.0, (fit_restarts.value, len(_current)))
    )
    _best, _best_key = _current, None
    for _point in mo.status.progress_bar(_starts, title="Fitting thresholds", remove_on_exit=True):
        _parameters, _key = coordinate_search(_objective, _point, _grids)
        if _best_key is None or _key > _best_key:
            _best, _best_key = _parameters, _key

    def _evaluate(parameters):
        """Shares [trajectory, band, label], margins [trajectory, band], all shares [group, label]."""
        shares = _shares_at(parameters)
        margins = band_margins(shares, _group_targets)
        return (
            shares.reshape(len(_used), len(BANDS), -1),
            margins.reshape(len(_used), len(BANDS)),
            shares,
        )

    _now_shares, _now_margins, _now_flat = _evaluate(_current)
    _fit_shares, _fit_margins, _fit_flat = _evaluate(_best)
    _conditions = list(dict.fromkeys(_t[0] for _t in _used))

    def _most_common(shares, rows, band):
        mean = np.nanmean(shares[rows, band], axis=0) if not np.isnan(shares[rows, band]).all() else None
        return None if mean is None else _names[int(np.nanargmax(mean))]

    _rows = []
    for _condition in _conditions:
        _members = [_i for _i, _t in enumerate(_used) if _t[0] == _condition]
        for _b, _band_name in enumerate(BANDS):
            if _band_name not in _targets.get(_condition, {}):
                continue
            _target = _names.index(_targets[_condition][_band_name])
            _rows.append(
                {
                    "condition": _condition,
                    "band": _band_name,
                    "target": _names[_target],
                    "target % (start)": round(100 * float(np.nanmean(_now_shares[_members, _b, _target])), 1),
                    "target % (fitted)": round(100 * float(np.nanmean(_fit_shares[_members, _b, _target])), 1),
                    "replicates won (start)": f"{int(np.sum(_now_margins[_members, _b] > 0))}/{len(_members)}",
                    "replicates won (fitted)": f"{int(np.sum(_fit_margins[_members, _b] > 0))}/{len(_members)}",
                    "most common (fitted)": _most_common(_fit_shares, _members, _b) or "no pixels",
                }
            )
    _constraint_rows = [
        {
            "condition": _row["condition"],
            "cell type": _row["cell type"],
            "direction": _row["direction"],
            "weight": _row["weight"],
            "colony % (start)": round(100 * colony_share(_now_flat, _area, _colonies_, _label), 1),
            "colony % (fitted)": round(100 * colony_share(_fit_flat, _area, _colonies_, _label), 1),
        }
        for _row, (_colonies_, _label, _) in zip(_active_constraints, _constraints)
    ]
    _threshold_table = [
        {"marker": _m, "start": round(float(_c), 2), "fitted": round(float(_f), 2)}
        for _m, _c, _f in zip(FATE_MARKERS, _current, _best)
    ]
    _weight_table = [
        {"row": f"{_c} row {_row + 1}", "start": round(float(_s), 2), "fitted": round(float(_f), 2)}
        for (_c, _row), _s, _f in zip(_free, _current[_n_markers:], _best[_n_markers:])
    ]
    _schematics = []
    for _condition in _conditions:
        _members = [_i for _i, _t in enumerate(_used) if _t[0] == _condition]
        _schematics.append(
            (
                _condition,
                len(_members),
                {_band_name: _most_common(_fit_shares, _members, _b) for _b, _band_name in enumerate(BANDS)},
            )
        )

    # The fitted thresholds and row weights become the slider defaults under
    # "Cell types at 48h" (which redraws every cell-type view), and the
    # result is kept for the cell below, since this cell stops once the
    # sliders change.
    set_threshold_defaults({_m: round(float(_t), 2) for _m, _t in zip(FATE_MARKERS, _best)})
    set_weight_defaults(_weights(_best))
    set_fit_result(
        {
            "thresholds": _threshold_table,
            "weights": _weight_table,
            "bands": _rows,
            "constraints": _constraint_rows,
            "schematics": _schematics,
            "edges": tuple(band_edges.value),
            "gap": band_gap.value,
            "won": (int(np.sum(_now_margins > 0)), int(np.sum(_fit_margins > 0))),
            "targeted": int(np.sum(~np.isnan(_fit_margins))),
            "band score": (target_score(_now_margins), target_score(_fit_margins)),
            "score": (_objective(_current)[0], _best_key[0]),
            "skipped": _skipped,
        }
    )
    return


@app.cell(hide_code=True)
def _(
    CONDITION_COLOURS,
    draw_band_schematic,
    get_fit_result,
    mo,
    plt,
    show_figure,
):
    _fit = get_fit_result()
    mo.stop(_fit is None)
    _figure, _axes = plt.subplots(
        1, len(_fit["schematics"]), figsize=(max(2.2 * len(_fit["schematics"]), 5.0), 2.4), squeeze=False
    )
    for _axis, (_condition, _n, _band_types) in zip(_axes[0], _fit["schematics"]):
        draw_band_schematic(
            _axis,
            _band_types,
            _fit["edges"],
            _fit["gap"],
            f"{_condition} (n={_n})",
            CONDITION_COLOURS.get(_condition, "black"),
        )
    _figure.suptitle("Most common cell type, fitted thresholds\n(mean over replicates)", fontsize=9)
    _figure.tight_layout()
    _total = _fit["targeted"]
    _summary = (
        "**Last fit.** The threshold and row weight sliders under **Cell "
        "types at 48h** now start at the fitted values; tweak them from "
        f"there. Score: {100 * _fit['score'][0]:.1f} at the start, "
        f"**{100 * _fit['score'][1]:.1f}** fitted."
    )
    if _total:
        _summary += (
            f" Bands won (over all replicates): {_fit['won'][0]}/{_total} at the "
            f"start, **{_fit['won'][1]}/{_total}** fitted (band part of the score: "
            f"{100 * _fit['band score'][0]:.1f} → {100 * _fit['band score'][1]:.1f}, best 10.0)."
        )
    _summary += f" Band edges {_fit['edges'][0]:.2f}–{_fit['edges'][1]:.2f}, gap {_fit['gap']:.2f}."
    mo.vstack(
        [mo.md(_summary), mo.ui.table(_fit["thresholds"], selection=None)]
        + ([mo.ui.table(_fit["weights"], selection=None)] if _fit["weights"] else [])
        + ([mo.ui.table(_fit["bands"], selection=None)] if _fit["bands"] else [])
        + ([mo.ui.table(_fit["constraints"], selection=None)] if _fit["constraints"] else [])
        + [show_figure(_figure)]
        + (
            [mo.md("Skipped (a stain is missing): " + ", ".join(_fit["skipped"]))]
            if _fit["skipped"]
            else []
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Export and load cell types
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cell_types_path = mo.ui.text(
        value="Experiments/micropatterns/conf/cell_types/260726_cell_types.yaml",
        label="Cell type file (relative to the repository root)",
        full_width=True,
    )
    export_cell_types = mo.ui.run_button(label="Export cell types")
    load_cell_types = mo.ui.run_button(label="Load cell types")
    mo.vstack(
        [
            mo.md(
                "**Export cell types** writes the thresholds, cell-type rules "
                "(with their *or* rows) and stains above to a file, together "
                "with the **Training settings** they were tuned with: the "
                "thresholds only hold for images processed the same way. "
                "Read it elsewhere with "
                "`Common.dataloader.cell_types.load_cell_type_rules`, and "
                "estimate cell-type shares with "
                "`Common.dataloader.cell_type_shares.share_estimator` (for "
                "model output too, so it is scored like the data). **Load "
                "cell types** puts a file's thresholds and rules back into "
                "the controls above."
            ),
            cell_types_path,
            mo.hstack([export_cell_types, load_cell_types], justify="start"),
        ]
    )
    return cell_types_path, export_cell_types, load_cell_types


@app.cell(hide_code=True)
def _(
    cell_type_rules,
    cell_types_path,
    export_cell_types,
    mo,
    resolve_alignment_path,
    save_cell_type_rules,
    training_mismatches,
):
    mo.stop(not export_cell_types.value)
    mo.stop(
        cell_type_rules is None,
        mo.callout("Load the cell-fate groups that contain every marker first.", kind="warn"),
    )
    _path = resolve_alignment_path(cell_types_path.value)
    save_cell_type_rules(_path, cell_type_rules)
    mo.md(
        f"Exported **{len(cell_type_rules.rules)}** cell types to `{_path}`."
        + (
            "\n\n**The thresholds were tuned on images training will not "
            "reproduce, because " + "; ".join(training_mismatches) + ".**"
            if training_mismatches
            else ""
        )
    )
    return


@app.cell(hide_code=True)
def _(
    cell_types_path,
    load_cell_type_rules,
    load_cell_types,
    mo,
    resolve_alignment_path,
    set_loaded_cell_types,
    set_threshold_defaults,
    set_weight_defaults,
    training_settings,
):
    mo.stop(not load_cell_types.value)
    _rules = load_cell_type_rules(resolve_alignment_path(cell_types_path.value))
    set_loaded_cell_types(_rules)
    set_threshold_defaults(dict(_rules.thresholds))
    set_weight_defaults({_c: list(_w) for _c, _w in _rules.clause_weights.items()})
    _differs = _rules.preprocessing != training_settings
    mo.md(
        f"Loaded **{len(_rules.rules)}** cell types (tuned at {_rules.hour}h)."
        + (
            "\n\n**The file was made with different pre-processing from the "
            "current Training settings, so its thresholds may not fit the "
            "images shown.**"
            if _differs
            else ""
        )
    )
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Whole selection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    overview_button = mo.ui.run_button(label="Clean all loaded images")
    outlier_ratio = mo.ui.slider(
        1.1, 4.0, step=0.1, value=1.5, label="Flag images off by more than", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                "Replaces hot pixels and removes the background in every "
                "loaded image with the current per-channel controls (about "
                "0.1–0.2 s per channel for the background with shrink 4, plus "
                "0.4 s per channel with hot pixel replacement on). Changing any "
                "control in section 2 clears the results until you click "
                "again. Mask and "
                "normalisation controls update without re-running."
            ),
            mo.hstack([overview_button, outlier_ratio], justify="start"),
        ]
    )
    return outlier_ratio, overview_button


@app.cell(hide_code=True)
def _(
    GROUP_CHANNELS,
    block_mean,
    clean_image,
    cleaning_settings,
    colony_statistics,
    mo,
    overview_button,
    raw_images,
):
    mo.stop(
        not overview_button.value,
        mo.md("Click **Clean all loaded images** to process the selection."),
    )
    _statistics = []
    _images = []
    for _index, _entry in enumerate(
        mo.status.progress_bar(raw_images, title="Cleaning images", remove_on_exit=True)
    ):
        _record = _entry["record"]
        _steps = clean_image(_entry, cleaning_settings)
        _scaled, _cleaned = _steps["scaled"], _steps["cleaned"]
        _hot_counts = _steps["hot"].sum(axis=(0, 1))
        _raw_inside, _raw_outside = colony_statistics(_scaled, _entry["foreground"])
        _clean_inside, _clean_outside = colony_statistics(_cleaned, _entry["foreground"])
        for _channel, _name in enumerate(GROUP_CHANNELS[_record.group]):
            _statistics.append(
                {
                    "index": _index,
                    "channel": _name,
                    "condition": _record.condition,
                    "timestep": _record.timestep,
                    "replicate": _record.replicate + 1,
                    "raw_colony_p99": float(_raw_inside[_channel]),
                    "raw_outside_median": float(_raw_outside[_channel]),
                    "cleaned_colony_p99": float(_clean_inside[_channel]),
                    "cleaned_outside_median": float(_clean_outside[_channel]),
                    "hot_pixels": int(_hot_counts[_channel]),
                }
            )
        # Everything after background removal works at the training resolution.
        _images.append(
            {
                "record": _record,
                "raw": block_mean(_scaled),
                "cleaned": block_mean(_cleaned),
            }
        )
    overview = {"statistics": _statistics, "images": _images}
    return (overview,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-image intensities
    """)
    return


@app.cell(hide_code=True)
def _(
    CONDITION_COLOURS,
    MICROPATTERN_260726_SCHEMA,
    mo,
    np,
    outlier_ratio,
    overview,
    plt,
):
    _statistics = overview["statistics"]
    _names = [
        _name
        for _name in MICROPATTERN_260726_SCHEMA.measurement_names
        if any(_row["channel"] == _name for _row in _statistics)
    ]
    _offsets = {"ctrl": -1.0, "sl0": 0.0, "sl24": 1.0}

    def _metric_figure(raw_key, cleaned_key, ylabel):
        columns = min(4, len(_names))
        rows = int(np.ceil(len(_names) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(3.4 * columns, 2.6 * rows), squeeze=False
        )
        for axis, name in zip(axes.flat, _names):
            for row in _statistics:
                if row["channel"] != name:
                    continue
                x = row["timestep"] + _offsets[row["condition"]]
                colour = CONDITION_COLOURS[row["condition"]]
                axis.scatter(x, row[raw_key], facecolors="none", edgecolors=colour, s=18)
                axis.scatter(x, row[cleaned_key], color=colour, s=12)
            axis.set_title(name, fontsize=8)
            axis.set_xlabel("hours", fontsize=7)
            axis.set_ylabel(ylabel, fontsize=7)
            axis.tick_params(labelsize=7)
        for axis in axes.flat[len(_names):]:
            axis.axis("off")
        figure.tight_layout()
        return figure

    # Compare each image with the median of its replicates (same channel,
    # condition and timestep). Large ratios point at imaging artifacts.
    _groups = {}
    for _row in _statistics:
        _key = (_row["channel"], _row["condition"], _row["timestep"])
        _groups.setdefault(_key, []).append(_row["cleaned_colony_p99"])
    _flagged = []
    for _row in _statistics:
        _median = np.median(
            _groups[(_row["channel"], _row["condition"], _row["timestep"])]
        )
        _ratio = _row["cleaned_colony_p99"] / _median if _median > 0 else np.nan
        if np.isfinite(_ratio) and max(_ratio, 1 / _ratio) > outlier_ratio.value:
            _flagged.append(
                {
                    "channel": _row["channel"],
                    "condition": _row["condition"],
                    "timestep": _row["timestep"],
                    "replicate": _row["replicate"],
                    "colony p99 / replicate median": round(float(_ratio), 2),
                }
            )

    mo.vstack(
        [
            mo.md(
                "Each point is one image: hollow = raw (after the intensity factor), "
                "filled = after hot pixel replacement and background removal. Colours: control blue, "
                "knockout at 0h orange, knockout at 24h green. Replicates of "
                "the same timestep should sit close together."
            ),
            mo.ui.tabs(
                {
                    "Colony brightness (p99)": _metric_figure(
                        "raw_colony_p99", "cleaned_colony_p99", "colony p99"
                    ),
                    "Background (off-colony median)": _metric_figure(
                        "raw_outside_median", "cleaned_outside_median", "off-colony median"
                    ),
                    "Hot pixels replaced": (
                        mo.ui.table(
                            [
                                {
                                    _key: _row[_key]
                                    for _key in ("channel", "condition", "timestep", "replicate", "hot_pixels")
                                }
                                for _row in _statistics
                                if _row["hot_pixels"]
                            ],
                            selection=None,
                        )
                        if any(_row["hot_pixels"] for _row in _statistics)
                        else mo.md("No hot pixels replaced (all thresholds are 0, or nothing was found).")
                    ),
                    f"Flagged images ({len(_flagged)})": (
                        mo.ui.table(_flagged, selection=None)
                        if _flagged
                        else mo.md("No image is off by more than the chosen ratio.")
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    GROUP_CHANNELS,
    MICROPATTERN_260726_SCHEMA,
    raw_images,
    selection,
    source_condition,
):
    # A trajectory is one condition and replicate over all timesteps. As in
    # the loader, timepoints before a knockout use the control images.
    _lookup = {
        (
            _entry["record"].condition,
            _entry["record"].group,
            _entry["record"].timestep,
            _entry["record"].replicate,
        ): _index
        for _index, _entry in enumerate(raw_images)
    }
    def _members(condition, replicate):
        members = {}
        for hour in selection["timesteps"]:
            for group in GROUP_CHANNELS:
                key = (source_condition(condition, hour), group, hour, replicate)
                if key in _lookup:
                    members[(group, hour)] = _lookup[key]
        return members

    trajectories = {
        (_condition, _replicate): _members(_condition, _replicate)
        for _condition in selection["conditions"]
        for _replicate in selection["replicates"]
        if _members(_condition, _replicate)
    }
    # Trajectories of every loaded condition for every replicate, also when
    # control is not a selected condition (its images are always loaded with
    # knockouts). Used for the control and joint knockout bounds.
    condition_trajectories = {
        (_condition, _replicate): _members(_condition, _replicate)
        for _condition in ("ctrl", "sl0", "sl24")
        if _condition == "ctrl" or _condition in selection["conditions"]
        for _replicate in selection["replicates"]
        if _members(_condition, _replicate)
    }
    _loaded_groups = {_entry["record"].group for _entry in raw_images}
    loaded_names = [
        _name
        for _group in MICROPATTERN_260726_SCHEMA.experiment_groups
        if _group.name in _loaded_groups
        for _name in GROUP_CHANNELS[_group.name]
        if _name not in selection["excluded"]
    ]
    return condition_trajectories, loaded_names, trajectories


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Intensity rescaling (normalisation)
    """)
    return


@app.cell(hide_code=True)
def _(KNOCKOUT_BOUNDS, NORMALISATION_MODES, mo):
    normalise_mode = mo.ui.dropdown(
        options={
            _description[0].upper() + _description[1:]: _mode
            for _mode, _description in NORMALISATION_MODES.items()
        },
        value="Per channel, bounds averaged over replicates",
        label="Normalisation",
    )
    inside_only = mo.ui.checkbox(value=True, label="Percentiles from pixels inside the mask")
    image_source = mo.ui.radio(
        options={"Background removed": "cleaned", "Raw (intensity factor only)": "raw"},
        value="Background removed",
        label="Images",
        inline=True,
    )
    zero_outside = mo.ui.checkbox(value=True, label="Zero outside the mask")
    knockout_bounds = mo.ui.dropdown(
        options={
            f"{_mode} ({_description})": _mode for _mode, _description in KNOCKOUT_BOUNDS.items()
        },
        value=f"selected ({KNOCKOUT_BOUNDS['selected']})",
        label="Knockout bounds",
    )
    mo.vstack(
        [
            mo.md(
                "Each channel is rescaled linearly so its low percentile maps "
                "to 0 and its high percentile to 1, with clipping. The "
                "percentiles are taken over all timesteps of a trajectory "
                "together, so changes over time are kept.\n\n"
                "- *Bounds averaged over replicates*: percentiles are found "
                "for each trajectory, then averaged, and every trajectory uses "
                "the average.\n"
                "- *Per channel and replicate*: each trajectory keeps its own "
                "percentiles, which evens out brightness differences between "
                "replicates (real ones included).\n"
                "- *Pooled*: percentiles of all pixels together, as the "
                "training loader does now.\n\n"
                "**Knockout bounds**:\n\n"
                "- *selected*: knockouts share bounds as set above.\n"
                "- *control*: knockouts use the control bounds (per "
                "replicate: the control replicate with the same number), so a "
                "change in overall intensity is kept.\n"
                "- *joint*: each condition is rescaled on its own, then all "
                "use the widest range (per replicate: within each replicate). "
                "Control bounds change only for channels where a knockout goes "
                "outside them, listed in the table below. Needs sl0 and sl24 "
                "loaded.\n\n"
                "Control images are used even if control is not a selected "
                "condition. These settings also apply to the live views in "
                "the middle column."
            ),
            normalise_mode,
            knockout_bounds,
            mo.hstack([image_source, inside_only, zero_outside], justify="start"),
        ]
    )
    return (
        image_source,
        inside_only,
        knockout_bounds,
        normalise_mode,
        zero_outside,
    )


@app.cell(hide_code=True)
def _(condition_trajectories, selection, trajectories):
    # Knockout conditions that joint bounds need but were not loaded.
    joint_missing = sorted({"sl0", "sl24"} - set(selection["conditions"]))

    def bound_sources(knockout_mode):
        """Trajectories whose pixels set the bounds: the selected ones, plus
        the control (control bounds) or every condition (joint bounds) of
        each selected replicate."""
        if knockout_mode == "joint" and joint_missing:
            raise ValueError("Joint knockout bounds need sl0 and sl24 loaded")
        return {
            **{
                trajectory: members
                for trajectory, members in condition_trajectories.items()
                if knockout_mode == "joint"
                or (knockout_mode == "control" and trajectory[0] == "ctrl")
            },
            **trajectories,
        }

    return bound_sources, joint_missing


@app.cell(hide_code=True)
def _(bound_sources, joint_missing, knockout_bounds, mo):
    mo.stop(
        knockout_bounds.value == "joint" and bool(joint_missing),
        mo.md("Joint knockout bounds need all conditions loaded; also load " + ", ".join(joint_missing) + "."),
    )
    rescaling_trajectories = bound_sources(knockout_bounds.value)
    return (rescaling_trajectories,)


@app.cell(hide_code=True)
def _(block_mean, raw_images):
    # Colony pixels at the training resolution, used to fit the masks.
    small_foregrounds = [
        block_mean(_entry["foreground"].astype("float32")) > 0.5 for _entry in raw_images
    ]
    return (small_foregrounds,)


@app.cell(hide_code=True)
def _(DISPLAY_DOWNSAMPLE, coverage_blur, coverage_map, raw_images):
    # Colony coverage at the training resolution: the fraction of colony
    # pixels in each block, scaled so the densest part of each image is 1
    # (Common/dataloader/alignment.py, as used for the exported file).
    coverage_maps = [
        coverage_map(_entry["foreground"], DISPLAY_DOWNSAMPLE, coverage_blur.value)
        for _entry in raw_images
    ]
    return (coverage_maps,)


@app.cell(hide_code=True)
def _(Path, selection):
    from Common.dataloader.micropattern_projections import image_key as _image_key

    def image_key(record):
        """Image path relative to the dataset root, without the projection:
        the key in the alignment and quality flags files."""
        return _image_key(Path(record.path).relative_to(selection["root"]).as_posix())

    return (image_key,)


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    circular_colony_mask,
    get_loaded_alignment,
    np,
    pattern_radius_override,
    radius_quantile,
    raw_images,
    selection,
    small_foregrounds,
    use_loaded_alignment,
):
    # One pattern radius for all images, from the latest timesteps, where the
    # colony fills the pattern. Each image's radius comes from the area of its
    # fitted circle (the "Fitted to each image" circle).
    _late = [_hour for _hour in selection["timesteps"] if _hour >= 36] or [
        max(selection["timesteps"])
    ]
    _radii = [
        np.sqrt(
            circular_colony_mask(_foreground, _entry["record"].group, radius_quantile.value, 1.0).sum()
            / np.pi
        )
        * DISPLAY_DOWNSAMPLE
        for _entry, _foreground in zip(raw_images, small_foregrounds)
        if _entry["record"].timestep in _late
    ]
    estimated_radius = float(np.median(_radii))
    _loaded = get_loaded_alignment()
    if _loaded is not None and use_loaded_alignment.value:
        pattern_radius = _loaded["alignment"].pattern_radius
        pattern_radius_source = f"from {_loaded['path']}"
    elif pattern_radius_override.value:
        pattern_radius = float(pattern_radius_override.value)
        pattern_radius_source = "set by hand"
    else:
        pattern_radius = estimated_radius
        pattern_radius_source = "estimated"
    return estimated_radius, pattern_radius, pattern_radius_source


@app.cell(hide_code=True)
def _(mo):
    # Manual centre corrections (down, right) in full-resolution pixels, keyed
    # by image path relative to the dataset root, without the projection (see image_key).
    get_offsets, set_offsets = mo.state({})
    # The alignment file loaded under "Export and load alignment".
    get_loaded_alignment, set_loaded_alignment = mo.state(None)
    # Images flagged as low quality: {image_key: reason}, shared by all projections.
    get_flags, set_flags = mo.state({})
    # The cell type file loaded under "Cell types at 48h" (a CellTypeRules).
    get_loaded_cell_types, set_loaded_cell_types = mo.state(None)
    # Default threshold of each marker for the sliders under "Cell types at
    # 48h", {marker: threshold}, set by a loaded cell type file or a fit.
    get_threshold_defaults, set_threshold_defaults = mo.state({})
    # Default row (clause) weights for the cell types with several rules
    # rows, {cell type: [weight, ...]}, set by a loaded file or a fit.
    get_weight_defaults, set_weight_defaults = mo.state({})
    # Result of the last "Fit thresholds" run (a dict, see that cell), or None.
    get_fit_result, set_fit_result = mo.state(None)
    # Target cell type of each radial band, {condition: {band: cell type}}.
    # Kept here so the targets survive the cell-type controls being rebuilt.
    get_band_targets, set_band_targets = mo.state(
        {"ctrl": {"centre": "Mesoderm", "ring": "Endoderm", "periphery": "Endoderm"}}
    )
    return (
        get_band_targets,
        get_fit_result,
        get_flags,
        get_loaded_alignment,
        get_loaded_cell_types,
        get_offsets,
        get_threshold_defaults,
        get_weight_defaults,
        set_band_targets,
        set_fit_result,
        set_flags,
        set_loaded_alignment,
        set_loaded_cell_types,
        set_offsets,
        set_threshold_defaults,
        set_weight_defaults,
    )


@app.cell(hide_code=True)
def _(get_flags, image_key, raw_images):
    # Indices of the loaded images flagged as low quality. Like training,
    # the notebook leaves them out of the normalisation bounds.
    flagged_indices = frozenset(
        _index
        for _index, _entry in enumerate(raw_images)
        if image_key(_entry["record"]) in get_flags()
    )
    return (flagged_indices,)


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    coverage_maps,
    edge_width,
    fit_colony_centre,
    fit_score,
    get_loaded_alignment,
    get_offsets,
    grid_position,
    image_key,
    pattern_radius,
    raw_images,
    record_label,
    use_loaded_alignment,
):
    # The centre of each colony on the training-resolution grid: the
    # automatic fit (or, with a loaded alignment file, its automatic centre)
    # plus the manual correction.
    _radius = pattern_radius / DISPLAY_DOWNSAMPLE
    _offsets = get_offsets()
    _loaded = get_loaded_alignment()
    _file = _loaded["alignment"] if _loaded is not None and use_loaded_alignment.value else None
    colony_fits = []
    for _entry, _coverage in zip(raw_images, coverage_maps):
        _key = image_key(_entry["record"])
        if _file is not None and _key in _file.centres:
            _automatic = _file.details.get(_key, {}).get("automatic", _file.centres[_key])
            _fitted = tuple(float(_v) for _v in grid_position(_automatic, DISPLAY_DOWNSAMPLE))
            _source = "file"
        else:
            _fitted, _ = fit_colony_centre(_coverage, _radius, edge_width.value)
            _source = "fit"
        _offset = _offsets.get(_key, (0.0, 0.0))
        _centre = (
            _fitted[0] + _offset[0] / DISPLAY_DOWNSAMPLE,
            _fitted[1] + _offset[1] / DISPLAY_DOWNSAMPLE,
        )
        colony_fits.append(
            {
                "key": _key,
                "label": record_label(_entry["record"]),
                "centre": _centre,
                "fitted": _fitted,
                "offset": _offset,
                "score": fit_score(_coverage, _centre, _radius, edge_width.value),
                "source": _source,
            }
        )
    return (colony_fits,)


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    align,
    centred_disk,
    circular_colony_mask,
    colony_fits,
    manual_radius,
    mask_mode,
    ndi,
    np,
    pattern_radius,
    radius_quantile,
    radius_scale,
    raw_images,
    small_foregrounds,
    trajectories,
):
    # For every loaded image: the shift that centres it (or None without
    # centring) and the mask chosen in section 3, both at the training
    # resolution. Shared by the live views and the whole selection.
    _shape = small_foregrounds[0].shape
    _target = ((_shape[0] - 1) / 2.0, (_shape[1] - 1) / 2.0)
    image_geometry = []
    if mask_mode.value == "fixed":
        # Centres from the template fit; every mask has the pattern radius.
        _radius = pattern_radius * radius_scale.value / DISPLAY_DOWNSAMPLE
        _centred = centred_disk(_shape, _radius)
        _rows, _columns = np.ogrid[: _shape[0], : _shape[1]]
        for _fit in colony_fits:
            _centre = _fit["centre"]
            if align.value:
                _shift = (_target[0] - _centre[0], _target[1] - _centre[1])
                image_geometry.append((_shift, _centred))
            else:
                _disk = (_rows - _centre[0]) ** 2 + (_columns - _centre[1]) ** 2 <= _radius**2
                image_geometry.append((None, _disk))
    else:
        # Centres and radii from a circle fitted to each image's colony pixels.
        for _entry, _foreground in zip(raw_images, small_foregrounds):
            _circle = circular_colony_mask(
                _foreground, _entry["record"].group, radius_quantile.value, radius_scale.value
            )
            _shift = None
            if align.value:
                _centre = ndi.center_of_mass(_circle)
                _shift = (_target[0] - _centre[0], _target[1] - _centre[1])
                _circle = (
                    ndi.shift(_circle.astype(np.float32), _shift, order=0, prefilter=False)
                    > 0.5
                )
            image_geometry.append((_shift, _circle))

    if mask_mode.value == "common":
        _groups = {_group for _members in trajectories.values() for _group, _ in _members}
        _primary = "cell_fate_s1" if "cell_fate_s1" in _groups else sorted(_groups)[0]
        _candidates = []
        for _members in trajectories.values():
            _hours = sorted(_hour for _group, _hour in _members if _group == _primary)
            if _hours:
                _candidates.append(image_geometry[_members[(_primary, _hours[-1])]][1])
        _common = np.mean(_candidates, axis=0) >= 0.5
        image_geometry = [(_shift, _common) for _shift, _ in image_geometry]
    elif mask_mode.value == "manual":
        _disk = centred_disk(_shape, manual_radius.value / DISPLAY_DOWNSAMPLE)
        image_geometry = [(_shift, _disk) for _shift, _ in image_geometry]
    return (image_geometry,)


@app.cell(hide_code=True)
def _(image_geometry, image_source, overview, place_image):
    prepared = [
        {
            "record": _image["record"],
            "values": place_image(_image[image_source.value], _shift),
            "mask": _mask,
        }
        for _image, (_shift, _mask) in zip(overview["images"], image_geometry)
    ]
    return (prepared,)


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    flagged_indices,
    high_percentiles,
    condition_bins,
    inside_only,
    knockout_bounds,
    loaded_names,
    low_percentiles,
    mo,
    normalise_mode,
    np,
    prepared,
    rescaling_trajectories,
    trajectory_bins,
):
    _invalid = [
        _name
        for _name in loaded_names
        if low_percentiles.value[_name] >= high_percentiles.value[_name]
    ]
    mo.stop(
        bool(_invalid),
        mo.md("Low % must be below high % for: " + ", ".join(f"`{_n}`" for _n in _invalid)),
    )
    # Clipping bounds for every (channel, trajectory), from all its timesteps.
    normalisation_bins = {}
    joint_bounds_rows = {}
    for _name in loaded_names:
        _group, _channel = CHANNEL_SOURCE[_name]
        _samples = {}
        for _trajectory, _members in rescaling_trajectories.items():
            _values = []
            for (_member_group, _), _index in _members.items():
                if _member_group != _group or _index in flagged_indices:
                    continue
                _item = prepared[_index]
                _pixels = _item["values"][..., _channel]
                _values.append(_pixels[_item["mask"]] if inside_only.value else _pixels.ravel())
            if _values:
                _samples[_trajectory] = np.concatenate(_values)
        if not _samples:
            continue
        _percentiles = (low_percentiles.value[_name], high_percentiles.value[_name])
        for _trajectory, _bounds in trajectory_bins(
            _samples, *_percentiles, normalise_mode.value, knockout_bounds.value
        ).items():
            normalisation_bins[(_name, _trajectory)] = _bounds
        if knockout_bounds.value == "joint":
            # Each condition's own bounds next to the joint ones, one row per
            # channel (per channel and replicate with per-replicate bounds).
            for _trajectory, _own in condition_bins(
                _samples, *_percentiles, normalise_mode.value
            ).items():
                _replicate = (
                    _trajectory[1] + 1 if normalise_mode.value == "per_replicate" else "all"
                )
                _row = joint_bounds_rows.setdefault(
                    (_name, _replicate), {"channel": _name, "replicate": _replicate}
                )
                _row[_trajectory[0]] = "{:.1f} – {:.1f}".format(*_own)
                _row["joint"] = "{:.1f} – {:.1f}".format(
                    *normalisation_bins[(_name, _trajectory)]
                )
    for _row in joint_bounds_rows.values():
        _row["widened"] = _row.get("ctrl") != _row["joint"]
    joint_bounds_rows = list(joint_bounds_rows.values())
    return joint_bounds_rows, normalisation_bins


@app.cell(hide_code=True)
def _(joint_bounds_rows, knockout_bounds, mo):
    mo.stop(knockout_bounds.value != "joint")
    _widened = sorted({_row["channel"] for _row in joint_bounds_rows if _row["widened"]})
    mo.vstack(
        [
            mo.md(
                "**Joint knockout bounds.** Each condition's own bounds and "
                "the joint bounds all conditions use. *widened*: a knockout "
                "goes outside the control range, so the control bounds change. "
                + (
                    "Widened: " + ", ".join(f"`{_n}`" for _n in _widened) + "."
                    if _widened
                    else "No channel is widened."
                )
            ),
            mo.ui.table(joint_bounds_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    normalisation_bins,
    prepared,
    rescale,
    trajectories,
    zero_outside,
):
    def normalised_image(name, trajectory, hour):
        """Normalised image and mask, or ``None`` if the image was not measured."""
        group, channel = CHANNEL_SOURCE[name]
        index = trajectories[trajectory].get((group, hour))
        if index is None or (name, trajectory) not in normalisation_bins:
            return None
        item = prepared[index]
        values = rescale(item["values"][..., channel], *normalisation_bins[(name, trajectory)])
        if zero_outside.value:
            values = values * item["mask"]
        return values, item["mask"]

    return (normalised_image,)


@app.cell(hide_code=True)
def _(mo, trajectories, trajectory_label):
    grid_trajectory = mo.ui.dropdown(
        options={trajectory_label(_trajectory): _trajectory for _trajectory in trajectories},
        value=trajectory_label(next(iter(trajectories))),
        label="Trajectory",
    )
    grid_trajectory
    return (grid_trajectory,)


@app.cell(hide_code=True)
def _(
    CHANNEL_SOURCE,
    CONDITION_COLOURS,
    flagged_indices,
    grid_trajectory,
    loaded_names,
    mo,
    normalisation_bins,
    normalise_mode,
    normalised_image,
    np,
    outline_flagged,
    plt,
    selection,
    show_figure,
    trajectories,
    trajectory_label,
    zero_outside,
):
    _hours = selection["timesteps"]
    _trajectory = grid_trajectory.value
    _grid, _axes = plt.subplots(
        len(loaded_names),
        len(_hours),
        figsize=(1.9 * len(_hours), 1.9 * len(loaded_names)),
        squeeze=False,
    )
    for _row, _name in enumerate(loaded_names):
        for _column, _hour in enumerate(_hours):
            _axis = _axes[_row, _column]
            _axis.set_xticks([])
            _axis.set_yticks([])
            if _row == 0:
                _axis.set_title(f"{_hour}h", fontsize=8)
            if _column == 0:
                _axis.set_ylabel(_name, fontsize=7)
            _result = normalised_image(_name, _trajectory, _hour)
            if trajectories[_trajectory].get((CHANNEL_SOURCE[_name][0], _hour)) in flagged_indices:
                outline_flagged(_axis)
            if _result is None:
                _axis.set_facecolor("0.9")
                continue
            _values, _mask = _result
            _axis.imshow(_values, cmap="magma", vmin=0, vmax=1)
            if not zero_outside.value:
                _axis.contour(_mask, levels=[0.5], colors="cyan", linewidths=0.4)
    _grid.tight_layout()

    # Mean normalised intensity inside the mask over time, one line per trajectory.
    _columns = min(4, len(loaded_names))
    _plot_rows = int(np.ceil(len(loaded_names) / _columns))
    _curves, _curve_axes = plt.subplots(
        _plot_rows,
        _columns,
        figsize=(3.4 * _columns, 2.4 * _plot_rows),
        sharey=True,
        squeeze=False,
    )
    for _axis, _name in zip(_curve_axes.flat, loaded_names):
        for _other in trajectories:
            _points = [
                (_hour, float(np.mean(_result[0][_result[1]])))
                for _hour in _hours
                if trajectories[_other].get((CHANNEL_SOURCE[_name][0], _hour)) not in flagged_indices
                and (_result := normalised_image(_name, _other, _hour)) is not None
            ]
            if _points:
                _axis.plot(
                    *zip(*_points),
                    marker="o",
                    markersize=3,
                    linewidth=2.0 if _other == _trajectory else 0.8,
                    color=CONDITION_COLOURS[_other[0]],
                    alpha=1.0 if _other == _trajectory else 0.5,
                )
        _axis.set_title(_name, fontsize=8)
        _axis.set_xticks(_hours)
        _axis.tick_params(labelsize=7)
    for _axis in _curve_axes.flat[len(loaded_names):]:
        _axis.axis("off")
    _curves.tight_layout()

    _shared = normalise_mode.value != "per_replicate"
    _bin_rows = []
    for (_name, _other), (_low, _high) in normalisation_bins.items():
        if _other not in trajectories or (_shared and _other != next(iter(trajectories))):
            continue
        _row_entry = {"channel": _name}
        if not _shared:
            _row_entry["trajectory"] = trajectory_label(_other)
        _row_entry.update({"low": round(_low, 2), "high": round(_high, 2)})
        _bin_rows.append(_row_entry)

    mo.vstack(
        [
            mo.md(
                "All panels share the scale 0–1. Grey panels were not measured "
                "for this trajectory; red borders mark images flagged as low "
                "quality (left out of the bounds and the curves, as in training)."
            ),
            show_figure(_grid),
            mo.md(
                "**Mean normalised intensity inside the mask.** One line per "
                "trajectory (colour = condition), the selected one in bold. "
                "Compare normalisation modes here: with the bounds shared "
                "across replicates, real differences between replicates stay "
                "visible."
            ),
            _curves,
            mo.accordion(
                {
                    "Clipping bounds (intensity units before rescaling)": mo.ui.table(
                        _bin_rows, selection=None
                    )
                }
            ),
        ]
    )
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Cleaning steps figure
    """)
    return


@app.cell(hide_code=True)
def _(image_key, mo, raw_images, trajectories):
    # Images that can be shown, as (condition, replicate, group, hour) -> raw image index
    steps_images = {
        (*_trajectory, _group, _hour): _index
        for _trajectory, _members in trajectories.items()
        for (_group, _hour), _index in _members.items()
    }
    # Default: the colony of the 3D -> 2D schematic (Experiments/micropatterns/schematics.py)
    _schematic_key = "cell_fate_markers/sl24_s2/B3_F4_48h"
    _default = next(
        (
            _key
            for _key, _index in steps_images.items()
            if image_key(raw_images[_index]["record"]) == _schematic_key
        ),
        None,
    )
    _start = _default or next(iter(steps_images))

    def _choices(position):
        return sorted({_key[position] for _key in steps_images})

    steps_condition = mo.ui.dropdown(options=_choices(0), value=_start[0], label="Condition")
    steps_replicate = mo.ui.dropdown(
        options={str(_r + 1): _r for _r in _choices(1)}, value=str(_start[1] + 1), label="Replicate"
    )
    steps_group = mo.ui.dropdown(options=_choices(2), value=_start[2], label="Experiment group")
    steps_hour = mo.ui.dropdown(
        options={f"{_h}h": _h for _h in _choices(3)}, value=f"{_start[3]}h", label="Timestep"
    )
    steps_run = mo.ui.run_button(label="Draw cleaning steps")
    mo.vstack(
        [
            mo.md(
                "- One image after each pre-processing step, with the current settings\n"
                "- The last two panels need **Clean all loaded images** (section 5)\n"
                + (
                    ""
                    if _default
                    else f"- The schematic's colony (`{_schematic_key}`) is not loaded: load "
                    "**Nodal knockout at 24h** and **Cell fate (stain 2)** in section 1\n"
                )
            ),
            mo.hstack([steps_condition, steps_replicate, steps_group, steps_hour], justify="start"),
            steps_run,
        ]
    )
    return (
        steps_condition,
        steps_group,
        steps_hour,
        steps_images,
        steps_replicate,
        steps_run,
    )


@app.cell(hide_code=True)
def _(GROUP_CHANNELS, mo, steps_group):
    _names = list(GROUP_CHANNELS[steps_group.value])
    steps_channels = mo.ui.multiselect(options=_names, value=_names, label="Channels")
    steps_layout = mo.ui.dropdown(
        options={"One composite": "composite", "One row per channel": "rows"},
        value="One composite",
        label="Layout",
    )
    steps_scale = mo.ui.dropdown(
        options={
            "Each panel its own": "panel",
            "Same as the raw panel": "raw",
            "One scale for all (true intensities)": "all",
        },
        value="Each panel its own",
        label="Display scale",
    )
    steps_percentiles = mo.ui.range_slider(
        start=0.0, stop=100.0, step=0.1, value=[0.5, 99.5], label="Display percentiles", show_value=True
    )
    steps_show_background = mo.ui.checkbox(value=False, label="Show background estimate")
    steps_show_normalised = mo.ui.checkbox(value=True, label="Show normalised panel")
    steps_mark_hot = mo.ui.checkbox(value=True, label="Circle replaced hot pixels")
    steps_pixel_um = mo.ui.number(
        start=0.01, stop=10.0, step=0.001, value=0.592, label="Pixel size (µm, full resolution)"
    )
    steps_scale_bar = mo.ui.number(start=10, stop=500, step=10, value=100, label="Scale bar (µm)")
    steps_panel_size = mo.ui.slider(1.0, 4.0, step=0.1, value=2.0, label="Panel size (in)", show_value=True)
    # Total downsampling of the last panels, as data.downsample in training
    steps_downsample = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16, "32": 32}, value="4", label="Downsampling"
    )
    steps_page = mo.ui.dropdown(options=["white", "black"], value="white", label="Figure background")
    steps_histograms = mo.ui.checkbox(value=True, label="Intensity histograms")
    steps_log_counts = mo.ui.checkbox(value=True, label="Log counts")
    # Zoom square, as fractions of the raw image
    steps_zoom = mo.ui.checkbox(value=True, label="Zoomed squares")
    steps_zoom_x = mo.ui.slider(0.0, 1.0, step=0.005, value=0.5, label="Zoom centre x", show_value=True)
    steps_zoom_y = mo.ui.slider(0.0, 1.0, step=0.005, value=0.5, label="Zoom centre y", show_value=True)
    steps_zoom_size = mo.ui.slider(0.02, 0.5, step=0.005, value=0.1, label="Zoom size", show_value=True)
    steps_zoom_colour = mo.ui.text(value="red", label="Zoom outline colour (name or hex)")
    steps_name = mo.ui.text(value="cleaning_steps", label="File name")
    steps_save = mo.ui.run_button(label="Save PDF + PNG")
    mo.vstack(
        [
            mo.hstack([steps_channels, steps_layout, steps_scale], justify="start"),
            mo.hstack(
                [steps_percentiles, steps_show_background, steps_show_normalised, steps_mark_hot], justify="start"
            ),
            mo.hstack([steps_pixel_um, steps_scale_bar, steps_panel_size, steps_downsample], justify="start"),
            mo.hstack([steps_page, steps_histograms, steps_log_counts], justify="start"),
            mo.hstack([steps_zoom, steps_zoom_colour], justify="start"),
            mo.hstack([steps_zoom_x, steps_zoom_y, steps_zoom_size], justify="start"),
            mo.hstack([steps_name, steps_save], justify="start"),
        ]
    )
    return (
        steps_channels,
        steps_downsample,
        steps_histograms,
        steps_layout,
        steps_log_counts,
        steps_mark_hot,
        steps_name,
        steps_page,
        steps_panel_size,
        steps_percentiles,
        steps_pixel_um,
        steps_save,
        steps_scale,
        steps_scale_bar,
        steps_show_background,
        steps_show_normalised,
        steps_zoom,
        steps_zoom_colour,
        steps_zoom_size,
        steps_zoom_x,
        steps_zoom_y,
    )


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    GROUP_CHANNELS,
    clean_image,
    cleaning_settings,
    image_geometry,
    intensity_factor,
    mo,
    normalised_image,
    np,
    overview,
    place_image,
    raw_images,
    steps_condition,
    steps_group,
    steps_hour,
    steps_images,
    steps_replicate,
    steps_run,
):
    mo.stop(not steps_run.value, mo.md("Press **Draw cleaning steps** to draw the figure."))
    _key = (steps_condition.value, steps_replicate.value, steps_group.value, steps_hour.value)
    mo.stop(
        _key not in steps_images,
        mo.callout("This combination was not loaded; pick another one.", kind="warn"),
    )
    _index = steps_images[_key]
    _record = raw_images[_index]["record"]
    _names = GROUP_CHANNELS[steps_group.value]
    _steps = clean_image(raw_images[_index], cleaning_settings)

    # Centring, mask and normalisation as in section 5, at the training resolution
    _shift, _mask = image_geometry[_index]
    _masked = place_image(overview["images"][_index]["cleaned"], _shift) * _mask[..., None]
    _normalised = []
    for _name in _names:
        _result = normalised_image(_name, _key[:2], _key[3])
        _normalised.append(np.zeros(_mask.shape) if _result is None else _result[0])

    _scaled = any(
        intensity_factor(cleaning_settings["scales"], _name, _record.timestep) != 1.0 for _name in _names
    )
    _striped = any(
        cleaning_settings["h_stripes"].get(_name, 0) or cleaning_settings["v_stripes"].get(_name, 0)
        for _name in _names
    )
    # Each panel: title, [X, Y, C] values, downsampling, kind and the centring
    # shift (rows, columns) it went through (None if not moved)
    _panels = [
        ("raw × intensity factor" if _scaled else "raw", _steps["scaled"], 1, "raw", None),
        ("hot pixels replaced", _steps["despeckled"], 1, "step", None),
    ]
    if _striped:
        _panels.append(("stripes removed", _steps["destriped"], 1, "step", None))
    _panels += [
        ("background estimate", _steps["background"], 1, "background", None),
        ("background removed", _steps["cleaned"], 1, "step", None),
        ("centred, masked and downsampled", _masked, DISPLAY_DOWNSAMPLE, "step", _shift),
        ("normalised", np.stack(_normalised, axis=-1), DISPLAY_DOWNSAMPLE, "normalised", _shift),
    ]
    cleaning_steps = {
        "record": _record,
        "names": _names,
        "hot": _steps["hot"],
        "panels": [
            {"title": _t, "values": _v, "downsample": _d, "kind": _k, "shift": _s}
            for _t, _v, _d, _k, _s in _panels
        ],
    }
    return (cleaning_steps,)


@app.cell(hide_code=True)
def _(
    DISPLAY_DOWNSAMPLE,
    MICROPATTERN_260726_SCHEMA,
    block_mean,
    cleaning_steps,
    mo,
    np,
    plt,
    record_label,
    selection,
    show_figure,
    steps_channels,
    steps_downsample,
    steps_histograms,
    steps_layout,
    steps_log_counts,
    steps_mark_hot,
    steps_page,
    steps_panel_size,
    steps_percentiles,
    steps_pixel_um,
    steps_scale,
    steps_scale_bar,
    steps_show_background,
    steps_show_normalised,
    steps_zoom,
    steps_zoom_colour,
    steps_zoom_size,
    steps_zoom_x,
    steps_zoom_y,
):
    from matplotlib.colors import is_color_like as _is_color_like
    from matplotlib.patches import Rectangle as _Rectangle

    # Marker colours as in the 3D -> 2D schematic; other markers take the
    # spare colours in schema order
    _marker_colours = {
        "SOX2": (0.0, 0.78, 0.9),
        "SOX17": (1.0, 0.78, 0.0),
        "TBXT": (0.92, 0.12, 0.72),
        "LMBR": (1.0, 1.0, 1.0),
    }
    _spare_colours = [(0.0, 0.78, 0.9), (0.92, 0.12, 0.72), (1.0, 0.78, 0.0), (1.0, 0.5, 0.0), (0.15, 0.8, 0.2)]
    _markers = {_c.name: _c.marker for _c in MICROPATTERN_260726_SCHEMA.measurement_channels}
    _names = cleaning_steps["names"]
    _fixed = {_n: _marker_colours[_markers[_n]] for _n in _names if _markers[_n] in _marker_colours}
    _spare = iter([_c for _c in _spare_colours if _c not in _fixed.values()])
    _colours = {_n: _fixed.get(_n) or next(_spare) for _n in _names}
    _shown = [_n for _n in _names if _n in steps_channels.value]
    # Images stay on black; the page around them can be white or black
    _page = steps_page.value
    _text = "black" if _page == "white" else "white"
    # Histogram line colours: white lines would vanish on a white page
    _line_colours = {
        _n: (0.4, 0.4, 0.4) if _page == "white" and min(_c) > 0.85 else _c for _n, _c in _colours.items()
    }
    mo.stop(not _shown, mo.md("Pick at least one channel."))

    _panels = [
        _p
        for _p in cleaning_steps["panels"]
        if (_p["kind"] != "background" or steps_show_background.value)
        and (_p["kind"] != "normalised" or steps_show_normalised.value)
    ]
    # The downsampled panels are at DISPLAY_DOWNSAMPLE; like the training loader,
    # average further blocks after masking and normalisation to reach the
    # chosen downsampling. "extra" is that further factor.
    _extra = max(1, steps_downsample.value // DISPLAY_DOWNSAMPLE)
    _panels = [
        {**_p, "values": block_mean(_p["values"], _extra), "extra": _extra}
        if _p["downsample"] > 1 and _extra > 1
        else {**_p, "extra": 1}
        for _p in _panels
    ]
    _low, _high = steps_percentiles.value
    # With one scale for all, each channel runs from 0 to the high percentile of
    # its brightest panel (normalised values keep their own 0 to 1)
    _intensity_panels = [_p for _p in _panels if _p["kind"] != "normalised"]
    _shared_high = (
        {
            _c: max(float(np.percentile(_p["values"][..., _c], _high)) for _p in _intensity_panels)
            for _c in range(len(_names))
        }
        if steps_scale.value == "all" and _intensity_panels
        else {}
    )

    def _limits(values, channel):
        return np.percentile(values[..., channel], (_low, _high))

    _raw = cleaning_steps["panels"][0]["values"]

    def _scaled(panel, channel):
        """One channel of a panel in [0, 1] for display."""
        values = panel["values"][..., channel]
        if panel["kind"] == "normalised":
            return np.clip(values, 0.0, 1.0)
        if steps_scale.value == "all":
            lo, hi = 0.0, _shared_high[channel]
        else:
            lo, hi = _limits(_raw if steps_scale.value == "raw" else panel["values"], channel)
        return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    def _rgb(panel, channels):
        """Composite of some channels of a panel, on black."""
        return np.clip(
            sum(_scaled(panel, _names.index(n))[..., None] * np.asarray(_colours[n]) for n in channels), 0.0, 1.0
        )

    _zoom_colour = steps_zoom_colour.value.strip() if _is_color_like(steps_zoom_colour.value.strip()) else "red"

    def _zoom_window(panel):
        """(top, bottom, left, right) of the zoom square in a panel's pixel coordinates.

        The square is placed on the raw image. Downsampled panels average
        blocks of ``downsample`` pixels, may then be shifted by centring, and
        then average blocks of ``extra`` of those pixels.
        """
        height, width = _raw.shape[:2]
        work, extra = panel["downsample"], panel["extra"]
        half = steps_zoom_size.value * min(height, width) / 2 / (work * extra)
        centre = (steps_zoom_y.value * height - 0.5, steps_zoom_x.value * width - 0.5)
        shift = panel["shift"] or (0.0, 0.0)
        row, column = (
            ((c - (work - 1) / 2) / work + s - (extra - 1) / 2) / extra for c, s in zip(centre, shift)
        )
        return row - half, row + half, column - half, column + half

    def _mark_hot(axis, channels):
        hot = cleaning_steps["hot"][..., [_names.index(n) for n in channels]].any(axis=-1)
        hot_rows, hot_columns = np.nonzero(hot)
        axis.scatter(hot_columns, hot_rows, s=14, facecolors="none", edgecolors="lime", linewidths=0.6)

    _rows = [_shown] if steps_layout.value == "composite" else [[_n] for _n in _shown]
    _size = steps_panel_size.value
    # Figure rows, top to bottom, with their heights in inches
    _layout = [("image", _size)] * len(_rows)
    if steps_zoom.value:
        _layout += [("gap", 0.08), ("zoom", _size)]
    if steps_histograms.value:
        _layout += [("gap", 0.08), ("histogram", 0.5 * _size)]
    # Room above the panels for their titles, and for the legend with a composite;
    # below them for the histogram tick labels
    _top = 0.45 if steps_layout.value == "rows" else 0.75
    _bottom = 0.35 if _layout[-1][0] == "histogram" else 0.05
    _height_in = sum(_h for _, _h in _layout) + _top + _bottom
    steps_figure = plt.figure(figsize=(_size * len(_panels), _height_in), facecolor=_page)
    _grid = steps_figure.add_gridspec(
        len(_layout),
        len(_panels),
        height_ratios=[_h for _, _h in _layout],
        wspace=0.04,
        hspace=0.02,
        left=0.03,
        right=0.99,
        bottom=_bottom / _height_in,
        top=1 - _top / _height_in,
    )
    _row_of = {_kind: _i for _i, (_kind, _) in enumerate(_layout)}
    for _row, _channels in enumerate(_rows):
        for _column, _panel in enumerate(_panels):
            _axis = steps_figure.add_subplot(_grid[_row, _column])
            _axis.imshow(_rgb(_panel, _channels), interpolation="antialiased")
            if steps_zoom.value:
                _top_edge, _bottom_edge, _left, _right = _zoom_window(_panel)
                _axis.add_patch(_Rectangle((_left, _top_edge), _right - _left, _bottom_edge - _top_edge,
                                           fill=False, edgecolor=_zoom_colour, linewidth=1.0))
            _axis.set_xticks([])
            _axis.set_yticks([])
            for _spine in _axis.spines.values():
                _spine.set_visible(False)
            if _row == 0:
                _axis.set_title(f"{_column + 1}. {_panel['title']}", fontsize=8, color=_text)
            if _column == 0 and steps_layout.value == "rows":
                _axis.set_ylabel(_markers[_channels[0]], fontsize=8, color=_text)
            if _column == 0 and steps_mark_hot.value:
                _mark_hot(_axis, _channels)
            if _row == len(_rows) - 1 and _column == 0:
                # Every panel shows the same field, so one scale bar serves all
                _height, _width = _panel["values"].shape[:2]
                _length = steps_scale_bar.value / (steps_pixel_um.value * _panel["downsample"])
                _pad = 0.05 * _width
                _axis.plot([_width - _pad - _length, _width - _pad], [_height - _pad] * 2, color="white", lw=2,
                           solid_capstyle="butt")
                _axis.text(_width - _pad - _length / 2, _height - _pad - 0.02 * _height,
                           f"{steps_scale_bar.value:g} µm", color="white", ha="center", va="bottom", fontsize=6)

    # Intensity histogram of each step (all pixels, in that step's own units)
    if steps_histograms.value:
        for _column, _panel in enumerate(_panels):
            _axis = steps_figure.add_subplot(_grid[_row_of["histogram"], _column])
            _channel_values = [_panel["values"][..., _names.index(_n)].ravel() for _n in _shown]
            _range = (min(float(_v.min()) for _v in _channel_values), max(float(_v.max()) for _v in _channel_values))
            if steps_scale.value == "all" and _panel["kind"] != "normalised":
                # One intensity axis for all steps before normalisation
                _range = (
                    min(float(_p["values"][..., [_names.index(_n) for _n in _shown]].min()) for _p in _intensity_panels),
                    max(float(_p["values"][..., [_names.index(_n) for _n in _shown]].max()) for _p in _intensity_panels),
                )
            if _range[1] <= _range[0]:
                _range = (_range[0], _range[0] + 1.0)
            for _n, _v in zip(_shown, _channel_values):
                _axis.hist(_v, bins=100, range=_range, histtype="step", color=_line_colours[_n], lw=0.8,
                           log=steps_log_counts.value)
            _axis.set_facecolor(_page)
            _axis.set_yticks([])
            _axis.minorticks_off()
            _axis.tick_params(colors=_text, labelsize=6, length=2)
            for _side in ("top", "right", "left"):
                _axis.spines[_side].set_visible(False)
            _axis.spines["bottom"].set_color(_text)
            if _column == 0:
                _axis.set_ylabel("count (log)" if steps_log_counts.value else "count", fontsize=7, color=_text)

    # The zoom square of every step, pixels drawn as they are (no smoothing)
    if steps_zoom.value:
        for _column, _panel in enumerate(_panels):
            _axis = steps_figure.add_subplot(_grid[_row_of["zoom"], _column])
            _axis.imshow(_rgb(_panel, _shown), interpolation="nearest")
            _top_edge, _bottom_edge, _left, _right = _zoom_window(_panel)
            _axis.set_xlim(_left, _right)
            _axis.set_ylim(_bottom_edge, _top_edge)
            _axis.set_xticks([])
            _axis.set_yticks([])
            for _spine in _axis.spines.values():
                _spine.set_edgecolor(_zoom_colour)
                _spine.set_linewidth(1.5)
            if _column == 0:
                _axis.set_ylabel("zoom", fontsize=8, color=_text)
                if steps_mark_hot.value:
                    _mark_hot(_axis, _shown)

    if steps_layout.value == "composite":
        _handles = [plt.Line2D([], [], color=_colours[_n], lw=2) for _n in _shown]
        steps_figure.legend(_handles, [_markers[_n] for _n in _shown], loc="upper right", ncol=len(_shown), fontsize=7,
                            bbox_to_anchor=(0.99, 0.995), labelcolor="white",
                            frameon=True, fancybox=False, facecolor="black", edgecolor="black", framealpha=1.0)
    _hot_count = int(cleaning_steps["hot"][..., [_names.index(_n) for _n in _shown]].sum())
    mo.vstack(
        [
            show_figure(steps_figure, dpi=150),
            mo.md(
                f"- {selection['projection']} projection, {record_label(cleaning_steps['record'])}\n"
                f"- {_hot_count} hot pixels replaced in the channels shown"
            ),
        ]
    )
    return (steps_figure,)


@app.cell(hide_code=True)
def _(Path, mo, steps_figure, steps_name, steps_save):
    mo.stop(not steps_save.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    for _suffix in ("pdf", "png"):
        steps_figure.savefig(_directory / f"{steps_name.value}.{_suffix}", dpi=400,
                             facecolor=steps_figure.get_facecolor())
    mo.md(f"Saved `{steps_name.value}.pdf` and `.png` to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()
