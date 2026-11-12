"""3D confocal stack → 2D slice, max and sum projections, drawn from a real stack.

Run with:
    marimo edit Experiments/micropatterns/figures/imaging_schematic.py

The 3D stack path is read from ``MICROPATTERN_3D_STACK`` (e.g. in ``.env``)
or typed into the notebook.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="imaging schematic")

with app.setup:
    import os
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv
    from matplotlib.patches import ConnectionPatch

    from Experiments.micropatterns.figures.data import (
        load_training_projection,
        locate_image,
        normalise,
        project_stack,
        projection_data_config,
        read_stack,
    )
    from Experiments.micropatterns.figures.drawing import (
        add_length_marker,
        add_scale_bar,
        composite,
        draw_profile,
        draw_stack,
    )
    from Experiments.micropatterns.figures.style import (
        ACCENT,
        CHANNEL_COLOURS,
        FIGURE_CHANNELS,
        FONT,
        INK,
        LINE,
        MARKER_COLOURS,
        MUTED,
        PROJECTION_LABELS,
        STAIN_NAMES,
        WIDTH,
        legend_options,
        legend_swatch,
        paper_style,
        save_figure,
    )

    load_dotenv()

    # Channels of files without marker names get these colours in turn
    FALLBACK_COLOURS = ("yellow", "cyan", "magenta")
    SHOWN_MARKERS = {marker for markers in FIGURE_CHANNELS.values() for marker in markers}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Imaging schematic

    - 3D confocal stack → 2D slice, max and sum projections
    - Optional last column: the same colony as the training dataloader returns it
    - Saved to `figures/schematics/`
    """)
    return


@app.cell(hide_code=True)
def _():
    stack_path = mo.ui.text(
        value=os.getenv("MICROPATTERN_3D_STACK", ""),
        label="3D stack (.ome.tiff)",
        full_width=True,
    )
    pyramid_level = mo.ui.dropdown(
        options={"540 px (fast)": 1, "1080 px (full)": 0}, value="540 px (fast)", label="Resolution"
    )
    stain = mo.ui.dropdown(options=["fluorophores (from file)", *STAIN_NAMES], value="cell fate s2",
                           label="Channel names")
    custom_names = mo.ui.text(placeholder="e.g. SOX17, SOX2, TBXT, LMBR", label="or type them (file order)")
    mo.vstack([stack_path, mo.hstack([pyramid_level, stain, custom_names], justify="start")])
    return custom_names, pyramid_level, stack_path, stain


@app.cell
def _(custom_names, pyramid_level, stack_path, stain):
    mo.stop(not stack_path.value, mo.callout("Set `MICROPATTERN_3D_STACK` or type a path above.", kind="warn"))
    with mo.status.spinner("Reading stack"):
        stack, dx, dz, file_channel_names = read_stack(Path(stack_path.value).expanduser(), pyramid_level.value)
    # The file only records the fluorophore of each channel, not the marker it labels
    _typed = [n.strip() for n in custom_names.value.split(",") if n.strip()]
    if len(_typed) == stack.shape[1]:
        channel_names = _typed
    else:
        channel_names = list(STAIN_NAMES.get(stain.value) or file_channel_names)
    mo.md(f"Stack `{stack.shape}` (Z, C, Y, X), {dx:.2f} µm pixels, {dz:.2f} µm z-step, channels: {', '.join(channel_names)}")
    field_width_um = stack.shape[-1] * dx
    return channel_names, dx, dz, field_width_um, stack


@app.cell(hide_code=True)
def _(channel_names, dz, stack):
    channel_shown = mo.ui.array([
        mo.ui.checkbox(value=name in SHOWN_MARKERS or name not in MARKER_COLOURS, label=name)
        for name in channel_names
    ])
    _depth = stack.shape[0]
    z_range = mo.ui.range_slider(start=0, stop=_depth, step=1, value=[0, _depth], label="z range (projections)",
                                 show_value=True)
    z_slice = mo.ui.slider(start=0, stop=_depth - 1, step=1, value=_depth // 2,
                           label=f"Slice depth z₀ (index, {dz:g} µm steps)", show_value=True, full_width=True)
    crop_size = mo.ui.slider(start=0.2, stop=1.0, step=0.01, value=1.0, label="Crop size", show_value=True)
    crop_x = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Crop centre x", show_value=True)
    crop_y = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Crop centre y", show_value=True)
    probe_x = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Probe x", show_value=True, full_width=True)
    probe_y = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Probe y", show_value=True, full_width=True)
    probe_radius = mo.ui.slider(start=0, stop=10, step=1, value=2, label="Probe radius (px)", show_value=True)
    column_width = mo.ui.slider(start=0.01, stop=0.15, step=0.005, value=0.05, label="Column width",
                                show_value=True)
    mo.vstack([
        mo.md("**Channels shown**"),
        mo.hstack(list(channel_shown), justify="start"),
        mo.md("**Depth**"),
        z_range,
        mo.md("**Field of view**"),
        mo.hstack([crop_size, crop_x, crop_y], justify="start"),
        mo.md("**Probe column** (position sliders are above the figure)"),
        mo.hstack([probe_radius, column_width], justify="start"),
    ])
    return (
        channel_shown,
        column_width,
        crop_size,
        crop_x,
        crop_y,
        probe_radius,
        probe_x,
        probe_y,
        z_range,
        z_slice,
    )


@app.cell(hide_code=True)
def _():
    low_percentile = mo.ui.number(start=0.0, stop=50.0, step=0.1, value=1.0, label="Low percentile")
    high_percentile = mo.ui.number(start=50.0, stop=100.0, step=0.01, value=99.8, label="High percentile")
    gamma = mo.ui.slider(start=0.3, stop=2.0, step=0.05, value=0.8, label="Gamma", show_value=True)
    n_planes = mo.ui.slider(start=3, stop=24, step=1, value=8, label="Planes drawn", show_value=True)
    plane_opacity = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.6, label="Plane opacity", show_value=True)
    lmbr_alpha = mo.ui.switch(value=False, label="LMBR as plane alpha")
    empty_opacity = mo.ui.slider(start=0.0, stop=0.6, step=0.01, value=0.05, label="Empty plane opacity (LMBR)",
                                 show_value=True)
    mask_gain = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=2.0, label="Cell opacity gain (LMBR)",
                             show_value=True)
    stack_height = mo.ui.slider(start=0.4, stop=2.5, step=0.05, value=1.2, label="Stack height", show_value=True)
    z_exaggeration = mo.ui.slider(start=1.0, stop=10.0, step=0.5, value=3.0, label="XZ z exaggeration",
                                  show_value=True)
    scale_bar_um = mo.ui.number(start=10, stop=500, step=10, value=100, label="Scale bar (µm)")
    figure_height = mo.ui.slider(start=2.5, stop=8.0, step=0.1, value=4.7, label="Height (in)", show_value=True)
    figure_width = mo.ui.slider(start=5.0, stop=10.0, step=0.05, value=round(WIDTH["double"], 2), label="Width (in)",
                                show_value=True)
    stack_column = mo.ui.slider(start=1.5, stop=5.0, step=0.1, value=2.8, label="Stack column width",
                                show_value=True)
    stack_share = mo.ui.slider(start=1.0, stop=5.0, step=0.1, value=2.0, label="Stack : XZ height",
                               show_value=True)
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([low_percentile, high_percentile, gamma], justify="start"),
        mo.hstack([n_planes, plane_opacity, stack_height, z_exaggeration], justify="start"),
        mo.hstack([lmbr_alpha, empty_opacity, mask_gain], justify="start"),
        mo.hstack([scale_bar_um, figure_width, figure_height], justify="start"),
        mo.hstack([stack_column, stack_share], justify="start"),
    ])
    return (
        empty_opacity,
        figure_height,
        figure_width,
        gamma,
        high_percentile,
        lmbr_alpha,
        low_percentile,
        mask_gain,
        n_planes,
        plane_opacity,
        scale_bar_um,
        stack_column,
        stack_height,
        stack_share,
        z_exaggeration,
    )


@app.cell(hide_code=True)
def _():
    show_training = mo.ui.switch(value=False, label="Show training data column")
    dataset_root = mo.ui.text(
        value=str(Path(os.getenv("DATA_PATH_BASE", "")) / "260726_nca_dataset"),
        label="2D dataset root",
        full_width=True,
    )
    image_key_text = mo.ui.text(value="cell_fate_markers/sl24_s2/B3_F4_48h", label="Image (same colony as the stack)")
    replicate = mo.ui.number(start=0, stop=20, step=1, value=0, label="Replicate (0-based)")
    training_downsample = mo.ui.dropdown(options={"1": 1, "4": 4, "8": 8, "16": 16}, value="8", label="Downsample")
    mo.vstack([
        mo.md("**Training data** (current micropattern dataloader, projection sweep settings)"),
        show_training,
        dataset_root,
        mo.hstack([image_key_text, replicate, training_downsample], justify="start"),
    ])
    return (
        dataset_root,
        image_key_text,
        replicate,
        show_training,
        training_downsample,
    )


@app.cell(hide_code=True)
def _(
    dataset_root,
    field_width_um,
    image_key_text,
    replicate,
    show_training,
    training_downsample,
):
    # Each projection is loaded as training loads it, so the clipping bounds (over
    # the whole trajectory), centring and boundary mask match training exactly.
    training_images = None
    _message = None
    _match = re.fullmatch(r"cell_fate_markers/(\w+?)_(s\d)/.+_\d+h", image_key_text.value)
    if show_training.value and _match is None:
        _message = mo.callout("The image key should look like `cell_fate_markers/sl24_s2/B3_F4_48h`.", kind="warn")
    elif show_training.value:
        _condition, _group = _match.group(1), f"cell_fate_{_match.group(2)}"
        _images, _missing, _names = {}, [], ()
        with mo.status.progress_bar(total=len(PROJECTION_LABELS), title="Loading training data") as _bar:
            for _projection in PROJECTION_LABELS:
                _data_config = projection_data_config(_projection, training_downsample.value)
                _dataset = load_training_projection(_data_config, dataset_root.value, _condition, _group,
                                                    int(replicate.value))
                _index = locate_image(np.asarray(_dataset.aux["source_files"]), dataset_root.value,
                                      image_key_text.value)
                if _index is None:
                    _missing.append(_projection)
                else:
                    _image = np.asarray(_dataset.data[_index[0], _index[1]])
                    _images[_projection] = {
                        "image": _image,
                        "dx": field_width_um / _image.shape[-1],
                        "downsample": training_downsample.value,
                    }
                    _names = _dataset.channel_names
                _bar.update()
        if _missing:
            _message = mo.callout(
                f"`{image_key_text.value}` is not replicate {replicate.value} of {_condition} {_group} "
                f"({', '.join(_missing)}). Try another replicate number.",
                kind="warn",
            )
        else:
            training_images = _images
            _message = mo.md(f"Training channels (drawn in the stack's channel order): {', '.join(_names)}")
    _message
    return (training_images,)


@app.cell(hide_code=True)
def _(
    crop_size,
    crop_x,
    crop_y,
    high_percentile,
    low_percentile,
    stack,
    z_range,
    z_slice,
):
    # Square crop of the field of view (a view, not a copy)
    _side = int(crop_size.value * min(stack.shape[-2:]))
    _row0 = int(np.clip(crop_y.value * stack.shape[-2] - _side / 2, 0, stack.shape[-2] - _side))
    _col0 = int(np.clip(crop_x.value * stack.shape[-1] - _side / 2, 0, stack.shape[-1] - _side))
    cropped = stack[:, :, _row0:_row0 + _side, _col0:_col0 + _side]

    z_bounds = (z_range.value[0], max(z_range.value[1], z_range.value[0] + 1))
    projections = {mode: project_stack(cropped, mode, z_slice.value, z_bounds) for mode in PROJECTION_LABELS}

    # Slice, max and stack share limits (lower from the stack, upper from the max
    # projection); the sum has its own.
    _step = max(1, _side // 256)
    _sample = cropped[z_bounds[0]:z_bounds[1]:2, :, ::_step, ::_step]
    _lo = np.percentile(_sample, low_percentile.value, axis=(0, 2, 3))
    _hi = np.percentile(projections["max"], high_percentile.value, axis=(1, 2))
    limits = {
        "slice": (_lo, _hi),
        "max": (_lo, _hi),
        "sum": (
            np.percentile(projections["sum"], low_percentile.value, axis=(1, 2)),
            np.percentile(projections["sum"], high_percentile.value, axis=(1, 2)),
        ),
    }
    return cropped, limits, projections, z_bounds


@app.cell(hide_code=True)
def _(
    channel_names,
    channel_shown,
    column_width,
    cropped,
    dx,
    dz,
    empty_opacity,
    figure_height,
    figure_width,
    gamma,
    limits,
    lmbr_alpha,
    mask_gain,
    n_planes,
    plane_opacity,
    probe_radius,
    probe_x,
    probe_y,
    projections,
    scale_bar_um,
    stack_column,
    stack_height,
    stack_share,
    training_images,
    z_bounds,
    z_exaggeration,
    z_slice,
):
    _colours = [
        CHANNEL_COLOURS[MARKER_COLOURS.get(name, FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)])] if shown else None
        for i, (name, shown) in enumerate(zip(channel_names, channel_shown.value))
    ]
    # Line colours for the profiles: white lines would vanish on the page
    _line_colours = [None if c is None else ((0.4, 0.4, 0.4) if min(c) > 0.85 else c) for c in _colours]
    _lo, _hi = limits["slice"]
    _z0, _z1 = z_bounds
    _side = cropped.shape[-1]
    _probe_col = int(probe_x.value * (_side - 1))
    _probe_row = int(probe_y.value * (_side - 1))
    _training = training_images is not None
    # LMBR marks cell presence, so it can serve as the plane alpha even when not shown
    _lmbr = channel_names.index("LMBR") if lmbr_alpha.value and "LMBR" in channel_names else None

    def _rgba(planes_czyx):
        """Composite and alpha of channel-first data; alpha follows LMBR or the brightest shown channel."""
        _norm = normalise(planes_czyx, _lo, _hi, gamma.value)
        _rgb = composite(_norm, _colours)
        _shown = [i for i, c in enumerate(_colours) if c is not None]
        if _lmbr is not None:
            _alpha = np.clip(mask_gain.value * _norm[_lmbr], 0.0, 1.0)
        else:
            _alpha = _norm[_shown].max(axis=0) if _shown else np.zeros(_norm.shape[1:])
        return np.concatenate([_rgb, _alpha[..., None]], axis=-1)

    with paper_style():
        schematic = plt.figure(figsize=(figure_width.value, figure_height.value))
        # Optional last column: the same projections as the training dataloader returns them
        _ratios = [stack_column.value, 0.2, 0.75, 1.0] + ([1.0] if _training else [])
        _grid = schematic.add_gridspec(3, len(_ratios), width_ratios=_ratios, height_ratios=[1, 1, 1],
                                       wspace=0.08, hspace=0.25, left=0.02, right=0.98, top=0.9, bottom=0.05)
        # The left column has its own split, so the stack can grow without resizing the projection rows
        _left = _grid[:, 0].subgridspec(2, 1, height_ratios=[stack_share.value, 1], hspace=0.25)

        # Oblique stack: evenly spaced planes plus the slice plane
        _ax_stack = schematic.add_subplot(_left[0])
        _plane_z = np.unique(np.r_[np.linspace(_z0, _z1 - 1, n_planes.value).round().astype(int), z_slice.value])
        _plane_z = _plane_z[(_plane_z >= _z0) & (_plane_z < _z1) | (_plane_z == z_slice.value)]
        _stride = max(1, _side // 256)
        _planes = [_rgba(cropped[z, :, ::_stride, ::_stride].astype(np.float32)) for z in _plane_z]
        _span = max(_z1 - 1 - _z0, 1)
        _heights = [(z - _z0) / _span * stack_height.value for z in _plane_z]
        _highlight = int(np.flatnonzero(_plane_z == z_slice.value)[0])

        # Probe window: its mean colours the column and gives the z-profiles below
        _r = probe_radius.value
        _window = cropped[_z0:_z1, :, max(_probe_row - _r, 0):_probe_row + _r + 1,
                          max(_probe_col - _r, 0):_probe_col + _r + 1]
        _window_mean = _window.mean(axis=(2, 3))
        _voxel_colours = composite(normalise(_window_mean.T, _lo, _hi, gamma.value), _colours)
        _voxel_heights = (np.arange(_z0, _z1) - _z0) / _span * stack_height.value
        _column = ((probe_x.value, probe_y.value), column_width.value, _voxel_colours, _voxel_heights,
                   z_slice.value - _z0)
        # With the LMBR mask, the black backing of each plane is faded so empty areas are see-through
        _backing = empty_opacity.value if _lmbr is not None else plane_opacity.value
        _column_top = draw_stack(_ax_stack, _planes, _heights, _highlight, _backing, _column)
        # Label above the stack, joined to the top of the column
        _stack_top = _heights[-1] + 0.3
        _ax_stack.annotate("single pixel z column", xy=_column_top, xytext=(_column_top[0], _stack_top + 0.1),
                           ha="center", va="bottom", fontsize=FONT["annotation"],
                           arrowprops=dict(arrowstyle="-", color=MUTED, lw=LINE["axis"], shrinkA=1, shrinkB=0),
                           zorder=100)
        _ax_stack.set_ylim(top=_stack_top + 0.22)
        _ax_stack.annotate("", xy=(-0.08, _heights[-1]), xytext=(-0.08, _heights[0]),
                           arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
        _ax_stack.text(-0.1, (_heights[0] + _heights[-1]) / 2, f"z  ({(_z1 - _z0) * dz:.0f} µm)", rotation=90,
                       ha="right", va="center")
        # X along the front edge of the bottom plane, Y along its right edge
        _length = f"{_side * dx:.0f} µm"
        add_length_marker(_ax_stack, (0, _heights[0] - 0.05), (1, _heights[0] - 0.05), f"x  {_length}")
        add_length_marker(_ax_stack, (1.05, _heights[0] - 0.05), (1.6, _heights[0] + 0.25), f"y  {_length}",
                          offset=(3, -3), rotation=np.degrees(np.arctan2(0.3, 0.55)))
        _ax_stack.text(1.57, _heights[_highlight] + 0.3, r"$z_0$", color=ACCENT, ha="left", va="center")
        _ax_stack.set_title("3D confocal stack")

        # XZ section through the probe row, showing the z range and slice
        _ax_section = schematic.add_subplot(_left[1])
        _section = cropped[:, :, _probe_row, :].transpose(1, 0, 2).astype(np.float32)
        _ax_section.imshow(composite(normalise(_section, _lo, _hi, gamma.value), _colours),
                           origin="lower", extent=(0, _side * dx, 0, cropped.shape[0] * dz),
                           aspect=z_exaggeration.value, interpolation="antialiased")
        _ax_section.axhspan(_z0 * dz, _z1 * dz, color=MUTED, alpha=0.12, lw=0)
        _ax_section.axhline((z_slice.value + 0.5) * dz, color=ACCENT, lw=LINE["data"])
        _ax_section.axvline((_probe_col + 0.5) * dx, color="white", lw=0.8)
        add_length_marker(_ax_section, (0, -0.06), (_side * dx, -0.06), f"x  {_side * dx:.0f} µm",
                          xycoords=("data", "axes fraction"))
        add_length_marker(_ax_section, (-0.02, 0), (-0.02, cropped.shape[0] * dz),
                          f"z  {cropped.shape[0] * dz:.0f} µm", xycoords=("axes fraction", "data"),
                          offset=(-3, 0), rotation=90, ha="center", va="bottom")
        _ax_section.set_title("XZ section" if z_exaggeration.value == 1 else f"XZ section (z ×{z_exaggeration.value:g})")
        _ax_section.axis("off")

        # One row per projection: z-profile at the probe, the 2D result, and
        # optionally the training image. The profile is on the slice's scale (not clipped).
        _profile = (_window_mean - _lo) / np.maximum(_hi - _lo, 1e-6)
        _z_um = (np.arange(_z0, _z1) + 0.5) * dz
        _scale_px = scale_bar_um.value / dx
        for _row, (_mode, (_name, _formula)) in enumerate(PROJECTION_LABELS.items()):
            _ax_profile = schematic.add_subplot(_grid[_row, 2])
            draw_profile(_ax_profile, _profile, _z_um, _mode, (z_slice.value + 0.5) * dz, _line_colours,
                         channel_names)
            _ax_profile.set_title(_name)
            if _row == 1:
                _ax_profile.set_ylabel("z (µm)")
            if _row == 2:
                _ax_profile.set_xlabel("intensity at probe")

            _ax_image = schematic.add_subplot(_grid[_row, 3])
            _image = normalise(projections[_mode], *limits[_mode], gamma.value)
            _ax_image.imshow(composite(_image, _colours), interpolation="antialiased")
            _ax_image.plot(_probe_col, _probe_row, "s", ms=5, mfc="none", mec="white", mew=0.9)
            _ax_image.set_title(_formula, loc="center", fontweight="normal")
            _ax_image.axis("off")
            if _row == 0:
                add_scale_bar(_ax_image, _scale_px, f"{scale_bar_um.value:g} µm", (_side, _side))
            _arrows = [((1.04, 0.5), _ax_profile, (-0.04, 0.5), _ax_image)]

            if _training:
                # Already in [0, 1], so drawn without extra rescaling or gamma
                _train = training_images[_mode]
                _ax_train = schematic.add_subplot(_grid[_row, 4])
                _ax_train.imshow(composite(_train["image"], _colours), interpolation="antialiased")
                _ax_train.axis("off")
                if _row == 0:
                    _ax_train.set_title("processed training data", loc="center", fontweight="normal")
                    add_scale_bar(_ax_train, scale_bar_um.value / _train["dx"], f"{scale_bar_um.value:g} µm",
                                  _train["image"].shape[-2:])
                _arrows.append(((1.04, 0.5), _ax_image, (-0.04, 0.5), _ax_train))

            for _start, _start_axes, _end, _end_axes in _arrows:
                schematic.add_artist(ConnectionPatch(
                    xyA=_start, coordsA=_start_axes.transAxes, xyB=_end, coordsB=_end_axes.transAxes,
                    arrowstyle="-|>", color=MUTED, lw=LINE["data"], mutation_scale=9,
                ))

        _handles = [legend_swatch(c) for c in _colours if c is not None]
        _labels = [n for n, c in zip(channel_names, _colours) if c is not None]
        schematic.legend(_handles, _labels, loc="upper right", ncol=max(1, len(_labels)),
                         bbox_to_anchor=(0.98, 1.0), **legend_options("white"))
    _warning = ([mo.callout("No `LMBR` channel in the channel names, so the plane alpha is unchanged.", kind="warn")]
                if lmbr_alpha.value and _lmbr is None else [])
    mo.vstack([*_warning, z_slice, mo.hstack([probe_x, probe_y], widths="equal"), schematic])
    return (schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="projection_schematic", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_button = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_button], justify="start")
    return figure_name, save_button, save_svg


@app.cell
def _(figure_name, save_button, save_svg, schematic):
    mo.stop(not save_button.value)
    _paths = save_figure(schematic, figure_name.value, "schematics", svg=save_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
