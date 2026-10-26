"""Figures and schematics made from micropattern data.

Run with:
    marimo edit Experiments/micropatterns/schematics.py

The 3D stack path is read from ``MICROPATTERN_3D_STACK`` (e.g. in ``.env``)
or typed into the notebook.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="schematics")

with app.setup:
    import os
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    from dotenv import load_dotenv
    from matplotlib.patches import ConnectionPatch, Polygon, Rectangle
    from matplotlib.colors import is_color_like
    from matplotlib.transforms import Affine2D

    load_dotenv()

    # Channel colours, added like light on a black background
    COLOURS = {
        "off": None,
        "cyan": (0.0, 0.78, 0.9),
        "magenta": (0.92, 0.12, 0.72),
        "yellow": (1.0, 0.78, 0.0),
        "red": (0.9, 0.15, 0.1),
        "green": (0.15, 0.8, 0.2),
        "blue": (0.15, 0.35, 1.0),
        "grey": (0.6, 0.6, 0.6),
        "white": (1.0, 1.0, 1.0),
    }
    MARKER_COLOURS = {"SOX2": "cyan", "SOX17": "yellow", "TBXT": "magenta", "LMBR": "white"}

    THEMES = {
        "light": {"bg": "white", "fg": "#222222", "muted": "#9a9a9a", "accent": "#d62728", "column": "white"},
        "dark": {"bg": "#111111", "fg": "#eeeeee", "muted": "#777777", "accent": "#ff5a5a", "column": "white"},
    }

    # Marker names of the cell fate stainings, in file channel order
    # (see Common/dataloader/micropattern_schemas.py).
    STAIN_NAMES = {
        "fluorophores (from file)": None,
        "cell fate s1": ("SOX17", "SOX2", "TBXT", "LMBR"),
        "cell fate s2": ("SOX17", "FOXA2", "TBXT", "LMBR"),
    }

    # Per-projection cleaning of the projection training runs, copied from
    # conf/experiments/micropattern_projections_training_sweep.yaml
    PROJECTION_CLEANING = {
        "max": {"horizontal_stripe_widths": {}, "stripe_block": 0},
        "sum": {"horizontal_stripe_widths": {"cell_fate_s1/TBXT": 10, "cell_fate_s2/TBXT": 10}, "stripe_block": 270},
        "slice": {"horizontal_stripe_widths": {}, "stripe_block": 0},
    }

    PROJECTION_LABELS = {
        "slice": ("Slice", r"$I(x, y, z_0)$"),
        "max": ("Max projection", r"$\max_z\, I(x, y, z)$"),
        "sum": ("Sum projection", r"$\sum_z\, I(x, y, z)$"),
    }


@app.function(hide_code=True)
def read_stack(path, level=1):
    """[Z, C, Y, X] stack of one OME-TIFF pyramid level, its voxel size (µm) and channel names."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "ZCYX":
            raise ValueError(f"Expected ZCYX axes, got {series.axes}")
        stack = series.levels[level].asarray()
        ome = tif.ome_metadata or ""

    def size(name):
        match = re.search(rf'{name}="([^"]+)"', ome)
        return float(match.group(1)) if match else 1.0

    dx = size("PhysicalSizeX") * series.shape[-1] / stack.shape[-1]
    dz = size("PhysicalSizeZ")
    names = re.findall(r'<Channel [^>]*Name="([^"]+)"', ome)
    if len(names) != stack.shape[1]:
        names = [f"ch{i}" for i in range(stack.shape[1])]
    return stack, dx, dz, names


@app.function(hide_code=True)
def project_stack(stack, mode, z_slice, z_range):
    """[C, Y, X] image from a [Z, C, Y, X] stack: one slice, or the max or sum over z in z_range."""
    z0, z1 = z_range
    if mode == "slice":
        return stack[z_slice].astype(np.float32)
    if mode == "max":
        return stack[z0:z1].max(axis=0).astype(np.float32)
    if mode == "sum":
        return stack[z0:z1].sum(axis=0, dtype=np.float32)
    raise ValueError(f"Unknown projection {mode!r}")


@app.function(hide_code=True)
def normalise(image, lo, hi, gamma=1.0):
    """Scale channel-first [C, ...] data to [0, 1] between per-channel limits."""
    shape = (-1,) + (1,) * (image.ndim - 1)
    scaled = (image - lo.reshape(shape)) / np.maximum(hi - lo, 1e-6).reshape(shape)
    return np.clip(scaled, 0.0, 1.0) ** gamma


@app.function(hide_code=True)
def composite(channels, colours):
    """RGB image [..., 3] on black from normalised channels [C, ...], one colour (or None) per channel."""
    rgb = np.zeros(channels.shape[1:] + (3,))
    for channel, colour in zip(channels, colours):
        if colour is not None:
            rgb += channel[..., None] * np.asarray(colour)
    return np.clip(rgb, 0.0, 1.0)


@app.function(hide_code=True)
def training_data_config(projection, downsample):
    """Typed data config of a projection training run: the base config plus the sweep's data settings."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.micropatterns.config_helpers import REPOSITORY_ROOT, measurement_count

    base = OmegaConf.load(REPOSITORY_ROOT / "Experiments" / "micropatterns" / "conf" / "base_config.yaml")
    overrides = {"data": {
        "dataset": "micropatterns_260726",
        "downsample": downsample,
        "micropattern": {
            "task": "micropattern_multi_replicate",
            "data_channels": measurement_count(None, base.data.micropattern.excluded_measurements),
            "experiment_groups": ["cell_fate_s1", "cell_fate_s2", "rna_expression", "protein_response"],
            "projection": projection,
            "cleaning": PROJECTION_CLEANING[projection],
        },
    }}
    return load_experiment_config(OmegaConf.merge(base, overrides)).data


@app.function(hide_code=True)
def load_training_projection(data_config, root, condition, group, replicate):
    """One replicate of one stain group, loaded with the same settings as ``config_helpers.load_data``."""
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.micropatterns.config_helpers import _cleaning_for_loader, _excluded_images

    micropattern = data_config.micropattern
    return load_micropattern_260726(
        root,
        conditions=(condition,),
        projection=micropattern.projection,
        timesteps=tuple(micropattern.timesteps),
        downsample=data_config.downsample,
        replicate_count=1,
        replicate_indices=(replicate,),
        pool_copies=1,
        experiment_groups=(group,),
        excluded_measurements=micropattern.excluded_measurements,
        hist_eqs=micropattern.histogram_percentiles,
        intensity_factors=micropattern.intensity_factors,
        cleaning=_cleaning_for_loader(micropattern.cleaning),
        excluded_images=_excluded_images(micropattern),
        cache_dir=cache_dir_from_environment(),
    )


@app.function(hide_code=True)
def locate_image(source_files, root, key):
    """(batch, time, group) index of the image with projection-free ``key`` in [B, T, G] source paths, or None."""
    from Common.dataloader.micropattern_projections import image_key

    root = Path(root).resolve()
    for index in np.ndindex(source_files.shape):
        path = source_files[index]
        if path and image_key(Path(path).resolve().relative_to(root).as_posix()) == key:
            return index
    return None


@app.function(hide_code=True)
def plane_transform(ax, height, shear):
    """Maps the unit square (image coordinates) to a tilted plane at ``height``."""
    sx, sy = shear
    return Affine2D.from_values(1.0, 0.0, sx, sy, 0.0, height) + ax.transData


@app.function(hide_code=True)
def draw_stack(ax, planes, heights, highlight, theme, opacity, column, shear=(0.55, 0.3)):
    """Oblique view of RGBA planes stacked at ``heights``, pierced by an exaggerated column of voxels.

    ``highlight`` is the index of the plane outlined as the slice. ``column`` holds
    the probe position (x, y as fractions of the image), its width (in plane
    widths), the voxel RGB colours [Z, 3], the voxel heights [Z] and the index
    of the slice voxel.
    """
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for k, (rgba, h) in enumerate(zip(planes, heights)):
        transform = plane_transform(ax, h, shear)
        zorder = 10 + 3 * k
        ax.add_patch(Polygon(unit_square, closed=True, transform=transform, facecolor="black",
                             alpha=opacity, edgecolor="none", zorder=zorder))
        ax.imshow(rgba, extent=(0, 1, 0, 1), origin="upper", transform=transform,
                  interpolation="antialiased", zorder=zorder + 1)
        is_slice = k == highlight
        ax.add_patch(Polygon(unit_square, closed=True, transform=transform, fill=False,
                             edgecolor=theme["accent"] if is_slice else theme["muted"],
                             linewidth=1.4 if is_slice else 0.5, zorder=zorder + 2))

    # Each voxel is drawn between the plane below it and the plane above it, so
    # the column passes through the stack. Only the front and right faces show.
    (px, py), width, colours, voxel_heights, voxel_highlight = column
    sx, sy = shear
    u, v = px - width / 2, 1.0 - py - width / 2
    x0, y0 = u + sx * v, sy * v
    side_dx, side_dy = sx * width, sy * width
    cell = voxel_heights[1] - voxel_heights[0] if len(voxel_heights) > 1 else 0.05
    edge = theme["muted"] if len(colours) <= 40 else "none"
    for z, (colour, h) in enumerate(zip(colours, voxel_heights)):
        zorder = 10 + 3 * (np.searchsorted(heights, h, side="right") - 1) + 2.5
        bottom = y0 + h - cell / 2
        front = [(x0, bottom), (x0 + width, bottom), (x0 + width, bottom + cell), (x0, bottom + cell)]
        side = [(x0 + width, bottom), (x0 + width + side_dx, bottom + side_dy),
                (x0 + width + side_dx, bottom + cell + side_dy), (x0 + width, bottom + cell)]
        ax.add_patch(Polygon(front, facecolor=colour, edgecolor=edge, lw=0.3, zorder=zorder))
        ax.add_patch(Polygon(side, facecolor=0.6 * np.asarray(colour), edgecolor=edge, lw=0.3, zorder=zorder))
        if z == voxel_highlight:
            for face in (front, side):
                ax.add_patch(Polygon(face, fill=False, edgecolor=theme["accent"], lw=1.0, zorder=zorder + 0.1))

    # Outline: the three vertical edges, drawn in pieces between planes so that
    # planes above still cover it, plus the bottom edges and the top face
    orders = 10 + 3 * (np.searchsorted(heights, voxel_heights, side="right") - 1) + 2.6
    for order in np.unique(orders):
        hs = voxel_heights[orders == order]
        y_low, y_high = y0 + hs[0] - cell / 2, y0 + hs[-1] + cell / 2
        for x, dy in ((x0, 0.0), (x0 + width, 0.0), (x0 + width + side_dx, side_dy)):
            ax.plot([x, x], [y_low + dy, y_high + dy], color=theme["column"], lw=0.7, zorder=order,
                    solid_capstyle="butt")
    base = y0 + voxel_heights[0] - cell / 2
    ax.plot([x0, x0 + width, x0 + width + side_dx], [base, base, base + side_dy], color=theme["column"], lw=0.7,
            zorder=orders[0])
    top = y0 + voxel_heights[-1] + cell / 2
    top_face = [(x0, top), (x0 + width, top), (x0 + width + side_dx, top + side_dy), (x0 + side_dx, top + side_dy)]
    ax.add_patch(Polygon(top_face, facecolor=0.8 * np.asarray(colours[-1]), edgecolor=theme["column"], lw=0.7,
                         zorder=10 + 3 * len(heights)))

    ax.set_xlim(-0.14, 1 + sx + 0.14)
    ax.set_ylim(-0.14, heights[-1] + sy + 0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    return (x0 + width / 2 + side_dx / 2, top + side_dy)


@app.function(hide_code=True)
def add_length_marker(ax, start, end, label, theme, xycoords="data", offset=(0, -3), rotation=0, ha="center",
                      va="top"):
    """Double-headed arrow from ``start`` to ``end`` with a label at its middle."""
    ax.annotate("", xy=end, xytext=start, xycoords=xycoords, textcoords=xycoords, annotation_clip=False,
                arrowprops=dict(arrowstyle="<->", color=theme["fg"], lw=0.6, shrinkA=0, shrinkB=0))
    middle = tuple((a + b) / 2 for a, b in zip(start, end))
    ax.annotate(label, xy=middle, xycoords=xycoords, xytext=offset, textcoords="offset points", ha=ha, va=va,
                rotation=rotation, rotation_mode="anchor", fontsize=6, color=theme["fg"], annotation_clip=False)


@app.function(hide_code=True)
def add_scale_bar(ax, length_px, label, shape):
    """White scale bar in the lower right of an image (on black) of the given [H, W] shape."""
    height, width = shape
    pad = 0.05 * width
    ax.add_patch(Rectangle((width - pad - length_px, height - pad - 0.012 * height), length_px,
                           0.012 * height, color="white", linewidth=0))
    ax.text(width - pad - length_px / 2, height - pad - 0.03 * height, label, color="white",
            ha="center", va="bottom", fontsize=6)


@app.function(hide_code=True)
def draw_profile(ax, profile, z_um, mode, z_slice_um, colours, names, theme):
    """Intensity against depth at the probe, marking what each projection keeps."""
    for c, colour in enumerate(colours):
        if colour is None:
            continue
        values = profile[:, c]
        ax.plot(values, z_um, color=colour, lw=1.1, label=names[c])
        if mode == "slice":
            ax.plot(np.interp(z_slice_um, z_um, values), z_slice_um, "o", ms=3.5, color=colour,
                    markeredgecolor=theme["fg"], markeredgewidth=0.4)
        elif mode == "max":
            peak = int(np.argmax(values))
            ax.plot(values[peak], z_um[peak], "o", ms=3.5, color=colour, markeredgecolor=theme["fg"],
                    markeredgewidth=0.4)
            ax.axvline(values[peak], color=colour, lw=0.7, ls=":")
        elif mode == "sum":
            ax.fill_betweenx(z_um, 0, values, color=colour, alpha=0.25, linewidth=0)
    if mode == "slice":
        ax.axhline(z_slice_um, color=theme["accent"], lw=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(z_um[0], z_um[-1])
    ax.set_facecolor(theme["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], labelsize=6, length=2)
    ax.set_xticks([])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Schematics

    - 3D confocal stack → 2D slice, max and sum projections
    - Figures are saved to `figures/schematics/`
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
    stain = mo.ui.dropdown(options=list(STAIN_NAMES), value="cell fate s2", label="Channel names")
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
        channel_names = list(STAIN_NAMES[stain.value] or file_channel_names)
    mo.md(f"Stack `{stack.shape}` (Z, C, Y, X), {dx:.2f} µm pixels, {dz:.2f} µm z-step, channels: {', '.join(channel_names)}")
    field_width_um = stack.shape[-1] * dx
    return channel_names, dx, dz, field_width_um, stack


@app.cell(hide_code=True)
def _(channel_names, dz, stack):
    _fallback = ["yellow", "cyan", "magenta", "white"]
    channel_colours = mo.ui.array([
        mo.ui.dropdown(options=list(COLOURS), value=MARKER_COLOURS.get(name, _fallback[i % 4]), label=name)
        for i, name in enumerate(channel_names)
    ])
    channel_shown = mo.ui.array([mo.ui.checkbox(value=True, label="show") for _ in channel_names])
    _depth = stack.shape[0]
    z_range = mo.ui.range_slider(start=0, stop=_depth, step=1, value=[0, _depth], label="z range (projections)",
                                 show_value=True)
    z_slice = mo.ui.slider(start=0, stop=_depth - 1, step=1, value=_depth // 2, label=f"Slice depth z₀ (index, {dz:g} µm steps)",
                          show_value=True, full_width=True)
    crop_size = mo.ui.slider(start=0.2, stop=1.0, step=0.01, value=1.0, label="Crop size", show_value=True)
    crop_x = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Crop centre x", show_value=True)
    crop_y = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Crop centre y", show_value=True)
    probe_x = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Probe x", show_value=True, full_width=True)
    probe_y = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Probe y", show_value=True, full_width=True)
    probe_radius = mo.ui.slider(start=0, stop=10, step=1, value=2, label="Probe radius (px)", show_value=True)
    column_width = mo.ui.slider(start=0.01, stop=0.15, step=0.005, value=0.05, label="Column width",
                                show_value=True)
    mo.vstack([
        mo.md("**Channels**"),
        mo.hstack([mo.vstack([c, s]) for c, s in zip(channel_colours, channel_shown)], justify="start"),
        mo.md("**Depth**"),
        z_range,
        mo.md("**Field of view**"),
        mo.hstack([crop_size, crop_x, crop_y], justify="start"),
        mo.md("**Probe column** (position sliders are above the figure)"),
        mo.hstack([probe_radius, column_width], justify="start"),
    ])
    return (
        channel_colours,
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
    background = mo.ui.dropdown(options=list(THEMES), value="light", label="Page background")
    low_percentile = mo.ui.number(start=0.0, stop=50.0, step=0.1, value=1.0, label="Low percentile")
    high_percentile = mo.ui.number(start=50.0, stop=100.0, step=0.01, value=99.8, label="High percentile")
    gamma = mo.ui.slider(start=0.3, stop=2.0, step=0.05, value=0.8, label="Gamma", show_value=True)
    n_planes = mo.ui.slider(start=3, stop=24, step=1, value=8, label="Planes drawn", show_value=True)
    plane_opacity = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.6, label="Plane opacity", show_value=True)
    stack_height = mo.ui.slider(start=0.4, stop=2.5, step=0.05, value=1.2, label="Stack height", show_value=True)
    z_exaggeration = mo.ui.slider(start=1.0, stop=10.0, step=0.5, value=3.0, label="XZ z exaggeration",
                                  show_value=True)
    scale_bar_um = mo.ui.number(start=10, stop=500, step=10, value=100, label="Scale bar (µm)")
    figure_width = mo.ui.slider(start=5.0, stop=14.0, step=0.25, value=8.5, label="Width (in)", show_value=True)
    figure_height = mo.ui.slider(start=3.0, stop=10.0, step=0.25, value=5.5, label="Height (in)", show_value=True)
    # Any matplotlib colour: a name ("orange") or hex code ("#d62728")
    slice_colour = mo.ui.text(value="#d62728", label="z₀ / probe line colour")
    column_colour = mo.ui.text(value="white", label="Column outline colour")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([background, low_percentile, high_percentile, gamma], justify="start"),
        mo.hstack([n_planes, plane_opacity, stack_height, z_exaggeration], justify="start"),
        mo.hstack([scale_bar_um, figure_width, figure_height], justify="start"),
        mo.hstack([slice_colour, column_colour], justify="start"),
    ])
    return (
        background,
        column_colour,
        figure_height,
        figure_width,
        gamma,
        high_percentile,
        low_percentile,
        n_planes,
        plane_opacity,
        scale_bar_um,
        slice_colour,
        stack_height,
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
                _data_config = training_data_config(_projection, training_downsample.value)
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
    background,
    channel_colours,
    channel_names,
    channel_shown,
    column_colour,
    column_width,
    cropped,
    dx,
    dz,
    figure_height,
    figure_width,
    gamma,
    limits,
    n_planes,
    plane_opacity,
    probe_radius,
    probe_x,
    probe_y,
    projections,
    scale_bar_um,
    slice_colour,
    stack_height,
    training_images,
    z_bounds,
    z_exaggeration,
    z_slice,
):
    _theme = dict(THEMES[background.value])
    # Indicator colours from the text boxes, ignored if they aren't valid colours
    for _key, _text in (("accent", slice_colour.value), ("column", column_colour.value)):
        if is_color_like(_text.strip()):
            _theme[_key] = _text.strip()
    _colours = [COLOURS[c] if shown else None for c, shown in zip(channel_colours.value, channel_shown.value)]
    # Line colours for the profiles: white lines would vanish on a light page
    _line_colours = [
        None if c is None else ((0.4, 0.4, 0.4) if background.value == "light" and min(c) > 0.85 else c)
        for c in _colours
    ]
    _lo, _hi = limits["slice"]
    _z0, _z1 = z_bounds
    _side = cropped.shape[-1]
    _probe_col = int(probe_x.value * (_side - 1))
    _probe_row = int(probe_y.value * (_side - 1))
    _training = training_images is not None

    def _rgba(planes_czyx):
        """Composite and alpha of channel-first data; alpha follows the brightest shown channel."""
        _norm = normalise(planes_czyx, _lo, _hi, gamma.value)
        _rgb = composite(_norm, _colours)
        _shown = [i for i, c in enumerate(_colours) if c is not None]
        _alpha = _norm[_shown].max(axis=0) if _shown else np.zeros(_norm.shape[1:])
        return np.concatenate([_rgb, _alpha[..., None]], axis=-1)

    with plt.rc_context({"font.size": 7, "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
                         "text.color": _theme["fg"], "axes.labelcolor": _theme["fg"]}):
        # Optional last column: the same projections as the training dataloader returns them
        _width = figure_width.value + (1.9 if _training else 0.0)
        schematic = plt.figure(figsize=(_width, figure_height.value), facecolor=_theme["bg"], dpi=300)
        _ratios = [2.8, 0.2, 0.75, 1.0] + ([1.0] if _training else [])
        _grid = schematic.add_gridspec(3, len(_ratios), width_ratios=_ratios, height_ratios=[1, 1, 1],
                                       wspace=0.08, hspace=0.18, left=0.02, right=0.98, top=0.9, bottom=0.05)

        # Oblique stack: evenly spaced planes plus the slice plane
        _ax_stack = schematic.add_subplot(_grid[0:2, 0])
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
        _column_top = draw_stack(_ax_stack, _planes, _heights, _highlight, _theme, plane_opacity.value, _column)
        # Label above the stack, joined to the top of the column
        _stack_top = _heights[-1] + 0.3
        _ax_stack.annotate("single pixel z column", xy=_column_top, xytext=(_column_top[0], _stack_top + 0.1),
                           ha="center", va="bottom", fontsize=6, color=_theme["fg"],
                           arrowprops=dict(arrowstyle="-", color=_theme["muted"], lw=0.6, shrinkA=1, shrinkB=0),
                           zorder=100)
        _ax_stack.set_ylim(top=_stack_top + 0.22)
        _ax_stack.annotate("", xy=(-0.08, _heights[-1]), xytext=(-0.08, _heights[0]),
                           arrowprops=dict(arrowstyle="-|>", color=_theme["fg"], lw=0.8))
        _ax_stack.text(-0.1, (_heights[0] + _heights[-1]) / 2, f"z  ({(_z1 - _z0) * dz:.0f} µm)", rotation=90,
                       ha="right", va="center", color=_theme["fg"])
        # X along the front edge of the bottom plane, Y along its right edge
        _length = f"{_side * dx:.0f} µm"
        add_length_marker(_ax_stack, (0, _heights[0] - 0.05), (1, _heights[0] - 0.05), f"x  {_length}", _theme)
        add_length_marker(_ax_stack, (1.05, _heights[0] - 0.05), (1.6, _heights[0] + 0.25), f"y  {_length}", _theme,
                          offset=(3, -3), rotation=np.degrees(np.arctan2(0.3, 0.55)))
        _ax_stack.text(1.57, _heights[_highlight] + 0.3, r"$z_0$", color=_theme["accent"], ha="left", va="center")
        _ax_stack.set_title("3D confocal stack", loc="left", color=_theme["fg"], fontsize=8)

        # XZ section through the probe row, showing the z range and slice
        _ax_section = schematic.add_subplot(_grid[2, 0])
        _section = cropped[:, :, _probe_row, :].transpose(1, 0, 2).astype(np.float32)
        _ax_section.imshow(composite(normalise(_section, _lo, _hi, gamma.value), _colours),
                           origin="lower", extent=(0, _side * dx, 0, cropped.shape[0] * dz),
                           aspect=z_exaggeration.value, interpolation="antialiased")
        _ax_section.axhspan(_z0 * dz, _z1 * dz, color=_theme["muted"], alpha=0.12, lw=0)
        _ax_section.axhline((z_slice.value + 0.5) * dz, color=_theme["accent"], lw=0.9)
        _ax_section.axvline((_probe_col + 0.5) * dx, color=_theme["column"], lw=0.8)
        add_length_marker(_ax_section, (0, -0.06), (_side * dx, -0.06), f"x  {_side * dx:.0f} µm", _theme,
                          xycoords=("data", "axes fraction"))
        add_length_marker(_ax_section, (-0.02, 0), (-0.02, cropped.shape[0] * dz),
                          f"z  {cropped.shape[0] * dz:.0f} µm", _theme, xycoords=("axes fraction", "data"),
                          offset=(-3, 0), rotation=90, ha="center", va="bottom")
        _ax_section.set_title(f"XZ section (z ×{z_exaggeration.value:g})", loc="left", color=_theme["fg"],
                              fontsize=8)
        _ax_section.axis("off")

        # One row per projection: z-profile at the probe, the 2D result, and
        # optionally the training image. The profile is on the slice's scale (not clipped).
        _profile = (_window_mean - _lo) / np.maximum(_hi - _lo, 1e-6)
        _z_um = (np.arange(_z0, _z1) + 0.5) * dz
        _scale_px = scale_bar_um.value / dx
        for _row, (_mode, (_name, _formula)) in enumerate(PROJECTION_LABELS.items()):
            _ax_profile = schematic.add_subplot(_grid[_row, 2])
            draw_profile(_ax_profile, _profile, _z_um, _mode, (z_slice.value + 0.5) * dz, _line_colours,
                         channel_names, _theme)
            _ax_profile.set_title(_name, loc="left", color=_theme["fg"], fontsize=8)
            if _row == 1:
                _ax_profile.set_ylabel("z (µm)", fontsize=7)
            if _row == 2:
                _ax_profile.set_xlabel("intensity at probe", fontsize=7)

            _ax_image = schematic.add_subplot(_grid[_row, 3])
            _image = normalise(projections[_mode], *limits[_mode], gamma.value)
            _ax_image.imshow(composite(_image, _colours), interpolation="antialiased")
            _ax_image.plot(_probe_col, _probe_row, "s", ms=5, mfc="none", mec=_theme["column"], mew=0.9)
            _ax_image.set_title(_formula, color=_theme["fg"], fontsize=8)
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
                    _ax_train.set_title("processed training data", color=_theme["fg"], fontsize=8)
                    add_scale_bar(_ax_train, scale_bar_um.value / _train["dx"], f"{scale_bar_um.value:g} µm",
                                  _train["image"].shape[-2:])
                _arrows.append(((1.04, 0.5), _ax_image, (-0.04, 0.5), _ax_train))

            for _start, _start_axes, _end, _end_axes in _arrows:
                schematic.add_artist(ConnectionPatch(
                    xyA=_start, coordsA=_start_axes.transAxes, xyB=_end, coordsB=_end_axes.transAxes,
                    arrowstyle="-|>", color=_theme["muted"], lw=0.9, mutation_scale=9,
                ))

        # Image colours on black, like the images themselves
        _handles = [plt.Line2D([], [], color=c, lw=2) for c in _colours if c is not None]
        _labels = [n for n, c in zip(channel_names, _colours) if c is not None]
        _legend = schematic.legend(_handles, _labels, loc="upper right", ncol=max(1, len(_labels)), fontsize=7,
                                   bbox_to_anchor=(0.98, 1.0), labelcolor="white", frameon=True, fancybox=False,
                                   facecolor="black", edgecolor="black", framealpha=1.0)
    mo.vstack([z_slice, mo.hstack([probe_x, probe_y], widths="equal"), schematic])
    return (schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="projection_schematic", label="File name")
    save_figure = mo.ui.run_button(label="Save PDF + PNG")
    mo.hstack([figure_name, save_figure], justify="start")
    return figure_name, save_figure


@app.cell
def _(figure_name, save_figure, schematic):
    mo.stop(not save_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    for _suffix in ("pdf", "png"):
        schematic.savefig(_directory / f"{figure_name.value}.{_suffix}", dpi=400,
                          facecolor=schematic.get_facecolor())
    mo.md(f"Saved `{figure_name.value}.pdf` and `.png` to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()
