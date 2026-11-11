"""Figures and schematics made from micropattern data.

Run with:
    marimo edit Experiments/micropatterns/schematics.py

The 3D stack path is read from ``MICROPATTERN_3D_STACK`` (e.g. in ``.env``)
or typed into the notebook. The model section reads ``MODEL_STORE_ROOT`` and
``DATA_PATH_BASE``.
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
    from matplotlib.patches import ConnectionPatch, Patch, Polygon, Rectangle, Wedge
    from matplotlib.colors import is_color_like
    from matplotlib.transforms import Affine2D

    # Colony discs with one sector per experiment group, as in the NODAL-KO figures
    from Experiments.micropatterns import nodal_ko_schematic as ko_drawing

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

    # Outline colours of the zoom boxes, visible on the black colonies and a white page
    ZOOM_COLOURS = ("#ff7f0e", "#e41a1c", "#2ca02c", "#1e90ff")

    LEGEND_STYLES = {"White, boxed colours": "white", "Black, like the images": "black"}


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


@app.function(hide_code=True)
def legend_swatch(colour, style):
    """Legend handle of a channel colour: a black-edged box on a white legend, a line on a black one."""
    if style == "white":
        return Patch(facecolor=colour, edgecolor="black", lw=0.6)
    return plt.Line2D([], [], color=colour, lw=2)


@app.function(hide_code=True)
def legend_options(style):
    """``figure.legend`` keyword arguments of a white or black legend style."""
    white = style == "white"
    return {
        "labelcolor": "#222222" if white else "white",
        "frameon": True,
        "fancybox": False,
        "facecolor": "white" if white else "black",
        "edgecolor": "white" if white else "black",
        "framealpha": 1.0,
        "handler_map": {ko_drawing.SectorHandle: (
            ko_drawing.SectorHandler(fill="#666666", edge="#222222") if white else ko_drawing.SectorHandler()
        )},
    }


@app.function(hide_code=True)
def load_selection_records(path, store_root):
    """Models of a selection exported by ``model_registry_explorer.py``, with each model's
    projection and downsample (from its bundle config) and a picker label."""
    import Experiments.micropatterns.nodal_ko_eval as ko_eval
    from Experiments.model_registry import ModelRegistry, load_selection

    records = load_selection(path)
    settings = ko_eval.read_model_settings(
        ModelRegistry(Path(store_root).expanduser()), [r["model_id"] for r in records]
    )
    return [
        {**r, **settings[r["model_id"]], "option_label": ko_eval.model_option_label({**r, **settings[r["model_id"]]})}
        for r in records
    ]


@app.function(hide_code=True)
def baseline_rollout(store_root, model_id, data_root, replicate, seed):
    """Baseline (no KO) rollout of a registry model from the 0h state of ``replicate`` (one-based),
    with that replicate's data loaded and scaled as in the model's training."""
    import jax.random as jr

    import Experiments.micropatterns.nodal_ko_eval as ko_eval
    from Experiments.model_registry import ModelRegistry

    bundle = ModelRegistry(Path(store_root).expanduser()).get(model_id)
    model = bundle.load_model(key=jr.PRNGKey(seed), implementation="portable")
    root = Path(data_root).expanduser()
    bins = ko_eval.load_training_histogram_bins(bundle, root)
    data = ko_eval.load_condition_data(bundle, root, "baseline", replicate, bins)
    prediction, time_labels = ko_eval.evaluate_condition(model, bundle, data, "baseline", seed)
    return {
        "prediction": prediction,
        "target": data["target"],
        "measured": data["measured"],
        "channel_names": data["channel_names"],
        "time_labels": time_labels,
        "boundary": np.asarray(data["boundary"]).squeeze().astype(bool),
        "train_replicates": tuple(bundle.config.data.micropattern.train_replicates or ()),
    }


@app.function(hide_code=True)
def group_colours(names, group):
    """Marker colour of each channel of ``group`` (None: other groups, and LMBR)."""
    return [
        ko_drawing.COLOURS[ko_drawing.MARKER_COLOURS.get(n.rsplit("/", 1)[-1], "grey")]
        if n.split("/")[0] == group else None
        for n in names
    ]


@app.function(hide_code=True)
def trajectory_tile(rollout, source, i, groups, gamma):
    """RGBA disc of time index i with one sector per group, and which sectors were not measured."""
    names = rollout["channel_names"]
    rgbas = [
        ko_drawing.colony_rgba(rollout[source][i], rollout["boundary"], group_colours(names, g), gamma)
        for g in groups
    ]
    unmeasured = [
        source == "target" and not any(f for n, f in zip(names, rollout["measured"][i]) if n.split("/")[0] == g)
        for g in groups
    ]
    return ko_drawing.sliced_rgba(rgbas), unmeasured


@app.function(hide_code=True)
def trajectory_figure(rollouts, blocks, groups, gamma, colony_size, font, hatch, zoom=None, width=7.0,
                      legend_style="white"):
    """Trajectories as rows of colony discs (one sector per group), with a bracketed label per block.

    ``blocks`` are (label, colour, rows), and each row is (rollout key, source,
    row label, row label colour). Positions are in hours of the time axis.
    ``zoom``, if given, adds magnified squares right of the trajectories: a dict
    with ``time`` (index), ``centres`` ((x, y) as fractions of the disc, y down),
    ``width`` (fraction of the disc), ``colours`` (one per centre) and ``title``.
    With ``pixels`` ({rollout key: n}) each box is instead n x n whole pixels of
    that rollout, on its pixel grid. ``legend_style`` is "white" or "black".
    """
    theme = ko_drawing.THEMES["light"]
    first = next(iter(rollouts.values()))
    names = list(first["channel_names"])
    times = [float(t.rstrip("h")) for t in first["time_labels"]]
    s, text = colony_size, font / 7.0
    row_step, block_gap = s + 1.5, 3.0
    placed, y = [], 0.0
    for label, colour, rows in blocks:
        ys = [y - k * row_step for k in range(len(rows))]
        placed.append((label, colour, list(zip(rows, ys))))
        y = ys[-1] - s - block_gap
    label_lines = max(label.count("\n") + 1 for label, _, _ in blocks)
    left = times[0] - s / 2 - 6.0 * text - 1.8 * text * (label_lines - 1) - 0.5
    zoom_x = []
    if zoom is not None:
        zoom_x = [times[-1] + s + 3.0 + k * (s + 1.5) for k in range(len(zoom["centres"]))]
    right = (zoom_x[-1] if zoom_x else times[-1]) + s / 2 + 2.0
    bottom = placed[-1][2][-1][1] - s / 2 - 1.0
    axis_y = s / 2 + 1.5
    top = axis_y + 3.0 * text
    unit = width / (right - left)  # inches per hour
    legend_height = 0.2 * (len(groups) + 1) + 0.15  # inches
    panel_height = unit * (top - bottom)
    hatched = False

    def zoom_box(zx, zy, rgba, key):
        """(left, top, width, height) of a zoom box, as fractions of the disc image."""
        if zoom.get("pixels") is None:
            return zx - zoom["width"] / 2, zy - zoom["width"] / 2, zoom["width"], zoom["width"]
        n = zoom["pixels"][key]
        rows, cols = rgba.shape[:2]
        col = int(np.clip(round(zx * cols - n / 2), 0, max(cols - n, 0)))
        row = int(np.clip(round(zy * rows - n / 2), 0, max(rows - n, 0)))
        return col / cols, row / rows, n / cols, n / rows

    with plt.rc_context({"font.size": font, "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
                         "text.color": theme["fg"]}):
        figure = plt.figure(figsize=(width, panel_height + legend_height), facecolor=theme["bg"], dpi=200)
        ax = figure.add_axes((0.0, legend_height / (panel_height + legend_height), 1.0,
                              panel_height / (panel_height + legend_height)))
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([times[0] - s / 2, times[-1] + s / 2], [axis_y] * 2, color=theme["fg"], lw=0.7)
        for t in times:
            ax.plot([t, t], [axis_y, axis_y + 0.8], color=theme["fg"], lw=0.7)
            ax.text(t, axis_y + 1.2, f"{t:g}h", ha="center", va="bottom")
        if zoom_x:
            ax.plot([zoom_x[0] - s / 2, zoom_x[-1] + s / 2], [axis_y] * 2, color=theme["fg"], lw=0.7)
            ax.text((zoom_x[0] + zoom_x[-1]) / 2, axis_y + 1.2, zoom["title"], ha="center", va="bottom")

        x_row = times[0] - s / 2 - text
        x_bracket = x_row - 2.4 * text
        for label, colour, rows in placed:
            for (key, source, row_label, row_colour), row_y in rows:
                ax.text(x_row, row_y, row_label, rotation=90, ha="right", va="center", fontsize=font - 0.5,
                        color=row_colour)
                rollout = rollouts[key]
                for i, t in enumerate(times[:len(rollout[source])]):
                    centre = (t, row_y)
                    rgba, unmeasured = trajectory_tile(rollout, source, i, groups, gamma)
                    ko_drawing.draw_colony(ax, rgba, centre, s, theme)
                    for k, missing in enumerate(unmeasured):
                        start, stop = ko_drawing.sector_angles(k, len(unmeasured))
                        if missing and hatch:
                            hatched = True
                            ax.add_patch(Wedge(centre, s / 2, start, stop, facecolor="#d9d9d9",
                                               edgecolor="#8c8c8c", hatch="////", lw=0, zorder=4.5))
                        if len(unmeasured) > 1:
                            ax.plot([centre[0], centre[0] + s / 2 * np.cos(np.radians(start))],
                                    [centre[1], centre[1] + s / 2 * np.sin(np.radians(start))],
                                    color="white", lw=0.6, zorder=5)
                    if zoom is None or i != zoom["time"]:
                        continue
                    # Boxes on the disc, and the disc magnified so each box fills its
                    # square, with nearest-pixel drawing so the grid of the data shows
                    for (zx, zy), zoom_colour, x in zip(zoom["centres"], zoom["colours"], zoom_x):
                        fx, fy, fw, fh = zoom_box(zx, zy, rgba, key)
                        ax.add_patch(Rectangle((t + (fx - 0.5) * s, row_y + (0.5 - fy - fh) * s), fw * s, fh * s,
                                               fill=False, edgecolor=zoom_colour, lw=0.8, zorder=6))
                        tile = Rectangle((x - s / 2, row_y - s / 2), s, s, transform=ax.transData)
                        ax.add_patch(Rectangle((x - s / 2, row_y - s / 2), s, s, facecolor="black", lw=0,
                                               zorder=3))
                        image_left = x - s / 2 - fx * s / fw
                        image_top = row_y + s / 2 + fy * s / fh
                        image = ax.imshow(rgba, extent=(image_left, image_left + s / fw, image_top - s / fh,
                                                        image_top),
                                          origin="upper", interpolation="nearest", zorder=4)
                        image.set_clip_path(tile)
                        ax.add_patch(Rectangle((x - s / 2, row_y - s / 2), s, s, fill=False,
                                               edgecolor=zoom_colour, lw=1.0, zorder=6))
            ys = [row_y for _, row_y in rows]
            ax.plot([x_bracket, x_bracket], [ys[-1] - s / 2 + 0.5, ys[0] + s / 2 - 0.5], color=colour, lw=0.7)
            ax.text(x_bracket - 0.6 * text, (ys[0] + ys[-1]) / 2, label, rotation=90, ha="right", va="center",
                    multialignment="center", fontsize=font + 0.5, color=colour)
        # Limits again, since imshow resets them
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        # A legend row per group: its sector, then its channel colours
        legend_rows = [
            [(ko_drawing.SectorHandle(k, len(groups)) if len(groups) > 1 else plt.Line2D([], [], lw=0),
              ko_drawing.GROUP_LABELS.get(g, g))]
            + [(legend_swatch(c, legend_style), n.rsplit("/", 1)[-1])
               for n, c in zip(names, group_colours(names, g)) if c is not None]
            for k, g in enumerate(groups)
        ]
        if hatched:
            legend_rows.append([(Patch(facecolor="#d9d9d9", edgecolor="#8c8c8c",
                                       hatch="////", lw=0), "not measured")])
        length = max(len(r) for r in legend_rows)
        padded = [r + [(plt.Line2D([], [], lw=0), "")] * (length - len(r)) for r in legend_rows]
        # Legend columns fill top to bottom, so entries go in column by column
        entries = [r[c] for c in range(length) for r in padded]
        figure.legend(
            [h for h, _ in entries], [l for _, l in entries], ncol=length, loc="lower center",
            bbox_to_anchor=(0.5, 0.0), fontsize=font - 0.5, handlelength=1.2, columnspacing=0.8,
            **legend_options(legend_style),
        )
    return figure


@app.function(hide_code=True)
def colour_or(text, default):
    """``text`` if it is a matplotlib colour, else ``default``."""
    return text.strip() if is_color_like(text.strip()) else default


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Schematics

    - 3D confocal stack → 2D slice, max and sum projections
    - Baseline NCA trajectories trained on each projection, against their data
    - Baseline NCA trajectories at several resolutions of one projection, with zoomed snapshots
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
    legend_style = mo.ui.dropdown(options=LEGEND_STYLES, value="White, boxed colours", label="Legend style")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([background, low_percentile, high_percentile, gamma], justify="start"),
        mo.hstack([n_planes, plane_opacity, stack_height, z_exaggeration], justify="start"),
        mo.hstack([scale_bar_um, figure_width, figure_height], justify="start"),
        mo.hstack([slice_colour, column_colour, legend_style], justify="start"),
    ])
    return (
        background,
        column_colour,
        figure_height,
        figure_width,
        gamma,
        high_percentile,
        legend_style,
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
    legend_style,
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

        _handles = [legend_swatch(c, legend_style.value) for c in _colours if c is not None]
        _labels = [n for n, c in zip(channel_names, _colours) if c is not None]
        _legend = schematic.legend(_handles, _labels, loc="upper right", ncol=max(1, len(_labels)), fontsize=7,
                                   bbox_to_anchor=(0.98, 1.0), **legend_options(legend_style.value))
    mo.vstack([z_slice, mo.hstack([probe_x, probe_y], widths="equal"), schematic])
    return (schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="projection_schematic", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_figure], justify="start")
    return figure_name, save_figure, save_svg


@app.cell
def _(figure_name, save_figure, schematic, save_svg):
    mo.stop(not save_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    _suffixes = ("pdf", "png", "svg") if save_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            schematic.savefig(_directory / f"{figure_name.value}.{_suffix}", dpi=400,
                              facecolor=schematic.get_facecolor())
    mo.md(f"Saved `{figure_name.value}` as " + ", ".join(_suffixes) + f" to `{_directory}`")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Baseline NCA trajectories per projection

    - One baseline (no KO) model per projection, from a selection exported by `model_registry_explorer.py`
    - Each model starts from the 0h state of the chosen replicate, loaded and scaled as for its training
    - Rows: true data and NCA for each projection; one sector per experiment group
    """)
    return


@app.cell(hide_code=True)
def _():
    projection_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    projection_store_root = mo.ui.text(
        value=os.getenv("MODEL_STORE_ROOT", str(Path(mo.notebook_dir()).parents[1] / "models")),
        label="Model store",
        full_width=True,
    )
    projection_data_root = mo.ui.text(
        value=os.getenv("DATA_PATH_BASE", ""),
        label="Data root (contains 260726_nca_dataset)",
        full_width=True,
    )
    mo.vstack([projection_selection_path, projection_store_root, projection_data_root])
    return projection_data_root, projection_selection_path, projection_store_root


@app.cell(hide_code=True)
def _(projection_selection_path, projection_store_root):
    _path = Path(projection_selection_path.value).expanduser()
    mo.stop(not projection_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    _records = load_selection_records(_path, projection_store_root.value)
    _options = {
        _projection: {_r["option_label"]: _r["model_id"] for _r in _records if _r["projection"] == _projection}
        for _projection in PROJECTION_LABELS
    }
    mo.stop(not any(_options.values()), mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    # One model per projection; projections without a model are left out of the figure
    projection_models = mo.ui.dictionary({
        _projection: mo.ui.dropdown(options=_o, value=next(iter(_o)), label=PROJECTION_LABELS[_projection][0],
                                    full_width=True)
        for _projection, _o in _options.items() if _o
    })
    projection_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    projection_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_projection_rollouts = mo.ui.run_button(label="Roll out models")
    mo.vstack([
        projection_models,
        mo.hstack([projection_replicate, projection_seed, run_projection_rollouts], justify="start"),
    ])
    return (
        projection_models,
        projection_replicate,
        projection_seed,
        run_projection_rollouts,
    )


@app.cell(hide_code=True)
def _(
    projection_data_root,
    projection_models,
    projection_replicate,
    projection_seed,
    projection_store_root,
    run_projection_rollouts,
):
    mo.stop(not run_projection_rollouts.value, mo.md("Choose the models, then roll them out."))
    mo.stop(not projection_data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
    _replicate = int(projection_replicate.value)
    projection_rollouts = {}
    with mo.status.progress_bar(total=len(projection_models.value), title="Loading data and rolling out") as _bar:
        for _index, (_projection, _model_id) in enumerate(projection_models.value.items()):
            projection_rollouts[_projection] = baseline_rollout(
                projection_store_root.value, _model_id, projection_data_root.value, _replicate,
                int(projection_seed.value) + _index,
            )
            _bar.update()
    _in_training = [_p for _p, _r in projection_rollouts.items() if _replicate in _r["train_replicates"]]
    mo.md(
        f"Replicate {_replicate} is in the training split of: {', '.join(_in_training)}"
        if _in_training else f"Replicate {_replicate} is held out from every model's training split."
    )
    return (projection_rollouts,)


@app.cell(hide_code=True)
def _(projection_rollouts):
    _names = next(iter(projection_rollouts.values()))["channel_names"]
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n))
    projection_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    projection_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    projection_row_order = mo.ui.dropdown(
        options={"True and NCA per projection": "projection", "True data, then NCA": "source"},
        value="True and NCA per projection",
        label="Rows",
    )
    projection_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    projection_font_size = mo.ui.slider(4.0, 12.0, step=0.5, value=7.0, show_value=True, label="Font size (pt)")
    projection_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    # Any matplotlib colour; invalid text keeps the default
    projection_true_colour = mo.ui.text(value="#4646c8", label="True data label colour")
    projection_model_colour = mo.ui.text(value="#d97706", label="NCA label colour")
    projection_legend_style = mo.ui.dropdown(options=LEGEND_STYLES, value="White, boxed colours",
                                             label="Legend style")
    mo.vstack([
        mo.hstack([projection_groups, projection_gamma, projection_row_order], justify="start"),
        mo.hstack([projection_colony_size, projection_font_size, projection_hatch], justify="start"),
        mo.hstack([projection_true_colour, projection_model_colour, projection_legend_style], justify="start"),
    ])
    return (
        projection_colony_size,
        projection_font_size,
        projection_gamma,
        projection_groups,
        projection_hatch,
        projection_legend_style,
        projection_model_colour,
        projection_row_order,
        projection_true_colour,
    )


@app.cell(hide_code=True)
def _(
    projection_colony_size,
    projection_font_size,
    projection_gamma,
    projection_groups,
    projection_hatch,
    projection_legend_style,
    projection_model_colour,
    projection_rollouts,
    projection_row_order,
    projection_true_colour,
):
    mo.stop(not projection_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(projection_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in projection_groups.value]
    _fg = ko_drawing.THEMES["light"]["fg"]
    _source_names = {"target": "True data", "prediction": "NCA"}
    _source_colours = {
        "target": colour_or(projection_true_colour.value, "#4646c8"),
        "prediction": colour_or(projection_model_colour.value, "#d97706"),
    }
    # Blocks of rows, each with a bracketed label: (label, colour, rows), and a
    # row is (projection, source, row label, row label colour).
    _projections = [_p for _p in PROJECTION_LABELS if _p in projection_rollouts]
    if projection_row_order.value == "projection":
        _blocks = [
            (f"{PROJECTION_LABELS[_p][0]}\n{PROJECTION_LABELS[_p][1]}", _fg,
             [(_p, _src, _source_names[_src], _source_colours[_src]) for _src in _source_names])
            for _p in _projections
        ]
    else:
        _blocks = [
            (_source_names[_src], _source_colours[_src],
             [(_p, _src, PROJECTION_LABELS[_p][0], _fg) for _p in _projections])
            for _src in _source_names
        ]
    projection_trajectories = trajectory_figure(
        projection_rollouts, _blocks, _groups, projection_gamma.value, projection_colony_size.value,
        projection_font_size.value, projection_hatch.value, legend_style=projection_legend_style.value,
    )
    projection_trajectories
    return (projection_trajectories,)


@app.cell(hide_code=True)
def _():
    projection_figure_name = mo.ui.text(value="projection_trajectories", label="File name")
    save_projection_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_projection_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([projection_figure_name, save_projection_svg, save_projection_figure], justify="start")
    return projection_figure_name, save_projection_figure, save_projection_svg


@app.cell
def _(projection_figure_name, projection_trajectories, save_projection_figure, save_projection_svg):
    mo.stop(not save_projection_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    _suffixes = ("pdf", "png", "svg") if save_projection_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            projection_trajectories.savefig(_directory / f"{projection_figure_name.value}.{_suffix}", dpi=400,
                                            facecolor=projection_trajectories.get_facecolor())
    mo.md(f"Saved `{projection_figure_name.value}` as " + ", ".join(_suffixes) + f" to `{_directory}`")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NCA at different resolutions

    - Baseline models of one projection at several downsampling ratios, from an exported selection
    - Model store and data root are set in the section above
    - Zoom squares magnify the same part of each colony at one timepoint, drawn pixel by pixel
    """)
    return


@app.cell(hide_code=True)
def _():
    resolution_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    resolution_selection_path
    return (resolution_selection_path,)


@app.cell(hide_code=True)
def _(projection_store_root, resolution_selection_path):
    _path = Path(resolution_selection_path.value).expanduser()
    mo.stop(not resolution_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    resolution_records = load_selection_records(_path, projection_store_root.value)
    _present = [_p for _p in PROJECTION_LABELS if any(_r["projection"] == _p for _r in resolution_records)]
    mo.stop(not _present, mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    resolution_projection = mo.ui.dropdown(
        options={PROJECTION_LABELS[_p][0]: _p for _p in _present},
        value=PROJECTION_LABELS[_present[0]][0],
        label="Projection",
    )
    resolution_projection
    return resolution_projection, resolution_records


@app.cell(hide_code=True)
def _(resolution_projection, resolution_records):
    _records = [_r for _r in resolution_records if _r["projection"] == resolution_projection.value]
    # One model per downsampling ratio, finest first
    _options = {}
    for _downsample in sorted({_r["downsample"] for _r in _records}):
        _options[str(_downsample)] = {
            _r["option_label"]: _r["model_id"] for _r in _records if _r["downsample"] == _downsample
        }
    resolution_models = mo.ui.dictionary({
        _downsample: mo.ui.dropdown(options=_o, value=next(iter(_o)), label=f"Δ = {_downsample}",
                                    full_width=True)
        for _downsample, _o in _options.items()
    })
    resolution_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    resolution_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_resolution_rollouts = mo.ui.run_button(label="Roll out models")
    mo.vstack([
        resolution_models,
        mo.hstack([resolution_replicate, resolution_seed, run_resolution_rollouts], justify="start"),
    ])
    return (
        resolution_models,
        resolution_replicate,
        resolution_seed,
        run_resolution_rollouts,
    )


@app.cell(hide_code=True)
def _(
    projection_data_root,
    projection_store_root,
    resolution_models,
    resolution_replicate,
    resolution_seed,
    run_resolution_rollouts,
):
    mo.stop(not run_resolution_rollouts.value, mo.md("Choose the models, then roll them out."))
    mo.stop(not projection_data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
    _replicate = int(resolution_replicate.value)
    resolution_rollouts = {}
    with mo.status.progress_bar(total=len(resolution_models.value), title="Loading data and rolling out") as _bar:
        for _index, (_downsample, _model_id) in enumerate(resolution_models.value.items()):
            resolution_rollouts[int(_downsample)] = baseline_rollout(
                projection_store_root.value, _model_id, projection_data_root.value, _replicate,
                int(resolution_seed.value) + _index,
            )
            _bar.update()
    _in_training = [str(_d) for _d, _r in resolution_rollouts.items() if _replicate in _r["train_replicates"]]
    mo.md(
        f"Replicate {_replicate} is in the training split of downsample: {', '.join(_in_training)}"
        if _in_training else f"Replicate {_replicate} is held out from every model's training split."
    )
    return (resolution_rollouts,)


@app.cell(hide_code=True)
def _(resolution_rollouts):
    _first = next(iter(resolution_rollouts.values()))
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _first["channel_names"] if "/" in _n))
    resolution_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    resolution_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    resolution_row_order = mo.ui.dropdown(
        options={"True and NCA per resolution": "resolution", "True data, then NCA": "source"},
        value="True and NCA per resolution",
        label="Rows",
    )
    resolution_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    resolution_font_size = mo.ui.slider(4.0, 12.0, step=0.5, value=7.0, show_value=True, label="Font size (pt)")
    resolution_width = mo.ui.slider(5.0, 14.0, step=0.25, value=9.0, show_value=True, label="Width (in)")
    resolution_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    # Any matplotlib colour; invalid text keeps the default
    resolution_true_colour = mo.ui.text(value="#4646c8", label="True data label colour")
    resolution_model_colour = mo.ui.text(value="#d97706", label="NCA label colour")
    resolution_zoom_time = mo.ui.dropdown(
        options={_t: _i for _i, _t in enumerate(_first["time_labels"])},
        value=_first["time_labels"][-1],
        label="Zoom timepoint",
    )
    resolution_zoom_count = mo.ui.slider(0, len(ZOOM_COLOURS), step=1, value=2, show_value=True, label="Zooms")
    resolution_zoom_width = mo.ui.slider(0.05, 0.5, step=0.01, value=0.2, show_value=True,
                                         label="Zoom box (fraction of colony)")
    # Whole-pixel boxes: N x N pixels at the coarsest resolution, 2N x 2N at the next, ...
    resolution_zoom_lock = mo.ui.checkbox(value=True, label="Lock zoom boxes to whole pixels")
    resolution_zoom_pixels = mo.ui.slider(1, 32, step=1, value=6, show_value=True,
                                          label="Pixels per box side, coarsest resolution")
    resolution_legend_style = mo.ui.dropdown(options=LEGEND_STYLES, value="White, boxed colours",
                                             label="Legend style")
    mo.vstack([
        mo.hstack([resolution_groups, resolution_gamma, resolution_row_order], justify="start"),
        mo.hstack([resolution_colony_size, resolution_font_size, resolution_width, resolution_hatch],
                  justify="start"),
        mo.hstack([resolution_true_colour, resolution_model_colour, resolution_legend_style], justify="start"),
        mo.hstack([resolution_zoom_time, resolution_zoom_count], justify="start"),
        mo.hstack([resolution_zoom_lock, resolution_zoom_pixels, resolution_zoom_width], justify="start"),
    ])
    return (
        resolution_colony_size,
        resolution_font_size,
        resolution_gamma,
        resolution_groups,
        resolution_hatch,
        resolution_legend_style,
        resolution_model_colour,
        resolution_row_order,
        resolution_true_colour,
        resolution_width,
        resolution_zoom_count,
        resolution_zoom_lock,
        resolution_zoom_pixels,
        resolution_zoom_time,
        resolution_zoom_width,
    )


@app.cell(hide_code=True)
def _(resolution_zoom_count):
    # Zoom centres as fractions of the colony (x right, y down)
    _defaults = [(0.5, 0.5), (0.5, 0.12), (0.2, 0.5), (0.8, 0.5)]
    resolution_zoom_centres = mo.ui.array([
        mo.ui.dictionary({
            "x": mo.ui.slider(0.0, 1.0, step=0.01, value=_x, show_value=True, label="x"),
            "y": mo.ui.slider(0.0, 1.0, step=0.01, value=_y, show_value=True, label="y"),
        })
        for _x, _y in _defaults[:resolution_zoom_count.value]
    ])
    mo.hstack([
        mo.vstack([mo.md(f"<span style='color:{_c}'>■</span> Zoom {_k + 1}"), _centre])
        for _k, (_c, _centre) in enumerate(zip(ZOOM_COLOURS, resolution_zoom_centres))
    ], justify="start")
    return (resolution_zoom_centres,)


@app.cell(hide_code=True)
def _(
    resolution_colony_size,
    resolution_font_size,
    resolution_gamma,
    resolution_groups,
    resolution_hatch,
    resolution_legend_style,
    resolution_model_colour,
    resolution_rollouts,
    resolution_row_order,
    resolution_true_colour,
    resolution_width,
    resolution_zoom_centres,
    resolution_zoom_lock,
    resolution_zoom_pixels,
    resolution_zoom_time,
    resolution_zoom_width,
):
    mo.stop(not resolution_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(resolution_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in resolution_groups.value]
    _fg = ko_drawing.THEMES["light"]["fg"]
    _source_names = {"target": "True data", "prediction": "NCA"}
    _source_colours = {
        "target": colour_or(resolution_true_colour.value, "#4646c8"),
        "prediction": colour_or(resolution_model_colour.value, "#d97706"),
    }

    # Blocks of rows as in the projection figure, keyed by downsampling ratio
    _downsamples = sorted(resolution_rollouts)
    if resolution_row_order.value == "resolution":
        _blocks = [
            (f"Δ = {_d}", _fg,
             [(_d, _src, _source_names[_src], _source_colours[_src]) for _src in _source_names])
            for _d in _downsamples
        ]
    else:
        _blocks = [
            (_source_names[_src], _source_colours[_src],
             [(_d, _src, f"Δ = {_d}", _fg) for _d in _downsamples])
            for _src in _source_names
        ]
    _zoom = None
    if resolution_zoom_centres.value:
        _time_label = next(iter(resolution_rollouts.values()))["time_labels"][resolution_zoom_time.value]
        _zoom = {
            "time": resolution_zoom_time.value,
            "centres": [(_c["x"], _c["y"]) for _c in resolution_zoom_centres.value],
            "width": resolution_zoom_width.value,
            "colours": ZOOM_COLOURS,
            "title": f"Zoom ×{1 / resolution_zoom_width.value:.1f} at {_time_label}",
        }
        if resolution_zoom_lock.value:
            # Downsampling ratios are powers of 2, so each finer one has a whole multiple of the pixels
            _coarsest = max(_downsamples)
            _n = resolution_zoom_pixels.value
            _zoom["pixels"] = {_d: _n * _coarsest // _d for _d in _downsamples}
            _zoom["title"] = f"Zoom at {_time_label}, {_n}×{_n} px at Δ = {_coarsest}"
    resolution_trajectories = trajectory_figure(
        resolution_rollouts, _blocks, _groups, resolution_gamma.value, resolution_colony_size.value,
        resolution_font_size.value, resolution_hatch.value, zoom=_zoom, width=resolution_width.value,
        legend_style=resolution_legend_style.value,
    )
    resolution_trajectories
    return (resolution_trajectories,)


@app.cell(hide_code=True)
def _():
    resolution_figure_name = mo.ui.text(value="resolution_trajectories", label="File name")
    save_resolution_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_resolution_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([resolution_figure_name, save_resolution_svg, save_resolution_figure], justify="start")
    return resolution_figure_name, save_resolution_figure, save_resolution_svg


@app.cell
def _(resolution_figure_name, resolution_trajectories, save_resolution_figure, save_resolution_svg):
    mo.stop(not save_resolution_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    _suffixes = ("pdf", "png", "svg") if save_resolution_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            resolution_trajectories.savefig(_directory / f"{resolution_figure_name.value}.{_suffix}", dpi=400,
                                            facecolor=resolution_trajectories.get_facecolor())
    mo.md(f"Saved `{resolution_figure_name.value}` as " + ", ".join(_suffixes) + f" to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()
