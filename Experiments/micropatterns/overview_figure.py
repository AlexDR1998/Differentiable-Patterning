"""Overview figure of the micropattern NCA results: the NCA idea, the data,
baseline results and NODAL knockouts, in one rearrangeable figure.

Run with:
    marimo edit Experiments/micropatterns/overview_figure.py

- Rollouts (one model, three conditions) are cheap and are cached in
  ``figures/overview/cache/``.
- The knockout-time curve comes from the fine KO timing sweep of
  ``nodal_ko_eval.py`` ("Save for overview figure"), which writes to
  ``figures/overview/``. Until then the panel shows a marked placeholder.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="overview figure")

with app.setup:
    import os
    import pickle
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from matplotlib.colors import to_rgb
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

    from Experiments.micropatterns import nodal_ko_schematic as ko_drawing
    from Experiments.micropatterns.schematics import (
        colour_or,
        group_colours,
        legend_options,
        legend_swatch,
        load_selection_records,
        trajectory_tile,
    )

    load_dotenv()

    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    FIGURE_DIRECTORY = REPOSITORY_ROOT / "figures" / "overview"
    CACHE_DIRECTORY = FIGURE_DIRECTORY / "cache"

    # Written by nodal_ko_eval.py (fine KO timing sweep, "Save for overview figure"); the
    # CSVs can also be the ones from its download buttons
    KO_SAMPLES_CSV = FIGURE_DIRECTORY / "nodal_ko_prevalence_samples.csv"
    KO_SUMMARY_CSV = FIGURE_DIRECTORY / "nodal_ko_prevalence_summary.csv"
    KO_TRUE_CSV = FIGURE_DIRECTORY / "nodal_ko_true_prevalence.csv"
    KO_MAPS_NPZ = FIGURE_DIRECTORY / "nodal_ko_cell_maps.npz"
    KO_PNG = FIGURE_DIRECTORY / "nodal_ko_prevalence.png"

    # Knockout hour of each condition (None: no knockout), in display order
    CONDITIONS = {"baseline": None, "ko_24h": 24, "ko_0h": 0}
    CONDITION_LABELS = {"baseline": "Control", "ko_24h": "NODAL KO\nfrom 24h", "ko_0h": "NODAL KO\nfrom 0h"}
    TIMEPOINTS = (0, 12, 24, 36, 48)

    # Default cell-type colours, as in nodal_ko_eval.py (changeable in the notebook)
    CELL_TYPE_COLOURS = {
        "Notochord": "#66b399",
        "Endoderm": "#e6e64d",
        "Mesoderm": "#ff0000",
        "Several": "#0000ff",
        "Other": "#808080",
    }


@app.function(hide_code=True)
def rollout_cache_path(model_id, replicate, seed):
    """Cache file of the rollouts of one model, replicate and seed."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(model_id))
    return CACHE_DIRECTORY / f"rollouts_{safe}_r{int(replicate)}_s{int(seed)}.pkl"


@app.function(hide_code=True)
def compute_rollouts(store_root, model_id, data_root, replicate, seed):
    """Data and NCA trajectories of one model for every condition, from the 0h
    state of ``replicate`` (one-based), loaded and scaled as in training."""
    import jax.random as jr

    import Experiments.micropatterns.nodal_ko_eval as ko_eval
    from Experiments.model_registry import ModelRegistry

    bundle = ModelRegistry(Path(store_root).expanduser()).get(model_id)
    model = bundle.load_model(key=jr.PRNGKey(seed), implementation="portable")
    root = Path(data_root).expanduser()
    bins = ko_eval.load_training_histogram_bins(bundle, root)
    rollouts = {}
    for k, condition in enumerate(CONDITIONS):
        data = ko_eval.load_condition_data(bundle, root, condition, replicate, bins)
        prediction, time_labels = ko_eval.evaluate_condition(model, bundle, data, condition, seed + k)
        rollouts[condition] = {
            "prediction": np.asarray(prediction),
            "target": np.asarray(data["target"]),
            "measured": np.asarray(data["measured"]),
            "channel_names": tuple(data["channel_names"]),
            "time_labels": tuple(time_labels),
            "boundary": np.asarray(data["boundary"]).squeeze().astype(bool),
            "train_replicates": tuple(bundle.config.data.micropattern.train_replicates or ()),
        }
    return rollouts


@app.function(hide_code=True)
def cached_rollouts(store_root, model_id, data_root, replicate, seed, refresh=False):
    """``compute_rollouts``, read from (or written to) the cache."""
    path = rollout_cache_path(model_id, replicate, seed)
    if path.is_file() and not refresh:
        with open(path, "rb") as file:
            return pickle.load(file)
    rollouts = compute_rollouts(store_root, model_id, data_root, replicate, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(rollouts, file)
    return rollouts


@app.function(hide_code=True)
def read_model_prevalence(path):
    """Model prevalences from a CSV of nodal_ko_eval.py: ``(samples, None)`` for the
    per-sample export (one row per model run and KO time, a ``<cell type>_prevalence``
    column per cell type), ``(None, summary)`` for the grouped summary."""
    table = pd.read_csv(path)
    if "mean_prevalence" in table.columns:
        return None, table
    values = [c for c in table.columns if c.endswith("_prevalence")]
    samples = table.melt(id_vars=[c for c in table.columns if c not in values], value_vars=values,
                         var_name="cell_type", value_name="prevalence")
    # Cell types were lower-cased in the column names
    samples["cell_type"] = samples["cell_type"].str.removesuffix("_prevalence").str.capitalize()
    return samples.rename(columns={"requested_knockout_hour": "knockout_hour"}), None


@app.function(hide_code=True)
def load_measured_data(data_root, replicate, projection, downsample, refresh=False):
    """Measured control trajectory of one replicate (one-based) at ``downsample`` (1: full
    resolution), with the standard preprocessing of conf/base_config.yaml, read from (or
    written to) the cache. Same keys as a rollout, without the prediction."""
    path = CACHE_DIRECTORY / f"data_{projection}_ds{int(downsample)}_r{int(replicate)}.pkl"
    if path.is_file() and not refresh:
        with open(path, "rb") as file:
            return pickle.load(file)
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.micropatterns.config_helpers import _cleaning_for_loader, _excluded_images

    micropattern = ko_drawing.knockout_data_config(ko_drawing.EXPERIMENT_GROUPS, projection, int(downsample)).micropattern
    dataset = load_micropattern_260726(
        Path(data_root).expanduser() / "260726_nca_dataset",
        conditions=("ctrl",),
        projection=micropattern.projection,
        timesteps=TIMEPOINTS,
        downsample=int(downsample),
        replicate_indices=(int(replicate) - 1,),
        experiment_groups=tuple(micropattern.experiment_groups),
        excluded_measurements=micropattern.excluded_measurements,
        hist_eqs=micropattern.histogram_percentiles,
        intensity_factors=micropattern.intensity_factors,
        cleaning=_cleaning_for_loader(micropattern.cleaning),
        excluded_images=_excluded_images(micropattern),
        cache_dir=cache_dir_from_environment(),
    )
    data = {
        "target": np.asarray(dataset.data[0]),
        "measured": np.asarray(dataset.measurement_mask[0]),
        "channel_names": tuple(dataset.channel_names),
        "time_labels": tuple(f"{t}h" for t in TIMEPOINTS),
        "boundary": np.asarray(dataset.boundary_mask[0, 0]).astype(bool),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)
    return data


@app.function(hide_code=True)
def load_ko_sources(model_csv, true_csv):
    """Files for the knockout-time panel (None where missing): model and true-data
    prevalence CSVs, and the cell-type maps and PNG saved to figures/overview/."""
    model_csv, true_csv = Path(model_csv).expanduser(), Path(true_csv).expanduser()
    samples, summary = read_model_prevalence(model_csv) if model_csv.is_file() else (None, None)
    maps = None
    if KO_MAPS_NPZ.is_file():
        with np.load(KO_MAPS_NPZ) as file:
            maps = {key: file[key] for key in file.files}
    return {
        "samples": samples,
        "summary": summary,
        "true": pd.read_csv(true_csv) if true_csv.is_file() else None,
        "maps": maps,
        "png": KO_PNG if KO_PNG.is_file() else None,
    }


@app.function(hide_code=True)
def summarise_prevalence(samples, uncertainty):
    """Mean prevalence per curve group, cell type and KO time, with a band:
    standard deviation, standard error, 95% percentile interval, or none."""
    rows = []
    for (group, cell_type, hour), values in samples.groupby(["curve_group", "cell_type", "knockout_hour"]):
        v = values["prevalence"].to_numpy()
        mean = float(v.mean())
        lower = upper = mean
        if len(v) > 1 and uncertainty in ("std", "sem"):
            error = float(v.std(ddof=1)) / (np.sqrt(len(v)) if uncertainty == "sem" else 1.0)
            lower, upper = mean - error, mean + error
        elif len(v) > 1 and uncertainty == "percentile":
            lower, upper = np.percentile(v, [2.5, 97.5])
        rows.append({"group": group, "cell_type": cell_type, "knockout_hour": float(hour), "mean_prevalence": mean,
                     "lower": max(0.0, float(lower)), "upper": min(1.0, float(upper)), "sample_count": len(v)})
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def hours_of(rollout):
    """Hours of the timepoints of a rollout."""
    return [float(t.rstrip("h")) for t in rollout["time_labels"]]


@app.function(hide_code=True)
def draw_sector_disc(ax, rollout, source, i, groups, centre, size, style, edge=None):
    """Colony disc of time index i, one sector per experiment group (hatched where not measured).
    Returns the RGBA image drawn."""
    rgba, unmeasured = trajectory_tile(rollout, source, i, groups, style["gamma"])
    ko_drawing.draw_colony(ax, rgba, centre, size, style["theme"], edge=edge)
    for k, missing in enumerate(unmeasured):
        start, stop = ko_drawing.sector_angles(k, len(groups))
        if missing and style["hatch"]:
            ax.add_patch(Wedge(centre, size / 2, start, stop, facecolor="#d9d9d9", edgecolor="#8c8c8c",
                               hatch="////", lw=0, zorder=4.5))
        if len(groups) > 1:
            ax.plot([centre[0], centre[0] + size / 2 * np.cos(np.radians(start))],
                    [centre[1], centre[1] + size / 2 * np.sin(np.radians(start))],
                    color="white", lw=0.6, zorder=5)
    return rgba


@app.function(hide_code=True)
def add_marker_legend(ax, rollout, style, in_layout, **kwargs):
    """The one marker legend of the figure: each group's name, then its marker colours.

    ``ax`` is an axes or the whole figure. Groups drawn as sectors in the NCA, baseline
    and KO panels lead with their sector. ``in_layout`` False: the legend cannot push
    other panels aside (it may overlap them)."""
    names = rollout["channel_names"]
    sectors = style["groups"]
    order = list(ko_drawing.EXPERIMENT_GROUPS)
    shown = [g for g in order if g in style["hero_groups"] or g in sectors]
    blank = plt.Line2D([], [], lw=0)
    blocks = [
        [(ko_drawing.SectorHandle(sectors.index(g), len(sectors)) if g in sectors and len(sectors) > 1 else blank,
          ko_drawing.GROUP_LABELS.get(g, g))]
        + [(legend_swatch(c, "white"), n.rsplit("/", 1)[-1])
           for n, c in zip(names, group_colours(names, g)) if c is not None]
        for g in shown
    ]
    if style["legend_layout"] == "rows":
        # One row per group; legend columns fill top to bottom
        length = max(len(b) for b in blocks)
        padded = [b + [(blank, "")] * (length - len(b)) for b in blocks]
        entries, ncol = [b[c] for c in range(length) for b in padded], length
    else:
        # Groups stacked into this many columns, optionally a blank line between groups
        count = len(blocks) if style["legend_layout"] == "groups" else int(style["legend_layout"])
        per = int(np.ceil(len(blocks) / count))
        columns = []
        for i in range(0, len(blocks), per):
            column = []
            for block in blocks[i:i + per]:
                column += ([(blank, "")] if column and style["legend_gaps"] else []) + block
            columns.append(column)
        length = max(len(c) for c in columns)
        entries = [e for c in columns for e in c + [(blank, "")] * (length - len(c))]
        ncol = len(columns)
    options = legend_options("white")
    if style["legend_border"]:
        options["edgecolor"] = "#9a9a9a"
    legend = ax.legend([h for h, _ in entries], [l for _, l in entries], ncol=ncol,
                       fontsize=style["legend_font"], handlelength=1.2, columnspacing=1.0,
                       **options, **kwargs)
    legend.get_frame().set_linewidth(0.8)
    legend.set_in_layout(in_layout)
    headers = {ko_drawing.GROUP_LABELS.get(g, g) for g in shown}
    for text in legend.get_texts():
        if text.get_text() in headers:
            text.set_fontweight("bold")


@app.function(hide_code=True)
def draw_legend_panel(ax, rollout, style):
    """The marker legend as a panel of its own."""
    ax.axis("off")
    if rollout is None:
        ax.text(0.5, 0.5, "marker legend", ha="center", va="center", transform=ax.transAxes,
                color=style["theme"]["muted"])
        return
    # Left out of the layout, so a legend larger than its slot cannot squeeze the other panels
    add_marker_legend(ax, rollout, style, False, loc=style["legend_loc"])


@app.function(hide_code=True)
def draw_arrow(ax, start, end, style, **kwargs):
    """Plain arrow from ``start`` to ``end`` (data units)."""
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=style["theme"]["fg"], lw=0.8,
                                                          shrinkA=0, shrinkB=0, **kwargs))


@app.function(hide_code=True)
def draw_nca_panel(ax, rollout, style):
    """NCA idea: a colony, one cell's neighbourhood, its perception, the shared
    update rule and the change of state, repeated in every cell."""
    theme = style["theme"]
    fg, muted, accent = theme["fg"], theme["muted"], theme["accent"]
    font, small = style["font"], style["font"] - 1
    label_y = 1.05

    # Colony, with the zoomed pixels boxed
    disc_centre, disc_size = (1.25, 2.4), 2.4
    n = style["zoom_pixels"]
    if rollout is None:
        ko_drawing.draw_placeholder(ax, disc_centre, disc_size, theme, "NCA")
        patch = np.tile([0.45, 0.45, 0.45, 1.0], (n, n, 1))
        box = (disc_centre[0] + 0.2, disc_centre[1] - 0.3, 0.25, 0.25)
    else:
        rgba = draw_sector_disc(ax, rollout, "prediction", style["nca_time"], style["groups"], disc_centre,
                                disc_size, style)
        rows, cols = rgba.shape[:2]
        zx, zy = style["zoom_centre"]
        r0 = int(np.clip(round(zy * rows) - n // 2, 0, rows - n))
        c0 = int(np.clip(round(zx * cols) - n // 2, 0, cols - n))
        patch = rgba[r0:r0 + n, c0:c0 + n]
        box = (disc_centre[0] - disc_size / 2 + c0 / cols * disc_size,
               disc_centre[1] + disc_size / 2 - (r0 + n) / rows * disc_size,
               n / cols * disc_size, n / rows * disc_size)
    ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor=accent, lw=0.8, zorder=7))
    ax.text(disc_centre[0], label_y, "colony state\n$X_n$", ha="center", va="top", fontsize=font, linespacing=1.3)

    # Neighbourhood: the zoomed pixels, with channel planes behind
    gx, gy, gsize = 3.05, 1.45, 1.9
    cell = gsize / n
    for k, shade in ((2, "#d0d0d0"), (1, "#a8a8a8")):
        ax.add_patch(Rectangle((gx + 0.11 * k, gy + 0.11 * k), gsize, gsize, facecolor=shade, edgecolor=muted,
                               lw=0.4, zorder=1))
    for i in range(n):
        for j in range(n):
            colour = patch[i, j, :3] if patch[i, j, 3] > 0.5 else "#efefef"
            ax.add_patch(Rectangle((gx + j * cell, gy + gsize - (i + 1) * cell), cell, cell, facecolor=colour,
                                   edgecolor="#555555", lw=0.3, zorder=3))
    m = n // 2
    ax.add_patch(Rectangle((gx + (m - 1) * cell, gy + gsize - (m + 2) * cell), 3 * cell, 3 * cell, fill=False,
                           edgecolor=accent, lw=1.2, zorder=4))
    ax.add_patch(Rectangle((gx + m * cell, gy + gsize - (m + 1) * cell), cell, cell, fill=False,
                           edgecolor="white", lw=1.0, zorder=4))
    for y_box, y_grid in ((box[1] + box[3], gy + gsize), (box[1], gy)):
        ax.plot([box[0] + box[2], gx], [y_box, y_grid], color=accent, lw=0.5, alpha=0.8, zorder=2)
    ax.text(gx + gsize / 2, label_y, "neighbourhood\n" + style["nca_state_label"], ha="center", va="top",
            fontsize=font, linespacing=1.3)

    # Perception kernels, applied to every channel
    kx, ksize = 6.6, 0.6
    kernels = {
        r"$I$": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        r"$\nabla^2$": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        r"$\nabla$": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    }
    cmap = plt.get_cmap("RdBu_r")
    for (name, kernel), ky in zip(kernels.items(), (3.35, 2.4, 1.45)):
        scaled = kernel / np.abs(kernel).max()
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((kx - ksize / 2 + j * ksize / 3, ky + ksize / 2 - (i + 1) * ksize / 3),
                                       ksize / 3, ksize / 3, facecolor=cmap(0.5 + 0.5 * scaled[i, j]),
                                       edgecolor="#555555", lw=0.3, zorder=3))
        ax.text(kx + ksize / 2 + 0.08, ky, name, ha="left", va="center", fontsize=small)
    draw_arrow(ax, (gx + gsize + 0.3, 2.4), (kx - ksize / 2 - 0.1, 2.4), style)
    ax.text(kx, label_y, "perceive", ha="center", va="top", fontsize=font)

    # Update rule: one small network shared by every cell
    layers = ((8.0, 6), (8.8, 8), (9.6, 6))
    nodes = [[(x, y) for y in np.linspace(2.4 - 0.13 * (count - 1), 2.4 + 0.13 * (count - 1), count)]
             for x, count in layers]
    for left, right in zip(nodes[:-1], nodes[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=muted, lw=0.25, alpha=0.6, zorder=2)
    for k, layer in enumerate(nodes):
        for xy in layer:
            ax.add_patch(Circle(xy, 0.075, facecolor="white" if k == 0 else style["nca_colour"],
                                edgecolor=style["nca_colour"], lw=0.6, zorder=3))
    draw_arrow(ax, (kx + ksize / 2 + 0.4, 2.4), (layers[0][0] - 0.15, 2.4), style)
    ax.text(layers[1][0], label_y, "update rule\n(shared)", ha="center", va="top", fontsize=font, linespacing=1.3)

    # Change of state: one value per channel
    dx, count = 10.6, 9
    heights = np.linspace(2.4 + 0.12 * (count - 1), 2.4 - 0.12 * (count - 1), count)
    values = np.random.default_rng(1).uniform(-1, 1, count)
    for y, value in zip(heights, values):
        ax.add_patch(Rectangle((dx - 0.13, y - 0.11), 0.26, 0.22, facecolor=cmap(0.5 + 0.45 * value),
                               edgecolor="#555555", lw=0.3, zorder=3))
    draw_arrow(ax, (layers[-1][0] + 0.15, 2.4), (dx - 0.25, 2.4), style)
    ax.text(dx, label_y, "change\n$\\Delta X_n$", ha="center", va="top", fontsize=font, linespacing=1.3)

    # The change is added back to the cell, in every cell, at every step
    centre_top = (gx + (m + 0.5) * cell, gy + gsize - m * cell)
    ax.add_patch(FancyArrowPatch((dx, heights[0] + 0.15), centre_top, connectionstyle="arc3,rad=0.33",
                                 arrowstyle="-|>", mutation_scale=7, color=accent, lw=1.0, zorder=8))
    middle = (dx + centre_top[0]) / 2
    ax.text(middle, 4.55, style["nca_loop_note"], ha="center", va="bottom", fontsize=small, color=accent)
    ax.text(middle, 4.95, style["nca_equation"], ha="center", va="bottom", fontsize=font + 4, color=fg)

    ax.set_xlim(-0.1, 11.2)
    ax.set_ylim(0.35, 5.75)
    ax.set_aspect("equal")
    ax.axis("off")


@app.function(hide_code=True)
def group_image(data, group, i, style):
    """RGBA of the channels of one group at time index i, cropped to the colony, or None
    where that group was not measured."""
    names = data["channel_names"]
    in_group = [n.split("/")[0] == group for n in names]
    if not np.asarray(data["measured"][i])[in_group].any():
        return None
    return ko_drawing.colony_rgba(data["target"][i], data["boundary"], group_colours(names, group), style["gamma"])


@app.function(hide_code=True)
def resize_rgba(rgba, shape):
    """``rgba`` [H, W, 4] scaled (bilinear) to ``shape``."""
    from scipy.ndimage import zoom

    if rgba.shape[:2] == tuple(shape):
        return rgba
    scaled = zoom(rgba, (shape[0] / rgba.shape[0], shape[1] / rgba.shape[1], 1), order=1, mode="nearest",
                  grid_mode=True)[:shape[0], :shape[1]]
    missing = (shape[0] - scaled.shape[0], shape[1] - scaled.shape[1])
    return np.clip(np.pad(scaled, ((0, missing[0]), (0, missing[1]), (0, 0)), mode="edge"), 0.0, 1.0)


@app.function(hide_code=True)
def draw_data_panel(ax, rollout, style, model_data=None):
    """Micropattern data: a grid of colonies, one column per measurement group and one
    row per time, and the colonies of one row combined into one disc, one sector per group.

    Groups in ``style["data_model_groups"]`` are taken from ``model_data`` (the images
    the model was trained on) instead of ``rollout``."""
    theme = style["theme"]
    fg, muted, accent = theme["fg"], theme["muted"], theme["accent"]
    font, small = style["font"], style["font"] - 1
    groups = style["hero_groups"]
    chosen = style["hero_time"]
    labelled = style["data_group_labels"] or style["data_markers"]

    def source(group):
        if model_data is not None and group in style["data_model_groups"]:
            return model_data
        return rollout

    # Grid of colonies: column = measurement group, row = time
    step, well = 0.72, 0.6
    rows = len(TIMEPOINTS)
    px, py = 0.55, 0.95
    pw, ph = len(groups) * step + 0.2, rows * step + 0.2
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.2", facecolor="#f2f2f2",
                                edgecolor=muted, lw=0.6, zorder=1))

    def centre(r, c):
        return px + 0.1 + step / 2 + c * step, py + ph - 0.1 - step / 2 - r * step

    for r, hour in enumerate(TIMEPOINTS):
        ax.text(px - 0.08, centre(r, 0)[1], f"{hour}h", ha="right", va="center", fontsize=small)
        for c, group in enumerate(groups):
            xy = centre(r, c)
            rgba = group_image(source(group), group, r, style) if source(group) is not None else None
            if rgba is None:
                ax.add_patch(Circle(xy, well / 2, facecolor="white", edgecolor=muted, lw=0.5, ls=(0, (2, 2)),
                                    zorder=2))
                continue
            # Grid colonies stay small, whatever the data resolution
            shrink = max(1, max(rgba.shape[:2]) // 150)
            ko_drawing.draw_colony(ax, rgba[::shrink, ::shrink], xy, well, theme,
                                   alpha=1.0 if r == chosen else style["data_dim"])
    for c, group in enumerate(groups):
        x, y = centre(0, c)
        ax.text(x - 0.1, y + well / 2 + 0.12, ko_drawing.GROUP_LABELS.get(group, group), rotation=40,
                ha="left", va="bottom", fontsize=small, rotation_mode="anchor")
    ax.text(px + pw / 2, py - 0.15, style["data_caption"], ha="center", va="top", fontsize=small, linespacing=1.3)

    # The chosen row, all groups, widening into the combined colony
    hero_size = 3.4
    hero_centre = (px + pw + 0.7 + hero_size / 2 + (2.1 if labelled else 0.0), py + ph / 2)
    row_y = centre(chosen, 0)[1]
    half = well / 2 + 0.07
    left, right = px + 0.05, px + pw - 0.05
    ax.add_patch(FancyBboxPatch((left, row_y - half), right - left, 2 * half,
                                boxstyle="round,pad=0,rounding_size=0.1", fill=False, edgecolor=accent, lw=1.2,
                                zorder=6))
    radius = hero_size / 2
    upper = (hero_centre[0] + radius * np.cos(np.radians(115)), hero_centre[1] + radius * np.sin(np.radians(115)))
    lower = (hero_centre[0] + radius * np.cos(np.radians(245)), hero_centre[1] + radius * np.sin(np.radians(245)))
    ax.add_patch(plt.Polygon([(right, row_y + half), upper, lower, (right, row_y - half)], closed=True,
                             facecolor=accent, alpha=0.08, edgecolor="none", zorder=0))
    for a, b in (((right, row_y + half), upper), ((right, row_y - half), lower)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=accent, lw=0.6, alpha=0.7, zorder=0)

    # One colony, one sector per staining, each group at the finest resolution shown
    if rollout is None:
        ko_drawing.draw_placeholder(ax, hero_centre, hero_size, theme, "data")
        names, time_label = (), ""
    else:
        images = [group_image(source(g), g, chosen, style) for g in groups]
        shown = [i for i in images if i is not None]
        if shown:
            shape = max((i.shape[:2] for i in shown), key=lambda hw: hw[0] * hw[1])
            reference = next(i for i in shown if i.shape[:2] == shape)
            # Unmeasured groups: a black sector
            blank = np.concatenate([np.zeros(shape + (3,)), reference[..., 3:]], axis=-1)
            images = [blank if i is None else resize_rgba(i, shape) for i in images]
            ko_drawing.draw_colony(ax, ko_drawing.sliced_rgba(images), hero_centre, hero_size, theme, edge=fg)
            for k in range(len(groups) if len(groups) > 1 else 0):
                start, _ = ko_drawing.sector_angles(k, len(groups))
                ax.plot([hero_centre[0], hero_centre[0] + radius * np.cos(np.radians(start))],
                        [hero_centre[1], hero_centre[1] + radius * np.sin(np.radians(start))],
                        color="white", lw=0.6, zorder=5)
        else:
            ko_drawing.draw_placeholder(ax, hero_centre, hero_size, theme, "–")
        names = rollout["channel_names"]
        time_label = rollout["time_labels"][chosen]
    ax.text(hero_centre[0], hero_centre[1] - hero_size / 2 - 0.12, time_label, ha="center", va="top", fontsize=font)

    # Marker names beside the disc, level with their sector
    line = 0.27
    backing = dict(facecolor=theme["bg"], edgecolor="none", pad=0.4)
    for k, group in enumerate(groups if labelled else []):
        start, stop = ko_drawing.sector_angles(k, len(groups))
        middle = np.radians((start + stop) / 2)
        right = np.cos(middle) >= 0
        sign = 1 if right else -1
        x = hero_centre[0] + sign * (hero_size / 2 + 0.2)
        markers = [(n.rsplit("/", 1)[-1], c) for n, c in zip(names, group_colours(names, group))
                   if c is not None and style["data_markers"]]
        height = line * len(markers)
        top = np.clip(hero_centre[1] + hero_size / 2 * np.sin(middle) + height / 2,
                      hero_centre[1] - hero_size / 2 + height, hero_centre[1] + hero_size / 2)
        ax.text(x, top, ko_drawing.GROUP_LABELS.get(group, group), ha="left" if right else "right", va="center",
                fontsize=font, fontweight="bold", bbox=backing, zorder=8)
        for row, (marker, colour) in enumerate(markers, start=1):
            ax.plot(x + sign * 0.1, top - row * line, "s", ms=4, color=colour, mec="#333333", mew=0.4, zorder=9)
            ax.text(x + sign * 0.25, top - row * line, marker, ha="left" if right else "right", va="center",
                    fontsize=small, bbox=backing, zorder=8)

    # The marker legend, beside or below this panel
    if rollout is not None and style["legend_place"] == "right":
        add_marker_legend(ax, rollout, style, True, loc="center left", bbox_to_anchor=(1.0, 0.5))
    elif rollout is not None and style["legend_place"] == "below":
        add_marker_legend(ax, rollout, style, True, loc="upper center", bbox_to_anchor=(0.5, 0.0))

    ax.set_xlim(-0.1, hero_centre[0] + hero_size / 2 + (2.1 if labelled else 0.2))
    ax.set_ylim(0.05, py + ph + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


@app.function(hide_code=True)
def draw_time_axis(ax, times, y, size, style):
    """Time axis above a row of discs (x in hours)."""
    fg = style["theme"]["fg"]
    ax.plot([times[0] - size / 2, times[-1] + size / 2], [y, y], color=fg, lw=0.7)
    for t in times:
        ax.plot([t, t], [y, y + 0.8], color=fg, lw=0.7)
        ax.text(t, y + 1.2, f"{t:g}h", ha="center", va="bottom")


@app.function(hide_code=True)
def draw_baseline_panel(ax, rollout, style):
    """Baseline trajectories: measured data and NCA, one sector per group, over time."""
    s = style["colony_size"]
    groups = style["groups"]
    times = hours_of(rollout) if rollout is not None else list(TIMEPOINTS)
    rows = (("target", style["data_label"], style["data_colour"]),
            ("prediction", style["nca_label"], style["nca_colour"]))
    step = s + 1.5
    for r, (source, label, colour) in enumerate(rows):
        y = -r * step
        ax.text(times[0] - s / 2 - 1.0, y, label, rotation=90, ha="right", va="center", color=colour,
                fontweight="bold")
        for i, t in enumerate(times):
            if rollout is None:
                ko_drawing.draw_placeholder(ax, (t, y), s, style["theme"], "")
            else:
                draw_sector_disc(ax, rollout, source, i, groups, (t, y), s, style)
    axis_y = s / 2 + 1.5
    draw_time_axis(ax, times, axis_y, s, style)
    ax.set_xlim(times[0] - s / 2 - 4.5, times[-1] + s / 2 + 0.5)
    ax.set_ylim(-step - s / 2 - 0.5, axis_y + 4.0)
    ax.set_aspect("equal")
    ax.axis("off")


@app.function(hide_code=True)
def draw_ko_outcomes_panel(ax, rollouts, style):
    """48h patterns of the control and both measured knockouts, data and NCA,
    each under a bar showing when NODAL is inhibited."""
    theme = style["theme"]
    fg, muted, accent = theme["fg"], theme["muted"], theme["accent"]
    s, gap = style["colony_size"], 2.0
    groups = style["groups"]
    rows = (("target", style["data_label"], style["data_colour"]),
            ("prediction", style["nca_label"], style["nca_colour"]))
    step = s + 1.5
    end = TIMEPOINTS[-1]
    bar_y = s / 2 + 1.4
    for c, (condition, hour) in enumerate(CONDITIONS.items()):
        x = c * (s + gap)
        # Inhibition bar: 0h to 48h, red from the knockout time
        ax.plot([x - s / 2, x + s / 2], [bar_y, bar_y], color=muted, lw=1.5, solid_capstyle="butt")
        if hour is not None:
            ax.plot([x - s / 2 + s * hour / end, x + s / 2], [bar_y, bar_y], color=accent, lw=3.0,
                    solid_capstyle="butt")
            ax.plot([x - s / 2 + s * hour / end] * 2, [bar_y - 0.7, bar_y + 0.7], color=accent, lw=0.8)
        if c == 0:
            for t, ha in ((0, "left"), (end, "right")):
                ax.text(x - s / 2 + s * t / end, bar_y - 0.8, f"{t}h", ha=ha, va="top", fontsize=style["font"] - 1.5,
                        color=muted)
        ax.text(x, bar_y + 1.2, CONDITION_LABELS[condition], ha="center", va="bottom", linespacing=1.2,
                color=fg if hour is None else accent)
        for r, (source, label, colour) in enumerate(rows):
            y = -r * step
            if rollouts is None:
                ko_drawing.draw_placeholder(ax, (x, y), s, theme, "")
            else:
                rollout = rollouts[condition]
                draw_sector_disc(ax, rollout, source, len(rollout["time_labels"]) - 1, groups, (x, y), s, style)
    for r, (_, label, colour) in enumerate(rows):
        ax.text(-s / 2 - 1.0, -r * step, label, rotation=90, ha="right", va="center", color=colour,
                fontweight="bold")
    ax.set_xlim(-s / 2 - 4.5, 2 * (s + gap) + s / 2 + 0.5)
    ax.set_ylim(-step - s / 2 - 0.5, bar_y + 6.5)
    ax.set_aspect("equal")
    ax.axis("off")


@app.function(hide_code=True)
def cell_type_rgba(shares, cell_types, mask, shown, palette):
    """RGBA of per-pixel cell-type shares [K, H, W] on black, cropped to the colony."""
    rgb = np.zeros(mask.shape + (3,))
    for share, cell_type in zip(np.asarray(shares, dtype=np.float32), cell_types):
        if cell_type in shown:
            rgb += share[..., None] * np.asarray(to_rgb(palette.get(str(cell_type), "#777777")))
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    return np.concatenate([np.clip(rgb[crop], 0.0, 1.0), mask[crop][..., None].astype(float)], axis=-1)


@app.function(hide_code=True)
def draw_png_panel(ax, path, style):
    """A saved image (path absolute, or relative to figures/overview/)."""
    path = Path(path)
    path = path if path.is_absolute() else FIGURE_DIRECTORY / path
    if path.is_file():
        ax.imshow(plt.imread(path), interpolation="antialiased")
    else:
        ax.text(0.5, 0.5, f"missing image\n{path.name}", ha="center", va="center", transform=ax.transAxes,
                color=style["theme"]["muted"])
    ax.axis("off")


@app.function(hide_code=True)
def draw_ko_curve_panel(ax, sources, style):
    """Cell-type prevalence at 48h against NODAL knockout time: NCA curves (mean
    and band over models, replicates and seeds) and measured knockouts.

    Returns the extra space (points) above the axes taken by cell-type maps."""
    theme = style["theme"]
    fg, muted = theme["fg"], theme["muted"]
    mode = style["ko_source"]
    if mode == "auto":
        mode = "csv" if sources["summary"] is not None else "placeholder"
    if mode == "png":
        draw_png_panel(ax, sources["png"] or KO_PNG, style)
        return 0.0
    shown = style["ko_cell_types"]
    palette = style["cell_type_colours"]
    end = float(TIMEPOINTS[-1])
    ax.set_facecolor("white")

    dashes = ("-", "--", ":", "-.")
    groups = []
    if mode == "csv" and sources["summary"] is not None:
        summary = sources["summary"]
        groups = [g for g in style["ko_groups"] if g in set(summary["group"])]
        samples = sources.get("samples")
        # One run of one model from one initial condition
        run_columns = [c for c in ("model_realisation_id", "initial_condition_replicate", "random_seed")
                       if samples is not None and c in samples.columns]
        for k, group in enumerate(groups):
            curves = summary[summary["group"] == group]
            for cell_type in shown:
                curve = curves[curves["cell_type"] == cell_type].sort_values("knockout_hour")
                if curve.empty:
                    continue
                colour = palette.get(cell_type, "#444444")
                ax.plot(curve["knockout_hour"], curve["mean_prevalence"], color=colour, lw=1.6,
                        ls=dashes[k % len(dashes)], zorder=3)
                if style["ko_uncertainty"] == "samples" and run_columns:
                    runs = samples[(samples["curve_group"] == group) & (samples["cell_type"] == cell_type)]
                    for _, run in runs.groupby(run_columns, dropna=False):
                        run = run.sort_values("knockout_hour")
                        ax.plot(run["knockout_hour"], run["prevalence"], color=colour, lw=0.5, alpha=0.3,
                                ls=dashes[k % len(dashes)], zorder=2)
                elif (curve["sample_count"] > 1).any() and (curve["upper"] > curve["lower"]).any():
                    ax.fill_between(curve["knockout_hour"], curve["lower"], curve["upper"], color=colour, alpha=0.2,
                                    lw=0)
        true = sources["true"]
        if true is not None and not true.empty:
            if style["ko_projection"] != "all":
                true = true[true["projection"] == style["ko_projection"]]
            for (cell_type, hour), values in true[true["cell_type"].isin(shown)].groupby(["cell_type",
                                                                                           "knockout_hour"]):
                v = values["relative_prevalence"].to_numpy()
                colour = palette.get(cell_type, "#444444")
                if style["ko_true_display"] == "replicates":
                    # Each replicate, spread a little along x so they do not hide each other
                    offsets = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.6
                    ax.scatter(hour + offsets, v, s=14, color=colour, edgecolor=fg, linewidth=0.5, zorder=5)
                else:
                    ax.errorbar(hour, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else None, fmt="o", ms=4,
                                mfc=colour, mec=fg, mew=0.6, ecolor=colour, elinewidth=0.8, capsize=1.5, zorder=5)
    else:
        # Made-up curves, only to lay out the figure
        hours = np.linspace(0, end, 49)
        made_up = {
            "Notochord": 0.05 + 0.3 / (1 + np.exp(-(hours - 18) / 4)),
            "Endoderm": 0.4 - 0.25 / (1 + np.exp(-(hours - 14) / 5)),
            "Mesoderm": 0.15 + 0.1 * np.sin(hours / end * np.pi),
        }
        for k, cell_type in enumerate(shown):
            colour = palette.get(cell_type, "#444444")
            curve = made_up.get(cell_type, 0.05 + 0.02 * k + 0 * hours)
            ax.plot(hours, curve, color=colour, lw=1.6, alpha=0.5)
            ax.fill_between(hours, curve - 0.04, curve + 0.04, color=colour, alpha=0.08, lw=0)
        ax.text(0.5, 0.5, "PLACEHOLDER\nrun the fine KO sweep in nodal_ko_eval.py,\nthen 'Save for overview figure'",
                transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=style["font"],
                zorder=6, bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.9, boxstyle="round,pad=0.4"))

    if style["ko_legend"]:
        handles = [plt.Line2D([], [], color=palette.get(c, "#444444"), lw=1.6, label=c) for c in shown]
        handles += [plt.Line2D([], [], color=muted, lw=1.6, label=style["nca_label"]),
                    plt.Line2D([], [], color=muted, marker="o", ls="", mec=fg, mew=0.6, ms=4,
                               label=style["data_label"])]
        if len(groups) > 1:
            handles += [plt.Line2D([], [], color=fg, lw=1.2, ls=dashes[k % len(dashes)], label=str(g))
                        for k, g in enumerate(groups)]
        ax.legend(handles=handles, fontsize=style["font"] - 1, frameon=False, loc=style["ko_legend_loc"],
                  handlelength=1.4)
    ticks = list(TIMEPOINTS)
    ax.set_xticks(ticks, [f"{t}h" for t in ticks[:-1]] + ["no KO"])
    ax.set_xlim(-2.0, end + 2.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("NODAL inhibited from")
    ax.set_ylabel("share of colony at 48h")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2)

    # Cell-type maps above the axes, at their knockout times
    hours_shown = style["ko_map_hours"]
    size = style["ko_map_size"]
    if not hours_shown:
        return 0.0
    maps = sources["maps"] if mode == "csv" else None
    for hour in hours_shown:
        if maps is not None:
            index = int(np.argmin(np.abs(maps["hours"] - hour)))
            rgba = cell_type_rgba(maps["shares"][index], maps["cell_types"], maps["mask"], shown, palette)
            box = OffsetImage(rgba, zoom=size / max(rgba.shape[:2]), interpolation="antialiased")
            x = float(maps["hours"][index])
        else:
            box = DrawingArea(size, size)
            box.add_artist(Circle((size / 2, size / 2), size / 2 - 0.5, fill=False, edgecolor=muted, lw=0.6,
                                  ls=(0, (2, 2))))
            x = hour
        ax.add_artist(AnnotationBbox(box, (x, 1.0), xycoords=("data", "axes fraction"), box_alignment=(0.5, -0.1),
                                     frameon=False, pad=0, annotation_clip=False))
    return size + 4.0


@app.function(hide_code=True)
def label_panel(ax, letter, title, style, above=0.0):
    """Panel letter and title above the top left corner of ``ax`` (``above``: extra points)."""
    if letter:
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5 + above), textcoords="offset points",
                    ha="left", va="bottom", fontsize=style["font"] + 3, fontweight="bold")
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(13 if letter else 0, 5 + above),
                    textcoords="offset points", ha="left", va="bottom", fontsize=style["font"] + 0.5)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Overview figure

    - Panels: `nca` (idea), `data`, `baseline`, `ko_outcomes`, `ko_curve`, `legend` (markers), or any PNG path
    - Layout (rows of panels with their widths), panel order and titles: the code cell below
    - Rollouts are cached in `figures/overview/cache/`; the figure is saved to `figures/overview/`
    """)
    return


@app.cell
def _():
    # One entry per row: its panels (one letter each, "." for a gap) and their relative widths.
    # Panel letters a, b, c... follow the order here (the legend gets none).
    ROWS = [
        ("AB", [7.2, 4.6]),
        ("CD", [7.2, 4.6]),
    ]
    PANELS = {
        "A": "nca",
        "B": "data",
        "C": "baseline",
        "D": "ko_curve",
        # Also available: "ko_outcomes" (48h control and KO colonies) and "legend"
        # (marker legend as its own panel; set Legend place to "own panel")
        # Any saved image, relative to figures/overview/, e.g.
        # "F": "nodal_ko_cell_maps_0.png",
    }
    TITLES = {
        "nca": "Neural cellular automaton",
        "data": "Micropatterned colonies",
        "baseline": "Learned dynamics",
        "ko_outcomes": "NODAL knockout",
        "ko_curve": "Predicting untested knockout times",
    }
    HEIGHT_RATIOS = [1.0, 1.25]
    return HEIGHT_RATIOS, PANELS, ROWS, TITLES


@app.cell(hide_code=True)
def _():
    selection_path = mo.ui.text(placeholder="/path/to/model_registry_selection.yaml (optional)",
                                label="Exported registry selection", full_width=True)
    store_root = mo.ui.text(value=os.getenv("MODEL_STORE_ROOT", str(REPOSITORY_ROOT / "models")),
                            label="Model store", full_width=True)
    data_root = mo.ui.text(value=os.getenv("DATA_PATH_BASE", ""), label="Data root (contains 260726_nca_dataset)",
                           full_width=True)
    mo.vstack([mo.md("**Models and data**"), selection_path, store_root, data_root])
    return data_root, selection_path, store_root


@app.cell(hide_code=True)
def _(selection_path, store_root):
    _path = Path(selection_path.value).expanduser()
    _options = {}
    if selection_path.value.strip() and _path.is_file():
        _options = {_r["option_label"]: _r["model_id"]
                    for _r in load_selection_records(_path, store_root.value)}
    # From the selection if there is one, else typed model IDs
    if _options:
        baseline_model = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Baseline model",
                                        full_width=True)
        ko_model = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Knockout model",
                                  full_width=True)
    else:
        baseline_model = mo.ui.text(placeholder="model ID", label="Baseline model", full_width=True)
        ko_model = mo.ui.text(placeholder="model ID (blank: baseline model)", label="Knockout model",
                              full_width=True)
    replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    refresh = mo.ui.checkbox(value=False, label="Ignore cache")
    run_rollouts = mo.ui.run_button(label="Roll out")
    mo.vstack([
        baseline_model,
        ko_model,
        mo.hstack([replicate, seed, refresh, run_rollouts], justify="start"),
        mo.md("- Cached rollouts load without pressing *Roll out*"),
    ])
    return baseline_model, ko_model, refresh, replicate, run_rollouts, seed


@app.cell(hide_code=True)
def _(
    baseline_model,
    data_root,
    ko_model,
    refresh,
    replicate,
    run_rollouts,
    seed,
    store_root,
):
    _baseline_id = (baseline_model.value or "").strip()
    _ko_id = (ko_model.value or "").strip() or _baseline_id
    _models = list(dict.fromkeys(_m for _m in (_baseline_id, _ko_id) if _m))
    _cached = bool(_models) and all(rollout_cache_path(_m, replicate.value, seed.value).is_file() for _m in _models)
    if not _models or not (run_rollouts.value or _cached):
        rollouts = {"baseline": None, "ko": None}
        _status = mo.md("No rollouts yet: panels show placeholders. Choose models, then *Roll out*.")
    else:
        mo.stop(not data_root.value.strip() and not _cached, mo.callout("Set the data root first.", kind="warn"))
        _by_model = {}
        with mo.status.progress_bar(total=len(_models), title="Loading data and rolling out") as _bar:
            for _m in _models:
                _by_model[_m] = cached_rollouts(store_root.value, _m, data_root.value, int(replicate.value),
                                                int(seed.value), refresh=run_rollouts.value and refresh.value)
                _bar.update()
        rollouts = {
            "baseline": _by_model[_baseline_id]["baseline"] if _baseline_id else None,
            "ko": _by_model[_ko_id] if _ko_id else None,
        }
        _held_out = [
            _name for _name, _m in (("baseline", _baseline_id), ("knockout", _ko_id))
            if _m and int(replicate.value) not in _by_model[_m]["baseline"]["train_replicates"]
        ]
        _status = mo.md(
            f"Rollouts ready. Replicate {int(replicate.value)} is held out from the "
            f"{' and '.join(_held_out)} model's training." if _held_out else
            f"Rollouts ready. Replicate {int(replicate.value)} is in the training split."
        )
    _status
    return (rollouts,)


@app.cell(hide_code=True)
def _():
    data_projection = mo.ui.dropdown(options=["max", "sum", "slice"], value="max", label="Data panel projection")
    data_downsample = mo.ui.dropdown(options={"1 (full resolution)": 1, "2": 2, "4": 4},
                                     value="1 (full resolution)", label="Data panel downsample")
    load_data_panel = mo.ui.run_button(label="Load data panel images")
    mo.vstack([
        mo.hstack([data_projection, data_downsample, load_data_panel], justify="start"),
        mo.md("- Data panel: measured control colony of the same replicate, standard preprocessing "
              "(`conf/base_config.yaml`), cached; until loaded it uses the model's (downsampled) data"),
    ])
    return data_downsample, data_projection, load_data_panel


@app.cell(hide_code=True)
def _(data_downsample, data_projection, data_root, load_data_panel, refresh, replicate):
    _path = CACHE_DIRECTORY / f"data_{data_projection.value}_ds{data_downsample.value}_r{int(replicate.value)}.pkl"
    if not (load_data_panel.value or _path.is_file()):
        measured_data = None
        _status = mo.md("Data panel images not loaded (using the model's data).")
    else:
        mo.stop(not data_root.value.strip() and not _path.is_file(),
                mo.callout("Set the data root first.", kind="warn"))
        with mo.status.spinner("Loading and cleaning the data panel images"):
            measured_data = load_measured_data(data_root.value, int(replicate.value), data_projection.value,
                                               data_downsample.value, refresh=load_data_panel.value and refresh.value)
        _status = mo.md(f"Data panel images: {measured_data['boundary'].shape[0]} px across.")
    _status
    return (measured_data,)


@app.cell(hide_code=True)
def _():
    _model_default = KO_SAMPLES_CSV if KO_SAMPLES_CSV.is_file() or not KO_SUMMARY_CSV.is_file() else KO_SUMMARY_CSV
    ko_model_csv = mo.ui.text(value=str(_model_default), label="Model prevalence CSV", full_width=True)
    ko_true_csv = mo.ui.text(value=str(KO_TRUE_CSV), label="True-data prevalence CSV", full_width=True)
    reload_ko_files = mo.ui.button(value=0, on_click=lambda _v: _v + 1, label="Reload KO files")
    mo.vstack([
        mo.md("**Knockout-time data** (from the fine KO timing sweep of `nodal_ko_eval.py`)"),
        ko_model_csv,
        ko_true_csv,
        reload_ko_files,
        mo.md("- Model CSV: the per-sample export (`nodal_ko_cell_type_prevalence.csv` download, or "
              "`nodal_ko_prevalence_samples.csv` saved here) lets grouping and bands change here; "
              "the grouped summary is drawn as exported"),
    ])
    return ko_model_csv, ko_true_csv, reload_ko_files


@app.cell(hide_code=True)
def _(ko_model_csv, ko_true_csv, reload_ko_files):
    reload_ko_files.value
    ko_sources = load_ko_sources(ko_model_csv.value, ko_true_csv.value)
    _samples = ko_sources["samples"]
    if _samples is not None:
        _counts = {_c: _samples[_c].nunique() for _c in ("model_realisation_id", "initial_condition_replicate",
                                                          "random_seed", "knockout_hour") if _c in _samples}
        _model_note = "per-sample: " + ", ".join(f"{_n} {_c.replace('_', ' ')}s" for _c, _n in _counts.items())
    elif ko_sources["summary"] is not None:
        _model_note = "grouped summary (grouping and bands fixed at export)"
    else:
        _model_note = "not found (placeholder)"
    _true = ko_sources["true"]
    mo.md(
        f"- Model prevalence: {_model_note}\n"
        f"- True data: {'not found' if _true is None else f'{len(_true)} rows'}\n"
        f"- Cell-type maps: {'found' if ko_sources['maps'] is not None else 'not found'}"
    )
    return (ko_sources,)


@app.cell(hide_code=True)
def _():
    _times = {f"{_t}h": _i for _i, _t in enumerate(TIMEPOINTS)}
    _groups = list(ko_drawing.EXPERIMENT_GROUPS)
    figure_width = mo.ui.slider(4.0, 16.0, step=0.01, value=13.33, show_value=True, label="Width (in)")
    figure_height = mo.ui.slider(3.0, 12.0, step=0.01, value=7.5, show_value=True, label="Height (in, 16:9 slide: 13.33 x 7.5)")
    font_size = mo.ui.slider(5.0, 18.0, step=0.5, value=11.0, show_value=True, label="Font (pt)")
    gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    colony_size = mo.ui.slider(7.0, 11.5, step=0.5, value=10.5, show_value=True, label="Colony size (h)")
    groups = mo.ui.multiselect(options=_groups, value=["cell_fate_s1", "cell_fate_s2"],
                               label="Sectors (baseline, KO, NCA)")
    hero_groups = mo.ui.multiselect(options=_groups, value=_groups, label="Sectors (data)")
    hero_time = mo.ui.dropdown(options=_times, value="48h", label="Data time")
    nca_time = mo.ui.dropdown(options=_times, value="48h", label="NCA panel time")
    zoom_x = mo.ui.slider(0.0, 1.0, step=0.01, value=0.7, show_value=True, label="Zoom x")
    zoom_y = mo.ui.slider(0.0, 1.0, step=0.01, value=0.35, show_value=True, label="Zoom y")
    zoom_pixels = mo.ui.dropdown(options={"5": 5, "7": 7, "9": 9}, value="7", label="Zoom pixels")
    hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured")
    legend_layout = mo.ui.dropdown(
        options={"one tall column": "1", "two columns": "2", "one column per group": "groups",
                 "one row per group": "rows"},
        value="one tall column", label="Legend")
    legend_place = mo.ui.dropdown(
        options={"right edge of the figure": "figure", "beside the data panel": "right",
                 "below the data panel": "below", "own panel ('legend' in ROWS)": "panel"},
        value="right edge of the figure", label="Legend place")
    # Top right corner of the legend, as fractions of the figure (right edge only)
    legend_x = mo.ui.slider(0.5, 1.0, step=0.005, value=1.0, show_value=True, label="Legend x")
    legend_y = mo.ui.slider(0.0, 1.0, step=0.005, value=0.55, show_value=True, label="Legend top")
    legend_font = mo.ui.slider(5.0, 16.0, step=0.5, value=9.0, show_value=True, label="Legend font (pt)")
    legend_gaps = mo.ui.checkbox(value=False, label="Gaps between legend groups")
    legend_border = mo.ui.checkbox(value=False, label="Legend border")
    legend_loc = mo.ui.dropdown(options=["center", "center left", "upper left", "upper center", "lower left"],
                                value="center left", label="Legend position")
    data_markers = mo.ui.checkbox(value=False, label="Marker names beside data disc")
    data_group_labels = mo.ui.checkbox(value=False, label="Group names beside data disc")
    data_model_groups = mo.ui.multiselect(options=_groups, value=["rna_expression"],
                                          label="Data panel groups from the model's data (as panel c)")
    data_dim = mo.ui.slider(0.2, 1.0, step=0.05, value=0.6, show_value=True, label="Other rows opacity")
    figure_title = mo.ui.text(value="", placeholder="none", label="Figure title", full_width=True)
    title_size = mo.ui.slider(8.0, 28.0, step=0.5, value=16.0, show_value=True, label="Title (pt)")
    title_align = mo.ui.dropdown(options=["left", "center"], value="left", label="Title alignment")
    data_label = mo.ui.text(value="Data", label="Data label")
    data_colour = mo.ui.text(value="#4646c8", label="Data colour")
    nca_label = mo.ui.text(value="NCA", label="NCA label")
    nca_colour = mo.ui.text(value="#d97706", label="NCA colour")
    data_caption = mo.ui.text(value="hESC colonies, imaged 0-48h,\n4 stainings", label="Data caption",
                              full_width=True)
    nca_state_label = mo.ui.text(value="(markers + hidden)", label="NCA state label")
    nca_equation = mo.ui.text(value=r"$X_{n+1} = X_n + \Delta X_n$", label="NCA equation", full_width=True)
    nca_loop_note = mo.ui.text(value="in every cell, every step", label="NCA loop note")
    letters = mo.ui.dropdown(options=["lower", "upper", "none"], value="lower", label="Panel letters")
    show_titles = mo.ui.checkbox(value=True, label="Panel titles")
    anchor = mo.ui.dropdown(options=["NW", "N", "C", "W"], value="NW", label="Panel anchor")
    mo.vstack([
        mo.md("**Figure**"),
        mo.hstack([figure_title, title_size, title_align], justify="start"),
        mo.hstack([figure_width, figure_height, font_size, letters, show_titles, anchor], justify="start"),
        mo.hstack([gamma, colony_size, groups, hero_groups], justify="start"),
        mo.hstack([hero_time, nca_time, zoom_x, zoom_y, zoom_pixels, hatch], justify="start"),
        mo.hstack([legend_place, legend_layout, legend_loc, legend_x, legend_y], justify="start"),
        mo.hstack([legend_font, legend_gaps, legend_border, data_group_labels, data_markers, data_dim],
                  justify="start"),
        data_model_groups,
        mo.hstack([data_label, data_colour, nca_label, nca_colour], justify="start"),
        mo.hstack([nca_state_label, nca_loop_note, nca_equation], justify="start"),
        data_caption,
    ])
    return (
        anchor,
        colony_size,
        data_caption,
        data_colour,
        data_dim,
        data_group_labels,
        data_label,
        data_markers,
        data_model_groups,
        figure_height,
        figure_title,
        figure_width,
        font_size,
        gamma,
        groups,
        hatch,
        hero_groups,
        hero_time,
        legend_border,
        legend_font,
        legend_gaps,
        legend_layout,
        legend_loc,
        legend_place,
        legend_x,
        legend_y,
        letters,
        nca_colour,
        nca_equation,
        nca_label,
        nca_loop_note,
        nca_state_label,
        nca_time,
        show_titles,
        title_align,
        title_size,
        zoom_pixels,
        zoom_x,
        zoom_y,
    )


@app.cell(hide_code=True)
def _(ko_sources):
    _samples, _summary, _true = ko_sources["samples"], ko_sources["summary"], ko_sources["true"]
    _models = _samples if _samples is not None else _summary
    _types = (list(dict.fromkeys(_models["cell_type"])) if _models is not None
              else ["Notochord", "Endoderm", "Mesoderm"])
    # Columns of the per-sample export that can group the model runs into curves
    _group_columns = {"Grouping chosen at export": "group", "Projection": "projection", "Resolution": "downsample",
                      "Each model": "model_realisation_label", "Initial condition": "initial_condition_replicate"}
    _group_options = {_l: _c for _l, _c in _group_columns.items() if _samples is not None and _c in _samples}
    _group_options["All models together"] = "all"
    ko_group_by = mo.ui.dropdown(options=_group_options, value=next(iter(_group_options)), label="Curves per")
    ko_uncertainty = mo.ui.dropdown(
        options={"standard deviation": "std", "standard error": "sem", "95% interval": "percentile",
                 "each run as a thin line": "samples", "none": "none"},
        value="standard deviation", label="Band")
    ko_true_display = mo.ui.dropdown(options={"mean ± sd": "mean", "each replicate": "replicates"},
                                     value="mean ± sd", label="True data")
    _projections = ["all"] + (list(dict.fromkeys(_true["projection"])) if _true is not None else [])
    ko_source = mo.ui.dropdown(
        options={"auto (CSV, else placeholder)": "auto", "saved curves (CSV)": "csv", "saved PNG": "png",
                 "placeholder": "placeholder"},
        value="auto (CSV, else placeholder)", label="Source")
    ko_cell_types = mo.ui.multiselect(options=_types, value=[_t for _t in _types if _t not in ("Other", "Several")],
                                      label="Cell types")
    ko_projection = mo.ui.dropdown(options=_projections, value=_projections[-1], label="Data projection")
    ko_map_hours = mo.ui.text(value="", placeholder="e.g. 0, 24, 48", label="Cell-type maps at KO times (h)")
    ko_map_size = mo.ui.slider(10.0, 40.0, step=1.0, value=22.0, show_value=True, label="Map size (pt)")
    ko_legend = mo.ui.checkbox(value=True, label="Legend")
    ko_legend_loc = mo.ui.dropdown(options=["best", "upper left", "upper right", "center right", "lower right"],
                                   value="best", label="Legend position")
    # Any matplotlib colour; invalid text keeps the default
    cell_type_colours = mo.ui.dictionary({
        _t: mo.ui.text(value=CELL_TYPE_COLOURS.get(_t, "#444444"), label=_t)
        for _t in dict.fromkeys([*_types, *CELL_TYPE_COLOURS])
    })
    mo.vstack([
        mo.md("**Knockout-time curve**"),
        mo.hstack([ko_source, ko_group_by, ko_uncertainty, ko_cell_types], justify="start"),
        mo.hstack([ko_projection, ko_true_display], justify="start"),
        mo.hstack([ko_map_hours, ko_map_size, ko_legend, ko_legend_loc], justify="start"),
        mo.md("Cell-type colours"),
        mo.hstack(list(cell_type_colours.values()), justify="start"),
    ])
    return (
        cell_type_colours,
        ko_cell_types,
        ko_group_by,
        ko_legend,
        ko_legend_loc,
        ko_map_hours,
        ko_map_size,
        ko_projection,
        ko_source,
        ko_true_display,
        ko_uncertainty,
    )


@app.cell(hide_code=True)
def _(ko_group_by, ko_sources, ko_uncertainty):
    # Model curves, regrouped and summarised here when the per-sample export is loaded
    _samples = ko_sources["samples"]
    if _samples is not None:
        _samples = _samples.assign(curve_group=(_samples[ko_group_by.value].astype(str)
                                                if ko_group_by.value != "all" else "All models"))
        ko_curves = {**ko_sources, "samples": _samples,
                     "summary": summarise_prevalence(_samples, ko_uncertainty.value)}
    else:
        ko_curves = ko_sources
    _groups = (list(dict.fromkeys(ko_curves["summary"]["group"])) if ko_curves["summary"] is not None
               else ["(placeholder)"])
    ko_groups = mo.ui.multiselect(options=_groups, value=_groups[:1], label="Curves shown (line style each)")
    ko_groups
    return ko_curves, ko_groups


@app.cell(hide_code=True)
def _(
    cell_type_colours,
    colony_size,
    data_caption,
    data_colour,
    data_dim,
    data_group_labels,
    data_label,
    data_markers,
    data_model_groups,
    font_size,
    gamma,
    groups,
    hatch,
    hero_groups,
    hero_time,
    ko_cell_types,
    ko_groups,
    ko_legend,
    ko_legend_loc,
    ko_map_hours,
    ko_map_size,
    ko_projection,
    ko_source,
    ko_true_display,
    ko_uncertainty,
    legend_border,
    legend_font,
    legend_gaps,
    legend_layout,
    legend_loc,
    legend_place,
    legend_x,
    legend_y,
    nca_colour,
    nca_equation,
    nca_label,
    nca_loop_note,
    nca_state_label,
    nca_time,
    zoom_pixels,
    zoom_x,
    zoom_y,
):
    _order = list(ko_drawing.EXPERIMENT_GROUPS)
    try:
        _map_hours = [float(_h) for _h in ko_map_hours.value.split(",") if _h.strip()]
    except ValueError:
        _map_hours = []
    style = {
        "theme": ko_drawing.THEMES["light"],
        "font": font_size.value,
        "gamma": gamma.value,
        "colony_size": colony_size.value,
        "groups": [_g for _g in _order if _g in groups.value] or _order[:1],
        "hero_groups": [_g for _g in _order if _g in hero_groups.value] or _order[:1],
        "hero_time": hero_time.value,
        "nca_time": nca_time.value,
        "zoom_centre": (zoom_x.value, zoom_y.value),
        "zoom_pixels": zoom_pixels.value,
        "hatch": hatch.value,
        "legend_layout": legend_layout.value,
        "legend_loc": legend_loc.value,
        "legend_place": legend_place.value,
        "legend_anchor": (legend_x.value, legend_y.value),
        "legend_font": legend_font.value,
        "legend_gaps": legend_gaps.value,
        "legend_border": legend_border.value,
        "data_markers": data_markers.value,
        "data_group_labels": data_group_labels.value,
        "data_dim": data_dim.value,
        "data_model_groups": list(data_model_groups.value),
        "data_label": data_label.value,
        "data_colour": colour_or(data_colour.value, "#4646c8"),
        "nca_label": nca_label.value,
        "nca_colour": colour_or(nca_colour.value, "#d97706"),
        "data_caption": data_caption.value.replace("\\n", "\n"),
        "nca_state_label": nca_state_label.value,
        "nca_equation": nca_equation.value,
        "nca_loop_note": nca_loop_note.value,
        "ko_source": ko_source.value,
        "ko_groups": list(ko_groups.value),
        "ko_uncertainty": ko_uncertainty.value,
        "ko_true_display": ko_true_display.value,
        "ko_cell_types": list(ko_cell_types.value),
        "cell_type_colours": {_t: colour_or(_c, CELL_TYPE_COLOURS.get(_t, "#444444"))
                              for _t, _c in cell_type_colours.value.items()},
        "ko_projection": ko_projection.value,
        "ko_map_hours": _map_hours,
        "ko_map_size": ko_map_size.value,
        "ko_legend": ko_legend.value,
        "ko_legend_loc": ko_legend_loc.value,
    }
    return (style,)


@app.cell(hide_code=True)
def _(
    HEIGHT_RATIOS,
    PANELS,
    ROWS,
    TITLES,
    anchor,
    figure_height,
    figure_title,
    figure_width,
    ko_curves,
    letters,
    measured_data,
    rollouts,
    show_titles,
    style,
    title_align,
    title_size,
):
    _drawers = {
        "nca": lambda ax: draw_nca_panel(ax, rollouts["baseline"], style),
        "data": lambda ax: draw_data_panel(ax, measured_data or rollouts["baseline"], style, rollouts["baseline"]),
        "baseline": lambda ax: draw_baseline_panel(ax, rollouts["baseline"], style),
        "ko_outcomes": lambda ax: draw_ko_outcomes_panel(ax, rollouts["ko"], style),
        "ko_curve": lambda ax: draw_ko_curve_panel(ax, ko_curves, style),
        "legend": lambda ax: draw_legend_panel(ax, rollouts["baseline"], style),
    }
    _order = list(dict.fromkeys(_c for _row, _ in ROWS for _c in _row if _c != "."))
    _lettered = [_key for _key in _order if PANELS.get(_key) != "legend"]
    _alphabet = "abcdefghijklmnopqrstuvwxyz"
    with plt.style.context("default"), plt.rc_context({
                         "font.size": style["font"], "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
                         "text.color": style["theme"]["fg"], "axes.labelsize": style["font"],
                         "xtick.labelsize": style["font"] - 1, "ytick.labelsize": style["font"] - 1}):
        overview = plt.figure(figsize=(figure_width.value, figure_height.value), facecolor="white",
                              layout="constrained", dpi=150)
        # One subfigure per row, so each row sets its own widths
        _rows = overview.subfigures(len(ROWS), 1, height_ratios=HEIGHT_RATIOS)
        overview_axes = {}
        for _subfigure, (_row, _widths) in zip(np.atleast_1d(_rows), ROWS):
            _subfigure.set_facecolor("white")
            overview_axes.update(_subfigure.subplot_mosaic([list(_row)], empty_sentinel=".", width_ratios=_widths))
        for _key in _order:
            _ax = overview_axes[_key]
            _ax.set_anchor(anchor.value)
            _panel = PANELS.get(_key, "")
            if _panel in _drawers:
                _above = _drawers[_panel](_ax) or 0.0
            elif _panel.lower().endswith(".png"):
                _above = draw_png_panel(_ax, _panel, style) or 0.0
            else:
                _ax.text(0.5, 0.5, f"unknown panel {_panel!r}", ha="center", va="center", transform=_ax.transAxes)
                _ax.axis("off")
                _above = 0.0
            _letter = ""
            if _key in _lettered:
                _letter = _alphabet[_lettered.index(_key)]
                _letter = {"lower": _letter, "upper": _letter.upper()}.get(letters.value, "")
            label_panel(_ax, _letter, TITLES.get(_panel, "") if show_titles.value else "", style, _above)
        if figure_title.value.strip():
            _left = title_align.value == "left"
            overview.suptitle(figure_title.value.replace("\\n", "\n"), fontsize=title_size.value,
                              fontweight="bold", x=0.01 if _left else 0.5, ha="left" if _left else "center")
        if style["legend_place"] == "figure" and rollouts["baseline"] is not None:
            add_marker_legend(overview, rollouts["baseline"], style, False, loc="upper right",
                              bbox_to_anchor=style["legend_anchor"], labelspacing=0.25)
    overview
    return overview, overview_axes


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="micropattern_overview", label="File name")
    save_panels = mo.ui.checkbox(value=False, label="Also each panel separately")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_panels, save_svg, save_figure], justify="start")
    return figure_name, save_figure, save_panels, save_svg


@app.cell(hide_code=True)
def _(figure_name, overview, overview_axes, save_figure, save_panels, save_svg):
    mo.stop(not save_figure.value)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _saved = []
    _suffixes = ("pdf", "png", "svg") if save_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            overview.savefig(FIGURE_DIRECTORY / f"{figure_name.value}.{_suffix}", dpi=600,
                             facecolor=overview.get_facecolor())
            _saved.append(f"{figure_name.value}.{_suffix}")
        if save_panels.value:
            # Each panel's area of the full figure, so panels match the composite
            overview.canvas.draw()
            _renderer = overview.canvas.get_renderer()
            for _key, _ax in overview_axes.items():
                _box = _ax.get_tightbbox(_renderer).transformed(overview.dpi_scale_trans.inverted()).padded(0.04)
                for _suffix in ("png", "svg") if save_svg.value else ("png",):
                    overview.savefig(FIGURE_DIRECTORY / f"{figure_name.value}_{_key}.{_suffix}", dpi=600,
                                     bbox_inches=_box, facecolor=overview.get_facecolor())
                    _saved.append(f"{figure_name.value}_{_key}.{_suffix}")
    mo.md(f"Saved to `{FIGURE_DIRECTORY}`:\n\n" + "\n".join(f"- `{_n}`" for _n in _saved))
    return


if __name__ == "__main__":
    app.run()
