"""High-level overview schematic of the project: imaging, training an NCA on the
data, and probing signalling perturbations at any time with the model.

Run with:
    marimo edit Experiments/micropatterns/project_overview.py

- Panel a: cartoon microscope and a time series of real colonies, one paper stack
  of replicates per timepoint (only the top sheet is clearly visible)
- Panel b: data stack → training → local pixelwise updates → time series (v1), or
  two spaced 3D slices with an update cone between them (v2); the patterning
  process is a toy local rule, for illustration only
- Panel c: 0h data → final data, joined by the rollout; a perturbation block under
  the 0h image points into it, and a mock curve of one cell type against
  perturbation time links back to the arrow ("find optimal timing")
- The dataset root defaults to ``$DATA_PATH_BASE/260726_nca_dataset``; without it
  the colonies are drawn as grey placeholders
- Figures are saved to ``figures/schematics/``
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="project overview")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

    from Experiments.micropatterns import nodal_ko_schematic as ko_drawing
    from Experiments.micropatterns.schematics import colour_or, group_colours, legend_options, legend_swatch

    load_dotenv()

    TIMEPOINTS = ko_drawing.TIMEPOINTS
    CONDITION_LABELS = {"ctrl": "Control", "sl24": "perturbed at 24h", "sl0": "perturbed at 0h"}
    PROJECTION_LABELS = {"max": "max projection", "sum": "sum projection", "slice": "single slice"}

    # Toy fates of panel b, drawn like the SOX2 / TBXT / SOX17 stainings
    TOY_COLOURS = (ko_drawing.COLOURS["cyan"], ko_drawing.COLOURS["red"], ko_drawing.COLOURS["yellow"])


@app.function(hide_code=True)
def load_overview_data(root, conditions, groups, projection, downsample, replicates):
    """All timepoints of the given conditions and one-based replicates, on one shared
    intensity scale per channel (pooled normalisation, as in nodal_ko_schematic.py).

    Returns images [B, T, C, H, W], measured [B, T, C], masks [B, H, W], the
    (condition, replicate) of each batch entry and the channel names.
    """
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.micropatterns.config_helpers import _cleaning_for_loader, _excluded_images

    micropattern = ko_drawing.knockout_data_config(groups, projection, downsample, "pooled").micropattern
    dataset = load_micropattern_260726(
        root,
        conditions=tuple(conditions),
        projection=micropattern.projection,
        timesteps=TIMEPOINTS,
        downsample=downsample,
        replicate_indices=tuple(r - 1 for r in replicates),
        experiment_groups=tuple(micropattern.experiment_groups),
        excluded_measurements=micropattern.excluded_measurements,
        hist_eqs=micropattern.histogram_percentiles,
        intensity_factors=micropattern.intensity_factors,
        cleaning=_cleaning_for_loader(micropattern.cleaning),
        excluded_images=_excluded_images(micropattern),
        cache_dir=cache_dir_from_environment(),
    )
    return {
        "images": np.asarray(dataset.data),
        "measured": np.asarray(dataset.measurement_mask),
        "masks": np.asarray(dataset.boundary_mask[:, 0]).astype(bool),
        "keys": list(zip(dataset.aux["batch_conditions"], dataset.aux["batch_replicates"])),
        "channel_names": tuple(dataset.channel_names),
    }


@app.function(hide_code=True)
def placeholder_rgba(size=64):
    """Grey disc, drawn where no data is loaded."""
    rows, cols = np.indices((size, size))
    inside = np.hypot(rows - (size - 1) / 2, cols - (size - 1) / 2) <= size / 2
    rgba = np.zeros((size, size, 4))
    rgba[..., :3] = 0.35
    rgba[..., 3] = inside
    return rgba


@app.function(hide_code=True)
def colony_tile(data, condition, replicate, hour, groups, gamma):
    """RGBA disc of one colony at ``hour``, one sector per group, and which sectors
    were not measured. A grey placeholder without data."""
    if data is None or (condition, replicate) not in data["keys"]:
        return placeholder_rgba(), ()
    batch = data["keys"].index((condition, replicate))
    time_index = TIMEPOINTS.index(hour)
    names, image = data["channel_names"], data["images"][batch, time_index]
    measured = data["measured"][batch, time_index]
    rgbas = [ko_drawing.colony_rgba(image, data["masks"][batch], group_colours(names, g), gamma) for g in groups]
    unmeasured = tuple(not any(m for n, m in zip(names, measured) if n.split("/")[0] == g) for g in groups)
    return ko_drawing.sliced_rgba(rgbas), unmeasured


@app.function(hide_code=True)
def toy_patterning(size, frame_count, steps_per_frame, seed):
    """Toy patterning process made only of local updates, for illustration.

    A signal enters at the colony edge and spreads by diffusion (a 4-neighbour
    Laplacian); each pixel then relaxes its three fates towards levels set by the
    local signal: high → SOX17-like, middle → TBXT-like, low → SOX2-like. Returns
    fates [F, 3, size, size] and the colony mask.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((size, size))
    centre = (size - 1) / 2
    mask = np.hypot(rows - centre, cols - centre) <= size / 2 - 0.5
    shifts = ((1, 0), (-1, 0), (1, 1), (-1, 1))
    neighbour_masks = [np.roll(mask, s, axis=a) for s, a in shifts]
    neighbour_count = sum(neighbour_masks)
    rim = mask & (neighbour_count < 4)
    # Fixed per-pixel jitter, so ring borders are not perfectly round
    jitter = 0.06 * rng.normal(size=(size, size))
    rate = 0.06 * (1 + 0.5 * rng.uniform(-1, 1, size=(size, size)))

    signal = np.zeros((size, size))
    fates = np.zeros((3, size, size))
    fates[0] = np.clip(0.85 + 0.1 * rng.normal(size=(size, size)), 0, 1)
    frames = [fates * mask]
    for _ in range(frame_count - 1):
        for _ in range(steps_per_frame):
            # Zero-flux Laplacian: only neighbours inside the colony
            laplacian = sum(np.roll(signal, s, axis=a) * m for (s, a), m in zip(shifts, neighbour_masks))
            laplacian = laplacian - neighbour_count * signal
            signal = (signal + 0.2 * laplacian - 0.02 * signal) * mask
            signal[rim] += 0.3 * (1.0 - signal[rim])
            level = signal + jitter
            high = 1 / (1 + np.exp(-(level - 0.5) / 0.04))
            middle = (1 / (1 + np.exp(-(level - 0.22) / 0.04))) * (1 - high)
            target = np.stack([np.clip(1 - high - middle, 0, 1), middle, high])
            fates = fates + rate * (target - fates)
        frames.append(fates * mask)
    return np.stack(frames), mask


@app.function(hide_code=True)
def toy_rgba(fates, mask):
    """RGBA [H, W, 4] of toy fates [3, H, W] on black, transparent outside the colony."""
    rgb = ko_drawing.composite(np.clip(fates, 0, 1), TOY_COLOURS)
    return np.concatenate([rgb, mask[..., None].astype(float)], axis=-1)


@app.function(hide_code=True)
def draw_arrow(ax, start, end, colour, lw=1.0, **kwargs):
    """Plain arrow from ``start`` to ``end`` (data units)."""
    ax.annotate("", xy=end, xytext=start, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9,
                                **kwargs))


@app.function(hide_code=True)
def draw_zoom_outline(ax, corner, size, dashes, z):
    """Square outline of black dashes over a solid white line, so it shows on both
    light pages and dark images. Marks the zoomed neighbourhood in panel b."""
    for colour, style in (("white", "-"), ("black", (0, dashes))):
        ax.add_patch(Rectangle(corner, size, size, fill=False, edgecolor=colour, lw=0.9, linestyle=style,
                               zorder=z))


@app.function(hide_code=True)
def draw_card(ax, corner, size, z, style, rgba=None, unmeasured=(), alpha=1.0, edge=None, pixelated=False):
    """One microscopy image: a black square with the colony disc in it.

    Hatched sectors were not measured; ``z`` sets the drawing order of the card."""
    theme = style["theme"]
    x, y = corner
    ax.add_patch(Rectangle(corner, size, size, facecolor="black", edgecolor=edge or theme["muted"], lw=0.6,
                           zorder=z))
    if rgba is None:
        return
    disc = 0.9 * size
    cx, cy = x + size / 2, y + size / 2
    shown = rgba.copy()
    shown[..., 3] *= alpha
    ax.imshow(shown, extent=(cx - disc / 2, cx + disc / 2, cy - disc / 2, cy + disc / 2), origin="upper",
              interpolation="nearest" if pixelated else "antialiased", zorder=z + 0.1)
    for k, missing in enumerate(unmeasured):
        start, stop = ko_drawing.sector_angles(k, len(unmeasured))
        if missing:
            ax.add_patch(Wedge((cx, cy), disc / 2, start, stop, facecolor="#3a3a3a", edgecolor="#8c8c8c",
                               hatch="////", lw=0, alpha=alpha, zorder=z + 0.2))
        if len(unmeasured) > 1:
            angle = np.radians(start)
            ax.plot([cx, cx + disc / 2 * np.cos(angle)], [cy, cy + disc / 2 * np.sin(angle)], color="white",
                    lw=0.5, alpha=alpha, zorder=z + 0.3)


@app.function(hide_code=True)
def draw_paper_stack(ax, tiles, corner, size, offset, style, pixelated=False):
    """Sheets drawn back to front, each shifted by ``offset`` from the one in front.

    ``tiles`` holds (rgba, unmeasured) per sheet, front first. Only the front sheet is
    clearly visible; ``style["back_cards"]`` sets the others (blank, faded or full)."""
    count = len(tiles)
    for k in reversed(range(count)):
        rgba, unmeasured = tiles[k]
        back = k > 0
        if back and style["back_cards"] == "blank":
            rgba, unmeasured = None, ()
        alpha = 0.4 if back and style["back_cards"] == "faded" else 1.0
        draw_card(ax, (corner[0] + k * offset[0], corner[1] + k * offset[1]), size, 10 + count - k, style, rgba,
                  unmeasured, alpha, pixelated=pixelated)


@app.function(hide_code=True)
def draw_microscope(ax, origin, style):
    """Cartoon objective above a dish of colonies, focused on one colony."""
    theme = style["theme"]
    muted, accent = theme["muted"], theme["accent"]
    ox, oy = origin
    cx = ox + 1.25

    # Dish: base, wall and top surface
    ax.add_patch(Ellipse((cx, oy + 0.25), 2.4, 0.7, facecolor="#dcdcdc", edgecolor=muted, lw=0.6, zorder=2))
    ax.add_patch(Rectangle((cx - 1.2, oy + 0.25), 2.4, 0.18, facecolor="#dcdcdc", edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((cx, oy + 0.43), 2.4, 0.7, facecolor="#f4f4f4", edgecolor=muted, lw=0.6, zorder=3))
    for side in (-1.2, 1.2):
        ax.plot([cx + side, cx + side], [oy + 0.25, oy + 0.43], color=muted, lw=0.6, zorder=3)
    colonies = [(u, v) for v in (-0.15, 0.0, 0.15) for u in (-0.72, -0.36, 0.0, 0.36, 0.72)
                if (u / 1.05) ** 2 + (v / 0.3) ** 2 < 1]
    for u, v in colonies:
        ax.add_patch(Ellipse((cx + u, oy + 0.43 + v), 0.22, 0.08, facecolor="#2a2a2a", edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((cx, oy + 0.43), 0.3, 0.12, fill=False, edgecolor=accent, lw=0.8, zorder=5))

    # Light cone focused on that colony
    nose = oy + 1.75
    ax.add_patch(Polygon([(cx - 0.16, nose), (cx + 0.16, nose), (cx, oy + 0.45)], facecolor=style["light_colour"],
                         alpha=0.5, lw=0, zorder=4))

    # Objective: nose, barrel with a coloured ring, thread
    ax.add_patch(Polygon([(cx - 0.17, nose), (cx + 0.17, nose), (cx + 0.3, nose + 0.3), (cx - 0.3, nose + 0.3)],
                         facecolor="#b0b0b0", edgecolor=muted, lw=0.6, zorder=6))
    ax.add_patch(Rectangle((cx - 0.3, nose + 0.3), 0.6, 0.8, facecolor="#d0d0d0", edgecolor=muted, lw=0.6,
                           zorder=6))
    ax.add_patch(Rectangle((cx - 0.3, nose + 0.52), 0.6, 0.08, facecolor=accent, edgecolor="none", zorder=7))
    ax.add_patch(Rectangle((cx - 0.22, nose + 1.1), 0.44, 0.15, facecolor="#a8a8a8", edgecolor=muted, lw=0.6,
                           zorder=6))
    ax.text(cx, oy - 0.2, style["imaging_label"], ha="center", va="top", fontsize=style["font"] - 1,
            linespacing=1.3)
    return cx + 1.3, nose + 1.25


@app.function(hide_code=True)
def draw_marker_legend(fig, data, style):
    """Marker colours of each staining group in one column, each headed by the colony
    sector that group fills, in a grey box floating over the figure (left out of the
    layout). Its top right corner is at ``style["legend_anchor"]`` (fractions of the figure)."""
    names = data["channel_names"]
    blank = plt.Line2D([], [], lw=0)
    legend_style = "white" if style["theme"]["bg"] == "white" else "black"
    groups = style["groups"]
    entries = []
    for k, group in enumerate(groups):
        if entries:
            entries.append((blank, ""))
        # Header: a small disc with this group's sector filled (when the colonies are split)
        header = ko_drawing.SectorHandle(k, len(groups)) if len(groups) > 1 else blank
        entries.append((header, ko_drawing.GROUP_LABELS.get(group, group)))
        entries += [(legend_swatch(c, legend_style), n.rsplit("/", 1)[-1])
                    for n, c in zip(names, group_colours(names, group)) if c is not None]
    legend = fig.legend([h for h, _ in entries], [l for _, l in entries], ncol=1, loc="upper right",
                        bbox_to_anchor=style["legend_anchor"], fontsize=style["font"] - 1, handlelength=1.0,
                        labelspacing=0.25, **legend_options(legend_style))
    legend.get_frame().set_edgecolor(style["theme"]["muted"])
    legend.get_frame().set_linewidth(0.6)
    legend.set_in_layout(False)
    legend.set_zorder(100)
    headers = {ko_drawing.GROUP_LABELS.get(g, g) for g in style["groups"]}
    for text in legend.get_texts():
        if text.get_text() in headers:
            text.set_fontweight("bold")


@app.function(hide_code=True)
def draw_imaging_panel(ax, data, style):
    """Panel a: microscope cartoon and one paper stack of replicates per timepoint."""
    theme = style["theme"]
    fg, muted = theme["fg"], theme["muted"]
    font, small = style["font"], style["font"] - 1
    size, step = style["card_size"], style["sheet_offset"]
    replicates = style["replicates"]
    y0 = 1.0
    bottom = y0 - 0.85

    x, microscope_top = draw_microscope(ax, (0.0, y0 - 0.35), style)
    x += 0.15
    draw_arrow(ax, (x, y0 + size / 2), (x + 0.55, y0 + size / 2), fg)
    x += 0.8
    first = x
    stack_width = size + (len(replicates) - 1) * step
    for hour in style["times"]:
        tiles = [colony_tile(data, style["condition"], r, hour, style["groups"], style["gamma"]) for r in replicates]
        draw_paper_stack(ax, tiles, (x, y0), size, (step, step), style)
        ax.text(x + size / 2, y0 - 0.08, f"{hour}h", ha="center", va="top", fontsize=font)
        x += stack_width + style["stack_gap"]
    last = x - style["stack_gap"]
    draw_arrow(ax, (first, y0 - 0.5), (last, y0 - 0.5), muted, lw=0.8)
    ax.text(last, y0 - 0.58, "time", ha="right", va="top", fontsize=small, color=muted)

    # Replicates: along the right edge of the last stack
    top = y0 + size + (len(replicates) - 1) * step
    right = last
    if style["replicate_label"] and len(replicates) > 1:
        # Offset to the lower right so the arrow runs parallel to the stack edge, not along it
        start = (last - stack_width + size + 0.25, y0)
        end = (start[0] + (len(replicates) - 1) * step, start[1] + (len(replicates) - 1) * step)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->", color=muted, lw=0.6, shrinkA=0,
                                                              shrinkB=0))
        ax.text(end[0] + 0.08, (start[1] + end[1]) / 2, f"{len(replicates)}\nreplicates", ha="left",
                va="center", fontsize=small, color=muted, linespacing=1.2)
        right = end[0] + 0.85
    ax.text(first, top + 0.12, style["condition_note"], ha="left", va="bottom", fontsize=small, color=muted)

    ax.set_xlim(-0.1, right + 0.1)
    ax.set_ylim(bottom, max(top + 0.3, microscope_top) + 0.45)


@app.function(hide_code=True)
def draw_training_panel(ax, data, toy, style):
    """Panel b: data stack → training → the patterning dynamics, as one of two layouts:
    v1: zoomed neighbourhood → updated pixel → time series of frames
    v2: two spaced slices (t, t+1) in 3D, with a local update cone between them"""
    theme = style["theme"]
    fg, muted, model = theme["fg"], theme["muted"], style["model_colour"]
    font, small = style["font"], style["font"] - 1
    frames, mask = toy
    centre_y = 1.7

    # Data stack: timepoints of one replicate, or the toy frames
    size, step = 1.3, 0.14
    if style["stack_source"] == "data" and data is not None:
        tiles = [colony_tile(data, style["condition"], style["replicates"][0], h, style["groups"], style["gamma"])
                 for h in style["times"]]
    else:
        tiles = [(toy_rgba(f, mask), ()) for f in frames]
    tiles = tiles[:style["stack_count"]]
    depth = (len(tiles) - 1) * step
    stack_y = centre_y - (size + depth) / 2
    draw_paper_stack(ax, tiles, (0.0, stack_y), size, (step, step), style,
                     pixelated=style["stack_source"] != "data")
    ax.text((size + depth) / 2, stack_y - 0.1, style["data_label"], ha="center", va="top", fontsize=font,
            linespacing=1.3)
    x = size + depth + 0.2

    # Training block
    draw_arrow(ax, (x, centre_y), (x + 0.5, centre_y), fg)
    x += 0.6
    block_w, block_h = 1.8, 1.0
    ax.add_patch(FancyBboxPatch((x, centre_y - block_h / 2), block_w, block_h,
                                boxstyle="round,pad=0,rounding_size=0.15", facecolor=model, alpha=0.15,
                                edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x, centre_y - block_h / 2), block_w, block_h,
                                boxstyle="round,pad=0,rounding_size=0.15", fill=False, edgecolor=model, lw=1.0,
                                zorder=3))
    ax.text(x + block_w / 2, centre_y + 0.1, style["training_label"], ha="center", va="center",
            fontsize=font + 1, fontweight="bold", zorder=4)
    ax.text(x + block_w / 2, centre_y - 0.22, style["training_note"], ha="center", va="center", fontsize=small,
            color=muted, zorder=4)
    x += block_w + 0.1
    draw_arrow(ax, (x, centre_y), (x + 0.5, centre_y), fg)
    x += 0.65

    draw_dynamics = draw_update_slices if style["training_layout"] == "v2" else draw_update_series
    right, label_top = draw_dynamics(ax, x, centre_y, toy, style)
    ax.set_xlim(-0.1, right + 0.1)
    ax.set_ylim(centre_y - 1.55, label_top + (1.0 if style["equation"] else 0.6))


@app.function(hide_code=True)
def draw_update_series(ax, x, centre_y, toy, style):
    """Panel b, v1: zoomed neighbourhood at t → its centre pixel at t+1 → model rollout
    of toy frames, with the zoom marked on its frame. The rollout frames sit on a
    continuous time axis at evenly spaced times, to show the model can be evaluated
    at any time. Returns (right edge, top of the art)."""
    theme = style["theme"]
    fg, muted, accent = theme["fg"], theme["muted"], theme["accent"]
    font, small = style["font"], style["font"] - 1
    frames, mask = toy

    # Local update: a neighbourhood at one step → the centre pixel at the next
    n, grid = 7, 1.5
    zoom = min(style["zoom_frame"], len(frames) - 2)
    half = frames.shape[-1] // 2
    c0 = int(np.flatnonzero(mask[half])[0])
    r0 = half - n // 2
    before = toy_rgba(frames[zoom], mask)[r0:r0 + n, c0:c0 + n]
    after = toy_rgba(frames[zoom + 1], mask)[r0:r0 + n, c0:c0 + n]
    cell = grid / n
    gy = centre_y - grid / 2
    for i in range(n):
        for j in range(n):
            colour = before[i, j, :3] if before[i, j, 3] > 0.5 else "#efefef"
            ax.add_patch(Rectangle((x + j * cell, gy + grid - (i + 1) * cell), cell, cell, facecolor=colour,
                                   edgecolor="#555555", lw=0.3, zorder=3))
    # Dashed outline: the same outline marks where this zoom sits in the time series
    pad = 0.05
    draw_zoom_outline(ax, (x - pad, gy - pad), grid + 2 * pad, (3, 1.5), 4)
    m = n // 2
    ax.add_patch(Rectangle((x + (m - 1) * cell, gy + grid - (m + 2) * cell), 3 * cell, 3 * cell, fill=False,
                           edgecolor=style["neighbourhood_colour"], lw=1.4, zorder=4))
    ax.add_patch(Rectangle((x + m * cell, gy + grid - (m + 1) * cell), cell, cell, fill=False,
                           edgecolor="white", lw=0.9, zorder=4))
    # Baseline alignment keeps $t$ and $t+1$ at the same height
    ax.text(x + grid / 2, gy - 0.25, "$t$", ha="center", va="baseline", fontsize=small)
    update_x = x
    x += grid + 0.15
    draw_arrow(ax, (x, centre_y), (x + 0.55, centre_y), accent)
    ax.text(x + 0.27, centre_y + 0.08, r"$f_\theta$", ha="center", va="bottom", fontsize=font, color=accent)
    x += 0.65
    single = 1.2 * cell
    ax.add_patch(Rectangle((x, centre_y - single / 2), single, single, facecolor=after[m, m, :3],
                           edgecolor="white", lw=0.9, zorder=3))
    ax.add_patch(Rectangle((x, centre_y - single / 2), single, single, fill=False, edgecolor="#555555", lw=0.3,
                           zorder=4))
    ax.text(x + single / 2, gy - 0.25, "$t+1$", ha="center", va="baseline", fontsize=small)
    middle = (update_x + x + single) / 2
    ax.text(middle, centre_y + grid / 2 + 0.12, style["update_label"], ha="center", va="bottom", fontsize=font,
            linespacing=1.3)
    if style["equation"]:
        ax.text(middle, centre_y + grid / 2 + 0.55, style["equation"], ha="center", va="bottom", fontsize=font)
    x += single + 0.2

    # Model rollout: toy frames at evenly spaced (not only imaged) times. The row keeps
    # the same total width, so frames shrink as more are added.
    draw_arrow(ax, (x, centre_y), (x + 0.5, centre_y), fg)
    x += 0.65
    gap = 0.2
    frame_size = min(1.2, (6.8 - gap * (len(frames) - 1)) / len(frames))
    first = x
    hours = np.linspace(0, TIMEPOINTS[-1], len(frames))
    for k, (fates, hour) in enumerate(zip(frames, hours)):
        draw_card(ax, (x, centre_y - frame_size / 2), frame_size, 10, style, toy_rgba(fates, mask), pixelated=True)
        if k == zoom:
            # Where the zoomed neighbourhood comes from
            disc, pixels = 0.9 * frame_size, frames.shape[-1]
            left = x + 0.05 * frame_size + c0 / pixels * disc
            low = centre_y - frame_size / 2 + 0.05 * frame_size + (1 - (r0 + n) / pixels) * disc
            draw_zoom_outline(ax, (left, low), n / pixels * disc, (2, 1), 12)
        ax.text(x + frame_size / 2, centre_y - frame_size / 2 - 0.08, f"{hour:.3g}h", ha="center", va="top",
                fontsize=small)
        x += frame_size + gap
    last = x - gap

    # Continuous time axis: frame centres sit at their times, with a tick under each frame
    axis_y = centre_y - frame_size / 2 - 0.45

    def at(hour):
        return first + frame_size / 2 + (last - first - frame_size) * hour / TIMEPOINTS[-1]

    draw_arrow(ax, (first, axis_y), (last, axis_y), muted, lw=0.8)
    for hour in hours:
        ax.plot([at(hour)] * 2, [axis_y - 0.05, axis_y + 0.05], color=fg, lw=0.6, zorder=5)
    ax.text(last, axis_y - 0.08, "time", ha="right", va="top", fontsize=small, color=muted)
    ax.text((first + last) / 2, centre_y + frame_size / 2 + 0.12, style["series_label"], ha="center",
            va="bottom", fontsize=font, linespacing=1.3)
    return last, centre_y + grid / 2


@app.function(hide_code=True)
def draw_update_slices(ax, x, centre_y, toy, style):
    """Panel b, v2: toy frames at t and t+1 as two spaced slices in an oblique 3D view,
    with a few earlier frames packed tightly behind t (a time series). A 3x3
    neighbourhood on t feeds a cone to one pixel on t+1: the local update f_theta. The
    pixel is the one that changes most between the two frames. Slices lie flat and
    time runs upwards, or with ``style["slices_transposed"]`` stand upright and time
    runs left to right. Returns (right edge, top of the art)."""
    theme = style["theme"]
    muted, local = theme["muted"], style["neighbourhood_colour"]
    font, small = style["font"], style["font"] - 1
    frames, mask = toy
    k = min(style["zoom_frame"], len(frames) - 2)
    pixels = frames.shape[-1]
    transposed = style["slices_transposed"]

    # Edge vectors of a slice: e_u along image columns, e_v along image rows (upwards).
    # The receding direction (into the page) is the same in both orientations.
    recede = np.array([0.7, style["slice_depth"]])
    if transposed:
        e_u, e_v = recede, np.array([0.0, 1.6])
        e_next, e_pack = np.array([style["slice_gap"], 0.0]), np.array([-0.08, 0.0])
    else:
        e_u, e_v = np.array([2.2, 0.0]), recede
        e_next, e_pack = np.array([0.0, style["slice_gap"]]), np.array([0.0, -0.08])
    # Slices from the back: earlier frames, t, then t+1; corners first relative to t
    packed = style["packed_slices"]
    offsets = [j * e_pack for j in range(packed, 0, -1)] + [np.zeros(2), e_next]
    shown = [frames[max(k - j, 0)] for j in range(packed, 0, -1)] + [frames[k], frames[k + 1]]
    corners = np.array([o + c for o in offsets for c in (0 * e_u, e_u, e_v, e_u + e_v)])
    shift = np.array([x, centre_y]) - [corners[:, 0].min(), (corners[:, 1].min() + corners[:, 1].max()) / 2]
    origins = [o + shift for o in offsets]
    low_corner, high_corner = corners.min(axis=0) + shift, corners.max(axis=0) + shift

    def pixel_corners(origin, i, j, size=1):
        """Corners of the ``size``x``size`` block with top-left pixel (row i, column j)."""
        u0, u1 = j / pixels, (j + size) / pixels
        v0, v1 = 1 - (i + size) / pixels, 1 - i / pixels
        return [origin + u * e_u + v * e_v for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]

    def draw_slice(origin, fates, z):
        outline = [origin, origin + e_u, origin + e_u + e_v, origin + e_v]
        ax.add_patch(Polygon(outline, closed=True, facecolor="black", edgecolor=muted, lw=0.6, zorder=z))
        rgba = toy_rgba(fates, mask)
        cells = [(i, j) for i in range(pixels) for j in range(pixels) if mask[i, j]]
        colours = [rgba[i, j, :3] for i, j in cells]
        # Edges in the face colour hide the hairline seams between pixels
        ax.add_collection(PolyCollection([pixel_corners(origin, i, j) for i, j in cells], facecolors=colours,
                                         edgecolors=colours, linewidths=0.2, zorder=z + 0.1))

    # Target pixel: the largest change between the two frames, with its whole 3x3 inside the colony
    change = np.abs(frames[k + 1] - frames[k]).sum(axis=0)
    inside = mask & (sum(np.roll(np.roll(mask, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)) == 9)
    ti, tj = np.unravel_index(np.argmax(np.where(inside, change, -1.0)), change.shape)

    # Back to front, so each slice hides the ones behind it
    for z, (origin, fates) in enumerate(zip(origins[:-1], shown[:-1])):
        draw_slice(origin, fates, 1 + 0.2 * z)
    here, after = origins[-2], origins[-1]
    ax.add_patch(Polygon(pixel_corners(here, ti - 1, tj - 1, 3), closed=True, fill=False, edgecolor=local,
                         lw=1.4, zorder=5))
    ax.add_patch(Polygon(pixel_corners(here, ti, tj), closed=True, fill=False, edgecolor="white", lw=0.7,
                         zorder=5))

    # Cone from the neighbourhood corners to the target pixel: solid behind the t+1
    # slice, faint over it, so it reads as passing through the slice
    target = np.mean(pixel_corners(after, ti, tj), axis=0)
    for corner in pixel_corners(here, ti - 1, tj - 1, 3):
        for z, alpha in ((6, 1.0), (9, 0.35)):
            ax.plot([corner[0], target[0]], [corner[1], target[1]], color=local, lw=0.8, alpha=alpha, zorder=z)
    draw_slice(after, shown[-1], 7)
    ax.add_patch(Polygon(pixel_corners(after, ti, tj), closed=True, fill=False, edgecolor="white", lw=0.9,
                         zorder=10))
    cone_mid = (np.mean(pixel_corners(here, ti - 1, tj - 1, 3), axis=0) + target) / 2
    if transposed:
        ax.text(cone_mid[0], cone_mid[1] + 0.15, r"$f_\theta$", ha="center", va="bottom", fontsize=font,
                color=local, zorder=10)
    else:
        ax.text(cone_mid[0] + 0.15, cone_mid[1], r"$f_\theta$", ha="left", va="center", fontsize=font,
                color=local, zorder=10)

    # Time labels beside (flat) or under (upright) the t and t+1 slices, joined by an arrow
    centres = [o + (e_u + e_v) / 2 for o in (here, after)]
    if transposed:
        label_y = low_corner[1] - 0.25
        for (cx, _), text in zip(centres, ("$t$", "$t+1$")):
            ax.text(cx, label_y, text, ha="center", va="baseline", fontsize=small)
        draw_arrow(ax, (centres[0][0] + 0.2, label_y + 0.05), (centres[1][0] - 0.3, label_y + 0.05), muted,
                   lw=0.8)
        right = high_corner[0]
    else:
        label_x = high_corner[0] + 0.15
        for (_, cy), text in zip(centres, ("$t$", "$t+1$")):
            ax.text(label_x, cy, text, ha="left", va="center", fontsize=small)
        draw_arrow(ax, (label_x + 0.07, centres[0][1] + 0.15), (label_x + 0.07, centres[1][1] - 0.15), muted,
                   lw=0.8)
        right = label_x + 0.45

    top = high_corner[1]
    middle = (low_corner[0] + high_corner[0]) / 2
    ax.text(middle, top + 0.12, style["update_label"], ha="center", va="bottom", fontsize=font, linespacing=1.3)
    if style["equation"]:
        ax.text(middle, top + 0.55, style["equation"], ha="center", va="bottom", fontsize=font)
    return right, top


@app.function(hide_code=True)
def parse_hours(text):
    """``"12, 36"`` → [12.0, 36.0], skipping anything that is not a number."""
    hours = []
    for item in text.split(","):
        try:
            hours.append(float(item))
        except ValueError:
            continue
    return hours


@app.function(hide_code=True)
def draw_perturbation_panel(ax, data, style):
    """Panel c: 0h data → final data, joined by a long arrow (the rollout). Under the
    0h image, a perturbation block points into the rollout at any time and an initial
    cell density box points back up into the image; mock curves of one cell type
    against perturbation time (one per density) point back between the two boxes."""
    theme = style["theme"]
    fg, muted, accent = theme["fg"], theme["muted"], theme["accent"]
    font, small = style["font"], style["font"] - 1
    size, y0 = 1.8, 1.2
    arrow_y = y0 + size / 2
    final_hour = TIMEPOINTS[-1]
    condition, replicate = style["final_condition"], style["final_replicate"]

    rgba, unmeasured = colony_tile(data, condition, replicate, 0, style["groups"], style["gamma"])
    draw_card(ax, (0.0, y0), size, 10, style, rgba, unmeasured)
    ax.text(size / 2, y0 + size + 0.1, style["initial_label"], ha="center", va="bottom", fontsize=font,
            linespacing=1.3)

    # Long arrow: the rollout, with the timepoints marked
    start, end = size + 0.25, size + 0.25 + style["arrow_length"]

    def at(hour):
        return start + (end - start - 0.15) * hour / final_hour

    draw_arrow(ax, (start, arrow_y), (end, arrow_y), fg, lw=1.6)
    measured = [h for h in style["measured_hours"] if 0 <= h <= final_hour]
    for hour in TIMEPOINTS:
        ax.plot([at(hour)] * 2, [arrow_y - 0.06, arrow_y + 0.06], color=fg, lw=0.8)
        ax.text(at(hour), arrow_y + 0.12, f"{hour}h", ha="center", va="bottom", fontsize=small,
                color=fg if hour in measured else muted, fontweight="bold" if hour in measured else "normal")
    # The only perturbation times with data: the measured data is sparse
    for hour in measured:
        ax.plot(at(hour), arrow_y, "D", color=fg, markeredgecolor=theme["bg"], markersize=6, zorder=12)
    if measured and style["measured_label"]:
        ax.plot(start + 0.05, arrow_y + 0.5, "D", color=fg, markeredgecolor=theme["bg"], markersize=5)
        ax.text(start + 0.15, arrow_y + 0.5, style["measured_label"], ha="left", va="center", fontsize=small)
    ax.text(end, arrow_y + 0.42, style["rollout_label"], ha="right", va="bottom", fontsize=font)

    def red_box(bottom, height, label):
        for kwargs in (dict(facecolor=accent, alpha=0.15, edgecolor="none", zorder=2),
                       dict(fill=False, edgecolor=accent, lw=1.0, zorder=3)):
            ax.add_patch(FancyBboxPatch((0.0, bottom), size, height, boxstyle="round,pad=0,rounding_size=0.12",
                                        **kwargs))
        ax.text(size / 2, bottom + height / 2, label, ha="center", va="center", fontsize=font,
                fontweight="bold", linespacing=1.2, zorder=4)

    # Perturbation block under the 0h image; its arrow runs right, then turns up into the timeline
    block_h = 0.8
    by = y0 - 0.3 - block_h
    level = by + block_h / 2
    red_box(by, block_h, style["perturbation_label"])

    # Initial cell density box below that; its arrow curves up the left side into the image
    density_h = 0.6
    gap = 0.3
    dy = by - gap - density_h
    red_box(dy, density_h, style["density_label"])
    left = -0.55
    ax.add_patch(FancyArrowPatch((-0.04, dy + density_h / 2), (-0.04, y0 + 0.25),
                                 connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=9,
                                 color=accent, lw=1.4, shrinkA=0, shrinkB=0, zorder=12))

    def elbow_arrow(hour, **kwargs):
        ax.annotate("", xy=(at(hour), arrow_y - 0.08), xytext=(size + 0.04, level), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=accent, shrinkA=0, shrinkB=0, mutation_scale=9,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=0", **kwargs))

    for hour in style["ghost_hours"]:
        if 0 <= hour <= final_hour:
            elbow_arrow(hour, lw=0.8, alpha=0.35, linestyle=(0, (2, 2)))
    elbow_arrow(style["perturbation_hour"], lw=1.4)
    ax.text(at(style["perturbation_hour"]) + 0.1, (level + arrow_y) / 2, style["perturbation_time_label"],
            ha="left", va="center", fontsize=small, color=accent)

    # Final image, then a mock curve of one cell type against perturbation time
    fx = end + 0.25
    rgba, unmeasured = colony_tile(data, condition, replicate, final_hour, style["groups"], style["gamma"])
    draw_card(ax, (fx, y0), size, 10, style, rgba, unmeasured)
    ax.text(fx + size / 2, y0 + size + 0.1, style["final_label"], ha="center", va="bottom", fontsize=font,
            linespacing=1.3)

    px, pw = fx + size + 0.75, 2.4
    backing = dict(facecolor=theme["bg"], edgecolor="none", pad=0.5)

    def plot_x(hour):
        return px + pw * hour / final_hour

    def plot_y(value):
        return y0 + size * value

    hours = np.linspace(0, final_hour, 200)
    ax.plot([px, px], [y0, y0 + size], color=fg, lw=0.7, zorder=3)
    ax.plot([px, px + pw], [y0, y0], color=fg, lw=0.7, zorder=3)
    curves = mock_curves(style)
    for curve in curves:
        ax.plot(plot_x(hours), plot_y(mock_prevalence(hours, curve)), color=curve["colour"], lw=1.3,
                ls=curve["line"], zorder=4)
    for hour in (0, final_hour):
        ax.text(plot_x(hour), y0 - 0.08, f"{hour}h", ha="center", va="top", fontsize=small, color=muted,
                bbox=backing, zorder=6)
    ax.text(px + pw / 2, y0 - 0.08, style["curve_x_label"], ha="center", va="top", fontsize=small,
            bbox=backing, zorder=6)
    # Rotated anchor frame: ha centres the label along the axis, va="bottom" puts it left of the axis
    ax.text(px - 0.12, y0 + size / 2, style["curve_y_label"], ha="center", va="bottom", rotation=90,
            rotation_mode="anchor", fontsize=small, linespacing=1.2)
    ax.text(px + pw / 2, y0 + size + 0.1, style["curve_title"], ha="center", va="bottom", fontsize=font,
            linespacing=1.3)

    # Density legend, beside the plot
    for k, curve in enumerate(curves):
        ly = y0 + size - 0.15 - k * 0.27
        ax.plot([px + pw + 0.15, px + pw + 0.45], [ly, ly], color=curve["colour"], lw=1.3, ls=curve["line"])
        ax.text(px + pw + 0.52, ly, curve["label"], ha="left", va="center", fontsize=small)

    # The arrow's perturbation time, as a vertical line in the plot
    hour = style["perturbation_hour"]
    vx = plot_x(hour)
    ax.plot([vx] * 2, [y0, y0 + size], color=accent, lw=1.0, ls=(0, (3, 2)), zorder=4)
    for curve in curves:
        value = float(mock_prevalence(np.array([hour]), curve)[0])
        ax.plot(vx, plot_y(value), "o", color=curve["colour"], markeredgecolor=accent, markersize=4, zorder=5)

    # Back from the plot to the gap between the two boxes: timing and density are
    # optimised together
    between = by - gap / 2
    ax.annotate("", xy=(size + 0.08, between), xytext=(vx, y0), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=accent, lw=1.8, ls=(0, (0.1, 2.2)), capstyle="round",
                                shrinkA=0, shrinkB=0,
                                mutation_scale=9, connectionstyle="angle,angleA=90,angleB=0,rad=0"))
    ax.text((size + vx) / 2, between - 0.08, style["search_label"], ha="center", va="top", fontsize=small,
            color=accent)
    ax.set_xlim(left - 0.1, px + pw + 0.55 + 0.09 * max(len(c["label"]) for c in curves) * small / 7)
    ax.set_ylim(dy - 0.15, y0 + size + 0.85)


@app.function(hide_code=True)
def mock_prevalence(hours, curve):
    """Mock prevalence of one cell type against perturbation time: a bump at
    ``curve["peak"]`` hours, ``curve["width"]`` hours wide."""
    return 0.15 + curve["height"] * np.exp(-(((hours - curve["peak"]) / curve["width"]) ** 2))


@app.function(hide_code=True)
def mock_curves(style):
    """The main mock curve and up to two others for different cell densities, each
    shifted, wider or narrower, and lower, so they look a bit different."""
    shapes = (
        {"shift": 0.0, "scale": 1.0, "height": 0.65, "colour": style["model_colour"], "line": "-"},
        {"shift": -9.0, "scale": 0.75, "height": 0.45, "colour": "#8c6bb1", "line": (0, (4, 1.5))},
        {"shift": 8.0, "scale": 1.4, "height": 0.35, "colour": "#41ab5d", "line": (0, (1.5, 1))},
    )
    labels = [label.strip() for label in style["density_labels"].split(",")]
    curves = []
    for k, shape in enumerate(shapes[:1 + style["extra_curves"]]):
        curves.append({
            "peak": style["curve_peak"] + shape["shift"],
            "width": style["curve_width"] * shape["scale"],
            "height": shape["height"],
            "colour": shape["colour"],
            "line": shape["line"],
            "label": labels[k] if k < len(labels) else f"density {k + 1}",
        })
    return curves


@app.function(hide_code=True)
def overview_figure(panels, data, toy, style):
    """The chosen panels stacked top to bottom; with ``style["same_scale"]`` every panel
    uses the same data-units-per-inch."""
    theme = style["theme"]
    draw = {
        "a": lambda ax: draw_imaging_panel(ax, data, style),
        "b": lambda ax: draw_training_panel(ax, data, toy, style),
        "c": lambda ax: draw_perturbation_panel(ax, data, style),
    }
    with plt.rc_context({"text.color": theme["fg"], "font.size": style["font"]}):
        fig = plt.figure(figsize=(style["width"], 2.0), facecolor=theme["bg"], layout="constrained")
        axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
        for ax, key in zip(axes, panels):
            ax.set_facecolor(theme["bg"])
            draw[key](ax)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(0.0, 1.0, key, transform=ax.transAxes, ha="left", va="top", fontsize=style["font"] + 3,
                    fontweight="bold")
            if style["titles"][key]:
                ax.text(0.03, 1.0, style["titles"][key], transform=ax.transAxes, ha="left", va="top",
                        fontsize=style["font"] + 1)
        widths = [ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes]
        if style["same_scale"]:
            for ax in axes:
                ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max(widths))
        ratios = [(ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) for ax in axes]
        axes[0].get_subplotspec().get_gridspec().set_height_ratios(ratios)
        fig.set_size_inches(style["width"], style["width"] * sum(ratios) + 0.1 * len(panels))
        if style["legend"] and data is not None:
            draw_marker_legend(fig, data, style)
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Project overview

    - a: imaging and the replicate time series; b: training an NCA; c: probing perturbations with the model
    - Panel b's patterning process and panel c's curve are illustrative, not results
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_root = mo.ui.text(
        value=str(Path(os.getenv("DATA_PATH_BASE", "")) / "260726_nca_dataset"),
        label="Dataset root",
        full_width=True,
    )
    groups = mo.ui.multiselect(options=list(ko_drawing.EXPERIMENT_GROUPS), value=["cell_fate_s1"],
                               label="Experiment groups")
    projection = mo.ui.dropdown(options=list(ko_drawing.PROJECTION_CLEANING), value="max", label="Projection")
    downsample = mo.ui.dropdown(options={"4": 4, "8": 8, "16": 16}, value="8", label="Downsample")
    condition = mo.ui.dropdown(options=list(CONDITION_LABELS), value="ctrl", label="Condition (a)")
    times = mo.ui.multiselect(options=[f"{t}h" for t in TIMEPOINTS], value=[f"{t}h" for t in TIMEPOINTS],
                              label="Timepoints (a)")
    replicates = mo.ui.text(value="1, 2, 3, 4", label="Replicates (a, one-based)")
    final_condition = mo.ui.dropdown(options=list(CONDITION_LABELS), value="sl24", label="Condition (c)")
    final_replicate = mo.ui.number(start=1, stop=20, step=1, value=1, label="Replicate (c)")
    mo.vstack([
        mo.md("**Data**"),
        dataset_root,
        mo.hstack([groups, projection, downsample], justify="start"),
        mo.hstack([condition, times, replicates], justify="start"),
        mo.hstack([final_condition, final_replicate], justify="start"),
    ])
    return (
        condition,
        dataset_root,
        downsample,
        final_condition,
        final_replicate,
        groups,
        projection,
        replicates,
        times,
    )


@app.cell(hide_code=True)
def _(
    condition,
    dataset_root,
    downsample,
    final_condition,
    final_replicate,
    groups,
    projection,
    replicates,
    times,
):
    selected_groups = [g for g in ko_drawing.EXPERIMENT_GROUPS if g in groups.value] or ["cell_fate_s1"]
    selected_times = sorted(int(t.rstrip("h")) for t in times.value) or [0]
    selected_replicates = [int(r) for r in replicates.value.replace(",", " ").split() if r.isdigit()] or [1]
    _conditions = list(dict.fromkeys(["ctrl", condition.value, final_condition.value]))
    _replicates = sorted(set(selected_replicates) | {int(final_replicate.value)})
    if Path(dataset_root.value).expanduser().is_dir():
        with mo.status.spinner("Loading colonies"):
            data = load_overview_data(Path(dataset_root.value).expanduser(), _conditions, selected_groups,
                                      projection.value, downsample.value, _replicates)
        _note = mo.md(f"Loaded {len(data['keys'])} trajectories of {', '.join(_conditions)}")
    else:
        data = None
        _note = mo.callout("Dataset root not found: colonies are drawn as placeholders.", kind="warn")
    _note
    return data, selected_groups, selected_replicates, selected_times


@app.cell(hide_code=True)
def _():
    background = mo.ui.dropdown(options=list(ko_drawing.THEMES), value="light", label="Page background")
    gamma = mo.ui.slider(start=0.3, stop=2.0, step=0.05, value=1.0, label="Gamma", show_value=True)
    figure_width = mo.ui.slider(start=5.0, stop=14.0, step=0.25, value=7.5, label="Width (in)", show_value=True)
    font_size = mo.ui.slider(start=5, stop=12, step=0.5, value=7, label="Font size", show_value=True)
    panels = mo.ui.multiselect(options=["a", "b", "c"], value=["a", "b", "c"], label="Panels")
    same_scale = mo.ui.switch(value=True, label="Same scale in every panel")
    back_cards = mo.ui.dropdown(options=["blank", "faded", "full"], value="blank", label="Sheets behind the top")
    card_size = mo.ui.slider(start=0.8, stop=2.0, step=0.05, value=1.3, label="Image size (a)", show_value=True)
    sheet_offset = mo.ui.slider(start=0.03, stop=0.25, step=0.01, value=0.09, label="Sheet offset (a)",
                                show_value=True)
    stack_gap = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.35, label="Gap between stacks (a)",
                             show_value=True)
    replicate_label = mo.ui.switch(value=True, label="Replicate label")
    legend = mo.ui.switch(value=True, label="Marker legend")
    # Top right corner of the legend, as fractions of the figure
    legend_x = mo.ui.slider(start=0.5, stop=1.0, step=0.005, value=1.0, label="Legend x", show_value=True)
    legend_y = mo.ui.slider(start=0.0, stop=1.0, step=0.005, value=0.97, label="Legend top", show_value=True)
    model_colour = mo.ui.text(value="", placeholder="theme", label="Model colour")
    accent_colour = mo.ui.text(value="", placeholder="theme", label="Accent colour")
    light_colour = mo.ui.text(value="#8fd0ff", label="Light cone colour")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([background, gamma, figure_width, font_size], justify="start"),
        mo.hstack([panels, same_scale, back_cards], justify="start"),
        mo.hstack([card_size, sheet_offset, stack_gap, replicate_label], justify="start"),
        mo.hstack([legend, legend_x, legend_y], justify="start"),
        mo.hstack([model_colour, accent_colour, light_colour], justify="start"),
    ])
    return (
        accent_colour,
        back_cards,
        background,
        card_size,
        figure_width,
        font_size,
        gamma,
        legend,
        legend_x,
        legend_y,
        light_colour,
        model_colour,
        panels,
        replicate_label,
        same_scale,
        sheet_offset,
        stack_gap,
    )


@app.cell(hide_code=True)
def _():
    stack_source = mo.ui.dropdown(options=["data", "toy"], value="data", label="Data stack (b)")
    stack_count = mo.ui.slider(start=1, stop=8, step=1, value=4, label="Sheets in data stack", show_value=True)
    toy_size = mo.ui.slider(start=12, stop=48, step=2, value=24, label="Toy grid (px)", show_value=True)
    # Rollout images over a fixed total number of toy steps
    toy_frames = mo.ui.slider(start=3, stop=12, step=1, value=7, label="Rollout images", show_value=True)
    toy_steps = mo.ui.slider(start=20, stop=600, step=10, value=160, label="Total toy steps", show_value=True)
    toy_seed = mo.ui.number(start=0, stop=1000, step=1, value=0, label="Seed")
    zoom_frame = mo.ui.slider(start=0, stop=6, step=1, value=2, label="Zoomed frame", show_value=True)
    # v1: zoom → pixel → time series; v2: two spaced 3D slices with an update cone
    training_layout = mo.ui.dropdown(options=["v1", "v2"], value="v1", label="Layout (b)")
    slice_gap = mo.ui.slider(start=0.6, stop=2.0, step=0.05, value=1.1, label="Slice spacing (v2)",
                             show_value=True)
    slice_depth = mo.ui.slider(start=0.3, stop=1.2, step=0.05, value=0.7, label="Slice depth (v2)",
                               show_value=True)
    packed_slices = mo.ui.slider(start=0, stop=6, step=1, value=3, label="Slices behind t (v2)",
                                 show_value=True)
    slices_transposed = mo.ui.switch(value=False, label="Time left to right (v2)")
    neighbourhood_colour = mo.ui.text(value="#00b300", label="Neighbourhood colour")
    mo.vstack([
        mo.md("**Panel b** (toy local rule)"),
        mo.hstack([training_layout, slices_transposed, slice_gap, slice_depth, packed_slices], justify="start"),
        mo.hstack([stack_source, stack_count, zoom_frame, neighbourhood_colour], justify="start"),
        mo.hstack([toy_size, toy_frames, toy_steps, toy_seed], justify="start"),
    ])
    return (
        neighbourhood_colour,
        packed_slices,
        slice_depth,
        slice_gap,
        slices_transposed,
        stack_count,
        stack_source,
        toy_frames,
        toy_seed,
        toy_size,
        toy_steps,
        training_layout,
        zoom_frame,
    )


@app.cell(hide_code=True)
def _():
    perturbation_hour = mo.ui.slider(start=0, stop=TIMEPOINTS[-1], step=1, value=TIMEPOINTS[-1] // 2,
                                     label="Perturbation time of the arrow (h)", show_value=True)
    ghost_hours = mo.ui.text(value="", placeholder="e.g. 12, 36", label="Faint perturbation arrows at (h)")
    measured_hours = mo.ui.text(value="0, 24", label="Measured perturbation times (h)")
    arrow_length = mo.ui.slider(start=3.0, stop=10.0, step=0.25, value=5.5, label="Arrow length",
                                show_value=True)
    curve_peak = mo.ui.slider(start=0, stop=TIMEPOINTS[-1], step=1, value=30, label="Mock curve peak (h)",
                              show_value=True)
    curve_width = mo.ui.slider(start=3, stop=40, step=1, value=12, label="Mock curve width (h)",
                               show_value=True)
    extra_curves = mo.ui.slider(start=0, stop=2, step=1, value=2, label="Other density curves",
                                show_value=True)
    density_labels = mo.ui.text(value="medium density, low density, high density",
                                label="Density curve labels (main first)", full_width=True)
    mo.vstack([
        mo.md("**Panel c**"),
        mo.hstack([perturbation_hour, ghost_hours, measured_hours, arrow_length], justify="start"),
        mo.hstack([curve_peak, curve_width, extra_curves], justify="start"),
        density_labels,
    ])
    return (
        arrow_length,
        curve_peak,
        curve_width,
        density_labels,
        extra_curves,
        ghost_hours,
        measured_hours,
        perturbation_hour,
    )


@app.cell(hide_code=True)
def _():
    labels = mo.ui.dictionary({
        "title_a": mo.ui.text(value="Imaging micropatterned colonies over time", label="Title a"),
        "title_b": mo.ui.text(value="Learning the patterning process from data", label="Title b"),
        "title_c": mo.ui.text(value="Probing perturbations at any time", label="Title c"),
        "imaging_label": mo.ui.text(value="confocal z-stacks", label="Imaging"),
        "data_label": mo.ui.text(value="micropattern data", label="Data stack"),
        "training_label": mo.ui.text(value="training", label="Training block"),
        "training_note": mo.ui.text(value="NCA", label="Training note"),
        "update_label": mo.ui.text(value="local pixelwise update", label="Update"),
        "equation": mo.ui.text(value="", placeholder=r"$x_{t+1} = x_t + f_\theta(\mathcal{N}(x_t))$",
                               label="Equation"),
        "series_label": mo.ui.text(value="model rollout at any time", label="Time series"),
        "initial_label": mo.ui.text(value="0h data", label="Initial"),
        "final_label": mo.ui.text(value="", placeholder="automatic", label="Final"),
        "rollout_label": mo.ui.text(value="model rollout", label="Rollout"),
        "perturbation_label": mo.ui.text(value="model\\nsignalling\\nperturbation", label="Perturbation block"),
        "perturbation_time_label": mo.ui.text(value=r"any $t$", label="Perturbation time"),
        "curve_title": mo.ui.text(value="cell type X at 48h", label="Curve title"),
        "curve_x_label": mo.ui.text(value="perturbation time", label="Curve x"),
        "curve_y_label": mo.ui.text(value="prevalence", label="Curve y"),
        "search_label": mo.ui.text(value="jointly optimise timing and density", label="Search"),
        "measured_label": mo.ui.text(value="measured perturbations", label="Measured times"),
        "density_label": mo.ui.text(value="initial\\ncell density", label="Density box"),
    })
    mo.vstack([mo.md("**Labels** (`\\n` for a new line, `$...$` for maths)"),
               mo.hstack(list(labels.values()), wrap=True, justify="start")])
    return (labels,)


@app.cell(hide_code=True)
def _(toy_frames, toy_seed, toy_size, toy_steps):
    _per_frame = max(1, round(int(toy_steps.value) / (int(toy_frames.value) - 1)))
    toy = toy_patterning(int(toy_size.value), int(toy_frames.value), _per_frame, int(toy_seed.value))
    return (toy,)


@app.cell(hide_code=True)
def _(
    accent_colour,
    arrow_length,
    back_cards,
    background,
    card_size,
    condition,
    curve_peak,
    curve_width,
    data,
    density_labels,
    extra_curves,
    figure_width,
    final_condition,
    final_replicate,
    font_size,
    gamma,
    ghost_hours,
    labels,
    legend,
    legend_x,
    legend_y,
    light_colour,
    measured_hours,
    model_colour,
    neighbourhood_colour,
    packed_slices,
    panels,
    perturbation_hour,
    projection,
    replicate_label,
    same_scale,
    selected_groups,
    selected_replicates,
    selected_times,
    sheet_offset,
    slice_depth,
    slice_gap,
    slices_transposed,
    stack_count,
    stack_gap,
    stack_source,
    toy,
    training_layout,
    zoom_frame,
):
    mo.stop(not panels.value, mo.callout("Select at least one panel.", kind="warn"))
    _theme = dict(ko_drawing.THEMES[background.value])
    _theme["accent"] = colour_or(accent_colour.value, _theme["accent"])
    _text = {k: v.replace("\\n", "\n") for k, v in labels.value.items()}
    _final = _text["final_label"] or f"{TIMEPOINTS[-1]}h data\n({CONDITION_LABELS[final_condition.value]})"
    style = {
        "theme": _theme,
        "font": float(font_size.value),
        "width": float(figure_width.value),
        "gamma": float(gamma.value),
        "same_scale": same_scale.value,
        "groups": selected_groups,
        "times": selected_times,
        "replicates": selected_replicates,
        "condition": condition.value,
        "condition_note": f"{CONDITION_LABELS[condition.value]}, {PROJECTION_LABELS[projection.value]}",
        "back_cards": back_cards.value,
        "card_size": float(card_size.value),
        "sheet_offset": float(sheet_offset.value),
        "stack_gap": float(stack_gap.value),
        "replicate_label": replicate_label.value,
        "legend": legend.value,
        "legend_anchor": (float(legend_x.value), float(legend_y.value)),
        "light_colour": colour_or(light_colour.value, "#8fd0ff"),
        "model_colour": colour_or(model_colour.value, _theme["model"]),
        "stack_source": stack_source.value,
        "stack_count": int(stack_count.value),
        "zoom_frame": int(zoom_frame.value),
        "training_layout": training_layout.value,
        "slice_gap": float(slice_gap.value),
        "slice_depth": float(slice_depth.value),
        "slices_transposed": slices_transposed.value,
        "packed_slices": int(packed_slices.value),
        "neighbourhood_colour": colour_or(neighbourhood_colour.value, "#00b300"),
        "final_condition": final_condition.value,
        "final_replicate": int(final_replicate.value),
        "perturbation_hour": float(perturbation_hour.value),
        "ghost_hours": parse_hours(ghost_hours.value),
        "measured_hours": parse_hours(measured_hours.value),
        "arrow_length": float(arrow_length.value),
        "curve_peak": float(curve_peak.value),
        "curve_width": float(curve_width.value),
        "extra_curves": int(extra_curves.value),
        "density_labels": density_labels.value,
        "titles": {"a": _text["title_a"], "b": _text["title_b"], "c": _text["title_c"]},
        "final_label": _final,
        **{k: v for k, v in _text.items() if not k.startswith("title_") and k != "final_label"},
    }
    overview = overview_figure(sorted(panels.value), data, toy, style)
    overview
    return (overview,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="project_overview", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_figure], justify="start")
    return figure_name, save_figure, save_svg


@app.cell
def _(figure_name, overview, save_figure, save_svg):
    mo.stop(not save_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    _suffixes = ("pdf", "png", "svg") if save_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            overview.savefig(_directory / f"{figure_name.value}.{_suffix}", dpi=400,
                             facecolor=overview.get_facecolor())
    mo.md(f"Saved `{figure_name.value}` as " + ", ".join(_suffixes) + f" to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()
