"""Drawing helpers shared by the micropattern figure notebooks.

Colours, fonts and line widths come from ``style.py``. Call these inside
``with paper_style():`` so the matplotlib settings match.
"""

import io
import re

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D

from Experiments.micropatterns.figures.data import crop_box
from Experiments.micropatterns.figures.style import (
    ACCENT,
    BAND_ALPHA,
    FONT,
    GROUP_LABELS,
    IMAGE_BG,
    INK,
    LINE,
    MUTED,
    PAGE,
    SOURCE_COLOURS,
    SOURCE_LABELS,
    UNMEASURED,
    ZOOM_OUTLINE,
    SectorHandle,
    figure_width,
    group_colours,
    label_panel,
    legend_options,
    legend_swatch,
    sector_angles,
    unmeasured_patch,
)

# --- Images --------------------------------------------------------------------


def composite(channels, colours):
    """RGB image [..., 3] on black from normalised channels [C, ...], one colour (or None) per channel."""
    rgb = np.zeros(channels.shape[1:] + (3,))
    for channel, colour in zip(channels, colours):
        if colour is not None:
            rgb += channel[..., None] * np.asarray(colour)
    return np.clip(rgb, 0.0, 1.0)


def colony_rgba(image, mask, colours, gamma=1.0):
    """RGBA of a [C, H, W] image in [0, 1], cropped to its colony mask and transparent outside it."""
    crop = crop_box(mask)
    rgb = composite(np.clip(image[:, crop[0], crop[1]], 0.0, 1.0) ** gamma, colours)
    return np.concatenate([rgb, mask[crop][..., None].astype(float)], axis=-1)


def sector_map(shape, count):
    """Sector index [H, W] of each pixel of an image drawn with row 0 at the top."""
    rows, cols = np.indices(shape)
    angle = np.degrees(np.arctan2((shape[0] - 1) / 2 - rows, cols - (shape[1] - 1) / 2))
    return (np.mod(angle - 90.0, 360.0) // (360.0 / count)).astype(int).clip(0, count - 1)


def sliced_rgba(group_rgbas):
    """One RGBA [H, W, 4] taking sector k from the k-th RGBA of ``group_rgbas``."""
    stacked = np.stack(group_rgbas)
    index = sector_map(stacked.shape[1:3], len(group_rgbas))
    return np.take_along_axis(stacked, index[None, ..., None], axis=0)[0]


def placeholder_rgba(size=64):
    """Grey disc, drawn where no data is loaded."""
    rows, cols = np.indices((size, size))
    inside = np.hypot(rows - (size - 1) / 2, cols - (size - 1) / 2) <= size / 2
    rgba = np.zeros((size, size, 4))
    rgba[..., :3] = 0.35
    rgba[..., 3] = inside
    return rgba


def grey_rgba(values, mask):
    """RGBA of one channel in [0, 1], white on black, transparent outside ``mask``."""
    rgb = np.repeat(np.clip(values, 0, 1)[..., None], 3, axis=-1)
    return np.concatenate([rgb, mask[..., None].astype(float)], axis=-1)


def show_png(fig, dpi=110):
    """PNG bytes of ``fig`` (closed afterwards), for quick previews in notebooks."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


# --- Colony discs ----------------------------------------------------------------


def draw_colony(ax, rgba, centre, size, alpha=1.0, edge=None):
    """Colony disc of diameter ``size`` at ``centre`` (data units), on black."""
    x, y = centre
    faded = rgba.copy()
    faded[..., 3] *= alpha
    ax.add_patch(Circle(centre, size / 2, facecolor=IMAGE_BG, edgecolor="none", alpha=alpha, zorder=3))
    ax.imshow(faded, extent=(x - size / 2, x + size / 2, y - size / 2, y + size / 2), origin="upper",
              interpolation="antialiased", zorder=4)
    ax.add_patch(Circle(centre, size / 2, fill=False, edgecolor=edge or MUTED, lw=LINE["axis"], alpha=alpha,
                        zorder=5))


def draw_placeholder(ax, centre, size, text, colour=None):
    """Dashed empty disc for a colony that was not measured (or is unknown)."""
    colour = colour or MUTED
    ax.add_patch(Circle(centre, size / 2, facecolor=PAGE, edgecolor=colour, lw=LINE["axis"], ls=(0, (2, 2)),
                        zorder=3))
    ax.text(*centre, text, ha="center", va="center", color=colour, fontsize=FONT["label"], zorder=5)


def draw_sectors(ax, centre, radius, unmeasured, hatch=True, z=4.5, alpha=1.0):
    """Hatch the sectors of a disc that were not measured, and divide the sectors with white lines.

    ``unmeasured`` holds one flag per sector. Returns True if any sector was hatched.
    """
    hatched = False
    count = len(unmeasured)
    for k, missing in enumerate(unmeasured):
        start, stop = sector_angles(k, count)
        if missing and hatch:
            hatched = True
            ax.add_patch(Wedge(centre, radius, start, stop, facecolor=UNMEASURED["face"],
                               edgecolor=UNMEASURED["edge"], hatch=UNMEASURED["hatch"], lw=0, alpha=alpha,
                               zorder=z))
        if count > 1:
            angle = np.radians(start)
            ax.plot([centre[0], centre[0] + radius * np.cos(angle)], [centre[1], centre[1] + radius * np.sin(angle)],
                    color="white", lw=LINE["axis"], alpha=alpha, zorder=z + 0.5)
    return hatched


def trajectory_tile(rollout, source, i, groups, gamma):
    """RGBA disc of time index i with one sector per group, and which sectors were not measured."""
    names = rollout["channel_names"]
    rgbas = [colony_rgba(rollout[source][i], rollout["boundary"], group_colours(names, g), gamma) for g in groups]
    unmeasured = [
        source == "target" and not any(f for n, f in zip(names, rollout["measured"][i]) if n.split("/")[0] == g)
        for g in groups
    ]
    return sliced_rgba(rgbas), unmeasured


def marker_legend_entries(names, groups, sectors=(), kind="white"):
    """Legend rows (one per group): the group's sector (if it is one of ``sectors``) and name,
    then a swatch per marker shown for that group. Returns a list of [(handle, label), ...]."""
    blank = plt.Line2D([], [], lw=0)
    sectors = list(sectors)
    return [
        [(SectorHandle(sectors.index(g), len(sectors)) if g in sectors and len(sectors) > 1 else blank,
          GROUP_LABELS.get(g, g))]
        + [(legend_swatch(c, kind), n.rsplit("/", 1)[-1]) for n, c in zip(names, group_colours(names, g))
           if c is not None]
        for g in groups
    ]


def rows_to_columns(rows):
    """Handles and labels of legend ``rows`` for ``legend(ncol=...)``, which fills columns top to bottom.

    Returns handles, labels and the column count.
    """
    blank = plt.Line2D([], [], lw=0)
    length = max(len(r) for r in rows)
    padded = [r + [(blank, "")] * (length - len(r)) for r in rows]
    entries = [r[c] for c in range(length) for r in padded]
    return [h for h, _ in entries], [label for _, label in entries], length


# --- Schematic shapes --------------------------------------------------------------


def draw_arrow(ax, start, end, colour=INK, lw=1.0, **kwargs):
    """Plain arrow from ``start`` to ``end`` (data units)."""
    ax.annotate("", xy=end, xytext=start, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9,
                                **kwargs))


def polyline_arrow(ax, points, colour=INK, lw=0.9, **kwargs):
    """Line through ``points`` with an arrow head on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, solid_capstyle="butt", **kwargs)
    draw_arrow(ax, points[-2], points[-1], colour, lw=lw, **kwargs)


def rounded_box(ax, corner, width, height, colour, z=2, rounding=0.1):
    """Filled (faint) and outlined rounded box."""
    for kwargs in (dict(facecolor=colour, alpha=0.15, edgecolor="none", zorder=z),
                   dict(fill=False, edgecolor=colour, lw=1.0, zorder=z + 0.1)):
        ax.add_patch(FancyBboxPatch(corner, width, height, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                    **kwargs))


def draw_zoom_outline(ax, corner, size, dashes=None, z=6, height=None, colour=None, lw=None, **kwargs):
    """Outline of a zoom box: black dashes over a solid white line (``ZOOM_OUTLINE``),
    so it shows on both light pages and dark images.

    ``size`` is the width (and the height, unless ``height`` is given). A ``colour``
    draws a solid line of that colour instead. Other keywords go to the patches,
    e.g. ``transform=ax.transAxes, clip_on=False`` to frame a whole axes.
    """
    lw = ZOOM_OUTLINE["width"] if lw is None else lw
    if colour is not None:
        lines = ((colour, "-"),)
    else:
        background, foreground = ZOOM_OUTLINE["colours"]
        lines = ((background, "-"), (foreground, (0, dashes or ZOOM_OUTLINE["dashes"])))
    for line_colour, linestyle in lines:
        ax.add_patch(Rectangle(corner, size, size if height is None else height, fill=False,
                               edgecolor=line_colour, lw=lw, linestyle=linestyle, zorder=z, **kwargs))


def frame_zoom_axes(ax, colour=None, lw=None):
    """Frame a whole axes (a magnified view) with the zoom box outline instead of its spines."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_zoom_outline(ax, (0, 0), 1, z=10, colour=colour, lw=lw, transform=ax.transAxes, clip_on=False)


def draw_card(ax, corner, size, z, rgba=None, unmeasured=(), alpha=1.0, edge=None, pixelated=False):
    """One microscopy image: a black square with the colony disc in it.

    Hatched sectors were not measured; ``z`` sets the drawing order of the card."""
    x, y = corner
    ax.add_patch(Rectangle(corner, size, size, facecolor=IMAGE_BG, edgecolor=edge or MUTED, lw=LINE["axis"],
                           zorder=z))
    if rgba is None:
        return
    disc = 0.9 * size
    cx, cy = x + size / 2, y + size / 2
    shown = rgba.copy()
    shown[..., 3] *= alpha
    ax.imshow(shown, extent=(cx - disc / 2, cx + disc / 2, cy - disc / 2, cy + disc / 2), origin="upper",
              interpolation="nearest" if pixelated else "antialiased", zorder=z + 0.1)
    draw_sectors(ax, (cx, cy), disc / 2, unmeasured, z=z + 0.2, alpha=alpha)


def draw_paper_stack(ax, tiles, corner, size, offset, back_cards="faded", pixelated=False):
    """Sheets drawn back to front, each shifted by ``offset`` from the one in front.

    ``tiles`` holds (rgba, unmeasured) per sheet, front first. Only the front sheet is
    clearly visible; ``back_cards`` ("blank", "faded" or "full") sets the others."""
    count = len(tiles)
    for k in reversed(range(count)):
        rgba, unmeasured = tiles[k]
        back = k > 0
        if back and back_cards == "blank":
            rgba, unmeasured = None, ()
        alpha = 0.4 if back and back_cards == "faded" else 1.0
        draw_card(ax, (corner[0] + k * offset[0], corner[1] + k * offset[1]), size, 10 + count - k, rgba,
                  unmeasured, alpha, pixelated=pixelated)


def draw_time_axis(ax, times, y, size, label="{:g}h"):
    """Time axis above a row of discs of diameter ``size`` (x in hours)."""
    ax.plot([times[0] - size / 2, times[-1] + size / 2], [y, y], color=INK, lw=LINE["axis"])
    for t in times:
        ax.plot([t, t], [y, y + 0.8], color=INK, lw=LINE["axis"])
        ax.text(t, y + 1.2, label.format(t), ha="center", va="bottom", fontsize=FONT["tick"])


def add_length_marker(ax, start, end, label, xycoords="data", offset=(0, -3), rotation=0, ha="center", va="top"):
    """Double-headed arrow from ``start`` to ``end`` with a label at its middle."""
    ax.annotate("", xy=end, xytext=start, xycoords=xycoords, textcoords=xycoords, annotation_clip=False,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=LINE["axis"], shrinkA=0, shrinkB=0))
    middle = tuple((a + b) / 2 for a, b in zip(start, end))
    ax.annotate(label, xy=middle, xycoords=xycoords, xytext=offset, textcoords="offset points", ha=ha, va=va,
                rotation=rotation, rotation_mode="anchor", fontsize=FONT["annotation"], color=INK,
                annotation_clip=False)


def add_scale_bar(ax, length_px, label, shape):
    """White scale bar in the lower right of an image (on black) of the given [H, W] shape."""
    height, width = shape
    pad = 0.05 * width
    ax.add_patch(Rectangle((width - pad - length_px, height - pad - 0.012 * height), length_px,
                           0.012 * height, color="white", linewidth=0))
    ax.text(width - pad - length_px / 2, height - pad - 0.03 * height, label, color="white",
            ha="center", va="bottom", fontsize=FONT["annotation"])


# --- Text size in data units ---------------------------------------------------------

# Points per data unit of the schematic panels when they are not measured (see stack_panels)
DEFAULT_PT_PER_UNIT = 45.0


def text_width(text, size, pt_per_unit=DEFAULT_PT_PER_UNIT, bold=False):
    """Rough width of ``text`` at ``size`` points in data units (maths markup counted as its symbols)."""

    def visible(line):
        line = re.sub(r"\\[a-zA-Z]+", "x", line)
        return len(re.sub(r"[${}_^]", "", line))

    return (0.64 if bold else 0.58) * size / pt_per_unit * max(visible(line) for line in text.split("\n"))


def text_height(text, size, pt_per_unit=DEFAULT_PT_PER_UNIT, spacing=1.25):
    """Height of ``text`` at ``size`` points in data units."""
    return size / pt_per_unit * spacing * (text.count("\n") + 1)


# --- Panels ------------------------------------------------------------------------


def stack_panels(draw, panels, width="double", titles=None, same_scale=True, measure_scale=False, letters=True):
    """Schematic panels stacked top to bottom, each an equal-aspect axes without axis lines.

    ``draw`` maps a panel key to a function drawing it on an axes. With
    ``measure_scale`` the function also gets the panel's points per data unit,
    and the panel is redrawn until that value settles (for text-heavy panels).
    ``same_scale`` gives every panel the same data units per inch. Panels are
    labelled with their key (left out with ``letters=False``) and ``titles[key]``.
    """
    width = figure_width(width)
    titles = titles or {}
    fig = plt.figure(figsize=(width, 2.0), layout="constrained")
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, key in zip(axes, panels):
        if measure_scale:
            pt_per_unit = 40.0
            for _ in range(12):
                ax.cla()
                draw[key](ax, pt_per_unit)
                measured = 0.97 * width * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
                if abs(measured - pt_per_unit) < 0.01 * measured:
                    break
                # Damped, as text-heavy panels widen when their text grows
                pt_per_unit = 0.5 * (pt_per_unit + measured)
        else:
            draw[key](ax)
        ax.set_aspect("equal")
        ax.axis("off")
        label_panel(ax, key if letters else None, titles.get(key) or None, offset=(0, 0))
    widths = [ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes]
    if same_scale:
        for ax in axes:
            ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max(widths))
    ratios = [(ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) for ax in axes]
    axes[0].get_subplotspec().get_gridspec().set_height_ratios(ratios)
    fig.set_size_inches(width, width * sum(ratios) + 0.15 * len(panels))
    return fig


# --- 3D stack ----------------------------------------------------------------------


def plane_transform(ax, height, shear):
    """Maps the unit square (image coordinates) to a tilted plane at ``height``."""
    sx, sy = shear
    return Affine2D.from_values(1.0, 0.0, sx, sy, 0.0, height) + ax.transData


def draw_stack(ax, planes, heights, highlight, opacity, column, shear=(0.55, 0.3), accent=None,
               column_colour="white"):
    """Oblique view of RGBA planes stacked at ``heights``, pierced by an exaggerated column of voxels.

    ``highlight`` is the index of the plane outlined (in ``accent``) as the slice. ``column`` holds
    the probe position (x, y as fractions of the image), its width (in plane
    widths), the voxel RGB colours [Z, 3], the voxel heights [Z] and the index
    of the slice voxel.
    """
    accent = accent or ACCENT
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for k, (rgba, h) in enumerate(zip(planes, heights)):
        transform = plane_transform(ax, h, shear)
        zorder = 10 + 3 * k
        ax.add_patch(Polygon(unit_square, closed=True, transform=transform, facecolor=IMAGE_BG,
                             alpha=opacity, edgecolor="none", zorder=zorder))
        ax.imshow(rgba, extent=(0, 1, 0, 1), origin="upper", transform=transform,
                  interpolation="antialiased", zorder=zorder + 1)
        is_slice = k == highlight
        ax.add_patch(Polygon(unit_square, closed=True, transform=transform, fill=False,
                             edgecolor=accent if is_slice else MUTED,
                             linewidth=LINE["emphasis"] if is_slice else LINE["thin"], zorder=zorder + 2))

    # Each voxel is drawn between the plane below it and the plane above it, so
    # the column passes through the stack. Only the front and right faces show.
    (px, py), width, colours, voxel_heights, voxel_highlight = column
    sx, sy = shear
    u, v = px - width / 2, 1.0 - py - width / 2
    x0, y0 = u + sx * v, sy * v
    side_dx, side_dy = sx * width, sy * width
    cell = voxel_heights[1] - voxel_heights[0] if len(voxel_heights) > 1 else 0.05
    edge = MUTED if len(colours) <= 40 else "none"
    for z, (colour, h) in enumerate(zip(colours, voxel_heights)):
        zorder = 10 + 3 * (np.searchsorted(heights, h, side="right") - 1) + 2.5
        bottom = y0 + h - cell / 2
        front = [(x0, bottom), (x0 + width, bottom), (x0 + width, bottom + cell), (x0, bottom + cell)]
        side = [(x0 + width, bottom), (x0 + width + side_dx, bottom + side_dy),
                (x0 + width + side_dx, bottom + cell + side_dy), (x0 + width, bottom + cell)]
        ax.add_patch(Polygon(front, facecolor=colour, edgecolor=edge, lw=LINE["hairline"], zorder=zorder))
        ax.add_patch(Polygon(side, facecolor=0.6 * np.asarray(colour), edgecolor=edge, lw=LINE["hairline"],
                             zorder=zorder))
        if z == voxel_highlight:
            for face in (front, side):
                ax.add_patch(Polygon(face, fill=False, edgecolor=accent, lw=1.0, zorder=zorder + 0.1))

    # Outline: the three vertical edges, drawn in pieces between planes so that
    # planes above still cover it, plus the bottom edges and the top face
    orders = 10 + 3 * (np.searchsorted(heights, voxel_heights, side="right") - 1) + 2.6
    for order in np.unique(orders):
        hs = voxel_heights[orders == order]
        y_low, y_high = y0 + hs[0] - cell / 2, y0 + hs[-1] + cell / 2
        for x, dy in ((x0, 0.0), (x0 + width, 0.0), (x0 + width + side_dx, side_dy)):
            ax.plot([x, x], [y_low + dy, y_high + dy], color=column_colour, lw=0.7, zorder=order,
                    solid_capstyle="butt")
    base = y0 + voxel_heights[0] - cell / 2
    ax.plot([x0, x0 + width, x0 + width + side_dx], [base, base, base + side_dy], color=column_colour, lw=0.7,
            zorder=orders[0])
    top = y0 + voxel_heights[-1] + cell / 2
    top_face = [(x0, top), (x0 + width, top), (x0 + width + side_dx, top + side_dy), (x0 + side_dx, top + side_dy)]
    ax.add_patch(Polygon(top_face, facecolor=0.8 * np.asarray(colours[-1]), edgecolor=column_colour, lw=0.7,
                         zorder=10 + 3 * len(heights)))

    ax.set_xlim(-0.14, 1 + sx + 0.14)
    ax.set_ylim(-0.14, heights[-1] + sy + 0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    return (x0 + width / 2 + side_dx / 2, top + side_dy)


def draw_profile(ax, profile, z_um, mode, z_slice_um, colours, names, accent=None):
    """Intensity against depth at the probe, marking what each projection keeps."""
    for c, colour in enumerate(colours):
        if colour is None:
            continue
        values = profile[:, c]
        ax.plot(values, z_um, color=colour, lw=LINE["data"], label=names[c])
        marker = dict(ms=3.5, color=colour, markeredgecolor=INK, markeredgewidth=0.4)
        if mode == "slice":
            ax.plot(np.interp(z_slice_um, z_um, values), z_slice_um, "o", **marker)
        elif mode == "max":
            peak = int(np.argmax(values))
            ax.plot(values[peak], z_um[peak], "o", **marker)
            ax.axvline(values[peak], color=colour, lw=LINE["axis"], ls=":")
        elif mode == "sum":
            ax.fill_betweenx(z_um, 0, values, color=colour, alpha=BAND_ALPHA, linewidth=0)
    if mode == "slice":
        ax.axhline(z_slice_um, color=accent or ACCENT, lw=LINE["data"])
    ax.set_xlim(left=0)
    ax.set_ylim(z_um[0], z_um[-1])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_xticks([])


# --- Trajectory figure ---------------------------------------------------------------


def trajectory_figure(rollouts, blocks, groups, gamma, colony_size, hatch, zoom=None, width="double",
                      legend_kind="white"):
    """Trajectories as rows of colony discs (one sector per group), with a bracketed label per block.

    ``blocks`` are (label, colour, rows), and each row is (rollout key, source,
    row label, row label colour). Positions are in hours of the time axis.
    ``zoom``, if given, adds magnified squares right of the trajectories: a dict
    with ``time`` (index), ``centres`` ((x, y) as fractions of the disc, y down),
    ``width`` (fraction of the disc) and ``title``. Boxes have the striped zoom
    outline, numbered when there are several; ``colours`` (one per centre) gives
    solid coloured outlines instead.
    With ``pixels`` ({rollout key: n}) each box is instead n x n whole pixels of
    that rollout, on its pixel grid. The box is placed on the coarsest grid (the
    smallest n), so every rollout shows the same region; finer n must be whole
    multiples of it, and the grids must share their top-left corner (as when
    downsampling). ``legend_kind`` is "white" or "black".
    """
    width = figure_width(width)
    first = next(iter(rollouts.values()))
    names = list(first["channel_names"])
    times = [float(t.rstrip("h")) for t in first["time_labels"]]
    s, text = colony_size, FONT["label"] / 7.0
    row_step, block_gap = s + 1.5, 3.0
    placed, y = [], 0.0
    for label, colour, rows in blocks:
        ys = [y - k * row_step for k in range(len(rows))]
        placed.append((label, colour, list(zip(rows, ys))))
        y = ys[-1] - s - block_gap
    label_lines = max(label.count("\n") + 1 for label, _, _ in blocks)
    left = times[0] - s / 2 - 6.0 * text - 1.8 * text * (label_lines - 1) - 0.5
    zoom_x, zoom_colours = [], []
    if zoom is not None:
        zoom_x = [times[-1] + s + 3.0 + k * (s + 1.5) for k in range(len(zoom["centres"]))]
        zoom_colours = zoom.get("colours") or [None] * len(zoom["centres"])
    # Several striped boxes look alike, so they are numbered
    numbered = len(zoom_x) > 1 and zoom_colours[0] is None
    right = (zoom_x[-1] if zoom_x else times[-1]) + s / 2 + 2.0
    bottom = placed[-1][2][-1][1] - s / 2 - 1.0
    axis_y = s / 2 + 1.5
    top = axis_y + 3.0 * text
    unit = width / (right - left)  # inches per hour
    legend_height = 0.2 * (len(groups) + 1) + 0.15  # inches
    panel_height = unit * (top - bottom)
    hatched = False

    def crop_corner_and_shape(key):
        """Top-left pixel and (rows, cols) of the colony crop drawn for rollout ``key``."""
        crop = crop_box(rollouts[key]["boundary"])
        return (crop[0].start, crop[1].start), (crop[0].stop - crop[0].start, crop[1].stop - crop[1].start)

    def zoom_box(zx, zy, key):
        """(left, top, width, height) of a zoom box, as fractions of the disc image."""
        if zoom.get("pixels") is None:
            return zx - zoom["width"] / 2, zy - zoom["width"] / 2, zoom["width"], zoom["width"]
        # Snap once on the grid of the coarsest rollout (fewest pixels), with the
        # centre as a fraction of its colony crop, then take the same region on
        # every finer grid. Each colony is cropped separately, so the crop offsets
        # are needed to line the grids up.
        coarse = min(zoom["pixels"], key=zoom["pixels"].get)
        n = zoom["pixels"][coarse]
        (top_c, left_c), (rows_c, cols_c) = crop_corner_and_shape(coarse)
        col = int(np.clip(np.floor(zx * cols_c - n / 2 + 0.5), 0, max(cols_c - n, 0)))
        row = int(np.clip(np.floor(zy * rows_c - n / 2 + 0.5), 0, max(rows_c - n, 0)))
        scale = zoom["pixels"][key] // n  # finer pixels per coarse pixel
        (top, left), (rows, cols) = crop_corner_and_shape(key)
        col, row = (left_c + col) * scale - left, (top_c + row) * scale - top
        return col / cols, row / rows, n * scale / cols, n * scale / rows

    figure = plt.figure(figsize=(width, panel_height + legend_height))
    ax = figure.add_axes((0.0, legend_height / (panel_height + legend_height), 1.0,
                          panel_height / (panel_height + legend_height)))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_time_axis(ax, times, axis_y, s)
    if zoom_x:
        ax.plot([zoom_x[0] - s / 2, zoom_x[-1] + s / 2], [axis_y] * 2, color=INK, lw=LINE["axis"])
        ax.text((zoom_x[0] + zoom_x[-1]) / 2, axis_y + 1.2, zoom["title"], ha="center", va="bottom",
                fontsize=FONT["tick"])
        for k, x in enumerate(zoom_x if numbered else []):
            ax.text(x, axis_y - 0.4, str(k + 1), ha="center", va="top", fontsize=FONT["tick"])

    x_row = times[0] - s / 2 - text
    x_bracket = x_row - 2.4 * text
    for label, colour, rows in placed:
        for (key, source, row_label, row_colour), row_y in rows:
            ax.text(x_row, row_y, row_label, rotation=90, ha="right", va="center", fontsize=FONT["annotation"],
                    color=row_colour)
            rollout = rollouts[key]
            for i, t in enumerate(times[:len(rollout[source])]):
                centre = (t, row_y)
                rgba, unmeasured = trajectory_tile(rollout, source, i, groups, gamma)
                draw_colony(ax, rgba, centre, s)
                hatched |= draw_sectors(ax, centre, s / 2, unmeasured, hatch=hatch)
                if zoom is None or i != zoom["time"]:
                    continue
                # Boxes on the disc, and the disc magnified so each box fills its
                # square, with nearest-pixel drawing so the grid of the data shows
                for k, ((zx, zy), zoom_colour, x) in enumerate(zip(zoom["centres"], zoom_colours, zoom_x)):
                    fx, fy, fw, fh = zoom_box(zx, zy, key)
                    box_left, box_low = t + (fx - 0.5) * s, row_y + (0.5 - fy - fh) * s
                    draw_zoom_outline(ax, (box_left, box_low), fw * s, height=fh * s, colour=zoom_colour, z=6)
                    if numbered:
                        ax.text(box_left + fw * s + 0.15, box_low + fh * s, str(k + 1), ha="left", va="top",
                                fontsize=FONT["tick"], color="white", zorder=7,
                                path_effects=[patheffects.withStroke(linewidth=1.2, foreground="black")])
                    tile = Rectangle((x - s / 2, row_y - s / 2), s, s, transform=ax.transData)
                    ax.add_patch(Rectangle((x - s / 2, row_y - s / 2), s, s, facecolor=IMAGE_BG, lw=0, zorder=3))
                    image_left = x - s / 2 - fx * s / fw
                    image_top = row_y + s / 2 + fy * s / fh
                    image = ax.imshow(rgba, extent=(image_left, image_left + s / fw, image_top - s / fh, image_top),
                                      origin="upper", interpolation="nearest", zorder=4)
                    image.set_clip_path(tile)
                    draw_zoom_outline(ax, (x - s / 2, row_y - s / 2), s, colour=zoom_colour, lw=1.0, z=6)
        ys = [row_y for _, row_y in rows]
        ax.plot([x_bracket, x_bracket], [ys[-1] - s / 2 + 0.5, ys[0] + s / 2 - 0.5], color=colour, lw=LINE["axis"])
        ax.text(x_bracket - 0.6 * text, (ys[0] + ys[-1]) / 2, label, rotation=90, ha="right", va="center",
                multialignment="center", fontsize=FONT["label"], color=colour)
    # Limits again, since imshow resets them
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    add_group_legend(figure, names, groups, hatched, legend_kind)
    return figure


def add_group_legend(figure, names, groups, hatched, legend_kind="white"):
    """Legend below the figure: a row per group with its sector, then its channel colours."""
    legend_rows = marker_legend_entries(names, groups, sectors=groups, kind=legend_kind)
    if hatched:
        legend_rows.append([(unmeasured_patch(), "not measured")])
    handles, labels, ncol = rows_to_columns(legend_rows)
    figure.legend(handles, labels, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                  **legend_options(legend_kind))


def draw_disc_grid(ax, rollouts, blocks, columns, groups, gamma, colony_size, hatch, time_label, title=None):
    """Draw the discs of ``grid_figure`` on ``ax`` (see there for the arguments).

    Returns the disc centres along x, the (left, right, bottom, top) data limits
    and whether any sector was hatched.
    """
    s, text = colony_size, FONT["label"] / 7.0
    step, block_gap = s + 1.5, 3.0
    xs = [k * step for k in range(len(columns))]
    placed, y = [], 0.0
    for label, colour, rows in blocks:
        ys = [y - k * step for k in range(len(rows))]
        placed.append((label, colour, list(zip(rows, ys))))
        y = ys[-1] - s - block_gap
    label_lines = max(label.count("\n") + 1 for label, _, _ in blocks)
    left = xs[0] - s / 2 - 6.0 * text - 1.8 * text * (label_lines - 1) - 0.5
    right = xs[-1] + s / 2 + 1.0
    bottom = placed[-1][2][-1][1] - s / 2 - 1.0
    header_y = s / 2 + 0.8
    top = header_y + (3.5 if title else 1.5) * text

    ax.set_aspect("equal")
    ax.axis("off")
    for x, (_, column_label) in zip(xs, columns):
        ax.text(x, header_y, column_label, ha="center", va="bottom", fontsize=FONT["tick"])
    if title:
        ax.text((xs[0] + xs[-1]) / 2, header_y + 1.6 * text, title, ha="center", va="bottom",
                fontsize=FONT["tick"])

    hatched = False
    x_row = xs[0] - s / 2 - text
    x_bracket = x_row - 2.4 * text
    for label, colour, rows in placed:
        for (row_key, source, row_label, row_colour), row_y in rows:
            ax.text(x_row, row_y, row_label, rotation=90, ha="right", va="center", fontsize=FONT["annotation"],
                    color=row_colour)
            for x, (column_key, _) in zip(xs, columns):
                rollout = rollouts.get((row_key, column_key))
                if rollout is None or time_label not in rollout["time_labels"]:
                    draw_placeholder(ax, (x, row_y), s, "no model" if rollout is None else f"no {time_label}")
                    continue
                rgba, unmeasured = trajectory_tile(rollout, source, rollout["time_labels"].index(time_label),
                                                   groups, gamma)
                draw_colony(ax, rgba, (x, row_y), s)
                hatched |= draw_sectors(ax, (x, row_y), s / 2, unmeasured, hatch=hatch)
        ys = [row_y for _, row_y in rows]
        ax.plot([x_bracket, x_bracket], [ys[-1] - s / 2 + 0.5, ys[0] + s / 2 - 0.5], color=colour, lw=LINE["axis"])
        ax.text(x_bracket - 0.6 * text, (ys[0] + ys[-1]) / 2, label, rotation=90, ha="right", va="center",
                multialignment="center", fontsize=FONT["label"], color=colour)
    # Limits last, since imshow resets them
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    return xs, (left, right, bottom, top), hatched


def grid_figure(rollouts, blocks, columns, groups, gamma, colony_size, hatch, time_label, title=None,
                width="double", legend_kind="white"):
    """Colony discs at one timepoint, in a grid of rows and columns, with a bracketed label per block.

    ``rollouts`` are keyed by (row key, column key). ``blocks`` are (label, colour,
    rows), and each row is (row key, source, row label, row label colour), as in
    ``trajectory_figure``. ``columns`` are (column key, column label). Each disc is
    the rollout's frame labelled ``time_label`` (e.g. "48h"); cells without a
    rollout get a dashed "no model" circle. ``title`` goes above the column labels.
    """
    width = figure_width(width)
    names = list(next(iter(rollouts.values()))["channel_names"])
    figure = plt.figure()
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    _, (left, right, bottom, top), hatched = draw_disc_grid(
        ax, rollouts, blocks, columns, groups, gamma, colony_size, hatch, time_label, title)
    legend_height = 0.2 * (len(groups) + 1) + 0.15  # inches
    panel_height = width / (right - left) * (top - bottom)
    total = panel_height + legend_height
    figure.set_size_inches(width, total)
    ax.set_position((0.0, legend_height / total, 1.0, panel_height / total))
    add_group_legend(figure, names, groups, hatched, legend_kind)
    return figure


def snapshot_loss_figure(rollouts, columns, groups, gamma, colony_size, hatch, time_label, losses, loss_label,
                         title=None, width="double", legend_kind="white", bar_height=1.3):
    """Discs of true data and NCA at one timepoint per column, above a bar chart of losses.

    ``rollouts`` (one model per column) and ``losses`` are keyed by the column keys
    of ``columns`` [(column key, column label)]. ``losses[key]`` is
    ``{"train": [...], "validation": [...]}`` with one value per model; each bar is
    their mean and the dots are the single models. Each pair of bars sits under its
    column of discs. ``loss_label`` labels the y axis; ``bar_height`` is in inches.
    """
    width = figure_width(width)
    names = list(next(iter(rollouts.values()))["channel_names"])
    figure = plt.figure()
    grid_ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    rows = [("model", source, SOURCE_LABELS[source], SOURCE_COLOURS[source]) for source in SOURCE_LABELS]
    xs, (left, right, bottom, top), hatched = draw_disc_grid(
        grid_ax, {("model", key): rollout for key, rollout in rollouts.items()}, [(f"At {time_label}", INK, rows)],
        columns, groups, gamma, colony_size, hatch, time_label, title)

    # Heights in inches, from the bottom: group legend, tick labels, bars, bar legend, discs
    legend_height = 0.2 * (len(groups) + 1) + 0.15
    tick_height, gap = 0.3, 0.3
    panel_height = width / (right - left) * (top - bottom)
    total = legend_height + tick_height + bar_height + gap + panel_height
    figure.set_size_inches(width, total)
    grid_ax.set_position((0.0, 1.0 - panel_height / total, 1.0, panel_height / total))

    # Bar axes over the disc columns, with the same x units as the discs
    step = xs[1] - xs[0] if len(xs) > 1 else colony_size + 1.5
    x0, x1 = xs[0] - step / 2, xs[-1] + step / 2
    bar_ax = figure.add_axes(((x0 - left) / (right - left), (legend_height + tick_height) / total,
                              (x1 - x0) / (right - left), bar_height / total))
    bar_width = 0.35 * step
    for offset, split, colour, label in ((-0.5, "train", MUTED, "Training replicates"),
                                         (0.5, "validation", INK, "Validation replicates")):
        for k, (x, (key, _)) in enumerate(zip(xs, columns)):
            values = np.asarray(losses.get(key, {}).get(split, []), dtype=float)
            if values.size == 0:
                continue
            centre = x + offset * bar_width
            bar_ax.bar(centre, values.mean(), bar_width, color=colour, label=label if k == 0 else None, zorder=2)
            bar_ax.plot(np.full(values.size, centre), values, "o", ms=2.5, color=INK, mec=PAGE, mew=0.5,
                        zorder=3)
    bar_ax.set_xlim(x0, x1)
    bar_ax.set_xticks(xs, [column_label for _, column_label in columns])
    bar_ax.set_ylabel(loss_label)
    bar_ax.spines[["top", "right"]].set_visible(False)
    # The bars are on white whatever the legend kind of the discs
    bar_ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, **legend_options("white"))
    add_group_legend(figure, names, groups, hatched, legend_kind)
    return figure
