"""Drawing helpers shared by the micropattern figure notebooks.

Colours, fonts and line widths come from ``style.py``. Call these inside
``with paper_style():`` so the matplotlib settings match.
"""

import io
import re

import matplotlib.pyplot as plt
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
    UNMEASURED,
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


def draw_zoom_outline(ax, corner, size, dashes, z):
    """Square outline of black dashes over a solid white line, so it shows on both
    light pages and dark images."""
    for colour, linestyle in (("white", "-"), ("black", (0, dashes))):
        ax.add_patch(Rectangle(corner, size, size, fill=False, edgecolor=colour, lw=0.9, linestyle=linestyle,
                               zorder=z))


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


def stack_panels(draw, panels, width="double", titles=None, same_scale=True, measure_scale=False):
    """Schematic panels stacked top to bottom, each an equal-aspect axes without axis lines.

    ``draw`` maps a panel key to a function drawing it on an axes. With
    ``measure_scale`` the function also gets the panel's points per data unit,
    and the panel is redrawn until that value settles (for text-heavy panels).
    ``same_scale`` gives every panel the same data units per inch. Panels are
    labelled with their key and ``titles[key]``.
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
        label_panel(ax, key, titles.get(key) or None, offset=(0, 0))
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
    ``width`` (fraction of the disc), ``colours`` (one per centre) and ``title``.
    With ``pixels`` ({rollout key: n}) each box is instead n x n whole pixels of
    that rollout, on its pixel grid. ``legend_kind`` is "white" or "black".
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
                for (zx, zy), zoom_colour, x in zip(zoom["centres"], zoom["colours"], zoom_x):
                    fx, fy, fw, fh = zoom_box(zx, zy, rgba, key)
                    ax.add_patch(Rectangle((t + (fx - 0.5) * s, row_y + (0.5 - fy - fh) * s), fw * s, fh * s,
                                           fill=False, edgecolor=zoom_colour, lw=0.8, zorder=6))
                    tile = Rectangle((x - s / 2, row_y - s / 2), s, s, transform=ax.transData)
                    ax.add_patch(Rectangle((x - s / 2, row_y - s / 2), s, s, facecolor=IMAGE_BG, lw=0, zorder=3))
                    image_left = x - s / 2 - fx * s / fw
                    image_top = row_y + s / 2 + fy * s / fh
                    image = ax.imshow(rgba, extent=(image_left, image_left + s / fw, image_top - s / fh, image_top),
                                      origin="upper", interpolation="nearest", zorder=4)
                    image.set_clip_path(tile)
                    ax.add_patch(Rectangle((x - s / 2, row_y - s / 2), s, s, fill=False, edgecolor=zoom_colour,
                                           lw=1.0, zorder=6))
        ys = [row_y for _, row_y in rows]
        ax.plot([x_bracket, x_bracket], [ys[-1] - s / 2 + 0.5, ys[0] + s / 2 - 0.5], color=colour, lw=LINE["axis"])
        ax.text(x_bracket - 0.6 * text, (ys[0] + ys[-1]) / 2, label, rotation=90, ha="right", va="center",
                multialignment="center", fontsize=FONT["label"], color=colour)
    # Limits again, since imshow resets them
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    # A legend row per group: its sector, then its channel colours
    legend_rows = marker_legend_entries(names, groups, sectors=groups, kind=legend_kind)
    if hatched:
        legend_rows.append([(unmeasured_patch(), "not measured")])
    handles, labels, ncol = rows_to_columns(legend_rows)
    figure.legend(handles, labels, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                  **legend_options(legend_kind))
    return figure
