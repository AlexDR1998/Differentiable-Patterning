"""Schematic of the micropattern loss functions, drawn with measured colonies.

Run with:
    marimo edit Experiments/micropatterns/figures/loss_schematic.py

- Two measured control colonies (two replicates) of the 260726 data, loaded with
  the training preprocessing of ``conf/base_config.yaml`` (default: slice
  projection, downsampled x4). One stands in for a prediction x̂, the other is
  the data d
- Panel a: one experiment group at one time, compared three ways: pixelwise L2
  (squared error map), radial profiles (ring means per channel) and texture
  (RGB triplets of channels through VGG)
- Panel b: three colonies on a sheet standing for VGG feature space, each with a
  stack of zoomed patches of its cells (a chosen front patch and random patches at
  several scales behind it, as the texture loss looks at many features and scales):
  x̂ is close to d at the same time and far from d at another time. Positions are
  illustrative, not computed
- The two colonies have the same names in panels a and b (x̂ and d, or colony 1 and
  colony 2; see *Colony names*). Panel b also gives each colony's time
- Panel c: the texture loss over experiment groups, as in
  ``NCA/trainer/loss_multi_target.py``. Each state channel is compared with every
  group that measured its marker; within a group the channels are zero-padded to
  a multiple of 3 and split into RGB triplets (randomly shuffled, re-drawn every
  training step, when ``random_channel_shuffle`` is set), so channels
  are only seen together within one group's colonies. Colonies are matched
  separately for each group
- Figures are saved to ``figures/schematics/``; loaded data is cached in
  ``figures/schematics/cache/``
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Loss schematic")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

    from Experiments.micropatterns.figures.data import (
        CACHE_DIRECTORY,
        TIMEPOINTS,
        cache_name,
        cached,
        crop_box,
        load_colonies,
    )
    from Experiments.micropatterns.figures.drawing import (
        colony_rgba,
        draw_arrow,
        draw_card,
        rounded_box,
        stack_panels,
        text_height,
        text_width,
    )
    from Experiments.micropatterns.figures.style import (
        ACCENT,
        CMAP,
        EXPERIMENT_GROUPS,
        FONT,
        GROUP_COLOURS,
        GROUP_LABELS,
        MODEL,
        SERIES,
        THEME,
        WIDTH,
        group_colours,
        paper_style,
        save_figure,
    )
    from Experiments.micropatterns.figures.style import marker_colour as channel_colour

    load_dotenv()

    # Channels of a triplet go to the R, G and B inputs of VGG
    RGB = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    RGB_TEXT = ("#e00000", "#00a000", "#0050ff")
    # Where d, x̂ at the same time and d at another time sit on the feature
    # sheet of panel b (sheet coordinates in [0, 1]²; illustrative, not computed)
    FEATURE_POSITIONS = {"data": (0.25, 0.55), "prediction": (0.36, 0.38), "time": (0.85, 0.6)}


@app.function(hide_code=True)
def em(size, style):
    """Font size ``size`` (points) in data units, from the panel's measured scale
    ``style["pt_per_unit"]`` (see loss_figure)."""
    return size / style["pt_per_unit"]


@app.function(hide_code=True)
def text_w(text, size, style, bold=False):
    """Rough width of ``text`` in data units (maths markup counted as its symbols)."""
    return text_width(text, size, style["pt_per_unit"], bold)


@app.function(hide_code=True)
def text_h(text, size, style, spacing=1.25):
    """Height of ``text`` in data units."""
    return text_height(text, size, style["pt_per_unit"], spacing)


@app.function(hide_code=True)
def draw_colour_labels(ax, centre_x, top_y, texts, colours, size, style, max_width=None, zorder=12):
    """Labels in a row, centred on ``centre_x`` below ``top_y``; stacked one per line when the
    row is wider than ``max_width``. Returns the bottom."""
    gap = 0.6 * em(size, style)
    widths = [text_w(t, size, style) for t in texts]
    total = sum(widths) + gap * (len(texts) - 1)
    line = text_h("x", size, style)
    if max_width is None or total <= max_width:
        x = centre_x - total / 2
        for text, colour, width in zip(texts, colours, widths):
            ax.text(x + width / 2, top_y, text, ha="center", va="top", fontsize=size, color=colour, zorder=zorder)
            x += width + gap
        return top_y - line
    for k, (text, colour) in enumerate(zip(texts, colours)):
        ax.text(centre_x, top_y - k * line, text, ha="center", va="top", fontsize=size, color=colour, zorder=zorder)
    return top_y - len(texts) * line


@app.function(hide_code=True)
def marker_colour(marker, theme):
    """Display colour of a marker, as in the composites (LMBR, shown in no colour: muted)."""
    colour = channel_colour(marker)
    return theme["muted"] if colour is None else colour


@app.function(hide_code=True)
def group_triplets(channels, shuffle, seed, group_index):
    """RGB triplets of a group's channels, as the texture loss forms them: zero-padded
    (None) to a multiple of 3 and, with ``shuffle``, permuted once (one random draw)."""
    padded = list(channels) + [None] * ((-len(channels)) % 3)
    if shuffle:
        order = np.random.default_rng(seed + group_index).permutation(len(padded))
        padded = [padded[k] for k in order]
    return [padded[k:k + 3] for k in range(0, len(padded), 3)]


@app.function(hide_code=True)
def triplet_rgba(image, mask, triplet, gamma):
    """RGBA of three channels (None: zero padding) as the R, G and B inputs of VGG."""
    stack = np.stack([image[c] if c is not None else np.zeros_like(image[0]) for c in triplet])
    return colony_rgba(stack, mask, RGB, gamma)


@app.function(hide_code=True)
def error_rgba(prediction, target, mask, colour_map):
    """RGBA of the squared error per pixel (mean over channels [C, H, W]), cropped to the
    colony and scaled to its 99th percentile."""
    error = ((prediction - target) ** 2).mean(axis=0)
    crop = crop_box(mask)
    error, inside = error[crop], mask[crop]
    top = np.percentile(error[inside], 99) if inside.any() else 1.0
    rgba = plt.get_cmap(colour_map)(np.clip(error / max(top, 1e-12), 0, 1))
    rgba[..., 3] = inside
    return rgba


@app.function(hide_code=True)
def radial_profiles(image, mask, bins):
    """Mean of each channel [C, H, W] in ``bins`` rings around the colony's centre, as
    Common/trainer/loss.py:radial_profiles (radius scaled to 1 at the furthest pixel)."""
    rows, cols = np.indices(mask.shape)
    weight = mask.astype(float)
    centre = ((rows * weight).sum() / weight.sum(), (cols * weight).sum() / weight.sum())
    radius = np.hypot(rows - centre[0], cols - centre[1])
    radius /= max(radius[mask].max(), 1e-8)
    ring = np.minimum((radius * bins).astype(int), bins - 1)[mask]
    counts = np.bincount(ring, minlength=bins)
    sums = np.stack([np.bincount(ring, weights=channel[mask], minlength=bins) for channel in image])
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


@app.function(hide_code=True)
def draw_unmeasured(ax, corner, size, style):
    """Empty card for a colony that was not measured."""
    draw_card(ax, corner, size, 10)
    ax.text(corner[0] + size / 2, corner[1] + size / 2, "not\nmeasured", ha="center", va="center",
            fontsize=style["font"] - 2, color=style["theme"]["muted"], zorder=12)


@app.function(hide_code=True)
def draw_rings(ax, corner, size, bins, highlight, style):
    """Ring boundaries over a card's colony disc, with ring ``highlight`` shaded."""
    centre = (corner[0] + size / 2, corner[1] + size / 2)
    outer = 0.45 * size
    for k in range(1, bins + 1):
        ax.add_patch(Circle(centre, outer * k / bins, fill=False, edgecolor="white", lw=0.3, alpha=0.45,
                            zorder=11))
    if 0 <= highlight < bins:
        ax.add_patch(Wedge(centre, outer * (highlight + 1) / bins, 0, 360, width=outer / bins,
                           facecolor=style["ring_colour"], edgecolor="none", alpha=0.6, zorder=11.5))


@app.function(hide_code=True)
def draw_profiles(ax, corner, width, height, profiles, colours, style):
    """Radial profiles of x̂ (solid) and d (dashed), one colour per channel, as a small plot."""
    theme = style["theme"]
    fg, small = theme["fg"], style["font"] - 1
    x0, y0 = corner
    bins = profiles[0].shape[1]
    top = np.nanmax(profiles) * 1.08 or 1.0
    radius = (np.arange(bins) + 0.5) / bins

    def to_xy(r, value):
        return x0 + width * r, y0 + height * value / top

    ax.plot([x0, x0], [y0, y0 + height], color=fg, lw=0.7)
    ax.plot([x0, x0 + width], [y0, y0], color=fg, lw=0.7)
    ring = style["ring"]
    if 0 <= ring < bins:
        ax.add_patch(Rectangle((x0 + width * ring / bins, y0), width / bins, height,
                               facecolor=style["ring_colour"], alpha=0.25, edgecolor="none", zorder=2))
    for channel, colour in enumerate(colours):
        xs, ys_x = to_xy(radius, profiles[0][channel])
        _, ys_d = to_xy(radius, profiles[1][channel])
        if style["shade_profiles"]:
            ax.fill_between(xs, ys_x, ys_d, color=style["loss_colour"], alpha=0.12, lw=0, zorder=3)
        ax.plot(xs, ys_x, color=colour, lw=1.1, zorder=4)
        ax.plot(xs, ys_d, color=colour, lw=1.1, ls=(0, (2.5, 1.5)), zorder=4)
    ax.text(x0 + width / 2, y0 - 0.06, style["radius_label"], ha="center", va="top", fontsize=small)
    ax.text(x0 - 0.08, y0 + height / 2, style["ring_mean_label"], ha="center", va="bottom", rotation=90,
            rotation_mode="anchor", fontsize=small)
    # Key: line style of x̂ and d
    y = y0 + height + 0.1
    keys = ((None, style["names"]["prediction"]), ((0, (2.5, 1.5)), style["names"]["data"]))
    kx = x0 + width - sum(0.4 + text_w(text, small, style) for _, text in keys)
    for line, text in keys:
        ax.plot([kx, kx + 0.25], [y, y], color=fg, lw=1.0, ls=line or "-")
        ax.text(kx + 0.3, y, text, ha="left", va="center", fontsize=small)
        kx += 0.4 + text_w(text, small, style)


@app.function(hide_code=True)
def draw_vgg(ax, x, centre_y, width, height, style):
    """VGG as a narrowing block (opaque, so lines behind it are hidden)."""
    corners = [(x, centre_y - height / 2), (x, centre_y + height / 2), (x + width, centre_y + height / 4),
               (x + width, centre_y - height / 4)]
    ax.add_patch(Polygon(corners, closed=True, facecolor=style["theme"]["bg"], edgecolor="none", zorder=4))
    ax.add_patch(Polygon(corners, closed=True, facecolor=style["model_colour"], alpha=0.2,
                         edgecolor=style["model_colour"], lw=0.8, zorder=4.1))
    ax.text(x + width / 2, centre_y, style["vgg_label"], ha="center", va="center", fontsize=style["font"] - 1,
            rotation=90, zorder=5)


@app.function(hide_code=True)
def draw_manifold(ax, corner, width, height, style, grid=7):
    """A curved sheet standing in for the space of VGG features, with grid lines. It bulges
    upwards in the middle by ``style["sheet_bulge"]`` (negative: a bowl).

    Returns ``to_xy(u, v)``, mapping sheet coordinates in [0, 1]² to the page."""
    model = style["model_colour"]
    bulge = style["sheet_bulge"]
    x0, y0 = corner

    def to_xy(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        x = x0 + width * (0.8 * u + 0.2 * v)
        # An arch along u, a little flatter towards the back edge
        dome = (1 - (2 * u - 1) ** 2) * (1 - 0.3 * v)
        y = y0 + height * (max(-bulge, 0) + 0.5 * v + bulge * dome) / (0.5 + abs(bulge))
        return x, y

    line = np.linspace(0, 1, 60)
    edge_u = np.concatenate([line, np.ones(60), line[::-1], np.zeros(60)])
    edge_v = np.concatenate([np.zeros(60), line, np.ones(60), line[::-1]])
    ax.add_patch(Polygon(np.stack(to_xy(edge_u, edge_v), axis=1), closed=True, facecolor=model, alpha=0.08,
                         edgecolor=model, lw=0.8, zorder=2))
    for k in np.linspace(0, 1, grid)[1:-1]:
        for u, v in ((np.full(60, k), line), (line, np.full(60, k))):
            ax.plot(*to_xy(u, v), color=model, lw=0.3, alpha=0.5, zorder=2.1)
    return to_xy


@app.function(hide_code=True)
def draw_geodesic(ax, to_xy, start, end, colour, lw=1.2, zorder=5):
    """Path between two sheet points, straight in sheet coordinates (so it follows the sheet)."""
    fraction = np.linspace(0, 1, 40)
    u = start[0] + (end[0] - start[0]) * fraction
    v = start[1] + (end[1] - start[1]) * fraction
    ax.plot(*to_xy(u, v), color=colour, lw=lw, solid_capstyle="round", zorder=zorder)


@app.function(hide_code=True)
def draw_dot(ax, xy, colour, style, radius=0.07):
    ax.add_patch(Circle(xy, radius, facecolor=colour, edgecolor=style["theme"]["bg"], lw=0.6, zorder=7))


@app.function(hide_code=True)
def draw_features(ax, x, centre_y, size, style):
    """VGG mapping x̂ and d onto the feature sheet, where they are compared. Returns the right edge."""
    small = style["font"] - 1
    narrow = 0.4 * size
    draw_vgg(ax, x, centre_y, narrow, size, style)
    to_xy = draw_manifold(ax, (x + narrow + 0.25, centre_y - 0.45 * size), 1.35 * size, 0.85 * size, style, grid=5)
    points = {"data": (0.22, 0.62), "prediction": (0.62, 0.32)}
    draw_geodesic(ax, to_xy, points["data"], points["prediction"], style["loss_colour"])
    colours = view_colours(style)
    for key, (u, v) in points.items():
        draw_dot(ax, to_xy(u, v), colours[key], style)
    for key, side in (("prediction", 1), ("data", -1)):
        px, py = to_xy(*points[key])
        ax.text(px + side * 0.12, py, f"$\\phi$({style['names'][key]})", ha="left" if side > 0 else "right",
                va="center", fontsize=small, zorder=8)
    return x + narrow + 0.25 + 1.35 * size


@app.function(hide_code=True)
def colony_caption(key, style, panel):
    """Caption of a colony ("data", "prediction", or "time": d at the far time) in panel a or b."""
    name = style["names"]["data" if key == "time" else key]
    hour = style["far_hour"] if key == "time" else style["hour"]
    return style[f"caption_{panel}"].format(name=name, hour=hour)


@app.function(hide_code=True)
def view_colours(style):
    """Colour of d and of the colonies compared with it."""
    return {"data": style["theme"]["fg"], "prediction": style["model_colour"], "time": style["far_colour"]}


@app.function(hide_code=True)
def zoom_crop(rgba, radius, angle, fraction):
    """Square crop of a colony RGBA (cropped to its mask), centred at ``radius`` (fraction of
    the colony radius) and ``angle`` (degrees, anticlockwise from +x), ``fraction`` of its
    width across. Also returns the crop centre and half-width in [0, 1] image units."""
    size = rgba.shape[0]
    centre = (0.5 + 0.5 * radius * np.cos(np.radians(angle)), 0.5 - 0.5 * radius * np.sin(np.radians(angle)))
    half = fraction / 2
    left = int(round((centre[0] - half) * size))
    top = int(round((centre[1] - half) * size))
    width = max(int(round(fraction * size)), 2)
    left, top = np.clip(left, 0, size - width), np.clip(top, 0, size - width)
    return rgba[top:top + width, left:left + width], centre, half


@app.function(hide_code=True)
def patch_samples(key, style):
    """(radius, angle, fraction) of each patch in the stack of ``key``, front first. The
    front patch is the chosen one; the ones behind are random samples from the same image
    (radius in ``style["sample_radius"]``, size scaled within ``style["scale_range"]``)."""
    position = style["crop_positions"]["data" if style["same_position"] else key]
    patches = [(position["radius"], position["angle"], style["crop_size"])]
    rng = np.random.default_rng(style["stack_seed"] + 101 * list(FEATURE_POSITIONS).index(key))
    low, high = style["sample_radius"]
    small, large = style["scale_range"]
    for _ in range(style["stack_depth"] - 1):
        scale = np.exp(rng.uniform(np.log(small), np.log(large))) if style["vary_scale"] else 1.0
        patches.append((rng.uniform(low, high), rng.uniform(0, 360), min(style["crop_size"] * scale, 1.0)))
    return patches


@app.function(hide_code=True)
def draw_patch_stack(ax, corner, rgba, patches, colour, style):
    """Crops of ``rgba`` as a stack of squares going back up and to the right, front first
    in ``patches``. With ``style["show_crop_location"]`` the front square has the whole
    colony in its corner, with every patch marked (the front one in white)."""
    zoom, step = style["zoom"], style["stack_offset"]
    count = len(patches)
    crops = [zoom_crop(rgba, *patch) for patch in patches]
    for k in reversed(range(count)):
        x, y = corner[0] + k * step, corner[1] + k * step
        z = 9 + (count - k) * 0.3
        ax.add_patch(Rectangle((x, y), zoom, zoom, facecolor="black", edgecolor="none", zorder=z))
        ax.imshow(crops[k][0], extent=(x, x + zoom, y, y + zoom), origin="upper",
                  interpolation="nearest" if style["pixelated"] else "antialiased", zorder=z + 0.1)
        ax.add_patch(Rectangle((x, y), zoom, zoom, fill=False, edgecolor=colour, lw=1.4 if k == 0 else 0.6,
                               zorder=z + 0.2))
    if style["show_crop_location"]:
        inset = 0.32 * zoom
        ix, iy = corner[0] + zoom - inset - 0.03, corner[1] + zoom - inset - 0.03
        z = 9 + count * 0.3 + 0.3
        ax.add_patch(Rectangle((ix, iy), inset, inset, facecolor="black", edgecolor="white", lw=0.4, zorder=z))
        ax.imshow(rgba, extent=(ix, ix + inset, iy, iy + inset), origin="upper", zorder=z + 0.1)
        for k, (_, centre, half) in reversed(list(enumerate(crops))):
            ax.add_patch(Rectangle((ix + inset * (centre[0] - half), iy + inset * (1 - centre[1] - half)),
                                   inset * 2 * half, inset * 2 * half, fill=False,
                                   edgecolor="white" if k == 0 else "#aaaaaa", lw=0.5 if k == 0 else 0.3,
                                   zorder=z + 0.2))


@app.function(hide_code=True)
def zoom_views(data, style):
    """Colony RGBA of panel b's RGB triplet for d, another colony at the same time and d at
    another time (None where not measured), and the triplet. Panel b uses its own group
    and triplet (``style["b_group"]``, None: panel a's)."""
    group = style["b_group"] or style["group"]
    group_index = data["groups"].index(group)
    channels = list(data["group_indices"][group_index])
    triplets = group_triplets(channels, style["shuffle"], style["shuffle_seed"], group_index)
    triplet = triplets[min(style["b_triplet"] if style["b_group"] else style["triplet"], len(triplets) - 1)]
    target, prediction = style["target_index"], style["prediction_index"]
    mask = data["masks"][target]
    sources = {
        "data": (target, TIMEPOINTS.index(style["hour"])),
        "prediction": (prediction, TIMEPOINTS.index(style["hour"])),
        "time": (target, TIMEPOINTS.index(style["far_hour"])),
    }
    rgbas = {
        key: triplet_rgba(data["images"][b, i], mask, triplet, style["gamma"])
        if data["measured"][b, i, channels].all() else None
        for key, (b, i) in sources.items()
    }
    return rgbas, triplet


@app.function(hide_code=True)
def draw_feature_space_panel(ax, data, style):
    """Panel b: three colonies on the VGG feature sheet, each with a zoomed crop of its
    cells. x̂ is close to d at the same time and far from d at another time.
    Positions are illustrative, not computed."""
    theme = style["theme"]
    muted = theme["muted"]
    font, small = style["font"], style["font"] - 1
    zoom = style["zoom"]
    rgbas, triplet = zoom_views(data, style)
    colours = view_colours(style)
    # How far a stack reaches up and to the right of its front square
    extent = (style["stack_depth"] - 1) * style["stack_offset"]
    # Zoom stacks of the two close colonies on the left, the far one on the right
    gap = 0.6
    sheet_x, sheet_w = zoom + extent + gap, 4.6
    sheet_h = max(2.0 * (zoom + extent) + 0.5, 2.2)
    sheet_y = 0.0
    to_xy = draw_manifold(ax, (sheet_x, sheet_y), sheet_w, sheet_h, style)
    ax.text(sheet_x + sheet_w * 0.55, sheet_y + sheet_h * 1.02, style["feature_space_label"], ha="center",
            va="bottom", fontsize=font, color=style["model_colour"])
    points = {k: to_xy(*FEATURE_POSITIONS[k]) for k in ("data", "prediction", "time")}

    # Distances along the sheet, from x̂ to each sample of d
    for key in ("data", "time"):
        draw_geodesic(ax, to_xy, FEATURE_POSITIONS["prediction"], FEATURE_POSITIONS[key], style["loss_colour"],
                      lw=1.4 if key == "time" else 1.8)

    label_room = 0.3
    defaults = {
        "data": (0.0, sheet_y + sheet_h - zoom - extent),
        "prediction": (0.0, sheet_y + label_room),
        "time": (sheet_x + sheet_w + gap, sheet_y + (sheet_h - zoom) / 2 + 0.2),
    }
    # Moved by the user (data units)
    corners = {k: (x + style["stack_shifts"][k]["x"], y + style["stack_shifts"][k]["y"])
               for k, (x, y) in defaults.items()}
    low, high, left, right = sheet_y, sheet_y + sheet_h * 1.02 + 0.3, 0.0, sheet_x + sheet_w
    for key, point in points.items():
        colour = colours[key]
        corner = corners[key]
        cx, cy = corner[0] + zoom / 2, corner[1] + zoom / 2
        # Callout: lines from the point to the two corners of the box side facing it
        near_x = corner[0] + zoom if cx < point[0] else corner[0]
        for y in (corner[1], corner[1] + zoom):
            ax.plot([point[0], near_x], [point[1], y], color=colour, lw=0.6, alpha=0.7, zorder=6)
        if rgbas[key] is None:
            ax.add_patch(Rectangle(corner, zoom, zoom, facecolor="black", edgecolor=colour, lw=1.4, zorder=9))
            ax.text(cx, cy, "not\nmeasured", ha="center", va="center", fontsize=small - 1, color=muted, zorder=10)
        else:
            draw_patch_stack(ax, corner, rgbas[key], patch_samples(key, style), colour, style)
        label = colony_caption(key, style, "b")
        ax.text(cx, corner[1] - 0.05, label, ha="center", va="top", fontsize=small, color=colour, zorder=12)
        draw_dot(ax, point, colour, style, radius=0.09)
        low = min(low, corner[1] - 0.1 - text_h(label, small, style))
        high = max(high, corner[1] + zoom + extent + 0.1)
        left = min(left, corner[0], cx - text_w(label, small, style) / 2)
        right = max(right, corner[0] + zoom + extent, cx + text_w(label, small, style) / 2)

    # Which channel goes to which VGG input (one random draw when shuffled), under the far box
    names = [f"{'RGB'[j]}: {data['markers'][c] if c is not None else '0'}" for j, c in enumerate(triplet)]
    far_label = colony_caption("time", style, "b")
    key_y = corners["time"][1] - 0.15 - text_h(far_label, small, style)
    bottom = draw_colour_labels(ax, corners["time"][0] + zoom / 2, key_y, names, RGB_TEXT, small - 1.5, style,
                                max_width=zoom + extent)
    low = min(low, bottom - 0.1)
    ax.set_xlim(left - 0.1, right)
    ax.set_ylim(low - 0.1, high + 0.25)


@app.function(hide_code=True)
def draw_comparison_panel(ax, data, style):
    """Panel a: one experiment group of x̂ and d at one time, compared by each loss (rows)."""
    theme = style["theme"]
    fg = theme["fg"]
    font, small = style["font"], style["font"] - 1
    size = style["card"]
    group_index = data["groups"].index(style["group"])
    channels = list(data["group_indices"][group_index])
    t = TIMEPOINTS.index(style["hour"])
    batch = (style["prediction_index"], style["target_index"])
    mask = data["masks"][batch[1]]
    images = [data["images"][b, t] for b in batch]
    measured = all(data["measured"][b, t, channels].all() for b in batch)
    colours = group_colours(data["channel_names"], style["group"])
    rows = [r for r in ("l2", "radial", "texture") if r in style["loss_rows"]]

    label_width = max(max(text_w(style[f"{r}_name"], font, style, bold=True),
                          text_w(style[f"{r}_formula"], small, style)) for r in rows) + 0.3
    x_prediction = label_width
    x_data = x_prediction + size + 0.3
    x_result = x_data + size + 0.75
    result_width = 2.0 * size
    x_note = x_result + result_width + 0.4
    row_height = size + style["row_spacing"]
    header_y = size / 2 + 0.1
    for x, key in ((x_prediction, "prediction"), (x_data, "data")):
        ax.text(x + size / 2, header_y, colony_caption(key, style, "a"), ha="center", va="bottom",
                fontsize=font + 1)

    for k, row in enumerate(rows):
        cy = -k * row_height
        ax.text(0, cy + 0.04, style[f"{row}_name"], ha="left", va="bottom", fontsize=font, fontweight="bold")
        ax.text(0, cy - 0.04, style[f"{row}_formula"], ha="left", va="top", fontsize=small)
        ax.text(x_note, cy, style[f"{row}_note"], ha="left", va="center", fontsize=small, linespacing=1.3,
                color=theme["muted"] if style["muted_notes"] else fg)
        corners = [(x_prediction, cy - size / 2), (x_data, cy - size / 2)]
        if not measured:
            for corner in corners:
                draw_unmeasured(ax, corner, size, style)
            continue
        if row == "texture":
            triplets = group_triplets(channels, style["shuffle"], style["shuffle_seed"], group_index)
            triplet = triplets[min(style["triplet"], len(triplets) - 1)]
            for corner, image in zip(corners, images):
                draw_card(ax, corner, size, 10, triplet_rgba(image, mask, triplet, style["gamma"]))
            # Which channel feeds which input of VGG
            names = [data["markers"][c] if c is not None else "0" for c in triplet]
            for corner in corners:
                draw_colour_labels(ax, corner[0] + size / 2, cy - size / 2 - 0.05, names, RGB_TEXT, small - 1.5,
                                   style, max_width=size + 0.25)
        else:
            for corner, image in zip(corners, images):
                draw_card(ax, corner, size, 10,
                                   colony_rgba(image, mask, colours, style["gamma"]))
                if row == "radial":
                    draw_rings(ax, corner, size, style["radial_bins"], style["ring"], style)
        draw_arrow(ax, (x_data + size + 0.08, cy), (x_result - 0.1, cy), fg)

        if row == "l2":
            corner = (x_result, cy - size / 2)
            draw_card(ax, corner, size, 10,
                               error_rgba(images[0][channels], images[1][channels], mask, style["error_map"]))
            bar = np.linspace(1, 0, 64)[:, None]
            ax.imshow(bar, cmap=style["error_map"], extent=(x_result + size + 0.08, x_result + size + 0.16,
                                                             cy - 0.45 * size, cy + 0.45 * size),
                      aspect="auto", zorder=10)
            ax.text(x_result + size + 0.22, cy + 0.45 * size, style["error_high"], ha="left", va="top",
                    fontsize=small - 1)
            ax.text(x_result + size + 0.22, cy - 0.45 * size, "0", ha="left", va="bottom", fontsize=small - 1)
            ax.text(x_result + size / 2, cy + size / 2 + 0.06, style["error_label"], ha="center", va="bottom",
                    fontsize=small)
        elif row == "radial":
            profiles = [radial_profiles(image[channels], mask, style["radial_bins"]) for image in images]
            line_colours = [marker_colour(data["markers"][c], theme) for c in channels]
            draw_profiles(ax, (x_result + 0.3, cy - size / 2 + 0.1), result_width - 0.3, size - 0.1, profiles,
                          line_colours, style)
        else:
            draw_features(ax, x_result, cy, size, style)

    right = x_note + max(text_w(style[f"{r}_note"], small, style) for r in rows)
    bottom = -(len(rows) - 1) * row_height - size / 2 - 0.4
    ax.set_xlim(-0.05, right)
    ax.set_ylim(bottom, header_y + 0.75)


@app.function(hide_code=True)
def draw_state_column(ax, x, markers, counts, style):
    """NCA state as a column of marker chips; ×n marks markers measured in n groups.
    Returns the right edge and the top."""
    theme = style["theme"]
    small = style["font"] - 1
    pad = 0.4 * em(small, style)
    chip_h = 1.6 * em(small, style)
    chip_gap = 0.3 * chip_h
    swatch = chip_h - 2 * pad
    badge = text_w("×3", small - 0.5, style, bold=True) if any(n > 1 for n in counts.values()) else 0.0
    chip_w = pad + swatch + pad + max(text_w(m, small, style) for m in markers) + 2 * pad + badge + pad
    top = (len(markers) * chip_h + (len(markers) - 1) * chip_gap) / 2
    for k, marker in enumerate(markers):
        y = top - k * (chip_h + chip_gap) - chip_h
        ax.add_patch(Rectangle((x, y), chip_w, chip_h, facecolor=theme["bg"], edgecolor=theme["muted"], lw=0.6,
                               zorder=4))
        ax.add_patch(Rectangle((x + pad, y + pad), swatch, swatch, facecolor=marker_colour(marker, theme),
                               edgecolor="none", zorder=5))
        ax.text(x + 2 * pad + swatch, y + chip_h / 2, marker, ha="left", va="center", fontsize=small, zorder=6)
        if counts[marker] > 1:
            ax.text(x + chip_w - pad, y + chip_h / 2, f"×{counts[marker]}", ha="right", va="center",
                    fontsize=small - 0.5, color=style["shared_colour"], fontweight="bold", zorder=6)
    label_y = top + pad
    ax.text(x + chip_w / 2, label_y, style["state_label"], ha="center", va="bottom", fontsize=small,
            linespacing=1.2)
    return x + chip_w, label_y + text_h(style["state_label"], small, style)


@app.function(hide_code=True)
def draw_pair_matrix(ax, x, block_top, entries, style):
    """Channel pairs that enter the texture loss together: the same RGB triplet, the same
    group (only with channel shuffling, over iterations) or never (different groups).

    ``entries`` holds (group, marker, triplet index) per channel; ``block_top`` is the top of
    its title. Returns (right, top, bottom)."""
    theme = style["theme"]
    small = style["font"] - 1
    label_size = small - 1
    # Cells at least as tall as a label line
    cell = max(style["cell"], text_h("x", label_size, style, spacing=1.1))
    count = len(entries)
    pad = 0.4 * em(label_size, style)
    label_w = max(text_w(m, label_size, style) for _, m, _ in entries)
    left = x + label_w + pad
    title_y = block_top - text_h(style["matrix_title"], small, style)
    top = title_y - 3 * pad - label_w
    colour = style["pair_colour"]
    for a, (group_a, marker_a, triplet_a) in enumerate(entries):
        y = top - (a + 1) * cell
        group_colour = GROUP_COLOURS.get(group_a, theme["fg"])
        ax.text(left - pad, y + cell / 2, marker_a, ha="right", va="center", fontsize=label_size, color=group_colour)
        ax.text(left + (a + 0.5) * cell, top + pad, marker_a, ha="center", va="bottom", rotation=90,
                fontsize=label_size, color=group_colour)
        for b, (group_b, _, triplet_b) in enumerate(entries):
            corner = (left + b * cell, y)
            if group_a == group_b and triplet_a == triplet_b:
                ax.add_patch(Rectangle(corner, cell, cell, facecolor=colour, alpha=0.85, edgecolor="none", zorder=3))
            elif group_a == group_b and style["shuffle"]:
                ax.add_patch(Rectangle(corner, cell, cell, facecolor=colour, alpha=0.3, edgecolor="none", zorder=3))
            elif group_a != group_b:
                ax.add_patch(Rectangle(corner, cell, cell, fill=False, hatch="//////", edgecolor=theme["muted"],
                                       lw=0, alpha=0.6, zorder=3))
            ax.add_patch(Rectangle(corner, cell, cell, fill=False, edgecolor=theme["muted"], lw=0.2, zorder=4))
    # Group blocks on the diagonal
    start = 0
    for group in dict.fromkeys(g for g, _, _ in entries):
        size = sum(g == group for g, _, _ in entries)
        ax.add_patch(Rectangle((left + start * cell, top - (start + size) * cell), size * cell, size * cell,
                               fill=False, edgecolor=GROUP_COLOURS.get(group, theme["fg"]), lw=1.3, zorder=5))
        start += size
    ax.text(left + count * cell / 2, title_y, style["matrix_title"], ha="center", va="bottom", fontsize=small)
    right = max(left + count * cell, left + count * cell / 2 + text_w(style["matrix_title"], small, style) / 2)

    # Key
    key_size = small - 0.5
    swatch = 1.2 * em(key_size, style)
    keys = [(dict(facecolor=colour, alpha=0.85, edgecolor="none"), style["key_triplet"])]
    if style["shuffle"]:
        keys.append((dict(facecolor=colour, alpha=0.3, edgecolor="none"), style["key_group"]))
    keys.append((dict(fill=False, hatch="//////", edgecolor=theme["muted"], lw=0, alpha=0.6), style["key_cross"]))
    y = top - count * cell - 2 * pad
    for kwargs, text in keys:
        ax.add_patch(Rectangle((x, y - swatch), swatch, swatch, **kwargs))
        ax.add_patch(Rectangle((x, y - swatch), swatch, swatch, fill=False, edgecolor=theme["muted"], lw=0.3))
        ax.text(x + swatch + pad, y, text, ha="left", va="top", fontsize=key_size, linespacing=1.2)
        right = max(right, x + swatch + pad + text_w(text, key_size, style))
        y -= max(swatch, text_h(text, key_size, style)) + pad
    return right, title_y + text_h(style["matrix_title"], small, style), y


@app.function(hide_code=True)
def draw_groups_panel(ax, data, style):
    """Panel c: the texture loss over experiment groups, for the data colony d at one time.
    Spacing around text follows the font size (see em)."""
    theme = style["theme"]
    fg, muted = theme["fg"], theme["muted"]
    font, small = style["font"], style["font"] - 1
    tile_label = small - 1.5
    pad = 0.5 * em(small, style)
    tile = style["tile"]
    t = TIMEPOINTS.index(style["hour"])
    target = style["target_index"]
    image, mask = data["images"][target, t], data["masks"][target]
    groups = [g for g in data["groups"] if g in style["shown_groups"]]
    indices = {g: data["groups"].index(g) for g in groups}
    triplets = {g: group_triplets(data["group_indices"][indices[g]], style["shuffle"], style["shuffle_seed"],
                                  indices[g]) for g in groups}
    most = max(len(triplets[g]) for g in groups)
    names = [data["markers"][c] for g in groups for c in data["group_indices"][indices[g]]] + ["pad"]
    pitch = tile + 0.08
    triplet_gap = max(0.2, pad)
    triplet_label = style["triplet_label"].format(most)
    composite_pitch = tile + 0.1
    # Labels wider than a tile are turned upright, which costs height instead of width
    longest = max(text_w(n, tile_label, style) for n in names + [triplet_label])
    upright = longest > pitch - 0.3 * pad
    label_kwargs = dict(rotation=90, ha="center", va="top") if upright else dict(ha="center", va="top")

    # NCA state: one channel per marker, compared with every group that measured it
    markers = [m for m in data["state_channels"] if m in names]
    counts = {m: sum(any(data["markers"][c] == m for c in data["group_indices"][indices[g]]) for g in groups)
              for m in markers}
    x = 0.0
    top = 0.0
    if style["show_state"]:
        x, top = draw_state_column(ax, 0.0, markers, counts, style)
        length = max(0.6, text_w(style["select_label"], small - 0.5, style) + 2 * pad)
        draw_arrow(ax, (x + pad, 0), (x + pad + length, 0), fg)
        ax.text(x + pad + length / 2, pad / 2, style["select_label"], ha="center", va="bottom", fontsize=small - 0.5,
                linespacing=1.2)
        x += length + 2 * pad

    # One row per experiment group: tiles, their labels, a bracket per triplet and its label
    # Capitals run wider than the estimate, so upright labels get some extra room
    label_line = 1.2 * longest + 0.3 * pad if upright else text_h("x", tile_label, style)
    bracket = 0.6 * pad
    below = 0.05 + label_line + pad + bracket + text_h("x", tile_label, style) + pad
    row_h = pad + tile + below
    row_gap = pad
    total = len(groups) * row_h + (len(groups) - 1) * row_gap
    excluded_text = {g: style["excluded_label"].format(", ".join(n.split("/")[-1] for n in data["excluded"]
                                                                  if n.split("/")[0] == g)) for g in groups}
    shown_notes = [excluded_text[g] for g in groups if style["show_excluded"]
                   and any(n.split("/")[0] == g for n in data["excluded"])]
    label_w = max([text_w(GROUP_LABELS.get(g, g), small, style, bold=True) for g in groups]
                  + [text_w(n, small - 1.5, style) for n in shown_notes]) + 2 * pad
    tiles_x = x + label_w
    composite_x = tiles_x + 3 * most * pitch + (most - 1) * triplet_gap + 0.4
    loss_x = composite_x + most * composite_pitch
    loss_w = max(text_w(style["group_loss_label"].format(len(groups)), font, style), 0.3)
    box_right = loss_x + loss_w + 2 * pad
    stubs = []
    for i, group in enumerate(groups):
        row_top = total / 2 - i * (row_h + row_gap)
        tiles_y = row_top - pad - tile
        middle = tiles_y + tile / 2
        colour = GROUP_COLOURS.get(group, muted)
        rounded_box(ax, (x, row_top - row_h), box_right - x, row_h, colour, z=1)
        ax.text(x + pad, row_top - pad, GROUP_LABELS.get(group, group), ha="left", va="top", fontsize=small,
                fontweight="bold", color=colour)
        if style["show_excluded"] and group in [g for g in groups if any(n.split("/")[0] == g for n in data["excluded"])]:
            ax.text(x + pad, row_top - pad - text_h("x", small, style), excluded_text[group], ha="left", va="top",
                    fontsize=small - 1.5, color=muted, linespacing=1.2)
        measured = data["measured"][target, t, list(data["group_indices"][indices[group]])].all()
        label_y = tiles_y - 0.05
        bracket_y = label_y - label_line - pad
        for k, triplet in enumerate(triplets[group]):
            first = tiles_x + 3 * k * pitch + k * triplet_gap
            for j, c in enumerate(triplet):
                px = first + j * pitch + (pitch - tile) / 2
                if c is None:
                    ax.add_patch(Rectangle((px, tiles_y), tile, tile, facecolor=theme["bg"], edgecolor=muted,
                                           lw=0.6, ls=(0, (2, 1.5)), zorder=10))
                    ax.text(px + tile / 2, tiles_y + tile / 2, "0", ha="center", va="center", fontsize=small,
                            color=muted, zorder=11)
                elif measured:
                    draw_card(ax, (px, tiles_y), tile, 10,
                                       colony_rgba(image[[c]], mask, [(1.0, 1.0, 1.0)], style["gamma"]))
                else:
                    draw_unmeasured(ax, (px, tiles_y), tile, style)
                ax.text(px + tile / 2, label_y, data["markers"][c] if c is not None else "pad",
                        fontsize=tile_label, color=RGB_TEXT[j], **label_kwargs)
            # Bracket under the triplet
            last = first + 3 * pitch
            ax.plot([first, first, last, last], [bracket_y + bracket, bracket_y, bracket_y, bracket_y + bracket],
                    color=fg, lw=0.6)
            ax.text((first + last) / 2, bracket_y - 0.03, style["triplet_label"].format(k + 1), ha="center",
                    va="top", fontsize=tile_label)
            corner = (composite_x + k * composite_pitch + (composite_pitch - tile) / 2, tiles_y)
            if measured:
                draw_card(ax, corner, tile, 10, triplet_rgba(image, mask, triplet, style["gamma"]))
            else:
                draw_unmeasured(ax, corner, tile, style)
            ax.text(corner[0] + tile / 2, label_y, style["triplet_label"].format(k + 1), fontsize=tile_label,
                    **label_kwargs)
        row_end = tiles_x + 3 * len(triplets[group]) * pitch + (len(triplets[group]) - 1) * triplet_gap
        draw_arrow(ax, (row_end + 0.02, middle), (composite_x - 0.08, middle), fg, lw=0.8)
        ax.text(loss_x + pad, middle, style["group_loss_label"].format(i + 1), ha="left", va="center",
                fontsize=font)
        stubs.append(middle)
    composite_text = style["composite_label"]
    if style["shuffle"] and style["shuffle_note"]:
        composite_text += "\n" + style["shuffle_note"]
    ax.text(composite_x + most * composite_pitch / 2, total / 2 + pad / 2, composite_text, ha="center",
            va="bottom", fontsize=small - 0.5, linespacing=1.2)

    # Mean over groups
    bar_x = box_right + 2 * pad
    for y in stubs:
        ax.plot([box_right - 0.02, bar_x], [y, y], color=fg, lw=0.7)
    ax.plot([bar_x, bar_x], [min(stubs), max(stubs)], color=fg, lw=0.7)
    centre = (min(stubs) + max(stubs)) / 2
    column_x = bar_x + 0.3 + pad
    # With the matrix the labels sit at the top of the column, the matrix under them
    label_y = total / 2 - text_h(style["combine_label"], font, style) if style["show_matrix"] else centre
    arrow_y = label_y if style["show_matrix"] else centre
    if style["show_matrix"]:
        ax.plot([bar_x, bar_x], [centre, arrow_y], color=fg, lw=0.7)
    draw_arrow(ax, (bar_x, arrow_y), (column_x - pad, arrow_y), fg, lw=0.7)
    ax.text(column_x, label_y + pad / 2, style["combine_label"], ha="left", va="bottom", fontsize=font)
    ax.text(column_x, label_y - pad / 2, style["matching_label"], ha="left", va="top",
            fontsize=small - 0.5, color=muted, linespacing=1.2)
    right = column_x + max(text_w(style["combine_label"], font, style),
                           text_w(style["matching_label"], small - 0.5, style))
    low = -total / 2
    high = max(total / 2 + pad + text_h(composite_text, small - 0.5, style), top,
               label_y + pad + text_h(style["combine_label"], font, style))

    if style["show_matrix"]:
        # (group, marker, triplet index) per channel, in measurement order
        triplet_of = {(g, c): k for g in groups for k, tr in enumerate(triplets[g]) for c in tr if c is not None}
        entries = [(g, data["markers"][c], triplet_of[(g, c)]) for g in groups
                   for c in data["group_indices"][indices[g]]]
        matching_bottom = label_y - pad - text_h(style["matching_label"], small - 0.5, style)
        matrix_right, matrix_top, matrix_bottom = draw_pair_matrix(ax, column_x, matching_bottom - 2 * pad,
                                                                   entries, style)
        right = max(right, matrix_right)
        low, high = min(low, matrix_bottom), max(high, matrix_top)

    ax.set_xlim(-0.05, right + 0.05)
    ax.set_ylim(low - pad, high + 2 * em(font + 3, style))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Loss schematic

    - a: L2, radial profile and texture losses on one experiment group
    - b: colonies on the VGG feature sheet, with zoomed crops (positions illustrative, not computed)
    - c: texture loss over experiment groups (`loss_multi_target.py`)
    - x̂ and d (colony 1 and 2) are two measured control colonies (two replicates); x̂ stands in for a prediction
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_root = mo.ui.text(value=str(Path(os.getenv("DATA_PATH_BASE", "")) / "260726_nca_dataset"),
                              label="Dataset root", full_width=True)
    projection = mo.ui.dropdown(options=["slice", "max", "sum"], value="slice", label="Projection")
    downsample = mo.ui.dropdown(options={"1": 1, "2": 2, "4": 4, "8": 8}, value="4", label="Downsample")
    replicate_a = mo.ui.number(start=1, stop=99, step=1, value=1, label="Replicate for x̂ (colony 1)")
    replicate_b = mo.ui.number(start=1, stop=99, step=1, value=2, label="Replicate for d (colony 2)")
    refresh = mo.ui.checkbox(value=False, label="Ignore cache")
    load_button = mo.ui.run_button(label="Load data")
    mo.vstack([
        mo.md("**Data** (control colonies)"),
        dataset_root,
        mo.hstack([projection, downsample, replicate_a, replicate_b, refresh, load_button], justify="start"),
        mo.md("- Cached data loads without pressing *Load data*"),
    ])
    return (
        dataset_root,
        downsample,
        load_button,
        projection,
        refresh,
        replicate_a,
        replicate_b,
    )


@app.cell(hide_code=True)
def _(
    dataset_root,
    downsample,
    load_button,
    projection,
    refresh,
    replicate_a,
    replicate_b,
):
    _replicates = (int(replicate_a.value), int(replicate_b.value))
    mo.stop(_replicates[0] == _replicates[1], mo.callout("Choose two different replicates.", kind="warn"))
    _name = cache_name("loss_data", projection.value, f"ds{downsample.value}", *_replicates)
    mo.stop(not (load_button.value or (CACHE_DIRECTORY / _name).is_file()),
            mo.md("Data not loaded yet: press *Load data*."))
    with mo.status.spinner("Loading and cleaning the colonies"):
        loss_data = cached(_name, lambda: load_colonies(dataset_root.value, ("ctrl",), EXPERIMENT_GROUPS,
                                                        projection.value, downsample.value, _replicates),
                           refresh=load_button.value and refresh.value)
    replicate_entries = tuple(loss_data["replicates"].index(_r) for _r in _replicates)
    mo.md(f"Loaded: {loss_data['masks'].shape[-1]} px across, channels {', '.join(loss_data['channel_names'])}")
    return loss_data, replicate_entries


@app.cell(hide_code=True)
def _():
    figure_width = mo.ui.dropdown(options={"double column": WIDTH["double"], "single column": WIDTH["single"]},
                                  value="double column", label="Width")
    panels = mo.ui.multiselect(options=["a", "b", "c"], value=["a", "b", "c"], label="Panels")
    same_scale = mo.ui.switch(value=False, label="Same scale in every panel")
    gamma = mo.ui.slider(start=0.3, stop=1.5, step=0.05, value=0.8, label="Image gamma", show_value=True)
    hour = mo.ui.dropdown(options={f"{h}h": h for h in TIMEPOINTS}, value="48h", label="Time")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([figure_width, panels, same_scale, gamma, hour], justify="start"),
    ])
    return figure_width, gamma, hour, panels, same_scale


@app.cell(hide_code=True)
def _():
    group = mo.ui.dropdown(options={GROUP_LABELS[g]: g for g in EXPERIMENT_GROUPS},
                           value=GROUP_LABELS["cell_fate_s1"], label="Group")
    loss_rows = mo.ui.multiselect(options={"L2": "l2", "radial": "radial", "texture": "texture"},
                                  value=["L2", "radial", "texture"], label="Loss rows")
    card_size = mo.ui.slider(start=0.6, stop=1.8, step=0.05, value=1.2, label="Image size", show_value=True)
    row_spacing = mo.ui.slider(start=0.2, stop=1.2, step=0.05, value=0.55, label="Row spacing", show_value=True)
    radial_bins = mo.ui.slider(start=4, stop=32, step=1, value=16, label="Radial bins", show_value=True)
    ring = mo.ui.slider(start=-1, stop=31, step=1, value=10, label="Highlighted ring (-1: none)", show_value=True)
    shade_profiles = mo.ui.switch(value=True, label="Shade profile differences")
    triplet = mo.ui.slider(start=0, stop=1, step=1, value=0, label="Triplet shown", show_value=True)
    muted_notes = mo.ui.switch(value=False, label="Grey notes")
    mo.vstack([
        mo.md("**Panel a** (one group, three losses)"),
        mo.hstack([group, loss_rows, card_size, row_spacing], justify="start"),
        mo.hstack([radial_bins, ring, shade_profiles, triplet, muted_notes], justify="start"),
    ])
    return (
        card_size,
        group,
        loss_rows,
        muted_notes,
        radial_bins,
        ring,
        row_spacing,
        shade_profiles,
        triplet,
    )


@app.cell(hide_code=True)
def _():
    far_hour = mo.ui.dropdown(options={f"{h}h": h for h in TIMEPOINTS}, value="0h", label="Time of d, far")
    zoom_size = mo.ui.slider(start=0.6, stop=2.0, step=0.05, value=1.2, label="Zoom box size", show_value=True)
    crop_size = mo.ui.slider(start=0.05, stop=0.6, step=0.01, value=0.2, label="Front patch (fraction of colony)",
                             show_value=True)
    same_position = mo.ui.switch(value=True, label="Same front patch position in every colony")
    crop_positions = mo.ui.dictionary({
        _key: mo.ui.dictionary({
            "radius": mo.ui.slider(start=0.0, stop=0.9, step=0.05, value=0.5, label="radius", show_value=True),
            "angle": mo.ui.slider(start=0, stop=355, step=5, value=45, label="angle", show_value=True),
        }, label=_label)
        for _key, _label in (("data", "d"), ("prediction", "x̂"), ("time", "d, other time"))
    }, label="Front patch position (radius as a fraction of the colony radius; angle anticlockwise from +x)")
    b_group = mo.ui.dropdown(options={"same as panel a": None,
                                      **{GROUP_LABELS[g]: g for g in EXPERIMENT_GROUPS}},
                             value="same as panel a", label="Group")
    b_triplet = mo.ui.slider(start=0, stop=1, step=1, value=0, label="Triplet (own group)", show_value=True)
    stack_shifts = mo.ui.dictionary({
        _key: mo.ui.dictionary({
            "x": mo.ui.slider(start=-3.0, stop=3.0, step=0.05, value=0.0, label="x", show_value=True),
            "y": mo.ui.slider(start=-3.0, stop=3.0, step=0.05, value=0.0, label="y", show_value=True),
        }, label=_label)
        for _key, _label in (("data", "d"), ("prediction", "x̂"), ("time", "d, other time"))
    }, label="Move patch stacks (data units)")
    stack_depth = mo.ui.slider(start=1, stop=10, step=1, value=4, label="Patches per stack", show_value=True)
    stack_offset = mo.ui.slider(start=0.02, stop=0.3, step=0.01, value=0.1, label="Stack offset", show_value=True)
    stack_seed = mo.ui.number(start=0, stop=10000, step=1, value=0, label="Patch sample seed")
    sample_radius = mo.ui.range_slider(start=0.0, stop=0.95, step=0.05, value=[0.1, 0.8],
                                       label="Sampled patches: radius range", show_value=True)
    vary_scale = mo.ui.switch(value=True, label="Sampled patches at several scales")
    scale_range = mo.ui.range_slider(start=0.25, stop=4.0, step=0.25, value=[0.5, 2.0],
                                     label="Sampled patch size (× front patch)", show_value=True)
    sheet_bulge = mo.ui.slider(start=-0.5, stop=0.6, step=0.05, value=0.15, label="Sheet bulge (negative: bowl)",
                               show_value=True)
    pixelated = mo.ui.switch(value=True, label="Show pixels")
    show_crop_location = mo.ui.switch(value=False, label="Patch location inset")
    mo.vstack([
        mo.md("**Panel b** (feature sheet; panel a's time)"),
        mo.hstack([b_group, b_triplet, far_hour, zoom_size, sheet_bulge], justify="start"),
        mo.hstack([crop_size, same_position, pixelated, show_crop_location], justify="start"),
        crop_positions,
        mo.hstack([stack_depth, stack_offset, stack_seed], justify="start"),
        mo.hstack([sample_radius, vary_scale, scale_range], justify="start"),
        stack_shifts,
    ])
    return (
        b_group,
        b_triplet,
        crop_positions,
        crop_size,
        far_hour,
        pixelated,
        same_position,
        sample_radius,
        scale_range,
        sheet_bulge,
        show_crop_location,
        stack_depth,
        stack_offset,
        stack_seed,
        stack_shifts,
        vary_scale,
        zoom_size,
    )


@app.cell(hide_code=True)
def _():
    shown_groups = mo.ui.multiselect(options={GROUP_LABELS[g]: g for g in EXPERIMENT_GROUPS},
                                     value=[GROUP_LABELS[g] for g in EXPERIMENT_GROUPS], label="Groups")
    shuffle = mo.ui.switch(value=True, label="Random channel shuffle")
    shuffle_seed = mo.ui.number(start=0, stop=1000, step=1, value=0, label="Shuffle draw")
    tile_size = mo.ui.slider(start=0.4, stop=1.2, step=0.05, value=0.65, label="Tile size", show_value=True)
    cell_size = mo.ui.slider(start=0.1, stop=0.35, step=0.01, value=0.18, label="Matrix cell", show_value=True)
    show_state = mo.ui.switch(value=True, label="State column")
    show_matrix = mo.ui.switch(value=True, label="Pair matrix")
    show_excluded = mo.ui.switch(value=True, label="Excluded channels")
    mo.vstack([
        mo.md("**Panel c** (texture over groups; the shuffle shows one random draw)"),
        mo.hstack([shown_groups, shuffle, shuffle_seed, tile_size, cell_size], justify="start"),
        mo.hstack([show_state, show_matrix, show_excluded], justify="start"),
    ])
    return (
        cell_size,
        show_excluded,
        show_matrix,
        show_state,
        shown_groups,
        shuffle,
        shuffle_seed,
        tile_size,
    )


@app.cell(hide_code=True)
def _(
    b_group,
    b_triplet,
    card_size,
    cell_size,
    colony_names,
    crop_positions,
    crop_size,
    far_hour,
    figure_width,
    gamma,
    group,
    hour,
    labels,
    loss_data,
    loss_rows,
    muted_notes,
    panels,
    pixelated,
    radial_bins,
    replicate_entries,
    ring,
    row_spacing,
    same_position,
    same_scale,
    sample_radius,
    scale_range,
    shade_profiles,
    sheet_bulge,
    show_crop_location,
    show_excluded,
    show_matrix,
    show_state,
    shown_groups,
    shuffle,
    shuffle_seed,
    stack_depth,
    stack_offset,
    stack_seed,
    stack_shifts,
    tile_size,
    triplet,
    vary_scale,
    zoom_size,
):
    mo.stop(not panels.value, mo.callout("Select at least one panel.", kind="warn"))
    mo.stop(not loss_rows.value and "a" in panels.value, mo.callout("Select at least one loss row.", kind="warn"))
    mo.stop(not shown_groups.value and "c" in panels.value, mo.callout("Select at least one group.", kind="warn"))
    _text = {k: v.replace("\\n", "\n") for k, v in labels.value.items()}
    loss_style = {
        "theme": THEME,
        "font": FONT["label"],
        "width": float(figure_width.value),
        "same_scale": same_scale.value,
        "gamma": float(gamma.value),
        "hour": int(hour.value),
        "prediction_index": replicate_entries[0],
        "target_index": replicate_entries[1],
        "model_colour": MODEL,
        "loss_colour": ACCENT,
        "ring_colour": SERIES[3],
        "pair_colour": ACCENT,
        "shared_colour": ACCENT,
        "group": group.value,
        "loss_rows": loss_rows.value,
        "card": float(card_size.value),
        "row_spacing": float(row_spacing.value),
        "radial_bins": int(radial_bins.value),
        "ring": min(int(ring.value), int(radial_bins.value) - 1),
        "shade_profiles": shade_profiles.value,
        "triplet": int(triplet.value),
        "error_map": CMAP["intensity"],
        "muted_notes": muted_notes.value,
        "shown_groups": shown_groups.value,
        "shuffle": shuffle.value,
        "shuffle_seed": int(shuffle_seed.value),
        "tile": float(tile_size.value),
        "cell": float(cell_size.value),
        "show_state": show_state.value,
        "show_matrix": show_matrix.value,
        "show_excluded": show_excluded.value,
        "far_hour": int(far_hour.value),
        "zoom": float(zoom_size.value),
        "crop_size": float(crop_size.value),
        "crop_positions": crop_positions.value,
        "same_position": same_position.value,
        "stack_depth": int(stack_depth.value),
        "stack_offset": float(stack_offset.value),
        "stack_seed": int(stack_seed.value),
        "sample_radius": tuple(float(_v) for _v in sample_radius.value),
        "vary_scale": vary_scale.value,
        "scale_range": tuple(float(_v) for _v in scale_range.value),
        "sheet_bulge": float(sheet_bulge.value),
        "b_group": b_group.value,
        "b_triplet": int(b_triplet.value),
        "stack_shifts": stack_shifts.value,
        "pixelated": pixelated.value,
        "show_crop_location": show_crop_location.value,
        "far_colour": SERIES[4],
        "titles": {"a": _text["title_a"], "b": _text["title_b"], "c": _text["title_c"]},
        "names": dict(zip(("prediction", "data"), colony_names.value)),
        **{k: v for k, v in _text.items() if not k.startswith("title_")},
    }
    # Text-heavy panels are redrawn until their scale (points per data unit) settles
    _panel_draw = {"a": draw_comparison_panel, "b": draw_feature_space_panel, "c": draw_groups_panel}
    _draw = {_k: (lambda _f: lambda ax, pt: _f(ax, loss_data, dict(loss_style, pt_per_unit=pt)))(_f)
             for _k, _f in _panel_draw.items()}
    with paper_style():
        loss_schematic = stack_panels(_draw, sorted(panels.value), loss_style["width"], loss_style["titles"],
                                      loss_style["same_scale"], measure_scale=True)
    loss_schematic
    return (loss_schematic,)


@app.cell(hide_code=True)
def _():
    labels = mo.ui.dictionary({
        "title_a": mo.ui.text(value="Comparing a prediction with data", label="Title a"),
        "title_b": mo.ui.text(value="Texture: distances in VGG feature space", label="Title b"),
        "title_c": mo.ui.text(value="Texture loss within experiment groups", label="Title c"),
        "caption_a": mo.ui.text(value="{name}", label="Colony caption a ({name}, {hour})"),
        "caption_b": mo.ui.text(value="{name}, {hour}h", label="Colony caption b ({name}, {hour})"),
        "l2_name": mo.ui.text(value="pixelwise L2", label="L2"),
        "l2_formula": mo.ui.text(value="$\\sum_p (\\hat{x}_p - d_p)^2$", label="L2 formula"),
        "l2_note": mo.ui.text(value="every pixel compared:\\nseparate colonies never\\nline up cell by cell",
                              label="L2 note"),
        "radial_name": mo.ui.text(value="radial profile", label="Radial"),
        "radial_formula": mo.ui.text(value="$\\sum_{c,r} (\\bar{x}_{c,r} - \\bar{d}_{c,r})^2$",
                                     label="Radial formula"),
        "radial_note": mo.ui.text(value="mean of each ring:\\nkeeps the radial pattern,\\nignores positions in a ring",
                                  label="Radial note"),
        "texture_name": mo.ui.text(value="texture", label="Texture"),
        "texture_formula": mo.ui.text(value="$\\sum_l \\|\\phi_l(\\hat{x}) - \\phi_l(d)\\|^2$",
                                      label="Texture formula"),
        "texture_note": mo.ui.text(value="deep image features:\\ncell-scale structure and\\nco-expression in a triplet",
                                   label="Texture note"),
        "error_label": mo.ui.text(value="$(\\hat{x} - d)^2$", label="Error map"),
        "error_high": mo.ui.text(value="high", label="Error bar top"),
        "radius_label": mo.ui.text(value="radius", label="Radius"),
        "ring_mean_label": mo.ui.text(value="ring mean", label="Ring mean"),
        "vgg_label": mo.ui.text(value="VGG", label="VGG"),
        "feature_space_label": mo.ui.text(value="VGG feature space", label="Feature space"),
        "shuffle_note": mo.ui.text(value="(channel order re-drawn\\nevery training step)", label="Shuffle note"),
        "state_label": mo.ui.text(value="NCA state\\n(one channel\\nper marker)", label="State"),
        "select_label": mo.ui.text(value="markers\\nper group", label="Select"),
        "excluded_label": mo.ui.text(value="{}\\nexcluded\\n(artefacts)", label="Excluded ({} = names)"),
        "triplet_label": mo.ui.text(value="RGB {}", label="Triplet ({} = number)"),
        "composite_label": mo.ui.text(value="VGG inputs", label="Composites"),
        "group_loss_label": mo.ui.text(value="$\\mathcal{{L}}_{{{}}}$", label="Group loss ({} = number)"),
        "combine_label": mo.ui.text(value="$\\mathcal{L}_{\\mathrm{tex}}$: mean over groups", label="Combine"),
        "matching_label": mo.ui.text(value="colonies matched\\nseparately in each group", label="Matching"),
        "matrix_title": mo.ui.text(value="channel pairs seen together", label="Matrix title"),
        "key_triplet": mo.ui.text(value="same RGB triplet", label="Key: triplet"),
        "key_group": mo.ui.text(value="same group\\n(over shuffles)", label="Key: group"),
        "key_cross": mo.ui.text(value="different colonies:\\nnever compared", label="Key: cross"),
    })
    colony_names = mo.ui.dropdown(options={"x̂ and d": ("$\\hat{x}$", "$d$"),
                                           "colony 1 and colony 2": ("colony 1", "colony 2")},
                                  value="x̂ and d", label="Colony names (panels a and b)")
    mo.vstack([mo.md("**Labels** (`\\n` for a new line, `$...$` for maths)"),
               colony_names,
               mo.hstack(list(labels.values()), wrap=True, justify="start")])
    return colony_names, labels


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="loss_schematic", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_button = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_button], justify="start")
    return figure_name, save_button, save_svg


@app.cell
def _(figure_name, loss_schematic, save_button, save_svg):
    mo.stop(not save_button.value)
    _paths = save_figure(loss_schematic, figure_name.value, "schematics", svg=save_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
