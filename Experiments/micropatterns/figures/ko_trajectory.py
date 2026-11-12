"""True and NCA trajectories of the NODAL knockout conditions, as rows of colony discs.

Used by ``nodal_ko_schematic.py``. ``rollouts`` maps "baseline", "ko_24h" and "ko_0h"
to the dicts returned by ``data.model_rollouts``. Positions are in hours of the time axis.

- Layout "stainings": a track per condition, one disc sector per staining.
- Layout "conditions": a track per staining, one sector per condition (CMY colours).
- ``branch``: the KO 24h track branches off the baseline track at its knockout time.
- Each track holds a row of discs per source (true data, NCA), or the sources go
  in separate framed panels.
"""

from dataclasses import dataclass

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch

from Common.dataloader.cell_types import label_names, marker_values, pixel_label_shares
from Experiments.micropatterns.figures.data import cell_type_rgba, masked_rgba
from Experiments.micropatterns.figures.drawing import composite, draw_colony, draw_sectors, sliced_rgba
from Experiments.micropatterns.figures.style import (
    ACCENT,
    CHANNEL_COLOURS,
    DATA,
    INK,
    MARKER_COLOURS,
    MODEL,
    PAGE,
    SERIES,
    SOURCE_LABELS,
    SectorHandle,
    SectorHandler,
    cell_type_colour,
    unmeasured_patch,
)

# Rollout conditions in track order: label and knockout hour (None: no knockout)
CONDITIONS = {"baseline": ("Baseline", None), "ko_24h": ("KO 24h", 24.0), "ko_0h": ("KO 0h", 0.0)}
# Sector order when there is one sector per condition
SECTOR_CONDITIONS = ("baseline", "ko_0h", "ko_24h")
# Staining names of this figure. Track labels and a legend column per staining
# break at the newline; other legends show them on one line.
TRACK_LABELS = {
    "cell_fate_s1": "Cell fate markers\n(Staining set 1)",
    "cell_fate_s2": "Cell fate markers\n(Staining set 2)",
    "rna_expression": "Morphogen expression",
    "protein_response": "Signalling response",
}
# Row label of each source when a track has true and NCA rows
ROW_LABELS = {"target": "true", "prediction": "model"}
ROUNDING = 1.5  # corner radius of the backgrounds (h)


@dataclass(frozen=True)
class KoTrajectoryOptions:
    """Choices of ``ko_trajectory_figure``."""

    groups: tuple = ("cell_fate_s1",)  # stainings shown
    layout: str = "stainings"  # "stainings" or "conditions" (see module docstring)
    show_true: bool = True
    show_model: bool = True
    cell_type_rules: object = None  # CellTypeRules: colour the NCA by cell type, not by channel
    branch: bool = False  # KO 24h branches off the baseline (layout "stainings" only)
    panels: str = "rows"  # "rows": true then NCA rows; "panels": side by side; "stacked": true data on top
    hatch: bool = False  # hatch stainings without a measured image
    lmbr: bool = False  # LMBR in yellow in Fate S2 (whose SOX17 is then left out)
    gamma: float = 1.0
    font_size: float = 7.0  # pt
    colony_size: float = 9.0  # h; the time ticks are 12h apart
    track_gap: float = 5.0  # h
    start_height: float = 0.5  # branch view: 0 is level with KO 24h, 1 with the later baseline
    connectors: str = "arrows"  # "arrows", "flow" (Sankey-like backgrounds) or "both"
    true_colour: str = DATA
    model_colour: str = MODEL
    baseline_band: object = SERIES[2]  # background colours (None: no background)
    ko_band: object = ACCENT
    band_alpha: float = 0.12
    legend: str = "auto"  # "auto" (corner of the branch view, else right), "gap", "right", "top", "bottom"
    legend_shape: str = "auto"  # "auto" (by position), "rows", "columns" or "single"
    legend_style: str = "black"  # "black" (like the images) or "white"
    legend_font: float = 6.5  # pt
    dpi: int = 200

    @property
    def sources(self):
        """Sources shown, true data first (top or left of the NCA)."""
        return tuple(s for s, shown in (("target", self.show_true), ("prediction", self.show_model)) if shown)

    @property
    def source_colours(self):
        return {"target": self.true_colour, "prediction": self.model_colour}


def _marker(name):
    return name.rsplit("/", 1)[-1]


def _group(name):
    return name.split("/")[0]


def channel_colours(names, groups, by_condition=False, lmbr=False):
    """Colour (or None: not drawn) of each channel in ``names``, per staining in ``groups``.

    One sector per staining: the marker colours of ``style.py``. One sector per
    condition: cyan, magenta, yellow in channel order within each staining, as in
    ``data_cleaning.py``. LMBR is left out, except in Fate S2 with ``lmbr``.
    """
    colours = {}
    for group in groups:
        palette = iter([(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)])
        group_colours = []
        for name in names:
            is_lmbr = _marker(name) == "LMBR"
            if _group(name) != group or (is_lmbr and not (lmbr and group == "cell_fate_s2")):
                group_colours.append(None)
            elif by_condition:
                group_colours.append(next(palette, (1.0, 1.0, 1.0)))
            elif is_lmbr:
                group_colours.append(CHANNEL_COLOURS["yellow"])
            else:
                group_colours.append(CHANNEL_COLOURS[MARKER_COLOURS.get(_marker(name), "grey")])
        colours[group] = group_colours
    return colours


# --- Tracks --------------------------------------------------------------------------


def build_tracks(rollouts, names, groups, colours, type_colours, opts):
    """Tracks (one per condition, or per staining), each with a tile function per source.

    A tile function maps time index i to (RGBA, unmeasured flag of each sector).
    """
    rules = opts.cell_type_rules
    cell_types = rules is not None
    # Sector per condition: every condition is cropped to the box around all
    # colonies, so the sectors line up.
    union = np.any([rollouts[c]["boundary"] for c in SECTOR_CONDITIONS], axis=0)

    def channel_tile(image, mask, group_colours, crop_mask=None):
        rgb = composite(np.clip(image, 0.0, 1.0) ** opts.gamma, group_colours)
        return masked_rgba(rgb, mask, crop_mask)

    def type_tile(image, mask, crop_mask=None):
        shares = pixel_label_shares(marker_values(image, names, rules, axis=0), rules, mask)
        return cell_type_rgba(shares, type_colours, mask, crop_mask)

    def unmeasured(condition, source, i, group):
        """True where staining ``group`` of the true data has no image at time index i."""
        if source != "target" or group is None:
            return False
        flags = rollouts[condition]["measured"][i]
        return not any(f for n, f in zip(names, flags) if _group(n) == group)

    def condition_tile(condition, source):
        rollout = rollouts[condition]

        def tile(i):
            image, mask = rollout[source][i], rollout["boundary"]
            if cell_types and source == "prediction":
                return type_tile(image, mask), [False]
            tiles = [channel_tile(image, mask, colours[g]) for g in groups]
            return sliced_rgba(tiles), [unmeasured(condition, source, i, g) for g in groups]
        return tile

    def sliced(source, make, group=None):
        def tile(i):
            rgba = sliced_rgba([make(rollouts[c][source][i], rollouts[c]["boundary"]) for c in SECTOR_CONDITIONS])
            return rgba, [unmeasured(c, source, i, group) for c in SECTOR_CONDITIONS]
        return tile

    tracks = []
    if opts.layout != "conditions":
        for condition, (label, ko) in CONDITIONS.items():
            tracks.append({
                "key": condition, "label": label, "colour": INK if ko is None else ACCENT, "ko": ko,
                "entries": {s: condition_tile(condition, s) for s in opts.sources},
            })
        return tracks
    for group in groups:
        make = lambda image, mask, group=group: channel_tile(image, mask, colours[group], union)
        tracks.append({
            "key": group, "label": TRACK_LABELS.get(group, group), "colour": INK, "ko": None,
            "entries": {s: sliced(s, make, group) for s in opts.sources if s == "target" or not cell_types},
        })
    if cell_types and opts.show_model:
        # Last, so the NCA sits below the true data
        tracks.append({
            "key": "cell_types", "label": "Cell types", "colour": INK, "ko": None,
            "entries": {"prediction": sliced("prediction", lambda image, mask: type_tile(image, mask, union))},
        })
    return tracks


def split_panels(tracks, opts):
    """(title source or None, tracks) of each panel: one with every source, or one per source."""
    sources = opts.sources
    separate = opts.panels != "rows" and len(sources) > 1
    panels = []
    for panel_sources in ([(s,) for s in sources] if separate else [sources]):
        panel_tracks = [
            {**t, "entries": [(s, t["entries"][s]) for s in panel_sources if s in t["entries"]]} for t in tracks
        ]
        panel_tracks = [t for t in panel_tracks if t["entries"]]
        if panel_tracks:
            panels.append((panel_sources[0] if separate else None, panel_tracks))
    return panels


# --- Placement -------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Sizes in hours of the time axis."""

    times: tuple
    s: float  # colony diameter
    font: float  # pt
    track_gap: float

    @property
    def text(self):
        """Scale of the text spacing, which grows with the font size."""
        return self.font / 7.0

    @property
    def sub_step(self):
        """Distance between the true and NCA rows of one track."""
        return self.s + 1.5

    @property
    def end(self):
        return self.times[-1]

    def height(self, track):
        """Distance from the first to the last row of a track."""
        return (len(track["entries"]) - 1) * self.sub_step


def _segment(track, indices, y, band, side="left"):
    """A track drawn over some time indices, with its first row at height y."""
    return {**track, "indices": indices, "y": y, "band": band, "side": side}


def _band(start, stop, colour, opts):
    return None if colour is None else (start, stop, colour, opts.band_alpha)


def _track_band(ko, geo, opts):
    """Background of a condition track: from the knockout, or all of the baseline.
    Staining tracks (sector per condition) have none."""
    if opts.layout == "conditions":
        return None
    if ko is None:
        return _band(geo.times[0], geo.end, opts.baseline_band, opts)
    return _band(ko, geo.end, opts.ko_band, opts)


def place_tracks(tracks, geo, opts, branch):
    """Segments and branch arrows of one panel."""
    times, s = geo.times, geo.s
    every = list(range(len(times)))
    branched = [t for t in tracks if t["ko"] is not None and t["ko"] > times[0]]
    baseline = next((t for t in tracks if t["key"] == "baseline"), None)
    if not (branch and branched and baseline):
        segments, y = [], 0.0
        for track in tracks:
            segments.append(_segment(track, every, y, _track_band(track["ko"], geo, opts)))
            y -= geo.height(track) + s + geo.track_gap
        return segments, []

    # Baseline splits at the first knockout after 0h: later baseline on top,
    # then the knockouts, all evenly spaced; the shared start sits between the
    # later baseline and the first knockout (see start_height).
    others = [t for t in tracks if t not in branched and t["key"] != "baseline"]
    split = min(t["ko"] for t in branched)
    before = [i for i, t in enumerate(times) if t < split]
    segments = [_segment(baseline, [i for i in every if times[i] >= split], 0.0,
                         _band(split, geo.end, opts.baseline_band, opts), "right")]
    for track in branched + others:
        y = segments[-1]["y"] - geo.height(segments[-1]) - s - geo.track_gap
        if track in branched:
            after = [i for i in every if times[i] >= track["ko"]]
            segments.append(_segment(track, after, y, _track_band(track["ko"], geo, opts), "right"))
        else:
            segments.append(_segment(track, every, y, _track_band(track["ko"], geo, opts)))
    posts = segments[:1 + len(branched)]

    def middle(segment):
        return segment["y"] - geo.height(segment) / 2

    low, high = middle(posts[1]), middle(posts[0])
    centre = low + opts.start_height * (high - low)
    pre_half = geo.height(baseline) / 2 + s / 2
    segments.insert(0, _segment(baseline, before, centre + geo.height(baseline) / 2,
                                _band(times[before[0]], times[before[-1]], opts.baseline_band, opts)))
    # From above or below the last shared colony, turning towards each post;
    # straight across when a post is level with it.
    arrows = []
    for post in posts:
        target_y = middle(post)
        stop_x = times[post["indices"][0]] - s / 2 - 0.6
        if abs(target_y - centre) >= pre_half + 2.1:
            side = 1 if target_y > centre else -1
            start = (times[before[-1]], centre + side * (pre_half + 0.9))
            arrows.append((start, (stop_x, target_y), "angle,angleA=90,angleB=0,rad=8"))
        else:
            level = (target_y + centre) / 2
            arrows.append(((times[before[-1]] + s / 2 + 0.6, level), (stop_x, level), "arc3,rad=0"))
    return segments, arrows


# --- Drawing ---------------------------------------------------------------------------


def _band_box(segment, geo):
    """(left, right, bottom, top) of a segment's background."""
    start, stop, _, _ = segment["band"]
    s, first = geo.s, segment["y"]
    return start - s / 2 - 0.4, stop + s / 2 + 0.4, first - geo.height(segment) - s / 2 - 0.6, first + s / 2 + 0.6


def _band_colour(segment):
    """Background colour blended onto the page, so overlaps don't darken."""
    _, _, colour, alpha = segment["band"]
    return alpha * np.asarray(to_rgb(colour)) + (1 - alpha) * np.asarray(to_rgb(PAGE))


def draw_flows(ax, pre, posts, geo):
    """Sankey-like ribbons from the shared start's background to each post's, splitting
    its height between them and shading between their colours."""
    if pre["band"] is None:
        return
    _, right, pre_bottom, pre_top = _band_box(pre, geo)
    boxes = [_band_box(p, geo) if p["band"] is not None else None for p in posts]
    total = sum(b[3] - b[2] for b in boxes if b is not None)
    top = pre_top
    for post, box in zip(posts, boxes):
        if box is None:
            continue
        left, _, post_bottom, post_top = box
        bottom = top - (pre_top - pre_bottom) * (post_top - post_bottom) / total
        middle = (right + left) / 2
        # Square ends reach under the rounded corners of both backgrounds
        path = mpath.Path(
            [
                (right - ROUNDING, top), (right, top),
                (middle, top), (middle, post_top), (left, post_top),
                (left + ROUNDING, post_top), (left + ROUNDING, post_bottom), (left, post_bottom),
                (middle, post_bottom), (middle, bottom), (right, bottom),
                (right - ROUNDING, bottom), (right - ROUNDING, top),
            ],
            [mpath.Path.MOVETO, mpath.Path.LINETO,
             mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
             mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO,
             mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY],
        )
        xs = np.linspace(right - ROUNDING, left + ROUNDING, 128)
        mix = np.clip((xs - right) / (left - right), 0.0, 1.0)
        mix = mix * mix * (3 - 2 * mix)  # smooth start and end
        colours = (1 - mix[:, None]) * _band_colour(pre) + mix[:, None] * _band_colour(post)
        image = ax.imshow(colours[None], extent=(xs[0], xs[-1], min(bottom, post_bottom), max(top, post_top)),
                          interpolation="bilinear", zorder=0)
        image.set_clip_path(PathPatch(path, transform=ax.transData))
        top = bottom


def draw_track_labels(ax, segment, rows_y, geo, opts):
    """Rotated track label beside the first (left) or last (right) colony; with true and
    NCA rows, a label per row and a bracket."""
    s, text = geo.s, geo.text
    left = segment["side"] == "left"
    sign = -1 if left else 1
    ha = "right" if left else "left"
    x = geo.times[segment["indices"][0 if left else -1]] + sign * s / 2 + sign * text
    if len(rows_y) > 1:
        for (source, _), row_y in zip(segment["entries"], rows_y):
            ax.text(x, row_y, ROW_LABELS[source], rotation=90, ha=ha, va="center", fontsize=geo.font - 0.5,
                    color=opts.source_colours[source])
        x += sign * 2.4 * text
        ax.plot([x, x], [rows_y[-1] - s / 2 + 0.5, rows_y[0] + s / 2 - 0.5], color=segment["colour"], lw=0.7)
        x += sign * 0.6 * text
    ax.text(x, (rows_y[0] + rows_y[-1]) / 2, segment["label"], rotation=90, ha=ha, va="center",
            fontsize=geo.font + 0.5, color=segment["colour"])


def draw_panel(ax, title, segments, arrows, frame, geo, opts, branch):
    """One panel of tracks under a time axis; returns True if a sector was hatched.

    ``frame`` holds the panel's ``left``, ``right``, ``top`` and ``bottom`` and the
    width of the name ``bar`` of a titled panel.
    """
    s, times = geo.s, geo.times
    left, right, bottom = frame["left"], frame["right"], frame["bottom"]
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, frame["top"])
    ax.set_aspect("equal")
    ax.axis("off")
    axis_y = s / 2 + 1.5
    ax.plot([times[0] - s / 2, geo.end + s / 2], [axis_y] * 2, color=INK, lw=0.7)
    for t in times:
        ax.plot([t, t], [axis_y, axis_y + 0.8], color=INK, lw=0.7)
        ax.text(t, axis_y + 1.2, f"{t:g}h", ha="center", va="bottom")
    if title is not None:
        # Frame with the source named in a bar down its left side
        colour = opts.source_colours[title]
        frame_top, frame_bottom = s / 2 + 1.0, bottom + 0.5
        ax.add_patch(FancyBboxPatch((left + 0.5, frame_bottom), right - left - 1.0, frame_top - frame_bottom,
                                    boxstyle="round,pad=0,rounding_size=1.5", fill=False, edgecolor=colour, lw=1.2))
        ax.add_patch(FancyBboxPatch((left + 0.5, frame_bottom), frame["bar"], frame_top - frame_bottom,
                                    boxstyle="round,pad=0,rounding_size=1.5", facecolor=colour, edgecolor=colour,
                                    lw=1.2))
        ax.text(left + 0.5 + frame["bar"] / 2, (frame_top + frame_bottom) / 2, SOURCE_LABELS[title], rotation=90,
                ha="center", va="center", color="white", fontweight="bold", fontsize=geo.font + 1)

    hatched = False
    for segment in segments:
        entries, y = segment["entries"], segment["y"]
        rows_y = [y - k * geo.sub_step for k in range(len(entries))]
        draw_track_labels(ax, segment, rows_y, geo, opts)
        if segment["band"] is not None:
            x0, x1, y0, y1 = _band_box(segment, geo)
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={ROUNDING}",
                                        facecolor=_band_colour(segment), edgecolor="none", zorder=0))
        for (_, tile), row_y in zip(entries, rows_y):
            for i in segment["indices"]:
                centre = (times[i], row_y)
                rgba, unmeasured = tile(i)
                draw_colony(ax, rgba, centre, s)
                hatched |= draw_sectors(ax, centre, s / 2, unmeasured, hatch=opts.hatch)
    if branch and opts.connectors != "arrows":
        draw_flows(ax, segments[0], [seg for seg in segments if seg["side"] == "right"], geo)
    if opts.connectors != "flow":
        for start, stop, connection in arrows:
            ax.annotate("", xy=stop, xytext=start, arrowprops={
                "arrowstyle": "-|>,head_length=0.5,head_width=0.25", "color": INK, "lw": 0.9,
                "shrinkA": 0, "shrinkB": 0, "connectionstyle": connection,
            })
    return hatched


# --- Legend ----------------------------------------------------------------------------


def legend_rows(names, groups, colours, type_colours, hatched, opts):
    """Legend entries, a row each: the sectors (per condition), cell type colours, then
    each staining's channel colours."""
    white = opts.legend_style == "white"
    by_condition = opts.layout == "conditions"
    # On white, colour swatches get a black outline
    outline = [patheffects.Stroke(linewidth=3.4, foreground="black"), patheffects.Normal()] if white else []
    rows = []
    if by_condition:
        rows.append([(SectorHandle(k, len(SECTOR_CONDITIONS)), CONDITIONS[c][0])
                     for k, c in enumerate(SECTOR_CONDITIONS)])
    if type_colours:
        rows.append([(Patch(facecolor=to_rgb(c), edgecolor="black" if white else "none", lw=0.6), label)
                     for label, c in type_colours.items()])
    if not type_colours or opts.show_true:
        for k, group in enumerate(groups):
            header = (SectorHandle(k, len(groups)) if not by_condition and len(groups) > 1
                      else plt.Line2D([], [], lw=0))
            rows.append([(header, TRACK_LABELS.get(group, group))] + [
                (plt.Line2D([], [], color=c, lw=2, path_effects=outline), _marker(name))
                for name, c in zip(names, colours[group]) if c is not None
            ])
    if hatched:
        rows.append([(unmeasured_patch(), "not measured")])
    return rows


def add_legend(figure, rows, shape, opts, **placement):
    """Legend on black like the images (or on white), with a row per staining ("rows"),
    a column per staining ("columns") or a single column."""
    white = opts.legend_style == "white"
    length = max(len(r) for r in rows)
    blank = (plt.Line2D([], [], lw=0), "")
    padded = [r + [blank] * (length - len(r)) for r in rows]
    # Legend columns fill top to bottom, so entries go in column by column
    if shape == "rows":
        width, entries = length, [r[c] for c in range(length) for r in padded]
    elif shape == "columns":
        width, entries = len(padded), [e for r in padded for e in r]
    else:
        width, entries = 1, [e for r in rows for e in r]
    if shape != "columns":
        entries = [(h, label.replace("\n", " ")) for h, label in entries]
    return figure.legend(
        [h for h, _ in entries], [label for _, label in entries], ncol=width, fontsize=opts.legend_font,
        handlelength=1.2, columnspacing=0.8, labelcolor=INK if white else "white", frameon=not white,
        fancybox=False, facecolor="black", edgecolor="black", framealpha=1.0,
        handler_map={SectorHandle: SectorHandler(fill="#666666", edge="#222222") if white else SectorHandler()},
        **placement,
    )


# --- Figure ----------------------------------------------------------------------------


def ko_trajectory_figure(rollouts, opts=KoTrajectoryOptions()):
    """Figure of the true and NCA trajectories of every condition; see the module docstring."""
    first = rollouts["baseline"]
    names = list(first["channel_names"])
    times = tuple(float(t.rstrip("h")) for t in first["time_labels"])
    groups = [g for g in dict.fromkeys(_group(n) for n in names if "/" in n) if g in opts.groups]
    if not groups:
        raise ValueError("None of the chosen stainings are in the rollouts")
    by_condition = opts.layout == "conditions"
    branch = opts.branch and not by_condition
    colours = channel_colours(names, groups, by_condition, opts.lmbr)
    rules = opts.cell_type_rules
    type_colours = {} if rules is None else {label: cell_type_colour(label) for label in label_names(rules)}
    tracks = build_tracks(rollouts, names, groups, colours, type_colours, opts)
    panels = split_panels(tracks, opts)
    if not panels:
        raise ValueError("Nothing to draw: show the true data, or colour the NCA by channels")
    separate = panels[0][0] is not None

    geo = Geometry(times, opts.colony_size, opts.font_size, opts.track_gap)
    s, text = geo.s, geo.text
    # Tracks with true and NCA rows get a label per row, and each extra line of
    # a (rotated) track label needs more room
    two_rows = any(len(t["entries"]) > 1 for _, panel_tracks in panels for t in panel_tracks)
    label_lines = max(t["label"].count("\n") + 1 for t in tracks)
    label_room = (6.0 if two_rows else 3.0) * text + 1.6 * text * (label_lines - 1)

    placed = [(title, *place_tracks(panel_tracks, geo, opts, branch)) for title, panel_tracks in panels]
    bottoms = [min(seg["y"] - geo.height(seg) for seg in segs) - s / 2 - 1.5 for _, segs, _ in placed]
    top = s / 2 + 1.5 + 3.0 * text  # top of the time axis labels
    bar = 3.5 * text  # width of a panel's name bar
    left = times[0] - s / 2 - label_room - (bar + 1.5 if separate else 0.5)
    right = geo.end + s / 2 + (label_room if branch else 0.0) + 1.0
    panel_width, panel_gap = right - left, 2.0
    unit = 7.0 / panel_width  # inches per hour, whatever the number of panels

    # Axes boxes (x, y, bottom) in hours: side by side share one height, stacked
    # panels each fit their own tracks.
    if opts.panels == "stacked" and separate:
        figure_width = panel_width
        heights = [top - b for b in bottoms]
        figure_height = sum(heights) + panel_gap * (len(placed) - 1)
        boxes, cursor = [], figure_height
        for b, h in zip(bottoms, heights):
            boxes.append((0.0, cursor - h, b))
            cursor -= h + panel_gap
    else:
        figure_width = len(placed) * panel_width + (len(placed) - 1) * panel_gap
        figure_height = top - min(bottoms)
        boxes = [(p * (panel_width + panel_gap), 0.0, min(bottoms)) for p in range(len(placed))]

    with plt.rc_context({"font.size": opts.font_size, "font.family": "sans-serif", "text.color": INK}):
        # Panels first; the figure is resized around them once the legend's size is known
        figure = plt.figure(figsize=(unit * figure_width, unit * figure_height), facecolor=PAGE, dpi=opts.dpi)
        axes, hatched = [], False
        for (title, segments, arrows), (x, y, bottom) in zip(placed, boxes):
            ax = figure.add_axes((x / figure_width, y / figure_height, panel_width / figure_width,
                                  (top - bottom) / figure_height))
            frame = {"left": left, "right": right, "top": top, "bottom": bottom, "bar": bar}
            hatched |= draw_panel(ax, title, segments, arrows, frame, geo, opts, branch)
            axes.append((ax, x, y, top - bottom))

        rows = legend_rows(names, groups, colours, type_colours, hatched, opts)
        corner = None
        if branch:
            # Above the shared baseline start of the first panel, left of its
            # arrows and below the time axis
            pre = placed[0][1][0]
            corner = (left + 0.5 + (bar + 0.5 if separate else 0.0), times[pre["indices"][-1]] - 1.0,
                      pre["y"] + s / 2 + 1.2, s / 2 + 1.0)
        _place_legend(figure, axes, rows, corner, unit, panel_width, opts)
    return figure


def _place_legend(figure, axes, rows, corner, unit, panel_width, opts):
    """Legend in the corner of the branch view (if it fits), or beside the panels, with
    the figure resized around it. ``corner`` is (left, right, bottom, top) in hours."""

    def inches(legend):
        box = legend.get_window_extent(figure.canvas.get_renderer())
        return box.width / figure.dpi, box.height / figure.dpi

    panels_size = tuple(figure.get_size_inches())
    size, offset = panels_size, (0.0, 0.0)  # figure, and the panels' corner in it (in)
    pad = 0.08
    position, chosen = opts.legend, opts.legend_shape
    if position in ("auto", "gap") and corner is not None:
        gap_left, gap_right, gap_bottom, gap_top = corner
        # Whichever shape fits; a forced corner keeps the last one tried
        shapes = ("rows", "columns") if chosen == "auto" else (chosen,)
        for shape in shapes:
            legend = add_legend(figure, rows, shape, opts, loc="upper left", bbox_to_anchor=(gap_left, gap_top),
                                bbox_transform=axes[0][0].transData)
            w, h = inches(legend)
            if w <= unit * (gap_right - gap_left) and h <= unit * (gap_top - gap_bottom):
                position = "gap"
                break
            if shape != shapes[-1] or position != "gap":
                legend.remove()
        if position != "gap":
            position = "right"
    elif position in ("auto", "gap"):
        position = "right"  # no corner without the branch view
    if position == "right":
        legend = add_legend(figure, rows, "single" if chosen == "auto" else chosen, opts, loc="upper left")
        w, h = inches(legend)
        size = (panels_size[0] + w + 2 * pad, max(panels_size[1], h + 2 * pad))
        offset = (0.0, size[1] - panels_size[1])
        anchor = ((panels_size[0] + pad) / size[0], 1.0 - pad / size[1])
    elif position in ("top", "bottom"):
        on_top = position == "top"
        legend = add_legend(figure, rows, "rows" if chosen == "auto" else chosen, opts,
                            loc="upper center" if on_top else "lower center")
        w, h = inches(legend)
        size = (panels_size[0], panels_size[1] + h + 2 * pad)
        offset = (0.0, 0.0 if on_top else h + 2 * pad)
        anchor = (0.5, 1.0 - pad / size[1] if on_top else pad / size[1])
    figure.set_size_inches(*size)
    for ax, x, y, h in axes:
        ax.set_position(((offset[0] + unit * x) / size[0], (offset[1] + unit * y) / size[1],
                         unit * panel_width / size[0], unit * h / size[1]))
    if position != "gap":
        legend.set_bbox_to_anchor(anchor, transform=figure.transFigure)
