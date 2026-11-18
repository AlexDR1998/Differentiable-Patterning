"""Video of one NCA rolled out under the NODAL knockout conditions, for slides.

Used by ``nodal_ko_schematic.py`` beside its static figure. ``rollouts`` maps
"baseline", "ko_24h" and "ko_0h" to the dicts of ``data.model_rollouts`` with
``frames_per_12h`` set, so the NCA has many frames between observations.

The video is one tall column with a row per condition. Stainings are either
sectors of one disc (as in the static figure) or a column each. Colours are the
static figure's (``ko_trajectory.channel_colours``). Frames are written by
``Experiments/snowmelt/video.py:render_mp4``.
"""

from itertools import permutations

import numpy as np

from Experiments.micropatterns.figures.data import crop_box
from Experiments.micropatterns.figures.drawing import composite, draw_sectors, rows_to_columns, sector_map
from Experiments.micropatterns.figures.ko_trajectory import CONDITIONS, channel_colours
from Experiments.micropatterns.figures.style import GROUP_LABELS, SectorHandle, legend_options, legend_swatch
from Experiments.snowmelt.video import Panel

# Every order of the three conditions, top to bottom, labelled e.g. "Baseline → KO 24h → KO 0h"
CONDITION_ORDERS = {" → ".join(CONDITIONS[c][0] for c in order): order for order in permutations(CONDITIONS)}
# Images are drawn on [-0.5, 0.5] in both directions, so the disc centre is 0 and its radius 0.5
EXTENT = (-0.5, 0.5, -0.5, 0.5)


def _frames_rgb(prediction, crop, colours, gamma):
    """RGB [F, h, w, 3] of the NCA frames [F, C, H, W] in one staining's colours, cropped to the colony."""
    return np.stack([composite(np.clip(frame[:, crop[0], crop[1]], 0.0, 1.0) ** gamma, colours)
                     for frame in prediction])


def ko_video_panels(rollouts, order, groups, sectors=True, gamma=1.0, lmbr=False, hold_frames=0):
    """Panels for ``render_mp4``, its column count and the title of each frame.

    A row per condition in ``order``; with ``sectors`` one disc per row with a
    sector per staining in ``groups``, else a column per staining. The last frame
    is repeated ``hold_frames`` times so the video pauses on it.
    """
    names = rollouts[order[0]]["channel_names"]
    colours = channel_colours(names, groups, lmbr=lmbr)
    panels = []
    for condition in order:
        rollout = rollouts[condition]
        mask = np.asarray(rollout["boundary"], dtype=bool)
        crop = crop_box(mask)
        prediction = np.asarray(rollout["prediction"])
        if hold_frames:
            prediction = np.concatenate([prediction, np.repeat(prediction[-1:], hold_frames, axis=0)])
        rgbs = [_frames_rgb(prediction, crop, colours[g], gamma) for g in groups]
        label = CONDITIONS[condition][0]
        if sectors:
            # Sector k of each frame from staining k, as drawing.sliced_rgba does for one image
            index = sector_map(rgbs[0].shape[1:3], len(groups))
            rgb = np.take_along_axis(np.stack(rgbs), index[None, None, ..., None], axis=0)[0]
            panels.append(Panel(title=label, frames=rgb, mask=mask[crop]))
        else:
            panels += [Panel(title=f"{label} · {GROUP_LABELS.get(g, g)}", frames=rgb, mask=mask[crop])
                       for g, rgb in zip(groups, rgbs)]
    hours = rollouts[order[0]]["frame_hours"]
    titles = [f"{h:.0f} h" for h in hours] + [f"{hours[-1]:.0f} h"] * hold_frames
    return panels, (1 if sectors else len(groups)), titles


def sector_lines(count):
    """``draw_outline`` for ``render_mp4``: white lines between ``count`` sectors, as in the static figure."""
    def draw(ax, mask, extent):
        if count > 1:
            draw_sectors(ax, (0.0, 0.0), 0.5, [False] * count)
    return draw


def legend_decorator(names, groups, sectors=True, lmbr=False, font_size=9):
    """``decorate`` for ``render_mp4``: a legend under the panels, a row per staining with its
    sector (when stainings are sectors) and the colour of each marker."""
    colours = channel_colours(names, groups, lmbr=lmbr)

    def decorate(fig):
        import matplotlib.pyplot as plt

        rows = []
        for k, group in enumerate(groups):
            header = SectorHandle(k, len(groups)) if sectors and len(groups) > 1 else plt.Line2D([], [], lw=0)
            rows.append([(header, GROUP_LABELS.get(group, group))] + [
                (legend_swatch(c, "white"), n.rsplit("/", 1)[-1]) for n, c in zip(names, colours[group])
                if c is not None
            ])
        handles, labels, ncol = rows_to_columns(rows)
        fig.legend(handles, labels, ncol=ncol, loc="outside lower center",
                   **{**legend_options("white"), "fontsize": font_size})

    return decorate


__all__ = ["CONDITION_ORDERS", "EXTENT", "ko_video_panels", "legend_decorator", "sector_lines"]
