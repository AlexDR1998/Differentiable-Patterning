import numpy as np
import pytest

from Experiments.micropatterns.figures.ko_trajectory import CONDITIONS
from Experiments.micropatterns.figures.ko_video import (
    CONDITION_ORDERS,
    EXTENT,
    ko_video_panels,
    legend_decorator,
    sector_lines,
)

NAMES = ("cell_fate_s1/SOX17", "cell_fate_s1/SOX2", "cell_fate_s1/TBXT", "cell_fate_s2/FOXA2",
         "cell_fate_s2/TBXT", "cell_fate_s2/SOX17", "cell_fate_s2/LMBR")


def _rollouts(frames=5, size=16):
    rows, cols = np.indices((size, size))
    boundary = np.hypot(rows - size / 2, cols - size / 2) < size / 2 - 2
    rng = np.random.default_rng(0)
    return {
        condition: {
            "prediction": rng.uniform(size=(frames, len(NAMES), size, size)),
            "boundary": boundary,
            "channel_names": NAMES,
            "frame_hours": tuple(np.linspace(0, 48, frames)),
        }
        for condition in CONDITIONS
    }


def test_condition_orders_cover_every_order():
    assert len(CONDITION_ORDERS) == 6
    assert all(sorted(order) == sorted(CONDITIONS) for order in CONDITION_ORDERS.values())


def test_sectors_give_one_column_and_columns_one_per_staining():
    rollouts = _rollouts()
    order = ("ko_0h", "baseline", "ko_24h")
    groups = ["cell_fate_s1", "cell_fate_s2"]
    panels, columns, titles = ko_video_panels(rollouts, order, groups, sectors=True, hold_frames=3)
    assert columns == 1 and [p.title for p in panels] == ["KO 0h", "Baseline", "KO 24h"]
    assert panels[0].frames.shape[0] == 8 and len(titles) == 8 and titles[-1] == "48 h"
    assert panels[0].frames.shape[1:3] == panels[0].mask.shape and panels[0].frames.shape[-1] == 3

    panels, columns, _ = ko_video_panels(rollouts, order, groups, sectors=False)
    assert columns == 2 and len(panels) == 6 and panels[1].title.startswith("KO 0h")


def test_render_writes_video(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from Experiments.snowmelt.video import render_mp4

    rollouts = _rollouts(frames=3)
    groups = ["cell_fate_s1", "cell_fate_s2"]
    panels, columns, titles = ko_video_panels(rollouts, tuple(CONDITIONS), groups)
    path = render_mp4(tmp_path / "ko.mp4", panels, titles, EXTENT, fps=4, ncols=columns,
                      draw_outline=sector_lines(len(groups)), panel_size=(1.5, 1.7), show_axes=False,
                      background="white", decorate=legend_decorator(NAMES, groups))
    assert path.stat().st_size > 0
