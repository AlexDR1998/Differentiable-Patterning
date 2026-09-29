import numpy as np
import pytest

from Experiments.snowmelt.video import Panel, composite, render_mp4, stretch


def test_stretch_clips_to_unit_range():
    np.testing.assert_allclose(stretch(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0), [0.0, 0.5, 1.0])


def test_composites_map_channels_to_colours():
    one, zero = np.ones((2, 2)), np.zeros((2, 2))

    rgb = composite([one, None, zero], "rgb")
    cmy = composite([one, None, None], "cmy")

    np.testing.assert_array_equal(rgb[0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(cmy[0, 0], [0.0, 1.0, 1.0])  # full cyan on white
    with pytest.raises(ValueError):
        composite([one, one, one], "hsv")


def test_render_mp4_writes_h264_video(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = np.random.default_rng(0).uniform(0, 1, (3, 6, 6)).astype(np.float32)
    mask = np.ones((6, 6), dtype=bool)
    panels = [
        Panel("mapped", frames, mask, cmap="viridis"),
        None,
        Panel("mixed", composite([frames, frames, None], "cmy"), mask),
    ]

    path = render_mp4(tmp_path / "video.mp4", panels, ["a", "b", "c"], extent=(0, 1, 0, 1), fps=4)

    assert path.stat().st_size > 0
