import numpy as np
import pytest

from Experiments.micropatterns.schematics import composite, locate_image, normalise, project_stack


def test_project_stack_modes():
    stack = np.arange(4 * 2 * 3 * 3, dtype=np.uint16).reshape(4, 2, 3, 3)
    np.testing.assert_array_equal(project_stack(stack, "slice", 2, (0, 4)), stack[2])
    np.testing.assert_array_equal(project_stack(stack, "max", 0, (1, 3)), stack[2])
    np.testing.assert_array_equal(project_stack(stack, "sum", 0, (1, 3)), stack[1] + stack[2])
    with pytest.raises(ValueError):
        project_stack(stack, "mean", 0, (0, 4))


def test_normalise_per_channel_limits():
    image = np.array([[[0.0, 5.0, 10.0]], [[10.0, 20.0, 30.0]]])
    out = normalise(image, np.array([0.0, 10.0]), np.array([10.0, 20.0]))
    np.testing.assert_allclose(out, [[[0.0, 0.5, 1.0]], [[0.0, 1.0, 1.0]]])


def test_composite_adds_colours_on_black():
    channels = np.ones((3, 1, 1))
    colours = [(1.0, 0.0, 0.0), (0.0, 0.5, 0.0), None]
    np.testing.assert_allclose(composite(channels, colours)[0, 0], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(composite(0 * channels, colours)[0, 0], [0.0, 0.0, 0.0])


def test_locate_image_by_projection_free_key(tmp_path):
    source_files = np.empty((1, 2, 1), dtype=object)
    source_files[0, 0, 0] = str(tmp_path / "cell_fate_markers/ctrl_s2_sums/B2_F3_24h_sum.tif")
    source_files[0, 1, 0] = str(tmp_path / "cell_fate_markers/sl24_s2_sums/B3_F4_48h_sum.tif")
    assert locate_image(source_files, tmp_path, "cell_fate_markers/sl24_s2/B3_F4_48h") == (0, 1, 0)
    assert locate_image(source_files, tmp_path, "cell_fate_markers/sl24_s2/B3_F6_48h") is None
