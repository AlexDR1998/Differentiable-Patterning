import numpy as np

from Experiments.micropatterns.nodal_ko_schematic import colony_rgba, sliced_rgba


def test_colony_rgba_crops_to_mask_and_is_transparent_outside():
    image = np.zeros((2, 6, 6))
    image[0, 2:4, 2:5] = 1.0
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    mask[1, 1] = False
    rgba = colony_rgba(image, mask, [(1.0, 0.0, 0.0), None])
    assert rgba.shape == (4, 4, 4)
    np.testing.assert_array_equal(rgba[..., 3], mask[1:5, 1:5])
    np.testing.assert_allclose(rgba[1, 1], [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rgba[0, 1, :3], 0.0)


def test_sliced_rgba_two_groups_are_left_and_right_halves():
    left, right = np.zeros((4, 4, 4)), np.ones((4, 4, 4))
    out = sliced_rgba([left, right])
    np.testing.assert_array_equal(out[:, :2], 0.0)
    np.testing.assert_array_equal(out[:, 2:], 1.0)
