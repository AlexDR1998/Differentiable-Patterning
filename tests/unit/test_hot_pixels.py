import numpy as np
import pytest

from Common.dataloader.hot_pixels import replace_hot_pixels, replace_hot_pixels_per_channel


def _noisy_image_with_nucleus(shape=(48, 48)):
    """Noise around 100, plus one 12 x 12 bright 'nucleus'."""
    image = 100.0 + np.random.default_rng(0).normal(0.0, 5.0, shape)
    image[10:22, 10:22] += 400.0
    return image.astype(np.float32)


def test_isolated_spikes_are_replaced_by_their_neighbours():
    image = _noisy_image_with_nucleus()
    spiky = image.copy()
    spiky[35, 35] = 60000.0
    spiky[5, 40:42] = 30000.0  # a two-pixel speck
    cleaned, hot = replace_hot_pixels(spiky, threshold=10, size=5)
    assert hot[35, 35] and hot[5, 40] and hot[5, 41]
    assert abs(cleaned[35, 35] - 100.0) < 15.0
    assert abs(cleaned[5, 41] - 100.0) < 15.0
    # Pixels that were not hot are untouched.
    np.testing.assert_array_equal(cleaned[~hot], spiky[~hot])


def test_structures_larger_than_the_window_are_kept():
    image = _noisy_image_with_nucleus()
    cleaned, hot = replace_hot_pixels(image, threshold=10, size=5)
    assert not hot[12:20, 12:20].any()
    np.testing.assert_array_equal(cleaned[12:20, 12:20], image[12:20, 12:20])


def test_flat_image_has_no_hot_pixels():
    cleaned, hot = replace_hot_pixels(np.full((16, 16), 7.0), threshold=5)
    assert not hot.any()
    np.testing.assert_array_equal(cleaned, 7.0)


def test_per_channel_skips_channels_without_threshold():
    spiky = _noisy_image_with_nucleus()
    spiky[35, 35] = 60000.0
    stacked = np.stack([spiky, spiky], axis=-1)
    cleaned, hot = replace_hot_pixels_per_channel(stacked, [10, 0])
    assert hot[35, 35, 0] and not hot[..., 1].any()
    np.testing.assert_array_equal(cleaned[..., 1], spiky)


@pytest.mark.parametrize("threshold, size", [(0, 5), (5, 4), (5, 1)])
def test_replace_hot_pixels_rejects_bad_arguments(threshold, size):
    with pytest.raises(ValueError):
        replace_hot_pixels(np.ones((8, 8)), threshold, size)


def test_per_channel_needs_one_threshold_per_channel():
    with pytest.raises(ValueError):
        replace_hot_pixels_per_channel(np.ones((8, 8, 2)), [5])
