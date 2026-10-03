import numpy as np
import pytest
from skimage import restoration

from Common.dataloader.background import rolling_ball_background, subtract_background


def _spots_on_ramp(shape=(61, 67)):
    """Small bright spots on a slowly rising background."""
    rows, columns = np.mgrid[: shape[0], : shape[1]]
    background = 100.0 + 0.5 * columns
    spots = np.zeros(shape)
    for row, column in [(15, 15), (30, 40), (45, 20)]:
        spots += 500.0 * np.exp(-((rows - row) ** 2 + (columns - column) ** 2) / 4.0)
    return (background + spots).astype(np.float32), background


def test_shrink_one_matches_skimage_rolling_ball():
    image, _ = _spots_on_ramp()
    expected = restoration.rolling_ball(image, radius=6)
    np.testing.assert_allclose(rolling_ball_background(image, 6, shrink=1), expected)


def test_background_follows_ramp_but_not_spots():
    # The image size is not a multiple of shrink, to cover the edge padding.
    image, ramp = _spots_on_ramp()
    background = rolling_ball_background(image, radius=12, shrink=2)
    assert background.shape == image.shape
    assert np.max(np.abs(background - ramp)) < 15.0
    assert background[30, 40] < 200.0


def test_subtract_background_skips_channels_without_radius():
    image, _ = _spots_on_ramp()
    stacked = np.stack([image, image], axis=-1)
    cleaned, background = subtract_background(stacked, [12, None], shrink=2)
    assert cleaned.shape == stacked.shape
    assert cleaned.min() >= 0.0
    np.testing.assert_array_equal(cleaned[..., 1], image)
    np.testing.assert_array_equal(background[..., 1], 0.0)
    assert np.median(cleaned[..., 0]) < 15.0


@pytest.mark.parametrize(
    "radii, shrink",
    [([10], 2), ([10, 10, 10], 2), ([0, -1], 2), ([10, 10], 0)],
)
def test_subtract_background_rejects_bad_arguments(radii, shrink):
    image = np.ones((20, 20, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        subtract_background(image, radii, shrink)
