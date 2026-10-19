import numpy as np
import pytest

from Common.dataloader.stripes import remove_stripes, remove_stripes_per_channel

NOISE = 5.0


def _noisy_colony(shape=(128, 128)):
    """Background noise around 100, plus a bright disk 'colony' of radius 40."""
    image = 100.0 + np.random.default_rng(0).normal(0.0, NOISE, shape)
    rows, columns = np.ogrid[: shape[0], : shape[1]]
    image[(rows - 64) ** 2 + (columns - 64) ** 2 <= 40**2] += 500.0
    return image.astype(np.float32)


def test_horizontal_band_is_removed():
    clean = _noisy_colony()
    banded = clean.copy()
    banded[50:56] += 60.0
    cleaned, stripes = remove_stripes(banded, width=8)
    assert np.allclose(stripes[52], stripes[52, 0])  # constant along the row
    assert np.mean(stripes[50:56]) == pytest.approx(60.0, abs=NOISE)
    band_excess = np.mean(cleaned[50:56] - clean[50:56])
    assert abs(band_excess) < NOISE


def test_vertical_band_is_removed():
    clean = _noisy_colony()
    banded = clean.copy()
    banded[:, 30:34] += 60.0
    cleaned, stripes = remove_stripes(banded, width=8, direction="vertical")
    assert np.allclose(stripes[:, 32], stripes[0, 32])
    assert abs(np.mean(cleaned[:, 30:34] - clean[:, 30:34])) < NOISE


def test_band_over_part_of_the_row_needs_blocks():
    clean = _noisy_colony()
    banded = clean.copy()
    banded[50:54, :64] += 60.0  # one 'tile' only
    whole_rows, _ = remove_stripes(banded, width=8)
    blocks, _ = remove_stripes(banded, width=8, block=32)
    # A whole-row profile barely sees the band; blocks remove it.
    assert np.mean(whole_rows[50:54, :64] - clean[50:54, :64]) > 30.0
    assert abs(np.mean(blocks[50:54, :48] - clean[50:54, :48])) < NOISE
    assert abs(np.mean(blocks[50:54, 80:] - clean[50:54, 80:])) < NOISE


def test_colony_and_wide_bands_are_kept():
    clean = _noisy_colony()
    for block in (0, 32):
        cleaned, _ = remove_stripes(clean, width=8, block=block)
        assert np.mean(np.abs(cleaned - clean)) < NOISE
    banded = clean.copy()
    banded[20:60] += 60.0  # far wider than the setting
    cleaned, _ = remove_stripes(banded, width=4)
    assert np.mean(cleaned[30:50] - clean[30:50]) == pytest.approx(60.0, abs=NOISE)


def test_per_channel_skips_channels_without_width():
    banded = _noisy_colony()
    banded[50:56] += 60.0
    image = np.stack([banded, banded], axis=-1)
    cleaned, stripes = remove_stripes_per_channel(image, [8, 0], [0, None])
    assert np.array_equal(cleaned[..., 1], image[..., 1])
    assert not stripes[..., 1].any()
    assert stripes[50:56, :, 0].mean() > 40.0


@pytest.mark.parametrize(
    "width, direction, block",
    [(0, "horizontal", 0), (2.5, "horizontal", 0), (4, "diagonal", 0), (4, "horizontal", -1)],
)
def test_remove_stripes_rejects_bad_arguments(width, direction, block):
    with pytest.raises(ValueError):
        remove_stripes(np.zeros((16, 16)), width, direction, block)


def test_per_channel_needs_one_width_per_channel():
    with pytest.raises(ValueError, match="one per channel"):
        remove_stripes_per_channel(np.zeros((8, 8, 2)), [4], [0, 0])
