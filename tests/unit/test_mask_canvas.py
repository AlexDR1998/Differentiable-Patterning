import numpy as np
import pytest

pytest.importorskip("anywidget")

from Experiments.micropatterns.mask_canvas import MaskCanvas, decode_mask, encode_mask


def test_mask_round_trip():
    mask = np.random.default_rng(0).random((7, 11)) > 0.5
    np.testing.assert_array_equal(decode_mask(encode_mask(mask), 7, 11), mask)


def test_decode_rejects_wrong_size():
    with pytest.raises(ValueError):
        decode_mask(encode_mask(np.ones((4, 4), dtype=bool)), 5, 5)


def test_widget_holds_initial_mask():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    widget = MaskCanvas(height=6, width=8, mask=encode_mask(mask))
    np.testing.assert_array_equal(decode_mask(widget.mask, 6, 8), mask)
