"""Replace hot pixels: isolated specks far brighter than their surroundings.

A pixel is hot when it sits more than ``threshold`` noise levels above the
median of its ``size x size`` neighbourhood. The median ignores specks
smaller than about half the window but follows larger structures such as
nuclei, so the window should be wider than the specks and narrower than a
nucleus. The noise level is a robust estimate of the spread of
``image - median`` over the whole image.

Hot pixels are replaced by the mean of the other, non-hot pixels in their
window, rather than clipped. This is meant for full-resolution images: after
downsampling, a speck is already averaged into its block.
"""

from collections.abc import Sequence

import numpy as np
import scipy.ndimage as ndi


def replace_hot_pixels(image, threshold, size=5):
    """Return ``(cleaned, hot)`` for a 2D image, with ``hot`` a boolean map."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if size < 3 or int(size) != size or size % 2 == 0:
        raise ValueError("size must be an odd integer of at least 3")
    size = int(size)

    local_median = ndi.median_filter(image, size=size, mode="reflect")
    residual = image - local_median
    # 1.4826 x median absolute deviation estimates the standard deviation of
    # normal noise. Fall back to the plain standard deviation for very flat
    # (e.g. mostly zero) images, where the median deviation is zero.
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if noise <= 0:
        noise = float(np.std(residual))
    if noise <= 0:
        return image.copy(), np.zeros(image.shape, dtype=bool)
    hot = residual > threshold * noise

    # Mean of the non-hot pixels in each window. Where a whole window is hot,
    # use the local median instead.
    good = (~hot).astype(np.float32)
    good_sum = ndi.uniform_filter(image * good, size=size, mode="reflect")
    good_fraction = ndi.uniform_filter(good, size=size, mode="reflect")
    neighbour_mean = np.where(
        good_fraction > 0, good_sum / np.maximum(good_fraction, 1e-12), local_median
    )
    cleaned = np.where(hot, neighbour_mean, image).astype(np.float32)
    return cleaned, hot


def replace_hot_pixels_per_channel(image, thresholds, size=5):
    """Apply ``replace_hot_pixels`` to each channel of an ``[X, Y, C]`` image.

    ``thresholds`` holds one value per channel; ``None`` or ``0`` leaves that
    channel unchanged. Returns ``(cleaned, hot)``, both ``[X, Y, C]``.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected an [X, Y, C] image, got shape {image.shape}")
    if not isinstance(thresholds, Sequence) or len(thresholds) != image.shape[-1]:
        raise ValueError(
            f"Expected {image.shape[-1]} thresholds (one per channel), got {thresholds!r}"
        )
    cleaned = image.copy()
    hot = np.zeros(image.shape, dtype=bool)
    for channel, threshold in enumerate(thresholds):
        if threshold:
            cleaned[..., channel], hot[..., channel] = replace_hot_pixels(
                image[..., channel], threshold, size
            )
    return cleaned, hot
