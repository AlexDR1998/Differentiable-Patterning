"""Remove narrow horizontal or vertical bands of raised background.

Stitching microscope tiles can leave a band of rows (or columns) whose
background is brighter than its neighbours. Such a band is constant along the
rows it covers and narrow across them, so it shows up in a 1D profile:

1. the background level of each row is a low percentile of that row, which
   ignores the bright cells on it;
2. a median filter over ``4 * width + 1`` rows gives a smooth baseline of
   that profile. Like the hot pixel filter, the median ignores features
   narrower than half its window, so it follows the colony and the
   illumination but not the bands;
3. the stripe is the profile minus the baseline. It is subtracted from every
   pixel of its row, and the result clipped at zero.

In frequency terms, this removes the frequencies across the band (features
narrower than ``width``) from the row background only, so the colony itself
is left alone. ``width`` is in full-resolution pixels.

A seam may cover only part of a row (e.g. one stitched tile) or shift along
it. With ``block`` set, the profile is taken separately in blocks of about
``block`` columns, and the stripe is interpolated linearly between the block
centres. Smaller blocks follow seams more closely, but use fewer pixels per
row, so cell texture is more likely to be taken for a stripe.
"""

from collections.abc import Sequence

import numpy as np
import scipy.ndimage as ndi

DIRECTIONS = ("horizontal", "vertical")


def remove_stripes(image, width, direction="horizontal", block=0, percentile=10.0):
    """Return ``(cleaned, stripes)`` for a 2D image, with ``stripes`` the 2D map subtracted.

    ``direction`` is the orientation of the bands: ``"horizontal"`` bands
    run along rows, ``"vertical"`` bands along columns. ``block`` is the
    length (pixels) of the pieces of each row that get their own profile;
    0 uses whole rows.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if width <= 0 or int(width) != width:
        raise ValueError("width must be a positive integer")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if block < 0 or int(block) != block:
        raise ValueError("block must be a non-negative integer")
    rows = image if direction == "horizontal" else image.T
    length = rows.shape[1]
    count = max(1, round(length / block)) if block else 1
    pieces = np.array_split(np.arange(length), count)

    # Background profile and its stripes, one column per block: [rows, blocks].
    profile = np.stack(
        [np.percentile(rows[:, piece], percentile, axis=1) for piece in pieces], axis=1
    )
    baseline = ndi.median_filter(profile, size=(4 * int(width) + 1, 1), mode="reflect")
    # Linear interpolation between block centres (constant beyond the end ones).
    centres = [piece.mean() for piece in pieces]
    weights = np.stack(
        [np.interp(np.arange(length), centres, np.eye(count)[k]) for k in range(count)]
    )
    stripes = (profile - baseline) @ weights
    if direction == "vertical":
        stripes = stripes.T
    stripes = stripes.astype(np.float32)
    cleaned = np.clip(image - stripes, 0.0, None)
    return cleaned, stripes


def remove_stripes_per_channel(image, horizontal_widths, vertical_widths, block=0):
    """Apply ``remove_stripes`` to each channel of an ``[X, Y, C]`` image.

    ``horizontal_widths`` and ``vertical_widths`` hold one width per channel;
    ``None`` or ``0`` skips that direction for that channel. ``block`` is
    shared by all channels. Returns
    ``(cleaned, stripes)``, both ``[X, Y, C]`` float32.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected an [X, Y, C] image, got shape {image.shape}")
    for label, widths in (("horizontal", horizontal_widths), ("vertical", vertical_widths)):
        if not isinstance(widths, Sequence) or len(widths) != image.shape[-1]:
            raise ValueError(
                f"Expected {image.shape[-1]} {label} widths (one per channel), got {widths!r}"
            )
    cleaned = image.copy()
    stripes = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        for direction, width in (
            ("horizontal", horizontal_widths[channel]),
            ("vertical", vertical_widths[channel]),
        ):
            if width:
                cleaned[..., channel], removed = remove_stripes(
                    cleaned[..., channel], width, direction, block
                )
                stripes[..., channel] += removed
    return cleaned, stripes
