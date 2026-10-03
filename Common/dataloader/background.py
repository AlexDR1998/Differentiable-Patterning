"""Rolling-ball background removal for fluorescence images.

A ball is rolled underneath the intensity surface of an image. The surface
traced by the top of the ball is the background: it follows slowly varying
light (uneven illumination, out-of-focus haze) but cannot reach into features
narrower than the ball. Subtracting it leaves those features on a dark floor.

Rolling a large ball over a full-resolution 1080 x 1080 image takes tens of
seconds, so the ball is rolled under a block-averaged copy of the image and
the resulting background is upsampled again (the "shrink" option in ImageJ).
With ``shrink=1`` the result is identical to
``skimage.restoration.rolling_ball(image, radius=radius)``.
"""

from collections.abc import Sequence

import numpy as np
from skimage import restoration, transform


def _block_mean(image, factor):
    """Average non-overlapping ``factor x factor`` blocks, repeating edge pixels to fill partial blocks."""
    padding = [(0, (-size) % factor) for size in image.shape]
    padded = np.pad(image, padding, mode="edge")
    height, width = padded.shape[0] // factor, padded.shape[1] // factor
    return padded.reshape(height, factor, width, factor).mean(axis=(1, 3))


def rolling_ball_background(image, radius, shrink=4):
    """Return the rolling-ball background under a 2D image.

    ``radius`` is in full-resolution pixels. As in skimage, the ball is also
    ``radius`` intensity units tall, so for raw 16-bit images it behaves
    almost like a flat disk. Choose a radius a little larger than the biggest
    feature that should be kept; with a smaller radius, the inside of large
    bright regions is treated as background.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if shrink < 1 or int(shrink) != shrink:
        raise ValueError("shrink must be a positive integer")
    shrink = int(shrink)

    small = _block_mean(image, shrink) if shrink > 1 else image
    small_radius = max(1, round(radius / shrink))
    kernel = restoration.ellipsoid_kernel(
        (2 * small_radius + 1, 2 * small_radius + 1), radius
    )
    background = restoration.rolling_ball(small, kernel=kernel)
    if shrink == 1:
        return background.astype(np.float32)

    padded_shape = (small.shape[0] * shrink, small.shape[1] * shrink)
    background = transform.resize(
        background, padded_shape, order=1, mode="edge", anti_aliasing=False
    )
    return background[: image.shape[0], : image.shape[1]].astype(np.float32)


def subtract_background(image, radii, shrink=4):
    """Subtract a rolling-ball background from each channel of an ``[X, Y, C]`` image.

    ``radii`` holds one radius per channel. ``None`` or ``0`` leaves that
    channel unchanged. Returns ``(cleaned, background)``, both ``[X, Y, C]``
    float32. ``cleaned`` is clipped at zero, because the upsampled background
    can sit slightly above single pixels.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected an [X, Y, C] image, got shape {image.shape}")
    if not isinstance(radii, Sequence) or len(radii) != image.shape[-1]:
        raise ValueError(
            f"Expected {image.shape[-1]} radii (one per channel), got {radii!r}"
        )
    background = np.zeros_like(image)
    for channel, radius in enumerate(radii):
        if radius:
            background[..., channel] = rolling_ball_background(
                image[..., channel], radius, shrink
            )
    cleaned = np.clip(image - background, 0.0, None)
    return cleaned, background
