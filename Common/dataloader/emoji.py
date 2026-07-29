"""Load RGB or RGBA emoji images as image sequences."""

from pathlib import Path

import numpy as np
import skimage.io as sio

from Common.dataloader.results import ImageSequenceDataset


def load_emoji_sequence(
    filename_sequence,
    impath_emojis="../Data/Emojis/",
    downsample=2,
    crop_square=False,
):
    """Load images into a ``[batch, time, channel, x, y]`` sequence."""

    if downsample <= 0:
        raise ValueError("downsample must be positive")
    root = Path(impath_emojis)
    filenames = tuple(str(filename) for filename in filename_sequence)
    images = []
    for filename in filenames:
        image = sio.imread(root / filename)[::downsample, ::downsample]
        if crop_square:
            size = min(image.shape[:2])
            image = image[:size, :size]
        images.append(image.astype(np.float32) / 255.0)
    data = np.moveaxis(np.asarray(images)[None], -1, 2)
    return ImageSequenceDataset(
        data=data,
        filenames=filenames,
        metadata={"downsample": downsample, "crop_square": crop_square},
    )
