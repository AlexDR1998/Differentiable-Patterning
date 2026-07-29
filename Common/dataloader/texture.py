"""Load texture images as dense image sequences."""

from pathlib import Path

import numpy as np
import skimage.io as sio

from Common.dataloader.results import ImageSequenceDataset


def load_textures(
    filename_sequence,
    impath_textures="../Data/dtd/images/",
    downsample=2,
    crop_square=False,
    crop_factor=1,
):
    """Load textures into a ``[batch, time, channel, x, y]`` sequence."""

    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if crop_factor <= 0:
        raise ValueError("crop_factor must be positive")
    root = Path(impath_textures)
    filenames = tuple(str(filename) for filename in filename_sequence)
    images = []
    for filename in filenames:
        image = sio.imread(root / filename)[::downsample, ::downsample]
        if crop_square:
            size = int(min(image.shape[:2]) / crop_factor)
            image = image[:size, :size]
        images.append(image.astype(np.float32) / 255.0)
    if crop_square and images:
        size = min(min(image.shape[:2]) for image in images)
        images = [image[:size, :size] for image in images]
    data = np.moveaxis(np.asarray(images)[None], -1, 2)
    return ImageSequenceDataset(
        data=data,
        filenames=filenames,
        metadata={
            "downsample": downsample,
            "crop_square": crop_square,
            "crop_factor": crop_factor,
        },
    )
