"""Dense compatibility entry point for the maintained texture augmenter."""

from NCA.trainer.data_augmenter_nca_texture import (
    DataAugmenter as SharedTextureDataAugmenter,
)


class DataAugmenter(SharedTextureDataAugmenter):
    BATCH_MODE = "array"


__all__ = ["DataAugmenter"]
