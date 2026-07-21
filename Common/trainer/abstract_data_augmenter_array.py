"""Dense-array data augmenter compatibility class.

The maintained augmentation implementation lives in
``abstract_data_augmenter_tree`` and now supports both batch containers.  This
class preserves the historical import path while selecting dense
``[B,N,C,H,W]`` storage by default.
"""

from Common.trainer.abstract_data_augmenter_tree import (
    DataAugmenterAbstract as SharedDataAugmenterAbstract,
)


class DataAugmenterAbstract(SharedDataAugmenterAbstract):
    BATCH_MODE = "array"

    def __init__(self, data_true, hidden_channels=0, nca_model=None, batch_mode="array"):
        super().__init__(
            data_true=data_true,
            hidden_channels=hidden_channels,
            nca_model=nca_model,
            batch_mode=batch_mode,
        )


__all__ = ["DataAugmenterAbstract"]
