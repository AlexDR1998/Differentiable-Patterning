"""Compatibility wrapper for the grouped 12-measurement augmenter."""

from Common.dataloader.micropattern_schemas import MICROPATTERN_GROUPED_12CH_SCHEMA
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch


class DataAugmenter(DataAugmenter4Ch):
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA


__all__ = ["DataAugmenter"]
