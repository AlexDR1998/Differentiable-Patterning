"""Compatibility wrapper for the schema-driven 4-channel augmenter."""

from Common.dataloader.micropattern_schemas import MICROPATTERN_4CH_SCHEMA
from NCA.trainer.data_augmenter_micropattern import MicropatternDataAugmenter


class DataAugmenter(MicropatternDataAugmenter):
    schema = MICROPATTERN_4CH_SCHEMA


__all__ = ["DataAugmenter"]
