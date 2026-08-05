"""Compatibility wrapper for the schema-driven 260726 augmenter."""

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from NCA.trainer.data_augmenter_micropattern import MicropatternSnapshotAugmenter


class DataAugmenter(MicropatternSnapshotAugmenter):
    schema = MICROPATTERN_260726_SCHEMA


__all__ = ["DataAugmenter"]
