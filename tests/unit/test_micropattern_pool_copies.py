import jax.numpy as jnp

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from NCA.trainer.data_augmenter_260726 import DataAugmenter


def test_260726_augmenter_preserves_loader_batch_count():
    # These four batches represent the output of the loader, which applies the
    # configured pool copies together with its masks and metadata.
    data = jnp.zeros(
        (4, 2, MICROPATTERN_260726_SCHEMA.n_measurement_channels, 2, 2)
    )

    augmenter = DataAugmenter(data_true=data)
    augmenter.data_init()

    assert len(augmenter.return_saved_data()) == 4
