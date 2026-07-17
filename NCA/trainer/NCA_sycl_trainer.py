"""SYCL-specialized NCA trainer paths.

The base trainer remains the reference implementation. This subclass owns
batch flattening and is the intended home for future multi-step custom calls,
regulariser fusion, and other Intel-specific training transformations.
"""

from __future__ import annotations

from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.sycl_batching import apply_flat_batched_nca


class NCA_sycl_Trainer(NCA_Trainer):
    """Use one native call for compatible leaves across both ``B`` and ``N``."""

    def _make_batched_nca(self, nca):
        fallback = super()._make_batched_nca(nca)
        batched_call = getattr(nca, "batched_call", None)
        if batched_call is None:
            return fallback

        def apply_batched(x, callbacks, key_array):
            # Boundary callbacks remain the established JAX operations for
            # now. A future fused epilogue belongs in this subclass/path.
            return apply_flat_batched_nca(
                nca, x, callbacks, key_array, fallback
            )

        return apply_batched


SyclNCA_Trainer = NCA_sycl_Trainer

__all__ = ["NCA_sycl_Trainer", "SyclNCA_Trainer"]
