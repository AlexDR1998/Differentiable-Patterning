"""Execution policies used by the NCA training loop.

The reference trainer delegates backend-specific mechanics here so its loss,
optimiser, logging, and pool-control flow remain readable. Accelerator-specific
trainers can replace this object without adding flags throughout ``train``.
"""

from __future__ import annotations

import equinox as eqx
import jax.random as jr

from Common.trainer.loss_multi_target import multi_target_loss


class TrainingExecution:
    """Single-device reference behaviour for :class:`NCA_Trainer`."""

    def __init__(self, trainer):
        self.trainer = trainer

    def boundary_callbacks(self):
        return self.trainer.BOUNDARY_CALLBACK

    def loss_time_channel_mask(self):
        return self.trainer.LOSS_TIME_CHANNEL_MASK

    def loss_cache(self):
        return self.trainer.LOSS_CACHE

    def diagnostic_boundary_masks(self):
        masks = self.trainer.DIAGNOSTIC_BOUNDARY_MASK
        if masks is None:
            return None
        return [mask for mask in masks]

    def synchronise_loss(self, mean_loss, regulariser_losses):
        return mean_loss, regulariser_losses

    def multi_target_loss(
        self, prediction, target, boundary, schema, params, key, args
    ):
        """Evaluate the reference full-batch multi-target loss."""
        return multi_target_loss(
            prediction, target, boundary, schema, params, key, args
        )

    def transform_step(self, make_step):
        return eqx.filter_jit(make_step)

    def transform_loss(self, compute_loss):
        """Return a loss callable with the same input and output PyTrees."""
        return compute_loss

    def prepare_inputs(self, nca, x, y, opt_state, key):
        return nca, x, y, opt_state, key

    def fold_in_key(self, key, iteration):
        return jr.fold_in(key, iteration)

    def split_key(self, key):
        return jr.split(key)

    def apply_data_callback(self, x, y, iteration, key):
        return self.trainer.DATA_AUGMENTER.data_callback(x, y, iteration, key)

    def prepare_log_dict(self, log_dict):
        return log_dict


__all__ = ["TrainingExecution"]
