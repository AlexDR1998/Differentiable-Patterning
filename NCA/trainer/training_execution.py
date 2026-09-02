"""Execution policies used by the NCA training loop.

The reference trainer delegates backend-specific mechanics here so its loss,
optimiser, logging, and pool-control flow remain readable. Accelerator-specific
trainers can replace this object without adding flags throughout ``train``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from Common.trainer.loss_multi_target import multi_target_loss


class TrainingExecution:
    """Single-device reference behaviour for the config-driven trainer."""

    def __init__(self, trainer):
        self.trainer = trainer

    def boundary_callbacks(self):
        return self.trainer.boundary_callbacks

    def loss_time_channel_mask(self):
        return self.trainer.loss_time_channel_mask

    def loss_cache(self):
        return self.trainer.loss_cache

    def diagnostic_boundary_masks(self):
        masks = self.trainer.diagnostic_boundary_mask
        if masks is None:
            return None
        return [mask for mask in masks]

    def synchronise_loss(self, mean_loss, regulariser_losses):
        return mean_loss, regulariser_losses

    def multi_target_loss(
        self, prediction, target, boundary, schema, params, key, args,
        measurement_mask=None,
    ):
        """Evaluate the reference full-batch multi-target loss."""
        return multi_target_loss(
            prediction, target, boundary, schema, params, key, args,
            measurement_mask=measurement_mask,
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

    def apply_advance_pool(self, x, y, iteration, key):
        return self.trainer.data_augmenter.advance_pool(x, y, iteration, key)

    def prepare_log_dict(self, log_dict):
        return log_dict

    def prepare_admission_losses(self, losses):
        """Return one existing objective value per biological time slot."""
        losses = jnp.asarray(losses)
        if losses.ndim < 1:
            raise ValueError("Per-time pool admission requires a loss vector")
        if losses.ndim == 1:
            return losses
        return jnp.mean(losses, axis=tuple(range(losses.ndim - 1)))


__all__ = ["TrainingExecution"]
