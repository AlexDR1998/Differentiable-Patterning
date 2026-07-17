"""SYCL-specialized NCA trainer paths.

The base trainer remains the reference implementation. This subclass owns
batch flattening and is the intended home for future multi-step custom calls,
regulariser fusion, and other Intel-specific training transformations.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from Common.utils import key_pytree_gen
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.sycl_batching import apply_flat_batched_nca


class NCA_sycl_Trainer(NCA_Trainer):
    """Use one native call over N while retaining independent outer B leaves."""

    ROLLOUT_STEPS = 2

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

    @staticmethod
    def _boundary_spec(callback, dtype):
        if isinstance(callback, no_boundary):
            return 0, jnp.zeros((1,), dtype=dtype), 0
        if isinstance(callback, model_boundary):
            mask = jnp.asarray(callback.MASK, dtype=dtype)
            return 1, mask, mask.shape[0]
        if isinstance(callback, hard_boundary):
            return 2, jnp.asarray(callback.MASK, dtype=dtype), 0
        raise TypeError(f"Unsupported fused-rollout boundary: {type(callback)}")

    def _rollout_tree(self, nca, state, callbacks, keys):
        state_leaves, tree_definition = jtu.tree_flatten(state)
        key_leaves = tree_definition.flatten_up_to(keys)
        callback_leaves = tree_definition.flatten_up_to(callbacks)
        final_leaves = []
        trajectory_leaves = []
        for leaf, leaf_keys, callback in zip(
            state_leaves, key_leaves, callback_leaves
        ):
            boundary_code, boundary_mask, boundary_channels = (
                self._boundary_spec(callback, leaf.dtype)
            )
            final, trajectory = nca.batched_rollout(
                leaf,
                leaf_keys,
                boundary_code=boundary_code,
                boundary_mask=boundary_mask,
                boundary_channels=boundary_channels,
            )
            final_leaves.append(final)
            trajectory_leaves.append(trajectory)
        return (
            jtu.tree_unflatten(tree_definition, final_leaves),
            jtu.tree_unflatten(tree_definition, trajectory_leaves),
        )

    def _run_nca_steps(
        self,
        nca,
        vv_nca,
        x_latent,
        x_proc,
        reg_logs_internal,
        t,
        key,
        loop_autodiff,
        apply_intermediate_regs,
        vv_latent_to_real,
    ):
        rollout = getattr(nca, "batched_rollout", None)
        callbacks = jtu.tree_leaves(self.BOUNDARY_CALLBACK)
        supported_boundaries = all(
            isinstance(callback, (no_boundary, model_boundary, hard_boundary))
            for callback in callbacks
        )
        if (
            rollout is None
            or not isinstance(t, int)
            or t % self.ROLLOUT_STEPS != 0
            or not supported_boundaries
        ):
            return super()._run_nca_steps(
                nca,
                vv_nca,
                x_latent,
                x_proc,
                reg_logs_internal,
                t,
                key,
                loop_autodiff,
                apply_intermediate_regs,
                vv_latent_to_real,
            )

        state_shape = x_latent[0].shape[0]

        def rollout_chunk(carry, chunk_start):
            step_key, state, processed, reg_logs = carry
            keys_by_step = []
            step_keys = []
            for offset in range(self.ROLLOUT_STEPS):
                step_key = jr.fold_in(step_key, chunk_start + offset)
                step_keys.append(step_key)
                keys_by_step.append(
                    key_pytree_gen(step_key, (len(state), state_shape))
                )
            rollout_keys = jtu.tree_map(
                lambda *values: jnp.stack(values, axis=0), *keys_by_step
            )
            final_state, trajectory = self._rollout_tree(
                nca, state, self.BOUNDARY_CALLBACK, rollout_keys
            )

            previous_state = state
            previous_processed = processed
            for offset in range(self.ROLLOUT_STEPS):
                new_state = jtu.tree_map(
                    lambda values: values[offset], trajectory
                )
                new_processed = vv_latent_to_real(new_state)
                reg_logs = apply_intermediate_regs(
                    reg_logs,
                    previous_state,
                    new_state,
                    previous_processed,
                    new_processed,
                    vv_nca,
                    step_keys[offset],
                )
                previous_state = new_state
                previous_processed = new_processed
            return (
                step_key,
                final_state,
                previous_processed,
                reg_logs,
            ), None

        carry, _ = eqx.internal.scan(
            rollout_chunk,
            (key, x_latent, x_proc, reg_logs_internal),
            xs=jnp.arange(0, t, self.ROLLOUT_STEPS),
            kind=loop_autodiff,
        )
        return carry


SyclNCA_Trainer = NCA_sycl_Trainer

__all__ = ["NCA_sycl_Trainer", "SyclNCA_Trainer"]
