"""SYCL-specific batching, fused rollout, and two-tile execution hooks."""

from __future__ import annotations

import os
from numbers import Integral

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from Common.utils import key_pytree_gen
from NCA.model.NCA_sycl import FUSED_REGULARISER_FLAGS
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.sycl_batching import apply_flat_batched_nca
from NCA.trainer.sycl_execution import SyclTwoTileExecution
from NCA.trainer.sycl_scan import scan_carry_only


def _configure_boolean_environment(name, enabled):
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise TypeError(f"{name} must be boolean or None")
        if enabled:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)
    value = os.getenv(name, "")
    return value not in {"", "0", "false", "False"}


def configure_custom_call_synchronization(enabled):
    """Configure end-of-custom-call synchronization."""
    return _configure_boolean_environment(
        "NCA_SYCL_SYNCHRONIZE_CUSTOM_CALLS", enabled
    )


def configure_stage_synchronization(enabled):
    """Configure diagnostic waits after individual native stages."""
    return _configure_boolean_environment(
        "NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION", enabled
    )


def configure_regulariser_reduction(mode):
    """Select atomic or deterministic two-stage fused reduction."""
    name = "NCA_SYCL_REGULARISER_REDUCTION"
    if mode is not None:
        if mode not in {"atomic", "two_stage"}:
            raise ValueError(
                "SYCL_REGULARISER_REDUCTION must be 'atomic', 'two_stage', "
                "or None"
            )
        os.environ[name] = mode
    active = os.getenv(name, "atomic")
    if active not in {"atomic", "two_stage"}:
        raise ValueError(
            f"Unsupported {name} value {active!r}; expected 'atomic' or "
            "'two_stage'"
        )
    return active


class NCA_sycl_Trainer(NCA_Trainer):
    """Train the native SYCL NCA while retaining independent outer-B leaves.

    Each state leaf has shape ``[N, C, H, W]``. A fused native rollout advances
    all ``N`` states for ``sycl_fused_steps`` sequential NCA timesteps. With
    ``SHARDING=2``, an even number of outer-B leaves is divided equally between
    the two tiles while model and optimiser arrays remain replicated.
    """

    def __init__(
        self,
        *args,
        SYCL_FUSED_STEPS=2,
        SYCL_SYNCHRONIZE_CUSTOM_CALLS=None,
        SYCL_STRICT_STAGE_SYNCHRONIZATION=None,
        SYCL_REGULARISER_REDUCTION=None,
        SYCL_PMEAN_LOSS=True,
        SYCL_PMEAN_REGULARISERS=True,
        SYCL_SERIALIZE_CUSTOM_CALLS=None,
        **kwargs,
    ):
        self.synchronize_custom_calls = configure_custom_call_synchronization(
            SYCL_SYNCHRONIZE_CUSTOM_CALLS
        )
        self.strict_stage_synchronization = configure_stage_synchronization(
            SYCL_STRICT_STAGE_SYNCHRONIZATION
        )
        self.regulariser_reduction = configure_regulariser_reduction(
            SYCL_REGULARISER_REDUCTION
        )
        if not isinstance(SYCL_PMEAN_LOSS, bool):
            raise TypeError("SYCL_PMEAN_LOSS must be boolean")
        if not isinstance(SYCL_PMEAN_REGULARISERS, bool):
            raise TypeError("SYCL_PMEAN_REGULARISERS must be boolean")
        self.pmean_loss = SYCL_PMEAN_LOSS
        self.pmean_regularisers = SYCL_PMEAN_REGULARISERS
        self.serialize_custom_calls = _configure_boolean_environment(
            "NCA_SYCL_SERIALIZE_CUSTOM_CALLS", SYCL_SERIALIZE_CUSTOM_CALLS
        )
        super().__init__(*args, **kwargs)
        if isinstance(SYCL_FUSED_STEPS, bool) or not isinstance(
            SYCL_FUSED_STEPS, Integral
        ):
            raise TypeError("SYCL_FUSED_STEPS must be a positive integer")
        fused_steps = int(SYCL_FUSED_STEPS)
        if fused_steps < 1:
            raise ValueError("SYCL_FUSED_STEPS must be a positive integer")
        self.fused_steps = fused_steps
        print(
            "NCA SYCL custom-call synchronization: "
            f"{self.synchronize_custom_calls}",
            flush=True,
        )
        print(
            "NCA SYCL strict stage synchronization: "
            f"{self.strict_stage_synchronization}",
            flush=True,
        )
        print(
            f"NCA SYCL regulariser reduction: {self.regulariser_reduction}",
            flush=True,
        )
        print(f"NCA SYCL loss pmean: {self.pmean_loss}", flush=True)
        print(
            f"NCA SYCL regulariser pmean: {self.pmean_regularisers}",
            flush=True,
        )
        print(
            f"NCA SYCL serialized custom calls: {self.serialize_custom_calls}",
            flush=True,
        )

    def setup_logging(self, *args, **kwargs):
        self.TRAIN_CONFIG["SYCL_SYNCHRONIZE_CUSTOM_CALLS"] = (
            self.synchronize_custom_calls
        )
        self.TRAIN_CONFIG["SYCL_STRICT_STAGE_SYNCHRONIZATION"] = (
            self.strict_stage_synchronization
        )
        self.TRAIN_CONFIG["SYCL_REGULARISER_REDUCTION"] = (
            self.regulariser_reduction
        )
        self.TRAIN_CONFIG["SYCL_PMEAN_LOSS"] = self.pmean_loss
        self.TRAIN_CONFIG["SYCL_PMEAN_REGULARISERS"] = (
            self.pmean_regularisers
        )
        self.TRAIN_CONFIG["SYCL_SERIALIZE_CUSTOM_CALLS"] = (
            self.serialize_custom_calls
        )
        return super().setup_logging(*args, **kwargs)

    def _training_execution(self):
        if self.SHARDING in (None, 1):
            return super()._training_execution()
        if self.SHARDING == 2:
            return SyclTwoTileExecution(self)
        raise ValueError(
            "NCA_sycl_Trainer currently supports trainer.sharding=null, 1, "
            f"or 2; got {self.SHARDING}"
        )

    def _make_batched_nca(self, nca):
        """Return a PyTree call over state leaves shaped ``[N,C,H,W]``."""
        fallback = super()._make_batched_nca(nca)
        batched_call = getattr(nca, "batched_call", None)
        if batched_call is None:
            return fallback

        def apply_batched(x, callbacks, key_array):
            return apply_flat_batched_nca(
                nca, x, callbacks, key_array, fallback
            )

        return apply_batched

    @staticmethod
    def _boundary_spec(callback, dtype):
        """Encode a boundary callback for the native rollout ABI."""
        if isinstance(callback, no_boundary):
            return 0, jnp.zeros((1,), dtype=dtype), 0
        if isinstance(callback, model_boundary):
            mask = jnp.asarray(callback.MASK, dtype=dtype)
            return 1, mask, mask.shape[0]
        if isinstance(callback, hard_boundary):
            return 2, jnp.asarray(callback.MASK, dtype=dtype), 0
        raise TypeError(f"Unsupported fused-rollout boundary: {type(callback)}")

    def _rollout_tree(self, nca, state, callbacks, keys, regulariser_flags=0):
        """Apply one native fused rollout independently to each outer-B leaf.

        State leaves are ``[N,C,H,W]`` and key leaves are ``[K,N,2]``, where
        ``K`` is ``self.fused_steps``. Returned trajectory leaves are
        ``[K,N,C,H,W]``, final-state leaves are ``[N,C,H,W]``, and native
        regulariser leaves contain two FP32 sums shaped ``[2]``.
        """
        state_leaves, tree_definition = jtu.tree_flatten(state)
        key_leaves = tree_definition.flatten_up_to(keys)
        callback_leaves = tree_definition.flatten_up_to(callbacks)
        final_leaves = []
        trajectory_leaves = []
        regulariser_leaves = []
        for leaf, leaf_keys, callback in zip(
            state_leaves, key_leaves, callback_leaves
        ):
            boundary_code, boundary_mask, boundary_channels = (
                self._boundary_spec(callback, leaf.dtype)
            )
            rollout_result = nca.batched_rollout(
                leaf,
                leaf_keys,
                boundary_code=boundary_code,
                boundary_mask=boundary_mask,
                boundary_channels=boundary_channels,
                regulariser_flags=regulariser_flags,
            )
            if regulariser_flags:
                final, trajectory, regularisers = rollout_result
            else:
                final, trajectory = rollout_result
                regularisers = jnp.zeros((2,), dtype=leaf.dtype)
            final_leaves.append(final)
            trajectory_leaves.append(trajectory)
            regulariser_leaves.append(regularisers)
        return (
            jtu.tree_unflatten(tree_definition, final_leaves),
            jtu.tree_unflatten(tree_definition, trajectory_leaves),
            jtu.tree_unflatten(tree_definition, regulariser_leaves),
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
        training_execution,
    ):
        """Advance all state leaves for ``t`` timesteps in fused chunks.

        The carry is ``(key, latent_state, processed_state, regulariser_logs)``.
        Intermediate states remain visible so existing per-step regularisers
        retain the reference trainer's semantics.
        """
        rollout = getattr(nca, "batched_rollout", None)
        training_callbacks = training_execution.boundary_callbacks()
        callbacks = jtu.tree_leaves(training_callbacks)
        supported_boundaries = all(
            isinstance(callback, (no_boundary, model_boundary, hard_boundary))
            for callback in callbacks
        )
        if (
            rollout is None
            or not isinstance(t, int)
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
                training_execution,
            )
        if t % self.fused_steps != 0:
            raise ValueError(
                f"NCA timestep count t={t} must be divisible by "
                f"trainer.sycl_fused_steps={self.fused_steps}"
            )

        fused_regularisers = tuple(
            name for name in FUSED_REGULARISER_FLAGS if name in reg_logs_internal
        )
        regulariser_flags = sum(
            FUSED_REGULARISER_FLAGS[name] for name in fused_regularisers
        )

        inner_batch_size = x_latent[0].shape[0]

        def rollout_chunk(carry, chunk_start):
            step_key, state, processed, reg_logs = carry
            keys_by_step = []
            step_keys = []
            for offset in range(self.fused_steps):
                step_key = jr.fold_in(step_key, chunk_start + offset)
                step_keys.append(step_key)
                keys_by_step.append(
                    key_pytree_gen(step_key, (len(state), inner_batch_size))
                )
            rollout_keys = jtu.tree_map(
                lambda *values: jnp.stack(values, axis=0), *keys_by_step
            )
            final_state, trajectory, native_regularisers = self._rollout_tree(
                nca,
                state,
                training_callbacks,
                rollout_keys,
                regulariser_flags,
            )
            if fused_regularisers:
                native_regularisers = jnp.stack(
                    jtu.tree_leaves(native_regularisers), axis=0
                )
                for index, name in enumerate(FUSED_REGULARISER_FLAGS):
                    if name in fused_regularisers:
                        reg_logs[name] += native_regularisers[:, index]

            previous_state = state
            previous_processed = processed
            for offset in range(self.fused_steps):
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
                    skip=fused_regularisers,
                )
                previous_state = new_state
                previous_processed = new_processed
            return (
                step_key,
                final_state,
                previous_processed,
                reg_logs,
            ), None

        carry = scan_carry_only(
            rollout_chunk,
            (key, x_latent, x_proc, reg_logs_internal),
            jnp.arange(0, t, self.fused_steps),
            kind=loop_autodiff,
        )
        return carry


SyclNCA_Trainer = NCA_sycl_Trainer

__all__ = ["NCA_sycl_Trainer", "SyclNCA_Trainer"]
