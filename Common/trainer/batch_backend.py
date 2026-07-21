"""Batch-container policies for uniform arrays and heterogeneous trajectory lists.

The training code works with a logical batch of trajectories, each shaped
``[N, C, H, W]``.  Tree mode retains the historical list/PyTree
representation, while array mode keeps a uniform batch in one
``[B, N, C, H, W]`` array. Keeping representation-specific operations here
prevents the trainer, regularisers, and loggers from growing parallel paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from Common.model.boundary import hard_boundary, model_boundary, no_boundary


def _stack_pytrees(values):
    """Stack a non-empty sequence of identically structured array PyTrees."""
    return jtu.tree_map(lambda *items: jnp.stack(items), *values)


@dataclass(frozen=True)
class BatchBackend:
    """Operations whose implementation depends on the outer-batch container."""

    mode: str

    @property
    def is_array(self) -> bool:
        return self.mode == "array"

    def batch_size(self, value) -> int:
        return len(value)

    def time_size(self, value) -> int:
        return value.shape[1] if self.is_array else value[0].shape[0]

    def map(self, function: Callable, *values):
        if self.is_array:
            return jax.vmap(function)(*values)
        return jtu.tree_map(function, *values)

    def keys(self, key, batch_size: int, time_size: int | None = None):
        shape = (batch_size,) if time_size is None else (batch_size, time_size)
        keys = jax.random.randint(
            key,
            shape=(*shape, 2),
            minval=0,
            maxval=2_147_483_647,
            dtype=jnp.uint32,
        )
        return keys if self.is_array else list(keys)

    def to_list(self, value):
        if value is None or not self.is_array:
            return value
        return [value[index] for index in range(value.shape[0])]

    def from_list(self, value):
        if not self.is_array:
            return value
        return jnp.stack(value)

    def stack_batch_pytree(self, value):
        """Convert a list of per-batch PyTrees to arrays with leading ``B``."""
        if not self.is_array:
            return value
        return _stack_pytrees(value)

    def apply_model(self, nca, state, callbacks, keys):
        """Apply one NCA step while preserving the selected container."""
        apply_time = jax.vmap(
            lambda item, item_key: nca(item, key=item_key),
            in_axes=(0, 0),
            out_axes=0,
            axis_name="N",
        )
        if not self.is_array:
            apply_with_boundary = jax.vmap(
                nca, in_axes=(0, None, 0), out_axes=0, axis_name="N"
            )
            return jtu.tree_map(apply_with_boundary, state, callbacks, keys)

        # The model update is vmapped over both batch axes without embedding a
        # Python callback in the mapped computation. Boundary enforcement is a
        # cheap batched epilogue below.
        batched_call = getattr(nca, "batched_call", None)
        boundary_mode = getattr(nca, "BATCHED_BOUNDARY_MODE", "epilogue")
        has_boundary = not all(
            isinstance(callback, no_boundary) for callback in callbacks
        )
        if boundary_mode == "internal" and has_boundary:
            return jnp.stack([
                jax.vmap(nca, in_axes=(0, None, 0))(
                    state[index], callback, keys[index]
                )
                for index, callback in enumerate(callbacks)
            ])
        if batched_call is not None:
            batch, time = state.shape[:2]
            flat_state = state.reshape(batch * time, *state.shape[2:])
            flat_keys = keys.reshape(batch * time, *keys.shape[2:])
            updated = batched_call(flat_state, flat_keys).reshape(state.shape)
        else:
            updated = jax.vmap(apply_time, in_axes=(0, 0), out_axes=0)(state, keys)
        return self.apply_boundaries(updated, callbacks)

    def apply_boundaries(self, state, callbacks):
        if not self.is_array:
            return jtu.tree_map(lambda callback, value: callback(value), callbacks, state)
        if all(isinstance(callback, no_boundary) for callback in callbacks):
            return state
        if all(isinstance(callback, model_boundary) for callback in callbacks):
            masks = jnp.stack([callback.MASK for callback in callbacks])
            mask_channels = masks.shape[1]
            return state.at[:, :, -mask_channels:].set(masks[:, None])
        if all(isinstance(callback, hard_boundary) for callback in callbacks):
            masks = jnp.stack([callback.MASK for callback in callbacks])
            return state * masks[:, None, None]

        # Mixed boundary modes are unusual but remain supported. This branch is
        # statically unrolled at trace time and only affects the small epilogue.
        return jnp.stack(
            [jax.vmap(callback)(state[index]) for index, callback in enumerate(callbacks)]
        )

    def loss_map(
        self,
        function,
        x_proc,
        y_proc,
        x_latent,
        y_latent,
        masks,
        cache,
        keys,
    ):
        if not self.is_array:
            return jnp.asarray(
                jtu.tree_map(
                    function,
                    x_proc,
                    y_proc,
                    x_latent,
                    y_latent,
                    masks,
                    cache,
                    keys,
                )
            )

        if cache is None:
            return jax.vmap(
                lambda xp, yp, xl, yl, mask, item_key: function(
                    xp, yp, xl, yl, mask, {}, item_key
                )
            )(x_proc, y_proc, x_latent, y_latent, masks, keys)
        return jax.vmap(function)(
            x_proc, y_proc, x_latent, y_latent, masks, cache, keys
        )


def make_batch_backend(value: Any, requested: str | None = None) -> BatchBackend:
    """Return the backend matching ``value``, optionally validating a request."""
    inferred = "array" if hasattr(value, "ndim") and value.ndim == 5 else "tree"
    mode = inferred if requested in (None, "auto") else requested
    if mode not in {"tree", "array"}:
        raise ValueError("batch mode must be 'tree', 'array', or 'auto'")
    if mode != inferred:
        raise TypeError(
            f"Requested {mode!r} batching but augmenter returned {inferred!r} data"
        )
    return BatchBackend(mode)


__all__ = ["BatchBackend", "make_batch_backend"]
