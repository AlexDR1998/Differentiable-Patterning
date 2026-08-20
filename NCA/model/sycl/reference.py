"""Portable JAX definition used to validate and differentiate the SYCL NCA."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from Common.model.spatial_operators import safe_grad_norm


_PAD_MODES = {
    0: "constant",
    1: "reflect",
    2: "edge",
    3: "wrap",
}


def jax_nca_forward(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    *,
    kernel_flags: int,
    padding: int,
):
    """Evaluate the same update as the native forward custom call."""
    was_unbatched = state.ndim == 3
    if was_unbatched:
        state = state[None]
        update_mask = update_mask[None]

    channels = state.shape[1]
    kernel_size = kernels.shape[-1]
    radius = kernel_size // 2
    try:
        pad_mode = _PAD_MODES[padding]
    except KeyError as exc:
        raise ValueError(f"Unknown NCA SYCL padding code: {padding}") from exc

    pad_width = ((0, 0), (0, 0), (radius, radius), (radius, radius))
    if pad_mode == "constant":
        padded = jnp.pad(state, pad_width, mode=pad_mode, constant_values=0)
    else:
        padded = jnp.pad(state, pad_width, mode=pad_mode)

    # The output-feature ordering is [channel, operator], matching the grouped
    # convolution used by NCA_model_fast.
    grouped_kernels = jnp.tile(kernels[:, None], (channels, 1, 1, 1))
    filtered = lax.conv_general_dilated(
        padded,
        grouped_kernels,
        window_strides=(1, 1),
        padding="VALID",
        feature_group_count=channels,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    filtered = filtered.reshape(
        state.shape[0], channels, 4, state.shape[2], state.shape[3]
    )

    features = []
    if kernel_flags & (1 << 0):
        features.append(state)
    if kernel_flags & (1 << 1):
        features.append(safe_grad_norm(filtered[:, :, 0], filtered[:, :, 1]))
    if kernel_flags & (1 << 2):
        features.extend((filtered[:, :, 0], filtered[:, :, 1]))
    if kernel_flags & (1 << 3):
        features.append(filtered[:, :, 2])
    if kernel_flags & (1 << 4):
        features.append(filtered[:, :, 3])
    perception = jnp.concatenate(features, axis=1)

    hidden = jnp.einsum("of,bfhw->bohw", weight_hidden, perception)
    hidden = jnp.maximum(hidden, 0)
    update = jnp.einsum("cf,bfhw->bchw", weight_output, hidden)
    update = update + bias_output[None, :, None, None]
    if update.shape[1] == 2 * channels:
        values, gates = jnp.split(update, 2, axis=1)
        update = values * lax.logistic(gates)
    elif update.shape[1] != channels:
        raise ValueError("NCA output width must be C or 2C")
    result = state + update_mask * update
    return result[0] if was_unbatched else result


__all__ = ["jax_nca_forward"]
