"""Portable JAX implementation of the standard NCA update.

This module deliberately preserves the public interface and parameter layout of
``NCA.model.NCA_model.NCA``.  A standard NCA can therefore opt in by changing
only its import::

    from NCA.model.NCA_model_fast import NCA

The implementation removes the vmapped single-channel convolutions used by the
reference perception function.  All requested linear spatial filters are
evaluated by one grouped convolution.  The two 1x1 convolutions in the update
MLP are expressed as matrix products, which gives XLA a direct opportunity to
use the target's optimized GEMM implementation.

Only the main ``__call__`` path uses the matrix-product implementation.  The
inherited activation/SAE diagnostic methods still walk ``self.layers`` so that
their intermediate activation semantics remain unchanged.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from Common.model.spatial_operators import safe_grad_norm
from NCA.model.NCA_model import NCA as ReferenceNCA


_PAD_MODES = {
    "ZEROS": "constant",
    "REFLECT": "reflect",
    "REPLICATE": "edge",
    "CIRCULAR": "wrap",
}


def _same_pad_spatial(x: Array, kernel_size: tuple[int, int], mode: str) -> Array:
    """Apply the padding used by the finite-difference Conv2d modules."""
    mode = mode.upper()
    try:
        pad_mode = _PAD_MODES[mode]
    except KeyError as exc:
        supported = ", ".join(_PAD_MODES)
        raise ValueError(
            f"Unsupported padding mode {mode!r}; expected one of {supported}"
        ) from exc

    # Ops currently constructs odd, square kernels. Keeping the asymmetric
    # formula here also gives the same output size if an even kernel is added.
    pad_h = (kernel_size[0] - 1) // 2, kernel_size[0] // 2
    pad_w = (kernel_size[1] - 1) // 2, kernel_size[1] // 2
    pad_width = ((0, 0), (0, 0), pad_h, pad_w)
    if pad_mode == "constant":
        return jnp.pad(x, pad_width, mode=pad_mode, constant_values=0)
    return jnp.pad(x, pad_width, mode=pad_mode)


def _pointwise_conv_as_dot(layer: eqx.nn.Conv2d, x: Array) -> Array:
    """Apply a 1x1 Conv2d as a feature-by-pixel matrix product."""
    weight = layer.weight
    if weight.shape[-2:] != (1, 1) or layer.groups != 1:
        # This fallback makes the helper safe for subclasses which replace a
        # standard pointwise layer with a spatial or grouped convolution.
        return layer(x)

    in_channels, height, width = x.shape
    if weight.shape[1] != in_channels:
        raise ValueError(
            "Pointwise layer input does not match the perception output: "
            f"expected {weight.shape[1]} channels, got {in_channels}"
        )

    # [O, I] @ [I, H*W] -> [O, H*W]. Casting the weights permits an explicitly
    # low-precision state while gradients still flow to the original weights.
    weight_2d = weight[:, :, 0, 0].astype(x.dtype)
    x_2d = x.reshape(in_channels, height * width)
    y = lax.dot_general(
        weight_2d,
        x_2d,
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    y = y.reshape(weight.shape[0], height, width)
    if layer.bias is not None:
        y = y + layer.bias.astype(y.dtype)
    return y


def _batched_pointwise_conv_as_dot(layer: eqx.nn.Conv2d, x: Array) -> Array:
    """Apply a shared 1x1 convolution to ``[B,C,H,W]`` as one GEMM."""
    weight = layer.weight
    if weight.shape[-2:] != (1, 1) or layer.groups != 1:
        return jax.vmap(layer)(x)

    batch, in_channels, height, width = x.shape
    if weight.shape[1] != in_channels:
        raise ValueError(
            "Pointwise layer input does not match the batched feature array: "
            f"expected {weight.shape[1]} channels, got {in_channels}"
        )
    # [O,I] @ [I,B*H*W] -> [O,B*H*W]. Keeping all cells on the right-hand
    # side gives cuBLAS one large shared-weight matrix multiplication rather
    # than a strided batch of small products.
    x_2d = jnp.transpose(x, (1, 0, 2, 3)).reshape(
        in_channels, batch * height * width
    )
    y = lax.dot_general(
        weight[:, :, 0, 0].astype(x.dtype),
        x_2d,
        dimension_numbers=(((1,), (0,)), ((), ())),
    )
    y = y.reshape(weight.shape[0], batch, height, width)
    y = jnp.transpose(y, (1, 0, 2, 3))
    if layer.bias is not None:
        y = y + layer.bias.astype(y.dtype)
    return y


class NCA(ReferenceNCA):
    """Drop-in standard NCA with a more compiler-friendly JAX update path.

    Parameters, configuration, serialization leaves, stochastic update masks,
    boundary callbacks, and inherited analysis helpers match ``ReferenceNCA``.
    The input to ``__call__`` remains one state with shape ``[C, H, W]``; the
    existing trainer's ``vmap`` therefore continues to batch states unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Keep ``perception`` as the same public callable field exposed by the
        # reference model. The finite-difference kernels in ``op`` are static in
        # ReferenceNCA.partition, so closing over them introduces no parameters.
        self.perception = lambda x: self._fast_perception(x)

    def _linear_filter_bank(self) -> tuple[list[str], Array] | None:
        """Return the unique linear filters needed by this model."""
        filter_names: list[str] = []
        if "DIFF" in self.KERNEL_STR or "GRAD" in self.KERNEL_STR:
            filter_names.extend(("grad_x", "grad_y"))
        if "AV" in self.KERNEL_STR:
            filter_names.append("average")
        if "LAP" in self.KERNEL_STR:
            filter_names.append("laplacian")
        if not filter_names:
            return None

        # Ops.partition() defines these finite-difference stencils as static.
        # Stopping gradients here preserves that contract even when callers
        # differentiate the complete model without first partitioning it.
        kernels = jax.lax.stop_gradient(jnp.concatenate(
            [getattr(self.op, name).weight for name in filter_names], axis=0
        ))
        return filter_names, kernels

    def _fast_perception(
        self, x: Float[Array, "{self.N_CHANNELS} H W"]
    ) -> Float[Array, "{self.N_FEATURES} H W"]:
        if x.ndim != 3:
            raise ValueError(
                f"NCA perception expects [C, H, W], received shape {x.shape}"
            )
        if x.shape[0] != self.N_CHANNELS:
            raise ValueError(
                f"Expected {self.N_CHANNELS} state channels, got {x.shape[0]}"
            )

        filtered: dict[str, Array] = {}
        filter_bank = self._linear_filter_bank()
        if filter_bank is not None:
            filter_names, kernels = filter_bank
            filter_count = len(filter_names)
            kernel_size = kernels.shape[-2:]

            # XLA feature grouping partitions the output-feature dimension by
            # input group. Repeating [filter, 1, kh, kw] by channel therefore
            # produces [channel, filter, H, W] after the reshape below.
            grouped_kernels = jnp.tile(
                kernels.astype(x.dtype), (self.N_CHANNELS, 1, 1, 1)
            )
            x_batched = _same_pad_spatial(
                x[None], kernel_size, self.op.PADDING
            )
            conv = lax.conv_general_dilated(
                x_batched,
                grouped_kernels,
                window_strides=(1, 1),
                padding="VALID",
                feature_group_count=self.N_CHANNELS,
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )[0]
            conv = conv.reshape(
                self.N_CHANNELS, filter_count, x.shape[1], x.shape[2]
            )
            filtered = {
                name: conv[:, index]
                for index, name in enumerate(filter_names)
            }

        # The reference implementation uses this canonical order regardless of
        # the order in KERNEL_STR. This is important for checkpoint-compatible
        # interpretation of the first pointwise layer's input-feature axis.
        features: list[Array] = []
        if "ID" in self.KERNEL_STR:
            features.append(x)
        if "DIFF" in self.KERNEL_STR:
            gx = filtered["grad_x"]
            gy = filtered["grad_y"]
            features.append(safe_grad_norm(gx, gy))
        if "GRAD" in self.KERNEL_STR:
            features.extend((filtered["grad_x"], filtered["grad_y"]))
        if "AV" in self.KERNEL_STR:
            features.append(filtered["average"])
        if "LAP" in self.KERNEL_STR:
            features.append(filtered["laplacian"])

        if not features:
            raise ValueError("KERNEL_STR must contain at least one supported kernel")
        return jnp.concatenate(features, axis=0)

    def _fast_perception_batched(self, x: Array) -> Array:
        """Batched counterpart of :meth:`_fast_perception`."""
        if x.ndim != 4 or x.shape[1] != self.N_CHANNELS:
            raise ValueError(
                "Batched NCA perception expects [B,C,H,W] with "
                f"C={self.N_CHANNELS}, received {x.shape}"
            )

        filtered: dict[str, Array] = {}
        filter_bank = self._linear_filter_bank()
        if filter_bank is not None:
            filter_names, kernels = filter_bank
            filter_count = len(filter_names)
            grouped_kernels = jnp.tile(
                kernels.astype(x.dtype), (self.N_CHANNELS, 1, 1, 1)
            )
            padded = _same_pad_spatial(x, kernels.shape[-2:], self.op.PADDING)
            conv = lax.conv_general_dilated(
                padded,
                grouped_kernels,
                window_strides=(1, 1),
                padding="VALID",
                feature_group_count=self.N_CHANNELS,
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            conv = conv.reshape(
                x.shape[0], self.N_CHANNELS, filter_count, x.shape[2], x.shape[3]
            )
            filtered = {
                name: conv[:, :, index]
                for index, name in enumerate(filter_names)
            }

        features: list[Array] = []
        if "ID" in self.KERNEL_STR:
            features.append(x)
        if "DIFF" in self.KERNEL_STR:
            gx = filtered["grad_x"]
            gy = filtered["grad_y"]
            features.append(safe_grad_norm(gx, gy))
        if "GRAD" in self.KERNEL_STR:
            features.extend((filtered["grad_x"], filtered["grad_y"]))
        if "AV" in self.KERNEL_STR:
            features.append(filtered["average"])
        if "LAP" in self.KERNEL_STR:
            features.append(filtered["laplacian"])
        if not features:
            raise ValueError("KERNEL_STR must contain at least one supported kernel")
        return jnp.concatenate(features, axis=1)

    def batched_call(self, x: Array, keys: Array) -> Array:
        """Update a uniform ``[B,C,H,W]`` batch using consolidated GEMMs."""
        if x.ndim != 4:
            raise ValueError(f"batched_call expects [B,C,H,W], got {x.shape}")
        if keys.shape[0] != x.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} keys, got {keys.shape}")

        dx = self._fast_perception_batched(x)
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                dx = _batched_pointwise_conv_as_dot(layer, dx)
            else:
                dx = layer(dx)
        masks = jax.vmap(
            lambda key: jax.random.bernoulli(
                key, p=self.FIRE_RATE, shape=dx.shape[1:]
            )
        )(keys)
        return x + masks * dx

    def __call__(
        self,
        x: Float[Array, "{self.N_CHANNELS} H W"],
        boundary_callback: Callable[[Array], Array] = lambda value: value,
        key: Array | None = None,
    ) -> Float[Array, "{self.N_CHANNELS} H W"]:
        if key is None:
            # Match the reference model's fallback while keeping normal training
            # deterministic when a key is supplied.
            import time

            key = jax.random.PRNGKey(int(time.time()))

        dx = self.perception(x)
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                dx = _pointwise_conv_as_dot(layer, dx)
            else:
                dx = layer(dx)

        sigma = jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        return boundary_callback(x + sigma * dx)


# An explicit name is convenient in code that needs both implementations, while
# importing ``NCA`` from this module remains the minimal drop-in change.
FastNCA = NCA


__all__ = ["FastNCA", "NCA"]
