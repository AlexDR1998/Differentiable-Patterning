"""Baseline NCA backed by an Intel SYCL/XLA custom call.

The model keeps the parameter PyTree and public call signature of the standard
NCA. Only the main update is replaced; perception remains available as a JAX
callable for diagnostics and gradient-based losses.

Gradients for the state and trainable pointwise layers are supplied by a native
SYCL backward custom call. The fixed perception kernels and stochastic mask are
non-trainable and therefore receive zero cotangents.
"""

from collections.abc import Callable
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from NCA.model.NCA_model_fast import NCA as FastNCA


_KERNEL_FLAGS = {
    "ID": 1 << 0,
    "DIFF": 1 << 1,
    "GRAD": 1 << 2,
    "AV": 1 << 3,
    "LAP": 1 << 4,
}

_PADDING_CODES = {
    "ZEROS": 0,
    "REFLECT": 1,
    "REPLICATE": 2,
    "CIRCULAR": 3,
}


class NCA(FastNCA):
    """Standard NCA whose forward update is executed by a SYCL kernel.

    Set ``NCA_SYCL_LIBRARY`` to the compiled ``libnca_sycl.so`` path, or place
    the library beside ``NCA/model/sycl/files/nca_sycl.cpp``.
    """

    def get_config(self):
        config = super().get_config()
        config["MODEL"] = "NCA_sycl"
        return config

    def _validate_sycl_configuration(self, x: Array) -> tuple[int, int]:
        if x.ndim not in (3, 4) or x.shape[-3] != self.N_CHANNELS:
            raise ValueError(
                "NCA_sycl expects [C,H,W] or [B,C,H,W] with "
                f"C={self.N_CHANNELS}, got {x.shape}"
            )
        if x.dtype != jnp.float32:
            raise TypeError(f"NCA_sycl currently supports float32, got {x.dtype}")

        activation_name = getattr(self.layers[1], "__name__", None)
        if activation_name != "relu":
            raise NotImplementedError(
                "NCA_sycl currently implements only the ReLU baseline NCA"
            )

        unknown_kernels = set(self.KERNEL_STR) - set(_KERNEL_FLAGS)
        if unknown_kernels:
            raise NotImplementedError(
                f"Unsupported perception kernels: {sorted(unknown_kernels)}"
            )

        try:
            padding = _PADDING_CODES[self.op.PADDING.upper()]
        except KeyError as exc:
            raise NotImplementedError(
                f"Unsupported SYCL padding mode: {self.op.PADDING!r}"
            ) from exc

        flags = 0
        for kernel_name in self.KERNEL_STR:
            flags |= _KERNEL_FLAGS[kernel_name]
        if flags == 0:
            raise ValueError("NCA_sycl requires at least one perception kernel")

        # The tiled FP32 kernels currently keep this cap to bound generated
        # code and scratch-buffer sizes. Multiples of 16 use the fast path.
        if self.N_FEATURES > 256:
            raise NotImplementedError(
                "The baseline SYCL kernel supports at most 256 features; "
                f"this model has {self.N_FEATURES}"
            )
        return flags, padding

    def _sycl_parameters(self):
        kernels = jnp.concatenate(
            (
                self.op.grad_x.weight,
                self.op.grad_y.weight,
                self.op.average.weight,
                self.op.laplacian.weight,
            ),
            axis=0,
        )[:, 0]
        weight_hidden = self.layers[0].weight[:, :, 0, 0]
        weight_output = self.layers[2].weight[:, :, 0, 0]
        bias_output = self.layers[2].bias.reshape(self.N_CHANNELS)
        return kernels, weight_hidden, weight_output, bias_output

    def _sycl_update(self, x: Array, update_mask: Array) -> Array:
        flags, padding = self._validate_sycl_configuration(x)

        # Import lazily so ordinary CPU-side model/config inspection does not
        # require the Intel-specific JAX 0.5 custom-call API.
        from NCA.model.sycl.bridge import sycl_nca_forward

        kernels, weight_hidden, weight_output, bias_output = (
            self._sycl_parameters()
        )
        return sycl_nca_forward(
            x,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            update_mask,
            kernel_flags=flags,
            padding=padding,
        )

    def batched_call(self, x: Array, keys: Array) -> Array:
        """Update ``[B,C,H,W]`` directly with shared-parameter gradients.

        Unlike ``vmap(self)``, this enters the custom VJP with an already
        batched state. The native backward call can therefore sum parameter
        gradients over all examples with one set of GEMMs instead of emitting
        per-example matrices for JAX to reduce later.
        """
        if x.ndim != 4:
            raise ValueError(f"batched_call expects [B,C,H,W], got {x.shape}")
        if keys.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected {x.shape[0]} random keys, got shape {keys.shape}"
            )
        update_mask = jax.vmap(
            lambda item_key: jax.random.bernoulli(
                item_key, p=self.FIRE_RATE, shape=x.shape[1:]
            )
        )(keys).astype(x.dtype)
        return self._sycl_update(x, update_mask)

    def batched_rollout(
        self,
        x: Array,
        keys: Array,
        *,
        boundary_code: int = 0,
        boundary_mask: Array | None = None,
        boundary_channels: int = 0,
    ) -> tuple[Array, Array]:
        """Run K sequential updates of one N leaf in a native custom call.

        ``keys`` has shape ``[K,B,2]`` for legacy PRNG keys (or ``[K,B]`` for
        typed keys). The returned trajectory has shape ``[K,B,C,H,W]`` and is
        differentiable, allowing existing per-timestep regularisers to remain
        outside the native rollout.
        """
        if x.ndim != 4:
            raise ValueError(f"batched_rollout expects [B,C,H,W], got {x.shape}")
        if keys.ndim < 2 or keys.shape[1] != x.shape[0]:
            raise ValueError(
                f"Rollout keys must have leading shape [K,{x.shape[0]}], "
                f"got {keys.shape}"
            )
        flags, padding = self._validate_sycl_configuration(x)
        if boundary_mask is None:
            boundary_mask = jnp.zeros((1,), dtype=x.dtype)
        else:
            boundary_mask = jnp.asarray(boundary_mask, dtype=x.dtype)

        masks = jax.vmap(
            lambda step_keys: jax.vmap(
                lambda item_key: jax.random.bernoulli(
                    item_key, p=self.FIRE_RATE, shape=x.shape[1:]
                )
            )(step_keys)
        )(keys).astype(x.dtype)

        from NCA.model.sycl.bridge import sycl_nca_rollout

        kernels, weight_hidden, weight_output, bias_output = (
            self._sycl_parameters()
        )
        return sycl_nca_rollout(
            x,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            masks,
            boundary_mask,
            kernel_flags=flags,
            padding=padding,
            boundary_code=boundary_code,
            boundary_channels=boundary_channels,
        )

    def __call__(
        self,
        x: Float[Array, "{self.N_CHANNELS} H W"],
        boundary_callback: Callable[[Array], Array] = lambda value: value,
        key: Array | None = None,
    ) -> Float[Array, "{self.N_CHANNELS} H W"]:
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        update_mask = jax.random.bernoulli(
            key, p=self.FIRE_RATE, shape=x.shape
        ).astype(x.dtype)
        updated = self._sycl_update(x, update_mask)
        return boundary_callback(updated)


SyclNCA = NCA

__all__ = ["NCA", "SyclNCA"]
