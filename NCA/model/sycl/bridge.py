"""JAX primitive and Intel SYCL custom-call registration for ``NCA_sycl``."""

from __future__ import annotations

import ctypes
from functools import partial
import os
import pathlib
import struct

import jax
import numpy as np
from jax import core
from jax.interpreters import batching, mlir
from jaxlib.hlo_helpers import custom_call

from NCA.model.sycl.reference import jax_nca_forward


_TARGET_NAME = "differentiable_patterning_nca_sycl_forward"
_METADATA_VERSION = 1
_LIBRARY: ctypes.CDLL | None = None
_CAPSULE: object | None = None
_REGISTERED = False


def _default_library_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "files" / "libnca_sycl.so"


def _library_path() -> pathlib.Path:
    configured = os.environ.get("NCA_SYCL_LIBRARY")
    return pathlib.Path(configured).expanduser() if configured else _default_library_path()


def _register_custom_call() -> None:
    global _LIBRARY, _CAPSULE, _REGISTERED
    if _REGISTERED:
        return

    # Force PJRT plugin discovery before asking for the plugin-installed SYCL
    # custom-call registration handler.
    if jax.default_backend() != "sycl":
        raise RuntimeError(
            "NCA_sycl requires JAX's Intel backend; "
            f"the active backend is {jax.default_backend()!r}"
        )

    path = _library_path().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"NCA SYCL library not found at {path}. Build it with "
            "NCA/model/sycl/files/build_nca_sycl.sh or set NCA_SYCL_LIBRARY."
        )

    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    function = library.nca_sycl_forward
    address = ctypes.cast(function, ctypes.c_void_p).value
    if address is None:
        raise RuntimeError("nca_sycl_forward resolved to a null address")

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(
        ctypes.c_void_p(address), b"xla._CUSTOM_CALL_TARGET", None
    )

    from jax._src.lib import xla_client

    xla_client.register_custom_call_target(
        _TARGET_NAME, capsule, platform="SYCL", api_version=0
    )
    mlir.register_lowering(_nca_forward_p, _lowering, platform="sycl")
    _LIBRARY = library
    _CAPSULE = capsule
    _REGISTERED = True


_nca_forward_p = core.Primitive("nca_sycl_forward")


def _abstract_eval(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    *,
    kernel_flags,
    padding,
):
    del padding
    if state.ndim not in (3, 4):
        raise ValueError("NCA SYCL state must have shape [C,H,W] or [B,C,H,W]")
    if state.shape != update_mask.shape:
        raise ValueError("NCA SYCL state and update mask shapes must match")
    channels = state.shape[-3]
    features = weight_hidden.shape[0]
    kernel_size = kernels.shape[-1]
    feature_multiplier = (
        bool(kernel_flags & (1 << 0))
        + bool(kernel_flags & (1 << 1))
        + 2 * bool(kernel_flags & (1 << 2))
        + bool(kernel_flags & (1 << 3))
        + bool(kernel_flags & (1 << 4))
    )
    if features != channels * feature_multiplier:
        raise ValueError(
            f"NCA SYCL metadata implies {channels * feature_multiplier} "
            f"features, but the hidden layer has {features}"
        )
    if kernel_size % 2 != 1:
        raise ValueError("NCA SYCL perception kernels must have odd size")
    expected = {
        "kernels": (4, kernel_size, kernel_size),
        "weight_hidden": (features, features),
        "weight_output": (channels, features),
        "bias_output": (channels,),
    }
    actual = {
        "kernels": kernels.shape,
        "weight_hidden": weight_hidden.shape,
        "weight_output": weight_output.shape,
        "bias_output": bias_output.shape,
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(
                f"NCA SYCL {name} has shape {actual[name]}, expected {expected_shape}"
            )
    operands = (state, kernels, weight_hidden, weight_output, bias_output, update_mask)
    if any(value.dtype != np.dtype(np.float32) for value in operands):
        raise TypeError("The baseline NCA SYCL custom call accepts float32 only")
    return core.ShapedArray(state.shape, state.dtype)


_nca_forward_p.def_abstract_eval(_abstract_eval)


def _lowering(ctx, *operands, kernel_flags, padding):
    state_aval = ctx.avals_in[0]
    if state_aval.ndim == 3:
        batch = 1
        channels, height, width = state_aval.shape
    else:
        batch, channels, height, width = state_aval.shape
    features = ctx.avals_in[2].shape[0]
    kernel_size = ctx.avals_in[1].shape[-1]
    workgroup_size = 1
    while workgroup_size < max(features, channels):
        workgroup_size *= 2

    metadata = struct.pack(
        "=10q",
        _METADATA_VERSION,
        batch,
        channels,
        height,
        width,
        features,
        kernel_size,
        int(kernel_flags),
        int(padding),
        workgroup_size,
    )
    operand_layouts = [
        tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_in
    ]
    output_layout = tuple(range(state_aval.ndim - 1, -1, -1))
    operation = custom_call(
        _TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=operands,
        backend_config=metadata,
        # StableHLO/XLA's API_VERSION_ORIGINAL enum value is 1. Registration
        # still uses api_version=0 above because the PJRT registration API uses
        # 0 to distinguish an untyped target from typed FFI.
        api_version=1,
        operand_layouts=operand_layouts,
        result_layouts=[output_layout],
    )
    return operation.results


def _batching_rule(args, dimensions, *, kernel_flags, padding):
    state, kernels, weight_hidden, weight_output, bias_output, update_mask = args
    state_dim, kernels_dim, hidden_dim, output_dim, bias_dim, mask_dim = dimensions
    if any(dim is not None for dim in (kernels_dim, hidden_dim, output_dim, bias_dim)):
        raise NotImplementedError("NCA_sycl does not support vmapped model weights")
    if state_dim is None or mask_dim is None:
        raise ValueError("state and update mask must be batched together")
    state = batching.moveaxis(state, state_dim, 0)
    update_mask = batching.moveaxis(update_mask, mask_dim, 0)
    result = _nca_forward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags=kernel_flags,
        padding=padding,
    )
    return result, 0


batching.primitive_batchers[_nca_forward_p] = _batching_rule


def _bind_native_forward(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    kernel_flags,
    padding,
):
    return _nca_forward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags=kernel_flags,
        padding=padding,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _differentiable_forward(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    kernel_flags,
    padding,
):
    return _bind_native_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags,
        padding,
    )


def _forward_vjp_rule(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    kernel_flags,
    padding,
):
    result = _bind_native_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags,
        padding,
    )
    residuals = (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
    )
    return result, residuals


def _backward_vjp_rule(kernel_flags, padding, residuals, output_cotangent):
    def reference(*operands):
        return jax_nca_forward(
            *operands, kernel_flags=kernel_flags, padding=padding
        )

    _, pullback = jax.vjp(reference, *residuals)
    return pullback(output_cotangent)


_differentiable_forward.defvjp(_forward_vjp_rule, _backward_vjp_rule)


def sycl_nca_forward(
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
    """Execute one baseline NCA update using the registered SYCL kernel."""
    _register_custom_call()
    return _differentiable_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags,
        padding,
    )


__all__ = ["sycl_nca_forward"]
