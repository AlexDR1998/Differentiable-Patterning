"""JAX primitive and Intel SYCL custom-call registration for ``NCA_sycl``."""

from __future__ import annotations

import ctypes
from functools import partial
import os
import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import batching, mlir
from jaxlib.hlo_helpers import custom_call

_FORWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_forward"
_BACKWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_backward"
_METADATA_VERSION = 3
_LIBRARY: ctypes.CDLL | None = None
_CAPSULES: tuple[object, object] | None = None
_REGISTERED = False


def _default_library_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "files" / "libnca_sycl.so"


def _library_path() -> pathlib.Path:
    configured = os.environ.get("NCA_SYCL_LIBRARY")
    return pathlib.Path(configured).expanduser() if configured else _default_library_path()


def _xmx_mode() -> int:
    """Map the sweep's JAX precision setting onto a oneMKL XMX mode."""
    override = os.environ.get("NCA_SYCL_XMX_MODE")
    precision = (
        override
        if override is not None
        else str(jax.config.jax_default_matmul_precision)
    ).lower()
    modes = {
        "highest": 0,
        "none": 0,
        "float32": 0,
        "standard": 0,
        "tensorfloat32": 1,
        "tf32": 1,
        "bfloat16": 2,
        "bf16": 2,
        "bf16x2": 3,
        "bf16x3": 4,
    }
    try:
        return modes[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported NCA_SYCL_XMX_MODE/matmul precision: {precision!r}"
        ) from exc


def _register_custom_call() -> None:
    global _LIBRARY, _CAPSULES, _REGISTERED
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
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    def make_capsule(symbol_name: str):
        function = getattr(library, symbol_name)
        address = ctypes.cast(function, ctypes.c_void_p).value
        if address is None:
            raise RuntimeError(f"{symbol_name} resolved to a null address")
        return capsule_new(
            ctypes.c_void_p(address), b"xla._CUSTOM_CALL_TARGET", None
        )

    forward_capsule = make_capsule("nca_sycl_forward")
    backward_capsule = make_capsule("nca_sycl_backward")

    from jax._src.lib import xla_client

    xla_client.register_custom_call_target(
        _FORWARD_TARGET_NAME, forward_capsule, platform="SYCL", api_version=0
    )
    xla_client.register_custom_call_target(
        _BACKWARD_TARGET_NAME, backward_capsule, platform="SYCL", api_version=0
    )
    mlir.register_lowering(_nca_forward_p, _lowering, platform="sycl")
    mlir.register_lowering(
        _nca_backward_p, _backward_lowering, platform="sycl"
    )
    _LIBRARY = library
    _CAPSULES = (forward_capsule, backward_capsule)
    _REGISTERED = True


_nca_forward_p = core.Primitive("nca_sycl_forward")
_nca_forward_p.multiple_results = True


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
    if state.ndim == 3:
        scratch_shape = (features, *state.shape[-2:])
    else:
        scratch_shape = (state.shape[0], features, *state.shape[-2:])
    return (
        core.ShapedArray(state.shape, state.dtype),
        core.ShapedArray(scratch_shape, state.dtype),
        core.ShapedArray(scratch_shape, state.dtype),
        core.ShapedArray(state.shape, state.dtype),
    )


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
        "=11q",
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
        _xmx_mode(),
    )
    operand_layouts = [
        tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_in
    ]
    result_layouts = [
        tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_out
    ]
    operation = custom_call(
        _FORWARD_TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=operands,
        backend_config=metadata,
        # StableHLO/XLA's API_VERSION_ORIGINAL enum value is 1. Registration
        # still uses api_version=0 above because the PJRT registration API uses
        # 0 to distinguish an untyped target from typed FFI.
        api_version=1,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
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
    results = _nca_forward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags=kernel_flags,
        padding=padding,
    )
    return results, (0, 0, 0, 0)


batching.primitive_batchers[_nca_forward_p] = _batching_rule


_nca_backward_p = core.Primitive("nca_sycl_backward")
_nca_backward_p.multiple_results = True


def _backward_abstract_eval(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    output_cotangent,
    *,
    kernel_flags,
    padding,
    per_example_weights,
):
    del bias_output, kernel_flags, padding
    if output_cotangent.shape != state.shape:
        raise ValueError("NCA SYCL output cotangent must match the state shape")
    if state.shape != update_mask.shape:
        raise ValueError("NCA SYCL update mask must match the state shape")
    if state.ndim == 3:
        scratch_shape = (weight_hidden.shape[0], *state.shape[-2:])
    elif state.ndim == 4:
        scratch_shape = (
            state.shape[0],
            weight_hidden.shape[0],
            *state.shape[-2:],
        )
    else:
        raise ValueError("NCA SYCL backward expects rank-three or rank-four state")
    dtype = state.dtype
    operands = (
        state,
        kernels,
        weight_hidden,
        weight_output,
        update_mask,
        output_cotangent,
    )
    if any(value.dtype != np.dtype(np.float32) for value in operands):
        raise TypeError("The baseline NCA SYCL backward call accepts float32 only")
    if per_example_weights:
        if state.ndim != 4:
            raise ValueError(
                "Per-example NCA SYCL gradients require a batched state"
            )
        parameter_prefix = (state.shape[0],)
    else:
        parameter_prefix = ()
    return (
        core.ShapedArray(state.shape, dtype),
        core.ShapedArray((*parameter_prefix, *weight_hidden.shape), dtype),
        core.ShapedArray((*parameter_prefix, *weight_output.shape), dtype),
        core.ShapedArray((*parameter_prefix, state.shape[-3]), dtype),
        core.ShapedArray(scratch_shape, dtype),
        core.ShapedArray(scratch_shape, dtype),
        core.ShapedArray(scratch_shape, dtype),
    )


_nca_backward_p.def_abstract_eval(_backward_abstract_eval)


def _backward_batching_rule(
    args, dimensions, *, kernel_flags, padding, per_example_weights
):
    if per_example_weights:
        raise NotImplementedError("Nested vmap of NCA_sycl backward is unsupported")
    (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        output_cotangent,
    ) = args
    (
        state_dim,
        kernels_dim,
        hidden_dim,
        output_dim,
        bias_dim,
        mask_dim,
        cotangent_dim,
    ) = dimensions
    if any(
        dim is not None
        for dim in (kernels_dim, hidden_dim, output_dim, bias_dim)
    ):
        raise NotImplementedError(
            "NCA_sycl does not support vmapped model parameters"
        )
    if state_dim is None or mask_dim is None or cotangent_dim is None:
        raise ValueError(
            "state, update mask, and output cotangent must be batched together"
        )

    state = batching.moveaxis(state, state_dim, 0)
    update_mask = batching.moveaxis(update_mask, mask_dim, 0)
    output_cotangent = batching.moveaxis(output_cotangent, cotangent_dim, 0)

    # Emit one batched custom call, but retain a leading batch dimension on
    # parameter cotangents. JAX's transpose of vmap then performs the reduction
    # required for parameters shared by all examples.
    results = _nca_backward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        output_cotangent,
        kernel_flags=kernel_flags,
        padding=padding,
        per_example_weights=True,
    )
    return results, (0,) * len(results)


batching.primitive_batchers[_nca_backward_p] = _backward_batching_rule


def _metadata_from_avals(
    state_aval,
    weight_hidden_aval,
    kernels_aval,
    kernel_flags,
    padding,
    per_example_weights,
):
    if state_aval.ndim == 3:
        batch = 1
        channels, height, width = state_aval.shape
    else:
        batch, channels, height, width = state_aval.shape
    features = weight_hidden_aval.shape[0]
    kernel_size = kernels_aval.shape[-1]
    workgroup_size = 1
    while workgroup_size < max(features, channels):
        workgroup_size *= 2
    return struct.pack(
        "=12q",
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
        int(per_example_weights),
        _xmx_mode(),
    )


def _backward_lowering(
    ctx, *operands, kernel_flags, padding, per_example_weights
):
    metadata = _metadata_from_avals(
        ctx.avals_in[0], ctx.avals_in[2], ctx.avals_in[1],
        kernel_flags, padding, per_example_weights
    )
    operand_layouts = [
        tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_in
    ]
    result_layouts = [
        tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_out
    ]
    operation = custom_call(
        _BACKWARD_TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=operands,
        backend_config=metadata,
        api_version=1,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return operation.results


def _bind_native_backward(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    update_mask,
    output_cotangent,
    kernel_flags,
    padding,
    per_example_weights=False,
):
    results = _nca_backward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        output_cotangent,
        kernel_flags=kernel_flags,
        padding=padding,
        per_example_weights=per_example_weights,
    )
    state_gradient, hidden_gradient, output_gradient, bias_gradient = results[:4]
    return state_gradient, hidden_gradient, output_gradient, bias_gradient


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
    results = _nca_forward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
        kernel_flags=kernel_flags,
        padding=padding,
    )
    return results[0]


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
    (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        update_mask,
    ) = residuals
    state_gradient, hidden_gradient, output_gradient, bias_gradient = (
        _bind_native_backward(
            state,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            update_mask,
            output_cotangent,
            kernel_flags,
            padding,
        )
    )
    return (
        state_gradient,
        jnp.zeros_like(kernels),
        hidden_gradient,
        output_gradient,
        bias_gradient,
        jnp.zeros_like(update_mask),
    )


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
