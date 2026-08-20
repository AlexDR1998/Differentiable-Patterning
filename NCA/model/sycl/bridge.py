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
from jax.interpreters import batching, mlir, xla
from jaxlib.hlo_helpers import custom_call

_FORWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_forward"
_BACKWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_backward"
_ROLLOUT_FORWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_rollout_forward"
_ROLLOUT_BACKWARD_TARGET_NAME = "differentiable_patterning_nca_sycl_rollout_backward"
_METADATA_VERSION = 5
_ROLLOUT_SCRATCH_GUARD_FLOATS = 64
_LIBRARY: ctypes.CDLL | None = None
_CAPSULES: tuple[object, ...] | None = None
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
    rollout_forward_capsule = make_capsule("nca_sycl_rollout_forward")
    rollout_backward_capsule = make_capsule("nca_sycl_rollout_backward")

    from jax._src.lib import xla_client

    xla_client.register_custom_call_target(
        _FORWARD_TARGET_NAME, forward_capsule, platform="SYCL", api_version=0
    )
    xla_client.register_custom_call_target(
        _BACKWARD_TARGET_NAME, backward_capsule, platform="SYCL", api_version=0
    )
    xla_client.register_custom_call_target(
        _ROLLOUT_FORWARD_TARGET_NAME,
        rollout_forward_capsule,
        platform="SYCL",
        api_version=0,
    )
    xla_client.register_custom_call_target(
        _ROLLOUT_BACKWARD_TARGET_NAME,
        rollout_backward_capsule,
        platform="SYCL",
        api_version=0,
    )
    mlir.register_lowering(_nca_forward_p, _lowering, platform="sycl")
    mlir.register_lowering(
        _nca_backward_p, _backward_lowering, platform="sycl"
    )
    mlir.register_lowering(
        _nca_rollout_forward_p, _rollout_forward_lowering, platform="sycl"
    )
    mlir.register_lowering(
        _nca_rollout_backward_p, _rollout_backward_lowering, platform="sycl"
    )
    _LIBRARY = library
    _CAPSULES = (
        forward_capsule,
        backward_capsule,
        rollout_forward_capsule,
        rollout_backward_capsule,
    )
    _REGISTERED = True


_nca_forward_p = core.Primitive("nca_sycl_forward")
_nca_forward_p.multiple_results = True
_nca_forward_p.def_impl(partial(xla.apply_primitive, _nca_forward_p))


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
    output_channels = weight_output.shape[0]
    if output_channels not in (channels, 2 * channels):
        raise ValueError(
            "NCA SYCL output width must be C (baseline) or 2C (gated), "
            f"got {output_channels} for C={channels}"
        )
    expected = {
        "kernels": (4, kernel_size, kernel_size),
        "weight_hidden": (features, features),
        "weight_output": (output_channels, features),
        "bias_output": (output_channels,),
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
        core.ShapedArray(
            (*state.shape[:-3], output_channels, *state.shape[-2:]),
            state.dtype,
        ),
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
        "=12q",
        _METADATA_VERSION,
        batch,
        channels,
        height,
        width,
        features,
        ctx.avals_in[3].shape[0],
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


def _mapped_operands(args, dimensions):
    """Move a mapped replica axis to the front for a replica-local lax.map."""
    axis_sizes = {
        value.shape[dimension]
        for value, dimension in zip(args, dimensions)
        if dimension is not None
    }
    if len(axis_sizes) != 1:
        raise ValueError(
            f"NCA SYCL mapped operands have inconsistent axis sizes: {axis_sizes}"
        )
    axis_size = axis_sizes.pop()
    return tuple(
        jnp.broadcast_to(value, (axis_size, *value.shape))
        if dimension is None
        else batching.moveaxis(value, dimension, 0)
        for value, dimension in zip(args, dimensions)
    )


def _batching_rule(args, dimensions, *, kernel_flags, padding):
    state, kernels, weight_hidden, weight_output, bias_output, update_mask = args
    state_dim, kernels_dim, hidden_dim, output_dim, bias_dim, mask_dim = dimensions
    if any(dim is not None for dim in (kernels_dim, hidden_dim, output_dim, bias_dim)):
        raise NotImplementedError("NCA_sycl does not support vmapped model weights")
    if state_dim is None or mask_dim is None:
        raise ValueError("state and update mask must be batched together")
    state = batching.moveaxis(state, state_dim, 0)
    update_mask = batching.moveaxis(update_mask, mask_dim, 0)
    if state.ndim == 5:
        mapped_args = _mapped_operands(args, dimensions)

        def apply_one(replica_args):
            return _nca_forward_p.bind(
                *replica_args,
                kernel_flags=kernel_flags,
                padding=padding,
            )

        results = jax.lax.map(apply_one, mapped_args)
        return results, (0,) * len(results)
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
_nca_backward_p.def_impl(partial(xla.apply_primitive, _nca_backward_p))


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
    del kernel_flags, padding
    if output_cotangent.shape != state.shape:
        raise ValueError("NCA SYCL output cotangent must match the state shape")
    if state.shape != update_mask.shape:
        raise ValueError("NCA SYCL update mask must match the state shape")
    if weight_output.shape[0] not in (state.shape[-3], 2 * state.shape[-3]):
        raise ValueError("NCA SYCL output width must be C or 2C")
    if bias_output.shape != (weight_output.shape[0],):
        raise ValueError("NCA SYCL output bias must match the output width")
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
        core.ShapedArray((*parameter_prefix, *bias_output.shape), dtype),
        core.ShapedArray(scratch_shape, dtype),
        core.ShapedArray(scratch_shape, dtype),
        core.ShapedArray(scratch_shape, dtype),
        core.ShapedArray(
            (*state.shape[:-3], weight_output.shape[0], *state.shape[-2:]),
            dtype,
        ),
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

    if state.ndim == 5:
        mapped_args = _mapped_operands(args, dimensions)

        def apply_one(replica_args):
            return _nca_backward_p.bind(
                *replica_args,
                kernel_flags=kernel_flags,
                padding=padding,
                per_example_weights=False,
            )

        results = jax.lax.map(apply_one, mapped_args)
        return results, (0,) * len(results)

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
    weight_output_aval,
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
        "=13q",
        _METADATA_VERSION,
        batch,
        channels,
        height,
        width,
        features,
        weight_output_aval.shape[0],
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
        ctx.avals_in[0], ctx.avals_in[2], ctx.avals_in[1], ctx.avals_in[3],
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


def _rollout_metadata(
    state_aval,
    kernels_aval,
    weight_hidden_aval,
    weight_output_aval,
    masks_aval,
    *,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    if state_aval.ndim != 4:
        raise ValueError("NCA SYCL rollout state must have shape [B,C,H,W]")
    batch, channels, height, width = state_aval.shape
    features = weight_hidden_aval.shape[0]
    kernel_size = kernels_aval.shape[-1]
    steps = masks_aval.shape[0]
    workgroup_size = 1
    while workgroup_size < max(features, channels):
        workgroup_size *= 2
    return struct.pack(
        "=16q",
        _METADATA_VERSION,
        batch,
        channels,
        height,
        width,
        features,
        weight_output_aval.shape[0],
        kernel_size,
        int(kernel_flags),
        int(padding),
        workgroup_size,
        _xmx_mode(),
        steps,
        int(boundary_code),
        int(boundary_channels),
        int(regulariser_flags),
    )


_nca_rollout_forward_p = core.Primitive("nca_sycl_rollout_forward")
_nca_rollout_forward_p.multiple_results = True
_nca_rollout_forward_p.def_impl(
    partial(xla.apply_primitive, _nca_rollout_forward_p)
)


def _rollout_forward_abstract_eval(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    *,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    del padding
    if regulariser_flags not in (0, 1, 2, 3):
        raise ValueError(
            f"Unsupported fused regulariser flags: {regulariser_flags}"
        )
    if state.ndim != 4:
        raise ValueError("NCA SYCL rollout expects state [B,C,H,W]")
    if masks.ndim != 5 or masks.shape[1:] != state.shape:
        raise ValueError(
            f"Rollout masks must have shape [K,{state.shape}], got {masks.shape}"
        )
    if masks.shape[0] < 1:
        raise ValueError("NCA SYCL rollout requires at least one step")
    if any(
        value.dtype != np.dtype(np.float32)
        for value in (
            state,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            masks,
            boundary_mask,
        )
    ):
        raise TypeError("NCA SYCL rollout currently accepts float32 only")
    channels = state.shape[1]
    features = weight_hidden.shape[0]
    expected_features = channels * (
        bool(kernel_flags & (1 << 0))
        + bool(kernel_flags & (1 << 1))
        + 2 * bool(kernel_flags & (1 << 2))
        + bool(kernel_flags & (1 << 3))
        + bool(kernel_flags & (1 << 4))
    )
    if features != expected_features:
        raise ValueError(
            f"Rollout expected {expected_features} features, got {features}"
        )
    if weight_output.shape not in (
        (channels, features),
        (2 * channels, features),
    ) or bias_output.shape != (weight_output.shape[0],):
        raise ValueError("Rollout output layer must have width C or 2C")
    if boundary_code == 1:
        expected_boundary = (boundary_channels, *state.shape[-2:])
    elif boundary_code == 2:
        expected_boundary = state.shape[-2:]
    else:
        expected_boundary = (1,)
    if boundary_mask.shape != expected_boundary:
        raise ValueError(
            f"Boundary mask has shape {boundary_mask.shape}; expected "
            f"{expected_boundary} for boundary code {boundary_code}"
        )
    scratch_shape = (state.shape[0], features, *state.shape[-2:])
    outputs = [
        core.ShapedArray(state.shape, state.dtype),
        core.ShapedArray(masks.shape, state.dtype),
        core.ShapedArray(scratch_shape, state.dtype),
        core.ShapedArray(scratch_shape, state.dtype),
        core.ShapedArray(
            (state.shape[0], weight_output.shape[0], *state.shape[-2:]),
            state.dtype,
        ),
    ]
    if regulariser_flags:
        outputs.insert(2, core.ShapedArray((2,), state.dtype))
    return tuple(outputs)


_nca_rollout_forward_p.def_abstract_eval(_rollout_forward_abstract_eval)


def _rollout_forward_batching_rule(
    args,
    dimensions,
    *,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    mapped_args = _mapped_operands(args, dimensions)

    def apply_one(replica_args):
        return _nca_rollout_forward_p.bind(
            *replica_args,
            kernel_flags=kernel_flags,
            padding=padding,
            boundary_code=boundary_code,
            boundary_channels=boundary_channels,
            regulariser_flags=regulariser_flags,
        )

    results = jax.lax.map(apply_one, mapped_args)
    return results, (0,) * len(results)


batching.primitive_batchers[_nca_rollout_forward_p] = (
    _rollout_forward_batching_rule
)


def _rollout_forward_lowering(
    ctx,
    *operands,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    metadata = _rollout_metadata(
        ctx.avals_in[0],
        ctx.avals_in[1],
        ctx.avals_in[2],
        ctx.avals_in[3], ctx.avals_in[5],
        kernel_flags=kernel_flags,
        padding=padding,
        boundary_code=boundary_code,
        boundary_channels=boundary_channels,
        regulariser_flags=regulariser_flags,
    )
    operation = custom_call(
        _ROLLOUT_FORWARD_TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=operands,
        backend_config=metadata,
        api_version=1,
        operand_layouts=[
            tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_in
        ],
        result_layouts=[
            tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_out
        ],
    )
    return operation.results


_nca_rollout_backward_p = core.Primitive("nca_sycl_rollout_backward")
_nca_rollout_backward_p.multiple_results = True
_nca_rollout_backward_p.def_impl(
    partial(xla.apply_primitive, _nca_rollout_backward_p)
)


def _rollout_backward_abstract_eval(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    trajectory,
    output_cotangent,
    trajectory_cotangent,
    regulariser_cotangent=None,
    *,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    del (
        kernel_flags,
        padding,
        boundary_code,
        boundary_channels,
    )
    if trajectory.shape != masks.shape:
        raise ValueError("Rollout trajectory and masks must have equal shapes")
    if output_cotangent.shape != state.shape:
        raise ValueError("Rollout output cotangent must match state")
    if trajectory_cotangent.shape != trajectory.shape:
        raise ValueError("Rollout trajectory cotangent must match trajectory")
    channels = state.shape[1]
    if weight_output.shape[0] not in (channels, 2 * channels):
        raise ValueError("Rollout output width must be C or 2C")
    if bias_output.shape != (weight_output.shape[0],):
        raise ValueError("Rollout output bias must match the output width")
    if regulariser_flags:
        if regulariser_cotangent is None or regulariser_cotangent.shape != (2,):
            raise ValueError("Rollout regulariser cotangent must have shape [2]")
    elif regulariser_cotangent is not None:
        raise ValueError(
            "A regulariser cotangent was supplied while fused regularisers "
            "are disabled"
        )
    checked_values = [
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        trajectory,
        output_cotangent,
        trajectory_cotangent,
    ]
    if regulariser_cotangent is not None:
        checked_values.append(regulariser_cotangent)
    if any(
        value.dtype != np.dtype(np.float32)
        for value in checked_values
    ):
        raise TypeError("NCA SYCL rollout backward accepts float32 only")
    scratch_shape = (state.shape[0], weight_hidden.shape[0], *state.shape[-2:])
    diagnostic_mode = os.environ.get("NCA_SYCL_DIAGNOSTIC_SCRATCH", "").lower()
    if diagnostic_mode not in ("", "reuse", "per_step"):
        raise ValueError(
            "NCA_SYCL_DIAGNOSTIC_SCRATCH must be unset, 'reuse', or "
            f"'per_step', got {diagnostic_mode!r}"
        )

    def workspace(shape):
        if not diagnostic_mode:
            return core.ShapedArray(shape, state.dtype)
        slots = trajectory.shape[0] if diagnostic_mode == "per_step" else 1
        elements = int(np.prod(shape))
        guarded_elements = slots * (
            elements + 2 * _ROLLOUT_SCRATCH_GUARD_FLOATS
        )
        return core.ShapedArray((guarded_elements,), state.dtype)

    return (
        core.ShapedArray(state.shape, state.dtype),
        core.ShapedArray(weight_hidden.shape, state.dtype),
        core.ShapedArray(weight_output.shape, state.dtype),
        core.ShapedArray(bias_output.shape, state.dtype),
        workspace(state.shape),
        workspace(state.shape),
        workspace(weight_hidden.shape),
        workspace(weight_output.shape),
        workspace(bias_output.shape),
        workspace(scratch_shape),
        workspace(scratch_shape),
        workspace(scratch_shape),
        workspace((state.shape[0], weight_output.shape[0], *state.shape[-2:])),
    )


_nca_rollout_backward_p.def_abstract_eval(_rollout_backward_abstract_eval)


def _rollout_backward_batching_rule(
    args,
    dimensions,
    *,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    mapped_args = _mapped_operands(args, dimensions)

    def apply_one(replica_args):
        return _nca_rollout_backward_p.bind(
            *replica_args,
            kernel_flags=kernel_flags,
            padding=padding,
            boundary_code=boundary_code,
            boundary_channels=boundary_channels,
            regulariser_flags=regulariser_flags,
        )

    results = jax.lax.map(apply_one, mapped_args)
    return results, (0,) * len(results)


batching.primitive_batchers[_nca_rollout_backward_p] = (
    _rollout_backward_batching_rule
)


def _rollout_backward_lowering(
    ctx,
    *operands,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    metadata = _rollout_metadata(
        ctx.avals_in[0],
        ctx.avals_in[1],
        ctx.avals_in[2],
        ctx.avals_in[3], ctx.avals_in[5],
        kernel_flags=kernel_flags,
        padding=padding,
        boundary_code=boundary_code,
        boundary_channels=boundary_channels,
        regulariser_flags=regulariser_flags,
    )
    operation = custom_call(
        _ROLLOUT_BACKWARD_TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
        operands=operands,
        backend_config=metadata,
        api_version=1,
        operand_layouts=[
            tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_in
        ],
        result_layouts=[
            tuple(range(aval.ndim - 1, -1, -1)) for aval in ctx.avals_out
        ],
    )
    return operation.results


def _bind_rollout_forward(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    results = _nca_rollout_forward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        kernel_flags=kernel_flags,
        padding=padding,
        boundary_code=boundary_code,
        boundary_channels=boundary_channels,
        regulariser_flags=regulariser_flags,
    )
    regularisers = (
        results[2]
        if regulariser_flags
        else jnp.zeros((2,), dtype=state.dtype)
    )
    return results[0], results[1], regularisers


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _differentiable_rollout(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    return _bind_rollout_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        kernel_flags,
        padding,
        boundary_code,
        boundary_channels,
        regulariser_flags,
    )


def _rollout_vjp_fwd(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
):
    output, trajectory, regularisers = _bind_rollout_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        kernel_flags,
        padding,
        boundary_code,
        boundary_channels,
        regulariser_flags,
    )
    residuals = (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        trajectory,
    )
    return (output, trajectory, regularisers), residuals


def _rollout_vjp_bwd(
    kernel_flags,
    padding,
    boundary_code,
    boundary_channels,
    regulariser_flags,
    residuals,
    cotangents,
):
    (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        trajectory,
    ) = residuals
    output_cotangent, trajectory_cotangent, regulariser_cotangent = cotangents
    operands = (
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        trajectory,
        output_cotangent,
        trajectory_cotangent,
    )
    if regulariser_flags:
        operands = (*operands, regulariser_cotangent)
    results = _nca_rollout_backward_p.bind(
        *operands,
        kernel_flags=kernel_flags,
        padding=padding,
        boundary_code=boundary_code,
        boundary_channels=boundary_channels,
        regulariser_flags=regulariser_flags,
    )
    return (
        results[0],
        jnp.zeros_like(kernels),
        results[1],
        results[2],
        results[3],
        jnp.zeros_like(masks),
        jnp.zeros_like(boundary_mask),
    )


_differentiable_rollout.defvjp(_rollout_vjp_fwd, _rollout_vjp_bwd)


def sycl_nca_rollout(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    *,
    kernel_flags: int,
    padding: int,
    boundary_code: int,
    boundary_channels: int,
    regulariser_flags: int = 0,
):
    """Execute several sequential NCA steps in one native custom call."""
    _register_custom_call()
    return _differentiable_rollout(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        kernel_flags,
        padding,
        boundary_code,
        boundary_channels,
        regulariser_flags,
    )


def sycl_nca_rollout_backward_diagnostic(
    state,
    kernels,
    weight_hidden,
    weight_output,
    bias_output,
    masks,
    boundary_mask,
    output_cotangent,
    trajectory_cotangent,
    *,
    kernel_flags: int,
    padding: int,
    boundary_code: int,
    boundary_channels: int,
):
    """Return native two-pass gradients and all scratch results for guards.

    This entry point is intended only for the hardware corruption probe. Set
    ``NCA_SYCL_DIAGNOSTIC_SCRATCH`` to ``reuse`` or ``per_step`` before JAX
    traces the function. The C++ custom call reads the same process setting.
    """
    diagnostic_mode = os.environ.get("NCA_SYCL_DIAGNOSTIC_SCRATCH", "").lower()
    if diagnostic_mode not in ("reuse", "per_step"):
        raise RuntimeError(
            "sycl_nca_rollout_backward_diagnostic requires "
            "NCA_SYCL_DIAGNOSTIC_SCRATCH=reuse or per_step"
        )
    _register_custom_call()
    _, trajectory, _ = _bind_rollout_forward(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        kernel_flags,
        padding,
        boundary_code,
        boundary_channels,
        0,
    )
    return _nca_rollout_backward_p.bind(
        state,
        kernels,
        weight_hidden,
        weight_output,
        bias_output,
        masks,
        boundary_mask,
        trajectory,
        output_cotangent,
        trajectory_cotangent,
        kernel_flags=kernel_flags,
        padding=padding,
        boundary_code=boundary_code,
        boundary_channels=boundary_channels,
        regulariser_flags=0,
    )


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


__all__ = [
    "sycl_nca_forward",
    "sycl_nca_rollout",
    "sycl_nca_rollout_backward_diagnostic",
]
