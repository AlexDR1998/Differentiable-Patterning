#!/usr/bin/env python3
"""Register and execute a minimal SYCL kernel as a JAX custom call."""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import mlir
from jaxlib.hlo_helpers import custom_call


TARGET_NAME = "differentiable_patterning_sycl_axpy"
_LOADED_LIBRARY: ctypes.CDLL | None = None
_TARGET_CAPSULE: object | None = None


def register_custom_call(library_path: pathlib.Path) -> None:
    """Load the SYCL DSO and register its exported function with XLA."""
    global _LOADED_LIBRARY, _TARGET_CAPSULE

    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    function = library.jax_sycl_axpy
    address = ctypes.cast(function, ctypes.c_void_p).value
    if address is None:
        raise RuntimeError("jax_sycl_axpy resolved to a null address")

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(
        ctypes.c_void_p(address), b"xla._CUSTOM_CALL_TARGET", None
    )

    # Import this only after JAX has initialized its PJRT plugins. Intel's
    # plugin installs a registration handler for the uppercase SYCL platform.
    from jax._src.lib import xla_client  # pylint: disable=import-outside-toplevel

    xla_client.register_custom_call_target(
        TARGET_NAME, capsule, platform="SYCL", api_version=0
    )
    mlir.register_lowering(sycl_axpy_p, _sycl_axpy_lowering, platform="sycl")

    # Keep both objects alive for the lifetime of compiled executables.
    _LOADED_LIBRARY = library
    _TARGET_CAPSULE = capsule


sycl_axpy_p = core.Primitive("sycl_axpy")


def _sycl_axpy_abstract(x: core.ShapedArray, y: core.ShapedArray):
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if x.dtype != np.dtype(np.float32) or y.dtype != np.dtype(np.float32):
        raise TypeError("the bridge smoke kernel accepts float32 arrays only")
    return core.ShapedArray(x.shape, x.dtype)


sycl_axpy_p.def_abstract_eval(_sycl_axpy_abstract)


def _sycl_axpy_lowering(ctx: mlir.LoweringRuleContext, x, y):
    output_aval = ctx.avals_out[0]
    element_count = int(np.prod(output_aval.shape, dtype=np.int64))
    layout = tuple(range(len(output_aval.shape) - 1, -1, -1))
    operation = custom_call(
        TARGET_NAME,
        result_types=[mlir.aval_to_ir_type(output_aval)],
        operands=[x, y],
        backend_config=struct.pack("=Q", element_count),
        # StableHLO/XLA numbers the original custom-call ABI as 1. This is
        # distinct from register_custom_call_target's api_version=0 above,
        # where 0 means an untyped registration.
        api_version=1,
        operand_layouts=[layout, layout],
        result_layouts=[layout],
    )
    return operation.results


def sycl_axpy(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute ``1.5 * x + y`` through the custom SYCL kernel."""
    return sycl_axpy_p.bind(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=pathlib.Path)
    parser.add_argument("--size", type=int, default=1 << 20)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")

    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEFAULT_BACKEND={jax.default_backend()}")
    print(f"JAX_DEVICES={jax.devices()}")
    if jax.default_backend() != "sycl":
        raise RuntimeError("expected JAX's default backend to be 'sycl'")

    register_custom_call(args.library.resolve())

    x = jnp.arange(args.size, dtype=jnp.float32) * np.float32(0.125)
    y = jnp.arange(args.size, dtype=jnp.float32) * np.float32(-0.25)
    compiled_axpy = jax.jit(sycl_axpy)
    output = compiled_axpy(x, y)
    output.block_until_ready()

    expected = np.float32(1.5) * np.asarray(x) + np.asarray(y)
    actual = np.asarray(output)
    max_absolute_error = float(np.max(np.abs(actual - expected)))

    print("JAX_SYCL_BRIDGE_VERSION=1")
    print(f"ELEMENT_COUNT={args.size}")
    print(f"OUTPUT_DEVICE={output.device}")
    print(f"MAX_ABSOLUTE_ERROR={max_absolute_error}")
    if max_absolute_error != 0.0:
        raise RuntimeError("custom-call output did not match the JAX reference")
    print("JAX_SYCL_BRIDGE_RESULT=PASS")


if __name__ == "__main__":
    main()
