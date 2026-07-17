#!/usr/bin/env python3
"""End-to-end forward and gradient smoke test for the JAX/SYCL NCA."""

from __future__ import annotations

import time
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from NCA.model.NCA_model_fast import NCA as FastNCA
from NCA.model.NCA_sycl import NCA as SyclNCA


CHANNELS = 32
HEIGHT = int(os.environ.get("NCA_SYCL_SMOKE_HEIGHT", "17"))
WIDTH = int(os.environ.get("NCA_SYCL_SMOKE_WIDTH", "19"))
BATCH = 2
KERNEL_STR = ["ID", "LAP", "DIFF"]
FIRE_RATE = 0.5
PADDING = os.environ.get("NCA_SYCL_SMOKE_PADDING", "CIRCULAR").upper()
# BF16 XMX multiplies accumulate into FP32. These bounds detect structural
# gradient errors while allowing the intended operand-rounding difference.
RTOL = 2.0e-2
ATOL = 2.0e-3


def _max_absolute_error(actual, expected) -> float:
    actual_host = np.asarray(actual)
    expected_host = np.asarray(expected)
    return float(np.max(np.abs(actual_host - expected_host)))


def _assert_close(name: str, actual, expected) -> float:
    error = _max_absolute_error(actual, expected)
    print(f"{name}_MAX_ABSOLUTE_ERROR={error}")
    if not np.allclose(np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL):
        raise AssertionError(
            f"{name} differs from the portable JAX reference: max error {error}"
        )
    return error


def _batched_outputs(model, states, keys):
    return jax.vmap(lambda state, key: model(state, key=key))(states, keys)


def _joint_loss(model_and_states, keys):
    model, states = model_and_states
    outputs = _batched_outputs(model, states, keys)
    return jnp.mean(outputs**2), outputs


def _direct_batched_joint_loss(model_and_states, keys):
    model, states = model_and_states
    outputs = model.batched_call(states, keys)
    return jnp.mean(outputs**2), outputs


def _make_models(key):
    fast_model = FastNCA(
        CHANNELS,
        KERNEL_STR=KERNEL_STR,
        PADDING=PADDING,
        FIRE_RATE=FIRE_RATE,
        key=key,
    )
    sycl_model = SyclNCA(
        CHANNELS,
        KERNEL_STR=KERNEL_STR,
        PADDING=PADDING,
        FIRE_RATE=FIRE_RATE,
        key=key,
    )

    weight_keys = jax.random.split(jax.random.fold_in(key, 1), 3)
    weights = (
        0.02
        * jax.random.normal(weight_keys[0], fast_model.layers[0].weight.shape),
        0.02
        * jax.random.normal(weight_keys[1], fast_model.layers[2].weight.shape),
        0.02
        * jax.random.normal(weight_keys[2], fast_model.layers[2].bias.shape),
    )
    fast_model.set_weights(weights)
    sycl_model.set_weights(weights)
    return fast_model, sycl_model


def main() -> None:
    os.environ.setdefault("NCA_SYCL_XMX_MODE", "bf16")
    print("NCA_SYCL_SMOKE_VERSION=3")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEFAULT_BACKEND={jax.default_backend()}")
    print(f"JAX_DEVICES={jax.devices()}")
    print("BACKWARD_IMPLEMENTATION=SYCL_CUSTOM_CALL")
    print("BACKWARD_BATCHING=SINGLE_CUSTOM_CALL")
    print("TRAINER_BATCHING=FLATTEN_BXN_SHARED_GRADIENTS")
    print("FORWARD_DENSE_IMPLEMENTATION=ONEMKL_GEMM")
    print("PERCEPTION_SPATIAL_TILING=8X16_SLM")
    print("BACKWARD_DENSE_IMPLEMENTATION=ONEMKL_GEMM")
    print("DIFF_BACKWARD_STENCIL=TILED_ATOMIC_FREE_GATHER_WHEN_SUPPORTED")
    print(
        "NCA_SYCL_XMX_MODE="
        f"{os.environ.get('NCA_SYCL_XMX_MODE', str(jax.config.jax_default_matmul_precision))}"
    )
    print("DENSE_IMPLEMENTATION=ONEMKL_XMX_COMPUTE_MODE")
    print(f"NUMERICAL_TOLERANCE=rtol:{RTOL},atol:{ATOL}")
    print(f"TEST_SHAPE={BATCH}X{CHANNELS}X{HEIGHT}X{WIDTH}")
    print(f"TEST_PADDING={PADDING}")
    if jax.default_backend() != "sycl":
        raise RuntimeError("NCA SYCL smoke test requires the 'sycl' JAX backend")

    root_key = jax.random.PRNGKey(20260716)
    model_key, state_key, rollout_key = jax.random.split(root_key, 3)
    fast_model, sycl_model = _make_models(model_key)
    states = jax.random.normal(
        state_key, (BATCH, CHANNELS, HEIGHT, WIDTH), dtype=jnp.float32
    )
    keys = jax.random.split(rollout_key, BATCH)

    fast_forward = eqx.filter_jit(_batched_outputs)
    sycl_forward = eqx.filter_jit(_batched_outputs)

    print("PHASE=JAX_REFERENCE_FORWARD", flush=True)
    start = time.perf_counter()
    expected = fast_forward(fast_model, states, keys)
    expected.block_until_ready()
    print(f"JAX_REFERENCE_FORWARD_COMPILE_EXECUTE_SECONDS={time.perf_counter() - start}")

    print("PHASE=SYCL_FORWARD", flush=True)
    start = time.perf_counter()
    actual = sycl_forward(sycl_model, states, keys)
    actual.block_until_ready()
    print(f"SYCL_FORWARD_COMPILE_EXECUTE_SECONDS={time.perf_counter() - start}")
    _assert_close("FORWARD", actual, expected)
    print(f"OUTPUT_DEVICE={actual.device}")

    # Final trainer evaluation calls model.__call__ directly from a Python
    # rollout rather than through filter_jit. Exercise the primitive's eager
    # dispatch rule explicitly so that training cannot succeed and evaluation
    # subsequently fail.
    print("PHASE=SYCL_EAGER_FORWARD", flush=True)
    expected_eager = fast_model(states[0], key=keys[0])
    actual_eager = sycl_model(states[0], key=keys[0])
    actual_eager.block_until_ready()
    _assert_close("EAGER_FORWARD", actual_eager, expected_eager)

    value_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(_joint_loss, has_aux=True)
    )
    print("PHASE=JAX_REFERENCE_BACKWARD", flush=True)
    start = time.perf_counter()
    (expected_loss, _), (expected_gradients, expected_state_gradient) = (
        value_and_grad((fast_model, states), keys)
    )
    jax.block_until_ready(
        (
            expected_loss,
            expected_gradients.layers[0].weight,
            expected_gradients.layers[2].weight,
            expected_gradients.layers[2].bias,
            expected_state_gradient,
        )
    )
    print(
        "JAX_REFERENCE_BACKWARD_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - start}"
    )

    print("PHASE=SYCL_BACKWARD", flush=True)
    start = time.perf_counter()
    (actual_loss, _), (actual_gradients, actual_state_gradient) = value_and_grad(
        (sycl_model, states), keys
    )
    jax.block_until_ready(
        (
            actual_loss,
            actual_gradients.layers[0].weight,
            actual_gradients.layers[2].weight,
            actual_gradients.layers[2].bias,
            actual_state_gradient,
        )
    )
    print(f"SYCL_VJP_COMPILE_EXECUTE_SECONDS={time.perf_counter() - start}")
    _assert_close("LOSS", actual_loss, expected_loss)
    _assert_close(
        "HIDDEN_WEIGHT_GRADIENT",
        actual_gradients.layers[0].weight,
        expected_gradients.layers[0].weight,
    )
    _assert_close(
        "OUTPUT_WEIGHT_GRADIENT",
        actual_gradients.layers[2].weight,
        expected_gradients.layers[2].weight,
    )
    _assert_close(
        "OUTPUT_BIAS_GRADIENT",
        actual_gradients.layers[2].bias,
        expected_gradients.layers[2].bias,
    )
    _assert_close("STATE_GRADIENT", actual_state_gradient, expected_state_gradient)

    # Exercise the trainer-specialized route: the state is already batched
    # when it enters the custom VJP, so shared parameter gradients are reduced
    # inside one native backward call rather than materialized per example.
    direct_value_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(_direct_batched_joint_loss, has_aux=True)
    )
    print("PHASE=SYCL_DIRECT_BATCHED_BACKWARD", flush=True)
    start = time.perf_counter()
    (direct_loss, direct_outputs), (
        direct_gradients,
        direct_state_gradient,
    ) = direct_value_and_grad((sycl_model, states), keys)
    jax.block_until_ready(
        (
            direct_loss,
            direct_outputs,
            direct_gradients.layers[0].weight,
            direct_gradients.layers[2].weight,
            direct_gradients.layers[2].bias,
            direct_state_gradient,
        )
    )
    print(
        "SYCL_DIRECT_BATCHED_VJP_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - start}"
    )
    _assert_close("DIRECT_BATCHED_FORWARD", direct_outputs, expected)
    _assert_close("DIRECT_BATCHED_LOSS", direct_loss, expected_loss)
    _assert_close(
        "DIRECT_BATCHED_HIDDEN_WEIGHT_GRADIENT",
        direct_gradients.layers[0].weight,
        expected_gradients.layers[0].weight,
    )
    _assert_close(
        "DIRECT_BATCHED_OUTPUT_WEIGHT_GRADIENT",
        direct_gradients.layers[2].weight,
        expected_gradients.layers[2].weight,
    )
    _assert_close(
        "DIRECT_BATCHED_OUTPUT_BIAS_GRADIENT",
        direct_gradients.layers[2].bias,
        expected_gradients.layers[2].bias,
    )
    _assert_close(
        "DIRECT_BATCHED_STATE_GRADIENT",
        direct_state_gradient,
        expected_state_gradient,
    )

    print("NCA_SYCL_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
