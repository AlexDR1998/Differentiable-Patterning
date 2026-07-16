#!/usr/bin/env python3
"""End-to-end forward and gradient smoke test for the JAX/SYCL NCA."""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from NCA.model.NCA_model_fast import NCA as FastNCA
from NCA.model.NCA_sycl import NCA as SyclNCA


CHANNELS = 32
HEIGHT = 16
WIDTH = 16
BATCH = 2
KERNEL_STR = ["ID", "LAP", "DIFF"]
FIRE_RATE = 0.5
RTOL = 2.0e-4
ATOL = 2.0e-5


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


def _make_models(key):
    fast_model = FastNCA(
        CHANNELS,
        KERNEL_STR=KERNEL_STR,
        PADDING="CIRCULAR",
        FIRE_RATE=FIRE_RATE,
        key=key,
    )
    sycl_model = SyclNCA(
        CHANNELS,
        KERNEL_STR=KERNEL_STR,
        PADDING="CIRCULAR",
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
    print("NCA_SYCL_SMOKE_VERSION=1")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEFAULT_BACKEND={jax.default_backend()}")
    print(f"JAX_DEVICES={jax.devices()}")
    print("BACKWARD_IMPLEMENTATION=SYCL_CUSTOM_CALL")
    print("BACKWARD_BATCHING=SINGLE_CUSTOM_CALL")
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

    start = time.perf_counter()
    expected = fast_forward(fast_model, states, keys)
    expected.block_until_ready()
    print(f"JAX_REFERENCE_FORWARD_COMPILE_EXECUTE_SECONDS={time.perf_counter() - start}")

    start = time.perf_counter()
    actual = sycl_forward(sycl_model, states, keys)
    actual.block_until_ready()
    print(f"SYCL_FORWARD_COMPILE_EXECUTE_SECONDS={time.perf_counter() - start}")
    _assert_close("FORWARD", actual, expected)
    print(f"OUTPUT_DEVICE={actual.device}")

    value_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(_joint_loss, has_aux=True)
    )
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

    print("NCA_SYCL_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
