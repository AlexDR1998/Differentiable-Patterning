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


def _assert_state_gradient_close(name: str, actual, expected) -> float:
    """Compare gradients while handling sqrt(0) in the JAX DIFF reference.

    Reflect padding makes both finite-difference components exactly zero at
    some boundary locations. JAX's derivative of sqrt(gx**2 + gy**2) is NaN
    there; the native VJP deliberately chooses the finite zero subgradient.
    """
    actual_host = np.asarray(actual)
    expected_host = np.asarray(expected)
    reference_nan = np.isnan(expected_host)
    print(f"{name}_REFERENCE_NAN_COUNT={int(np.sum(reference_nan))}")
    if np.any(~np.isfinite(actual_host)):
        raise AssertionError(f"{name} contains non-finite SYCL gradients")
    finite_reference = ~reference_nan
    if not np.any(finite_reference):
        raise AssertionError(f"{name} has no finite JAX reference entries")
    error = float(
        np.max(np.abs(actual_host[finite_reference] - expected_host[finite_reference]))
    )
    print(f"{name}_MAX_ABSOLUTE_ERROR={error}")
    if not np.allclose(
        actual_host[finite_reference],
        expected_host[finite_reference],
        rtol=RTOL,
        atol=ATOL,
    ):
        raise AssertionError(
            f"{name} differs from finite JAX reference entries: max error {error}"
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


def _reference_rollout(model, states, keys):
    trajectory = []
    for step in range(keys.shape[0]):
        states = _batched_outputs(model, states, keys[step])
        trajectory.append(states)
    return states, jnp.stack(trajectory)


def _reference_rollout_loss(model_and_states, keys):
    model, states = model_and_states
    final, trajectory = _reference_rollout(model, states, keys)
    return jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2), (
        final,
        trajectory,
    )


def _sycl_rollout_loss(model_and_states, keys):
    model, states = model_and_states
    final, trajectory = model.batched_rollout(states, keys)
    return jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2), (
        final,
        trajectory,
    )


def _reference_fixed_boundary_rollout_loss(
    model_and_states, keys, boundary_mask
):
    model, states = model_and_states
    trajectory = []
    boundary_channels = boundary_mask.shape[0]
    for step in range(keys.shape[0]):
        states = _batched_outputs(model, states, keys[step])
        states = states.at[:, -boundary_channels:].set(boundary_mask)
        trajectory.append(states)
    trajectory = jnp.stack(trajectory)
    return jnp.mean(states**2) + 0.25 * jnp.mean(trajectory**2), (
        states,
        trajectory,
    )


def _sycl_fixed_boundary_rollout_loss(model_and_states, keys, boundary_mask):
    model, states = model_and_states
    final, trajectory = model.batched_rollout(
        states,
        keys,
        boundary_code=1,
        boundary_mask=boundary_mask,
        boundary_channels=boundary_mask.shape[0],
    )
    return jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2), (
        final,
        trajectory,
    )


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
    print("NCA_SYCL_SMOKE_VERSION=4")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEFAULT_BACKEND={jax.default_backend()}")
    print(f"JAX_DEVICES={jax.devices()}")
    print("BACKWARD_IMPLEMENTATION=SYCL_CUSTOM_CALL")
    print("BACKWARD_BATCHING=SINGLE_CUSTOM_CALL")
    print("TRAINER_BATCHING=SEPARATE_B_DIRECT_N_SHARED_GRADIENTS")
    print("TRAINER_ROLLOUT=2_TIMESTEPS_PER_CUSTOM_CALL")
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
    _assert_state_gradient_close(
        "STATE_GRADIENT", actual_state_gradient, expected_state_gradient
    )

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
    _assert_state_gradient_close(
        "DIRECT_BATCHED_STATE_GRADIENT",
        direct_state_gradient,
        expected_state_gradient,
    )

    rollout_keys = jax.random.split(jax.random.fold_in(rollout_key, 99), 2 * BATCH)
    rollout_keys = rollout_keys.reshape(2, BATCH, 2)
    reference_rollout_vg = eqx.filter_jit(
        eqx.filter_value_and_grad(_reference_rollout_loss, has_aux=True)
    )
    sycl_rollout_vg = eqx.filter_jit(
        eqx.filter_value_and_grad(_sycl_rollout_loss, has_aux=True)
    )
    print("PHASE=SYCL_TWO_STEP_ROLLOUT", flush=True)
    start = time.perf_counter()
    (expected_rollout_loss, expected_rollout), (
        expected_rollout_gradients,
        expected_rollout_state_gradient,
    ) = reference_rollout_vg((fast_model, states), rollout_keys)
    (actual_rollout_loss, actual_rollout), (
        actual_rollout_gradients,
        actual_rollout_state_gradient,
    ) = sycl_rollout_vg((sycl_model, states), rollout_keys)
    jax.block_until_ready(
        (
            actual_rollout_loss,
            actual_rollout,
            actual_rollout_gradients.layers[0].weight,
            actual_rollout_gradients.layers[2].weight,
            actual_rollout_gradients.layers[2].bias,
            actual_rollout_state_gradient,
        )
    )
    print(
        "SYCL_TWO_STEP_ROLLOUT_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - start}"
    )
    _assert_close("ROLLOUT_FINAL", actual_rollout[0], expected_rollout[0])
    _assert_close(
        "ROLLOUT_TRAJECTORY", actual_rollout[1], expected_rollout[1]
    )
    _assert_close(
        "ROLLOUT_LOSS", actual_rollout_loss, expected_rollout_loss
    )
    _assert_close(
        "ROLLOUT_HIDDEN_WEIGHT_GRADIENT",
        actual_rollout_gradients.layers[0].weight,
        expected_rollout_gradients.layers[0].weight,
    )
    _assert_close(
        "ROLLOUT_OUTPUT_WEIGHT_GRADIENT",
        actual_rollout_gradients.layers[2].weight,
        expected_rollout_gradients.layers[2].weight,
    )
    _assert_close(
        "ROLLOUT_OUTPUT_BIAS_GRADIENT",
        actual_rollout_gradients.layers[2].bias,
        expected_rollout_gradients.layers[2].bias,
    )
    _assert_state_gradient_close(
        "ROLLOUT_STATE_GRADIENT",
        actual_rollout_state_gradient,
        expected_rollout_state_gradient,
    )

    # The trainer's default soft-boundary mode fixes the final mask channels
    # after every NCA step. Verify both the saved intermediate states and the
    # reverse rule, including the zero derivative through overwritten values.
    boundary_mask = jax.random.normal(
        jax.random.fold_in(rollout_key, 100),
        (2, HEIGHT, WIDTH),
        dtype=jnp.float32,
    )
    reference_boundary_vg = eqx.filter_jit(
        eqx.filter_value_and_grad(
            _reference_fixed_boundary_rollout_loss, has_aux=True
        )
    )
    sycl_boundary_vg = eqx.filter_jit(
        eqx.filter_value_and_grad(
            _sycl_fixed_boundary_rollout_loss, has_aux=True
        )
    )
    print("PHASE=SYCL_TWO_STEP_FIXED_BOUNDARY_ROLLOUT", flush=True)
    (expected_boundary_loss, expected_boundary_rollout), (
        expected_boundary_gradients,
        expected_boundary_state_gradient,
    ) = reference_boundary_vg((fast_model, states), rollout_keys, boundary_mask)
    (actual_boundary_loss, actual_boundary_rollout), (
        actual_boundary_gradients,
        actual_boundary_state_gradient,
    ) = sycl_boundary_vg((sycl_model, states), rollout_keys, boundary_mask)
    jax.block_until_ready(
        (
            actual_boundary_loss,
            actual_boundary_rollout,
            actual_boundary_gradients.layers[0].weight,
            actual_boundary_gradients.layers[2].weight,
            actual_boundary_gradients.layers[2].bias,
            actual_boundary_state_gradient,
        )
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_FINAL",
        actual_boundary_rollout[0],
        expected_boundary_rollout[0],
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_TRAJECTORY",
        actual_boundary_rollout[1],
        expected_boundary_rollout[1],
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_LOSS", actual_boundary_loss, expected_boundary_loss
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_HIDDEN_WEIGHT_GRADIENT",
        actual_boundary_gradients.layers[0].weight,
        expected_boundary_gradients.layers[0].weight,
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_OUTPUT_WEIGHT_GRADIENT",
        actual_boundary_gradients.layers[2].weight,
        expected_boundary_gradients.layers[2].weight,
    )
    _assert_close(
        "BOUNDARY_ROLLOUT_OUTPUT_BIAS_GRADIENT",
        actual_boundary_gradients.layers[2].bias,
        expected_boundary_gradients.layers[2].bias,
    )
    _assert_state_gradient_close(
        "BOUNDARY_ROLLOUT_STATE_GRADIENT",
        actual_boundary_state_gradient,
        expected_boundary_state_gradient,
    )

    print("NCA_SYCL_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
