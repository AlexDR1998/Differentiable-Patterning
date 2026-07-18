#!/usr/bin/env python3
"""Verify two-tile NCA execution and replicated gradient reduction."""

from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from NCA.model.NCA_sycl import NCA as SyclNCA
from NCA.trainer.sycl_filter_pmap import filter_pmap_no_probe
from NCA.trainer.sycl_scan import scan_carry_only


CHANNELS = 32
HEIGHT = int(os.environ.get("NCA_SYCL_SMOKE_HEIGHT", "17"))
WIDTH = int(os.environ.get("NCA_SYCL_SMOKE_WIDTH", "19"))
INNER_BATCH = 2
STEPS = 2
RTOL = 2.0e-2
ATOL = 2.0e-3


def _loss(model, states, keys):
    def rollout_chunk(carry, chunk_keys):
        state, _ = carry
        final, trajectory = model.batched_rollout(state, chunk_keys)
        return (final, trajectory), None

    final, trajectory = scan_carry_only(
        rollout_chunk,
        (states, jnp.zeros((STEPS, *states.shape), dtype=states.dtype)),
        keys[None],
        kind="checkpointed",
    )
    loss = jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2)
    return loss, (final, trajectory)


def _mean_array_tree(tree, axis_name):
    return jtu.tree_map(
        lambda value: jax.lax.pmean(value, axis_name)
        if eqx.is_array(value)
        else value,
        tree,
    )


def _two_tile_value_and_grad_impl(model, states, keys):
    (local_loss, outputs), gradients = eqx.filter_value_and_grad(
        _loss, has_aux=True
    )(model, states, keys)
    mean_loss = jax.lax.pmean(local_loss, "tiles")
    mean_gradients = _mean_array_tree(gradients, "tiles")
    updates = jtu.tree_map(
        lambda value: -0.01 * value if eqx.is_array(value) else value,
        mean_gradients,
    )
    updated_model = eqx.apply_updates(model, updates)
    return updated_model, mean_loss, outputs, mean_gradients


_two_tile_value_and_grad = filter_pmap_no_probe(
    _two_tile_value_and_grad_impl,
    axis_name="tiles",
    in_axes=(None, 0, 0),
    out_axes=(None, None, 0, 0),
)


def _single_tile_reference(model, states, keys):
    def global_loss(candidate):
        losses = []
        outputs = []
        for tile in range(states.shape[0]):
            loss, output = _loss(candidate, states[tile], keys[tile])
            losses.append(loss)
            outputs.append(output)
        return jnp.mean(jnp.stack(losses)), outputs

    return eqx.filter_value_and_grad(global_loss, has_aux=True)(model)


def _assert_close(name, actual, expected):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    print(f"{name}_MAX_ABSOLUTE_ERROR={error}")
    if not np.allclose(
        np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL
    ):
        raise AssertionError(f"{name} mismatch: maximum error {error}")


def main():
    os.environ.setdefault("NCA_SYCL_XMX_MODE", "bf16")
    devices = [
        device for device in jax.local_devices() if device.platform == "sycl"
    ]
    print("NCA_SYCL_TWO_TILE_SMOKE_VERSION=6")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEVICES={devices}")
    if len(devices) != 2:
        raise RuntimeError(f"Expected exactly two SYCL tiles, found {devices}")

    root_key = jax.random.PRNGKey(20260717)
    model_key, state_key, rollout_key = jax.random.split(root_key, 3)
    model = SyclNCA(
        CHANNELS,
        KERNEL_STR=["ID", "LAP", "DIFF"],
        PADDING="CIRCULAR",
        FIRE_RATE=0.5,
        key=model_key,
    )
    weight_keys = jax.random.split(jax.random.fold_in(model_key, 1), 3)
    model.set_weights(
        (
            0.02 * jax.random.normal(weight_keys[0], model.layers[0].weight.shape),
            0.02 * jax.random.normal(weight_keys[1], model.layers[2].weight.shape),
            0.02 * jax.random.normal(weight_keys[2], model.layers[2].bias.shape),
        )
    )
    states = jax.random.normal(
        state_key,
        (2, INNER_BATCH, CHANNELS, HEIGHT, WIDTH),
        dtype=jnp.float32,
    )
    keys = jax.random.split(rollout_key, 2 * STEPS * INNER_BATCH).reshape(
        2, STEPS, INNER_BATCH, 2
    )

    # A collective-only check gives a much clearer error if oneCCL support is
    # unavailable than encountering it after compiling the complete NCA VJP.
    collective = jax.pmap(
        lambda value: jax.lax.pmean(value, "tiles"), axis_name="tiles"
    )
    collective_result = collective(jnp.asarray([1.0, 3.0], jnp.float32))
    collective_result.block_until_ready()
    _assert_close("COLLECTIVE_PMEAN", collective_result, jnp.asarray([2.0, 2.0]))
    print("TWO_TILE_COLLECTIVE_RESULT=PASS")

    print("PHASE=SINGLE_TILE_REFERENCE", flush=True)
    reference_start = time.perf_counter()
    (reference_loss, reference_outputs), reference_gradients = eqx.filter_jit(
        _single_tile_reference
    )(model, states, keys)
    jax.block_until_ready((reference_loss, reference_outputs, reference_gradients))
    print(
        "SINGLE_TILE_REFERENCE_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - reference_start}"
    )

    print("PHASE=TWO_TILE_FORWARD_BACKWARD", flush=True)
    two_tile_start = time.perf_counter()
    sharded_states = jax.device_put_sharded(
        [states[tile] for tile in range(2)], devices
    )
    sharded_keys = jax.device_put_sharded(
        [keys[tile] for tile in range(2)], devices
    )
    updated_model, mean_loss, outputs, gradients = _two_tile_value_and_grad(
        model, sharded_states, sharded_keys
    )
    jax.block_until_ready((updated_model, mean_loss, outputs, gradients))
    print(
        "TWO_TILE_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - two_tile_start}"
    )

    _assert_close("LOSS", mean_loss, reference_loss)
    _assert_close(
        "FINAL_STATE", outputs[0], jnp.stack([value[0] for value in reference_outputs])
    )
    _assert_close(
        "TRAJECTORY", outputs[1], jnp.stack([value[1] for value in reference_outputs])
    )
    _assert_close(
        "HIDDEN_WEIGHT_GRADIENT",
        gradients.layers[0].weight,
        jnp.repeat(reference_gradients.layers[0].weight[None], 2, axis=0),
    )
    _assert_close(
        "OUTPUT_WEIGHT_GRADIENT",
        gradients.layers[2].weight,
        jnp.repeat(reference_gradients.layers[2].weight[None], 2, axis=0),
    )
    _assert_close(
        "OUTPUT_BIAS_GRADIENT",
        gradients.layers[2].bias,
        jnp.repeat(reference_gradients.layers[2].bias[None], 2, axis=0),
    )
    reference_updates = jtu.tree_map(
        lambda value: -0.01 * value if eqx.is_array(value) else value,
        reference_gradients,
    )
    reference_updated_model = eqx.apply_updates(model, reference_updates)
    _assert_close(
        "UPDATED_HIDDEN_WEIGHT",
        updated_model.layers[0].weight,
        reference_updated_model.layers[0].weight,
    )
    _assert_close(
        "UPDATED_OUTPUT_WEIGHT",
        updated_model.layers[2].weight,
        reference_updated_model.layers[2].weight,
    )

    shard_devices = sorted(
        str(shard.device) for shard in outputs[0].addressable_shards
    )
    print(f"OUTPUT_SHARD_DEVICES={shard_devices}")
    if len(shard_devices) != 2:
        raise AssertionError("The NCA output was not sharded across both tiles")
    print("NCA_SYCL_TWO_TILE_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
