#!/usr/bin/env python3
"""Validate tile-local NCA training through explicit shard_map SPMD."""

from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax.experimental.shard_map import shard_map

    SHARD_MAP_API = "jax.experimental.shard_map"
except ImportError:
    shard_map = jax.shard_map
    SHARD_MAP_API = "jax.shard_map"

from NCA.model.NCA_sycl import NCA as SyclNCA
from NCA.trainer.sycl_scan import scan_carry_only


CHANNELS = 32
INNER_BATCH = 4
HEIGHT = int(os.environ.get("NCA_SYCL_SHARD_MAP_HEIGHT", "17"))
WIDTH = int(os.environ.get("NCA_SYCL_SHARD_MAP_WIDTH", "19"))
TOTAL_STEPS = int(os.environ.get("NCA_SYCL_SHARD_MAP_STEPS", "8"))
TIMING_REPEATS = int(os.environ.get("NCA_SYCL_SHARD_MAP_TIMING_REPEATS", "5"))
RTOL = 2.0e-2
ATOL = 2.0e-3


def _csv_values(name, default, convert=str):
    raw = os.environ.get(name, default)
    return [convert(value.strip()) for value in raw.split(",") if value.strip()]


def _mean_array_tree(tree, axis_name):
    return jtu.tree_map(
        lambda value: jax.lax.pmean(value, axis_name)
        if eqx.is_array(value)
        else value,
        tree,
    )


def _local_loss(model, state, chunk_keys, loop_kind):
    """Run only tile-local [N,C,H,W] states through fused rollout chunks."""

    def chunk(carry, keys):
        current, trajectory_energy = carry
        final, trajectory = model.batched_rollout(current, keys)
        trajectory_energy = trajectory_energy + jnp.mean(trajectory**2)
        return (final, trajectory_energy), None

    final, trajectory_energy = scan_carry_only(
        chunk,
        (state, jnp.asarray(0.0, dtype=state.dtype)),
        chunk_keys,
        kind=loop_kind,
    )
    loss = jnp.mean(final**2) + 0.25 * trajectory_energy / chunk_keys.shape[0]
    return loss, final


def _make_sharded_step(model_static, mesh, loop_kind):
    def tile_step(model_dynamic, state_shard, key_shard):
        # shard_map retains a size-one local shard dimension. Remove exactly
        # that explicit data-parallel axis before entering the NCA custom call.
        if state_shard.shape[0] != 1 or key_shard.shape[0] != 1:
            raise ValueError(
                "Expected one outer-B leaf per tile, got "
                f"state={state_shard.shape}, keys={key_shard.shape}"
            )
        local_state = state_shard[0]
        local_keys = key_shard[0]
        if local_state.ndim != 4 or local_keys.ndim != 4:
            raise ValueError(
                "SYCL custom-call inputs must be tile-local: "
                f"state={local_state.shape}, keys={local_keys.shape}"
            )

        def loss_function(candidate_dynamic):
            candidate = eqx.combine(candidate_dynamic, model_static)
            return _local_loss(candidate, local_state, local_keys, loop_kind)

        (local_loss, final), gradients = eqx.filter_value_and_grad(
            loss_function, has_aux=True
        )(model_dynamic)
        gradients = _mean_array_tree(gradients, "tile")
        mean_loss = jax.lax.pmean(local_loss, "tile")
        return mean_loss, final[None], gradients

    return shard_map(
        tile_step,
        mesh=mesh,
        in_specs=(P(), P("tile"), P("tile")),
        out_specs=(P(), P("tile"), P()),
        check_rep=False,
    )


def _reference_value_and_grad(model_dynamic, model_static, states, keys, kind):
    def global_loss(candidate_dynamic):
        candidate = eqx.combine(candidate_dynamic, model_static)
        losses = []
        finals = []
        for tile in range(2):
            loss, final = _local_loss(
                candidate, states[tile], keys[tile], kind
            )
            losses.append(loss)
            finals.append(final)
        return jnp.mean(jnp.stack(losses)), jnp.stack(finals)

    return eqx.filter_value_and_grad(global_loss, has_aux=True)(model_dynamic)


def _maximum_error(actual, expected):
    return float(
        np.max(np.abs(np.asarray(jax.device_get(actual)) - np.asarray(jax.device_get(expected))))
    )


def _assert_close(name, actual, expected):
    error = _maximum_error(actual, expected)
    print(f"{name}_MAX_ABSOLUTE_ERROR={error}")
    if not np.allclose(
        np.asarray(jax.device_get(actual)),
        np.asarray(jax.device_get(expected)),
        rtol=RTOL,
        atol=ATOL,
    ):
        raise AssertionError(f"{name} mismatch: maximum error {error}")


def main():
    os.environ.setdefault("NCA_SYCL_XMX_MODE", "bf16")
    devices = [device for device in jax.local_devices() if device.platform == "sycl"]
    print("NCA_SYCL_SHARD_MAP_SMOKE_VERSION=1")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"SHARD_MAP_API={SHARD_MAP_API}")
    print(f"JAX_DEVICES={devices}")
    if len(devices) != 2:
        raise RuntimeError(f"Expected exactly two SYCL tiles, found {devices}")

    mesh = Mesh(np.asarray(devices), ("tile",))
    replicated = NamedSharding(mesh, P())
    tile_sharding = NamedSharding(mesh, P("tile"))

    root_key = jax.random.PRNGKey(20260718)
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
    model_dynamic, model_static = model.partition()
    replicated_model_dynamic = jtu.tree_map(
        lambda value: jax.device_put(value, replicated)
        if eqx.is_array(value)
        else value,
        model_dynamic,
    )

    states = jax.random.normal(
        state_key,
        (2, INNER_BATCH, CHANNELS, HEIGHT, WIDTH),
        dtype=jnp.float32,
    )
    fusion_steps = _csv_values("NCA_SYCL_SHARD_MAP_FUSIONS", "1,2,4", int)
    loop_kinds = _csv_values(
        "NCA_SYCL_SHARD_MAP_LOOPS", "lax,checkpointed"
    )

    for fusion in fusion_steps:
        if fusion < 1 or TOTAL_STEPS % fusion != 0:
            raise ValueError(
                f"Fusion length {fusion} must divide TOTAL_STEPS={TOTAL_STEPS}"
            )
        chunks = TOTAL_STEPS // fusion
        keys = jax.random.split(
            jax.random.fold_in(rollout_key, fusion),
            2 * chunks * fusion * INNER_BATCH,
        ).reshape(2, chunks, fusion, INNER_BATCH, 2)

        sharded_states = jax.device_put(states, tile_sharding)
        sharded_keys = jax.device_put(keys, tile_sharding)

        for loop_kind in loop_kinds:
            label = f"FUSION_{fusion}_{loop_kind.upper()}"
            print(f"PHASE={label}", flush=True)

            reference_function = eqx.filter_jit(_reference_value_and_grad)
            reference_start = time.perf_counter()
            (reference_loss, reference_final), reference_gradients = reference_function(
                model_dynamic,
                model_static,
                states,
                keys,
                loop_kind,
            )
            jax.block_until_ready(
                (reference_loss, reference_final, reference_gradients)
            )
            print(
                f"{label}_REFERENCE_COMPILE_EXECUTE_SECONDS="
                f"{time.perf_counter() - reference_start}"
            )

            sharded_step = jax.jit(
                _make_sharded_step(model_static, mesh, loop_kind)
            )
            sharded_start = time.perf_counter()
            mean_loss, final, gradients = sharded_step(
                replicated_model_dynamic, sharded_states, sharded_keys
            )
            jax.block_until_ready((mean_loss, final, gradients))
            print(
                f"{label}_SHARDED_COMPILE_EXECUTE_SECONDS="
                f"{time.perf_counter() - sharded_start}"
            )

            _assert_close(f"{label}_LOSS", mean_loss, reference_loss)
            _assert_close(f"{label}_FINAL", final, reference_final)
            _assert_close(
                f"{label}_HIDDEN_WEIGHT_GRADIENT",
                gradients.layers[0].weight,
                reference_gradients.layers[0].weight,
            )
            _assert_close(
                f"{label}_OUTPUT_WEIGHT_GRADIENT",
                gradients.layers[2].weight,
                reference_gradients.layers[2].weight,
            )
            _assert_close(
                f"{label}_OUTPUT_BIAS_GRADIENT",
                gradients.layers[2].bias,
                reference_gradients.layers[2].bias,
            )

            shard_devices = sorted(
                str(shard.device) for shard in final.addressable_shards
            )
            print(f"{label}_OUTPUT_SHARD_DEVICES={shard_devices}")
            if shard_devices != sorted(str(device) for device in devices):
                raise AssertionError(
                    f"Output was not distributed over both tiles: {shard_devices}"
                )

            reference_steady_start = time.perf_counter()
            for _ in range(TIMING_REPEATS):
                reference_outputs = reference_function(
                    model_dynamic,
                    model_static,
                    states,
                    keys,
                    loop_kind,
                )
            jax.block_until_ready(reference_outputs)
            reference_steady = (
                time.perf_counter() - reference_steady_start
            ) / TIMING_REPEATS

            sharded_steady_start = time.perf_counter()
            for _ in range(TIMING_REPEATS):
                sharded_outputs = sharded_step(
                    replicated_model_dynamic, sharded_states, sharded_keys
                )
            jax.block_until_ready(sharded_outputs)
            sharded_steady = (
                time.perf_counter() - sharded_steady_start
            ) / TIMING_REPEATS
            print(f"{label}_REFERENCE_STEADY_SECONDS={reference_steady}")
            print(f"{label}_SHARDED_STEADY_SECONDS={sharded_steady}")
            print(
                f"{label}_TWO_TILE_SPEEDUP="
                f"{reference_steady / max(sharded_steady, 1e-12)}"
            )

    print("NCA_SYCL_SHARD_MAP_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
