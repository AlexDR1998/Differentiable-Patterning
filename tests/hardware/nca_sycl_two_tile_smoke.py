#!/usr/bin/env python3
"""Verify tile-local rollout and gradients through the production shard map."""

from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from NCA.model.NCA_sycl import FUSED_REGULARISER_FLAGS, NCA as SyclNCA
from NCA.trainer.backend.sycl.scan import scan_carry_only
from NCA.trainer.backend.sycl.shard_map import filter_shard_map


CHANNELS = 32
HEIGHT = int(os.environ.get("NCA_SYCL_SMOKE_HEIGHT", "17"))
WIDTH = int(os.environ.get("NCA_SYCL_SMOKE_WIDTH", "19"))
INNER_BATCH = int(os.environ.get("NCA_SYCL_SMOKE_INNER_BATCH", "2"))
OUTER_BATCHES_PER_TILE = int(
    os.environ.get("NCA_SYCL_SMOKE_BATCHES_PER_TILE", "2")
)
STEPS = int(os.environ.get("NCA_SYCL_SMOKE_STEPS", "2"))
RTOL = 2.0e-2
ATOL = 2.0e-3
REGULARISER_FLAGS = sum(FUSED_REGULARISER_FLAGS.values())


def _boundary_mask(dtype):
    """Return one fixed model-boundary channel shaped ``[1,H,W]``."""
    y = jnp.arange(HEIGHT)[:, None]
    x = jnp.arange(WIDTH)[None, :]
    inside = (
        (y >= HEIGHT // 4)
        & (y < HEIGHT - HEIGHT // 4)
        & (x >= WIDTH // 4)
        & (x < WIDTH - WIDTH // 4)
    )
    return inside[None].astype(dtype)


def _loss(model, states, keys):
    """Return loss and rollout outputs for ``[N,C,H,W]`` tile-local states."""
    boundary_mask = _boundary_mask(states.dtype)

    def rollout_chunk(carry, chunk_keys):
        state, _, _ = carry
        final, trajectory, regularisers = model.batched_rollout(
            state,
            chunk_keys,
            boundary_code=1,
            boundary_mask=boundary_mask,
            boundary_channels=1,
            regulariser_flags=REGULARISER_FLAGS,
        )
        return (final, trajectory, regularisers), None

    final, trajectory, regularisers = scan_carry_only(
        rollout_chunk,
        (
            states,
            jnp.zeros((STEPS, *states.shape), dtype=states.dtype),
            jnp.zeros((2,), dtype=states.dtype),
        ),
        keys[None],
        kind="checkpointed",
    )
    loss = (
        jnp.mean(final**2)
        + 0.25 * jnp.mean(trajectory**2)
        + 0.13 * regularisers[0]
        + 0.19 * regularisers[1]
    )
    return loss, (final, trajectory, regularisers)


def _two_tile_loss_impl(model, state_shards, key_shards):
    """Evaluate separate ``[N,C,H,W]`` outer-B leaves on each tile."""
    local_losses = []
    final_states = []
    trajectories = []
    regularisers = []
    for state_shard, key_shard in zip(state_shards, key_shards):
        if state_shard.shape[0] != 1 or key_shard.shape[0] != 1:
            raise ValueError("Expected one physical tile axis per outer-B slot")
        local_loss, (final, trajectory, regulariser) = _loss(
            model, state_shard[0], key_shard[0]
        )
        local_losses.append(local_loss)
        final_states.append(final[None])
        trajectories.append(trajectory[None])
        regularisers.append(regulariser[None])
    mean_loss = jax.lax.pmean(jnp.mean(jnp.stack(local_losses)), "tiles")
    return mean_loss, (final_states, trajectories, regularisers)


def _single_tile_reference(model, states, keys):
    """Evaluate all ``[tile,outer-B,...]`` inputs without a device map."""
    def global_loss(candidate):
        losses = []
        final_states = []
        trajectories = []
        regularisers = []
        for tile in range(states.shape[0]):
            tile_finals = []
            tile_trajectories = []
            tile_regularisers = []
            for batch in range(states.shape[1]):
                loss, (final, trajectory, regulariser) = _loss(
                    candidate, states[tile, batch], keys[tile, batch]
                )
                losses.append(loss)
                tile_finals.append(final)
                tile_trajectories.append(trajectory)
                tile_regularisers.append(regulariser)
            final_states.append(jnp.stack(tile_finals))
            trajectories.append(jnp.stack(tile_trajectories))
            regularisers.append(jnp.stack(tile_regularisers))
        return jnp.mean(jnp.stack(losses)), (
            jnp.stack(final_states),
            jnp.stack(trajectories),
            jnp.stack(regularisers),
        )

    return eqx.filter_value_and_grad(global_loss, has_aux=True)(model)


def _assert_close(name, actual, expected):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    print(f"{name}_MAX_ABSOLUTE_ERROR={error}")
    if not np.allclose(
        np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL
    ):
        raise AssertionError(f"{name} mismatch: maximum error {error}")


def _gathered_loss(values):
    gathered = jax.lax.all_gather(values, "tiles", axis=0, tiled=False)
    gathered = gathered.reshape((-1, *values.shape[1:]))
    return jax.lax.pmean(jnp.mean(gathered**2), "tiles")


def main():
    os.environ.setdefault("NCA_SYCL_XMX_MODE", "bf16")
    devices = [
        device for device in jax.local_devices() if device.platform == "sycl"
    ]
    print("NCA_SYCL_TWO_TILE_SMOKE_VERSION=14")
    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAX_DEVICES={devices}")
    print(f"OUTER_BATCHES_PER_TILE={OUTER_BATCHES_PER_TILE}")
    print(f"TEST_SHAPE={INNER_BATCH}X{CHANNELS}X{HEIGHT}X{WIDTH}")
    print(f"FUSED_STEPS={STEPS}")
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
        (
            2,
            OUTER_BATCHES_PER_TILE,
            INNER_BATCH,
            CHANNELS,
            HEIGHT,
            WIDTH,
        ),
        dtype=jnp.float32,
    )
    keys = jax.random.split(
        rollout_key,
        2 * OUTER_BATCHES_PER_TILE * STEPS * INNER_BATCH,
    ).reshape(2, OUTER_BATCHES_PER_TILE, STEPS, INNER_BATCH, 2)

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
    mesh = Mesh(np.asarray(devices), ("tiles",))
    tile_sharding = NamedSharding(mesh, P("tiles"))
    gathered_loss = filter_shard_map(
        _gathered_loss,
        mesh=mesh,
        in_specs=(P("tiles"),),
        out_specs=P(),
        check_rep=False,
    )
    gather_values = jax.device_put(
        jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3), tile_sharding
    )
    gather_loss, gather_gradient = jax.jit(jax.value_and_grad(gathered_loss))(
        gather_values
    )
    _assert_close("ALL_GATHER_LOSS", gather_loss, jnp.mean(gather_values**2))
    _assert_close(
        "ALL_GATHER_GRADIENT",
        gather_gradient,
        2.0 * gather_values / gather_values.size,
    )
    sharded_states = [
        jax.device_put(states[:, batch], tile_sharding)
        for batch in range(OUTER_BATCHES_PER_TILE)
    ]
    sharded_keys = [
        jax.device_put(keys[:, batch], tile_sharding)
        for batch in range(OUTER_BATCHES_PER_TILE)
    ]
    two_tile_loss = filter_shard_map(
        _two_tile_loss_impl,
        mesh=mesh,
        in_specs=(P(), P("tiles"), P("tiles")),
        out_specs=(P(), (P("tiles"), P("tiles"), P("tiles"))),
        check_rep=False,
    )
    (mean_loss, outputs), gradients = eqx.filter_jit(
        eqx.filter_value_and_grad(two_tile_loss, has_aux=True)
    )(
        model, sharded_states, sharded_keys
    )
    updates = jtu.tree_map(
        lambda value: -0.01 * value if eqx.is_array(value) else value,
        gradients,
    )
    updated_model = eqx.apply_updates(model, updates)
    jax.block_until_ready((updated_model, mean_loss, outputs, gradients))
    print(
        "TWO_TILE_COMPILE_EXECUTE_SECONDS="
        f"{time.perf_counter() - two_tile_start}"
    )

    _assert_close("LOSS", mean_loss, reference_loss)
    final_states = jnp.stack(outputs[0], axis=1)
    trajectories = jnp.stack(outputs[1], axis=1)
    regularisers = jnp.stack(outputs[2], axis=1)
    _assert_close(
        "FINAL_STATE", final_states, reference_outputs[0]
    )
    _assert_close(
        "TRAJECTORY", trajectories, reference_outputs[1]
    )
    _assert_close(
        "FUSED_REGULARISERS", regularisers, reference_outputs[2]
    )
    _assert_close(
        "HIDDEN_WEIGHT_GRADIENT",
        gradients.layers[0].weight,
        reference_gradients.layers[0].weight,
    )
    _assert_close(
        "OUTPUT_WEIGHT_GRADIENT",
        gradients.layers[2].weight,
        reference_gradients.layers[2].weight,
    )
    _assert_close(
        "OUTPUT_BIAS_GRADIENT",
        gradients.layers[2].bias,
        reference_gradients.layers[2].bias,
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
        str(shard.device) for shard in outputs[0][0].addressable_shards
    )
    print(f"OUTPUT_SHARD_DEVICES={shard_devices}")
    if len(shard_devices) != 2:
        raise AssertionError("The NCA output was not sharded across both tiles")
    print("NCA_SYCL_TWO_TILE_SMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
