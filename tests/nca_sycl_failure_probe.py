#!/usr/bin/env python3
"""Minimal two-tile probes for Intel OpenXLA/SYCL crash isolation."""

from __future__ import annotations

import argparse
import os
import socket
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from NCA.model.NCA_sycl import FUSED_REGULARISER_FLAGS, NCA as SyclNCA
from NCA.trainer.sycl_shard_map import filter_shard_map


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        required=True,
        choices=(
            "collective",
            "fused_rollout_forward",
            "fused_rollout",
            "fused_rollout_serialized",
            "fused_rollout_collective",
            "fused_rollout_regulariser",
            "fused_rollout_regulariser_collective",
        ),
    )
    return parser.parse_args()


def main():
    args = _arguments()
    devices = [d for d in jax.local_devices() if d.platform == "sycl"]
    print("NCA_SYCL_FAILURE_PROBE_VERSION=5", flush=True)
    forward_only = args.probe == "fused_rollout_forward"
    serialized = args.probe == "fused_rollout_serialized"
    if serialized:
        os.environ["NCA_SYCL_SERIALIZE_CUSTOM_CALLS"] = "1"
    print(
        f"AUTODIFF={'forward_only' if forward_only else 'value_and_grad'}",
        flush=True,
    )
    print(f"SERIALIZE_CUSTOM_CALLS={int(serialized)}", flush=True)
    regulariser_enabled = "regulariser" in args.probe
    print(
        "CUSTOM_CALL_COMPLEXITY="
        + (
            "fused_rollout_with_intermediate_regulariser"
            if regulariser_enabled
            else "fused_rollout_without_regularisers"
        ),
        flush=True,
    )
    print("FUSED_STEPS=2", flush=True)
    print(f"PROBE={args.probe}", flush=True)
    print(f"HOSTNAME={socket.gethostname()}", flush=True)
    print(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID', '<unset>')}", flush=True)
    print(f"SLURM_ARRAY_TASK_ID={os.getenv('SLURM_ARRAY_TASK_ID', '<unset>')}", flush=True)
    print(f"JAX_VERSION={jax.__version__}", flush=True)
    print(f"JAX_DEVICES={devices}", flush=True)
    if len(devices) != 2:
        raise RuntimeError(f"Expected exactly two SYCL tiles, found {devices}")

    mesh = Mesh(np.asarray(devices), ("tiles",))
    sharding = NamedSharding(mesh, P("tiles"))
    values = jax.device_put(
        jnp.arange(2 * 2 * 32 * 17 * 19, dtype=jnp.float32).reshape(
            2, 2, 32, 17, 19
        ) / 1000.0,
        sharding,
    )

    model = None
    if args.probe != "collective":
        model = SyclNCA(
            32,
            KERNEL_STR=["ID", "LAP", "DIFF"],
            PADDING="CIRCULAR",
            FIRE_RATE=0.5,
            key=jax.random.PRNGKey(17),
        )

    def probe(candidate, local_values):
        local_values = local_values[0]
        if candidate is not None:
            keys = jax.random.split(
                jax.random.PRNGKey(23), 2 * local_values.shape[0]
            ).reshape(2, local_values.shape[0], 2)
            regulariser_flags = (
                FUSED_REGULARISER_FLAGS["intermediate_state"]
                if regulariser_enabled
                else 0
            )
            rollout = candidate.batched_rollout(
                local_values,
                keys,
                regulariser_flags=regulariser_flags,
            )
            if regulariser_enabled:
                final, trajectory, regularisers = rollout
            else:
                final, trajectory = rollout
            result = jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2)
            if regulariser_enabled:
                result = result + 0.13 * regularisers[0]
        else:
            result = jnp.mean(local_values**2)
        if args.probe == "collective" or args.probe.endswith("_collective"):
            result = jax.lax.pmean(result, "tiles")
        return result

    mapped = filter_shard_map(
        probe,
        mesh=mesh,
        in_specs=(P(), P("tiles")),
        out_specs=P(),
        check_rep=False,
    )

    def objective(arguments):
        return mapped(*arguments)

    start = time.perf_counter()
    if forward_only:
        result = eqx.filter_jit(mapped)(model, values)
        result.block_until_ready()
        gradient_leaves = []
        gradient_norm = None
    else:
        value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))
        result, gradients = value_and_grad((model, values))
        jax.block_until_ready((result, gradients))
        gradient_leaves = [
            leaf for leaf in jtu.tree_leaves(gradients) if eqx.is_array(leaf)
        ]
        gradient_norm = jnp.sqrt(
            sum(
                jnp.sum(jnp.asarray(leaf, dtype=jnp.float32) ** 2)
                for leaf in gradient_leaves
            )
        )
        gradient_norm.block_until_ready()
    print(f"RESULT={float(result)}", flush=True)
    print(f"GRADIENT_ARRAY_LEAVES={len(gradient_leaves)}", flush=True)
    print(
        "GRADIENT_NORM="
        + ("not_run" if gradient_norm is None else str(float(gradient_norm))),
        flush=True,
    )
    print(f"ELAPSED_SECONDS={time.perf_counter() - start}", flush=True)
    print("NCA_SYCL_FAILURE_PROBE_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
