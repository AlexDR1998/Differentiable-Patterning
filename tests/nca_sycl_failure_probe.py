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

from NCA.model.NCA_sycl import NCA as SyclNCA
from NCA.trainer.sycl_shard_map import filter_shard_map


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        required=True,
        choices=("collective", "custom_call", "custom_call_collective"),
    )
    return parser.parse_args()


def main():
    args = _arguments()
    devices = [d for d in jax.local_devices() if d.platform == "sycl"]
    print("NCA_SYCL_FAILURE_PROBE_VERSION=2", flush=True)
    print("AUTODIFF=value_and_grad", flush=True)
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
            keys = jax.random.split(jax.random.PRNGKey(23), local_values.shape[0])
            local_values = candidate.batched_call(local_values, keys)
        result = jnp.mean(local_values**2)
        if args.probe != "custom_call":
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

    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))
    start = time.perf_counter()
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
    print(f"GRADIENT_NORM={float(gradient_norm)}", flush=True)
    print(f"ELAPSED_SECONDS={time.perf_counter() - start}", flush=True)
    print("NCA_SYCL_FAILURE_PROBE_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
