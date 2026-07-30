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


PROBE_ENVIRONMENTS = {
    "baseline": {"NCA_SYCL_XMX_MODE": "standard"},
    "strict_stages": {
        "NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION": "1",
        "NCA_SYCL_XMX_MODE": "standard",
    },
    "serialize_onemkl": {
        "NCA_SYCL_SERIALIZE_ONEMKL": "1",
        "NCA_SYCL_XMX_MODE": "standard",
    },
    "serialize_backward": {
        "NCA_SYCL_SERIALIZE_BACKWARD_CUSTOM_CALLS": "1",
        "NCA_SYCL_XMX_MODE": "standard",
    },
    "bf16_compute": {"NCA_SYCL_XMX_MODE": "bf16"},
}


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        required=True,
        choices=tuple(PROBE_ENVIRONMENTS),
    )
    return parser.parse_args()


def main():
    args = _arguments()
    os.environ["NCA_SYCL_REPORT_QUEUE_ORDERING"] = "1"
    for name in (
        "NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION",
        "NCA_SYCL_SERIALIZE_ONEMKL",
        "NCA_SYCL_SERIALIZE_BACKWARD_CUSTOM_CALLS",
        "NCA_SYCL_SERIALIZE_CUSTOM_CALLS",
        "NCA_SYCL_XMX_MODE",
    ):
        os.environ.pop(name, None)
    os.environ.update(PROBE_ENVIRONMENTS[args.probe])

    devices = [d for d in jax.local_devices() if d.platform == "sycl"]
    print("NCA_SYCL_FAILURE_PROBE_VERSION=6", flush=True)
    print("AUTODIFF=value_and_grad", flush=True)
    print("CUSTOM_CALL_COMPLEXITY=fused_rollout_without_regularisers", flush=True)
    print("FUSED_STEPS=2", flush=True)
    print(f"PROBE={args.probe}", flush=True)
    for name in (
        "NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION",
        "NCA_SYCL_SERIALIZE_ONEMKL",
        "NCA_SYCL_SERIALIZE_BACKWARD_CUSTOM_CALLS",
        "NCA_SYCL_SERIALIZE_CUSTOM_CALLS",
        "NCA_SYCL_XMX_MODE",
    ):
        print(f"{name}={os.getenv(name, '<unset>')}", flush=True)
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

    model = SyclNCA(
        32,
        KERNEL_STR=["ID", "LAP", "DIFF"],
        PADDING="CIRCULAR",
        FIRE_RATE=0.5,
        key=jax.random.PRNGKey(17),
    )

    def probe(candidate, local_values):
        local_values = local_values[0]
        keys = jax.random.split(
            jax.random.PRNGKey(23), 2 * local_values.shape[0]
        ).reshape(2, local_values.shape[0], 2)
        final, trajectory = candidate.batched_rollout(local_values, keys)
        return jnp.mean(final**2) + 0.25 * jnp.mean(trajectory**2)

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
    print(f"GRADIENT_NORM={float(gradient_norm)}", flush=True)
    print(f"ELAPSED_SECONDS={time.perf_counter() - start}", flush=True)
    print("NCA_SYCL_FAILURE_PROBE_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
