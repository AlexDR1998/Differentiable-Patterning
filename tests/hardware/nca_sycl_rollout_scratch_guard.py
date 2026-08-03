#!/usr/bin/env python3
"""Two-step fused-backward scratch corruption probe for Intel SYCL.

Run this script in separate processes with ``--mode reuse`` and
``--mode per_step``. The former preserves the production scratch reuse while
surrounding every allocation with canaries. The latter gives every reverse
step its own guarded slot, isolating reuse/lifetime failures from kernel OOBs.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from NCA.model.NCA_sycl import NCA as SyclNCA


STEPS = 2
BATCH = 2
CHANNELS = 32
HEIGHT = 17
WIDTH = 19
GUARD_FLOATS = 64
GUARD_VALUE = np.float32(-1234567.0)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("reuse", "per_step"))
    parser.add_argument("--tiles", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def _check_guarded_workspace(name, value, payload_elements, slots):
    host = np.asarray(value).reshape(-1)
    stride = payload_elements + 2 * GUARD_FLOATS
    expected_elements = slots * stride
    if host.size != expected_elements:
        raise AssertionError(
            f"{name} has {host.size} elements; expected {expected_elements}"
        )
    corruptions = []
    for slot in range(slots):
        start = slot * stride
        prefix = host[start : start + GUARD_FLOATS]
        suffix_start = start + GUARD_FLOATS + payload_elements
        suffix = host[suffix_start : suffix_start + GUARD_FLOATS]
        bad_prefix = np.flatnonzero(prefix != GUARD_VALUE)
        bad_suffix = np.flatnonzero(suffix != GUARD_VALUE)
        if bad_prefix.size or bad_suffix.size:
            corruptions.append(
                (slot, bad_prefix.tolist(), bad_suffix.tolist())
            )
    print(f"{name}_GUARD_CORRUPTIONS={corruptions}", flush=True)
    if corruptions:
        raise AssertionError(f"{name} scratch canary was overwritten")


def main():
    args = _arguments()
    os.environ["NCA_SYCL_DIAGNOSTIC_SCRATCH"] = args.mode
    print(f"NCA_SYCL_DIAGNOSTIC_SCRATCH={args.mode}", flush=True)

    # Import after selecting the mode: it controls the diagnostic result
    # shapes at JAX trace time as well as native pointer offsets at execution.
    from NCA.model.sycl.bridge import (
        _bind_native_backward,
        _bind_rollout_forward,
        sycl_nca_rollout_backward_diagnostic,
    )

    devices = [device for device in jax.local_devices() if device.platform == "sycl"]
    if len(devices) < args.tiles:
        raise RuntimeError(
            f"The scratch probe requested {args.tiles} SYCL tiles, found {devices}"
        )
    devices = devices[: args.tiles]
    print(f"NCA_SYCL_SCRATCH_GUARD_DEVICES={devices}", flush=True)

    model = SyclNCA(
        CHANNELS,
        KERNEL_STR=["ID", "LAP", "DIFF"],
        PADDING="CIRCULAR",
        FIRE_RATE=0.5,
        key=jax.random.PRNGKey(17),
    )
    flags, padding = model._validate_sycl_configuration(
        jnp.zeros((BATCH, CHANNELS, HEIGHT, WIDTH), dtype=jnp.float32)
    )
    kernels, weight_hidden, weight_output, bias_output = model._sycl_parameters()
    base_state = (
        jnp.arange(BATCH * CHANNELS * HEIGHT * WIDTH, dtype=jnp.float32)
        .reshape(BATCH, CHANNELS, HEIGHT, WIDTH)
        / 10000.0
    )
    state = jnp.stack(
        tuple(base_state + tile * 0.001 for tile in range(args.tiles))
    )
    keys = jax.random.split(
        jax.random.PRNGKey(23), args.tiles * STEPS * BATCH
    ).reshape(
        args.tiles, STEPS, BATCH, 2
    )
    masks = jax.vmap(
        lambda tile_keys: jax.vmap(
            lambda step_keys: jax.vmap(
                lambda key: jax.random.bernoulli(
                    key, p=0.5, shape=(CHANNELS, HEIGHT, WIDTH)
                )
            )(step_keys)
        )(tile_keys)
    )(keys).astype(jnp.float32)
    boundary_mask = jnp.zeros((1,), dtype=jnp.float32)
    output_cotangent = jnp.sin(state * 0.013).astype(jnp.float32)
    trajectory_cotangent = jnp.stack(
        (jnp.cos(state * 0.017), jnp.sin(state * 0.019)), axis=1
    ).astype(jnp.float32)

    def diagnostic_probe(local_state, local_masks, local_output_cotangent,
                         local_trajectory_cotangent):
        return sycl_nca_rollout_backward_diagnostic(
            local_state,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            local_masks,
            boundary_mask,
            local_output_cotangent,
            local_trajectory_cotangent,
            kernel_flags=flags,
            padding=padding,
            boundary_code=0,
            boundary_channels=0,
        )

    def explicit_probe(local_state, local_masks, local_output_cotangent,
                       local_trajectory_cotangent):
        _, trajectory, _ = _bind_rollout_forward(
            local_state,
            kernels,
            weight_hidden,
            weight_output,
            bias_output,
            local_masks,
            boundary_mask,
            flags,
            padding,
            0,
            0,
            0,
        )
        current = local_output_cotangent
        accumulated = (
            jnp.zeros_like(weight_hidden),
            jnp.zeros_like(weight_output),
            jnp.zeros_like(bias_output),
        )
        for step in range(STEPS - 1, -1, -1):
            step_state = local_state if step == 0 else trajectory[step - 1]
            current, step_dw0, step_dw1, step_db = _bind_native_backward(
                step_state,
                kernels,
                weight_hidden,
                weight_output,
                bias_output,
                local_masks[step],
                current + local_trajectory_cotangent[step],
                flags,
                padding,
            )
            accumulated = tuple(
                total + update for total, update in zip(
                    accumulated, (step_dw0, step_dw1, step_db)
                )
            )
        return current, *accumulated

    # Keep the oracle in a separate dispatch. This prevents an independent
    # custom call from overlapping the guarded fused call and muddying a
    # corruption result.
    dispatch_diagnostic = jax.pmap(diagnostic_probe, devices=devices)
    dispatch_explicit = jax.pmap(explicit_probe, devices=devices)

    diagnostic = dispatch_diagnostic(
        state, masks, output_cotangent, trajectory_cotangent
    )
    jax.block_until_ready(diagnostic)
    explicit = dispatch_explicit(
        state, masks, output_cotangent, trajectory_cotangent
    )
    jax.block_until_ready(explicit)

    for name, actual, expected in zip(
        ("STATE", "HIDDEN_WEIGHT", "OUTPUT_WEIGHT", "BIAS"),
        diagnostic[:4],
        explicit,
    ):
        actual_host = np.asarray(actual)
        expected_host = np.asarray(expected)
        error = float(np.max(np.abs(actual_host - expected_host)))
        print(f"{name}_MAX_ABSOLUTE_ERROR={error}", flush=True)
        print(f"{name}_GRADIENT_SUM={float(np.sum(actual_host))}", flush=True)
        if not np.allclose(actual_host, expected_host, rtol=2e-4, atol=2e-5):
            raise AssertionError(
                f"Fused {name} gradient differs from two one-step backward calls"
            )

    features = weight_hidden.shape[0]
    state_elements = BATCH * CHANNELS * HEIGHT * WIDTH
    activation_elements = BATCH * features * HEIGHT * WIDTH
    slots = STEPS if args.mode == "per_step" else 1
    scratch_specs = (
        ("BOUNDARY_COTANGENT", state_elements),
        ("ROLLING_STATE_GRADIENT", state_elements),
        ("STEP_HIDDEN_WEIGHT_GRADIENT", features * features),
        ("STEP_OUTPUT_WEIGHT_GRADIENT", CHANNELS * features),
        ("STEP_BIAS_GRADIENT", CHANNELS),
        ("PERCEPTION", activation_elements),
        ("HIDDEN", activation_elements),
        ("HIDDEN_GRADIENT", activation_elements),
    )
    for (name, elements), value in zip(scratch_specs, diagnostic[4:]):
        for tile in range(args.tiles):
            _check_guarded_workspace(
                f"TILE_{tile}_{name}", value[tile], elements, slots
            )

    print("NCA_SYCL_ROLLOUT_SCRATCH_GUARD_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
