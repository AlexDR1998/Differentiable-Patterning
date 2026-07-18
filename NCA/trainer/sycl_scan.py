"""Loop helpers for fused SYCL rollouts."""

from __future__ import annotations

import jax


def scan_carry_only(body, init, xs, *, kind):
    """Run a scan whose per-step output is intentionally unused.

    Equinox's checkpointed scan and while loop both retrace their body through
    auxiliary transformations. On the Intel JAX 0.5 pmap path those transforms
    materialise the mapped tile axis inside the body. Instead, checkpointed
    mode uses a two-level JAX scan: complete blocks are rematerialised during
    the backward pass, whilst an individual recomputed block uses an ordinary
    scan. Choosing square-root-sized blocks keeps only O(sqrt(T)) recurrent
    states live without placing another transform around the custom call.
    """
    if kind == "lax":
        carry, _ = jax.lax.scan(body, init, xs)
        return carry
    if kind != "checkpointed":
        raise ValueError(
            "SYCL rollout loop kind must be 'lax' or 'checkpointed'; "
            f"got {kind!r}"
        )

    length = int(xs.shape[0])
    if length == 0:
        return init

    block_size = max(1, int(length**0.5))
    complete_length = (length // block_size) * block_size
    complete_blocks = xs[:complete_length].reshape(
        (-1, block_size, *xs.shape[1:])
    )

    def run_block(carry, block_xs):
        carry, _ = jax.lax.scan(body, carry, block_xs)
        return carry, None

    # Rematerialise whole blocks. During their backward recomputation only one
    # block's ordinary scan residuals are live, giving approximately
    # n_blocks + block_size saved recurrent states.
    carry, _ = jax.lax.scan(
        jax.checkpoint(run_block), init, complete_blocks
    )

    remainder = xs[complete_length:]
    if remainder.shape[0] > 0:
        def run_remainder(value):
            value, _ = jax.lax.scan(body, value, remainder)
            return value

        carry = jax.checkpoint(run_remainder)(carry)
    return carry


__all__ = ["scan_carry_only"]
