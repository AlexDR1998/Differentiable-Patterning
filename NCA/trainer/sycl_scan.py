"""Loop helpers for fused SYCL rollouts."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def scan_carry_only(body, init, xs, *, kind):
    """Run a scan whose per-step output is intentionally unused.

    Equinox's checkpointed scan evaluates ``body`` separately to determine the
    shape of its output buffer. Fused NCA rollout chunks return no useful
    output, so allocating and shape-probing that buffer is unnecessary. A
    checkpointed while loop preserves Equinox's online checkpointing algorithm
    while carrying only the values required by the next chunk.
    """
    if kind == "lax":
        carry, _ = jax.lax.scan(body, init, xs)
        return carry
    if kind != "checkpointed":
        raise ValueError(
            "SYCL rollout loop kind must be 'lax' or 'checkpointed'; "
            f"got {kind!r}"
        )

    length = xs.shape[0]

    def condition(loop_state):
        index, _ = loop_state
        return index < length

    def step(loop_state):
        index, carry = loop_state
        carry, _ = body(carry, xs[index])
        return index + 1, carry

    _, carry = eqx.internal.while_loop(
        condition,
        step,
        (jnp.asarray(0, dtype=jnp.int32), init),
        max_steps=length,
        kind="checkpointed",
    )
    return carry


__all__ = ["scan_carry_only"]
