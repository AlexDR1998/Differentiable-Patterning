"""Loop helpers for fused SYCL rollouts."""

from __future__ import annotations

import jax
import jax.tree_util as jtu


def scan_carry_only(body, init, xs, *, kind):
    """Run a carry-only ``lax`` or checkpointed scan.

    ``xs`` is a PyTree whose leaves share leading length ``T``; the returned
    PyTree has the same structure and shapes as ``init``. Checkpointed mode
    rematerialises square-root-sized blocks, retaining approximately
    ``O(sqrt(T))`` recurrent states while preserving the ordinary scan body.
    """
    if kind == "lax":
        carry, _ = jax.lax.scan(body, init, xs)
        return carry
    if kind != "checkpointed":
        raise ValueError(
            "SYCL rollout loop kind must be 'lax' or 'checkpointed'; "
            f"got {kind!r}"
        )

    leaves = jtu.tree_leaves(xs)
    if not leaves:
        raise ValueError("scan_carry_only requires at least one scan input")
    lengths = {int(leaf.shape[0]) for leaf in leaves}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent SYCL scan input lengths: {lengths}")
    length = lengths.pop()
    if length == 0:
        return init

    block_size = max(1, int(length**0.5))
    complete_length = (length // block_size) * block_size
    complete_blocks = jtu.tree_map(
        lambda value: value[:complete_length].reshape(
            (-1, block_size, *value.shape[1:])
        ),
        xs,
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

    remainder = jtu.tree_map(lambda value: value[complete_length:], xs)
    if length > complete_length:
        def run_remainder(value):
            value, _ = jax.lax.scan(body, value, remainder)
            return value

        carry = jax.checkpoint(run_remainder)(carry)
    return carry


__all__ = ["scan_carry_only"]
