"""Filtered ``shard_map`` support for the two-tile SYCL trainer."""

from __future__ import annotations

import equinox as eqx
import jax

try:
    shard_map = jax.shard_map
    _CHECK_ARGUMENT = "check_vma"
except AttributeError:
    from jax.experimental.shard_map import shard_map

    _CHECK_ARGUMENT = "check_rep"


class _CompiledFilteredShardMap:
    def __init__(self, compiled, static_output):
        self._compiled = compiled
        self._static_output = static_output

    def __call__(self, *args):
        dynamic, _ = eqx.partition(args, eqx.is_array)
        dynamic_output = self._compiled(*dynamic)
        return eqx.combine(dynamic_output, self._static_output)


class _LoweredFilteredShardMap:
    def __init__(self, mapped, static_output):
        self._mapped = mapped
        self._static_output = static_output

    def compile(self):
        return _CompiledFilteredShardMap(
            self._mapped, self._static_output
        )


class FilteredShardMap:
    """Filter non-array leaves and expose the usual JIT lower/compile API."""

    def __init__(
        self,
        function,
        *,
        mesh,
        in_specs,
        out_specs,
        check_rep=False,
    ):
        self._function = function
        self._mesh = mesh
        self._in_specs = in_specs
        self._out_specs = out_specs
        self._check_rep = check_rep

    def _prepare(self, args):
        dynamic, static = eqx.partition(args, eqx.is_array)
        static_output = {}

        def array_function(*dynamic_args):
            full_args = eqx.combine(dynamic_args, static)
            output = self._function(*full_args)
            dynamic_output, output_static = eqx.partition(
                output, eqx.is_array
            )
            static_output["value"] = output_static
            return dynamic_output

        check_kwargs = {_CHECK_ARGUMENT: self._check_rep}
        # Cache the expensive tile-local trace without placing JIT outside the
        # SPMD transform. Thus JIT only ever sees physical tile-local shapes.
        local_jitted = jax.jit(array_function)
        mapped = shard_map(
            local_jitted,
            mesh=self._mesh,
            in_specs=self._in_specs,
            out_specs=self._out_specs,
            **check_kwargs,
        )
        return dynamic, mapped, static_output

    def lower(self, *args):
        dynamic, mapped, static_output = self._prepare(args)
        # Do not wrap the complete shard_map in another jax.jit. Intel JAX
        # 0.5 materialises the mesh axis inside nested remat/scan transforms
        # for jit(shard_map(...)), turning a tile-local [N,C,H,W] state back
        # into global [tiles,N,C,H,W]. Direct shard_map execution does not;
        # this is also the path exercised by the Intel shard-map smoke test.
        #
        # Calling once here populates both the tile-local JIT cache and the
        # shard_map executable cache. The returned callable reuses those exact
        # function objects during training.
        traced_output = mapped(*dynamic)
        jax.block_until_ready(traced_output)
        if "value" not in static_output:
            raise RuntimeError("shard_map tracing did not produce an output tree")
        return _LoweredFilteredShardMap(mapped, static_output["value"])

    def __call__(self, *args):
        dynamic, mapped, static_output = self._prepare(args)
        dynamic_output = mapped(*dynamic)
        if "value" not in static_output:
            raise RuntimeError("shard_map tracing did not produce an output tree")
        return eqx.combine(dynamic_output, static_output["value"])


def filter_shard_map(
    function,
    *,
    mesh,
    in_specs,
    out_specs,
    check_rep=False,
):
    return FilteredShardMap(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=check_rep,
    )


__all__ = ["FilteredShardMap", "filter_shard_map"]
