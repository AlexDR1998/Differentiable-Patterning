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
    def __init__(self, lowered, static_output):
        self._lowered = lowered
        self._static_output = static_output

    def compile(self):
        return _CompiledFilteredShardMap(
            self._lowered.compile(), self._static_output
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
        mapped = shard_map(
            array_function,
            mesh=self._mesh,
            in_specs=self._in_specs,
            out_specs=self._out_specs,
            **check_kwargs,
        )
        return dynamic, jax.jit(mapped), static_output

    def lower(self, *args):
        dynamic, mapped, static_output = self._prepare(args)
        lowered = mapped.lower(*dynamic)
        if "value" not in static_output:
            raise RuntimeError("shard_map tracing did not produce an output tree")
        return _LoweredFilteredShardMap(lowered, static_output["value"])

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
