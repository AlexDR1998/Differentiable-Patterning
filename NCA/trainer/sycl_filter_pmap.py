"""A small filtered pmap wrapper without Equinox's vmap shape probe.

Equinox 0.11/0.12 determines ``filter_pmap`` output structure by first running
the function under ``vmap``. That changes the semantics of nested scans and
custom calls: their bodies can observe a materialised mapped axis that is not
present when the function is genuinely compiled by ``pmap``. This wrapper
partitions array/static leaves like Equinox, but discovers static outputs while
the real pmap is traced.
"""

from __future__ import annotations

import equinox as eqx
import jax


class _CompiledFilteredPmap:
    def __init__(self, compiled, function, static_output):
        self._compiled = compiled
        self._function = function
        self._static_output = static_output

    def __call__(self, *args):
        dynamic, _ = eqx.partition(args, eqx.is_array)
        dynamic_output = self._compiled(dynamic)
        return eqx.combine(dynamic_output, self._static_output)


class _LoweredFilteredPmap:
    def __init__(self, lowered, function, static_output):
        self._lowered = lowered
        self._function = function
        self._static_output = static_output

    def compile(self):
        return _CompiledFilteredPmap(
            self._lowered.compile(),
            self._function,
            self._static_output,
        )


class FilteredPmapNoProbe:
    """Filter static leaves and trace outputs directly under ``jax.pmap``."""

    def __init__(self, function, *, in_axes, out_axes, axis_name):
        self._function = function
        self._in_axes = in_axes
        self._out_axes = out_axes
        self._axis_name = axis_name

    def _prepare(self, args):
        # Keep the mapped PyTree identical to the function's actual argument
        # tuple. In particular, do not prepend the function as a synthetic
        # static leaf: older pmap/PJRT implementations can resolve a prefix
        # ``in_axes`` against that artificial nesting differently.
        dynamic, static = eqx.partition(args, eqx.is_array)
        static_output = {}

        def array_function(dynamic_arguments):
            full_args = eqx.combine(dynamic_arguments, static)
            output = self._function(*full_args)
            dynamic_output, output_static = eqx.partition(
                output, eqx.is_array
            )
            static_output["value"] = output_static
            return dynamic_output

        mapped = jax.pmap(
            array_function,
            in_axes=(self._in_axes,),
            out_axes=self._out_axes,
            axis_name=self._axis_name,
        )
        return dynamic, mapped, static_output

    def lower(self, *args):
        dynamic, mapped, static_output = self._prepare(args)
        lowered = mapped.lower(dynamic)
        if "value" not in static_output:
            raise RuntimeError("pmap tracing did not produce an output tree")
        return _LoweredFilteredPmap(
            lowered, self._function, static_output["value"]
        )

    def __call__(self, *args):
        dynamic, mapped, static_output = self._prepare(args)
        dynamic_output = mapped(dynamic)
        if "value" not in static_output:
            raise RuntimeError("pmap tracing did not produce an output tree")
        return eqx.combine(dynamic_output, static_output["value"])


def filter_pmap_no_probe(function, *, in_axes, out_axes, axis_name):
    return FilteredPmapNoProbe(
        function,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
    )


__all__ = ["FilteredPmapNoProbe", "filter_pmap_no_probe"]
