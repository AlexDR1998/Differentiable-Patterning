"""Equinox-style filtering for a loss transformed by :func:`jax.shard_map`."""

from __future__ import annotations

import equinox as eqx
import jax

try:
    shard_map = jax.shard_map
    _CHECK_ARGUMENT = "check_vma"
except AttributeError:
    from jax.experimental.shard_map import shard_map

    _CHECK_ARGUMENT = "check_rep"


class FilteredShardMap:
    """Apply ``shard_map`` to array leaves while closing over static leaves.

    The callable preserves the input/output PyTree structures of ``function``.
    For data-parallel training, transform the scalar loss with this class and
    apply reverse-mode autodiff outside the resulting mapped loss.
    """

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
        """Partition arguments and construct the array-only mapped function."""
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
        return dynamic, mapped, static_output

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
    """Construct a filtered shard-map callable with matching PyTree specs."""
    return FilteredShardMap(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=check_rep,
    )


__all__ = ["FilteredShardMap", "filter_shard_map"]
