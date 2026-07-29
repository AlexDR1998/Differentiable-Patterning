"""Typed results returned by data loaders.

Keeping loader outputs in dataclasses makes array axes and optional metadata
explicit at call sites.  The dataclasses deliberately contain no loading or
processing behaviour; they are plain descriptions of already-loaded data.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ImageSequenceDataset:
    """A dense image sequence with axes ``[batch, time, channel, x, y]``."""

    data: Any
    filenames: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, item):
        """Temporary array-style access for downstream migration."""

        return self.data[item]

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self.data, dtype=dtype)


@dataclass(frozen=True)
class MicropatternDataset:
    """A micropattern target sequence and its measurement metadata.

    ``data`` normally has axes ``[batch, time, measurement, x, y]``.
    A measurement mask is either ``[batch, time, measurement]`` or ``None``
    for older datasets in which every returned value is observed.
    """

    data: Any
    channel_names: tuple[str, ...] = ()
    boundary_mask: Any | None = None
    measurement_mask: Any | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)

    @property
    def schema(self):
        """Channel schema attached by the loader, when available."""

        return self.aux.get("channel_schema")

    def __iter__(self):
        """Support the historical five-value API during the migration period."""

        yield self.data
        yield self.aux
        yield list(self.channel_names)
        yield self.boundary_mask
        yield self.measurement_mask


@dataclass(frozen=True)
class MicropatternShapeDataset:
    """Possibly ragged shape/radius data and its spatial masks."""

    data: Any
    masks: Any
    spatial_shapes: Sequence[Any] = ()
    channel_names: tuple[str, ...] = ()
    synthetic_initial_conditions: Any | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)
