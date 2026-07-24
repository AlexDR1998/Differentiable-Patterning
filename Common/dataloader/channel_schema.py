"""Metadata describing biological channels and co-measured image channels.

Micropattern datasets combine staining panels that measure overlapping sets of
biological markers.  A :class:`ChannelSchema` keeps the unique model state
separate from the measurement layout stored in the target tensor.  In
particular, experiment groups record which channels were acquired together and
can therefore support within-image correlation losses.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class MeasurementChannel:
    """One channel in an experimentally acquired image stack.

    Parameters
    ----------
    name:
        Unique, user-facing name in the assembled target tensor.
    marker:
        Biological state channel represented by this measurement.
    source_index:
        Channel index in the source image for this experiment.  ``None`` is
        useful for datasets assembled from separate single-channel files.
    """

    name: str
    marker: str
    source_index: Optional[int] = None


@dataclass(frozen=True)
class ExperimentChannelGroup:
    """Channels acquired together in one staining/imaging experiment."""

    name: str
    channels: Tuple[MeasurementChannel, ...]

    def __post_init__(self):
        if not self.name:
            raise ValueError("Experiment group names cannot be empty")
        if not self.channels:
            raise ValueError(f"Experiment group {self.name!r} has no channels")
        source_indices = [
            channel.source_index
            for channel in self.channels
            if channel.source_index is not None
        ]
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(
                f"Experiment group {self.name!r} has duplicate source indices"
            )


@dataclass(frozen=True)
class ChannelSchema:
    """Relationship between unique model state and experimental targets.

    Target tensors concatenate ``experiment_groups`` in their declared order,
    preserving repeated measurements of the same marker.  Model predictions
    contain one channel for each entry in ``state_channels``.
    """

    name: str
    state_channels: Tuple[str, ...]
    experiment_groups: Tuple[ExperimentChannelGroup, ...]

    def __post_init__(self):
        if not self.name:
            raise ValueError("Channel schema names cannot be empty")
        if not self.state_channels:
            raise ValueError("Channel schemas must define at least one state channel")
        if len(self.state_channels) != len(set(self.state_channels)):
            raise ValueError("State channel names must be unique")
        if not self.experiment_groups:
            raise ValueError("Channel schemas must define at least one experiment group")

        group_names = [group.name for group in self.experiment_groups]
        if len(group_names) != len(set(group_names)):
            raise ValueError("Experiment group names must be unique")

        measurement_names = [channel.name for channel in self.measurement_channels]
        if len(measurement_names) != len(set(measurement_names)):
            raise ValueError("Measurement channel names must be unique")

        state_channels = set(self.state_channels)
        unknown_markers = sorted(
            {
                channel.marker
                for channel in self.measurement_channels
                if channel.marker not in state_channels
            }
        )
        if unknown_markers:
            raise ValueError(
                "Measurement channels reference unknown state markers: "
                + ", ".join(unknown_markers)
            )

        measured_markers = {channel.marker for channel in self.measurement_channels}
        unmeasured_states = [
            marker for marker in self.state_channels if marker not in measured_markers
        ]
        if unmeasured_states:
            raise ValueError(
                "State channels have no corresponding measurement: "
                + ", ".join(unmeasured_states)
            )

    @property
    def measurement_channels(self) -> Tuple[MeasurementChannel, ...]:
        return tuple(
            channel
            for group in self.experiment_groups
            for channel in group.channels
        )

    @property
    def measurement_names(self) -> Tuple[str, ...]:
        return tuple(channel.name for channel in self.measurement_channels)

    @property
    def group_names(self) -> Tuple[str, ...]:
        return tuple(group.name for group in self.experiment_groups)

    def select_groups(self, group_names: Optional[Sequence[str]] = None):
        """Return a schema containing only the requested experiment groups.

        State channels are restricted to markers measured by the selected
        groups, while retaining their order in the parent schema.
        """

        if group_names is None:
            return self
        if isinstance(group_names, str):
            group_names = (group_names,)
        else:
            group_names = tuple(group_names)
        if not group_names:
            raise ValueError("At least one experiment group must be selected")
        if len(set(group_names)) != len(group_names):
            raise ValueError("Experiment group selections cannot contain duplicates")
        groups_by_name = {group.name: group for group in self.experiment_groups}
        unknown = [name for name in group_names if name not in groups_by_name]
        if unknown:
            raise ValueError(
                "Unknown experiment groups for schema "
                f"{self.name!r}: {', '.join(unknown)}"
            )
        selected_groups = tuple(groups_by_name[name] for name in group_names)
        selected_markers = {
            channel.marker
            for group in selected_groups
            for channel in group.channels
        }
        state_channels = tuple(
            marker for marker in self.state_channels if marker in selected_markers
        )
        return ChannelSchema(
            name=f"{self.name}[{','.join(group_names)}]",
            state_channels=state_channels,
            experiment_groups=selected_groups,
        )

    @property
    def group_sizes(self) -> Tuple[int, ...]:
        return tuple(len(group.channels) for group in self.experiment_groups)

    @property
    def group_measurement_indices(self) -> Tuple[Tuple[int, ...], ...]:
        groups = []
        start = 0
        for size in self.group_sizes:
            groups.append(tuple(range(start, start + size)))
            start += size
        return tuple(groups)

    @property
    def target_to_state(self) -> Tuple[int, ...]:
        state_index = {
            marker: index for index, marker in enumerate(self.state_channels)
        }
        return tuple(
            state_index[channel.marker] for channel in self.measurement_channels
        )

    @property
    def primary_measurements(self) -> Tuple[int, ...]:
        """First declared measurement for each biological state channel."""

        primary = []
        for state_index in range(self.n_state_channels):
            primary.append(self.target_to_state.index(state_index))
        return tuple(primary)

    @property
    def state_to_measurements(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            tuple(
                measurement_index
                for measurement_index, mapped_state in enumerate(self.target_to_state)
                if mapped_state == state_index
            )
            for state_index in range(self.n_state_channels)
        )

    @property
    def measurement_weights(self) -> Tuple[float, ...]:
        """Static weights giving each biological marker unit total weight."""

        occurrence_counts = tuple(
            len(indices) for indices in self.state_to_measurements
        )
        return tuple(
            1.0 / occurrence_counts[state_index]
            for state_index in self.target_to_state
        )

    @property
    def co_measurement_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """Target-channel pairs that were measured in the same image."""

        return tuple(
            pair
            for group_indices in self.group_measurement_indices
            for pair in combinations(group_indices, 2)
        )

    @property
    def correlation_pair_weights(self) -> Tuple[float, ...]:
        """Normalize biological pairs repeated across experimental groups."""

        state_pairs = tuple(
            tuple(sorted((self.target_to_state[left], self.target_to_state[right])))
            for left, right in self.co_measurement_pairs
        )
        pair_counts = {pair: state_pairs.count(pair) for pair in set(state_pairs)}
        return tuple(1.0 / pair_counts[pair] for pair in state_pairs)

    @property
    def duplicate_state_channels(self) -> Tuple[str, ...]:
        return tuple(
            marker
            for marker, measurements in zip(
                self.state_channels, self.state_to_measurements
            )
            if len(measurements) > 1
        )

    @property
    def n_state_channels(self) -> int:
        return len(self.state_channels)

    @property
    def n_measurement_channels(self) -> int:
        return len(self.measurement_channels)

    def validate_measurement_channel_count(self, channel_count: int):
        """Raise a descriptive error when a target tensor has the wrong width."""

        if channel_count != self.n_measurement_channels:
            raise ValueError(
                f"Schema {self.name!r} expects {self.n_measurement_channels} "
                f"measurement channels, got {channel_count}"
            )
