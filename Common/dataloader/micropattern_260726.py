"""Loader for the multichannel 260726 NCA micropattern dataset."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Mapping

import jax.numpy as jnp
import numpy as np
import scipy.ndimage as ndi
import skimage.io
from skimage import morphology

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA


_CONDITIONS = ("ctrl", "sl0", "sl24")
_GROUP_DIRECTORIES = {
    "cell_fate_s1": lambda condition: Path("cell_fate_markers")
    / f"{condition}_s1",
    "cell_fate_s2": lambda condition: Path("cell_fate_markers")
    / f"{condition}_s2",
    "rna_expression": lambda condition: Path("signalling")
    / "rna_expression"
    / condition,
    "protein_response": lambda condition: Path("signalling")
    / "protein_response"
    / condition,
}
_STRUCTURAL_SOURCE_CHANNEL = {
    "cell_fate_s1": 3,
    "cell_fate_s2": 3,
    # DAPI is used only to register the RNA stack. It is not included in the
    # measurement schema, normalization statistics, returned data, or losses.
    "rna_expression": 0,
    "protein_response": 1,
}
_TIME_PATTERN = re.compile(r"(?:^|_)(\d+)h(?:_|\.|-)", re.IGNORECASE)
_RNA_REPLICATE_PATTERN = re.compile(r"-(\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class MicropatternImageRecord:
    """Location and experimental identity of one multichannel TIFF."""

    path: str
    condition: str
    group: str
    timestep: int
    replicate: int


def _natural_sort_key(path):
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    )


def _parse_timestep(path):
    match = _TIME_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse a timestep from {path}")
    return int(match.group(1))


def _source_condition(condition, timestep, substitute_preperturbation):
    if not substitute_preperturbation:
        return condition
    if condition == "sl0" and timestep == 0:
        return "ctrl"
    if condition == "sl24" and timestep <= 24:
        return "ctrl"
    return condition


def _resolve_manifest_path(root, directory, value):
    value = Path(value)
    candidates = [value] if value.is_absolute() else [root / value, directory / value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Manifest file {value} was not found relative to {root} or {directory}"
    )


def _select_replicates(files, group, replicate_count):
    """Return fixed replicate slots plus any deliberately unused files."""

    files = sorted(files, key=_natural_sort_key)
    selected = [None] * replicate_count
    unselected = []
    if group == "rna_expression":
        for path in files:
            match = _RNA_REPLICATE_PATTERN.search(path.name)
            if match is None:
                unselected.append(path)
                continue
            replicate = int(match.group(1)) - 1
            if 0 <= replicate < replicate_count and selected[replicate] is None:
                selected[replicate] = path
            else:
                unselected.append(path)
    else:
        selected[: min(len(files), replicate_count)] = files[:replicate_count]
        unselected.extend(files[replicate_count:])
    return selected, unselected


def build_micropattern_260726_manifest(
    root,
    conditions=("ctrl",),
    timesteps=(0, 12, 24, 36, 48),
    replicate_count=3,
    replicate_manifest=None,
    experiment_groups=None,
    substitute_preperturbation=True,
):
    """Index selected files without reading image pixels.

    ``replicate_manifest`` may override automatic selection for any
    ``(condition, group, timestep)`` key with an ordered sequence of up to
    ``replicate_count`` relative or absolute filenames.

    ``experiment_groups`` restricts indexing to the named staining groups;
    ``None`` selects all groups.
    """

    schema = MICROPATTERN_260726_SCHEMA.select_groups(experiment_groups)
    selected_group_names = schema.group_names
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Micropattern dataset root does not exist: {root}")
    conditions = tuple(conditions)
    timesteps = tuple(int(timestep) for timestep in timesteps)
    if not conditions or any(condition not in _CONDITIONS for condition in conditions):
        raise ValueError(f"conditions must be selected from {_CONDITIONS}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions cannot contain duplicates")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("timesteps cannot contain duplicates")
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")
    required_conditions = set(conditions)
    if substitute_preperturbation and any(
        condition in {"sl0", "sl24"} for condition in conditions
    ):
        required_conditions.add("ctrl")

    selected = {}
    records = []
    unselected_files = []
    replicate_manifest = {} if replicate_manifest is None else replicate_manifest
    for condition in sorted(required_conditions):
        for group in selected_group_names:
            relative_directory = _GROUP_DIRECTORIES[group]
            directory = root / relative_directory(condition)
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Expected directory for {condition}/{group}: {directory}"
                )
            files_by_time = {timestep: [] for timestep in timesteps}
            for path in directory.glob("*.tif"):
                timestep = _parse_timestep(path)
                if timestep in files_by_time:
                    files_by_time[timestep].append(path.resolve())

            for timestep in timesteps:
                key = (condition, group, timestep)
                override = replicate_manifest.get(key)
                if override is None:
                    slots, unused = _select_replicates(
                        files_by_time[timestep], group, replicate_count
                    )
                else:
                    if len(override) > replicate_count:
                        raise ValueError(
                            f"Manifest entry {key} contains more than "
                            f"{replicate_count} files"
                        )
                    slots = [None] * replicate_count
                    for replicate, value in enumerate(override):
                        if value is not None:
                            resolved = _resolve_manifest_path(
                                root, directory, value
                            )
                            if resolved not in files_by_time[timestep]:
                                raise ValueError(
                                    f"Manifest file {resolved} does not belong to "
                                    f"{condition}/{group}/{timestep}h"
                                )
                            slots[replicate] = resolved
                    chosen = {path for path in slots if path is not None}
                    if len(chosen) != sum(path is not None for path in slots):
                        raise ValueError(f"Manifest entry {key} repeats a source file")
                    unused = [
                        path for path in files_by_time[timestep] if path not in chosen
                    ]
                selected[key] = tuple(slots)
                unselected_files.extend(unused)
                for replicate, path in enumerate(slots):
                    if path is not None:
                        records.append(
                            MicropatternImageRecord(
                                path=str(path),
                                condition=condition,
                                group=group,
                                timestep=timestep,
                                replicate=replicate,
                            )
                        )

    return {
        "records": tuple(records),
        "selected": selected,
        "unselected_files": tuple(
            str(path) for path in sorted(set(unselected_files), key=_natural_sort_key)
        ),
    }


def _source_channel_count(group):
    schema_group = next(
        item
        for item in MICROPATTERN_260726_SCHEMA.experiment_groups
        if item.name == group
    )
    return max(channel.source_index for channel in schema_group.channels) + 1


def _read_multichannel_image(path, group):
    image = np.asarray(skimage.io.imread(path))
    expected_channels = _source_channel_count(group)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional TIFF at {path}, got {image.shape}")
    if image.shape[-1] == expected_channels:
        return image
    if image.shape[0] == expected_channels:
        return np.moveaxis(image, 0, -1)
    raise ValueError(
        f"Expected {expected_channels} source channels for {group} at {path}, "
        f"got shape {image.shape}"
    )


def _channel_values(image, record, source_index):
    values = image[..., source_index].astype(np.float32)
    if (
        record.group == "cell_fate_s2"
        and record.condition == "ctrl"
        and record.timestep == 0
        and source_index == 1
    ):
        values = values * 0.075
    return values


def _percentile_from_histogram(histogram, percentile):
    if histogram.sum() == 0:
        raise ValueError("Cannot calculate percentiles from an empty histogram")
    threshold = percentile / 100.0 * histogram.sum()
    return float(np.searchsorted(np.cumsum(histogram), threshold, side="left"))


def _compute_histogram_bins(records, hist_eqs, schema):
    histograms = np.zeros((schema.n_measurement_channels, 65536), dtype=np.uint64)
    group_target_indices = {
        group.name: target_indices
        for group, target_indices in zip(
            schema.experiment_groups, schema.group_measurement_indices
        )
    }
    for record in records:
        image = _read_multichannel_image(record.path, record.group)
        schema_group = next(
            group for group in schema.experiment_groups if group.name == record.group
        )
        for target_index, channel in zip(
            group_target_indices[record.group], schema_group.channels
        ):
            values = _channel_values(image, record, channel.source_index)
            if np.min(values) < 0 or np.max(values) > 65535:
                raise ValueError(
                    "Automatic histogram calculation expects intensities in [0, 65535]; "
                    "provide histogram_bins explicitly for other data ranges"
                )
            integer_values = np.rint(values).astype(np.uint16)
            histograms[target_index] += np.bincount(
                integer_values.reshape(-1), minlength=65536
            ).astype(np.uint64)

    bins = np.zeros((schema.n_measurement_channels, 2), dtype=np.float32)
    for channel in range(schema.n_measurement_channels):
        bins[channel, 0] = _percentile_from_histogram(
            histograms[channel], hist_eqs[0]
        )
        bins[channel, 1] = _percentile_from_histogram(
            histograms[channel], hist_eqs[1]
        )
    return bins


def _coerce_histogram_bins(histogram_bins, schema):
    if isinstance(histogram_bins, Mapping):
        missing = [
            name for name in schema.measurement_names if name not in histogram_bins
        ]
        if missing:
            raise ValueError("Missing histogram bins for: " + ", ".join(missing))
        histogram_bins = [histogram_bins[name] for name in schema.measurement_names]
    bins = np.asarray(histogram_bins, dtype=np.float32)
    expected_shape = (schema.n_measurement_channels, 2)
    if bins.shape != expected_shape:
        raise ValueError(
            f"histogram_bins must have shape {expected_shape}, got {bins.shape}"
        )
    if np.any(bins[:, 1] <= bins[:, 0]):
        raise ValueError("Each histogram upper bound must exceed its lower bound")
    return bins


def _foreground_mask(image, group, boundary_radius_quantile, boundary_radius_scale):
    if group in _STRUCTURAL_SOURCE_CHANNEL:
        reference = image[..., _STRUCTURAL_SOURCE_CHANNEL[group]].astype(np.float32)
    else:
        schema_group = next(
            item
            for item in MICROPATTERN_260726_SCHEMA.experiment_groups
            if item.name == group
        )
        reference = np.mean(
            np.stack(
                [
                    image[..., channel.source_index].astype(np.float32)
                    for channel in schema_group.channels
                ],
                axis=0,
            ),
            axis=0,
        )
    if group == "rna_expression":
        smoothing_radius = max(1.0, min(reference.shape) / 200.0)
    else:
        smoothing_radius = 1.0
    smooth = ndi.gaussian_filter(reference, sigma=smoothing_radius)
    if np.all(smooth == smooth.flat[0]):
        raise ValueError(f"Cannot infer a foreground mask from constant {group} data")
    threshold = np.mean(smooth)
    foreground = smooth > threshold
    foreground = morphology.remove_small_objects(
        foreground,
        min_size=max(4, foreground.size // 20000),
    )
    if not np.any(foreground):
        raise ValueError(f"Could not identify foreground pixels for {group}")
    if group == "rna_expression":
        # RNA DAPI is dimmer and less cleanly separated than the cell-fate
        # structural channels. Connect nearby nuclei into a colony-sized blob,
        # then reject every disconnected artifact before estimating geometry.
        dilation_steps = max(1, round(min(reference.shape) / 200.0))
        connected = ndi.binary_dilation(foreground, iterations=dilation_steps)
        labels, component_count = ndi.label(connected)
        if component_count == 0:
            raise ValueError("Could not identify a connected RNA colony")
        component_sizes = np.bincount(labels.reshape(-1))
        component_sizes[0] = 0
        foreground = labels == np.argmax(component_sizes)
    foreground = ndi.binary_fill_holes(foreground)
    coordinates = np.argwhere(foreground)
    if group == "rna_expression":
        centre = np.mean(coordinates, axis=0)
    else:
        centre = np.median(coordinates, axis=0)
    distances = np.sqrt(np.sum((coordinates - centre[None]) ** 2, axis=1))
    radius = np.quantile(distances, boundary_radius_quantile)
    radius *= boundary_radius_scale
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Could not infer a circular boundary for {group}")
    rows, columns = np.ogrid[: image.shape[0], : image.shape[1]]
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius**2


def _align_image_and_mask(image, mask):
    centre = ndi.center_of_mass(mask)
    target = ((mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0)
    shift = (target[0] - centre[0], target[1] - centre[1])
    image = ndi.shift(
        image,
        shift=(shift[0], shift[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    mask = ndi.shift(
        mask.astype(np.float32),
        shift=shift,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5
    return image, mask


def _downsample_image_and_mask(image, mask, downsample):
    if downsample == 1:
        return image, mask
    height, width = image.shape[:2]
    pad_height = (-height) % downsample
    pad_width = (-width) % downsample
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)))
    mask = np.pad(mask, ((0, pad_height), (0, pad_width)))
    new_height = image.shape[0] // downsample
    new_width = image.shape[1] // downsample
    image = image.reshape(
        new_height, downsample, new_width, downsample, image.shape[-1]
    ).mean(axis=(1, 3))
    mask = mask.reshape(new_height, downsample, new_width, downsample).mean(
        axis=(1, 3)
    ) > 0.5
    return image, mask


def load_micropattern_260726(
    root,
    conditions=("ctrl",),
    timesteps=(0, 12, 24, 36, 48),
    downsample=4,
    replicate_count=3,
    replicate_manifest=None,
    experiment_groups=None,
    substitute_preperturbation=True,
    histogram_bins=None,
    hist_eqs=(0.5, 99.95),
    align=True,
    strict_replicates=False,
    boundary_radius_quantile=0.98,
    boundary_radius_scale=1.0,
    batch_multiplier=1,
):
    """Load physical replicates of the multichannel 260726 NCA dataset.

    ``boundary_radius_quantile`` robustly trims foreground pixels far from the
    inferred colony centre before constructing a circle.  Lower values or a
    ``boundary_radius_scale`` below one produce a stricter common boundary.
    ``batch_multiplier`` repeats each selected physical batch in the returned
    training batch while retaining the original replicate provenance.

    Returns
    -------
    targets:
        Float32 array ``[B, T, M, X, Y]`` where ``M`` is the number of
        channels in the selected experiment groups and
        ``B = replicate_count * len(conditions) * batch_multiplier``.
    aux:
        Selected schema, provenance, group masks, histogram bins, and inventory.
    measurement_names:
        Names aligned with the target channel axis.
    boundary_mask:
        Boolean array ``[B, 1, X, Y]`` derived from cell-fate S1 when selected,
        otherwise from the first selected group.
    measurement_mask:
        Boolean availability array ``[B, T, M]``.  Downstream one-step losses
        can use ``measurement_mask[:, 1:]``.
    """

    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if len(hist_eqs) != 2 or not 0 <= hist_eqs[0] < hist_eqs[1] <= 100:
        raise ValueError("hist_eqs must contain increasing percentiles in [0, 100]")
    if not 0.5 < boundary_radius_quantile <= 1.0:
        raise ValueError("boundary_radius_quantile must be in (0.5, 1.0]")
    if boundary_radius_scale <= 0:
        raise ValueError("boundary_radius_scale must be positive")
    if batch_multiplier <= 0 or int(batch_multiplier) != batch_multiplier:
        raise ValueError("batch_multiplier must be a positive integer")
    batch_multiplier = int(batch_multiplier)
    conditions = tuple(conditions)
    timesteps = tuple(int(timestep) for timestep in timesteps)
    schema = MICROPATTERN_260726_SCHEMA.select_groups(experiment_groups)
    inventory = build_micropattern_260726_manifest(
        root=root,
        conditions=conditions,
        timesteps=timesteps,
        replicate_count=replicate_count,
        replicate_manifest=replicate_manifest,
        experiment_groups=schema.group_names,
        substitute_preperturbation=substitute_preperturbation,
    )
    selected = inventory["selected"]
    if strict_replicates:
        missing = []
        for condition in conditions:
            for timestep in timesteps:
                source_condition = _source_condition(
                    condition, timestep, substitute_preperturbation
                )
                for group in schema.group_names:
                    slots = selected[(source_condition, group, timestep)]
                    for replicate, path in enumerate(slots):
                        if path is None:
                            missing.append(
                                f"{condition}/{group}/{timestep}h/replicate-{replicate + 1}"
                            )
        if missing:
            raise ValueError("Missing required measurements: " + ", ".join(missing))

    records = inventory["records"]
    if histogram_bins is None:
        histogram_bins = _compute_histogram_bins(records, hist_eqs, schema)
    else:
        histogram_bins = _coerce_histogram_bins(histogram_bins, schema)
    group_target_indices = {
        group.name: target_indices
        for group, target_indices in zip(
            schema.experiment_groups, schema.group_measurement_indices
        )
    }
    group_index = {
        group.name: index for index, group in enumerate(schema.experiment_groups)
    }
    record_lookup = {
        (record.condition, record.group, record.timestep, record.replicate): record
        for record in records
    }

    @lru_cache(maxsize=8)
    def load_processed(path, condition, group, timestep, replicate):
        record = MicropatternImageRecord(
            path, condition, group, timestep, replicate
        )
        raw = _read_multichannel_image(path, group)
        raw_mask = _foreground_mask(
            raw,
            group,
            boundary_radius_quantile,
            boundary_radius_scale,
        )
        schema_group = schema.experiment_groups[group_index[group]]
        channels = []
        for target_index, channel in zip(
            group_target_indices[group], schema_group.channels
        ):
            values = _channel_values(raw, record, channel.source_index)
            lower, upper = histogram_bins[target_index]
            channels.append(np.clip((values - lower) / (upper - lower), 0.0, 1.0))
        image = np.stack(channels, axis=-1).astype(np.float32)
        if align:
            image, raw_mask = _align_image_and_mask(image, raw_mask)
        image, raw_mask = _downsample_image_and_mask(
            image, raw_mask, downsample
        )
        return image, raw_mask

    first_record = next(iter(records), None)
    if first_record is None:
        raise ValueError("No images were selected from the requested dataset")
    first_image, _ = load_processed(
        first_record.path,
        first_record.condition,
        first_record.group,
        first_record.timestep,
        first_record.replicate,
    )
    spatial_shape = first_image.shape[:2]
    batch_count = len(conditions) * replicate_count
    targets = np.zeros(
        (
            batch_count,
            len(timesteps),
            schema.n_measurement_channels,
            *spatial_shape,
        ),
        dtype=np.float32,
    )
    measurement_mask = np.zeros(
        (batch_count, len(timesteps), schema.n_measurement_channels), dtype=bool
    )
    group_masks = np.zeros(
        (
            batch_count,
            len(timesteps),
            len(schema.experiment_groups),
            *spatial_shape,
        ),
        dtype=bool,
    )
    group_mask = np.zeros(
        (batch_count, len(timesteps), len(schema.experiment_groups)), dtype=bool
    )
    source_conditions = np.full(group_mask.shape, "", dtype=object)
    substituted = np.zeros(group_mask.shape, dtype=bool)
    source_files = np.full(group_mask.shape, "", dtype=object)

    batch_conditions = []
    batch_replicates = []
    for condition_index, condition in enumerate(conditions):
        for replicate in range(replicate_count):
            batch = condition_index * replicate_count + replicate
            batch_conditions.append(condition)
            batch_replicates.append(replicate + 1)
            for time_index, timestep in enumerate(timesteps):
                source_condition = _source_condition(
                    condition, timestep, substitute_preperturbation
                )
                for group in schema.group_names:
                    record = record_lookup.get(
                        (source_condition, group, timestep, replicate)
                    )
                    if record is None:
                        continue
                    image, mask = load_processed(
                        record.path,
                        record.condition,
                        record.group,
                        record.timestep,
                        record.replicate,
                    )
                    if image.shape[:2] != spatial_shape:
                        raise ValueError(
                            f"Processed spatial shape mismatch for {record.path}: "
                            f"expected {spatial_shape}, got {image.shape[:2]}"
                        )
                    target_indices = group_target_indices[group]
                    targets[batch, time_index, target_indices] = np.moveaxis(
                        image, -1, 0
                    )
                    measurement_mask[batch, time_index, target_indices] = True
                    current_group = group_index[group]
                    group_masks[batch, time_index, current_group] = mask
                    group_mask[batch, time_index, current_group] = True
                    source_conditions[batch, time_index, current_group] = source_condition
                    substituted[batch, time_index, current_group] = (
                        source_condition != condition
                    )
                    source_files[batch, time_index, current_group] = record.path

    boundary_candidates = np.zeros((batch_count, *spatial_shape), dtype=bool)
    boundary_candidate_mask = np.zeros((batch_count,), dtype=bool)
    primary_group = group_index.get("cell_fate_s1", 0)
    for batch in range(batch_count):
        available_times = np.flatnonzero(group_mask[batch, :, primary_group])
        if available_times.size:
            boundary_candidates[batch] = group_masks[
                batch, available_times[-1], primary_group
            ]
            boundary_candidate_mask[batch] = True
        else:
            available = np.argwhere(group_mask[batch])
            if available.size:
                time_index, current_group = available[-1]
                boundary_candidates[batch] = group_masks[
                    batch, time_index, current_group
                ]
                boundary_candidate_mask[batch] = True

    if not np.any(boundary_candidate_mask):
        raise ValueError("No measurements were available to infer a boundary mask")
    common_boundary = np.mean(
        boundary_candidates[boundary_candidate_mask].astype(np.float32),
        axis=0,
    ) >= 0.5
    boundary_mask = np.broadcast_to(
        common_boundary[None, None],
        (batch_count, 1, *spatial_shape),
    ).copy()

    # Apply one trajectory boundary to all independently stained panels. RNA
    # images have already been registered using DAPI, which is then discarded.
    targets *= boundary_mask[:, None].astype(targets.dtype)
    group_masks = boundary_mask[:, None] & group_mask[..., None, None]

    if batch_multiplier > 1:
        targets = np.concatenate([targets] * batch_multiplier, axis=0)
        measurement_mask = np.concatenate(
            [measurement_mask] * batch_multiplier, axis=0
        )
        boundary_mask = np.concatenate([boundary_mask] * batch_multiplier, axis=0)
        group_masks = np.concatenate([group_masks] * batch_multiplier, axis=0)
        group_mask = np.concatenate([group_mask] * batch_multiplier, axis=0)
        source_conditions = np.concatenate(
            [source_conditions] * batch_multiplier, axis=0
        )
        substituted = np.concatenate([substituted] * batch_multiplier, axis=0)
        source_files = np.concatenate([source_files] * batch_multiplier, axis=0)
        batch_conditions = batch_conditions * batch_multiplier
        batch_replicates = batch_replicates * batch_multiplier

    aux = {
        "channel_schema": schema,
        "selected_experiment_groups": schema.group_names,
        "batch_multiplier": batch_multiplier,
        "manifest": records,
        "unselected_files": inventory["unselected_files"],
        "histogram_bins": histogram_bins,
        "group_boundary_masks": group_masks,
        "group_mask": group_mask,
        "source_conditions": source_conditions,
        "is_substituted": substituted,
        "source_files": source_files,
        "batch_conditions": tuple(batch_conditions),
        "batch_replicates": tuple(batch_replicates),
        "timesteps": timesteps,
        "loss_measurement_mask": measurement_mask[:, 1:],
    }
    return (
        jnp.asarray(targets),
        aux,
        list(schema.measurement_names),
        jnp.asarray(boundary_mask),
        jnp.asarray(measurement_mask),
    )
