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

from Common.dataloader import background, hot_pixels
from Common.dataloader.alignment import grid_position, load_colony_alignment
from Common.dataloader.background import subtract_background
from Common.dataloader.disk_cache import cache_folder, cached_array
from Common.dataloader.hot_pixels import replace_hot_pixels_per_channel
from Common.dataloader.micropattern_schemas import (
    DEFAULT_260726_HISTOGRAM_PERCENTILES,
    DEFAULT_260726_INTENSITY_FACTORS,
    MICROPATTERN_260726_SCHEMA,
)
from Common.dataloader.normalisation import percentile_bins
from Common.dataloader.results import MicropatternDataset


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


def source_condition(condition, timestep, substitute_preperturbation=True):
    """Condition whose images stand in for ``condition`` at ``timestep``.

    Before a knockout the colony is still a control colony, so with
    ``substitute_preperturbation`` the control images are used there.
    """
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


def _select_replicates(files, group, replicate_indices):
    """Split ``files`` into the selected replicate slots and the unused rest."""

    files = sorted(files, key=_natural_sort_key)
    replicate_indices = tuple(replicate_indices)
    selected = [None] * len(replicate_indices)
    unselected = []
    if group == "rna_expression":
        for path in files:
            match = _RNA_REPLICATE_PATTERN.search(path.name)
            if match is None:
                unselected.append(path)
                continue
            replicate = int(match.group(1)) - 1
            if replicate in replicate_indices:
                slot = replicate_indices.index(replicate)
                if selected[slot] is None:
                    selected[slot] = path
                    continue
                unselected.append(path)
            else:
                unselected.append(path)
    else:
        for replicate, path in enumerate(files):
            if replicate in replicate_indices:
                selected[replicate_indices.index(replicate)] = path
            else:
                unselected.append(path)
    return selected, unselected


def build_micropattern_260726_manifest(
    root,
    conditions=("ctrl",),
    timesteps=(0, 12, 24, 36, 48),
    replicate_count=3,
    replicate_indices=None,
    replicate_manifest=None,
    experiment_groups=None,
    substitute_preperturbation=True,
):
    """List the selected image files, without reading them.

    ``replicate_indices`` selects zero-based replicate slots; by default the
    first ``replicate_count``. ``replicate_manifest`` can replace the
    automatic choice for any ``(condition, group, timestep)`` key with an
    ordered list of up to ``replicate_count`` relative or absolute filenames.
    ``experiment_groups`` restricts the staining groups (``None`` = all).
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
    if replicate_indices is None:
        if replicate_count <= 0:
            raise ValueError("replicate_count must be positive")
        replicate_indices = tuple(range(replicate_count))
    else:
        replicate_indices = tuple(int(index) for index in replicate_indices)
        if not replicate_indices or min(replicate_indices) < 0:
            raise ValueError("replicate_indices must contain non-negative indices")
        if len(set(replicate_indices)) != len(replicate_indices):
            raise ValueError("replicate_indices cannot contain duplicates")
        replicate_count = len(replicate_indices)
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
                        files_by_time[timestep], group, replicate_indices
                    )
                else:
                    slots = [None] * replicate_count
                    for slot, replicate in enumerate(replicate_indices):
                        value = override[replicate] if replicate < len(override) else None
                        if value is not None:
                            resolved = _resolve_manifest_path(
                                root, directory, value
                            )
                            if resolved not in files_by_time[timestep]:
                                raise ValueError(
                                    f"Manifest file {resolved} does not belong to "
                                    f"{condition}/{group}/{timestep}h"
                                )
                            slots[slot] = resolved
                    chosen = {path for path in slots if path is not None}
                    if len(chosen) != sum(path is not None for path in slots):
                        raise ValueError(f"Manifest entry {key} repeats a source file")
                    unused = [
                        path for path in files_by_time[timestep] if path not in chosen
                    ]
                selected[key] = tuple(slots)
                unselected_files.extend(unused)
                for slot, path in enumerate(slots):
                    if path is not None:
                        records.append(
                            MicropatternImageRecord(
                                path=str(path),
                                condition=condition,
                                group=group,
                                timestep=timestep,
                                replicate=replicate_indices[slot],
                            )
                        )

    return {
        "records": tuple(records),
        "selected": selected,
        "replicate_indices": replicate_indices,
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


def _channel_values(image, record, channel, intensity_factors):
    values = image[..., channel.source_index].astype(np.float32)
    factor = intensity_factors.get(channel.name, {}).get(record.timestep, 1.0)
    return values * factor if factor != 1.0 else values


def _coerce_intensity_factors(intensity_factors):
    if intensity_factors is None:
        intensity_factors = DEFAULT_260726_INTENSITY_FACTORS
    factors = {
        str(name): {int(hour): float(value) for hour, value in by_hour.items()}
        for name, by_hour in intensity_factors.items()
    }
    unknown = sorted(set(factors) - set(MICROPATTERN_260726_SCHEMA.measurement_names))
    if unknown:
        raise ValueError("Unknown intensity factor channels: " + ", ".join(unknown))
    values = [value for by_hour in factors.values() for value in by_hour.values()]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("intensity factors must be finite and non-negative")
    return factors


def _percentile_from_histogram(histogram, percentile):
    if histogram.sum() == 0:
        raise ValueError("Cannot calculate percentiles from an empty histogram")
    threshold = percentile / 100.0 * histogram.sum()
    return float(np.searchsorted(np.cumsum(histogram), threshold, side="left"))


def _compute_histogram_bins(records, hist_eqs, schema, intensity_factors):
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
            values = _channel_values(
                image, record, channel, intensity_factors
            )
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


def colony_foreground(image, group):
    """Boolean ``[X, Y]`` map of colony pixels in one raw ``[X, Y, C]`` image.

    Thresholds the smoothed structural channel (or the channel mean) at its
    mean and fills holes. ``circular_colony_mask`` fits a circle to the result.
    """
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
    return ndi.binary_fill_holes(foreground)


def circular_colony_mask(
    foreground, group, boundary_radius_quantile=0.98, boundary_radius_scale=1.0
):
    """Fit a circular colony mask to a ``colony_foreground`` map.

    The centre is the median (mean for RNA) of the foreground pixels. The
    radius is the ``boundary_radius_quantile`` of their distances from the
    centre, which ignores stray pixels far away, times
    ``boundary_radius_scale``.
    """
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
    rows, columns = np.ogrid[: foreground.shape[0], : foreground.shape[1]]
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius**2


def _foreground_mask(image, group, boundary_radius_quantile, boundary_radius_scale):
    return circular_colony_mask(
        colony_foreground(image, group),
        group,
        boundary_radius_quantile,
        boundary_radius_scale,
    )


def read_micropattern_260726_image(record):
    """Read one raw image and its colony foreground, before any processing.

    ``record`` is a ``MicropatternImageRecord`` from
    ``build_micropattern_260726_manifest``. Returns ``(channels, foreground)``:
    float32 raw intensities ``[X, Y, M]`` for the measurement channels of
    ``record.group`` in schema order (channels used only for registration,
    such as RNA DAPI, are dropped), and the boolean ``[X, Y]`` output of
    ``colony_foreground``. No 0h scaling, normalisation, alignment or
    downsampling is applied.
    """
    raw = _read_multichannel_image(record.path, record.group)
    schema_group = next(
        item
        for item in MICROPATTERN_260726_SCHEMA.experiment_groups
        if item.name == record.group
    )
    channels = np.stack(
        [raw[..., channel.source_index] for channel in schema_group.channels],
        axis=-1,
    ).astype(np.float32)
    return channels, colony_foreground(raw, record.group)


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


def _normalisation_downsample(downsample):
    """Downsampling at which cleaned images are centred and normalised.

    4, the data cleaning notebook's, when the training downsampling is a
    multiple of 4, so the bounds tuned there apply unchanged; otherwise the
    training downsampling itself.
    """
    return 4 if downsample % 4 == 0 else downsample


def _schema_group(group):
    return next(item for item in MICROPATTERN_260726_SCHEMA.experiment_groups if item.name == group)


def _clean_image(record, intensity_factors, cleaning, work):
    """Cleaning steps 1 to 4 for one image: ``[X, Y, C]`` at downsampling ``work``.

    These are the slow steps, which ``cleaned_image`` can cache on disk.
    """
    schema_group = _schema_group(record.group)
    raw = _read_multichannel_image(record.path, record.group)
    names = [channel.name for channel in schema_group.channels]
    image = np.stack(
        [_channel_values(raw, record, channel, intensity_factors) for channel in schema_group.channels],
        axis=-1,
    )
    image, _ = replace_hot_pixels_per_channel(
        image,
        [cleaning.hot_pixel_thresholds.get(name, 0) for name in names],
        cleaning.hot_pixel_window,
    )
    image, _ = subtract_background(
        image,
        [cleaning.background_radii.get(name, 0) for name in names],
        cleaning.background_shrink,
    )
    image, _ = _downsample_image_and_mask(image, np.zeros(image.shape[:2], dtype=bool), work)
    return image.astype(np.float32)


# The code behind _clean_image; editing any of it starts a new cache folder.
_CLEANING_SOURCES = (
    hot_pixels,
    background,
    _read_multichannel_image,
    _channel_values,
    _downsample_image_and_mask,
    _schema_group,
    _clean_image,
)


def cleaning_cache_folder(cache_dir, root, intensity_factors, cleaning, work):
    """The ``disk_cache`` folder for images cleaned with these settings.

    Only the settings of cleaning steps 1 to 4 count, so one folder serves
    every selection of conditions and replicates, every alignment and every
    normalisation setting.
    """
    settings = {
        "dataset": str(Path(root).expanduser().resolve()),
        "downsample": int(work),
        "channels": {
            group.name: [[channel.name, channel.source_index] for channel in group.channels]
            for group in MICROPATTERN_260726_SCHEMA.experiment_groups
        },
        "intensity_factors": _coerce_intensity_factors(intensity_factors),
        "hot_pixel_thresholds": dict(cleaning.hot_pixel_thresholds),
        "hot_pixel_window": cleaning.hot_pixel_window,
        "background_radii": dict(cleaning.background_radii),
        "background_shrink": cleaning.background_shrink,
    }
    return cache_folder(cache_dir, f"micropattern_260726_ds{work}", settings, _CLEANING_SOURCES)


def cleaned_image(record, intensity_factors, cleaning, work, root, folder=None):
    """``_clean_image``, read from or saved to the cache ``folder`` when given."""
    intensity_factors = _coerce_intensity_factors(intensity_factors)
    if folder is None:
        return _clean_image(record, intensity_factors, cleaning, work)
    relative = Path(record.path).relative_to(Path(root).expanduser().resolve()).as_posix()
    return cached_array(folder, relative, lambda: _clean_image(record, intensity_factors, cleaning, work))


def _centre_image(image, record, cleaning, alignment, root, work):
    """Cleaning step 5: centre ``image`` with ``alignment`` and make its circular mask."""
    centre = grid_position(alignment.centre(Path(record.path).relative_to(root).as_posix()), work)
    target = ((image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0)
    image = ndi.shift(
        image,
        shift=(target[0] - centre[0], target[1] - centre[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    radius = alignment.pattern_radius * cleaning.mask_radius_scale / work
    rows, columns = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (rows - target[0]) ** 2 + (columns - target[1]) ** 2 <= radius**2
    return image.astype(np.float32), mask


def _trajectory_bounds(
    prepared,
    record_lookup,
    schema,
    trajectories,
    timesteps,
    substitute_preperturbation,
    hist_eqs,
    cleaning,
    excluded=frozenset(),
):
    """Clipping bounds ``[M, 2]`` for each trajectory ``(condition, replicate)``.

    The percentiles of each channel are taken over all timesteps of a
    trajectory (inside the mask with ``percentiles_inside_mask``) and shared
    between trajectories as set by ``cleaning.normalisation``. With
    ``knockouts_use_control_bounds``, a knockout trajectory takes its bounds
    from the control trajectory of the same replicate. Images in
    ``excluded`` (flagged as low quality) do not count. Channels a trajectory
    did not measure get NaN.
    """
    if cleaning.knockouts_use_control_bounds:
        sources = list(dict.fromkeys(
            [("ctrl", replicate) for _, replicate in trajectories] + list(trajectories)
        ))
    else:
        sources = list(trajectories)
    channel_of = {}
    for group, target_indices in zip(schema.experiment_groups, schema.group_measurement_indices):
        for channel_index, (target_index, channel) in enumerate(zip(target_indices, group.channels)):
            channel_of[target_index] = (group.name, channel_index, channel.name)

    bounds = {trajectory: np.full((schema.n_measurement_channels, 2), np.nan) for trajectory in trajectories}
    for target_index, (group, channel_index, name) in channel_of.items():
        samples = {}
        for condition, replicate in sources:
            values = []
            for timestep in timesteps:
                record = record_lookup.get(
                    (
                        source_condition(condition, timestep, substitute_preperturbation),
                        group,
                        timestep,
                        replicate,
                    )
                )
                if record is None or record in excluded:
                    continue
                image, mask = prepared(record)
                pixels = image[..., channel_index]
                values.append(pixels[mask] if cleaning.percentiles_inside_mask else pixels.ravel())
            if values:
                samples[(condition, replicate)] = np.concatenate(values)
        if not samples:
            continue
        reference = {}
        if cleaning.knockouts_use_control_bounds:
            reference = {
                trajectory: ("ctrl", trajectory[1])
                for trajectory in samples
                if trajectory[0] != "ctrl" and ("ctrl", trajectory[1]) in samples
            }
            # Only the requested trajectories and their references set the bounds.
            keep = set(trajectories) | set(reference.values())
            samples = {key: value for key, value in samples.items() if key in keep}
        low, high = cleaning.channel_percentiles.get(name, hist_eqs)
        for trajectory, (lower, upper) in percentile_bins(
            samples, low, high, cleaning.normalisation, reference=reference
        ).items():
            if trajectory in bounds:
                bounds[trajectory][target_index] = (lower, upper)
    return bounds


def load_micropattern_260726(
    root,
    conditions=("ctrl",),
    timesteps=(0, 12, 24, 36, 48),
    downsample=4,
    replicate_count=3,
    replicate_indices=None,
    replicate_manifest=None,
    experiment_groups=None,
    substitute_preperturbation=True,
    histogram_bins=None,
    hist_eqs=DEFAULT_260726_HISTOGRAM_PERCENTILES,
    align=True,
    strict_replicates=False,
    boundary_radius_quantile=0.98,
    boundary_radius_scale=1.0,
    pool_copies=1,
    intensity_factors=None,
    cleaning=None,
    excluded_images=None,
    cache_dir=None,
):
    """Load replicates of the multichannel 260726 micropattern dataset.

    ``boundary_radius_quantile`` drops foreground pixels far from the colony
    centre before fitting a circle; lower values or a
    ``boundary_radius_scale`` below one give a tighter common boundary.
    ``pool_copies`` repeats each replicate along the batch axis (the
    provenance in ``aux`` is repeated too). ``replicate_indices`` selects
    zero-based replicate slots, in batch order.
    ``intensity_factors`` maps measurement names (e.g.
    ``"cell_fate_s2/FOXA2"``) to ``{timestep: factor}``; each factor multiplies
    the raw intensities of that channel at that timestep, for every
    condition, before anything else, to correct imaging artifacts. Unlisted
    channels and timesteps are left unchanged. ``None`` uses
    ``DEFAULT_260726_INTENSITY_FACTORS``; pass ``{}`` for no correction.

    ``cleaning`` is a ``MicropatternCleaningConfig``
    (``Common/dataloader/micropattern_cleaning.py``). When it is enabled, the
    images are cleaned, centred with its alignment file and normalised per
    trajectory as described there; ``align``, ``boundary_radius_quantile``
    and ``boundary_radius_scale`` are then not used. ``histogram_bins`` can
    pass shared bounds from another load (the training replicates, or the
    control data for knockouts) only when the bounds are shared between
    trajectories. Without ``cleaning``, every image is clipped to percentiles
    pooled over all loaded images at full resolution and centred on a circle
    fitted to its colony pixels.

    ``excluded_images`` lists image paths relative to ``root`` (e.g.
    ``cell_fate_markers/ctrl_s1/A2_F14_24h.ome.tif``) flagged as low quality.
    Each flagged image is one experiment group at one timestep of one
    replicate, and is treated as not measured: its channels are False in
    ``measurement_mask`` (so losses and reinjection skip them) and it does
    not count towards the normalisation bounds. Its slot in the targets is
    filled with the mean of the same group and timestep over the other
    replicates of the same condition, so the trajectory still has a
    sensible initial state; ``aux["imputed"]`` marks these, and
    ``aux["excluded"]`` all flagged slots.

    ``cache_dir`` keeps the cleaned images on disk (see
    ``Common/dataloader/disk_cache.py``), so later loads with the same
    cleaning settings skip the slow cleaning steps 1 to 4. It is only used
    with ``cleaning`` enabled; ``None`` turns it off.

    Returns
    -------
    MicropatternDataset
        ``data``: float32 ``[B, T, M, X, Y]``, where ``M`` is the number of
        channels in the selected experiment groups and
        ``B = replicate_count * len(conditions) * pool_copies``.
        ``aux``: selected schema, provenance, group masks, histogram bins
        and file inventory.
        ``channel_names``: measurement names along the channel axis.
        ``boundary_mask``: boolean ``[B, 1, X, Y]``, from cell-fate S1 when
        selected, otherwise from the first selected group.
        ``measurement_mask``: boolean ``[B, T, M]``, True where measured
        (``measurement_mask[:, 1:]`` for one-step losses).
    """

    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if len(hist_eqs) != 2 or not 0 <= hist_eqs[0] < hist_eqs[1] <= 100:
        raise ValueError("hist_eqs must contain increasing percentiles in [0, 100]")
    if not 0.5 < boundary_radius_quantile <= 1.0:
        raise ValueError("boundary_radius_quantile must be in (0.5, 1.0]")
    if boundary_radius_scale <= 0:
        raise ValueError("boundary_radius_scale must be positive")
    if pool_copies <= 0 or int(pool_copies) != pool_copies:
        raise ValueError("pool_copies must be a positive integer")
    pool_copies = int(pool_copies)
    intensity_factors = _coerce_intensity_factors(intensity_factors)
    cleaned = cleaning is not None and cleaning.enabled
    conditions = tuple(conditions)
    timesteps = tuple(int(timestep) for timestep in timesteps)
    schema = MICROPATTERN_260726_SCHEMA.select_groups(experiment_groups)
    inventory = build_micropattern_260726_manifest(
        root=root,
        conditions=conditions,
        timesteps=timesteps,
        replicate_count=replicate_count,
        replicate_indices=replicate_indices,
        replicate_manifest=replicate_manifest,
        experiment_groups=schema.group_names,
        substitute_preperturbation=substitute_preperturbation,
    )
    selected = inventory["selected"]
    replicate_indices = inventory["replicate_indices"]
    replicate_count = len(replicate_indices)
    if strict_replicates:
        missing = []
        for condition in conditions:
            for timestep in timesteps:
                source = source_condition(
                    condition, timestep, substitute_preperturbation
                )
                for group in schema.group_names:
                    slots = selected[(source, group, timestep)]
                    for slot, path in enumerate(slots):
                        if path is None:
                            missing.append(
                                f"{condition}/{group}/{timestep}h/replicate-{replicate_indices[slot] + 1}"
                            )
        if missing:
            raise ValueError("Missing required measurements: " + ", ".join(missing))

    records = inventory["records"]
    dataset_root = Path(root).expanduser().resolve()
    excluded_names = set(excluded_images or ())
    excluded = frozenset(
        record
        for record in records
        if Path(record.path).relative_to(dataset_root).as_posix() in excluded_names
    )
    if cleaned and histogram_bins is not None and not cleaning.shares_bounds:
        raise ValueError(
            "histogram_bins cannot be reused with per-replicate normalisation; "
            "each trajectory sets its own bounds"
        )
    if histogram_bins is not None:
        histogram_bins = _coerce_histogram_bins(histogram_bins, schema)
    elif not cleaned:
        histogram_bins = _compute_histogram_bins(
            [record for record in records if record not in excluded],
            hist_eqs,
            schema,
            intensity_factors,
        )
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

    trajectories = [
        (condition, replicate) for condition in conditions for replicate in replicate_indices
    ]
    if cleaned:
        work = _normalisation_downsample(downsample)
        alignment = load_colony_alignment(cleaning.alignment_file)
        # Every image is cleaned once and kept at the normalisation resolution.
        cleaned_images = {}

        folder = (
            None
            if cache_dir is None
            else cleaning_cache_folder(cache_dir, dataset_root, intensity_factors, cleaning, work)
        )

        def prepared(record):
            if record not in cleaned_images:
                image = cleaned_image(record, intensity_factors, cleaning, work, dataset_root, folder)
                cleaned_images[record] = _centre_image(image, record, cleaning, alignment, dataset_root, work)
            return cleaned_images[record]

        if histogram_bins is None:
            trajectory_bounds = _trajectory_bounds(
                prepared,
                record_lookup,
                schema,
                trajectories,
                timesteps,
                substitute_preperturbation,
                hist_eqs,
                cleaning,
                excluded,
            )
        else:
            trajectory_bounds = {
                trajectory: np.asarray(histogram_bins, dtype=float) for trajectory in trajectories
            }
        if cleaning.shares_bounds:
            # The bounds every trajectory shares, for reuse by other loads.
            # Each channel's row comes from a trajectory that measured it.
            histogram_bins = np.full((schema.n_measurement_channels, 2), np.nan, dtype=np.float32)
            for bounds in trajectory_bounds.values():
                missing = np.isnan(histogram_bins[:, 0]) & ~np.isnan(bounds[:, 0])
                histogram_bins[missing] = bounds[missing]
        else:
            histogram_bins = None

    def finished(record, trajectory):
        """The image of ``record`` for ``trajectory`` at the training resolution."""
        image, mask = load_processed(
            record.path, record.condition, record.group, record.timestep, record.replicate
        )
        if not cleaned:
            return image, mask
        target_indices = group_target_indices[record.group]
        lower, upper = np.moveaxis(trajectory_bounds[trajectory][list(target_indices)], -1, 0)
        image = np.clip((image - lower) / np.maximum(upper - lower, 1e-6), 0.0, 1.0)
        image = (image * mask[..., None]).astype(np.float32)
        return _downsample_image_and_mask(image, mask, downsample // work)

    @lru_cache(maxsize=8)
    def load_processed(path, condition, group, timestep, replicate):
        record = MicropatternImageRecord(
            path, condition, group, timestep, replicate
        )
        if cleaned:
            return prepared(record)
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
            values = _channel_values(
                raw, record, channel, intensity_factors
            )
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
    first_image, first_mask = load_processed(
        first_record.path,
        first_record.condition,
        first_record.group,
        first_record.timestep,
        first_record.replicate,
    )
    if cleaned:
        first_image, _ = _downsample_image_and_mask(first_image, first_mask, downsample // work)
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
    excluded_slots = []
    for condition_index, condition in enumerate(conditions):
        for slot, replicate in enumerate(replicate_indices):
            batch = condition_index * replicate_count + slot
            batch_conditions.append(condition)
            batch_replicates.append(replicate + 1)
            for time_index, timestep in enumerate(timesteps):
                source = source_condition(
                    condition, timestep, substitute_preperturbation
                )
                for group in schema.group_names:
                    record = record_lookup.get(
                        (source, group, timestep, replicate)
                    )
                    if record is None:
                        continue
                    if record in excluded:
                        excluded_slots.append((batch, time_index, group))
                        source_conditions[batch, time_index, group_index[group]] = source
                        source_files[batch, time_index, group_index[group]] = record.path
                        continue
                    image, mask = finished(record, (condition, replicate))
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
                    source_conditions[batch, time_index, current_group] = source
                    substituted[batch, time_index, current_group] = (
                        source != condition
                    )
                    source_files[batch, time_index, current_group] = record.path

    # Flagged images stay unmeasured; their slots get the mean of the same
    # group and timestep over the other replicates of the condition.
    excluded_mask = np.zeros(group_mask.shape, dtype=bool)
    imputed = np.zeros(group_mask.shape, dtype=bool)
    for batch, time_index, group in excluded_slots:
        current_group = group_index[group]
        target_indices = list(group_target_indices[group])
        excluded_mask[batch, time_index, current_group] = True
        peers = [
            other
            for other in range(batch_count)
            if other != batch
            and batch_conditions[other] == batch_conditions[batch]
            and group_mask[other, time_index, current_group]
        ]
        if peers:
            targets[batch, time_index, target_indices] = targets[peers][:, time_index][
                :, target_indices
            ].mean(axis=0)
            imputed[batch, time_index, current_group] = True

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

    if pool_copies > 1:
        targets = np.concatenate([targets] * pool_copies, axis=0)
        measurement_mask = np.concatenate(
            [measurement_mask] * pool_copies, axis=0
        )
        boundary_mask = np.concatenate([boundary_mask] * pool_copies, axis=0)
        group_masks = np.concatenate([group_masks] * pool_copies, axis=0)
        group_mask = np.concatenate([group_mask] * pool_copies, axis=0)
        source_conditions = np.concatenate(
            [source_conditions] * pool_copies, axis=0
        )
        substituted = np.concatenate([substituted] * pool_copies, axis=0)
        source_files = np.concatenate([source_files] * pool_copies, axis=0)
        excluded_mask = np.concatenate([excluded_mask] * pool_copies, axis=0)
        imputed = np.concatenate([imputed] * pool_copies, axis=0)
        batch_conditions = batch_conditions * pool_copies
        batch_replicates = batch_replicates * pool_copies

    aux = {
        "channel_schema": schema,
        "selected_experiment_groups": schema.group_names,
        "pool_copies": pool_copies,
        "manifest": records,
        "unselected_files": inventory["unselected_files"],
        "histogram_bins": histogram_bins,
        "intensity_factors": intensity_factors,
        "cleaning": cleaning if cleaned else None,
        "excluded_images": tuple(
            sorted(Path(record.path).relative_to(dataset_root).as_posix() for record in excluded)
        ),
        "excluded": excluded_mask,
        "imputed": imputed,
        "pattern_radius": alignment.pattern_radius if cleaned else None,
        # Clipping bounds per trajectory, keyed by (condition, one-based replicate).
        "trajectory_bounds": (
            {
                (condition, replicate + 1): {
                    name: tuple(float(value) for value in trajectory_bounds[(condition, replicate)][index])
                    for index, name in enumerate(schema.measurement_names)
                }
                for condition, replicate in trajectories
            }
            if cleaned
            else None
        ),
        "group_boundary_masks": group_masks,
        "group_mask": group_mask,
        "source_conditions": source_conditions,
        "is_substituted": substituted,
        "source_files": source_files,
        "batch_conditions": tuple(batch_conditions),
        "batch_replicates": tuple(batch_replicates),
        "replicate_indices": replicate_indices,
        "timesteps": timesteps,
        "measurement_mask": measurement_mask,
        "loss_measurement_mask": measurement_mask[:, 1:],
    }
    return MicropatternDataset(
        data=jnp.asarray(targets),
        aux=aux,
        channel_names=tuple(schema.measurement_names),
        boundary_mask=jnp.asarray(boundary_mask),
        measurement_mask=jnp.asarray(measurement_mask),
    )
