"""The 250K-plating Nodal knockout colonies as a held-out test of the density × KO sweep.

``$DATA_PATH_BASE/density_nodal_ko_finetune/`` holds one colony per
projection and knockout hour (``<MAX|SUM|SLICE>_KO<hour>.tif``), plated at
250K cells instead of 150K, fixed at 48 h, with Nodal inhibited from the hour
in the file name. Each image is one 4-channel panel: SOX17, TBXT, FOXA2, LMBR
(the markers of ``cell_fate_s2``, in another order). There are no 0 h images
or controls, and the pixels are smaller than in 260726.

This module reads and cleans the colonies like the 260726 loader (with the
cleaning settings of a trained model), normalises them in one of
``NORMALISATIONS``, and compares their cell type shares with the density × KO
sweep at 250/150 times the measured 150K density. Only SOX17, TBXT, FOXA2
(and LMBR) are imaged, so cell types use the clauses those markers can check
(``stain_rules``), for the data and the model alike.

Used by the "held-out 250K colonies" section of ``nodal_ko_eval.py``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile

from Common.dataloader.alignment import coverage_map, fit_colony_centre, load_colony_alignment
from Common.dataloader.background import subtract_background
from Common.dataloader.cell_type_fit import radial_rings
from Common.dataloader.cell_type_shares import (
    colony_label_shares,
    pattern_counts,
    stain_counts_from_patterns,
)
from Common.dataloader.cell_types import CellTypeRules
from Common.dataloader.hot_pixels import replace_hot_pixels_per_channel
from Common.dataloader.micropattern_260726 import (
    _downsample_image_and_mask,
    _normalisation_downsample,
    colony_foreground,
)
from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from Common.dataloader.stripes import remove_stripes_per_channel

TEST_FOLDER = "density_nodal_ko_finetune"
# Cells per plate of the test colonies over the 260726 colonies
PLATING_SCALE = 250 / 150
# Channel order in the test files
FILE_MARKERS = ("SOX17", "TBXT", "FOXA2", "LMBR")
GROUP = "cell_fate_s2"
FILE_PREFIXES = {"max": "MAX", "sum": "SUM", "slice": "SLICE"}
_FILE_PATTERN = re.compile(r"^(MAX|SUM|SLICE)_KO(\d+)\.tif$", re.IGNORECASE)

NORMALISATIONS = {
    "joint": "Joint: one range per channel over the knockout times",
    "per_image": "Per image: each colony on its own range",
    "match_150k": "Matched to the 150K KO 24 h colonies at 48 h",
}


# ---------------------------------------------------------------- reading


def test_image_paths(folder, projection):
    """``{knockout hour: path}`` of the test colonies of ``projection``."""
    prefix = FILE_PREFIXES[projection]
    paths = {}
    for path in sorted(Path(folder).expanduser().glob("*.tif")):
        match = _FILE_PATTERN.match(path.name)
        if match and match.group(1).upper() == prefix:
            paths[int(match.group(2))] = path
    return dict(sorted(paths.items()))


def pixel_size_um(path):
    """Microns per pixel, from the TIFF resolution tag (pixels per micron)."""
    with tifffile.TiffFile(path) as tif:
        numerator, denominator = tif.pages[0].tags["XResolution"].value
    return denominator / numerator


def reference_pixel_size_um(dataset_root, projection):
    """Pixel size of the 260726 ``cell_fate_s2`` images of ``projection``."""
    folder = {"max": "max", "sum": "sums", "slice": "slices"}[projection]
    paths = sorted((Path(dataset_root).expanduser() / "cell_fate_markers").glob(f"ctrl_s2_{folder}/*.tif"))
    if not paths:
        raise FileNotFoundError(f"No 260726 cell_fate_s2 {projection} images under {dataset_root}")
    return pixel_size_um(paths[0])


def group_channels():
    """``(measurement names, source index)`` of the ``cell_fate_s2`` channels."""
    group = next(item for item in MICROPATTERN_260726_SCHEMA.experiment_groups if item.name == GROUP)
    return [(channel.name, channel.source_index) for channel in group.channels]


def read_test_image(path):
    """``[X, Y, C]`` float32 in the 260726 ``cell_fate_s2`` source channel order."""
    raw = tifffile.imread(path).astype(np.float32)  # [C, X, Y]
    channels = group_channels()
    image = np.zeros(raw.shape[1:] + (len(channels),), dtype=np.float32)
    for name, source_index in channels:
        image[..., source_index] = raw[FILE_MARKERS.index(name.split("/")[1])]
    return image


def rescale(image, from_um, to_um):
    """Resample ``image`` ``[X, Y, C]`` from ``from_um`` to ``to_um`` pixels."""
    factor = from_um / to_um
    if abs(factor - 1.0) < 1e-6:
        return image
    return ndi.zoom(image, (factor, factor, 1.0), order=1).astype(np.float32)


# ---------------------------------------------------------------- cleaning


def source_names():
    """Channel names of the test images, in source (array) order."""
    return [name for name, _ in sorted(group_channels(), key=lambda item: item[1])]


def apply_brightness(image, brightness, hour):
    """``image`` ``[X, Y, C]`` with each channel times its factor for ``hour``.

    ``brightness`` is ``{channel name: {hour: factor}}`` (missing = 1), like
    the 260726 intensity factors. It matters when one colony is dimmer than
    the others: a factor shared by every colony of a channel cancels out in
    the normalisation.
    """
    if not brightness:
        return image
    factors = np.array(
        [float(brightness.get(name, {}).get(hour, 1.0)) for name in source_names()], dtype=np.float32
    )
    return image * factors


@lru_cache(maxsize=16)
def _raw_image(path, target_um):
    """``read_test_image`` resampled to ``target_um`` pixels (cached; do not modify)."""
    return rescale(read_test_image(path), pixel_size_um(path), target_um)


@lru_cache(maxsize=512)
def cleaned_channel(
    path, target_um, name, factor, hot_threshold, hot_window, h_stripe, v_stripe, stripe_block, radius, shrink, work,
):
    """One channel of one test colony after brightness and cleaning, at downsampling ``work``.

    Returns ``(image [X, Y], share of pixels replaced as hot, share of each
    block replaced [X, Y])``. Cached per
    colony, channel and setting, so changing one control only cleans the
    channel it affects again.
    """
    index = source_names().index(name)
    image = _raw_image(str(path), float(target_um))[..., index : index + 1] * np.float32(factor)
    image, hot = replace_hot_pixels_per_channel(image, [hot_threshold], int(hot_window))
    image, _ = remove_stripes_per_channel(image, [h_stripe], [v_stripe], stripe_block)
    image, _ = subtract_background(image, [radius], shrink)
    empty = np.zeros(image.shape[:2], dtype=bool)
    image, _ = _downsample_image_and_mask(image, empty, work)
    hot_blocks, _ = _downsample_image_and_mask(hot.astype(np.float32), empty, work)
    return image[..., 0].astype(np.float32), float(hot.mean()), hot_blocks[..., 0]


@lru_cache(maxsize=16)
def colony_centre(path, target_um, pattern_radius, work):
    """Fitted colony centre on the ``work`` grid and the fit score (cached)."""
    coverage = coverage_map(colony_foreground(_raw_image(str(path), float(target_um)), GROUP), work)
    return fit_colony_centre(coverage, pattern_radius / work)


def centre_on(image, centre, radius):
    """``image`` ``[X, Y, C]`` shifted so ``centre`` is in the middle, and a
    circular mask of ``radius`` pixels around it."""
    target = ((image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0)
    image = ndi.shift(
        image, shift=(target[0] - centre[0], target[1] - centre[1], 0),
        order=1, mode="constant", cval=0.0, prefilter=False,
    )
    rows, columns = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (rows - target[0]) ** 2 + (columns - target[1]) ** 2 <= radius**2
    return image.astype(np.float32), mask


# ---------------------------------------------------------------- normalisation


def channel_percentiles(micropattern):
    """``{channel name: (low, high)}`` percentiles of the test channels, as in training."""
    return {
        name: tuple(micropattern.cleaning.channel_percentiles.get(name, micropattern.histogram_percentiles))
        for name, _ in group_channels()
    }


def _bounds(pixels, low, high):
    lower, upper = np.percentile(pixels, [low, high])
    return float(lower), float(max(upper, lower + 1e-6))


def normalise(images, masks, percentiles, mode, reference=None):
    """Scale ``images`` ``[K, X, Y, C]`` to [0, 1] per channel (source order).

    ``percentiles`` gives each channel's low and high percentile. ``joint`` takes one range per channel from every colony's pixels inside
    its mask; ``per_image`` one range per colony. ``match_150k`` maps the
    joint range's percentiles onto ``reference`` (``{channel name: (low,
    high)}``: the same percentiles of the normalised 150K images), keeping
    the differences between knockout times. Returns the images and the
    ranges used ``{channel name: [(low, high) per colony]}``.
    """
    if mode not in NORMALISATIONS:
        raise ValueError(f"Unknown normalisation {mode!r}; use one of {list(NORMALISATIONS)}")
    images = np.asarray(images, dtype=np.float32).copy()
    ranges = {}
    for name, index in group_channels():
        low, high = percentiles[name]
        values = [image[..., index][mask] for image, mask in zip(images, masks)]
        if mode == "per_image":
            bounds = [_bounds(value, low, high) for value in values]
        else:
            bounds = [_bounds(np.concatenate(values), low, high)] * len(images)
        ranges[name] = bounds
        for k, (lower, upper) in enumerate(bounds):
            scaled = (images[k, ..., index] - lower) / (upper - lower)
            if mode == "match_150k":
                target_low, target_high = reference[name]
                scaled = target_low + scaled * (target_high - target_low)
            images[k, ..., index] = np.clip(scaled, 0.0, 1.0)
    return images, ranges


def reference_ranges(reference_images, reference_masks, channel_names, percentiles):
    """``{channel name: (low, high)}`` percentiles of normalised 150K colonies,
    and ``{channel name: the channel they were read from}``.

    ``reference_images`` ``[N, M, H, W]`` are named by ``channel_names`` (e.g.
    the 48 h images of the KO 24 h condition from
    ``density_ko_sweep.load_condition_data``). A ``cell_fate_s2`` channel the
    models leave out (``cell_fate_s2/SOX17``) is read from the same marker in
    another panel.
    """
    channel_names = list(channel_names)
    ranges, sources = {}, {}
    for name, _ in group_channels():
        marker = name.split("/")[1]
        source = name if name in channel_names else next(
            (other for other in channel_names if other.split("/")[-1] == marker), None
        )
        if source is None:
            raise ValueError(f"No 150K channel shows {marker} to match {name} to")
        low, high = percentiles[name]
        pixels = np.concatenate([
            np.asarray(image[channel_names.index(source)])[np.asarray(mask, dtype=bool)]
            for image, mask in zip(reference_images, reference_masks)
        ])
        ranges[name] = tuple(float(value) for value in np.percentile(pixels, [low, high]))
        sources[name] = source
    return ranges, sources


# ---------------------------------------------------------------- whole pipeline


def process_test_images(
    paths, target_um, cleaning, percentiles, pattern_radius, work, mode, reference=None, downsample=None,
    brightness=None,
):
    """Clean, centre and normalise the test colonies at ``paths`` (``{hour: path}``).

    ``cleaning`` has the fields of ``MicropatternCleaningConfig`` that cleaning
    and the mask use; ``percentiles`` is ``{channel name: (low, high)}``.
    Images are resampled to ``target_um`` pixels, cleaned and centred at
    downsampling ``work`` and, with ``downsample``, block-averaged to that
    downsampling. ``brightness`` (``apply_brightness``) scales channels of
    single colonies before cleaning. Returns ``{"hours", "images" [K, C, H,
    W] (``cell_fate_s2`` order, zero outside the colony), "mask" [K, H, W],
    "channel_names", "fit_scores", "ranges", "hot_shares" [K, C], "hot_maps"
    [K, C, H, W]}``.
    """
    cleaned, masks, scores, hot_shares, hot_maps = [], [], [], [], []
    for hour, path in paths.items():
        channels = [
            cleaned_channel(
                str(path), float(target_um), name,
                float((brightness or {}).get(name, {}).get(hour, 1.0)),
                float(cleaning.hot_pixel_thresholds.get(name, 0)), int(cleaning.hot_pixel_window),
                int(cleaning.horizontal_stripe_widths.get(name, 0)),
                int(cleaning.vertical_stripe_widths.get(name, 0)), int(cleaning.stripe_block),
                int(cleaning.background_radii.get(name, 0)), int(cleaning.background_shrink), int(work),
            )
            for name in source_names()
        ]
        centre, score = colony_centre(str(path), float(target_um), float(pattern_radius), int(work))
        # Images and hot pixel maps are centred together
        image, mask = centre_on(
            np.stack([channel[0] for channel in channels] + [channel[2] for channel in channels], axis=-1),
            centre,
            pattern_radius * cleaning.mask_radius_scale / work,
        )
        cleaned.append(image[..., : len(channels)])
        hot_maps.append(image[..., len(channels) :])
        masks.append(mask)
        scores.append(score)
        hot_shares.append([channel[1] for channel in channels])
    normalised, ranges = normalise(np.stack(cleaned), masks, percentiles, mode, reference)
    factor = 1 if downsample is None else int(downsample) // work
    images, final_masks, final_hot = [], [], []
    for image, mask, hot in zip(normalised, masks, hot_maps):
        image = np.where(mask[..., None], image, 0.0)
        both, final_mask = _downsample_image_and_mask(np.concatenate([image, hot], axis=-1), mask, factor)
        images.append(np.moveaxis(both[..., : image.shape[-1]], -1, 0))
        final_hot.append(np.moveaxis(both[..., image.shape[-1] :], -1, 0))
        final_masks.append(final_mask)
    names = source_names()
    return {
        "hours": np.asarray(list(paths), dtype=float),
        "images": np.stack(images).astype(np.float32),
        "mask": np.stack(final_masks),
        "channel_names": tuple(names),
        "fit_scores": np.asarray(scores),
        "ranges": ranges,
        "hot_shares": np.asarray(hot_shares),
        # [K, C, H, W]: share of each block replaced as hot
        "hot_maps": np.stack(final_hot).astype(np.float32),
    }


def prepare_test_colonies(folder, projection, model_config, dataset_root, mode, reference=None):
    """The test colonies of ``projection``, prepared like ``model_config``'s data
    (``process_test_images`` at the model's downsampling)."""
    paths = test_image_paths(folder, projection)
    if not paths:
        raise FileNotFoundError(f"No {FILE_PREFIXES[projection]}_KO<hour>.tif files in {folder}")
    micropattern = model_config.data.micropattern
    cleaning = micropattern.cleaning
    if not cleaning.alignment_file:
        raise ValueError("The model's cleaning settings have no alignment file (pattern radius)")
    from Experiments.notebook_jobs import repository_path

    pattern_radius = load_colony_alignment(repository_path(cleaning.alignment_file)).pattern_radius
    downsample = int(model_config.data.downsample)
    target_um = reference_pixel_size_um(dataset_root, projection)
    colonies = process_test_images(
        paths, target_um, cleaning, channel_percentiles(micropattern), pattern_radius,
        _normalisation_downsample(downsample), mode, reference, downsample,
    )
    return {**colonies, "pixel_um": target_um * downsample}


# ---------------------------------------------------------------- cell types


def stain_rules(rules, markers=FILE_MARKERS, group=GROUP):
    """``rules`` restricted to the markers imaged in the test panel.

    Each rule marker among ``markers`` is read from the test image's own
    channel (``<group>/<marker>``), with the rule's threshold. Clauses that
    need another marker (anything but "any") are left out, as
    ``classify_within_stain`` does, and cell types left without clauses are
    dropped. Returns the rules, the left-out ``(cell type, clause)`` pairs,
    and ``{marker: rule channel}`` for markers whose threshold was tuned on
    another channel than the one read here.
    """
    imaged = [marker for marker in rules.markers if marker in markers]
    channels = {marker: f"{group}/{marker}" for marker in imaged}
    borrowed = {
        marker: rules.channels[marker] for marker in imaged if rules.channels[marker] != channels[marker]
    }
    kept, weights, dropped = {}, {}, []
    for cell_type, clauses in rules.rules.items():
        own = []
        for clause, weight in zip(clauses, rules.weights_of(cell_type)):
            if all(marker in channels for marker, state in clause.items() if state != "any"):
                own.append(({m: s for m, s in clause.items() if m in channels}, weight))
            else:
                dropped.append((cell_type, dict(clause)))
        if own and any(weight > 0 for _, weight in own):
            kept[cell_type] = [clause for clause, _ in own]
            weights[cell_type] = [weight for _, weight in own]
    if not kept:
        raise ValueError("No cell type rule can be checked with the imaged markers " + ", ".join(markers))
    restricted = CellTypeRules(
        channels=channels,
        thresholds={marker: rules.thresholds[marker] for marker in imaged},
        rules=kept,
        hour=rules.hour,
        preprocessing=rules.preprocessing,
        stains={group: channels},
        clause_weights=weights,
    )
    return restricted, dropped, borrowed


def panel_label_shares(values, mask, restricted, n_rings, resolve_several=0.0):
    """Label shares ``[labels]`` of one panel ``{marker: image}`` with ``restricted``
    rules (``stain_rules``), estimated per ring as for the measured data."""
    rings = radial_rings(np.asarray(mask, dtype=bool), n_rings).ravel()
    markers = restricted.markers
    counts = pattern_counts(
        np.stack([np.ravel(values[marker]) for marker in markers]),
        [restricted.thresholds[marker] for marker in markers],
        rings,
        n_rings,
    )
    return colony_label_shares([(markers, counts)], restricted, resolve_several=resolve_several)


def model_stain_prevalence(pattern_counts, markers, restricted, resolve_several=0.0):
    """Label shares ``[..., labels]`` of sweep counts ``[..., rings, patterns]``
    over ``markers``, with only the markers of ``restricted`` (``stain_rules``)."""
    counts = np.asarray(pattern_counts, dtype=float)
    flat = counts.reshape(-1, *counts.shape[-2:])
    shares = [
        colony_label_shares(
            stain_counts_from_patterns(colony, markers, [restricted.markers]),
            restricted,
            resolve_several=resolve_several,
        )
        for colony in flat
    ]
    return np.asarray(shares).reshape(*counts.shape[:-2], -1)


def model_prevalence_rows(results, restricted, group_by, resolve_several=0.0):
    """One row per run, density, KO time and cell type of density × KO sweep
    ``results``: the share with ``restricted`` rules (``model_stain_prevalence``).

    ``density`` is the density the run actually used, in its ``density_unit``;
    ``group`` is the run's group under ``group_by`` (as in the notebook).
    """
    from Common.dataloader.cell_types import label_names
    from Experiments.micropatterns.density_ko_sweep import actual_densities

    labels = label_names(restricted)
    rows = []
    for run, result in enumerate(results):
        grid = model_stain_prevalence(
            result["pattern_counts"], result["cell_type_markers"], restricted, resolve_several=resolve_several,
        )
        for density_index, (requested, density) in enumerate(zip(result["densities"], actual_densities(result))):
            for ko_index, hour in enumerate(result["requested_ko_hours"]):
                rows.extend(
                    {
                        "group": result["groups"][group_by],
                        "model_id": result["model_id"],
                        "run": run,
                        "density_index": density_index,
                        "requested_density": float(requested),
                        "density": float(density),
                        "knockout_hour": float(hour),
                        "cell_type": label,
                        "prevalence": float(share),
                    }
                    for label, share in zip(labels, grid[density_index, ko_index])
                )
    return rows


def test_density(results):
    """The test density in the sweep's unit: ``PLATING_SCALE`` times the measured 150K density."""
    unit = results[0]["density_unit"]
    if unit == "share":
        raise ValueError(
            "The plating scale applies to cell numbers; run the sweep in cells per mm² or per colony"
        )
    return PLATING_SCALE * float(np.mean([result["measured_density"] for result in results]))


def predict_at(grid, densities, ko_hours, density, hour):
    """Linear interpolation of ``grid`` ``[densities, KO times, ...]`` at one
    point; NaN outside the swept range."""
    densities = np.asarray(densities, dtype=float)
    ko_hours = np.asarray(ko_hours, dtype=float)
    if not (densities.min() <= density <= densities.max() and ko_hours.min() <= hour <= ko_hours.max()):
        return np.full(np.asarray(grid).shape[2:], np.nan)
    order = np.argsort(densities, kind="stable")
    densities, grid = densities[order], np.asarray(grid, dtype=float)[order]

    def along(values, points, x):
        upper = int(np.clip(np.searchsorted(points, x, side="right"), 1, len(points) - 1))
        lower = upper - 1
        span = points[upper] - points[lower]
        weight = 0.0 if span == 0 else (x - points[lower]) / span
        return (1 - weight) * values[lower] + weight * values[upper]

    if len(ko_hours) == 1:
        by_density = grid[:, 0]
    else:
        by_density = np.stack([along(row, ko_hours, hour) for row in grid])
    return by_density[0] if len(densities) == 1 else along(by_density, densities, density)


# ---------------------------------------------------------------- threshold tuning


def test_point_states(bundle, settings, points, root, max_batch=64):
    """Final model states at chosen points of the density × KO map.

    ``points`` lists ``(label, density scale, knockout hour or None)``: the
    density is the scale times the measured 150K density (mean over
    ``settings.replicates``), and None means no knockout. Cells are placed
    as in the density × KO sweep (``settings`` is its ``DensityKoSettings``),
    once per replicate and run, and every point of the same density uses the
    same cells and seed. Unlike the sweep, the states are kept, so cell types
    can be counted with any thresholds.

    Returns ``{"labels", "densities_per_mm2" [points], "states" [points,
    samples, C, H, W] (measured state channels), "names", "mask"}``.
    """
    import jax.numpy as jnp
    import jax.random as jr

    from Experiments.micropatterns.density_ko_sweep import circle_mask, prepare_model_cells
    from Experiments.micropatterns.synthetic_ic import cell_states
    from NCA.trainer.intervention import rollout_batch_with_blocked_channel_final

    projection = bundle.config.data.micropattern.projection
    prepared = prepare_model_cells(bundle, settings, projection, Path(root))
    colonies = prepared["colonies"]
    shape = colonies["support"].shape[-2:]
    mask = circle_mask(shape, float(np.mean(np.sum(colonies["support"], axis=(1, 2)))))
    mask_mm2 = float(np.count_nonzero(mask)) * prepared["pixel_um"] ** 2 * 1e-6
    measured = float(np.nanmean([
        replicate["cell_counts"]["cells_per_mm2"] for replicate in prepared["replicates"]
    ]))
    if not np.isfinite(measured):
        raise ValueError("No measured 150K density: the nuclei of the 0h colonies are not counted")
    model = bundle.load_model(key=jr.PRNGKey(int(settings.seed)), implementation="portable")
    channels = int(model.N_CHANNELS)
    names = prepared["names"]
    steps_per_12h = int(bundle.config.run.t)
    total_steps = int(round(float(settings.snapshot_hour) * steps_per_12h / 12.0))
    boundary = jnp.broadcast_to(jnp.asarray(mask, dtype=prepared["boundary"].dtype), prepared["boundary"].shape)
    runs = int(settings.duplicate_runs)

    # Initial states of every (replicate, run) at each density used
    initial, seeds = {}, []
    for scale in sorted({float(scale) for _, scale, _ in points}):
        states = []
        for index, replicate in enumerate(prepared["replicates"]):
            for run in range(runs):
                seed = int(settings.seed) + index * runs + run
                placed = cell_states(
                    replicate["library"], colonies["schema"], mask, np.asarray([scale * measured * mask_mm2]),
                    False, seed, prepared["cell_settings"], replicate["per_piece"],
                )
                states.append(placed["initial_states"][0])
                if len(seeds) < len(prepared["replicates"]) * runs:
                    seeds.append(seed)
        stacked = np.stack(states)
        initial[scale] = np.pad(stacked, ((0, 0), (0, channels - stacked.shape[1]), (0, 0), (0, 0)))
    keys = jnp.stack([jr.PRNGKey(seed) for seed in seeds])

    finals = []
    for _, scale, hour in points:
        # Past the end: never blocked
        ko_step = total_steps + 1 if hour is None else int(round(float(hour) * steps_per_12h / 12.0))
        states = initial[float(scale)]
        out = []
        for start in range(0, len(states), max_batch):
            out.append(np.asarray(rollout_batch_with_blocked_channel_final(
                model, jnp.asarray(states[start : start + max_batch]), boundary,
                bundle.config.trainer.boundary_mode, keys[start : start + max_batch], total_steps,
                prepared["nodal_channel"], jnp.asarray(ko_step, dtype=jnp.int32),
            ))[:, : len(names)])
        finals.append(np.concatenate(out))
    return {
        "labels": [label for label, _, _ in points],
        "densities_per_mm2": np.asarray([float(scale) * measured for _, scale, _ in points]),
        "states": np.stack(finals).astype(np.float32),
        "names": tuple(names),
        "mask": mask,
    }


def measured_150k(bundle, root, conditions, replicates):
    """The 150K colonies at the last time point, normalised as for training but
    with every channel kept (the models leave out ``cell_fate_s2/SOX17``, which
    the S2 panel needs).

    Returns ``{condition: [(image [M, H, W], mask [H, W], channel names), ...]}``
    for the replicates whose ``cell_fate_s2`` panel was measured then.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from Experiments.micropatterns.density_ko_sweep import load_condition_data, load_training_histogram_bins

    micropattern = replace(
        bundle.config.data.micropattern,
        excluded_measurements=(),
        data_channels=MICROPATTERN_260726_SCHEMA.n_measurement_channels,
    )
    config = replace(bundle.config, data=replace(bundle.config.data, micropattern=micropattern))
    every_channel = SimpleNamespace(id=bundle.id, config=config)
    bins = load_training_histogram_bins(every_channel, Path(root))
    found = {}
    for condition in conditions:
        for replicate in replicates:
            data = load_condition_data(every_channel, Path(root), condition, replicate, bins)
            names = list(data["channel_names"])
            panel = [names.index(name) for name, _ in group_channels()]
            if not np.asarray(data["measured"])[-1][panel].all():
                continue
            found.setdefault(condition, []).append(
                (np.asarray(data["target"])[-1], np.asarray(data["boundary"]).squeeze() > 0, names)
            )
    return found


def shares_with(images, names, masks, rules, thresholds, n_rings, resolve_several=0.0):
    """Label shares ``[N, labels]`` of ``images`` ``[N, C, H, W]`` with ``rules``
    restricted to the test panel (``stain_rules``) and ``thresholds``
    (``{marker: threshold}``, others unchanged).

    A marker is read from the channel ``cell_fate_s2/<marker>`` or, for
    model states, ``<marker>``. ``masks`` is one mask or one per image.
    """
    from dataclasses import replace

    tuned = replace(rules, thresholds={**rules.thresholds, **thresholds})
    restricted, _, _ = stain_rules(tuned)
    names = list(names)
    masks = [masks] * len(images) if np.ndim(masks) == 2 else list(masks)
    shares = []
    for image, mask in zip(images, masks):
        values = {}
        for marker, channel in restricted.channels.items():
            index = names.index(channel) if channel in names else names.index(marker)
            values[marker] = np.asarray(image[index])
        shares.append(panel_label_shares(values, mask, restricted, n_rings, resolve_several))
    return np.asarray(shares), restricted


TEST_POINT_ARRAYS = ("states", "mask", "densities_per_mm2")


def save_test_points(path, results):
    """Save ``test_point_states`` results (one dict per model, with ``model_id``)."""
    from Experiments.notebook_jobs import save_results_npz

    return save_results_npz(path, results, TEST_POINT_ARRAYS)


def load_test_points(path):
    from Experiments.notebook_jobs import load_results_npz

    return load_results_npz(path, TEST_POINT_ARRAYS)
