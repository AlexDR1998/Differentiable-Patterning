"""Cell numbers of real and synthetic 0 h colonies.

Synthetic density initial conditions (``NCA/inverse_design/density.py``)
are set by the share of the colony covered by placed pieces. Experiments
report cell numbers, so this module counts cells and gives the conversion.

Terms:

- **cell**: a real cell, counted by its nucleus (one nucleus per cell).
- **segment**: a piece ``cell_library`` cuts out of a model-resolution
  colony. At model resolution touching nuclei merge, so a segment holds
  ``cells_per_segment`` cells on average (often more than one).

The pieces placed on a synthetic colony come from one of two libraries:

- ``cell_library`` (``reference.py``): segments. Cell numbers are segments
  times ``cells_per_segment``, measured on the same real colonies
  (``reference_cell_counts``, ``summarise_counts``).
- ``nuclear_cell_library``: whole cells, found from nuclei at full
  resolution and shrunk to model resolution, so ``cells_per_segment`` is 1.

Notes:

- Nuclei are found in the nuclear stain: LMBR (cell-fate and protein
  panels) or DAPI (RNA panel); LMBR is taken to outline each nucleus.
- The colonies are 500 µm across (``PATTERN_DIAMETER_UM``), so the pixel
  size follows from the alignment file's ``pattern_radius``.
- Cell-fate and protein images are projections of z stacks, so nuclei
  stacked in z are hidden: counts are a lower bound.
- A placed piece takes one piece from each staining panel, so
  ``cells_per_segment`` is the mean over the panels with cell markers.

Used by ``boundary_mask_explorer.py`` and ``nodal_ko_eval.py``.
"""

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from Experiments.micropatterns.texture_synthesis import cell_coverage, otsu_threshold, segment_cells

PATTERN_DIAMETER_UM = 500.0
# Typical nucleus diameter of the colony cells; sets the smoothing and the
# smallest distance between two nuclei
NUCLEUS_DIAMETER_UM = 8.0
# How far a cell reaches beyond its nucleus (nuclear cell library)
CELL_EXPANSION_UM = 2.0
# Touching nuclei are split at seeds at least this share of a nucleus apart
NUCLEUS_SEED_SPACING = 0.5
# The nuclear stain threshold, as a multiple of its Otsu threshold
NUCLEUS_THRESHOLD = 1.0
# How touching nuclei are told apart (count_nuclei)
NUCLEUS_SPLIT_METHODS = ("distance", "rim", "intensity")
NUCLEUS_SPLIT_METHOD = "distance"
# Nuclei less round than this (solidity: area over convex hull area) are split
# again; 0: never. Nuclei are round, so pinched shapes are usually merged ones.
NUCLEUS_ROUNDNESS = 0.0
# Nucleus size limits, as multiples of the area of the set nucleus diameter:
# smaller pieces are dropped (debris); larger ones are split into about
# area / nucleus area pieces (clumps; 0: never)
NUCLEUS_SMALLEST = 0.1
NUCLEUS_LARGEST = 0.0
# Default alignment file of the 260726 dataset (repository relative)
DEFAULT_ALIGNMENT_FILE = "Experiments/micropatterns/conf/alignment/260726_alignment.yaml"


def default_alignment():
    """The 260726 alignment file of the repository (``DEFAULT_ALIGNMENT_FILE``)."""

    from Common.dataloader.alignment import load_colony_alignment

    return load_colony_alignment(Path(__file__).resolve().parents[2] / DEFAULT_ALIGNMENT_FILE)


def um_per_pixel(pattern_radius, downsample=1):
    """Pixel size in µm, for a pattern radius in raw pixels and a downsampling."""

    return PATTERN_DIAMETER_UM / (2.0 * float(pattern_radius)) * downsample


# ---------------------------------------------------------------------------
# Nuclei at full resolution


def count_nuclei(channel, mask, nucleus_diameter_px, seed_spacing=NUCLEUS_SEED_SPACING,
                 threshold_scale=NUCLEUS_THRESHOLD, split_method=NUCLEUS_SPLIT_METHOD,
                 roundness=NUCLEUS_ROUNDNESS, smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST):
    """Nuclei in a raw nuclear stain ``[X, Y]`` inside ``mask``.

    The stain is smoothed and thresholded (``threshold_scale`` times its Otsu
    threshold inside the mask), and holes are filled, giving blobs of one or
    more nuclei. Each blob is split by a watershed from seeds at least
    ``seed_spacing`` nuclei apart, placed by ``split_method``:

    - ``distance``: the maxima of the distance to the background, so blobs
      split where they narrow
    - ``rim``: for stains of the nuclear rim (LMBR), the dark interior of
      each nucleus, enclosed by its rim in a lightly smoothed stain (enclosed
      regions under a tenth of a nucleus, such as gaps between nuclei, are
      left out). The watershed follows the stain, so touching nuclei split
      along their shared rim.
    - ``intensity``: the maxima of the more smoothed stain, for filled
      nuclear stains (DAPI), whose touching nuclei are dimmer between them

    Blobs given no seed get ``distance`` seeds. Then nuclei less round than
    ``roundness`` (solidity; 0: none) are split again at their own distance
    maxima, half as far apart, if no piece is under a quarter of a nucleus.
    Nuclei larger than ``largest`` nuclei (clumps; 0: none) are split into
    ``round(area / nucleus area)`` pieces (at least 2) by k-means of their
    pixel positions, so the pieces are about equal and round. Last, pieces
    smaller than ``smallest`` nuclei (debris) are dropped. Returns ``(count,
    labels [X, Y], mean nuclear area in px)``.
    """

    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    if split_method not in NUCLEUS_SPLIT_METHODS:
        raise ValueError(f"Unknown split method {split_method!r}; use one of {NUCLEUS_SPLIT_METHODS}")
    mask = np.asarray(mask, dtype=bool)
    channel = np.asarray(channel, dtype=np.float32)
    diameter = float(nucleus_diameter_px)
    nucleus_area = np.pi * (diameter / 2.0) ** 2
    spacing = max(int(round(float(seed_spacing) * diameter)), 1)
    smooth = ndi.gaussian_filter(channel, sigma=max(0.15 * diameter, 0.5))
    threshold = float(threshold_scale) * otsu_threshold(smooth[mask])
    stained = (smooth > threshold) & mask
    foreground = ndi.binary_fill_holes(stained)
    distance = ndi.distance_transform_edt(foreground)
    patches, _ = ndi.label(foreground)

    def _seeds(peaks, shape, first=1):
        seeds = np.zeros(shape, dtype=np.int32)
        seeds[tuple(np.asarray(peaks).reshape(-1, 2).T)] = np.arange(first, first + len(peaks))
        return seeds

    if split_method == "rim":
        # Interiors from a lightly smoothed stain: the usual smoothing blurs the thin rim into them
        light = ndi.gaussian_filter(channel, sigma=max(0.05 * diameter, 0.5))
        rim = (light > float(threshold_scale) * otsu_threshold(light[mask])) & mask
        interiors, _ = ndi.label(ndi.binary_fill_holes(rim) & ~rim & foreground)
        sizes = np.bincount(interiors.ravel())
        large = np.flatnonzero(sizes >= 0.1 * nucleus_area)
        large = large[large > 0]
        relabel = np.zeros(len(sizes), dtype=np.int32)
        relabel[large] = np.arange(1, len(large) + 1)
        seeds = relabel[interiors]
        elevation = smooth
    elif split_method == "intensity":
        bright = ndi.gaussian_filter(channel, sigma=max(0.25 * diameter, 0.5))
        peaks = peak_local_max(bright, min_distance=spacing, labels=patches, exclude_border=False)
        seeds = _seeds(peaks, foreground.shape)
        elevation = -bright
    else:
        peaks = peak_local_max(distance, min_distance=spacing, labels=patches, exclude_border=False)
        seeds = _seeds(peaks, foreground.shape)
        elevation = -distance
    # Blobs with no seed (no closed rim, or a flat stain): seeds at the distance maxima
    unseeded = np.setdiff1d(np.unique(patches[foreground]), np.unique(patches[seeds > 0]))
    if len(unseeded):
        peaks = peak_local_max(
            distance, min_distance=spacing, labels=np.where(np.isin(patches, unseeded), patches, 0),
            exclude_border=False,
        )
        extra = _seeds(peaks, foreground.shape, first=int(seeds.max()) + 1)
        seeds = np.where(extra > 0, extra, seeds)
    labels = watershed(elevation, seeds, mask=foreground)

    # Nuclei are round: split pinched ones again along their own shape
    if roundness > 0:
        next_label = int(labels.max()) + 1
        for region in regionprops(labels):
            if region.solidity >= roundness or region.area < 0.5 * nucleus_area:
                continue
            inside = region.image
            local = ndi.distance_transform_edt(inside)
            peaks = peak_local_max(
                local, min_distance=max(spacing // 2, 1), labels=inside.astype(np.int32), exclude_border=False,
            )
            if len(peaks) < 2:
                continue
            parts = watershed(-local, _seeds(peaks, inside.shape), mask=inside)
            if np.bincount(parts[inside])[1:].min() < 0.25 * nucleus_area:
                continue
            box = labels[region.slice]
            box[inside] = parts[inside] + next_label - 1
            next_label += len(peaks)

    # Clumps: split into about as many pieces as nuclei would fill them
    if largest > 0:
        from scipy.cluster.vq import kmeans2

        next_label = int(labels.max()) + 1
        for region in regionprops(labels):
            if region.area <= largest * nucleus_area:
                continue
            pieces = max(2, int(round(region.area / nucleus_area)))
            positions = region.coords.astype(np.float64)
            _, part = kmeans2(positions, pieces, minit="++", seed=0)
            labels[tuple(region.coords.T)] = part + next_label
            next_label += pieces

    areas = np.bincount(labels.ravel())[1:]
    kept = np.flatnonzero((areas > 0) & (areas >= smallest * nucleus_area)) + 1
    # Kept nuclei renumbered 1 .. count
    renumber = np.zeros(len(areas) + 1, dtype=np.int32)
    renumber[kept] = np.arange(1, len(kept) + 1)
    labels = renumber[labels]
    mean_area = float(np.mean(areas[kept - 1])) if len(kept) else 0.0
    return len(kept), labels, mean_area


def _circle(shape, centre, radius):
    rows, columns = np.ogrid[: shape[0], : shape[1]]
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius**2


def colony_mask(path, dataset_root, alignment, shape, mask_radius_scale=1.0):
    """The aligned circular colony ``[X, Y]`` (bool) of one raw image of ``shape``, as ``colony_nuclei`` uses it."""

    root = Path(dataset_root).expanduser().resolve()
    relative = Path(path).resolve().relative_to(root).as_posix()
    return _circle(shape, alignment.centre(relative), alignment.pattern_radius * mask_radius_scale)


def colony_nuclei(path, group, dataset_root, alignment, mask_radius_scale=1.0,
                  nucleus_diameter_um=NUCLEUS_DIAMETER_UM, cache_dir=None,
                  seed_spacing=NUCLEUS_SEED_SPACING, threshold_scale=NUCLEUS_THRESHOLD,
                  split_method=NUCLEUS_SPLIT_METHOD, roundness=NUCLEUS_ROUNDNESS,
                  smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST, segmentation=None):
    """``(nucleus labels [X, Y], colony mask [X, Y], µm per pixel)`` of one raw image.

    Nuclei are found inside the aligned circular colony (centre from
    ``alignment``, radius ``pattern_radius * mask_radius_scale``), as the
    loader masks it. With ``segmentation`` (a stored method of
    ``nucleus_segmentation.py``) its stored labels are used when it has this
    image, keeping the nuclei centred in the colony; otherwise, and without
    it, nuclei are found by ``count_nuclei``. With ``cache_dir`` those labels
    are kept on disk.
    """

    from Common.dataloader.disk_cache import cache_folder, cached_array
    from Common.dataloader.micropattern_260726 import read_nuclear_channel
    from Common.dataloader.micropattern_projections import image_key

    root = Path(dataset_root).expanduser().resolve()
    relative = Path(path).resolve().relative_to(root).as_posix()
    centre = alignment.centre(relative)
    radius = alignment.pattern_radius * mask_radius_scale
    pixel_um = um_per_pixel(alignment.pattern_radius)

    if segmentation:
        from Experiments.micropatterns.nucleus_segmentation import relabel_inside, stored_labels

        stored = stored_labels(root, segmentation, relative)
        if stored is not None:
            colony = _circle(stored.shape, centre, radius)
            return relabel_inside(stored, colony), colony, pixel_um

    def _labels():
        channel = read_nuclear_channel(path, group)
        mask = _circle(channel.shape, centre, radius)
        return count_nuclei(
            channel, mask, nucleus_diameter_um / pixel_um, seed_spacing, threshold_scale, split_method, roundness,
            smallest, largest,
        )[1]

    if cache_dir is None:
        labels = _labels()
    else:
        settings = {"dataset": str(root), "nucleus_diameter_um": nucleus_diameter_um}
        # Only settings changed from their defaults are named, so counts cached before they existed stay valid
        if seed_spacing != NUCLEUS_SEED_SPACING:
            settings["seed_spacing"] = float(seed_spacing)
        if threshold_scale != NUCLEUS_THRESHOLD:
            settings["threshold_scale"] = float(threshold_scale)
        if split_method != NUCLEUS_SPLIT_METHOD:
            settings["split_method"] = str(split_method)
        if roundness != NUCLEUS_ROUNDNESS:
            settings["roundness"] = float(roundness)
        if smallest != NUCLEUS_SMALLEST:
            settings["smallest"] = float(smallest)
        if largest != NUCLEUS_LARGEST:
            settings["largest"] = float(largest)
        folder = cache_folder(
            cache_dir, "micropattern_260726_nuclei",
            settings,
            (count_nuclei, read_nuclear_channel),
        )
        # The colony mask is part of the item name, so a new alignment is recounted
        item = f"{image_key(relative)}@{centre[0]:.1f}_{centre[1]:.1f}_{radius:.1f}"
        labels = cached_array(folder, item, _labels)
    return labels, _circle(labels.shape, centre, radius), pixel_um


def colony_cell_count(path, group, dataset_root, alignment, mask_radius_scale=1.0,
                      nucleus_diameter_um=NUCLEUS_DIAMETER_UM, cache_dir=None,
                      seed_spacing=NUCLEUS_SEED_SPACING, threshold_scale=NUCLEUS_THRESHOLD,
                  split_method=NUCLEUS_SPLIT_METHOD, roundness=NUCLEUS_ROUNDNESS,
                  smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST, segmentation=None):
    """``(cells, mean nuclear area in µm², colony area in mm²)`` of one raw image (``colony_nuclei``)."""

    labels, colony, pixel_um = colony_nuclei(
        path, group, dataset_root, alignment, mask_radius_scale, nucleus_diameter_um, cache_dir,
        seed_spacing=seed_spacing, threshold_scale=threshold_scale, split_method=split_method,
        roundness=roundness, smallest=smallest, largest=largest, segmentation=segmentation,
    )
    areas = np.bincount(labels.ravel())[1:]
    nuclear_area = float(np.mean(areas)) * pixel_um**2 if len(areas) else 0.0
    return len(areas), nuclear_area, float(np.count_nonzero(colony) * pixel_um**2 * 1e-6)


# ---------------------------------------------------------------------------
# Calibration: cells per model-resolution segment


def colony_segments(colony, support, indices, markers, split_distance, min_area):
    """``(segments, from_markers)`` of one staining panel of a model-resolution colony.

    Uses the segmentation of ``cell_library``, over the whole colony.
    """

    channel_markers = [channel.marker for channel in colony["schema"].measurement_channels]
    marker_indices = [i for i in indices if channel_markers[i] in markers]
    used = marker_indices or list(indices)
    measurements = colony["measurements"]
    coverage, _ = cell_coverage(measurements, support, used)
    found = segment_cells(
        measurements[indices], support, coverage, np.max(measurements[used], axis=0),
        split_distance, 1.0, min_area,
    )
    return len(found), bool(marker_indices)


def _measured_panels(colonies, source_files):
    """``(colony index, panel index, panel name, measurement indices, raw image path)`` of every measured panel."""

    schema = colonies["schema"]
    for index in range(len(colonies["measurements"])):
        for group_index, (name, indices) in enumerate(
            zip(schema.group_names, schema.group_measurement_indices)
        ):
            indices = list(indices)
            path = str(source_files[index][group_index])
            if path and np.all(colonies["available"][index, indices]):
                yield index, group_index, name, indices, path


def reference_cell_counts(colonies, source_files, dataset_root, alignment, markers,
                          split_distance=2, min_area=2, mask_radius_scale=1.0,
                          nucleus_diameter_um=NUCLEUS_DIAMETER_UM, cache_dir=None,
                          seed_spacing=NUCLEUS_SEED_SPACING, threshold_scale=NUCLEUS_THRESHOLD,
                  split_method=NUCLEUS_SPLIT_METHOD, roundness=NUCLEUS_ROUNDNESS,
                  smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST, segmentation=None):
    """Cells and model-resolution segments of every measured panel of the reference colonies.

    ``colonies`` is as ``cell_library`` reads it (``[B, C, H, W]``) and
    ``source_files`` ``[B, groups]`` the raw image of each panel at the same
    time (e.g. ``aux["source_files"][:, 0]``). ``markers``,
    ``split_distance`` and ``min_area`` are the segmentation settings of
    ``cell_library``. Returns one dict per replicate and panel:
    ``replicate``, ``panel``, ``cells``, ``segments``, ``from_markers``,
    ``colony_mm2``, ``cells_per_mm2`` and ``nuclear_area_um2``.
    """

    replicates = colonies.get("replicates", tuple(range(1, len(colonies["measurements"]) + 1)))
    rows = []
    for index, _, name, indices, path in _measured_panels(colonies, source_files):
        cells, nuclear_area, colony_mm2 = colony_cell_count(
            path, name, dataset_root, alignment, mask_radius_scale, nucleus_diameter_um, cache_dir,
            seed_spacing=seed_spacing, threshold_scale=threshold_scale, split_method=split_method,
        roundness=roundness, smallest=smallest, largest=largest, segmentation=segmentation,
        )
        segments, from_markers = colony_segments(
            {"measurements": np.asarray(colonies["measurements"][index]), "schema": colonies["schema"]},
            colonies["support"][index], indices, markers, split_distance, min_area,
        )
        rows.append({
            "replicate": replicates[index],
            "panel": name,
            "cells": cells,
            "segments": segments,
            "from_markers": from_markers,
            "colony_mm2": colony_mm2,
            "cells_per_mm2": cells / colony_mm2,
            "nuclear_area_um2": nuclear_area,
        })
    return rows


def summarise_counts(rows):
    """Mean cells per mm², cells per colony and cells per model-resolution segment.

    Cells per segment uses only panels with cell markers (as ``cell_library``).
    """

    if not rows:
        raise ValueError("No measured panel to count")
    calibrated = [row for row in rows if row["from_markers"] and row["segments"] > 0]
    if not calibrated:
        raise ValueError("No panel with cell markers to calibrate cells per segment")
    return {
        "cells_per_mm2": float(np.mean([row["cells_per_mm2"] for row in rows])),
        "cells_per_colony": float(np.mean([row["cells"] for row in rows])),
        "cells_per_segment": float(np.mean([row["cells"] / row["segments"] for row in calibrated])),
    }


# ---------------------------------------------------------------------------
# Whole cells from nuclei, at model resolution


def downsample_cell(mask, values, downsample):
    """One full-resolution cell at model resolution: ``(mask [h, w], values [C, h, w])``.

    ``mask`` ``[H, W]`` and ``values`` ``[C, H, W]`` are the cell's bounding
    box. Blocks of ``downsample`` pixels at least half covered by the cell
    are kept (the most covered one if none is), each with the mean of the
    cell's own pixels in it.
    """

    d = int(downsample)
    height, width = mask.shape
    rows, cols = -(-height // d), -(-width // d)
    covered = np.zeros((rows * d, cols * d))
    covered[:height, :width] = mask
    summed = np.zeros((values.shape[0], rows * d, cols * d))
    summed[:, :height, :width] = values * mask
    share = covered.reshape(rows, d, cols, d).mean(axis=(1, 3))
    means = summed.reshape(-1, rows, d, cols, d).sum(axis=(2, 4)) / np.maximum(share * d * d, 1e-12)
    kept = share >= 0.5
    if not np.any(kept):
        kept = share == share.max()
    found = np.argwhere(kept)
    (top, left), (bottom, right) = found.min(axis=0), found.max(axis=0) + 1
    return kept[top:bottom, left:right], means[:, top:bottom, left:right].astype(np.float32)


def _block_means(image, mask, downsample):
    """Block means ``[n, C]`` of an ``[X, Y, C]`` image over blocks at least half inside ``mask``."""

    d = int(downsample)
    rows, cols = image.shape[0] // d, image.shape[1] // d
    image = image[: rows * d, : cols * d].reshape(rows, d, cols, d, -1).mean(axis=(1, 3))
    inside = mask[: rows * d, : cols * d].reshape(rows, d, cols, d).mean(axis=(1, 3)) >= 0.5
    return image[inside]


def match_values(values, source, target, quantiles=101):
    """``values`` ``[C, ...]`` mapped channel by channel so that ``source`` ``[n, C]`` would match ``target`` ``[m, C]``.

    Quantile matching: a monotone map per channel, so the raw intensities
    take the model's scale (clipping, normalisation and cleaning) without
    registering the raw image to the model grid.
    """

    levels = np.linspace(0.0, 1.0, quantiles)
    matched = np.empty_like(values, dtype=np.float32)
    for channel in range(values.shape[0]):
        source_levels = np.quantile(source[:, channel], levels)
        # np.interp needs increasing points
        source_levels = source_levels + np.arange(quantiles) * 1e-9 * max(np.ptp(source_levels), 1.0)
        matched[channel] = np.interp(values[channel], source_levels, np.quantile(target[:, channel], levels))
    return matched


def nuclear_cells(path, group, names, dataset_root, alignment, downsample, target, support,
                  mask_radius_scale=1.0, interior_fraction=0.8, min_area=1,
                  nucleus_diameter_um=NUCLEUS_DIAMETER_UM, expansion_um=CELL_EXPANSION_UM,
                  cache_dir=None, seed_spacing=NUCLEUS_SEED_SPACING, threshold_scale=NUCLEUS_THRESHOLD,
                  split_method=NUCLEUS_SPLIT_METHOD, roundness=NUCLEUS_ROUNDNESS,
                  smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST, segmentation=None):
    """Whole cells of one raw image, at model resolution, as ``cell_library`` panels hold them.

    Each nucleus (``colony_nuclei``) is grown by ``expansion_um`` into its
    cell, without overlapping its neighbours. Cells reaching beyond
    ``interior_fraction`` of the colony radius (edge artefacts) are left
    out. The cells of the measurement channels ``names`` are shrunk to
    model resolution (``downsample``, ``downsample_cell``) and their values
    matched (``match_values``) to the model-resolution colony ``target``
    ``[C, H, W]`` inside ``support`` ``[H, W]``. Returns ``(cells,
    covered share)``: a list of ``(mask [h, w], values [C, h, w])``, and the
    share of the colony the grown cells cover (all of them, edge cells too),
    which is what the placed cells' covered share stands for.
    """

    from skimage.segmentation import expand_labels

    from Common.dataloader.hot_pixels import replace_hot_pixels_per_channel
    from Common.dataloader.micropattern_260726 import read_raw_channels
    from Common.dataloader.micropattern_cleaning import MicropatternCleaningConfig

    labels, colony, pixel_um = colony_nuclei(
        path, group, dataset_root, alignment, mask_radius_scale, nucleus_diameter_um, cache_dir,
        seed_spacing=seed_spacing, threshold_scale=threshold_scale, split_method=split_method,
        roundness=roundness, smallest=smallest, largest=largest, segmentation=segmentation,
    )
    channels, raw_names, _ = read_raw_channels(path, group)
    channels = channels[..., [raw_names.index(name) for name in names]]
    cleaning = MicropatternCleaningConfig()
    channels, _ = replace_hot_pixels_per_channel(
        channels, [cleaning.hot_pixel_thresholds.get(name, 0) for name in names], cleaning.hot_pixel_window,
    )
    cells = expand_labels(labels, distance=expansion_um / pixel_um) * colony
    source = _block_means(channels, colony, downsample)
    target = np.asarray(target, dtype=np.float32)[:, np.asarray(support, dtype=bool)].T
    # Edge artefacts: only cells within interior_fraction of the colony radius
    centre = np.argwhere(colony).mean(axis=0)
    radius = np.sqrt(np.count_nonzero(colony) / np.pi)
    interior = _circle(colony.shape, centre, interior_fraction * radius)
    found = []
    for label, box in enumerate(ndi.find_objects(cells), start=1):
        if box is None:
            continue
        mask = cells[box] == label
        if np.any(mask & ~interior[box]):
            continue
        small_mask, values = downsample_cell(mask, np.moveaxis(channels[box], -1, 0), downsample)
        if np.count_nonzero(small_mask) < min_area:
            continue
        found.append((small_mask, match_values(values, source, target)))
    return found, float(np.mean(cells[colony] > 0))


def nuclear_cell_library(colonies, source_files, dataset_root, alignment, downsample, markers,
                         split_distance=2, interior_fraction=0.8, min_area=1, mask_radius_scale=1.0,
                         nucleus_diameter_um=NUCLEUS_DIAMETER_UM, expansion_um=CELL_EXPANSION_UM,
                         cache_dir=None, seed_spacing=NUCLEUS_SEED_SPACING, threshold_scale=NUCLEUS_THRESHOLD,
                  split_method=NUCLEUS_SPLIT_METHOD, roundness=NUCLEUS_ROUNDNESS,
                  smallest=NUCLEUS_SMALLEST, largest=NUCLEUS_LARGEST, segmentation=None):
    """``cell_library`` with whole cells found from nuclei at full resolution instead of segments.

    The pieces are real cells (``cells_per_segment`` 1). Gap values are
    ``cell_library``'s (model resolution). Covered shares (``coverage`` rows
    and ``density``) are those of the grown cells (``nuclear_cells``), so
    they compare with the covered share of placed cells; every panel's
    cells come from its nuclear stain, so all rows count (``from_markers``).
    ``source_files`` ``[B, groups]`` are the raw images of ``colonies``
    (same time), ``downsample`` the model's.
    """

    from Experiments.micropatterns.reference import cell_library

    library = cell_library(colonies, markers, split_distance, interior_fraction, min_area)
    schema = colonies["schema"]
    replicates = colonies.get("replicates", tuple(range(1, len(colonies["measurements"]) + 1)))
    panels = [(np.asarray(indices, dtype=int), []) for indices in schema.group_measurement_indices]
    rows = []
    for index, group_index, name, indices, path in _measured_panels(colonies, source_files):
        cells, covered = nuclear_cells(
            path, name, [schema.measurement_names[i] for i in indices], dataset_root, alignment,
            downsample, np.asarray(colonies["measurements"][index])[indices], colonies["support"][index],
            mask_radius_scale, interior_fraction, min_area, nucleus_diameter_um, expansion_um, cache_dir,
            seed_spacing=seed_spacing, threshold_scale=threshold_scale, split_method=split_method,
        roundness=roundness, smallest=smallest, largest=largest, segmentation=segmentation,
        )
        panels[group_index][1].extend(cells)
        rows.append({
            "replicate": replicates[index],
            "panel": name,
            "coverage": covered,
            "cells": len(cells),
            "from_markers": True,
        })
    for name, (_, cells) in zip(schema.group_names, panels):
        if not cells:
            raise ValueError(f"No whole cell was found from the nuclei of panel {name!r}")
    return {
        **library,
        "panels": panels,
        "coverage": rows,
        "density": float(np.mean([row["coverage"] for row in rows])),
        "cells_per_segment": 1.0,
    }


# ---------------------------------------------------------------------------
# Covered share <-> cells of a placed texture


def rank_softness(softness_cells, ranks):
    """``CellGaps.softness`` (a share of a layer's ranks) for a cut ``softness_cells`` pieces wide.

    The softness is a width in rank, so it fades about ``softness x pieces
    per layer`` pieces at the cut. For few kept pieces this fringe would be
    many times the number asked for, so it is set in pieces. ``ranks``
    ``[L, P, H, W]`` is the placement (``place_cells``), uncropped.
    """

    from Experiments.micropatterns.texture_synthesis import layer_cell_counts

    per_layer = float(np.mean(layer_cell_counts(ranks, np.ones(np.shape(ranks)[-2:], dtype=bool))))
    return float(softness_cells) / max(per_layer, 1.0)


def cells_to_density(cells, layer_pieces, layer_coverage, cells_per_segment):
    """Covered share of a placed texture holding ``cells`` cells.

    ``layer_pieces`` ``[L]`` are the placed pieces of each layer in the
    counted region (``layer_cell_counts``), each ``cells_per_segment``
    cells, and ``layer_coverage`` ``[L]`` the share of the grid each layer
    covers. Counts above what the layers hold give their full coverage.
    """

    import jax.numpy as jnp

    from NCA.inverse_design.density import density_for_level, level_for_cell_count

    layer_pieces = jnp.asarray(layer_pieces, dtype=jnp.float32)
    level = level_for_cell_count(float(cells) / float(cells_per_segment), layer_pieces)
    return float(density_for_level(level, jnp.asarray(layer_coverage, dtype=jnp.float32)))


def density_to_cells(density, layer_pieces, layer_coverage, cells_per_segment):
    """Cells in a placed texture at covered share ``density`` (inverse of ``cells_to_density``)."""

    import jax.numpy as jnp

    from NCA.inverse_design.density import fill_level, kept_cell_count

    level = fill_level(float(density), jnp.asarray(layer_coverage, dtype=jnp.float32))
    return float(kept_cell_count(level, jnp.asarray(layer_pieces, dtype=jnp.float32))) * float(cells_per_segment)
