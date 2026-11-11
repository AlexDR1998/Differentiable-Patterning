"""Cell numbers and cell types of measured colonies over time, counted by nuclei.

A check of how cell-type prevalence by cells relates to prevalence by
pixels in the 260726 images. For every replicate, time and cell-fate
staining panel, nuclei are found in the raw image (``colony_nuclei``,
LMBR) and grown into cells (as ``nuclear_cells``). Each cell's marker
values are its mean raw values, mapped to the model's scale by quantile
matching. Cell types are then estimated twice with the same estimator
(``colony_label_shares``: per ring, combining the stains): once counting
every cell once, once counting the model-resolution pixels. So the census
shows

- how the number of cells in the colony changes over time, and
- whether the share of cells of a type differs from the share of pixels,
  i.e. whether cell types differ in area per cell.

Which images are read follows the model bundle's data settings, so a
slice-trained model gives the slices. Used by ``cell_census_explorer.py``.
"""

import numpy as np
import scipy.ndimage as ndi

from Common.dataloader.cell_type_fit import radial_rings
from Common.dataloader.cell_type_shares import colony_label_shares, pattern_counts
from Common.dataloader.cell_types import label_names, marker_values
from Experiments.micropatterns.cell_counts import (
    CELL_EXPANSION_UM,
    NUCLEUS_DIAMETER_UM,
    _block_means,
    colony_nuclei,
    match_values,
)


def panel_stain(cell_type_rules, names):
    """``(stain, {marker: channel})`` of the rules' stain imaged in a panel
    with measurement ``names``, or None if no stain is fully in it."""
    for stain, channels in cell_type_rules.stains.items():
        if channels and all(channel in names for channel in channels.values()):
            return stain, dict(channels)
    return None


def point_rings(points, centre, radius, n_rings):
    """Ring of each point ``[n, 2]`` (0 at the centre), the colony
    ``radius`` split into ``n_rings`` equal steps, as ``radial_rings``."""
    distance = np.hypot(*(np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)).T)
    return np.minimum((distance / max(float(radius), 1e-9) * n_rings).astype(int), n_rings - 1)


def cell_mean_values(cells, channels, count):
    """Mean of each channel of ``channels`` ``[X, Y, C]`` over each cell
    1 .. ``count`` of the label image ``cells``, as ``[C, count]``."""
    index = np.arange(1, int(count) + 1)
    if len(index) == 0:
        return np.zeros((channels.shape[-1], 0), dtype=np.float32)
    return np.stack([
        np.asarray(ndi.mean(channels[..., channel], cells, index), dtype=np.float32)
        for channel in range(channels.shape[-1])
    ])


def panel_census(path, group, names, dataset_root, alignment, downsample, target, support,
                 cell_type_rules, n_rings, mask_radius_scale=1.0,
                 nucleus_diameter_um=NUCLEUS_DIAMETER_UM, expansion_um=CELL_EXPANSION_UM,
                 cache_dir=None):
    """Census of one raw panel image, or None if the panel images no stain of the rules.

    ``names`` are the panel's measurement channels, ``target`` ``[C, H, W]``
    its model-resolution image (same channels) and ``support`` ``[H, W]``
    the colony there. Returns a dict with ``stain``, ``markers``,
    ``nuclei``, ``colony_mm2`` and, per ring, the counts of each marker
    pattern of the cells (``cell_counts``) and of the model pixels
    (``pixel_counts``), both ``[rings, patterns]``.
    """
    from skimage.segmentation import expand_labels

    from Common.dataloader.hot_pixels import replace_hot_pixels_per_channel
    from Common.dataloader.micropattern_260726 import read_raw_channels
    from Common.dataloader.micropattern_cleaning import MicropatternCleaningConfig

    found = panel_stain(cell_type_rules, names)
    if found is None:
        return None
    stain, stain_channels = found
    markers = tuple(stain_channels)
    thresholds = [cell_type_rules.thresholds[marker] for marker in markers]
    names = list(names)

    labels, colony, pixel_um = colony_nuclei(
        path, group, dataset_root, alignment, mask_radius_scale, nucleus_diameter_um, cache_dir,
    )
    channels, raw_names, _ = read_raw_channels(path, group)
    channels = channels[..., [raw_names.index(name) for name in names]]
    cleaning = MicropatternCleaningConfig()
    channels, _ = replace_hot_pixels_per_channel(
        channels, [cleaning.hot_pixel_thresholds.get(name, 0) for name in names], cleaning.hot_pixel_window,
    )
    cells = expand_labels(labels, distance=expansion_um / pixel_um) * colony
    support = np.asarray(support, dtype=bool)
    target = np.asarray(target, dtype=np.float32)
    # Each cell's mean raw values, on the model's scale
    index = np.arange(1, int(labels.max()) + 1)
    values = match_values(
        cell_mean_values(cells, channels, len(index)), _block_means(channels, colony, downsample),
        target[:, support].T,
    )
    centre = np.argwhere(colony).mean(axis=0)
    radius = np.sqrt(np.count_nonzero(colony) / np.pi)
    positions = np.asarray(ndi.center_of_mass(labels > 0, labels, index)).reshape(-1, 2)
    cell_counts = pattern_counts(
        np.stack([values[names.index(stain_channels[marker])] for marker in markers]).reshape(len(markers), -1),
        thresholds, point_rings(positions, centre, radius, n_rings), n_rings,
    )
    pixel_values = marker_values(target, names, cell_type_rules, axis=0, channels=stain_channels)
    pixel_counts = pattern_counts(
        np.stack([np.ravel(pixel_values[marker]) for marker in markers]),
        thresholds, radial_rings(support, n_rings).ravel(), n_rings,
    )
    return {
        "stain": stain,
        "markers": markers,
        "nuclei": int(len(index)),
        "colony_mm2": float(np.count_nonzero(colony)) * pixel_um**2 * 1e-6,
        "cell_counts": cell_counts,
        "pixel_counts": pixel_counts,
    }


def colony_census(bundle, data_root, condition, replicates, cell_type_rules, n_rings,
                  expansion_um=CELL_EXPANSION_UM, cache_dir=None, progress=None):
    """Census rows of every replicate, time and cell-fate panel of one condition.

    Images and scaling follow ``bundle``'s data settings (projection,
    downsample, alignment). One row per measured panel: ``replicate``,
    ``condition``, ``hour``, ``panel``, plus ``panel_census``'s fields.
    ``progress()`` is called after each image.
    """
    from pathlib import Path

    from Experiments.micropatterns.density_ko_sweep import (
        bundle_alignment,
        load_condition_data,
        load_training_histogram_bins,
    )

    root = Path(data_root).expanduser()
    bins = load_training_histogram_bins(bundle, root)
    alignment, radius_scale, _ = bundle_alignment(bundle)
    hours = tuple(bundle.config.data.micropattern.timesteps)
    rows = []
    for replicate in replicates:
        data = load_condition_data(bundle, root, condition, replicate, bins)
        schema = data["channel_schema"]
        support = np.asarray(data["boundary"]).squeeze() > 0.5
        for time_index, hour in enumerate(hours[: len(data["target"])]):
            for group_index, (panel, indices) in enumerate(
                zip(schema.group_names, schema.group_measurement_indices)
            ):
                indices = list(indices)
                path = str(data["source_files"][time_index][group_index])
                if not path or not np.all(data["measured"][time_index, indices]):
                    continue
                census = panel_census(
                    path, panel, [schema.measurement_names[i] for i in indices],
                    root / "260726_nca_dataset", alignment, int(bundle.config.data.downsample),
                    data["target"][time_index][indices], support, cell_type_rules, n_rings,
                    mask_radius_scale=radius_scale, expansion_um=expansion_um, cache_dir=cache_dir,
                )
                if progress is not None:
                    progress()
                if census is not None:
                    rows.append({
                        "replicate": replicate, "condition": condition, "hour": float(hour),
                        "panel": panel, **census,
                    })
    return rows


def census_label_shares(rows, cell_type_rules, resolve_several=0.0):
    """Cell-type shares by cells and by pixels, one row per (replicate,
    condition, hour, label) whose panels image every stain of the rules.

    Columns: ``share_cells``, ``share_pixels``, ``nuclei`` and
    ``colony_mm2`` (means over the panels), and ``area_per_cell_um2``, the
    colony area of the label per cell of the label (NaN if it has no cells).
    """
    needed = {stain for stain, channels in cell_type_rules.stains.items() if channels}
    labels = label_names(cell_type_rules)
    groups = {}
    for row in rows:
        groups.setdefault((row["replicate"], row["condition"], row["hour"]), []).append(row)
    shares = []
    for (replicate, condition, hour), panels in groups.items():
        by_stain = {row["stain"]: row for row in panels}
        if set(by_stain) < needed or not needed:
            continue
        stains = [by_stain[stain] for stain in sorted(needed)]
        by_cells = colony_label_shares(
            [(row["markers"], row["cell_counts"]) for row in stains], cell_type_rules,
            resolve_several=resolve_several,
        )
        by_pixels = colony_label_shares(
            [(row["markers"], row["pixel_counts"]) for row in stains], cell_type_rules,
            resolve_several=resolve_several,
        )
        nuclei = float(np.mean([row["nuclei"] for row in stains]))
        colony_mm2 = float(np.mean([row["colony_mm2"] for row in stains]))
        for label, cell_share, pixel_share in zip(labels, by_cells, by_pixels):
            cells_of_label = cell_share * nuclei
            shares.append({
                "replicate": replicate,
                "condition": condition,
                "hour": hour,
                "cell_type": label,
                "share_cells": float(cell_share),
                "share_pixels": float(pixel_share),
                "nuclei": nuclei,
                "colony_mm2": colony_mm2,
                "area_per_cell_um2": (
                    float(pixel_share * colony_mm2 * 1e6 / cells_of_label) if cells_of_label > 0 else float("nan")
                ),
            })
    return shares


# Arrays of a census row; everything else is saved as JSON
CENSUS_ARRAYS = ("cell_counts", "pixel_counts")


def save_census(path, rows, settings=None):
    """Write census rows (and the ``settings`` used, a JSON-able dict) to one ``.npz`` file."""
    import json
    from pathlib import Path

    arrays, metadata = {}, []
    for index, row in enumerate(rows):
        metadata.append({key: value for key, value in row.items() if key not in CENSUS_ARRAYS})
        for key in CENSUS_ARRAYS:
            arrays[f"{index}.{key}"] = np.asarray(row[key])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(
            {"version": 1, "settings": settings or {}, "rows": metadata},
            default=lambda value: value.item() if hasattr(value, "item") else list(value),
        )),
        **arrays,
    )


def load_census(path):
    """``(rows, settings)`` written by ``save_census``."""
    import json
    from pathlib import Path

    with np.load(Path(path), allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        rows = []
        for index, row in enumerate(metadata["rows"]):
            row["markers"] = tuple(row["markers"])
            for key in CENSUS_ARRAYS:
                row[key] = stored[f"{index}.{key}"]
            rows.append(row)
    return rows, metadata["settings"]
