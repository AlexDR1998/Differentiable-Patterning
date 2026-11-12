"""Cell density × NODAL-KO time sweep of trained micropattern NCAs.

Synthetic circular colonies are filled with whole cells at a range of
densities, and each is rolled out with NODAL blocked from a range of KO
times. How the cells are placed is the ``cells`` setting of each model's
projection in the texture settings file (``synthetic_ic.py``, tuned in
``synthetic_textured_ic_explorer.py``). Only the marker-pattern counts per
radial ring of the final state are kept (``[densities, KO times, rings,
patterns]`` per run), so results files stay small.

Used by ``nodal_ko_eval.py`` (section "Cell density × NODAL-KO time"), which
sets the sweep up, plots the results and can export it as a remote job. Run
one part of an exported job with:

    python Experiments/micropatterns/density_ko_sweep.py --manifest <manifest.yaml> [--index i]

See ``docs/remote_notebook_jobs.md``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from Common.dataloader.alignment import load_colony_alignment
from Common.dataloader.cell_type_fit import radial_rings
from Common.dataloader.cell_type_shares import pattern_counts
from Common.dataloader.cell_types import load_cell_type_rules, marker_values
from Common.dataloader.disk_cache import cache_dir_from_environment
from Experiments.micropatterns.cell_counts import (
    default_alignment,
    reference_cell_counts,
    summarise_counts,
    um_per_pixel,
)
from Experiments.micropatterns.synthetic_ic import cell_states, replicate_cell_library
from Experiments.micropatterns.texture_settings import CellSettings
from Experiments.notebook_jobs import (
    load_results_npz,
    part_path,
    read_job_entry,
    repository_path,
    save_results_npz,
)

repository_root = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class DensityKoSettings:
    """Everything the sweep depends on, apart from the models.

    Densities are in ``density_unit`` (``per_mm2``, ``per_colony`` or
    ``share``). ``cell_type_files`` maps a projection to its cell type file
    (relative to the repository root). ``data_root`` contains
    ``260726_nca_dataset``; ``data_root`` and ``store_root`` are replaced by
    ``DATA_PATH_BASE`` and ``MODEL_STORE_ROOT`` when a job runs remotely.
    """

    data_root: str
    store_root: str
    cell_type_files: dict = field(default_factory=dict)
    rings: int = 12
    density_unit: str = "per_mm2"
    density_start: float = 1000.0
    density_stop: float = 8000.0
    density_count: int = 7
    ko_start: float = 0.0
    ko_stop: float = 48.0
    ko_interval: float = 6.0
    snapshot_hour: float = 48.0
    replicates: tuple = (1,)
    seed: int = 0
    duplicate_runs: int = 3
    # How cells are placed, per projection: the cells sections of the texture
    # settings file (CellSettings as dicts), copied in so a job file keeps
    # the settings it ran with
    cells: dict = field(default_factory=dict)
    # Rollouts run on the GPU at once (runs × densities of one KO time)
    max_batch: int = 64

    def densities(self):
        return np.linspace(float(self.density_start), float(self.density_stop), int(self.density_count))

    def ko_hours(self):
        return np.arange(
            float(self.ko_start),
            float(self.ko_stop) + 0.5 * float(self.ko_interval),
            float(self.ko_interval),
        )

    def trajectories(self, models):
        """Number of rollouts for ``models`` models."""
        return (
            models * len(self.replicates) * int(self.duplicate_runs)
            * len(self.densities()) * len(self.ko_hours())
        )

    def check(self):
        if not str(self.data_root).strip():
            raise ValueError("Set Data root before running the density sweep.")
        if self.density_stop < self.density_start:
            raise ValueError("Highest density must be at or above the lowest.")
        if self.density_unit != "share" and self.density_start <= 0:
            raise ValueError("The lowest density must be above zero cells.")
        if self.ko_stop < self.ko_start:
            raise ValueError("Last KO time must be at or after first KO time.")
        if self.ko_stop > self.snapshot_hour:
            raise ValueError("KO times cannot be later than the snapshot time.")
        for projection in self.cell_type_files:
            self.cell_settings(projection)

    def cell_settings(self, projection):
        """``CellSettings`` for models of ``projection``."""
        if projection not in self.cells:
            raise ValueError(
                f"No cell settings for projection {projection!r}; save them with "
                "synthetic_textured_ic_explorer.py"
            )
        return CellSettings(**self.cells[projection])


def settings_from_dict(values):
    """``DensityKoSettings`` from a plain dict (as written into a job manifest)."""
    values = dict(values)
    values["replicates"] = tuple(int(_r) for _r in values.get("replicates", (1,)))
    values["cell_type_files"] = dict(values.get("cell_type_files") or {})
    # Job files from before the texture settings file held one set of cell
    # settings for every projection (placement "layers" by default)
    old = {_new: values.pop(_old) for _old, _new in OLD_CELL_KEYS.items() if _old in values}
    if "cells" not in values:
        cells = CellSettings(**{"placement": "layers", **old}).to_dict()
        values["cells"] = {_projection: cells for _projection in values["cell_type_files"]}
    return DensityKoSettings(**values)


# Cell settings of old job files, and their names in CellSettings
OLD_CELL_KEYS = {
    "piece_source": "pieces",
    "placement": "placement",
    "expansion_um": "expansion_um",
    "split_distance": "split_distance",
    "min_cell_area": "min_cell_area",
    "layers": "layers",
    "softness_cells": "softness_cells",
}


# ---------------------------------------------------------------- helpers


def record_label(record):
    """Short name of a selected model: its notes, alias, name or ID."""
    return str(
        record.get("notes")
        or record.get("alias")
        or record.get("display_name")
        or record["model_id"]
    )


def format_density(value, unit):
    return f"{value:.2f}" if unit == "share" else f"{value:.0f}"


def actual_densities(result):
    """Densities a run actually used, in its ``density_unit``.

    Requested densities above what the cell layers hold
    (``max_density``) are cut down to it, so these can be lower.
    """
    values = {
        "share": result["covered_shares"],
        "per_mm2": result["cells_per_mm2"],
        "per_colony": result["cells_per_colony"],
    }[result["density_unit"]]
    return np.asarray(result["densities"] if values is None else values, dtype=float)


def density_converter(values, other_values):
    """``(forward, inverse)`` functions between two density measures.

    ``values`` and ``other_values`` are the same sweep points in two units
    (e.g. cells per mm² and pixel coverage), both increasing with density.
    Between the points the functions interpolate linearly, and beyond them
    they continue the end segments, as matplotlib's secondary axes need.
    Points that do not increase in both units (e.g. densities cut down to
    the same value) are dropped. Returns None with fewer than two points.
    """
    values = np.asarray(values, dtype=float)
    other_values = np.asarray(other_values, dtype=float)
    keep = np.isfinite(values) & np.isfinite(other_values)
    order = np.argsort(values[keep], kind="stable")
    values, other_values = values[keep][order], other_values[keep][order]
    if len(values) == 0:
        return None
    increasing = [0]
    for index in range(1, len(values)):
        if values[index] > values[increasing[-1]] and other_values[index] > other_values[increasing[-1]]:
            increasing.append(index)
    values, other_values = values[increasing], other_values[increasing]
    if len(values) < 2:
        return None

    def _convert(x, xp, fp):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, xp, fp)
        low_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        high_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y = np.where(x < xp[0], fp[0] + (x - xp[0]) * low_slope, y)
        return np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * high_slope, y)

    return (
        lambda x: _convert(x, values, other_values),
        lambda y: _convert(y, other_values, values),
    )


def capped_densities(result):
    """True for the requested densities above what the cell layers hold."""
    return np.asarray(result["densities"], dtype=float) > float(result["max_density"])


def model_pattern_counts(image, channel_names, cell_type_rules, colony, n_rings, weights=None):
    """Pixels of each marker pattern in each ring ``[rings, patterns]`` of one model state ``[C, H, W]``.

    With ``weights`` ``[H, W]`` each pixel counts by its weight, e.g. its
    share of the cells (``tile_cells``), which gives counts of cells.
    """
    values = marker_values(image, channel_names, cell_type_rules, axis=0)
    markers = cell_type_rules.markers
    return pattern_counts(
        np.stack([np.ravel(values[_m]) for _m in markers]),
        [cell_type_rules.thresholds[_m] for _m in markers],
        radial_rings(colony, n_rings).ravel(),
        n_rings,
        weights=None if weights is None else np.ravel(weights),
    )


def ring_channel_means(state, colony, n_rings):
    """Mean of every channel of ``state`` ``[C, H, W]`` in each ring of the
    colony (``radial_rings``), as ``[rings, C]`` (NaN for empty rings)."""
    rings = radial_rings(colony, n_rings).ravel()
    keep = rings >= 0
    area = np.bincount(rings[keep], minlength=n_rings).astype(float)
    sums = np.stack([
        np.bincount(rings[keep], weights=np.ravel(channel)[keep], minlength=n_rings)
        for channel in np.asarray(state, dtype=float)
    ], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums / area[:, None]).astype(np.float32)


def load_training_histogram_bins(bundle, root):
    """Fit scaling once from the model's configured baseline train split."""

    from Experiments.micropatterns.config_helpers import load_data

    baseline_config = replace(
        bundle.config.data,
        knockout=replace(
            bundle.config.data.knockout,
            curriculum=("baseline",),
        ),
    )
    train_replicates = baseline_config.micropattern.train_replicates
    if train_replicates is None:
        train_replicates = tuple(range(1, baseline_config.batches + 1))
    baseline_data = load_data(
        baseline_config,
        impath=str(root / "260726_nca_dataset"),
        replicate_indices=tuple(index - 1 for index in train_replicates),
        pool_copies_override=1,
    )
    return baseline_data[1]["histogram_bins"]


def load_condition_data(bundle, root, condition, replicate, histogram_bins):
    """Load one matching experimental condition using training scaling."""

    if bundle.config.data.dataset != "micropatterns_260726":
        raise ValueError(
            f"{bundle.id} uses {bundle.config.data.dataset!r}; NODAL-KO evaluation "
            "requires 'micropatterns_260726'."
        )

    from Experiments.micropatterns.config_helpers import load_data

    data_config = replace(
        bundle.config.data,
        knockout=replace(
            bundle.config.data.knockout,
            curriculum=(condition,),
        ),
    )
    data_path = root / "260726_nca_dataset"
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"NODAL-KO dataset not found at {data_path}. Set Data root to the "
            "directory containing 260726_nca_dataset."
        )

    loaded_data = load_data(
        data_config,
        impath=str(data_path),
        replicate_indices=(replicate - 1,),
        histogram_bins=histogram_bins,
        pool_copies_override=1,
    )
    data, aux, channel_names, boundary, _, _ = loaded_data
    channel_schema = aux.get("channel_schema")
    if channel_schema is None or "NODAL" not in channel_schema.state_channels:
        raise ValueError(f"{bundle.id} has no NODAL channel in its data schema.")

    initial_measurements = np.asarray(data)[0, 0]
    initial_state = initial_measurements[
        np.asarray(channel_schema.primary_measurements)
    ]
    output_channels = tuple(channel_schema.target_to_state)
    displayed_count = len(output_channels)
    names = list(channel_names[:displayed_count])
    names.extend(
        f"Channel {index + 1}"
        for index in range(len(names), displayed_count)
    )
    return {
        "initial_state": initial_state,
        "boundary": np.asarray(boundary)[0],
        "target": np.asarray(data)[0],
        # [T, groups]: the raw image of each staining panel (for counting nuclei)
        "source_files": np.asarray(aux["source_files"])[0],
        # [T, channels]: False where the image was not measured
        "measured": np.asarray(
            aux.get("measurement_mask", np.ones(np.asarray(data).shape[:3], dtype=bool))
        )[0],
        "target_state": np.asarray(data)[0][
            :, np.asarray(channel_schema.primary_measurements)
        ],
        "output_channels": output_channels,
        "channel_names": tuple(names),
        "nodal_channel": channel_schema.state_channels.index("NODAL"),
        "state_channel_names": tuple(channel_schema.state_channels),
        "channel_schema": channel_schema,
    }


def stacked_colonies(condition_data_list, replicates):
    """The 0h colonies of several replicates (``load_condition_data``), as ``cell_library`` reads them."""
    return {
        "measurements": np.stack([_data["target"][0] for _data in condition_data_list]),
        "available": np.stack([_data["measured"][0] for _data in condition_data_list]),
        "support": np.stack([
            np.asarray(_data["boundary"]).squeeze() > 0.5 for _data in condition_data_list
        ]),
        "schema": condition_data_list[0]["channel_schema"],
        "replicates": tuple(replicates),
        # [B, groups]: the raw 0h image of each staining panel
        "source_files": np.stack([_data["source_files"][0] for _data in condition_data_list]),
    }


def bundle_alignment(bundle):
    """``(alignment, mask radius scale, µm per model pixel)`` of a bundle's data.

    The colony centres and pattern radius come from the bundle's alignment
    file when it cleaned its data, else from the repository's 260726 file.
    """
    cleaning = bundle.config.data.micropattern.cleaning
    if cleaning.enabled and cleaning.alignment_file:
        path = Path(cleaning.alignment_file)
        alignment = load_colony_alignment(path if path.is_absolute() else repository_root / path)
        scale = cleaning.mask_radius_scale
    else:
        alignment, scale = default_alignment(), 1.0
    return alignment, scale, um_per_pixel(alignment.pattern_radius, bundle.config.data.downsample)


def replicate_cell_counts(rows, replicate, panels=None):
    """``summarise_counts`` of one replicate's panels; what it lacks comes from every replicate.

    With ``panels`` (staining panel names), only those panels' counts are used.
    """
    if panels is not None:
        rows = [_row for _row in rows if _row["panel"] in panels]
    summary = summarise_counts(rows)
    try:
        summary.update(summarise_counts([_row for _row in rows if _row["replicate"] == replicate]))
    except ValueError:
        pass
    return summary


def check_stored_segmentation(colonies, dataset_root, method):
    """Raise unless ``method`` (a stored segmentation; "" for none) has labels for
    every colony's 0 h SOX2/LMBR image, so a missing copy cannot fall back to
    the classical nucleus counting unnoticed."""
    if not method:
        return
    from Experiments.micropatterns.nucleus_segmentation import GROUP, segmentation_root, stored_labels

    folder = segmentation_root(dataset_root) / method
    if not (folder / "manifest.yaml").exists():
        raise FileNotFoundError(
            f"The stored segmentation {method!r} is not at {folder}. Copy 260726_nca_segmentation next to "
            "the dataset folder (or set NUCLEUS_SEGMENTATION_ROOT)."
        )
    panel = list(colonies["schema"].group_names).index(GROUP)
    root = Path(dataset_root).expanduser().resolve()
    for path in colonies["source_files"][:, panel]:
        relative = Path(str(path)).resolve().relative_to(root).as_posix()
        if stored_labels(root, method, relative) is None:
            raise FileNotFoundError(f"The stored segmentation {method!r} has no labels for {relative} (in {folder}).")


def circle_mask(shape, area):
    """Centred disc of about ``area`` pixels on a grid ``shape``."""
    rows, cols = np.indices(shape)
    distance = np.hypot(rows - (shape[0] - 1) / 2, cols - (shape[1] - 1) / 2)
    return distance <= np.sqrt(area / np.pi)


# ---------------------------------------------------------------- results files

# Arrays of a density × KO result; everything else is saved as JSON
DENSITY_KO_ARRAYS = (
    "densities",
    "covered_shares",
    "cells_per_mm2",
    "cells_per_colony",
    "requested_ko_hours",
    "realised_ko_hours",
    "pattern_counts",
    "cell_pattern_counts",
    "channel_means",
    "initial_states",
    "colony",
)


def save_density_ko_results(path, results):
    """Write density × KO sweep results to one ``.npz`` file (``save_results_npz``)."""
    save_results_npz(path, results, DENSITY_KO_ARRAYS)


def _upgrade_result(result):
    """A loaded result, with the defaults of older files filled in."""
    result["state_channel_names"] = tuple(result["state_channel_names"])
    # Results from before cell numbers: densities are covered shares
    result.setdefault("density_unit", "share")
    result.setdefault("placement", "layers")
    result.setdefault("measured_covered_share", result["measured_density"])
    if result["covered_shares"] is None:
        result["covered_shares"] = result["densities"]
    return result


def load_density_ko_results(path):
    """Results written by ``save_density_ko_results``: one file, or a folder
    of ``part_*.npz`` files of a remote job (joined in part order)."""
    return tuple(_upgrade_result(_result) for _result in load_results_npz(path, DENSITY_KO_ARRAYS))


# ---------------------------------------------------------------- sweep


def _final_pattern_counts(
    model, states, keys, boundary, boundary_mode, total_steps, nodal_channel, ko_step,
    names, rules, mask, rings, max_batch, cell_weights=None,
):
    """Pattern counts ``[B, rings, patterns]`` of the final states of
    ``states`` ``[B, C, H, W]`` rolled out with ``keys`` ``[B]``, NODAL
    blocked from ``ko_step``. Rollouts run in chunks of ``max_batch``; the
    last chunk is padded, so every chunk uses the same compiled rollout.

    Returns ``(pixel counts, cell counts, channel means)``. With
    ``cell_weights`` ``[B, H, W]`` (each pixel's share of the 0h cells,
    ``synthetic_ic.cell_states``) the cell counts count every cell once, spread over its
    pixels; else None. Channel means are ``[B, rings, channels]`` of the
    measured state channels (``ring_channel_means``).
    """
    from NCA.trainer.intervention import rollout_batch_with_blocked_channel_final

    counts, cell_counts, means = [], [], []
    batch = max(1, min(int(max_batch), states.shape[0]))
    for start in range(0, states.shape[0], batch):
        chunk = np.arange(start, min(start + batch, states.shape[0]))
        padded = np.pad(chunk, (0, batch - len(chunk)), mode="edge")
        final = rollout_batch_with_blocked_channel_final(
            model,
            jnp.asarray(states[padded]),
            boundary,
            boundary_mode,
            keys[padded],
            total_steps,
            nodal_channel,
            jnp.asarray(ko_step, dtype=jnp.int32),
        )
        final = np.asarray(final)[: len(chunk), : len(names)]
        counts.extend(model_pattern_counts(_state, names, rules, mask, rings) for _state in final)
        means.extend(ring_channel_means(_state, mask, rings) for _state in final)
        if cell_weights is not None:
            cell_counts.extend(
                model_pattern_counts(_state, names, rules, mask, rings, weights=cell_weights[_index])
                for _state, _index in zip(final, chunk)
            )
    return (
        np.stack(counts),
        np.stack(cell_counts) if cell_weights is not None else None,
        np.stack(means),
    )


def prepare_model_cells(bundle, settings, projection, root):
    """What synthetic initial states of one model need from its measured colonies.

    Loads the 0h baseline colonies of ``settings.replicates`` (scaled as in
    training), counts their nuclei (``reference_cell_counts``, with the
    projection's cell settings, so a stored Cellpose segmentation is used
    when set) and cuts out each replicate's cells
    (``replicate_cell_library``). Shared by the density × KO and the size ×
    density sweeps, so both place cells the same way. ``settings`` needs
    ``replicates``, ``density_unit`` and ``cell_settings(projection)``.

    Returns a dict with ``cell_settings``, ``colonies`` (``stacked_colonies``),
    ``boundary`` (the data's boundary mask, for shape and dtype), ``names``
    (state channel names), ``nodal_channel``, ``pixel_um`` (µm per model
    pixel; NaN without nuclear counts), ``replicates`` (one dict per
    replicate: ``library``, ``measured_share``, ``cell_counts``,
    ``per_piece``) and ``warnings``.
    """
    cell_settings = settings.cell_settings(projection)
    replicates = tuple(settings.replicates)
    warnings = []
    bins = load_training_histogram_bins(bundle, root)
    data = [
        load_condition_data(bundle, root, "baseline", _replicate, bins)
        for _replicate in replicates
    ]
    colonies = stacked_colonies(data, replicates)
    # Nuclei of the raw 0h images (needed unless the sweep is in covered share)
    try:
        alignment, radius_scale, pixel_um = bundle_alignment(bundle)
        count_rows = reference_cell_counts(
            colonies, colonies["source_files"],
            root / "260726_nca_dataset", alignment,
            cell_settings.markers,
            cell_settings.split_distance, cell_settings.min_cell_area,
            mask_radius_scale=radius_scale,
            **cell_settings.nucleus_options(),
            cache_dir=cache_dir_from_environment(),
        )
    except (OSError, ValueError, KeyError) as count_error:
        if settings.density_unit != "share" or cell_settings.pieces == "nuclei":
            raise
        warnings.append(f"{bundle.id}: no nuclear counts ({count_error})")
        count_rows, pixel_um = None, float("nan")

    prepared_replicates = []
    for replicate_index, replicate in enumerate(replicates):
        library, measured_share = replicate_cell_library(
            colonies, replicate_index, cell_settings,
            raw=None if cell_settings.pieces == "segments" else {
                "source_files": colonies["source_files"],
                "dataset_root": root / "260726_nca_dataset",
                "alignment": alignment,
                "downsample": int(bundle.config.data.downsample),
                "mask_radius_scale": radius_scale,
                "cache_dir": cache_dir_from_environment(),
            },
        )
        if count_rows is None:
            cell_counts = {"cells_per_mm2": float("nan"), "cells_per_segment": float("nan")}
        else:
            cell_counts = replicate_cell_counts(count_rows, replicate, cell_settings.density_panels)
        prepared_replicates.append({
            "library": library,
            "measured_share": measured_share,
            "cell_counts": cell_counts,
            # Real cells per placed piece: 1 for real cells, else measured cells per segment
            "per_piece": library.get("cells_per_segment", cell_counts["cells_per_segment"]),
        })
    return {
        "cell_settings": cell_settings,
        "colonies": colonies,
        "boundary": np.asarray(data[0]["boundary"]),
        "names": data[0]["state_channel_names"],
        "nodal_channel": data[0]["nodal_channel"],
        "pixel_um": float(pixel_um),
        "replicates": prepared_replicates,
        "warnings": warnings,
    }


def run_density_ko_sweep(settings, records, job_offset=0, progress=None):
    """``(results, warnings)`` of the sweep for the models in ``records``.

    ``records`` are model-selection records (``load_selection``, with
    ``projection`` and ``downsample``). Each (model, replicate) pair is one
    job; run seeds are ``seed + job * duplicate_runs + run``, counting jobs
    from ``job_offset``, so a sweep split into parts keeps the seeds of a
    single run. ``progress(n)`` is called after every ``n`` rollouts.
    Results are the dicts ``save_density_ko_results`` writes, one per
    (model, replicate, run).
    """
    from Experiments.model_registry import ModelRegistry

    settings.check()
    unit = settings.density_unit
    densities = settings.densities()
    ko_hours = settings.ko_hours()
    registry = ModelRegistry(Path(settings.store_root).expanduser())
    root = Path(settings.data_root).expanduser()
    rules_by_projection = {
        _projection: load_cell_type_rules(repository_path(_path))
        for _projection, _path in settings.cell_type_files.items()
    }
    replicates = tuple(settings.replicates)
    runs = int(settings.duplicate_runs)
    rings = int(settings.rings)
    progress = progress or (lambda _n: None)

    results = []
    warnings = []
    job_index = int(job_offset)
    for record in records:
        projection = record["projection"]
        if projection not in rules_by_projection:
            raise ValueError(f"No cell type file is set for projection {projection!r}.")
        rules = rules_by_projection[projection]
        bundle = registry.get(record["model_id"])
        model = bundle.load_model(key=jr.PRNGKey(int(settings.seed)), implementation="portable")
        prepared = prepare_model_cells(bundle, settings, projection, root)
        warnings.extend(prepared["warnings"])
        cell_settings = prepared["cell_settings"]
        colonies = prepared["colonies"]
        # One circle for every replicate: the geometry stays fixed
        shape = colonies["support"].shape[-2:]
        mask = circle_mask(shape, float(np.mean(np.sum(colonies["support"], axis=(1, 2)))))
        mask_mm2 = float(np.count_nonzero(mask)) * prepared["pixel_um"]**2 * 1e-6
        boundary = jnp.broadcast_to(
            jnp.asarray(mask, dtype=prepared["boundary"].dtype), prepared["boundary"].shape,
        )
        names = prepared["names"]
        channels = int(model.N_CHANNELS)
        steps_per_12h = int(bundle.config.run.t)
        total_steps = int(round(float(settings.snapshot_hour) * steps_per_12h / 12.0))
        ko_steps = [int(round(float(_hour) * steps_per_12h / 12.0)) for _hour in ko_hours]
        curriculum = tuple(bundle.config.data.knockout.curriculum or ("baseline",))
        label = record_label(record)
        for replicate, prepared_replicate in zip(replicates, prepared["replicates"]):
            library = prepared_replicate["library"]
            measured_share = prepared_replicate["measured_share"]
            cell_counts = prepared_replicate["cell_counts"]
            per_piece = prepared_replicate["per_piece"]
            # Measured density of the replicate, in unit
            measured_density = {
                "share": measured_share,
                "per_mm2": cell_counts["cells_per_mm2"],
                "per_colony": cell_counts["cells_per_mm2"] * mask_mm2,
            }[unit]
            # Requested densities as real cells in the circle (or covered shares)
            targets = densities * mask_mm2 if unit == "per_mm2" else densities

            # Place the cells of every run, then roll them all out together
            placed = []
            for run_index in range(runs):
                run_seed = int(settings.seed) + job_index * runs + run_index
                states = cell_states(
                    library, colonies["schema"], mask, targets, unit == "share",
                    run_seed, cell_settings, per_piece,
                )
                max_density = states["capacity"] / (mask_mm2 if unit == "per_mm2" else 1.0)
                where = f"{label}, replicate {replicate}, run {run_index}"
                if densities[-1] > max_density:
                    warnings.append(
                        f"{where}: the {cell_settings.placement} cells hold at most "
                        f"{format_density(max_density, unit)}"
                    )
                if states["squeeze"] < 0.5:
                    warnings.append(
                        f"{where}: at the highest density cells have "
                        f"{states['squeeze']:.0%} of the measured cell area"
                    )
                placed.append({
                    "run_index": run_index,
                    "seed": run_seed,
                    "shares": states["shares"],
                    "cells": states["cells"],
                    "max_density": max_density,
                    "initial_states": states["initial_states"],
                    "cell_weights": states["cell_weights"],
                    "cell_layers": states["layers"],
                    "cell_area_px": states["cell_area"],
                })

            # [runs × densities, C, H, W], padded with zero channels to the model's
            states = np.concatenate([_run["initial_states"] for _run in placed])
            states = np.pad(states, ((0, 0), (0, channels - states.shape[1]), (0, 0), (0, 0)))
            keys = jnp.concatenate([
                jnp.broadcast_to(jr.PRNGKey(_run["seed"]), (len(densities), 2)) for _run in placed
            ])
            # Each pixel's share of the 0h cells that show (synthetic_ic.cell_states)
            weights = np.concatenate([_run["cell_weights"] for _run in placed])
            # [KO times, runs × densities, rings, patterns]
            counts, weighted_counts, channel_means = [], [], []
            for ko_step in ko_steps:
                pixel_counts, cell_weighted, ring_means = _final_pattern_counts(
                    model, states, keys, boundary, bundle.config.trainer.boundary_mode,
                    total_steps, prepared["nodal_channel"], ko_step,
                    names, rules, mask, rings, settings.max_batch, cell_weights=weights,
                )
                counts.append(pixel_counts)
                weighted_counts.append(cell_weighted)
                channel_means.append(ring_means)
                progress(states.shape[0])

            def _by_run(_counts):
                # [runs, densities, KO times, rings, patterns]
                return np.stack(_counts, axis=1).reshape(
                    runs, len(densities), len(ko_steps), *_counts[0].shape[1:]
                )

            counts = _by_run(counts)
            # Not cell_counts: that name holds the replicate's measured nuclei
            weighted_counts = _by_run(weighted_counts) if weights is not None else [None] * runs
            channel_means = _by_run(channel_means)

            for run, run_counts, run_cell_counts, run_means in zip(placed, counts, weighted_counts, channel_means):
                results.append({
                    "model_id": bundle.id,
                    "label": label,
                    "projection": projection,
                    "downsample": record["downsample"],
                    "replicate": replicate,
                    "run_index": run["run_index"],
                    "seed": run["seed"],
                    "groups": {
                        "projection": projection,
                        "projection_resolution": f"{projection} · ds{record['downsample']}",
                        "curriculum": " + ".join(map(str, curriculum)),
                        "experiment": str(record.get("experiment") or "Unlabelled"),
                        "family": str(record.get("family") or "Unlabelled"),
                        "notes": str(record.get("notes") or "Unlabelled"),
                        "individual": label,
                        "all": "All selected models",
                    },
                    # Requested, in density_unit; the covered shares used
                    "densities": densities,
                    "density_unit": unit,
                    "covered_shares": run["shares"],
                    "cells_per_colony": run["cells"],
                    "cells_per_mm2": run["cells"] / mask_mm2,
                    "piece_source": cell_settings.pieces,
                    "cells_per_segment": per_piece,
                    "measured_cells_per_mm2": cell_counts["cells_per_mm2"],
                    "colony_mm2": mask_mm2,
                    "measured_covered_share": measured_share,
                    "max_density": run["max_density"],
                    "measured_density": measured_density,
                    "requested_ko_hours": ko_hours,
                    "realised_ko_hours": np.asarray(
                        [12.0 * _step / steps_per_12h for _step in ko_steps]
                    ),
                    "snapshot_hour": float(settings.snapshot_hour),
                    "placement": cell_settings.placement,
                    # [densities, KO times, rings, patterns]: pixels, and cells
                    # (each 0h cell that shows counted once over its pixels)
                    "pattern_counts": run_counts,
                    "cell_pattern_counts": run_cell_counts,
                    # [densities, KO times, rings, channels]: mean of each measured
                    # state channel (state_channel_names) in each ring
                    "channel_means": run_means,
                    "cell_area_px": run["cell_area_px"],
                    # Kept for the first run only, to show the initial states
                    "initial_states": run["initial_states"] if not results else None,
                    "state_channel_names": names,
                    "colony": mask,
                    # What the pattern counts depend on, checked when loading
                    "cell_type_markers": list(rules.markers),
                    "cell_type_thresholds": {
                        _marker: float(rules.thresholds[_marker]) for _marker in rules.markers
                    },
                    "rings": rings,
                    "cell_settings": cell_settings.to_dict(),
                    "cell_layers": run["cell_layers"],
                })
            job_index += 1
    return tuple(results), warnings


# ---------------------------------------------------------------- remote jobs


def job_configs(settings, records, output_dir, split="model"):
    """One manifest entry per model (``split="model"``) or per (model,
    replicate) pair (``split="replicate"``), each writing a part into
    ``output_dir``. ``job_offset`` keeps the seeds of a single run."""
    configs = []
    job_offset = 0
    for record in records:
        replicate_groups = (
            [(_r,) for _r in settings.replicates] if split == "replicate"
            else [tuple(settings.replicates)]
        )
        for group in replicate_groups:
            configs.append({
                "settings": asdict(replace(settings, replicates=group)),
                "records": [dict(record)],
                # Seeds count (model, replicate) pairs over the whole sweep
                "job_offset": job_offset,
                "output_dir": str(output_dir),
            })
            job_offset += len(group)
    return configs


def run_job_part(config, index, overwrite=False):
    """Run one part of an exported job and write its results file."""
    settings = settings_from_dict(config["settings"])
    # On the cluster the data and models are in the pod's volumes
    settings = replace(
        settings,
        data_root=os.getenv("DATA_PATH_BASE") or settings.data_root,
        store_root=os.getenv("MODEL_STORE_ROOT") or settings.store_root,
    )
    output = part_path(repository_path(config["output_dir"]), index)
    if output.exists() and not overwrite:
        print(f"{output} exists; skipping part {index}")
        return output
    from tqdm.auto import tqdm

    records = list(config["records"])
    with tqdm(total=settings.trajectories(len(records)), desc=f"Part {index}", unit="trajectory") as bar:
        results, warnings = run_density_ko_sweep(
            settings, records, job_offset=int(config["job_offset"]), progress=bar.update
        )
    for warning in warnings:
        print(f"Warning: {warning}")
    save_density_ko_results(output, results)
    print(f"Saved part {index} to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run one part of a density × KO sweep job")
    parser.add_argument("--manifest", required=True, help="Job manifest exported by nodal_ko_eval.py")
    parser.add_argument("--index", type=int, default=None, help="Part to run (default: from the job environment)")
    parser.add_argument("--overwrite", action="store_true", help="Rerun a part whose results file exists")
    args = parser.parse_args()
    index, config = read_job_entry(args.manifest, args.index)
    run_job_part(config, index, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
