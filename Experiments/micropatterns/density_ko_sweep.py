"""Cell density × NODAL-KO time sweep of trained micropattern NCAs.

Synthetic circular colonies are filled with whole cells at a range of
densities, and each is rolled out with NODAL blocked from a range of KO
times. Only the marker-pattern counts per radial ring of the final state
are kept (``[densities, KO times, rings, patterns]`` per run), so results
files stay small.

Used by ``nodal_ko_eval.py`` (section "Cell density × NODAL-KO time"), which
sets the sweep up, plots the results and can export it as a remote job. Run
one part of an exported job with:

    python Experiments/micropatterns/density_ko_sweep.py --manifest <manifest.yaml> [--index i]

See ``docs/remote_notebook_jobs.md``.
"""

from __future__ import annotations

import argparse
import json
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
from Experiments.inverse_design.config import InverseDesignDensityConfig
from Experiments.micropatterns.cell_counts import (
    CELL_EXPANSION_UM,
    cells_to_density,
    default_alignment,
    density_to_cells,
    nuclear_cell_library,
    rank_softness,
    reference_cell_counts,
    summarise_counts,
    um_per_pixel,
)
from Experiments.micropatterns.reference import cell_library
from Experiments.micropatterns.texture_synthesis import layer_cell_counts, place_cells
from Experiments.notebook_jobs import part_path, part_paths, read_job_entry, repository_path
from NCA.inverse_design.density import CellDensity, CellGaps, pack_cells

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
    piece_source: str = "nuclei"
    expansion_um: float = CELL_EXPANSION_UM
    split_distance: int = InverseDesignDensityConfig().split_distance
    min_cell_area: int = InverseDesignDensityConfig().min_cell_area
    layers: int = InverseDesignDensityConfig().layers
    softness_cells: float = 0.5
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


def settings_from_dict(values):
    """``DensityKoSettings`` from a plain dict (as written into a job manifest)."""
    values = dict(values)
    values["replicates"] = tuple(int(_r) for _r in values.get("replicates", (1,)))
    values["cell_type_files"] = dict(values.get("cell_type_files") or {})
    return DensityKoSettings(**values)


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


def model_pattern_counts(image, channel_names, cell_type_rules, colony, n_rings):
    """Pixels of each marker pattern in each ring ``[rings, patterns]`` of one model state ``[C, H, W]``."""
    values = marker_values(image, channel_names, cell_type_rules, axis=0)
    markers = cell_type_rules.markers
    return pattern_counts(
        np.stack([np.ravel(values[_m]) for _m in markers]),
        [cell_type_rules.thresholds[_m] for _m in markers],
        radial_rings(colony, n_rings).ravel(),
        n_rings,
    )


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


def replicate_cell_counts(rows, replicate):
    """``summarise_counts`` of one replicate's panels; what it lacks comes from every replicate."""
    summary = summarise_counts(rows)
    try:
        summary.update(summarise_counts([_row for _row in rows if _row["replicate"] == replicate]))
    except ValueError:
        pass
    return summary


def replicate_cell_library(colonies, index, density_config, nuclei=None):
    """Whole pieces of colony ``index``. Staining panels it did not measure
    at 0h take their pieces from the other colonies.

    Pieces are segments (``cell_library``), or with ``nuclei`` (keyword
    arguments of ``nuclear_cell_library`` after the colonies: source files,
    dataset root, alignment, downsample, ...) real cells from the nuclei.
    """
    available = np.array(colonies["available"], dtype=bool)
    others = np.arange(len(available)) != index
    for indices in colonies["schema"].group_measurement_indices:
        indices = np.asarray(indices, dtype=int)
        if np.all(available[index, indices]):
            available[np.ix_(others, indices)] = False
    if nuclei is None:
        library = cell_library(
            {**colonies, "available": available},
            density_config.markers,
            density_config.split_distance,
            density_config.interior_fraction,
            density_config.min_cell_area,
        )
    else:
        library = nuclear_cell_library(
            {**colonies, "available": available},
            markers=density_config.markers,
            split_distance=density_config.split_distance,
            interior_fraction=density_config.interior_fraction,
            min_area=density_config.min_cell_area,
            **nuclei,
        )
    own = [
        _row["coverage"] for _row in library["coverage"]
        if _row["replicate"] == colonies["replicates"][index] and _row["from_markers"]
    ]
    # The colony's own covered share (else that of the colonies used)
    measured_density = float(np.mean(own)) if own else library["density"]
    return library, measured_density


def circle_mask(shape, area):
    """Centred disc of about ``area`` pixels on a grid ``shape``."""
    rows, cols = np.indices(shape)
    distance = np.hypot(rows - (shape[0] - 1) / 2, cols - (shape[1] - 1) / 2)
    return distance <= np.sqrt(area / np.pi)


def placed_cell_texture(
    library, schema, shape, density_config, seed, highest_density=0.0, max_layers=10,
    mask=None, highest_cells=0.0, softness_cells=None,
):
    """``(packed texture, CellGaps, cells per layer)``: whole cells placed
    at random over ``shape``, in state channel order
    (``NCA/inverse_design/density.py``).

    Cells are placed over a margin around the grid and cropped, so the
    grid edge has as many cells as the centre. One layer of
    non-overlapping cells jams well below full coverage, so layers are
    added (from ``density_config.layers`` up to ``max_layers``) until
    together they can cover ``highest_density`` and hold
    ``highest_cells`` cells in ``mask`` ``[H, W]``. Placement is layer by
    layer from ``seed``, so the extra layers go on top of the same cells.
    The cells per layer are counted in ``mask`` (None without a mask).
    ``softness_cells`` sets the cut softness in cells (``rank_softness``)
    instead of ``density_config.softness``.
    """
    panels = library["panels"]
    margin = max(max(_mask.shape) for _, _cells in panels for _mask, _ in _cells)

    def _place(layers):
        return place_cells(
            panels,
            len(library["measurement_gap_values"]),
            (shape[0] + 2 * margin, shape[1] + 2 * margin),
            layers,
            density_config.attempts,
            seed,
        )

    padded_mask = None if mask is None else np.pad(np.asarray(mask, dtype=bool), margin)

    def _cells(ranks):
        return None if padded_mask is None else layer_cell_counts(ranks, padded_mask)

    layers = density_config.layers
    textures, ranks, coverage = _place(layers)
    cells = _cells(ranks)
    uncovered = float(np.prod(1.0 - coverage))
    per_layer = 1.0 - uncovered ** (1.0 / layers)
    needed = layers
    if 1.0 - uncovered < highest_density and per_layer > 0.0:
        # Layers are about equally full and independent: 1 - (1 - c) ** L.
        # The small margin covers layers that come out a little emptier.
        target = min(float(highest_density) + 0.02, 0.999)
        needed = int(np.ceil(np.log(1.0 - target) / np.log(1.0 - per_layer)))
    if cells is not None and float(np.sum(cells)) < highest_cells and np.mean(cells) > 0.0:
        needed = max(needed, int(np.ceil(1.02 * highest_cells / float(np.mean(cells)))))
    if needed > layers and layers < max_layers:
        textures, ranks, coverage = _place(min(max(needed, layers + 1), max_layers))
        cells = _cells(ranks)
    crop = (
        slice(None), slice(None),
        slice(margin, margin + shape[0]), slice(margin, margin + shape[1]),
    )
    primary = np.asarray(schema.primary_measurements, dtype=int)
    panel_of = {
        int(_index): _panel
        for _panel, (_indices, _) in enumerate(panels) for _index in _indices
    }
    gaps = CellGaps(
        gap_values=jnp.asarray(library["gap_values"]),
        channel_panels=jnp.asarray([panel_of[int(_index)] for _index in primary]),
        layer_coverage=jnp.asarray(coverage),
        panels=len(panels),
        softness=(
            density_config.softness if softness_cells is None
            else rank_softness(softness_cells, ranks)
        ),
    )
    packed = pack_cells(jnp.asarray(textures[crop][:, primary]), jnp.asarray(ranks[crop]))
    return packed, gaps, cells


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
    "initial_states",
    "colony",
)


def save_density_ko_results(path, results):
    """Write density × KO sweep results to one ``.npz`` file.

    Arrays are stored as ``<result index>.<name>``, the rest of each
    result (labels, groups, settings) as JSON under ``metadata``.
    """
    arrays, metadata = {}, []
    for index, result in enumerate(results):
        metadata.append({key: value for key, value in result.items() if key not in DENSITY_KO_ARRAYS})
        for key in DENSITY_KO_ARRAYS:
            if result[key] is not None:
                arrays[f"{index}.{key}"] = np.asarray(result[key])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and then renamed, so a job killed while
    # writing leaves no half-written part (which would be skipped on a rerun)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as file:
        np.savez_compressed(
            file,
            metadata=np.asarray(json.dumps(
                {"version": 1, "results": metadata},
                default=lambda value: value.item() if hasattr(value, "item") else list(value),
            )),
            **arrays,
        )
    os.replace(temporary, path)


def _load_results_file(path):
    with np.load(Path(path), allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        results = []
        for index, result in enumerate(metadata["results"]):
            result["state_channel_names"] = tuple(result["state_channel_names"])
            for key in DENSITY_KO_ARRAYS:
                name = f"{index}.{key}"
                result[key] = stored[name] if name in stored.files else None
            # Results from before cell numbers: densities are covered shares
            result.setdefault("density_unit", "share")
            result.setdefault("measured_covered_share", result["measured_density"])
            if result["covered_shares"] is None:
                result["covered_shares"] = result["densities"]
            results.append(result)
    return results


def load_density_ko_results(path):
    """Results written by ``save_density_ko_results``: one file, or a folder
    of ``part_*.npz`` files of a remote job (joined in part order)."""
    path = Path(path)
    if path.is_dir():
        files = part_paths(path)
        if not files:
            raise FileNotFoundError(f"No part_*.npz files in {path}")
    else:
        files = [path]
    return tuple(_result for _file in files for _result in _load_results_file(_file))


# ---------------------------------------------------------------- sweep


def _final_pattern_counts(
    model, states, keys, boundary, boundary_mode, total_steps, nodal_channel, ko_step,
    names, rules, mask, rings, max_batch,
):
    """Pattern counts ``[B, rings, patterns]`` of the final states of
    ``states`` ``[B, C, H, W]`` rolled out with ``keys`` ``[B]``, NODAL
    blocked from ``ko_step``. Rollouts run in chunks of ``max_batch``; the
    last chunk is padded, so every chunk uses the same compiled rollout."""
    from NCA.trainer.intervention import rollout_batch_with_blocked_channel_final

    counts = []
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
    return np.stack(counts)


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
    density_config = replace(
        InverseDesignDensityConfig(),
        split_distance=int(settings.split_distance),
        min_cell_area=int(settings.min_cell_area),
        layers=int(settings.layers),
    )
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
        bins = load_training_histogram_bins(bundle, root)
        data = [
            load_condition_data(bundle, root, "baseline", _replicate, bins)
            for _replicate in replicates
        ]
        colonies = stacked_colonies(data, replicates)
        # One circle for every replicate: the geometry stays fixed
        shape = colonies["support"].shape[-2:]
        mask = circle_mask(shape, float(np.mean(np.sum(colonies["support"], axis=(1, 2)))))
        # Nuclei of the raw 0h images (needed unless the sweep is in covered share)
        try:
            alignment, radius_scale, pixel_um = bundle_alignment(bundle)
            count_rows = reference_cell_counts(
                colonies, colonies["source_files"],
                root / "260726_nca_dataset", alignment,
                density_config.markers,
                density_config.split_distance, density_config.min_cell_area,
                mask_radius_scale=radius_scale,
                cache_dir=cache_dir_from_environment(),
            )
            mask_mm2 = float(np.count_nonzero(mask)) * pixel_um**2 * 1e-6
        except (OSError, ValueError, KeyError) as count_error:
            if unit != "share" or settings.piece_source == "nuclei":
                raise
            warnings.append(f"{bundle.id}: no nuclear counts ({count_error})")
            count_rows, mask_mm2 = None, float("nan")
        boundary = jnp.broadcast_to(
            jnp.asarray(mask, dtype=jnp.asarray(data[0]["boundary"]).dtype),
            jnp.asarray(data[0]["boundary"]).shape,
        )
        names = data[0]["state_channel_names"]
        channels = int(model.N_CHANNELS)
        steps_per_12h = int(bundle.config.run.t)
        total_steps = int(round(float(settings.snapshot_hour) * steps_per_12h / 12.0))
        ko_steps = [int(round(float(_hour) * steps_per_12h / 12.0)) for _hour in ko_hours]
        curriculum = tuple(bundle.config.data.knockout.curriculum or ("baseline",))
        label = record_label(record)
        for replicate_index, replicate in enumerate(replicates):
            library, measured_share = replicate_cell_library(
                colonies, replicate_index, density_config,
                nuclei=None if settings.piece_source == "segments" else {
                    "source_files": colonies["source_files"],
                    "dataset_root": root / "260726_nca_dataset",
                    "alignment": alignment,
                    "downsample": int(bundle.config.data.downsample),
                    "mask_radius_scale": radius_scale,
                    "expansion_um": float(settings.expansion_um),
                    "cache_dir": cache_dir_from_environment(),
                },
            )
            if count_rows is None:
                cell_counts = {"cells_per_mm2": float("nan"), "cells_per_segment": float("nan")}
            else:
                cell_counts = replicate_cell_counts(count_rows, replicate)
            # Real cells per placed piece: 1 for real cells, else measured cells per segment
            per_piece = library.get("cells_per_segment", cell_counts["cells_per_segment"])
            # Measured density of the replicate, in unit
            measured_density = {
                "share": measured_share,
                "per_mm2": cell_counts["cells_per_mm2"],
                "per_colony": cell_counts["cells_per_mm2"] * mask_mm2,
            }[unit]
            # Requested densities as real cells in the circle
            targets = densities * mask_mm2 if unit == "per_mm2" else densities

            # Place the cells of every run, then roll them all out together
            placed = []
            for run_index in range(runs):
                run_seed = int(settings.seed) + job_index * runs + run_index
                # Layers are added until the highest density fits
                packed, gaps, layer_pieces = placed_cell_texture(
                    library, colonies["schema"], shape, density_config, run_seed,
                    highest_density=float(densities[-1]) if unit == "share" else 0.0,
                    mask=mask,
                    softness_cells=float(settings.softness_cells),
                    highest_cells=0.0 if unit == "share" else float(targets[-1]) / per_piece,
                )
                coverage = np.asarray(gaps.layer_coverage)
                capacity = float(np.sum(layer_pieces)) * per_piece
                if unit == "share":
                    shares = np.asarray(densities, dtype=float)
                    max_density = float(gaps.max_density)
                else:
                    shares = np.asarray([
                        cells_to_density(_n, layer_pieces, coverage, per_piece) for _n in targets
                    ])
                    max_density = capacity / (mask_mm2 if unit == "per_mm2" else 1.0)
                if densities[-1] > max_density:
                    warnings.append(
                        f"{label}, replicate {replicate}, run {run_index}: "
                        f"{int(coverage.shape[0])} layers hold at most "
                        f"{format_density(max_density, unit)}"
                    )
                shares = np.clip(shares, 1e-3, min(float(gaps.max_density), 0.999))
                # Nuclei in the circle at each covered share used
                cells = np.asarray([
                    density_to_cells(_d, layer_pieces, coverage, per_piece) for _d in shares
                ])
                initial_states = np.stack([
                    np.asarray(gaps.apply(packed, CellDensity(float(_d)), jr.PRNGKey(run_seed)))
                    * mask[None]
                    for _d in shares
                ])
                placed.append({
                    "run_index": run_index,
                    "seed": run_seed,
                    "shares": shares,
                    "cells": cells,
                    "max_density": max_density,
                    "initial_states": initial_states,
                    "cell_layers": int(gaps.layer_coverage.shape[0]),
                })

            # [runs × densities, C, H, W], padded with zero channels to the model's
            states = np.concatenate([_run["initial_states"] for _run in placed])
            states = np.pad(states, ((0, 0), (0, channels - states.shape[1]), (0, 0), (0, 0)))
            keys = jnp.concatenate([
                jnp.broadcast_to(jr.PRNGKey(_run["seed"]), (len(densities), 2)) for _run in placed
            ])
            # [KO times, runs × densities, rings, patterns]
            counts = []
            for ko_step in ko_steps:
                counts.append(_final_pattern_counts(
                    model, states, keys, boundary, bundle.config.trainer.boundary_mode,
                    total_steps, data[0]["nodal_channel"], ko_step,
                    names, rules, mask, rings, settings.max_batch,
                ))
                progress(states.shape[0])
            # [runs, densities, KO times, rings, patterns]
            counts = np.stack(counts, axis=1).reshape(
                runs, len(densities), len(ko_steps), *counts[0].shape[1:]
            )

            for run, run_counts in zip(placed, counts):
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
                    "piece_source": settings.piece_source,
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
                    # [densities, KO times, rings, patterns]
                    "pattern_counts": run_counts,
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
                    # Its softness is unused: the cut is softness_cells wide
                    "density_config": asdict(density_config),
                    "softness_cells": float(settings.softness_cells),
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
