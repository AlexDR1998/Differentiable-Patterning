"""Colony size × cell density sweep of trained micropattern NCAs, with radial kymographs.

Circular colonies of several diameters are filled with whole measured cells
at a range of densities (``synthetic_ic.py``, set up by
``density_ko_sweep.prepare_model_cells`` as in the density × KO sweep) and
rolled out, optionally with NODAL blocked from given hours. Every size is
drawn on one grid per model, at the model's own µm per pixel, so a
1000 µm colony is twice as wide as a 500 µm one (the models were trained on
500 µm colonies only).

Kept per run:

- cell-type pattern counts and channel means of the final state per
  relative ring (as in the density × KO sweep), and
- radial profiles over time: the mean of each state channel and the marker
  pattern counts in bins of distance from the colony centre (µm), every
  ``record_every_hours``. These give kymographs (radius × time).

Used by ``size_density_explorer.py``, which sets the sweep up, plots the
results and can export it as a remote job. Run one part of an exported job with:

    python Experiments/micropatterns/size_density_sweep.py --manifest <manifest.yaml> [--index i]

See ``docs/remote_notebook_jobs.md``.
"""

from __future__ import annotations

import argparse
import os
from warnings import catch_warnings, simplefilter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from Common.dataloader.cell_type_shares import combine_stains, rule_stains, stain_counts_from_patterns
from Common.dataloader.cell_types import label_names, load_cell_type_rules, marker_values, pattern_label_shares
from Experiments.micropatterns.density_ko_sweep import (
    circle_mask,
    format_density,
    model_pattern_counts,
    prepare_model_cells,
    record_label,
    ring_channel_means,
)
from Experiments.micropatterns.synthetic_ic import cell_states
from Experiments.micropatterns.texture_settings import CellSettings
from Experiments.notebook_jobs import (
    load_results_npz,
    part_path,
    read_job_entry,
    repository_path,
    save_results_npz,
)

# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class SizeDensitySettings:
    """Everything the sweep depends on, apart from the models.

    Densities are in ``density_unit``: ``per_mm2`` (real cells per mm², the
    same at every size) or ``share`` (covered share of the colony).
    ``cell_type_files`` maps a projection to its cell type file (relative to
    the repository root); ``cells`` holds each projection's ``CellSettings``
    (as dicts, from the texture settings file). ``data_root`` contains
    ``260726_nca_dataset``; ``data_root`` and ``store_root`` are replaced by
    ``DATA_PATH_BASE`` and ``MODEL_STORE_ROOT`` when a job runs remotely.
    """

    data_root: str
    store_root: str
    cell_type_files: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)
    diameters_um: tuple = (200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0)
    # The grid fits a colony of this diameter (0: the largest of diameters_um).
    # Set when a job is split by size, so every part uses the same grid.
    grid_diameter_um: float = 0.0
    rings: int = 12
    density_unit: str = "per_mm2"
    density_start: float = 1000.0
    density_stop: float = 8000.0
    density_count: int = 5
    # NODAL knockout hours; a baseline rollout (no knockout) is always run
    ko_hours: tuple = ()
    snapshot_hour: float = 48.0
    # Radial profiles: every this many hours, in bins this wide (at least one pixel)
    record_every_hours: float = 1.0
    profile_bin_um: float = 10.0
    replicates: tuple = (1,)
    seed: int = 0
    duplicate_runs: int = 2
    # Rollouts run on the GPU at once (runs × densities of one size and condition)
    max_batch: int = 32
    # References beside the sweep, rolled out with the same conditions and
    # run seeds: the true 0h control colony of each of ``replicates`` (its
    # own outline and size), and synthetic colonies of every size at the
    # measured density (the mean over the replicates)
    include_true_data: bool = False
    include_measured_density: bool = False
    # Replicates the measured density is the mean over (empty: replicates);
    # a job split by replicate keeps every replicate here
    measured_replicates: tuple = ()

    def densities(self):
        return np.linspace(float(self.density_start), float(self.density_stop), int(self.density_count))

    def conditions(self):
        """``(label, KO hour or None)`` of each rollout condition, baseline first."""
        return (("baseline", None), *((f"ko_{float(_h):g}h", float(_h)) for _h in self.ko_hours))

    def grid_diameter(self):
        return float(self.grid_diameter_um) or max(float(_d) for _d in self.diameters_um)

    def trajectories(self, models):
        """Number of rollouts for ``models`` models (references included)."""
        # Colonies per model and run: the sweep's, then one per reference
        colonies = len(self.replicates) * len(self.diameters_um) * len(self.densities())
        colonies += len(self.replicates) * len(self.diameters_um) if self.include_measured_density else 0
        colonies += len(self.replicates) if self.include_true_data else 0
        return models * int(self.duplicate_runs) * colonies * len(self.conditions())

    def check(self):
        if not str(self.data_root).strip():
            raise ValueError("Set Data root before running the size sweep.")
        if not self.diameters_um or min(self.diameters_um) <= 0:
            raise ValueError("Give at least one colony diameter, all above zero.")
        if self.grid_diameter_um and self.grid_diameter_um < max(self.diameters_um):
            raise ValueError("The grid diameter must fit the largest colony.")
        if self.density_unit not in ("per_mm2", "share"):
            raise ValueError("Sweep the density per mm² or as covered share, so sizes compare at equal density.")
        if self.density_stop < self.density_start:
            raise ValueError("Highest density must be at or above the lowest.")
        if self.density_unit != "share" and self.density_start <= 0:
            raise ValueError("The lowest density must be above zero cells.")
        if any(not 0.0 <= float(_h) <= float(self.snapshot_hour) for _h in self.ko_hours):
            raise ValueError("KO times must lie between 0h and the snapshot time.")
        if self.record_every_hours <= 0 or self.profile_bin_um <= 0:
            raise ValueError("The profile interval and bin width must be above zero.")
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
    """``SizeDensitySettings`` from a plain dict (as written into a job manifest)."""
    values = dict(values)
    for key in ("diameters_um", "ko_hours"):
        values[key] = tuple(float(_v) for _v in values.get(key, ()))
    values["replicates"] = tuple(int(_r) for _r in values.get("replicates", (1,)))
    values["measured_replicates"] = tuple(int(_r) for _r in values.get("measured_replicates", ()))
    # Job files that chose one true data replicate: it now comes from every replicate
    values.pop("true_data_replicate", None)
    values["cell_type_files"] = dict(values.get("cell_type_files") or {})
    values["cells"] = dict(values.get("cells") or {})
    return SizeDensitySettings(**values)


# ---------------------------------------------------------------- geometry


def sweep_grid_shape(data_shape, training_radius_px, largest_radius_px):
    """One odd square grid for every colony size of a model.

    At least the data grid, and wide enough to keep the training colony's
    margin (data grid half-width minus its radius) around the largest
    colony: padding is circular, so a colony too close to the edge would
    meet itself.
    """
    margin = max(0.0, min(data_shape) / 2.0 - float(training_radius_px))
    side = max(max(data_shape), int(np.ceil(2.0 * (float(largest_radius_px) + margin))))
    side += 1 - side % 2
    return (side, side)


def size_masks(shape, diameters_um, pixel_um):
    """Centred discs ``[sizes, H, W]`` of the given diameters on a grid ``shape``."""
    return np.stack([
        circle_mask(shape, np.pi * (float(_d) / 2.0 / pixel_um) ** 2) for _d in diameters_um
    ])


def radial_bins(shape, pixel_um, bin_um):
    """Bin of each pixel ``[H, W]`` by distance from the grid centre, in bins ``bin_um`` wide."""
    rows, cols = np.indices(shape)
    distance_um = np.hypot(rows - (shape[0] - 1) / 2.0, cols - (shape[1] - 1) / 2.0) * pixel_um
    return (distance_um // bin_um).astype(np.int32)


def sweep_geometry(prepared, settings):
    """Grid, colony discs and radial bins of one model (``prepare_model_cells``) for ``settings``.

    One grid for every size, fitting ``settings.grid_diameter()``. Returns a
    dict with ``shape``, ``masks`` ``[sizes, H, W]``, ``distance_bins``
    ``[H, W]``, ``n_bins`` (bins reaching the largest colony), ``radii_um``
    (bin centres) and ``bin_um`` (``profile_bin_um``, at least one pixel).
    """
    pixel_um = prepared["pixel_um"]
    support = prepared["colonies"]["support"]
    training_radius_px = float(np.sqrt(np.mean(np.sum(support, axis=(1, 2))) / np.pi))
    shape = sweep_grid_shape(support.shape[-2:], training_radius_px, settings.grid_diameter() / 2.0 / pixel_um)
    masks = size_masks(shape, settings.diameters_um, pixel_um)
    bin_um = max(float(settings.profile_bin_um), pixel_um)
    distance_bins = radial_bins(shape, pixel_um, bin_um)
    n_bins = int(distance_bins[masks.any(axis=0)].max()) + 1
    return {
        "shape": shape,
        "masks": masks,
        "distance_bins": distance_bins,
        "n_bins": n_bins,
        "radii_um": (np.arange(n_bins) + 0.5) * bin_um,
        "bin_um": bin_um,
        "training_diameter_um": 2.0 * training_radius_px * pixel_um,
    }


def size_initial_states(prepared, replicate_index, mask, densities, unit, seed):
    """Synthetic initial states of one replicate's cells in ``mask``, one per density.

    ``prepared`` comes from ``prepare_model_cells``. Returns the dict of
    ``synthetic_ic.cell_states``, plus ``mask_mm2``, the ``targets`` it was
    asked for and ``max_density`` (what the placement holds, in ``unit``).
    """
    replicate = prepared["replicates"][replicate_index]
    mask_mm2 = float(np.count_nonzero(mask)) * prepared["pixel_um"] ** 2 * 1e-6
    targets = np.asarray(densities, dtype=float) * (mask_mm2 if unit == "per_mm2" else 1.0)
    states = cell_states(
        replicate["library"], prepared["colonies"]["schema"], mask, targets, unit == "share",
        int(seed), prepared["cell_settings"], replicate["per_piece"],
    )
    states["mask_mm2"] = mask_mm2
    states["targets"] = targets
    states["max_density"] = states["capacity"] / (mask_mm2 if unit == "per_mm2" else 1.0)
    return states


def marker_channels(names, cell_type_rules):
    """Index of each rule marker (in ``cell_type_rules.markers`` order) among the state channels ``names``."""
    index = marker_values(np.arange(len(names)), names, cell_type_rules, axis=0)
    return [int(index[_marker]) for _marker in cell_type_rules.markers]


# ---------------------------------------------------------------- results files

# Arrays of a size × density result; everything else is saved as JSON
SIZE_DENSITY_ARRAYS = (
    "diameters_um",
    "colony_mm2",
    "densities",
    "covered_shares",
    "cells_per_mm2",
    "cells_per_colony",
    "max_density",
    "ko_hours",
    "realised_ko_hours",
    "pattern_counts",
    "cell_pattern_counts",
    "channel_means",
    "profiles",
    "profile_counts",
    "profile_hours",
    "profile_radii_um",
    "colony_masks",
    "initial_states",
    "final_states",
)


def save_size_density_results(path, results):
    """Write size × density results to one ``.npz`` file."""
    save_results_npz(path, results, SIZE_DENSITY_ARRAYS)


def load_size_density_results(path):
    """Results of ``save_size_density_results``: one file, or a folder of a remote job's parts."""
    results = load_results_npz(path, SIZE_DENSITY_ARRAYS)
    for result in results:
        result["state_channel_names"] = tuple(result["state_channel_names"])
        result["conditions"] = tuple(result["conditions"])
        # Results from before the references are all sweep results
        result.setdefault("source", "sweep")
    return results


# ---------------------------------------------------------------- sweep


def _profile_rollouts(
    model, states, keys, boundary, boundary_mode, total_steps, nodal_channel, ko_step,
    n_observed, fate_channels, thresholds, groups, n_bins, record_steps, max_batch,
):
    """Rollouts of ``states`` ``[B, C, H, W]`` in chunks of ``max_batch``.

    Returns numpy ``(final states [B, n_observed, H, W], profiles [B,
    records, bins, n_observed], counts [B, records, bins, patterns])`` at
    ``record_steps``. The last chunk is padded, so every chunk uses the same
    compiled rollout.
    """
    from NCA.trainer.intervention import rollout_batch_with_blocked_channel_profiles

    finals, profiles, counts = [], [], []
    batch = max(1, min(int(max_batch), states.shape[0]))
    for start in range(0, states.shape[0], batch):
        chunk = np.arange(start, min(start + batch, states.shape[0]))
        padded = np.pad(chunk, (0, batch - len(chunk)), mode="edge")
        final, profile, count = rollout_batch_with_blocked_channel_profiles(
            model,
            jnp.asarray(states[padded]),
            boundary,
            boundary_mode,
            keys[padded],
            total_steps,
            nodal_channel,
            jnp.asarray(ko_step, dtype=jnp.int32),
            n_observed,
            jnp.asarray(fate_channels, dtype=jnp.int32),
            jnp.asarray(thresholds, dtype=jnp.float32),
            jnp.asarray(groups),
            n_bins,
        )
        finals.append(np.asarray(final)[: len(chunk), :n_observed])
        profiles.append(np.asarray(profile)[: len(chunk)][:, record_steps])
        counts.append(np.asarray(count)[: len(chunk)][:, record_steps])
    return np.concatenate(finals), np.concatenate(profiles), np.concatenate(counts)


def centred_on_grid(image, shape):
    """``image`` ``[..., h, w]`` padded with zeros to ``shape`` ``(H, W)``, centred."""
    image = np.asarray(image)
    rows, cols = shape[0] - image.shape[-2], shape[1] - image.shape[-1]
    if rows < 0 or cols < 0:
        raise ValueError(f"An image of {image.shape[-2:]} pixels does not fit the grid {tuple(shape)}.")
    pad = [(0, 0)] * (image.ndim - 2) + [(rows // 2, rows - rows // 2), (cols // 2, cols - cols // 2)]
    return np.pad(image, pad)


def run_size_density_sweep(settings, records, job_offset=0, progress=None):
    """``(results, warnings)`` of the sweep for the models in ``records``.

    ``records`` are model-selection records (``load_selection``, with
    ``projection`` and ``downsample``). Each (model, replicate) pair is one
    job; run seeds are ``seed + job * duplicate_runs + run``, counting jobs
    from ``job_offset``, and the same at every size, so a sweep split into
    parts keeps the seeds of a single run. ``progress(n)`` is called after
    every ``n`` rollouts. Results are the dicts
    ``save_size_density_results`` writes, one per (model, replicate, run)
    and ``source``, with arrays ``[sizes, densities, conditions, ...]``.
    ``source`` is ``sweep``, or a reference the settings ask for:
    ``measured_density`` (every size at the mean measured density, one
    density, the sweep's seeds) or ``true_data`` (each replicate's true
    colony, one size: its own outline, centred on the grid, with radial bins
    over the whole grid; that replicate's seeds; replicates whose colony
    cannot be loaded are skipped with a warning).
    """
    from Experiments.micropatterns.density_ko_sweep import (
        mean_measured_density,
        measured_density_cells,
        true_data_colony,
    )
    from Experiments.model_registry import ModelRegistry

    settings.check()
    unit = settings.density_unit
    densities = settings.densities()
    diameters = np.asarray(settings.diameters_um, dtype=float)
    conditions = settings.conditions()
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
        pixel_um = prepared["pixel_um"]
        if not np.isfinite(pixel_um):
            raise ValueError(f"{bundle.id}: the size sweep needs the pixel size (nuclear counts failed).")

        geometry = sweep_geometry(prepared, settings)
        shape, masks, distance_bins = geometry["shape"], geometry["masks"], geometry["distance_bins"]
        bin_um = geometry["bin_um"]

        names = prepared["names"]
        n_observed = len(names)
        fate_channels = marker_channels(names, rules)
        thresholds = [float(rules.thresholds[_marker]) for _marker in rules.markers]
        channels = int(model.N_CHANNELS)
        boundary_mode = bundle.config.trainer.boundary_mode
        steps_per_12h = int(bundle.config.run.t)
        total_steps = int(round(float(settings.snapshot_hour) * steps_per_12h / 12.0))
        record_steps = np.unique(np.round(
            np.arange(0.0, float(settings.snapshot_hour) + 1e-9, float(settings.record_every_hours))
            * steps_per_12h / 12.0
        ).astype(int))
        record_steps = np.unique(np.append(record_steps[record_steps <= total_steps], total_steps))
        # Baseline: a knockout step after the end, so it never happens
        ko_steps = [
            total_steps + 1 if _hour is None else int(round(_hour * steps_per_12h / 12.0))
            for _, _hour in conditions
        ]
        curriculum = tuple(bundle.config.data.knockout.curriculum or ("baseline",))
        label = record_label(record)
        # Measured density of the real colonies, mean over the replicates
        measured_cells, measured_replicates = (
            measured_density_cells(bundle, settings, projection, root, prepared)
            if settings.include_true_data or settings.include_measured_density else (prepared, replicates)
        )
        mean_per_mm2, mean_share = mean_measured_density(measured_cells)

        def roll_sizes(_masks, _states_of, _seeds, n_bins=None):
            """Rollouts of every run's initial states at every size in ``_masks``.

            ``_states_of(size index, mask, seed)`` gives the dict of
            ``size_initial_states`` (``cell_weights`` may be None). Profiles
            have ``n_bins`` radial bins (default: those reaching the largest
            mask). Returns ``(per_run, n_bins, radii_um)``: arrays ``[runs, sizes, ...]``.
            """
            _n_bins = int(n_bins or int(distance_bins[np.any(_masks, axis=0)].max()) + 1)
            _by_size = []
            for _size_index, _mask in enumerate(_masks):
                _placed = [{**_states_of(_size_index, _mask, _seed), "seed": _seed} for _seed in _seeds]
                _points = len(_placed[0]["initial_states"])
                # [runs × points, C, H, W], padded with zero channels to the model's
                _initial = np.concatenate([_run["initial_states"] for _run in _placed])
                _padded = np.pad(_initial, ((0, 0), (0, channels - _initial.shape[1]), (0, 0), (0, 0)))
                _keys = jnp.concatenate([
                    jnp.broadcast_to(jr.PRNGKey(_run["seed"]), (_points, 2)) for _run in _placed
                ])
                _weights = (
                    None if _placed[0]["cell_weights"] is None
                    else np.concatenate([_run["cell_weights"] for _run in _placed])
                )
                _boundary = jnp.asarray(_mask[None], dtype=prepared["boundary"].dtype)
                _groups = np.where(_mask & (distance_bins < _n_bins), distance_bins, -1)

                _per_condition = []
                for _ko_step in ko_steps:
                    _final, _profiles, _counts = _profile_rollouts(
                        model, _padded, _keys, _boundary, boundary_mode, total_steps,
                        prepared["nodal_channel"], _ko_step, n_observed, fate_channels, thresholds,
                        _groups, _n_bins, record_steps, settings.max_batch,
                    )
                    _kept = {
                        "final_states": _final.astype(np.float16),
                        "pattern_counts": np.stack([
                            model_pattern_counts(_state, names, rules, _mask, rings) for _state in _final
                        ]),
                        "channel_means": np.stack([
                            ring_channel_means(_state, _mask, rings) for _state in _final
                        ]),
                        "profiles": _profiles.astype(np.float16),
                        "profile_counts": _counts.astype(np.uint32),
                    }
                    if _weights is not None:
                        _kept["cell_pattern_counts"] = np.stack([
                            model_pattern_counts(_state, names, rules, _mask, rings, weights=_w)
                            for _state, _w in zip(_final, _weights)
                        ])
                    _per_condition.append(_kept)
                    progress(_padded.shape[0])

                # [runs, points, conditions, ...]
                _stacked = {
                    _key: np.stack([_c[_key] for _c in _per_condition], axis=1).reshape(
                        len(_seeds), _points, len(ko_steps), *_per_condition[0][_key].shape[1:]
                    )
                    for _key in _per_condition[0]
                }
                _stacked["initial_states"] = _initial.astype(np.float16).reshape(
                    len(_seeds), _points, *_initial.shape[1:]
                )
                _stacked["covered_shares"] = np.stack([_run["shares"] for _run in _placed])
                _stacked["cells_per_colony"] = np.stack([_run["cells"] for _run in _placed])
                _stacked["max_density"] = np.asarray([_run["max_density"] for _run in _placed])
                _stacked["colony_mm2"] = np.full(len(_seeds), _placed[0]["mask_mm2"])
                _by_size.append(_stacked)
            _per_run = {_key: np.stack([_s[_key] for _s in _by_size], axis=1) for _key in _by_size[0]}
            return _per_run, _n_bins, (np.arange(_n_bins) + 0.5) * bin_um

        def add_results(_source, _replicate, _seeds, _rolled, _diameters, _masks, _densities, _measured, _images):
            """One result per run; ``_images``: keep the images of the first run."""
            _per_run, _n_bins, _radii_um = _rolled
            for _run_index, _seed in enumerate(_seeds):
                _keep = _images and _run_index == 0
                results.append({
                    "model_id": bundle.id,
                    "label": label,
                    "projection": projection,
                    "downsample": record["downsample"],
                    "replicate": _replicate,
                    "run_index": _run_index,
                    "seed": int(_seed),
                    # sweep, or a reference: measured_density or true_data
                    "source": _source,
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
                    # Sizes: [S]; grid and pixel size shared by every size
                    "diameters_um": np.asarray(_diameters, dtype=float),
                    "colony_mm2": _per_run["colony_mm2"][_run_index],
                    # Diameter of the training colonies' discs (nominally 500 µm)
                    "training_diameter_um": geometry["training_diameter_um"],
                    "pixel_um": pixel_um,
                    "grid_shape": list(shape),
                    "colony_masks": _masks,
                    # Requested densities (in density_unit) and what each size got: [S, D]
                    "densities": np.asarray(_densities, dtype=float),
                    "density_unit": unit,
                    "covered_shares": _per_run["covered_shares"][_run_index],
                    "cells_per_colony": _per_run["cells_per_colony"][_run_index],
                    "cells_per_mm2": (
                        _per_run["cells_per_colony"][_run_index] / _per_run["colony_mm2"][_run_index][:, None]
                    ),
                    "max_density": _per_run["max_density"][_run_index],
                    "measured_cells_per_mm2": _measured["cells_per_mm2"],
                    "measured_covered_share": _measured["share"],
                    "measured_density": {"share": _measured["share"], "per_mm2": _measured["cells_per_mm2"]}[unit],
                    "placement": cell_settings.placement,
                    "cell_settings": cell_settings.to_dict(),
                    # Conditions: [K]; NaN KO hour for baseline
                    "conditions": [_label for _label, _ in conditions],
                    "ko_hours": np.asarray([np.nan if _h is None else _h for _, _h in conditions]),
                    "realised_ko_hours": np.asarray([
                        np.nan if _h is None else 12.0 * _step / steps_per_12h
                        for (_, _h), _step in zip(conditions, ko_steps)
                    ]),
                    "snapshot_hour": float(settings.snapshot_hour),
                    # Final state per relative ring: [S, D, K, rings, patterns / channels]
                    "pattern_counts": _per_run["pattern_counts"][_run_index],
                    "cell_pattern_counts": (
                        _per_run["cell_pattern_counts"][_run_index] if "cell_pattern_counts" in _per_run else None
                    ),
                    "channel_means": _per_run["channel_means"][_run_index],
                    # Kymographs: [S, D, K, T, bins, channels / patterns], bins of
                    # distance from the centre (µm); NaN / 0 outside the colony
                    "profiles": _per_run["profiles"][_run_index],
                    "profile_counts": _per_run["profile_counts"][_run_index],
                    "profile_hours": 12.0 * record_steps / steps_per_12h,
                    "profile_radii_um": _radii_um,
                    "bin_um": bin_um,
                    # Images, for one run only (they are large)
                    "initial_states": _per_run["initial_states"][_run_index] if _keep else None,
                    "final_states": _per_run["final_states"][_run_index] if _keep else None,
                    "state_channel_names": names,
                    # What the pattern counts depend on, checked when loading
                    "cell_type_markers": list(rules.markers),
                    "cell_type_thresholds": {
                        _marker: float(rules.thresholds[_marker]) for _marker in rules.markers
                    },
                    "rings": rings,
                })

        for replicate_index, replicate in enumerate(replicates):
            prepared_replicate = prepared["replicates"][replicate_index]
            measured = {
                "cells_per_mm2": prepared_replicate["cell_counts"]["cells_per_mm2"],
                "share": prepared_replicate["measured_share"],
            }
            seeds = [int(settings.seed) + job_index * runs + _run for _run in range(runs)]

            def sweep_states(_size_index, _mask, _seed):
                _states = size_initial_states(prepared, replicate_index, _mask, densities, unit, _seed)
                _where = f"{label}, {diameters[_size_index]:g} µm, replicate {replicate}, seed {_seed}"
                if densities[-1] > _states["max_density"]:
                    warnings.append(
                        f"{_where}: the {cell_settings.placement} cells hold at most "
                        f"{format_density(_states['max_density'], unit)}"
                    )
                if _states["squeeze"] < 0.5:
                    warnings.append(
                        f"{_where}: at the highest density cells have "
                        f"{_states['squeeze']:.0%} of the measured cell area"
                    )
                return _states

            add_results(
                "sweep", replicate, seeds, roll_sizes(masks, sweep_states, seeds),
                diameters, masks, densities, measured, not results,
            )

            # Reference: the same cells at the measured density (mean over the replicates), every size
            if settings.include_measured_density:
                reference_density = mean_share if unit == "share" else mean_per_mm2
                if not np.isfinite(reference_density):
                    warnings.append(f"{label}: no measured density (nuclei not counted); skipped that reference")
                else:
                    add_results(
                        "measured_density", replicate, seeds,
                        roll_sizes(
                            masks,
                            lambda _s, _mask, _seed: size_initial_states(
                                prepared, replicate_index, _mask, [reference_density], unit, _seed,
                            ),
                            seeds,
                        ),
                        diameters, masks, [reference_density], measured, replicate_index == 0,
                    )

            # Reference: this replicate's true colony, in its own outline and size, with the sweep's seeds
            if settings.include_true_data:
                try:
                    true = true_data_colony(bundle, root, measured_cells, replicate, measured_replicates)
                except (OSError, ValueError, KeyError, IndexError) as error:
                    warnings.append(f"{label}: no true data for replicate {replicate} ({error}); skipped it")
                    true = None
                if true is not None:
                    true_mask = centred_on_grid(true["mask"], shape).astype(bool)
                    true_state = centred_on_grid(true["initial_state"], shape)
                    true_cells = true["cells_per_mm2"] * true["mask_mm2"]

                    def true_states(_s, _mask, _seed, _true=true, _state=true_state, _cells=true_cells):
                        return {
                            "initial_states": _state[None],
                            "cell_weights": None,
                            "shares": np.asarray([_true["covered_share"]]),
                            "cells": np.asarray([_cells]),
                            # Nothing was placed, so nothing was cut down
                            "max_density": _true["covered_share"] if unit == "share" else _true["cells_per_mm2"],
                            "mask_mm2": _true["mask_mm2"],
                        }

                    add_results(
                        "true_data", replicate, seeds,
                        # Bins over the whole grid, so every replicate's colony has the same bins
                        roll_sizes(true_mask[None], true_states, seeds, n_bins=int(distance_bins.max()) + 1),
                        # The diameter of a disc of the colony's area
                        [2.0 * np.sqrt(true["mask_mm2"] / np.pi) * 1000.0],
                        true_mask[None],
                        [true["covered_share"] if unit == "share" else true["cells_per_mm2"]],
                        {"cells_per_mm2": true["cells_per_mm2"], "share": true["covered_share"]},
                        True,
                    )
            job_index += 1
    return tuple(results), warnings


# ---------------------------------------------------------------- analysis


def profile_positions(result, size_index, axis="edge_um"):
    """Position of each radial bin of ``result`` for one colony size.

    ``axis``: ``edge_um`` (distance from the colony edge, µm), ``centre_um``
    (from the centre) or ``relative`` (fraction of the colony radius).
    """
    radius = float(result["diameters_um"][size_index]) / 2.0
    # Each bin at the middle of its part inside the colony, so the edge bin
    # does not reach past the edge
    inner = np.asarray(result["profile_radii_um"], dtype=float) - 0.5 * float(result["bin_um"])
    outer = np.minimum(inner + float(result["bin_um"]), max(radius, 0.0))
    centres = np.where(outer > inner, 0.5 * (inner + outer), inner)
    return {
        "edge_um": radius - centres,
        "centre_um": centres,
        "relative": centres / radius,
    }[axis]


def ring_label_shares(counts, cell_type_rules, resolve_several=0.0):
    """Label shares ``[..., bins, labels]`` (``label_names`` order) in each
    bin, from marker pattern counts ``[..., bins, patterns]`` of model
    states. Stains are combined as for whole colonies
    (``colony_label_shares``), but bins are not averaged. NaN for empty bins.
    """
    counts = np.asarray(counts, dtype=float)
    stains = [tuple(_channels) for _channels in rule_stains(cell_type_rules).values()]
    to_labels = pattern_label_shares(cell_type_rules, resolve_several=resolve_several)
    flat = counts.reshape(-1, *counts.shape[-2:])
    shares = np.stack([
        combine_stains(
            stain_counts_from_patterns(_counts, cell_type_rules.markers, stains), cell_type_rules.markers
        ) @ to_labels
        for _counts in flat
    ])
    return shares.reshape(*counts.shape[:-1], shares.shape[-1])


def kymograph_values(result, quantity, cell_type_rules=None, resolve_several=0.0):
    """``[S, D, K, T, bins]`` of one quantity: a state channel name (its
    mean), or a cell type label (its share; needs ``cell_type_rules``)."""
    names = list(result["state_channel_names"])
    if quantity in names:
        return np.asarray(result["profiles"], dtype=np.float32)[..., names.index(quantity)]
    if cell_type_rules is None:
        raise ValueError(f"{quantity!r} is not a state channel; give the cell type rules for cell types.")
    labels = list(label_names(cell_type_rules))
    shares = ring_label_shares(result["profile_counts"], cell_type_rules, resolve_several)
    return shares[..., labels.index(quantity)]


def mean_kymograph(results, quantity, cell_type_rules=None, resolve_several=0.0):
    """``[S, D, K, T, bins]`` of one quantity over several runs of one model.

    Channel means are averaged over the runs; cell type shares are
    estimated from the pattern counts of all runs pooled (``kymograph_values``).
    """
    if quantity in results[0]["state_channel_names"]:
        with catch_warnings():
            # Bins outside smaller colonies are empty (NaN) in every run
            simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack([kymograph_values(_r, quantity) for _r in results]), axis=0)
    pooled = {**results[0], "profile_counts": np.sum([_r["profile_counts"] for _r in results], axis=0)}
    return kymograph_values(pooled, quantity, cell_type_rules, resolve_several)


def front_positions(values, positions, level):
    """Where each profile first crosses ``level``, ``[T]`` (NaN where it does not).

    ``values`` is ``[T, bins]`` and ``positions`` ``[bins]``; profiles are
    read in order of increasing position (e.g. distance from the edge, so
    the front nearest the edge is found) and interpolated linearly between
    bins. NaN bins are skipped.
    """
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    order = np.argsort(positions)
    fronts = np.full(values.shape[0], np.nan)
    for time, profile in enumerate(values[:, order]):
        keep = np.isfinite(profile)
        value, position = profile[keep], positions[order][keep]
        above = value > level
        crossings = np.flatnonzero(above[:-1] != above[1:])
        if len(crossings):
            i = crossings[0]
            fronts[time] = position[i] + (level - value[i]) * (position[i + 1] - position[i]) / (value[i + 1] - value[i])
    return fronts


def front_speed(hours, fronts, window=None):
    """Slope of ``fronts`` over ``hours`` (position per hour) by least squares,
    within ``window`` ``(start, end)`` hours. NaN with fewer than two fronts."""
    hours = np.asarray(hours, dtype=float)
    fronts = np.asarray(fronts, dtype=float)
    keep = np.isfinite(fronts)
    if window is not None:
        keep &= (hours >= window[0]) & (hours <= window[1])
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(hours[keep], fronts[keep], 1)[0])


# ---------------------------------------------------------------- remote jobs


def job_configs(settings, records, output_dir, split="model"):
    """One manifest entry per model (``split="model"``), per (model,
    replicate) pair (``"replicate"``) or per (model, replicate, size)
    (``"size"``), each writing a part into ``output_dir``. ``job_offset``
    and the grid diameter keep the seeds and grid of a single run. Every
    part averages the measured density over all the replicates, and runs
    the true data of its own replicates (in the first size part only)."""
    configs = []
    job_offset = 0
    settings = replace(
        settings, grid_diameter_um=settings.grid_diameter(),
        measured_replicates=tuple(settings.measured_replicates) or tuple(settings.replicates),
    )
    for record in records:
        replicate_groups = (
            [(_r,) for _r in settings.replicates] if split in ("replicate", "size")
            else [tuple(settings.replicates)]
        )
        size_groups = (
            [(_d,) for _d in settings.diameters_um] if split == "size"
            else [tuple(settings.diameters_um)]
        )
        for group in replicate_groups:
            for size_index, sizes in enumerate(size_groups):
                configs.append({
                    "settings": asdict(replace(
                        settings, replicates=group, diameters_um=sizes,
                        include_true_data=settings.include_true_data and size_index == 0,
                    )),
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
        results, warnings = run_size_density_sweep(
            settings, records, job_offset=int(config["job_offset"]), progress=bar.update
        )
    for warning in warnings:
        print(f"Warning: {warning}")
    save_size_density_results(output, results)
    print(f"Saved part {index} to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run one part of a size × density sweep job")
    parser.add_argument("--manifest", required=True, help="Job manifest exported by size_density_explorer.py")
    parser.add_argument("--index", type=int, default=None, help="Part to run (default: from the job environment)")
    parser.add_argument("--overwrite", action="store_true", help="Rerun a part whose results file exists")
    args = parser.parse_args()
    index, config = read_job_entry(args.manifest, args.index)
    run_job_part(config, index, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
