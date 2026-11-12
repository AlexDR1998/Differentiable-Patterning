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

    def densities(self):
        return np.linspace(float(self.density_start), float(self.density_stop), int(self.density_count))

    def conditions(self):
        """``(label, KO hour or None)`` of each rollout condition, baseline first."""
        return (("baseline", None), *((f"ko_{float(_h):g}h", float(_h)) for _h in self.ko_hours))

    def grid_diameter(self):
        return float(self.grid_diameter_um) or max(float(_d) for _d in self.diameters_um)

    def trajectories(self, models):
        """Number of rollouts for ``models`` models."""
        return (
            models * len(self.replicates) * int(self.duplicate_runs)
            * len(self.diameters_um) * len(self.densities()) * len(self.conditions())
        )

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


def run_size_density_sweep(settings, records, job_offset=0, progress=None):
    """``(results, warnings)`` of the sweep for the models in ``records``.

    ``records`` are model-selection records (``load_selection``, with
    ``projection`` and ``downsample``). Each (model, replicate) pair is one
    job; run seeds are ``seed + job * duplicate_runs + run``, counting jobs
    from ``job_offset``, and the same at every size, so a sweep split into
    parts keeps the seeds of a single run. ``progress(n)`` is called after
    every ``n`` rollouts. Results are the dicts
    ``save_size_density_results`` writes, one per (model, replicate, run),
    with arrays ``[sizes, densities, conditions, ...]``.
    """
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
        n_bins, radii_um, bin_um = geometry["n_bins"], geometry["radii_um"], geometry["bin_um"]

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

        for replicate_index, replicate in enumerate(replicates):
            cell_counts = prepared["replicates"][replicate_index]["cell_counts"]
            measured_share = prepared["replicates"][replicate_index]["measured_share"]
            # [runs, sizes, ...] of everything kept per run
            by_size = []
            for size_index, mask in enumerate(masks):
                placed = []
                for run_index in range(runs):
                    run_seed = int(settings.seed) + job_index * runs + run_index
                    states = size_initial_states(prepared, replicate_index, mask, densities, unit, run_seed)
                    where = f"{label}, {diameters[size_index]:g} µm, replicate {replicate}, run {run_index}"
                    if densities[-1] > states["max_density"]:
                        warnings.append(
                            f"{where}: the {cell_settings.placement} cells hold at most "
                            f"{format_density(states['max_density'], unit)}"
                        )
                    if states["squeeze"] < 0.5:
                        warnings.append(
                            f"{where}: at the highest density cells have "
                            f"{states['squeeze']:.0%} of the measured cell area"
                        )
                    placed.append({**states, "seed": run_seed})

                # [runs × densities, C, H, W], padded with zero channels to the model's
                initial = np.concatenate([_run["initial_states"] for _run in placed])
                padded_states = np.pad(initial, ((0, 0), (0, channels - initial.shape[1]), (0, 0), (0, 0)))
                keys = jnp.concatenate([
                    jnp.broadcast_to(jr.PRNGKey(_run["seed"]), (len(densities), 2)) for _run in placed
                ])
                weights = np.concatenate([_run["cell_weights"] for _run in placed])
                boundary = jnp.asarray(mask[None], dtype=prepared["boundary"].dtype)
                groups = np.where(mask & (distance_bins < n_bins), distance_bins, -1)

                per_condition = []
                for ko_step in ko_steps:
                    final, profiles, counts = _profile_rollouts(
                        model, padded_states, keys, boundary, boundary_mode, total_steps,
                        prepared["nodal_channel"], ko_step, n_observed, fate_channels, thresholds,
                        groups, n_bins, record_steps, settings.max_batch,
                    )
                    per_condition.append({
                        "final_states": final.astype(np.float16),
                        "pattern_counts": np.stack([
                            model_pattern_counts(_state, names, rules, mask, rings) for _state in final
                        ]),
                        "cell_pattern_counts": np.stack([
                            model_pattern_counts(_state, names, rules, mask, rings, weights=_weights)
                            for _state, _weights in zip(final, weights)
                        ]),
                        "channel_means": np.stack([
                            ring_channel_means(_state, mask, rings) for _state in final
                        ]),
                        "profiles": profiles.astype(np.float16),
                        "profile_counts": counts.astype(np.uint32),
                    })
                    progress(padded_states.shape[0])

                # [runs, densities, conditions, ...]
                stacked = {
                    _key: np.stack([_c[_key] for _c in per_condition], axis=1).reshape(
                        runs, len(densities), len(ko_steps), *per_condition[0][_key].shape[1:]
                    )
                    for _key in per_condition[0]
                }
                stacked["initial_states"] = initial.astype(np.float16).reshape(
                    runs, len(densities), *initial.shape[1:]
                )
                stacked["covered_shares"] = np.stack([_run["shares"] for _run in placed])
                stacked["cells_per_colony"] = np.stack([_run["cells"] for _run in placed])
                stacked["max_density"] = np.asarray([_run["max_density"] for _run in placed])
                stacked["colony_mm2"] = np.full(runs, placed[0]["mask_mm2"])
                stacked["seed"] = np.asarray([_run["seed"] for _run in placed])
                by_size.append(stacked)

            # [runs, sizes, ...]
            per_run = {_key: np.stack([_s[_key] for _s in by_size], axis=1) for _key in by_size[0]}
            for run_index in range(runs):
                first = not results
                results.append({
                    "model_id": bundle.id,
                    "label": label,
                    "projection": projection,
                    "downsample": record["downsample"],
                    "replicate": replicate,
                    "run_index": run_index,
                    "seed": int(per_run["seed"][run_index, 0]),
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
                    "diameters_um": diameters,
                    "colony_mm2": per_run["colony_mm2"][run_index],
                    # Diameter of the training colonies' discs (nominally 500 µm)
                    "training_diameter_um": geometry["training_diameter_um"],
                    "pixel_um": pixel_um,
                    "grid_shape": list(shape),
                    "colony_masks": masks,
                    # Requested densities (in density_unit) and what each size got: [S, D]
                    "densities": densities,
                    "density_unit": unit,
                    "covered_shares": per_run["covered_shares"][run_index],
                    "cells_per_colony": per_run["cells_per_colony"][run_index],
                    "cells_per_mm2": per_run["cells_per_colony"][run_index] / per_run["colony_mm2"][run_index][:, None],
                    "max_density": per_run["max_density"][run_index],
                    "measured_cells_per_mm2": cell_counts["cells_per_mm2"],
                    "measured_covered_share": measured_share,
                    "measured_density": {"share": measured_share, "per_mm2": cell_counts["cells_per_mm2"]}[unit],
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
                    "pattern_counts": per_run["pattern_counts"][run_index],
                    "cell_pattern_counts": per_run["cell_pattern_counts"][run_index],
                    "channel_means": per_run["channel_means"][run_index],
                    # Kymographs: [S, D, K, T, bins, channels / patterns], bins of
                    # distance from the centre (µm); NaN / 0 outside the colony
                    "profiles": per_run["profiles"][run_index],
                    "profile_counts": per_run["profile_counts"][run_index],
                    "profile_hours": 12.0 * record_steps / steps_per_12h,
                    "profile_radii_um": radii_um,
                    "bin_um": bin_um,
                    # Images, for the first run of a job only (they are large)
                    "initial_states": per_run["initial_states"][run_index] if first else None,
                    "final_states": per_run["final_states"][run_index] if first else None,
                    "state_channel_names": names,
                    # What the pattern counts depend on, checked when loading
                    "cell_type_markers": list(rules.markers),
                    "cell_type_thresholds": {
                        _marker: float(rules.thresholds[_marker]) for _marker in rules.markers
                    },
                    "rings": rings,
                })
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
    and the grid diameter keep the seeds and grid of a single run."""
    configs = []
    job_offset = 0
    settings = replace(settings, grid_diameter_um=settings.grid_diameter())
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
            for sizes in size_groups:
                configs.append({
                    "settings": asdict(replace(settings, replicates=group, diameters_um=sizes)),
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
