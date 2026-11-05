"""Inverse design records: optimised colony geometries saved in the model store.

Each run of ``Experiments/inverse_design/optimise.py`` writes one record next
to the model bundles it used:

    <store>/inverse_design/<experiment>/<run-id>/
        geometry.eqx    best geometry (by binary-mask score)
        density.eqx     its cell density (only with the cell density model)
        config.yaml     resolved inverse design config
        summary.yaml    model link, cell type file, baseline and optimised
                        cell-type shares, evaluation history, provenance
        states.npz      masks and final states of the baselines and the best
                        geometry, the mask at each evaluation, the textures
                        (and with the cell density model, the gap values and
                        spectrum)

Records are never edited once written. They are not indexed by the registry.
"""

import shutil
from pathlib import Path

import equinox as eqx
import numpy as np
from omegaconf import OmegaConf

from Common.dataloader.cell_types import label_names, marker_values, pixel_label_shares
from Experiments.config import config_to_dict
from Experiments.model_registry import (
    _config_digest,
    _git_provenance,
    _identifier,
    _sha256,
    _slug,
    _utc_now,
    _write_yaml,
)
from NCA.inverse_design import CellDensity, FourierRadialGeometry

INVERSE_DESIGN_SCHEMA_VERSION = 1
GEOMETRY_FILE = "geometry.eqx"
DENSITY_FILE = "density.eqx"
STATES_FILE = "states.npz"


def create_run_id(cfg):
    """ID for one run: timestamp, config hash (without the store location) and a random part."""

    identity = config_to_dict(cfg)
    identity["checkpoint"] = {
        key: value for key, value in identity["checkpoint"].items() if key != "store_root"
    }
    return _identifier(_config_digest(identity))


def geometry_template(settings):
    """A geometry with the structure recorded in ``summary.yaml`` under ``geometry``."""

    lower, upper = settings["radius_range"]
    return FourierRadialGeometry(
        int(settings["harmonics"]),
        radius=0.5 * (lower + upper),
        radius_range=(float(lower), float(upper)),
        coefficient_limit=float(settings["coefficient_limit"]),
    )


def load_geometry(path):
    """The best geometry of a record folder."""

    path = Path(path)
    summary = OmegaConf.to_container(OmegaConf.load(path / "summary.yaml"))
    return eqx.tree_deserialise_leaves(path / GEOMETRY_FILE, geometry_template(summary["geometry"]))


def load_density(path):
    """The best cell density of a record folder, or None if it used no density model."""

    path = Path(path)
    if not (path / DENSITY_FILE).exists():
        return None
    return eqx.tree_deserialise_leaves(path / DENSITY_FILE, CellDensity(0.5))


def _density(density):
    return None if density is None else float(density.value)


def describe_geometry(geometry):
    """Constrained parameters of a geometry, for reading in ``summary.yaml``."""

    return {
        "radius": float(geometry.radius),
        "angle": float(geometry.angle),
        "coefficients_cos": [float(value) for value in geometry.coefficients_cos],
        "coefficients_sin": [float(value) for value in geometry.coefficients_sin],
    }


def label_fractions(evaluation, cell_type_rules, state_channel_names):
    """Mean share of every label of the rules inside the mask, over the rollouts."""

    mask = np.asarray(evaluation.mask) > 0.5
    totals = {name: 0.0 for name in label_names(cell_type_rules)}
    states = np.asarray(evaluation.final_states)
    for state in states:
        values = marker_values(state, state_channel_names, cell_type_rules, axis=0)
        shares = pixel_label_shares(values, cell_type_rules, mask)
        for name, share in shares.items():
            totals[name] += float(np.sum(share)) / max(int(np.sum(mask)), 1)
    return {name: total / len(states) for name, total in totals.items()}


def evaluation_summary(evaluation, cell_type_rules, state_channel_names):
    return {
        "target_share_mean": evaluation.mean,
        "target_share_std": evaluation.std,
        "border_fraction": evaluation.border_fraction,
        "density": evaluation.density,
        "area_pixels": int(np.sum(np.asarray(evaluation.mask) > 0.5)),
        "labels": label_fractions(evaluation, cell_type_rules, state_channel_names),
    }


def publish_inverse_design_record(
    *,
    store_root,
    run_id,
    cfg,
    bundle,
    cell_types_path,
    cell_type_rules,
    state_channel_names,
    geometry_settings,
    total_steps,
    boundary_mode,
    baselines,
    result,
    biology_bank,
    wandb_run_id=None,
    repository_root=None,
    gaps=None,
    coverage_rows=None,
    texture_settings=None,
):
    """Write a finished run as a new record in ``<store_root>/inverse_design/``.

    ``baselines`` maps a name to ``(geometry, BinaryEvaluation)``. ``gaps``
    (``CellGaps``) and ``coverage_rows`` (from ``fill_reference_gaps``) are
    given with the cell density model; ``texture_settings`` are the texture
    settings used (``optimise.texture_summary``). The files are written to a hidden
    folder that is renamed into place at the end.
    """

    repository_root = Path(repository_root or Path(__file__).resolve().parents[2])
    parent = Path(store_root) / "inverse_design" / _slug(str(cfg.experiment.name))
    final_path = parent / run_id
    staging = parent / f".{run_id}.tmp"
    if final_path.exists():
        raise FileExistsError(f"Inverse design record {final_path} already exists")
    staging.mkdir(parents=True)
    try:
        eqx.tree_serialise_leaves(staging / GEOMETRY_FILE, result.best_geometry)
        if result.best_density is not None:
            eqx.tree_serialise_leaves(staging / DENSITY_FILE, result.best_density)
        _write_yaml(staging / "config.yaml", config_to_dict(cfg))
        best = result.best_evaluation
        summary = {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "id": run_id,
            "created": _utc_now(),
            "experiment": str(cfg.experiment.name),
            "model": {
                "id": str(bundle.id),
                "path": str(bundle.path),
                "checkpoint_sha256": str(bundle.manifest.artifact.sha256),
            },
            "cell_types": {
                "file": str(cfg.inverse_design.cell_types_file),
                "sha256": _sha256(Path(cell_types_path)),
                "target": cfg.inverse_design.objective.target,
                "maximise": cfg.inverse_design.objective.maximise,
            },
            "rollout": {
                "hours": cfg.inverse_design.rollout.hours,
                "total_steps": int(total_steps),
                "boundary_mode": boundary_mode,
                "grid_shape": [int(size) for size in biology_bank.shape[-2:]],
            },
            "texture": texture_settings,
            "geometry": {
                **geometry_settings,
                "best": describe_geometry(result.best_geometry),
                "final": describe_geometry(result.geometry),
            },
            "density": None if gaps is None else {
                "mode": cfg.inverse_design.density.mode,
                "best": _density(result.best_density),
                "final": _density(result.density),
                "edge_width": float(gaps.edge_width),
                "training_colonies": coverage_rows,
            },
            "best_iteration": int(result.best_iteration),
            "optimised": evaluation_summary(best, cell_type_rules, state_channel_names),
            "baselines": {
                name: {
                    **evaluation_summary(evaluation, cell_type_rules, state_channel_names),
                    "geometry": describe_geometry(geometry),
                }
                for name, (geometry, evaluation) in baselines.items()
            },
            "evaluations": {
                "iterations": [int(iteration) for iteration in result.evaluation_iterations],
                "target_share_mean": [evaluation.mean for evaluation in result.evaluations],
                "target_share_std": [evaluation.std for evaluation in result.evaluations],
                "density": [evaluation.density for evaluation in result.evaluations],
            },
            "wandb_run_id": wandb_run_id,
            "provenance": _git_provenance(repository_root),
        }
        _write_yaml(staging / "summary.yaml", summary)
        density_arrays = {} if gaps is None else {
            "gap_values": np.asarray(gaps.gap_values),
            "gap_amplitude": np.asarray(gaps.amplitude),
        }
        np.savez_compressed(
            staging / STATES_FILE,
            **density_arrays,
            best_mask=np.asarray(best.mask),
            best_final_states=np.asarray(best.final_states),
            evaluation_iterations=np.asarray(result.evaluation_iterations),
            evaluation_masks=np.stack([np.asarray(evaluation.mask) for evaluation in result.evaluations]),
            evaluation_scores=np.stack([evaluation.scores for evaluation in result.evaluations]),
            baseline_names=np.asarray(list(baselines)),
            baseline_masks=np.stack([np.asarray(evaluation.mask) for _, evaluation in baselines.values()]),
            baseline_final_states=np.stack(
                [np.asarray(evaluation.final_states) for _, evaluation in baselines.values()]
            ),
            biology_bank=np.asarray(biology_bank),
            loss_history=np.asarray(result.loss_history),
        )
        staging.rename(final_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_path
