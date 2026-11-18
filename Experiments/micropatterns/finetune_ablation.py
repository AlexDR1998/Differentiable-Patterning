"""Evaluate the fine-tune vs joint training ablation the same way for every model.

W&B validation losses depend on each run's curriculum and loss weights, so
they cannot be compared across the arms of the ablation. Here every model is
rolled out on every condition (baseline, ko_0h, ko_24h) from the 0h image of
each of its **held-out** replicates, and scored with fixed measures:

* ``pixel_mse``: mean squared error inside the colony, over measured channels.
* ``radial_mse``: squared error of the ring-averaged profiles (``n_rings``
  rings), over measured channels. Less sensitive to where exactly cells sit.
* ``share_error`` (with a cell type rule file): mean absolute difference of
  the cell type shares at the last time point.

Each row also records the model's place in the ablation: its pretraining
iterations (from the parent), total iterations, and the baseline share of the
loss weight. ``finetune_ablation_explorer.py`` plots the results.

Run locally from a notebook (``evaluate_models``), or as a remote job: export
a manifest with ``job_configs`` + ``Experiments.notebook_jobs.write_job_manifest``
and launch

    bash launch/launch_batch_multi_job.sh Experiments/micropatterns/finetune_ablation.py <manifest> <workers> H200

Test one part locally with ``--manifest <manifest> --index 0``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from Experiments.notebook_jobs import (
    load_results_npz,
    part_path,
    read_job_entry,
    repository_path,
    save_results_npz,
)

CONDITIONS = ("baseline", "ko_0h", "ko_24h")
ARRAY_KEYS = ("radial_prediction", "radial_target")


@dataclass(frozen=True)
class AblationEvaluationSettings:
    data_root: str  # folder that holds 260726_nca_dataset (DATA_PATH_BASE)
    store_root: str  # model store (MODEL_STORE_ROOT)
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = (0, 1)  # rollouts per model, condition and replicate
    n_rings: int = 8
    cell_type_rules: str | None = None  # rule file for share_error; None skips it


def settings_from_dict(values) -> AblationEvaluationSettings:
    values = dict(values)
    for key in ("conditions", "seeds"):
        values[key] = tuple(values[key])
    return AblationEvaluationSettings(**values)


# ---------------------------------------------------------------------------
# Where a model sits in the ablation
# ---------------------------------------------------------------------------

def baseline_share(knockout) -> float:
    """Share of the training loss weight on the baseline condition."""
    curriculum = tuple(knockout.curriculum or ("baseline",))
    weights = dict(knockout.curriculum_weights or {})
    total = sum(weights.get(name, 1.0) for name in curriculum)
    baseline = sum(weights.get(name, 1.0) for name in curriculum if name == "baseline")
    return float(baseline / total)


def ablation_position(bundle, registry) -> dict:
    """Pretraining and total iterations, and the baseline share, of one model."""
    config = bundle.config
    parent_id = config.initialization.model_id
    pretrain = 0
    if parent_id is not None:
        pretrain = int(registry.get(parent_id).config.run.iterations)
    total = pretrain + int(config.run.iterations)
    return {
        "model_id": bundle.id,
        "parent_id": parent_id,
        "experiment": config.experiment.name,
        "repeat": config.run.repeat,
        "train_replicates": list(config.data.micropattern.train_replicates or ()),
        "pretrain_iterations": pretrain,
        "total_iterations": total,
        "pretrain_fraction": pretrain / total,
        "baseline_share": baseline_share(config.data.knockout),
    }


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def score_prediction(prediction, condition_data, n_rings, cell_type_rules=None):
    """Scores of one rollout ``[T, M, H, W]`` against its measured target.

    Returns one dict per time point after 0h, plus the ring profiles
    ``[T, rings, M]`` of prediction and target.
    """
    from Experiments.micropatterns.density_ko_sweep import ring_channel_means

    target = np.asarray(condition_data["target"], dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    measured = np.asarray(condition_data["measured"], dtype=bool)
    colony = np.asarray(condition_data["boundary"]).squeeze().astype(bool)
    radial_prediction = np.stack([ring_channel_means(frame, colony, n_rings) for frame in prediction])
    radial_target = np.stack([ring_channel_means(frame, colony, n_rings) for frame in target])

    rows = []
    for time_index in range(1, target.shape[0]):
        channels = measured[time_index]
        if not channels.any():
            continue
        difference = (prediction[time_index] - target[time_index])[channels][:, colony]
        radial = (radial_prediction[time_index] - radial_target[time_index])[:, channels]
        rows.append({
            "time_index": time_index,
            "pixel_mse": float(np.mean(difference**2)),
            "radial_mse": float(np.nanmean(radial**2)),
        })
    if cell_type_rules is not None and rows:
        rows[-1]["share_error"] = share_error(
            prediction[-1], target[-1], condition_data, cell_type_rules, n_rings
        )
    return rows, radial_prediction, radial_target


def share_error(prediction, target, condition_data, cell_type_rules, n_rings):
    """Mean absolute difference of model and measured cell type shares."""
    from Experiments.micropatterns.ko_evaluation import data_label_shares, model_label_shares

    colony = np.asarray(condition_data["boundary"]).squeeze().astype(bool)
    names = condition_data["channel_names"]
    model = np.asarray(model_label_shares(prediction, names, cell_type_rules, colony, n_rings))
    data = np.asarray(data_label_shares(target, names, cell_type_rules, colony, n_rings))
    return float(np.nanmean(np.abs(model - data)))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def evaluate_models(settings: AblationEvaluationSettings, model_ids, progress=None):
    """Result rows for ``model_ids``: one per model, condition, replicate, seed and time.

    Each row also holds the ring profiles (``radial_prediction``,
    ``radial_target``) of its rollout, on the row of its last time point.
    """
    import jax

    from Common.dataloader.cell_types import load_cell_type_rules
    from Experiments.micropatterns.density_ko_sweep import (
        load_condition_data,
        load_training_histogram_bins,
    )
    from Experiments.micropatterns.ko_evaluation import evaluate_condition
    from Experiments.model_registry import ModelRegistry

    registry = ModelRegistry(settings.store_root)
    data_root = Path(settings.data_root)
    rules = (
        None
        if settings.cell_type_rules is None
        else load_cell_type_rules(repository_path(settings.cell_type_rules))
    )
    results = []
    for model_id in model_ids:
        bundle = registry.get(model_id)
        position = ablation_position(bundle, registry)
        model = bundle.load_model(key=jax.random.PRNGKey(0))
        bins = load_training_histogram_bins(bundle, data_root)
        replicates = bundle.config.data.micropattern.validation_replicates
        if not replicates:
            raise ValueError(f"{model_id} has no held-out validation replicates")
        for condition in settings.conditions:
            for replicate in replicates:
                condition_data = load_condition_data(bundle, data_root, condition, replicate, bins)
                if rules is not None:
                    # A rule file that needs a left-out channel fails here,
                    # before any rollouts
                    final = np.asarray(condition_data["target"])[-1]
                    share_error(final, final, condition_data, rules, settings.n_rings)
                for seed in settings.seeds:
                    prediction, time_labels = evaluate_condition(
                        model, bundle, condition_data, condition, seed
                    )
                    rows, radial_prediction, radial_target = score_prediction(
                        prediction, condition_data, settings.n_rings, rules
                    )
                    for row in rows:
                        results.append({
                            **position,
                            "condition": condition,
                            "replicate": int(replicate),
                            "seed": int(seed),
                            "time": time_labels[row["time_index"]],
                            **row,
                        })
                    if rows:
                        results[-1]["radial_prediction"] = radial_prediction
                        results[-1]["radial_target"] = radial_target
                    if progress is not None:
                        progress(1)
    return results


def results_table(results):
    """The scalar columns of the results as a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame([
        {key: value for key, value in row.items() if key not in ARRAY_KEYS}
        for row in results
    ])


def save_results(path, results):
    return save_results_npz(path, results, ARRAY_KEYS)


def load_results(path):
    """Results from one file, or a folder of ``part_*.npz`` files of a job."""
    return load_results_npz(path, ARRAY_KEYS)


def job_configs(settings: AblationEvaluationSettings, model_ids, output_dir):
    """One manifest entry per model, each writing a part into ``output_dir``."""
    return [
        {"settings": asdict(settings), "model_ids": [model_id], "output_dir": str(output_dir)}
        for model_id in model_ids
    ]


def run_job_part(config, index, overwrite=False):
    """Run one part of an exported job and write its results file."""
    settings = settings_from_dict(config["settings"])
    # On the cluster the data and models are in the pod's volumes
    settings = AblationEvaluationSettings(**{
        **asdict(settings),
        "data_root": os.getenv("DATA_PATH_BASE") or settings.data_root,
        "store_root": os.getenv("MODEL_STORE_ROOT") or settings.store_root,
    })
    output = part_path(repository_path(config["output_dir"]), index)
    if output.exists() and not overwrite:
        print(f"{output} exists; skipping part {index}")
        return output
    results = evaluate_models(settings, config["model_ids"])
    save_results(output, results)
    print(f"Saved part {index} to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run one part of a fine-tune ablation evaluation job")
    parser.add_argument("--manifest", required=True, help="Job manifest exported by finetune_ablation_explorer.py")
    parser.add_argument("--index", type=int, default=None, help="Part to run (default: from the job environment)")
    parser.add_argument("--overwrite", action="store_true", help="Rerun a part whose results file exists")
    args = parser.parse_args()
    index, config = read_job_entry(args.manifest, args.index)
    run_job_part(config, index, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
