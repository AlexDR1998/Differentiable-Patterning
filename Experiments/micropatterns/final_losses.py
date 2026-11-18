"""Final losses of a saved micropattern model on its training and validation replicates.

The model is scored by the trainer's own validation code (``NCA/trainer/validation.py``)
with its training loss and the loss weights of its last training iteration (the end of
the curriculum). Models trained with the same loss and curriculum are therefore scored
the same way, whatever their projection or resolution.

Each split gets two scores, as on W&B:

- ``per_transition``: every 12 h step starts from the true image (``validation/loss``);
- ``rollout``: one rollout from the 0 h image (``validation_rollout/loss``).

The training split is scored like the validation split (one copy of each replicate,
no pool), so the two are directly comparable. This differs from the training loss
logged during training, which is measured on pool states.

Many models can be scored on the cluster as a job (``docs/remote_notebook_jobs.md``):
the notebook ``figures/projection_resolution_figure.py`` exports the manifest, and

    python Experiments/micropatterns/final_losses.py --manifest <manifest> [--index 0]

scores one part of it.
"""

import argparse
import os
from dataclasses import replace
from pathlib import Path

DATASET_FOLDER = "260726_nca_dataset"
# Prefix of the validation metrics -> prefix used here
SCORES = {"validation/": "per_transition/", "validation_rollout/": "rollout/"}


def _rename_losses(metrics):
    """Loss metrics of ``ValidationEvaluator`` with readable prefixes, as floats.

    The variation metrics (spread between replicates) are left out.
    """
    renamed = {}
    for name, value in metrics.items():
        for old, new in SCORES.items():
            if name.startswith(old) and "/variation/" not in name:
                renamed[new + name[len(old):]] = float(value)
    return renamed


def final_losses(bundle, data_root, seed=0):
    """``{"train": {score: value}, "validation": {score: value}, "replicates": {...}}`` for a
    registry bundle.

    Each model is scored on its own split: the training and validation replicates
    (one-based IDs, listed in ``replicates``) come from its saved config, and the
    data are scaled as in its training. Scores are ``per_transition/loss``, ``rollout/loss`` and their components
    (e.g. ``rollout/loss_component_raw/texture``). ``validation`` is missing when the
    model was trained without held-out replicates. ``seed`` sets the NCA's random
    updates and the loss's random parts; use the same seed for models you compare.
    """
    import jax
    import jax.random as jr

    from Experiments.micropatterns.config_helpers import (
        build_data_augmenter,
        expand_channel_timestep_mask_for_loss,
        load_train_validation_data,
    )
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.preparation import prepare_training
    from NCA.trainer.trainer import build_trainer
    from NCA.trainer.validation import ValidationEvaluator

    cfg = bundle.config
    # Always score the full rollout too, and load one copy of each replicate
    cfg = replace(
        cfg,
        trainer=replace(cfg.trainer, validation_rollout=True),
        data=replace(cfg.data, micropattern=replace(cfg.data.micropattern, pool_copies=1)),
    )
    root = Path(data_root).expanduser() / DATASET_FOLDER
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found at {root}; set the data root to the folder containing it.")
    training, validation = load_train_validation_data(cfg.data, impath=str(root))
    model_key, prepare_key, loss_key = jr.split(jr.PRNGKey(seed), 3)
    model = bundle.load_model(key=model_key)

    # The trainer is built as in Experiments/micropatterns/train.py, but only
    # used for its loss settings: nothing is trained.
    data, aux, channel_names, boundary, mask, _ = training
    schema = aux.get("channel_schema")
    observation_times = tuple(float(time) for time in cfg.data.micropattern.timesteps)
    augmenter, _ = build_data_augmenter(
        cfg.data,
        cfg.run.iterations,
        data,
        model.N_CHANNELS,
        mask,
        schema,
        aux.get("intervention_times"),
        observation_times=None if cfg.run.interval_mode == "uniform" else observation_times,
        condition_ids=aux.get("condition_ids"),
    )
    context = TrainerContext(
        run_name=bundle.id,
        model_directory="",
        data_augmenter=augmenter,
        boundary_mask=boundary,
        channel_names=channel_names,
        channel_schema=schema,
        loss_time_channel_mask=expand_channel_timestep_mask_for_loss(cfg.data, mask, schema),
        training_intervention_times=aux.get("intervention_times"),
        training_condition_ids=aux.get("condition_ids"),
        training_batch_weights=aux.get("batch_loss_weights"),
        observation_times=observation_times,
    )
    setup = prepare_training(build_trainer(cfg, model, data, context), key=prepare_key)
    final_weights = setup.loss_weight_schedule(setup.iterations - 1)

    micropattern = cfg.data.micropattern
    losses = {"replicates": {
        "train": tuple(micropattern.train_replicates or range(1, cfg.data.batches + 1)),
        "validation": tuple(micropattern.validation_replicates or ()),
    }}
    for split_name, split in (("train", training), ("validation", validation)):
        if split is None:
            continue
        split_data, split_aux, _, split_boundary, split_mask, _ = split
        # The evaluator reads knockout times and conditions from the context
        trainer = build_trainer(cfg, model, data, replace(
            context,
            validation_intervention_times=split_aux.get("intervention_times"),
            validation_condition_ids=split_aux.get("condition_ids"),
        ))
        evaluator = ValidationEvaluator(
            trainer,
            setup,
            split_data,
            split_boundary,
            expand_channel_timestep_mask_for_loss(cfg.data, split_mask, schema),
            loss_key,
        )
        losses[split_name] = _rename_losses(jax.device_get(evaluator(model, final_weights)))
    return losses


# ---------------------------------------------------------------- remote jobs


def job_configs(model_ids, seed, output_dir, store_root="", data_root="", models_per_part=1):
    """Manifest entries scoring ``model_ids`` in parts of ``models_per_part`` models.

    ``store_root`` and ``data_root`` are only used outside the cluster (pods use
    ``MODEL_STORE_ROOT`` and ``DATA_PATH_BASE``).
    """
    model_ids = list(model_ids)
    size = max(1, int(models_per_part))
    return [
        {
            "model_ids": model_ids[start:start + size],
            "seed": int(seed),
            "output_dir": str(output_dir),
            "store_root": str(store_root),
            "data_root": str(data_root),
        }
        for start in range(0, len(model_ids), size)
    ]


def run_job_part(config, index, overwrite=False):
    """Score the models of one part of an exported job and write its results file."""
    from Experiments.model_registry import ModelRegistry
    from Experiments.notebook_jobs import part_path, repository_path, save_results_npz

    output = part_path(repository_path(config["output_dir"]), index)
    if output.exists() and not overwrite:
        print(f"{output} exists; skipping part {index}")
        return output
    # On the cluster the data and models are in the pod's volumes
    store_root = os.getenv("MODEL_STORE_ROOT") or config["store_root"]
    data_root = os.getenv("DATA_PATH_BASE") or config["data_root"]
    registry = ModelRegistry(Path(store_root).expanduser())
    results = []
    for model_id in config["model_ids"]:
        print(f"Part {index}: scoring {model_id}")
        losses = final_losses(registry.get(model_id), data_root, int(config["seed"]))
        results.append({"model_id": model_id, "seed": int(config["seed"]), **losses})
    save_results_npz(output, results, array_keys=())
    print(f"Saved part {index} to {output}")
    return output


def load_final_losses(path):
    """``{model ID: losses}`` (as ``final_losses`` returns, plus ``seed``) from a results
    file or a folder of a job's ``part_*.npz`` files."""
    from Experiments.notebook_jobs import load_results_npz, repository_path

    results = load_results_npz(repository_path(path), array_keys=())
    return {result.pop("model_id"): result for result in results}


def main():
    parser = argparse.ArgumentParser(description="Score one part of a final-loss job")
    parser.add_argument("--manifest", required=True, help="Job manifest exported by projection_resolution_figure.py")
    parser.add_argument("--index", type=int, default=None, help="Part to run (default: from the job environment)")
    parser.add_argument("--overwrite", action="store_true", help="Rerun a part whose results file exists")
    args = parser.parse_args()
    from Experiments.notebook_jobs import read_job_entry

    index, config = read_job_entry(args.manifest, args.index)
    run_job_part(config, index, overwrite=args.overwrite)


__all__ = ["final_losses", "job_configs", "load_final_losses", "run_job_part"]


if __name__ == "__main__":
    main()
