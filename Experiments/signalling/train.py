"""Config-driven entrypoint for training NCAs on the signalling tasks.

Launched through the standard manifest workflow, e.g.::

    python Experiments/generate_configs.py Experiments/signalling ...
    python Experiments/run_config.py --manifest <manifest> --index 0

or for a single config file::

    python Experiments/run_config.py --config-file <config.yaml> \
        --entrypoint Experiments.signalling.train:run

The training pool is ``data.batches`` random instances of one task from
``Experiments/signalling/tasks.py``; part of it is replaced by fresh instances
after every step. The NCA learns each transition between frames in
``run.t`` steps. At the end of training the model is measured on held-out
instances (``Experiments/signalling/evaluation.py``), logged as TaskEvaluation/*.

State channel layout (``model.channels`` = C):

    [0, 4)      RGBA (alpha marks the tissue)
    [4, C)      hidden channels (for sNCA the last ones are signal and structure channels)
"""

import os
from functools import partial

from Experiments.config_helpers import build_loss_filename


def build_run_name(cfg, model_name, optimiser_name):
    task = cfg.data.signal_task
    params = "-".join(
        f"{name}{'-'.join(map(str, value)) if isinstance(value, tuple) else value}"
        for name, value in sorted(task.params.items())
    )
    details = (
        f"task-{task.task}_params{params or 'default'}_s{task.size}"
        f"_b{cfg.data.batches}_t{cfg.run.t}_lr{cfg.optimiser.learn_rate}"
    )
    if cfg.run.repeat is not None:
        details += f"_rep{cfg.run.repeat}"
    return f"{model_name}_{build_loss_filename(cfg.loss)}_{details}_{optimiser_name}"


def run(cfg):
    import jax
    import numpy as np
    from dotenv import load_dotenv

    from Experiments.model_registry import create_model_id, evaluation_input_provenance
    from Experiments.nca_training import run_training
    from Experiments.signalling.evaluation import evaluate_task
    from Experiments.signalling.tasks import sample_batch
    from NCA.model.factory import build_model
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.data_augmenter.procedural import ProceduralTaskAugmenter
    from NCA.trainer.optimiser import build_optimiser

    load_dotenv()
    model_root = cfg.model_store.root
    if not model_root:
        raise ValueError("model_store.root must be set for signalling task training.")

    task = cfg.data.signal_task
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model, model_name = build_model(cfg.model, key=model_key)
    if model.N_CHANNELS <= 4:
        raise ValueError("model.channels must leave hidden channels after the 4 RGBA channels")
    _, optimiser_name, _ = build_optimiser(cfg.optimiser, cfg.run.iterations, return_schedule=True)

    def sample(seed, count):
        return sample_batch(task.task, seed, count, task.size, task.target_repeats, **task.params)[0]

    augmenter = ProceduralTaskAugmenter(
        sample,
        cfg.data.batches,
        model.N_CHANNELS - 4,
        resample_fraction=task.resample_fraction,
        noise_strength=task.noise_strength,
        seed=task.seed,
        evaluate=partial(
            evaluate_task,
            task=task.task, size=task.size, params=task.params, target_repeats=task.target_repeats,
            t=cfg.run.t, samples=task.evaluation_samples, repeats=task.evaluation_repeats,
            horizons=task.evaluation_horizons,
        ),
    )
    # The first pool, [batches, time, 4, H, W]
    data = np.stack(augmenter.return_observed_data())
    frames = data.shape[1]
    context = TrainerContext(
        run_name=build_run_name(cfg, model_name, optimiser_name),
        storage_id=create_model_id(cfg),
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        data_augmenter=augmenter,
        observed_channels=4,
        data_channels=4,
        channel_names=("R", "G", "B", "A"),
        timepoint_names=tuple(f"frame {k}" for k in range(1, frames)),
        evaluation_input=evaluation_input_provenance(data),
    )
    return run_training(cfg, model=model, data=data, context=context, key=train_key)
