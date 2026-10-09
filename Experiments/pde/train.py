"""Config-driven entrypoint for training NCA on simulated PDE trajectories.

Launched through the standard manifest workflow, e.g.::

    python Experiments/generate_configs.py Experiments/pde ...
    python Experiments/run_config.py --manifest <manifest> --index 0

or for a single config file::

    python Experiments/run_config.py --config-file <config.yaml> \
        --entrypoint Experiments.pde.train:run

The training data is ``data.batches`` trajectories of one PDE from
``PDE/catalogue.py``, simulated at the start of the run. The NCA learns each
transition between consecutive frames in ``run.t`` steps.

State channel layout (``model.channels`` = C):

    [0, n_observed)     observed PDE channels (data.pde.observed_channels)
    [n_observed, C)     hidden channels
"""

import os

from Experiments.config_helpers import build_loss_filename


def build_run_name(cfg, model_name, optimiser_name):
    pde = cfg.data.pde
    parameters = "-".join(f"{name}{value:g}" for name, value in sorted(pde.parameters.items()))
    details = (
        f"pde-{pde.model}_{'-'.join(pde.channel_names)}"
        f"_params{parameters or 'default'}"
        f"_ic-{pde.initial_condition}_s{pde.size}_f{pde.frames}"
        f"_t{cfg.run.t}_lr{cfg.optimiser.learn_rate}"
        f"_irp{pde.reinjection_probability}"
    )
    if cfg.run.repeat is not None:
        details += f"_rep{cfg.run.repeat}"
    return f"{model_name}_{build_loss_filename(cfg.loss)}_{details}_{optimiser_name}"


def simulate_training_data(cfg):
    """Simulate the PDE trajectories described by ``cfg.data.pde``."""
    import jax
    from PDE.dataset import simulate_trajectories

    pde = cfg.data.pde
    return simulate_trajectories(
        pde.model,
        jax.random.PRNGKey(pde.seed),
        parameters=pde.parameters,
        batches=cfg.data.batches,
        size=pde.size,
        padding=pde.padding,
        dx=pde.dx,
        dt=pde.dt,
        solver=pde.solver,
        t_end=pde.t_end,
        frames=pde.frames,
        observed_channels=pde.observed_channels,
        initial_condition=pde.initial_condition,
        initial_condition_scale=pde.initial_condition_scale,
        initial_condition_offset=pde.initial_condition_offset,
        initial_condition_blur=pde.initial_condition_blur,
        normalisation=pde.normalisation,
    )


def run(cfg):
    import jax
    import jax.numpy as jnp
    from dotenv import load_dotenv

    from NCA.model.factory import build_model
    from Experiments.nca_training import run_training
    from Experiments.model_registry import create_model_id, evaluation_input_provenance
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.data_augmenter.pde import PdeAugmenter
    from NCA.trainer.optimiser import build_optimiser

    load_dotenv()
    model_root = cfg.model_store.root
    if not model_root:
        raise ValueError("model_store.root must be set for PDE training.")

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    trajectory = simulate_training_data(cfg)
    n_observed = trajectory.data.shape[2]
    if cfg.model.channels <= n_observed:
        raise ValueError(
            f"model.channels={cfg.model.channels} leaves no hidden channels for {n_observed} observed channels"
        )
    print(
        f"PDE {cfg.data.pde.model}: data {trajectory.data.shape} ({', '.join(trajectory.channel_names)}), "
        f"times {trajectory.observation_times}"
    )

    data = jnp.asarray(trajectory.data)
    model, model_name = build_model(cfg.model, key=model_key)
    _, optimiser_name, _ = build_optimiser(
        cfg.optimiser, cfg.run.iterations, return_schedule=True
    )
    augmenter = PdeAugmenter(
        data,
        model.N_CHANNELS - n_observed,
        reinjection_probability=cfg.data.pde.reinjection_probability,
        noise_strength=cfg.data.pde.noise_strength,
    )
    context = TrainerContext(
        run_name=build_run_name(cfg, model_name, optimiser_name),
        storage_id=create_model_id(cfg),
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        data_augmenter=augmenter,
        channel_names=trajectory.channel_names,
        timepoint_names=tuple(f"t={t:g}" for t in trajectory.observation_times[1:]),
        observation_times=trajectory.observation_times,
        evaluation_input=evaluation_input_provenance(data),
    )
    return run_training(cfg, model=model, data=data, context=context, key=train_key)
