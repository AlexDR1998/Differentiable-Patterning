"""Config-driven snowmelt NCA training entrypoint.

Launched through the standard manifest workflow, e.g.::

    python Experiments/generate_configs.py Experiments/snowmelt ...
    python Experiments/run_config.py --manifest <manifest> --index 0

or for a single config file::

    python Experiments/run_config.py --config <config.yaml> \
        --entrypoint Experiments.snowmelt.train:run

The target sequence is one Sentinel-2 acquisition series over the Nivolet
catchment (see ``Common/dataloader/snowmelt.py``). Acquisitions are unevenly
spaced in time; set ``run.interval_mode: steps`` so each transition's NCA step
count is proportional to its duration in days.

State channel layout (``model.channels`` = C):

    [0, n_targets)          target channels (training.loss compares these)
    [n_targets, C - m)      hidden channels
    [C - m, C)              fixed boundary channels: catchment mask, then
                            data.snowmelt.static_channels (e.g. DEM, slope)
"""

import os

from Experiments.config_helpers import _compact_value, build_loss_filename


def build_run_name(cfg, model_name, optimiser_name):
    snowmelt = cfg.data.snowmelt
    loop = cfg.run
    details = (
        f"snowmelt-{snowmelt.version}_{'-'.join(snowmelt.target_channels)}"
        f"_static{_compact_value(snowmelt.static_channels or None)}"
        f"_ds{cfg.data.downsample}"
        f"_t{loop.t}_{loop.interval_mode}"
        f"_lr{cfg.optimiser.learn_rate}"
        f"_irp{snowmelt.reinjection_probability}"
    )
    if snowmelt.hold_out_dates:
        details += f"_holdout{'-'.join(map(str, snowmelt.hold_out_dates))}"
    if loop.repeat is not None:
        details += f"_rep{loop.repeat}"
    return f"{model_name}_{build_loss_filename(cfg.loss)}_{details}_{optimiser_name}"


def load_snowmelt_training_data(cfg):
    """Build the target sequence, boundary array and acquisition times from config."""
    from Common.dataloader.snowmelt import build_snowmelt_sequence

    snowmelt = cfg.data.snowmelt
    if cfg.data.batches != 1:
        raise ValueError("Snowmelt training uses one acquisition sequence: set data.batches=1")
    if snowmelt.hold_out_dates and (cfg.run.interval_mode != "steps" or cfg.run.reference_interval is None):
        # Holding out a date merges its two intervals, so steps must follow days, and the
        # median interval (the default reference) must not change with the held-out dates.
        raise ValueError(
            "data.snowmelt.hold_out_dates needs run.interval_mode=steps and an explicit run.reference_interval"
        )
    return build_snowmelt_sequence(
        root=snowmelt.root,
        version=snowmelt.version,
        target_channels=snowmelt.target_channels,
        static_channels=snowmelt.static_channels,
        downsample=cfg.data.downsample,
        pad=snowmelt.pad,
        mask_threshold=snowmelt.mask_threshold,
        exclude_dates=snowmelt.exclude_dates,
        hold_out_dates=snowmelt.hold_out_dates,
    )


def run(cfg):
    import jax
    import jax.numpy as jnp
    from dotenv import load_dotenv

    from NCA.model.factory import build_model
    from Experiments.nca_training import run_training
    from Experiments.model_registry import create_model_id, evaluation_input_provenance
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.data_augmenter.snowmelt import SnowmeltAugmenter
    from NCA.trainer.optimiser import build_optimiser

    load_dotenv()
    model_root = cfg.model_store.root
    if not model_root:
        raise ValueError("model_store.root must be set for snowmelt training.")
    if cfg.trainer.boundary_mode != "soft":
        raise ValueError(
            "Snowmelt training writes the catchment and terrain into fixed state "
            "channels, which requires trainer.boundary_mode='soft'"
        )

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    sequence = load_snowmelt_training_data(cfg)
    n_targets = sequence.data.shape[2]
    n_boundary = sequence.boundary_mask.shape[1]
    if cfg.model.channels <= n_targets + n_boundary:
        raise ValueError(
            f"model.channels={cfg.model.channels} leaves no hidden channels for "
            f"{n_targets} target and {n_boundary} boundary channels"
        )
    print(
        f"Snowmelt sequence: data {sequence.data.shape}, boundary {sequence.boundary_mask.shape} "
        f"({', '.join(sequence.boundary_channel_names)}), days {sequence.observation_times}"
    )

    data = jnp.asarray(sequence.data)
    boundary = jnp.asarray(sequence.boundary_mask)
    model, model_name = build_model(cfg.model, key=model_key)
    _, optimiser_name, _ = build_optimiser(
        cfg.optimiser, cfg.run.iterations, return_schedule=True
    )
    snowmelt = cfg.data.snowmelt
    augmenter = SnowmeltAugmenter(
        data,
        model.N_CHANNELS - data.shape[2],
        boundary,
        reinjection_probability=snowmelt.reinjection_probability,
        noise_strength=snowmelt.noise_strength,
    )
    context = TrainerContext(
        run_name=build_run_name(cfg, model_name, optimiser_name),
        storage_id=create_model_id(cfg),
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        data_augmenter=augmenter,
        boundary_mask=boundary,
        channel_names=sequence.channel_names,
        timepoint_names=sequence.dates[1:],
        observation_times=sequence.observation_times,
        evaluation_input=evaluation_input_provenance(data, boundary_mask=boundary),
    )
    return run_training(cfg, model=model, data=data, context=context, key=train_key)
