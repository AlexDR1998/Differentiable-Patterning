# /// script
# dependencies = ["marimo", "jax", "matplotlib", "numpy", "pyyaml"]
# ///

"""Walkthrough: train a small emoji NCA with the experiment pipeline.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/nca_training_config_walkthrough.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import json
    import os
    import tempfile
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from Common.trainer.config import (
        LossConfig,
        OptimiserConfig,
        PointwiseLossConfig,
        ScheduleConfig,
    )
    from Experiments.config import (
        CONFIG_SCHEMA_VERSION,
        DataConfig,
        ExperimentConfig,
        ExperimentMetadataConfig,
        LoggingConfig,
        ModelStoreConfig,
        RunConfig,
        SystemConfig,
        WandbConfig,
        config_to_dict,
        experiment_config_from_mapping,
    )
    from Experiments.emoji.config import EmojiDataConfig, EmojiPairConfig
    from Experiments.emoji.config_helpers import build_data_augmenter, load_data
    from Experiments.model_registry import (
        ModelRegistry,
        create_model_id,
        evaluation_input_provenance,
        verify_evaluation_input,
    )
    from Experiments.nca_training import run_training
    from NCA.model.config import ModelConfig
    from NCA.model.factory import build_model
    from NCA.trainer.config import PoolAdmissionConfig, TrainerConfig
    from NCA.trainer.context import TrainerContext


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA training walkthrough

    This notebook is a first introduction to the codebase. It trains a small
    neural cellular automaton (NCA) to grow a sequence of emoji images, and
    goes through the same steps as a real experiment: choosing data, building
    a model, writing the experiment config, training, saving the model as a
    bundle, and loading it back to run it.

    ## How the pieces fit together

    A real experiment starts from a YAML file and runs on a remote GPU:

    ```text
    Experiments/emoji/conf/base_config.yaml   (+ sweep overrides, conf/experiments/*.yaml)
      └─ Experiments/run_config.py            reads the YAML, converts it with
                                              Experiments.config.load_experiment_config
         └─ ExperimentConfig                  frozen dataclasses, same names as the YAML
            └─ Experiments/emoji/train.py:run(cfg)
                 load_data, build_model, build_data_augmenter, TrainerContext
               └─ Experiments/nca_training.py:run_training
                    build_trainer → NcaTrainer.train → TrainingResult
                    publish_model_bundle → <model store>/bundles/...
    ```

    Here we build the `ExperimentConfig` directly in Python, one section at a
    time, then do by hand what `Experiments/emoji/train.py` does. Each section
    of this notebook fills in one section of the config:

    | Notebook section | Config section | Defined in |
    | --- | --- | --- |
    | 1. Data | `cfg.data` | `Experiments/config.py`, `Experiments/emoji/config.py` |
    | 2. Model | `cfg.model` | `NCA/model/config.py` |
    | 3. Training | `cfg.run`, `cfg.trainer`, `cfg.optimiser`, `cfg.loss` | `Experiments/config.py`, `NCA/trainer/config.py`, `Common/trainer/config.py` |
    | 4. The whole config | `ExperimentConfig` | `Experiments/config.py` |

    The default settings are tiny so that training takes a minute or two on a
    laptop CPU. The results will be poor; real runs use larger models, more
    iterations and a GPU. `docs/configuration.md` and `docs/nca_trainer.md`
    describe the config and the trainer in more detail.
    """)
    return


@app.cell(hide_code=True)
def _():
    _repo_root = Path(__file__).resolve().parents[1]
    data_path_base = mo.ui.text(
        value=os.environ.get(
            "DATA_PATH_BASE",
            str(_repo_root / "demo" / "demo_data"),
        ),
        label="DATA_PATH_BASE (must contain Emojis/)",
        full_width=True,
    )
    local_store_root = mo.ui.text(
        value=os.environ.get(
            "MODEL_STORE_ROOT",
            str(_repo_root / "models" / "local" / "notebook"),
        ),
        label="Model store root (cfg.model_store.root)",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            "Two paths come from the environment in real runs: `DATA_PATH_BASE` "
            "(where the datasets live) and `MODEL_STORE_ROOT` (where trained "
            "models are saved). A few example emojis are included in `demo/demo_data/`."
        ),
        data_path_base,
        local_store_root,
    ])
    return data_path_base, local_store_root


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Data

    `cfg.data` is a `DataConfig`. Its `dataset` field picks one of the data
    sections: here `dataset="emojis"`, so the emoji settings go in
    `data.emoji` (an `EmojiDataConfig`), and `data.micropattern` and
    `data.snowmelt` stay empty.

    There are two emoji tasks:

    - **Sequential morphing** (`task="sequence"`): one trajectory that goes
      through the listed images in order.
    - **Multi-attractor patterning** (`task="multi_attractor"`): several
      independent trajectories, each from an initial condition (for example a
      small patch of an image) to a target image repeated `target_repeats` times.

    Each domain has a `load_data(cfg.data)` function in
    `Experiments/<domain>/config_helpers.py`. It returns an array shaped
    `[batch, time, channels, H, W]`, where `data[:, 0]` is the initial condition
    and the later time slots are the targets.
    """)
    return


@app.cell(hide_code=True)
def _(data_path_base):
    _image_root = Path(data_path_base.value).expanduser() / "Emojis"
    if not _image_root.is_dir():
        _emoji_contents = mo.callout(
            "Select a DATA_PATH_BASE containing an `Emojis/` directory to list available files.",
            kind="info",
        )
    else:
        _image_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        _filenames = sorted(
            _path.name
            for _path in _image_root.iterdir()
            if _path.is_file() and _path.suffix.lower() in _image_suffixes
        )
        if _filenames:
            _emoji_contents = mo.md(
                "Available emoji files: "
                + ", ".join(f"`{_filename}`" for _filename in _filenames)
            )
        else:
            _emoji_contents = mo.callout(
                f"No supported image files were found in `{_image_root}`.",
                kind="warn",
            )
    _emoji_contents
    return


@app.cell(hide_code=True)
def _():
    task_picker = mo.ui.radio(
        options={
            "Sequential morphing": "sequence",
            "Multi-attractor patterning": "multi_attractor",
        },
        value="Sequential morphing",
        label="data.emoji.task",
    )
    emoji_sequence_text = mo.ui.text(
        value="crab.png, microbe.png",
        label="data.emoji.sequence (comma separated, used by 'sequence')",
        full_width=True,
    )
    attractor_pairs_text = mo.ui.text_area(
        value=(
            '[{"initial": {"image": "crab.png", "mode": "patch", "size": 4}, '
            '"target": "crab.png"}, '
            '{"initial": {"image": "microbe.png", "mode": "patch", "size": 4}, '
            '"target": "microbe.png"}]'
        ),
        label="data.emoji.pairs (JSON, used by 'multi_attractor')",
        full_width=True,
    )
    downsample_value = mo.ui.dropdown(
        options={
            "Very small (16)": 16,
            "Small (8)": 8,
            "Medium (4)": 4,
            "Full (1)": 1,
        },
        value="Small (8)",
        label="data.downsample",
    )
    target_repeats_value = mo.ui.slider(
        1,
        3,
        value=2,
        label="data.emoji.target_repeats",
    )
    mo.vstack([
        task_picker,
        emoji_sequence_text,
        attractor_pairs_text,
        mo.hstack([downsample_value, target_repeats_value]),
    ])
    return (
        attractor_pairs_text,
        downsample_value,
        emoji_sequence_text,
        target_repeats_value,
        task_picker,
    )


@app.cell
def _(
    attractor_pairs_text,
    downsample_value,
    emoji_sequence_text,
    target_repeats_value,
    task_picker,
):
    data_config = None
    data_config_error = None
    try:
        _sequence = tuple(
            _item.strip()
            for _item in emoji_sequence_text.value.split(",")
            if _item.strip()
        )
        _pairs = ()
        if task_picker.value == "multi_attractor":
            _pairs = tuple(
                EmojiPairConfig(initial=_item["initial"], target=_item["target"])
                for _item in json.loads(attractor_pairs_text.value)
            )

        data_config = DataConfig(
            dataset="emojis",
            batches=1,  # copies of each trajectory in the training pool
            downsample=int(downsample_value.value),
            emoji=EmojiDataConfig(
                task=task_picker.value,
                sequence=_sequence if task_picker.value == "sequence" else (),
                pairs=_pairs,
                target_repeats=int(target_repeats_value.value),
                pad=(2, 2, 2, 2),  # zero border added around each image
                # Augmentation is switched off here, and explored in the
                # second demo notebook
                shift_amount=0,
                noise_strength=0.0,
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as _error:
        data_config_error = str(_error)
    return data_config, data_config_error


@app.cell
def _(data_config, data_config_error, data_path_base):
    data = None
    data_name = None
    data_load_error = data_config_error
    _image_root = Path(data_path_base.value).expanduser() / "Emojis"
    if data_load_error is None and not _image_root.is_dir():
        data_load_error = "Set DATA_PATH_BASE to a directory containing an Emojis/ directory."
    if data_load_error is None:
        try:
            # Without impath, load_data reads $DATA_PATH_BASE/Emojis/
            data, data_name = load_data(data_config, impath=str(_image_root))
        except Exception as _error:
            data_load_error = str(_error)
    return data, data_load_error, data_name


@app.cell(hide_code=True)
def _(data, data_config, data_load_error, data_name):
    if data_load_error is not None:
        _data_output = mo.callout(data_load_error, kind="danger")
    else:
        _array = np.asarray(data)
        _batches, _times = _array.shape[:2]
        _figure, _axes = plt.subplots(
            _batches,
            _times,
            figsize=(2.5 * _times, 2.5 * _batches),
            squeeze=False,
        )
        for _batch in range(_batches):
            for _time in range(_times):
                _axes[_batch, _time].imshow(
                    np.clip(np.moveaxis(_array[_batch, _time, :3], 0, -1), 0.0, 1.0)
                )
                _axes[_batch, _time].set_title(
                    "initial condition" if _time == 0 else f"target {_time}",
                    fontsize=9,
                )
                _axes[_batch, _time].set_axis_off()
        _figure.suptitle(f"data {_array.shape} = [batch, time, channels, H, W]")
        _figure.tight_layout()
        _data_output = mo.vstack([
            mo.md(
                f"Loaded `{data_name}`. Each row is one trajectory, time runs "
                "left to right. The name is built from the data config and "
                "becomes part of the run name."
            ),
            _figure,
            mo.accordion({
                "data config (as YAML)": mo.md(
                    f"```yaml\n{yaml.safe_dump(config_to_dict(data_config), sort_keys=False)}```"
                )
            }),
        ])
    _data_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Model

    `cfg.model` is a `ModelConfig`, which describes only the NCA architecture:

    - `channels`: the size of each cell's state. The first
      `data.emoji.observed_channels` (4: RGBA) are compared to the images;
      the rest are hidden channels the NCA can use freely.
    - `kernel_str`: the fixed spatial filters each cell uses to sense its
      neighbours (`ID` identity, `LAP` Laplacian, `GRAD` gradients; see
      `Common/model/spatial_operators.py`).
    - `fire_rate`: the probability that each cell updates at each step
      (less than 1 makes the update asynchronous and stochastic).
    - `family`: which model to build. `NCA/model/factory.py:build_model` maps
      families to classes. `gNCA`, `nNCA` and `gnNCA` are the plain `NCA`
      with gating and/or parameter noise switched on.

    Models are [Equinox modules](https://docs.kidger.site/equinox/api/module/module/),
    so they are ordinary JAX PyTrees that can be passed through `jit`, `vmap`
    and `grad`. `model.partition()` splits a model into the parts that are
    trained and the parts that are fixed (such as the spatial kernels), for use
    with [`eqx.filter_grad`](https://docs.kidger.site/equinox/api/transformations/#equinox.filter_grad).
    The update rule itself is in `NCA/model/NCA_model.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    seed_value = mo.ui.number(0, 1_000_000, value=0, step=1, label="cfg.seed")
    channel_count = mo.ui.dropdown(
        options={
            "8 channels": 8,
            "12 channels": 12,
            "16 channels": 16,
            "24 channels": 24,
            "32 channels": 32,
        },
        value="8 channels",
        label="model.channels",
    )
    fire_rate_value = mo.ui.dropdown(
        options={"0.5 stochastic": 0.5, "1.0 deterministic": 1.0},
        value="0.5 stochastic",
        label="model.fire_rate",
    )
    mo.hstack([seed_value, channel_count, fire_rate_value])
    return channel_count, fire_rate_value, seed_value


@app.cell
def _(channel_count, fire_rate_value):
    model_config = ModelConfig(
        family="NCA",
        channels=int(channel_count.value),
        kernel_str=("ID", "LAP", "GRAD"),
        fire_rate=float(fire_rate_value.value),
        padding="CIRCULAR",  # how the grid edges are treated
        activation="relu",
    )
    return (model_config,)


@app.cell
def _(model_config, seed_value):
    # As in Experiments/emoji/train.py: one key for the model, one for training
    model_key, train_key = jax.random.split(jax.random.PRNGKey(int(seed_value.value)))
    model, model_name = build_model(model_config, key=model_key)
    return model, model_name, train_key


@app.cell(hide_code=True)
def _(model, model_name):
    _trainable, _fixed = model.partition()
    _parameter_count = sum(
        int(np.prod(_leaf.shape))
        for _leaf in jax.tree_util.tree_leaves(_trainable)
        if hasattr(_leaf, "shape")
    )
    mo.callout(
        f"Built `{model_name}` with {_parameter_count:,} trainable parameters.",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Training settings

    Four config sections control training:

    - `cfg.run` (`RunConfig`): `t`, the number of NCA steps between
      consecutive target images, and `iterations`, the number of gradient
      steps. The best checkpoint is only kept after `checkpoint_warmup`
      iterations.
    - `cfg.trainer` (`TrainerConfig`): how the trainer works.
      `loop_autodiff="checkpointed"` saves memory during backpropagation
      through the rollout; `"lax"` is faster but uses more memory.
      `pool_admission` is explained in the second demo notebook.
    - `cfg.optimiser` (`OptimiserConfig`): the optimiser (NAdam by default)
      and its learning-rate schedule.
    - `cfg.loss` (`LossConfig`): a list of loss `terms` (each with a `type`
      and relative `weight`) and a mapping of `regularisers` to coefficients.
      Available losses are listed in `Common/trainer/loss_table.py`, and their
      options in `Common/trainer/config.py`. As in most machine learning,
      choosing the loss is often the hardest part.

    `cfg.logging.backend` is `"wandb"` for real runs. Here it defaults to
    `"none"`; with `"wandb"` the notebook logs offline to `./wandb/`, which
    can be uploaded later with `wandb sync`.
    """)
    return


@app.cell(hide_code=True)
def _():
    rollout_steps = mo.ui.dropdown(
        options={"4 steps": 4, "8 steps": 8, "16 steps": 16, "32 steps": 32},
        value="16 steps",
        label="run.t",
    )
    iteration_count = mo.ui.dropdown(
        options={
            "10 iterations": 10,
            "100 iterations": 100,
            "500 iterations": 500,
            "1000 iterations": 1000,
        },
        value="100 iterations",
        label="run.iterations",
    )
    learning_rate_value = mo.ui.number(
        0.0001, 0.01, value=0.001, step=0.0001, label="optimiser.learn_rate"
    )
    logging_backend = mo.ui.radio(
        options={"none": "none", "wandb (offline)": "wandb"},
        value="none",
        label="logging.backend",
    )
    mo.vstack([
        mo.hstack([rollout_steps, iteration_count, learning_rate_value]),
        logging_backend,
    ])
    return iteration_count, learning_rate_value, logging_backend, rollout_steps


@app.cell
def _(iteration_count, learning_rate_value, rollout_steps):
    run_config = RunConfig(
        t=int(rollout_steps.value),
        iterations=int(iteration_count.value),
        checkpoint_warmup=0,
        write_images=False,
        write_videos=False,
    )
    trainer_config = TrainerConfig(
        loop_autodiff="checkpointed",
        log_every=max(1, int(iteration_count.value) // 2),
        pool_admission=PoolAdmissionConfig(enabled=False),
    )
    optimiser_config = OptimiserConfig(
        learn_rate=float(learning_rate_value.value),
        warmup_steps=0,
        schedule=ScheduleConfig(type="cosine", final_factor=0.2),
    )
    loss_config = LossConfig(
        terms=(PointwiseLossConfig(type="l2", weight=1.0),),
        regularisers={},
    )
    return loss_config, optimiser_config, run_config, trainer_config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. The whole experiment config

    The sections above, plus a name, the seed, logging and model-store
    settings, make up one `ExperimentConfig`. All config classes are frozen
    dataclasses, so a config can't be changed once built (use
    `dataclasses.replace` to make a modified copy). They also check their
    values when built: try an invalid value, such as a negative learning rate,
    and the error appears straight away rather than during training.

    The dataclasses have the same section and field names as the YAML files,
    so `run.t` in a YAML file is `cfg.run.t` in code. Below, the config is
    written out as YAML: this is what `Experiments/emoji/conf/base_config.yaml`
    looks like, and what is saved as `config.yaml` in each trained model's
    bundle. Converting that YAML back (with the same function `run_config.py`
    uses) gives the identical config.
    """)
    return


@app.cell
def _(
    data_config,
    data_config_error,
    local_store_root,
    logging_backend,
    loss_config,
    model_config,
    optimiser_config,
    run_config,
    seed_value,
    task_picker,
    trainer_config,
):
    experiment_config = None
    experiment_config_error = data_config_error
    if experiment_config_error is None:
        try:
            experiment_config = ExperimentConfig(
                schema_version=CONFIG_SCHEMA_VERSION,
                seed=int(seed_value.value),
                experiment=ExperimentMetadataConfig(name=f"notebook_emoji_{task_picker.value}"),
                system=SystemConfig(precision="highest"),
                data=data_config,
                model=model_config,
                run=run_config,
                trainer=trainer_config,
                optimiser=optimiser_config,
                loss=loss_config,
                logging=LoggingConfig(
                    backend=logging_backend.value,
                    wandb=WandbConfig(
                        project="NCA-notebook",
                        group=f"walkthrough-{task_picker.value}",
                    ),
                ),
                model_store=ModelStoreConfig(
                    enabled=True,
                    root=local_store_root.value,
                    collection="NCA-notebook",
                ),
            )
        except ValueError as _error:
            experiment_config_error = str(_error)
    return experiment_config, experiment_config_error


@app.cell(hide_code=True)
def _(experiment_config, experiment_config_error):
    if experiment_config_error is not None:
        _config_output = mo.callout(experiment_config_error, kind="danger")
    else:
        _as_yaml = config_to_dict(experiment_config)
        _round_trip = experiment_config_from_mapping(_as_yaml) == experiment_config
        _config_output = mo.vstack([
            mo.accordion({
                "The experiment config as YAML": mo.md(
                    f"```yaml\n{yaml.safe_dump(_as_yaml, sort_keys=False)}```"
                )
            }),
            mo.callout(
                "Converting this YAML back with `experiment_config_from_mapping` "
                + ("gives the same config." if _round_trip else "gives a different config!"),
                kind="success" if _round_trip else "danger",
            ),
        ])
    _config_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. The data augmenter

    The trainer never sees the images directly. It gets a *data augmenter*
    (`NCA/trainer/data_augmenter/`), built by each domain's
    `build_data_augmenter(cfg.data, data, model.N_CHANNELS)`. The augmenter

    - adds zero hidden channels so the data has as many channels as the NCA,
      and pads and copies the data as set in `cfg.data`;
    - holds the training *pool*: `x`, the states each time slot starts from,
      and `y`, the targets they should reach after `run.t` NCA steps;
    - after every training step, `advance_pool` builds the next pool from the
      states the NCA actually reached, mixed with the true images, and
      optionally shifted, damaged or noised.

    Training from its own earlier outputs is what teaches the NCA to keep a
    pattern stable, rather than only to reach it once. The second demo
    notebook looks at this in more detail.
    """)
    return


@app.cell
def _(data, data_load_error, experiment_config, model, model_name):
    augmenter = None
    run_name = None
    if data_load_error is None and experiment_config is not None:
        augmenter, _augmenter_name = build_data_augmenter(
            experiment_config.data, data, model.N_CHANNELS
        )
        run_name = f"{experiment_config.experiment.name}_{model_name}"
    return augmenter, run_name


@app.cell(hide_code=True)
def _(augmenter, data):
    if augmenter is None:
        _augmenter_output = mo.callout("Load the data first.", kind="info")
    else:
        _pool_x, _pool_y = augmenter.initialize_pool(jax.random.PRNGKey(0))
        _augmenter_output = mo.md(
            f"""
    - Loaded data: `{tuple(np.shape(data))}`
    - `augmenter.return_saved_data()`: {len(augmenter.return_saved_data())} trajectories of
      shape `{tuple(augmenter.return_saved_data()[0].shape)}` (hidden channels and padding added)
    - Initial pool: `x[0]` has shape `{tuple(_pool_x[0].shape)}`, `y[0]` has shape
      `{tuple(_pool_y[0].shape)}`. There is one slot per transition between images.
    """
        )
    _augmenter_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Training

    The next cell is what `Experiments/emoji/train.py` does after building
    the model and augmenter:

    1. It fills in a `TrainerContext` (`NCA/trainer/context.py`). This holds
       the values that are not user choices but are derived from the loaded
       data or the environment: the augmenter, the run name, where to save
       the checkpoint, channel counts, and a fingerprint of the input data
       (`evaluation_input`) so later evaluations can check they use the same
       initial condition.
    2. It calls `run_training`, which builds the trainer and calls
       `NcaTrainer.train`. That compiles one training step (the NCA rollout,
       loss, gradient and optimiser update) with JAX, then runs it in a
       Python loop that also updates the pool, logs, and keeps the best
       checkpoint. Afterwards `run_training` publishes the best checkpoint
       as a model bundle, because `model_store.enabled` is true.

    Here we also pass a `progress_callback`, which is called after every
    iteration and is used for the live loss plot. The first iteration takes
    longer, because that is when JAX compiles the step.
    """)
    return


@app.cell(hide_code=True)
def _(experiment_config_error):
    training_button = mo.ui.run_button(
        label="Train locally",
        disabled=experiment_config_error is not None,
    )
    training_button
    return (training_button,)


@app.cell
def _(
    augmenter,
    data,
    experiment_config,
    run_name,
    train_key,
    training_button,
    model,
):
    mo.stop(not training_button.value or augmenter is None)

    # run_config.py sets these from the config and environment in real runs
    jax.config.update("jax_default_matmul_precision", experiment_config.system.precision)
    os.environ["WANDB_MODE"] = "offline"

    trainer_context = TrainerContext(
        run_name=run_name,
        # A new ID each time: bundle directories are never overwritten
        storage_id=create_model_id(experiment_config),
        model_directory=os.path.join(
            experiment_config.model_store.root, experiment_config.logging.wandb.group, ""
        ),
        data_augmenter=augmenter,
        observed_channels=experiment_config.data.emoji.observed_channels,
        data_channels=experiment_config.data.emoji.data_channels,
        evaluation_input=evaluation_input_provenance(data),
    )

    _loss_history = []

    def _plot_loss(iteration, loss, metrics):
        _loss_history.append(loss)
        _figure, _axis = plt.subplots(figsize=(8, 3.5))
        _axis.plot(_loss_history, label="training loss")
        _axis.plot(np.minimum.accumulate(_loss_history), "--", label="best loss")
        _axis.set(
            xlabel="iteration",
            ylabel="loss",
            yscale="log" if min(_loss_history) > 0 else "linear",
            title=f"iteration {iteration + 1}/{experiment_config.run.iterations}",
        )
        _axis.grid(alpha=0.25)
        _axis.legend()
        _figure.tight_layout()
        mo.output.replace(_figure)
        plt.close(_figure)

    training_result = run_training(
        experiment_config,
        model=model,
        data=data,
        context=trainer_context,
        key=train_key,
        progress_callback=_plot_loss,
    )
    trained_model_id = trainer_context.storage_id
    return trained_model_id, training_result


@app.cell(hide_code=True)
def _(trained_model_id, training_result):
    mo.callout(
        f"Training finished. Best loss `{training_result.best_loss:.4g}` at iteration "
        f"`{training_result.best_iteration}`. Published model `{trained_model_id}`.",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7. Model registry and inference

    A model bundle is a directory under
    `<model store>/bundles/<collection>/<experiment>/` holding `model.eqx`
    (the weights), `config.yaml` (the full experiment config) and
    `manifest.yaml` (checksums, git state, data fingerprint and training
    summary). Bundles are never edited after they are written.

    `ModelRegistry` (`Experiments/model_registry.py`) indexes a model store
    into a small SQLite database and returns it as pandas dataframes; the same
    is available from the command line as `python -m Experiments.model_registry list`.
    `bundle.load_model()` rebuilds the model from its saved `ModelConfig`
    and loads the weights. See `docs/model_registry.md`.

    Press *Refresh* (for example after training) to index the model store.
    The table only shows models trained by this notebook. Select one to run it
    from its initial condition. The data is loaded again from the saved data
    config and checked against the fingerprint saved at training time.
    """)
    return


@app.cell(hide_code=True)
def _():
    bundle_refresh = mo.ui.run_button(label="Refresh model list")
    return (bundle_refresh,)


@app.cell(hide_code=True)
def _(bundle_refresh, local_store_root):
    _rows = []
    _store_root = Path(local_store_root.value).expanduser()
    if bundle_refresh.value and _store_root.is_dir():
        _registry = ModelRegistry(_store_root)
        _registry.reindex()
        _models = _registry.models_df()
        if len(_models):
            _notebook_models = _models[
                (_models["collection"].fillna("").str.lower() == "nca-notebook")
                & _models["experiment"].fillna("").str.startswith("notebook_emoji_")
            ]
            _rows = _notebook_models.to_dict("records")
    bundle_table = mo.ui.table(_rows, selection="single", page_size=10)
    mo.vstack([bundle_refresh, bundle_table])
    return (bundle_table,)


@app.cell(hide_code=True)
def _(bundle_table, local_store_root):
    selected_bundle = None
    _selected = bundle_table.value
    if _selected is None or len(_selected) == 0:
        _bundle_detail = mo.md("Select a model to inspect its bundle.")
    else:
        _model_id = (
            _selected.iloc[0]["model_id"]
            if hasattr(_selected, "iloc")
            else _selected[0]["model_id"]
        )
        selected_bundle = ModelRegistry(Path(local_store_root.value).expanduser()).get(
            str(_model_id)
        )
        _bundle_detail = mo.md(
            "### Selected bundle\n\n"
            f"- ID: `{selected_bundle.id}`\n"
            f"- Dataset: `{selected_bundle.manifest.data.dataset}`, "
            f"task: `{selected_bundle.manifest.data.task}`\n"
            f"- Model: `{selected_bundle.config.model.family}` with "
            f"{selected_bundle.config.model.channels} channels\n"
            f"- Path: `{selected_bundle.path}`"
        )
    _bundle_detail
    return (selected_bundle,)


@app.cell(hide_code=True)
def _():
    inference_steps = mo.ui.number(1, 512, value=128, step=1, label="Inference steps")
    inference_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Inference seed")
    inference_fps = mo.ui.slider(1, 60, value=20, step=1, label="Video frames per second")
    run_inference = mo.ui.run_button(label="Run inference and render video")
    mo.vstack([
        mo.hstack([inference_steps, inference_seed, inference_fps]),
        run_inference,
    ])
    return inference_fps, inference_seed, inference_steps, run_inference


@app.cell
def _(data_path_base, inference_seed, inference_steps, run_inference, selected_bundle):
    mo.stop(
        not run_inference.value or selected_bundle is None,
        mo.callout("Select one model above, then run inference.", kind="info"),
    )
    if "evaluation_input" not in selected_bundle.manifest:
        raise ValueError("This bundle has no data fingerprint; retrain it with this notebook.")

    # Rebuild the training data from the saved config, and check it matches
    _inference_data, _ = load_data(
        selected_bundle.config.data,
        impath=str(Path(data_path_base.value).expanduser() / "Emojis"),
    )
    verify_evaluation_input(_inference_data, selected_bundle.manifest.evaluation_input)

    _key = jax.random.PRNGKey(int(inference_seed.value))
    _model = selected_bundle.load_model(key=_key)
    # The augmenter adds the hidden channels and padding, as in training
    _inference_augmenter, _ = build_data_augmenter(
        selected_bundle.config.data, _inference_data, _model.N_CHANNELS
    )
    _initial_state = _inference_augmenter.return_saved_data()[0][0]
    # model.run returns [steps + 1, channels, H, W], starting with the initial state
    inference_trajectory = np.asarray(
        _model.run(int(inference_steps.value), jnp.asarray(_initial_state), key=_key)
    )
    return (inference_trajectory,)


@app.cell(hide_code=True)
def _(inference_fps, inference_trajectory, selected_bundle):
    _observed_channels = int(selected_bundle.config.data.emoji.observed_channels)
    _hidden_channel_count = inference_trajectory.shape[1] - _observed_channels
    _panel_columns = 5
    _hidden_rows = max(1, -(-_hidden_channel_count // _panel_columns))
    _figure, _axes = plt.subplots(
        1 + _hidden_rows,
        _panel_columns,
        figsize=(8, 2.25 * (1 + _hidden_rows)),
        squeeze=False,
    )

    def _rgb_frame(frame):
        return np.clip(np.moveaxis(frame[:3], 0, -1), 0.0, 1.0)

    _rgb_artist = _axes[0, 0].imshow(_rgb_frame(inference_trajectory[0]))
    _axes[0, 0].set_title("RGB", fontsize=8)
    _observable_artists = []
    for _channel in range(min(_observed_channels, _panel_columns - 1)):
        _observable_artists.append(
            _axes[0, _channel + 1].imshow(
                np.clip(inference_trajectory[0, _channel], 0.0, 1.0),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
        )
        _axes[0, _channel + 1].set_title(f"observed {_channel}", fontsize=8)

    _hidden_values = inference_trajectory[:, _observed_channels:]
    _hidden_scale = max(float(np.max(np.abs(_hidden_values))) if _hidden_values.size else 0.0, 1e-6)
    _hidden_artists = []
    for _hidden_channel in range(_hidden_channel_count):
        _row, _column = divmod(_hidden_channel, _panel_columns)
        _hidden_artists.append(
            _axes[1 + _row, _column].imshow(
                inference_trajectory[0, _observed_channels + _hidden_channel],
                cmap="coolwarm",
                vmin=-_hidden_scale,
                vmax=_hidden_scale,
            )
        )
        _axes[1 + _row, _column].set_title(f"hidden {_hidden_channel}", fontsize=8)
    for _axis in _axes.flat:
        _axis.set_axis_off()
    _figure.tight_layout()

    def _render_frame(frame_index):
        _frame = inference_trajectory[frame_index]
        _rgb_artist.set_data(_rgb_frame(_frame))
        for _channel, _artist in enumerate(_observable_artists):
            _artist.set_data(np.clip(_frame[_channel], 0.0, 1.0))
        for _hidden_channel, _artist in enumerate(_hidden_artists):
            _artist.set_data(_frame[_observed_channels + _hidden_channel])
        return [_rgb_artist, *_observable_artists, *_hidden_artists]

    _movie = animation.FuncAnimation(
        _figure,
        _render_frame,
        frames=len(inference_trajectory),
        interval=1000 / int(inference_fps.value),
        blit=True,
    )
    with tempfile.TemporaryDirectory() as _video_directory:
        _video_path = Path(_video_directory) / "nca-inference.mp4"
        _movie.save(
            _video_path,
            writer=animation.FFMpegWriter(fps=int(inference_fps.value)),
            dpi=100,
        )
        _video_bytes = _video_path.read_bytes()
    plt.close(_figure)
    mo.vstack([
        mo.md(
            f"`{selected_bundle.id}` run for {len(inference_trajectory) - 1} steps "
            "from its initial condition. Top row: the observed (RGBA) channels; "
            "below: the hidden channels."
        ),
        mo.video(_video_bytes, controls=True, muted=True, autoplay=True, loop=True, width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Where next

    - **Objectives and the training pool:** `demo/nca_objectives_and_pool_dynamics.py`
      looks at loss terms, regularisers, the data augmenter and pool admission.
    - **Running real experiments:** start from
      `Experiments/emoji/conf/base_config.yaml`. A sweep file in
      `conf/experiments/` lists the settings to vary (`grid:` of dotted keys
      such as `run.t` or `model.channels`).
      `python Experiments/generate_configs.py` turns it into a manifest of
      configs, and the scripts in `launch/` run the manifest on the cluster.
      See the README.
    - **Other data:** `Experiments/micropatterns/` and `Experiments/snowmelt/`
      follow the same pattern as `Experiments/emoji/`, with their own
      `load_data`, `build_data_augmenter` and `train.py`.
    - **Looking at results:** `marimo edit Experiments/model_registry_explorer.py`
      searches and compares trained bundles; `Experiments/dataset_explorer.py`
      and the other `*_explorer.py` notebooks look at datasets and evaluations.
    """)
    return


if __name__ == "__main__":
    app.run()
