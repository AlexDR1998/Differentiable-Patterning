# /// script
# dependencies = ["marimo", "jax", "matplotlib", "numpy"]
# ///

"""A typed-config emoji NCA training walkthrough.

Run from the repository root with:

    marimo edit demo/nca_training_config_walkthrough.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import json
    import os
    import sys
    import tempfile
    from pathlib import Path

    _repository_root = Path(__file__).resolve().parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.dataloader.preprocessing import PreprocessingConfig
    from Common.trainer.config import (
        LossConfig,
        OptimizerConfig,
        PointwiseLossConfig,
        ScheduleConfig,
    )
    from Experiments.config import (
        CheckpointConfig,
        DataConfig,
        ExperimentConfig,
        ExperimentMetadataConfig,
        LoggingConfig,
        ModelStoreConfig,
        RuntimeConfig,
        TrainingConfig,
        TrainingLoopConfig,
        WandbConfig,
        config_to_dict,
    )
    from Experiments.config_helpers import build_model, set_matmul_precision
    from Experiments.emoji.config import (
        EmojiDataConfig,
        EmojiPairConfig,
        ProbabilityScheduleConfig,
    )
    from Experiments.emoji.config_helpers import build_data_augmenter, load_data
    from NCA.model.config import ModelConfig
    from NCA.registry import (
        ModelRegistry,
        create_model_id,
        evaluation_input_provenance,
        publish_model_bundle,
        verify_evaluation_input,
    )
    from NCA.trainer.config import PoolAdmissionConfig, TrainerConfig
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.trainer import build_trainer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA training walkthrough

    This notebook is first tutorial on using this codebase to train NCA to perform dynamical self organisation based on a target sequence of data. This is a high level overview of the data selection, model construction, training and evaluation workflow.

    Each section constructs the relevant
    configuration dataclasses owned by that part of the workflow.

    The example supports sequential emoji morphing and multi-attractor
    patterning. This notebook performs training locally, which should be quick (and low quality) with the default parameters.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Data selection and loading

    `PreprocessingConfig` owns reusable preprocessing, while `EmojiDataConfig`
    owns emoji-specific task and augmentation choices. They are composed here
    into the experiment-level `DataConfig`, then used immediately to load and
    preview the selected trajectories.
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
        label="Local model-store root",
        full_width=True,
    )
    mo.vstack([data_path_base, local_store_root])
    return data_path_base, local_store_root


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
                "### Available emoji files\n\n"
                + "\n".join(f"- `{_filename}`" for _filename in _filenames)
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
        label="Training objective",
    )
    emoji_sequence_text = mo.ui.text(
        value="crab.png, microbe.png",
        label="Sequential filenames (comma separated)",
        full_width=True,
    )
    attractor_pairs_text = mo.ui.text_area(
        value=(
            '[{"initial": {"image": "crab.png", "mode": "patch", "size": 4}, '
            '"target": "crab.png"}, '
            '{"initial": {"image": "microbe.png", "mode": "patch", "size": 4}, '
            '"target": "microbe.png"}]'
        ),
        label="Attractor pairs (JSON)",
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
        label="PreprocessingConfig.downsample",
    )
    target_repeats_value = mo.ui.slider(
        1,
        3,
        value=2,
        label="EmojiDataConfig.target_repeats",
    )
    mo.vstack([
        task_picker,
        mo.md(
            "Sequential training uses the filename sequence. Multi-attractor "
            "training uses the initial-condition/target pairs."
        ),
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


@app.cell(hide_code=True)
def _(data_config, data_config_error, data_path_base):
    data = None
    data_name = None
    data_load_error = data_config_error
    _image_root = Path(data_path_base.value).expanduser() / "Emojis"
    if data_load_error is None and not _image_root.is_dir():
        data_load_error = (
            "Set DATA_PATH_BASE to a directory containing an Emojis/ directory."
        )
    if data_load_error is None:
        try:
            data, data_name = load_data(data_config, impath=str(_image_root))
        except Exception as _error:
            data_load_error = str(_error)
    return data, data_load_error, data_name


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data details

    shown below are the constructed DataConfig, and images of the actual data. The images show time series targets from left to right, with any duplicate batches/trajectories vertically
    """)
    return


@app.cell(hide_code=True)
def _(
    attractor_pairs_text,
    downsample_value,
    emoji_sequence_text,
    target_repeats_value,
    task_picker,
):
    data_config = None
    data_config_error = None
    _task = task_picker.value
    try:
        _sequence = tuple(
            _item.strip()
            for _item in emoji_sequence_text.value.split(",")
            if _item.strip()
        )
        if _task == "sequence" and not _sequence:
            raise ValueError("At least one sequential filename is required.")
        _pairs = ()
        if _task == "multi_attractor":
            _raw_pairs = json.loads(attractor_pairs_text.value)
            _pairs = tuple(
                EmojiPairConfig(
                    initial=_item["initial"],
                    target=_item["target"],
                )
                for _item in _raw_pairs
            )
            if not _pairs:
                raise ValueError("At least one attractor pair is required.")

        data_config = DataConfig(
            dataset="emojis",
            batches=1,
            preprocessing=PreprocessingConfig(
                downsample=int(downsample_value.value)
            ),
            augmentation=EmojiDataConfig(
                task=_task,
                sequence=_sequence if _task == "sequence" else (),
                pairs=_pairs,
                target_repeats=int(target_repeats_value.value),
                pad=(2, 2, 2, 2),
                shift_amount=0,
                noise_strength=0.0,
                regenerate=False,
                terminal_carry=ProbabilityScheduleConfig(),
                regeneration=ProbabilityScheduleConfig(),
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as _error:
        data_config_error = str(_error)
    mo.md(
        "### Constructed `DataConfig`\n\n" f"```json\n{json.dumps(config_to_dict(data_config), indent=2)}\n```"
    )
    return data_config, data_config_error


@app.cell(hide_code=True)
def _(data, data_load_error, data_name):
    if data_load_error is not None:
        _data_output = mo.callout(data_load_error, kind="info")
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
                    np.clip(
                        np.moveaxis(_array[_batch, _time, :3], 0, -1),
                        0.0,
                        1.0,
                    )
                )
                _axes[_batch, _time].set_title(
                    f"trajectory {_batch}, state {_time}"
                )
                _axes[_batch, _time].set_axis_off()
        _figure.suptitle(f"{data_name} — {_array.shape}")
        _figure.tight_layout()
        # _data_output = mo.vstack([
        #     mo.md(
        #         "### Constructed `DataConfig`\n\n"
        #         f"```json\n{json.dumps(config_to_dict(data_config), indent=2)}\n```"
        #     ),
        #     _figure,
        # ])
    # _data_output
    _figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Model construction

    `ModelConfig` describes only the NCA architecture. NCA models are all subclasses of [Equinox modules](https://docs.kidger.site/equinox/api/module/module/), which allows for very flexible gradient based optimisation. The NCA class has custom `partition()` and `combine()` methods, which tell equinox exactly which parts of the NCA model to apply gradient updates to via [`filter_grad`](https://docs.kidger.site/equinox/api/transformations/#equinox.filter_grad).


    The selected seed is
    used to initialize its Equinox parameters and is later reused as the
    experiment seed when the complete configuration is assembled.
    """)
    return


@app.cell(hide_code=True)
def _():
    seed_value = mo.ui.number(
        0,
        1_000_000,
        value=0,
        step=1,
        label="Experiment seed",
    )
    channel_count = mo.ui.dropdown(
        options={
            "8 channels": 8,
            "12 channels": 12,
            "16 channels": 16,
            "24 channels": 24,
            "32 channels": 32,
        },
        value="8 channels",
        label="ModelConfig.channels",
    )
    fire_rate_value = mo.ui.dropdown(
        options={"0.5 stochastic": 0.5, "1.0 deterministic": 1.0},
        value="0.5 stochastic",
        label="ModelConfig.fire_rate",
    )
    mo.hstack([seed_value, channel_count, fire_rate_value])
    return channel_count, fire_rate_value, seed_value


@app.cell
def _(channel_count, fire_rate_value):
    model_config = ModelConfig(
        family="NCA",
        channels=int(channel_count.value),
        activation="relu",
        kernel_str=("ID", "LAP", "GRAD"),
        fire_rate=float(fire_rate_value.value),
        padding="CIRCULAR",
    )
    return (model_config,)


@app.cell
def _(model_config, seed_value):
    model_key, train_key = jax.random.split(
        jax.random.PRNGKey(int(seed_value.value))
    )
    model, model_name = build_model(model_config, key=model_key)
    return model, model_name, train_key


@app.cell(hide_code=True)
def _(model, model_config, model_name):
    _differentiable, _static = model.partition()
    _parameter_count = sum(
        int(np.prod(_leaf.shape))
        for _leaf in jax.tree_util.tree_leaves(_differentiable)
        if hasattr(_leaf, "shape")
    )
    mo.vstack([
        mo.md(
            "### Constructed `ModelConfig`\n\n"
            f"```json\n{json.dumps(config_to_dict(model_config), indent=2)}\n```"
        ),
        mo.callout(
            f"Built `{model_name}` with {_parameter_count:,} trainable parameters.",
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # 3. Training and optimizer setup

    This section constructs the loop, trainer, optimizer, loss, and checkpoint
    dataclasses. It then combines those values with the `DataConfig` and
    `ModelConfig` from the previous sections to create the one
    `ExperimentConfig` consumed by the trainer.

    Some important parameters include:
     - Learning rate
     - loop_autodiff: set to lax for fastest runtime but higher memory usage, set to checkpointed if lax causes OOM errors
     - LossConfig: as with most ML, constructing the right loss function is often the hardest part
    """)
    return


@app.cell(hide_code=True)
def _():
    rollout_steps = mo.ui.dropdown(
        options={
            "4 steps": 4,
            "8 steps": 8,
            "12 steps": 12,
            "16 steps": 16,
            "32 steps": 32,
        },
        value="16 steps",
        label="TrainingLoopConfig.t",
    )
    iteration_count = mo.ui.dropdown(
        options={
            "5 iterations": 5,
            "10 iterations": 10,
            "100 iterations": 100,
            "1000 iterations": 1000,
        },
        value="100 iterations",
        label="TrainingLoopConfig.iterations",
    )
    learning_rate_value = mo.ui.number(
        0.0001,
        0.01,
        value=0.001,
        step=0.0001,
        label="OptimizerConfig.learn_rate",
    )
    mo.vstack([rollout_steps, iteration_count, learning_rate_value])
    return iteration_count, learning_rate_value, rollout_steps


@app.cell
def _(iteration_count, learning_rate_value, rollout_steps):
    training_loop_config = TrainingLoopConfig(
        t=int(rollout_steps.value),
        iterations=int(iteration_count.value),
        write_images=False,
    )
    trainer_config = TrainerConfig(
        grad_loss=False,
        loop_autodiff="checkpointed",
        log_every=max(1, int(iteration_count.value) // 2),
        pool_admission=PoolAdmissionConfig(enabled=False),
    )
    optimizer_config = OptimizerConfig(
        learn_rate=float(learning_rate_value.value),
        warmup_steps=0,
        schedule=ScheduleConfig(type="cosine",final_factor=0.2)
    )
    loss_config = LossConfig(
        terms=(PointwiseLossConfig(type="l2"),),
        regularisers={},
    )
    checkpoint_config = CheckpointConfig(warmup=0)
    training_config = TrainingConfig(
        loop=training_loop_config,
        trainer=trainer_config,
        optimizer=optimizer_config,
        loss=loss_config,
        checkpoint=checkpoint_config,
    )
    return (training_config,)


@app.cell
def _(
    data_config,
    data_config_error,
    local_store_root,
    model_config,
    seed_value,
    task_picker,
    training_config,
):
    experiment_config = None
    experiment_config_error = data_config_error
    if experiment_config_error is None:
        try:
            experiment_config = ExperimentConfig(
                schema_version=1,
                seed=int(seed_value.value),
                experiment=ExperimentMetadataConfig(
                    name=f"notebook_emoji_{task_picker.value}"
                ),
                runtime=RuntimeConfig(precision="highest"),
                data=data_config,
                model=model_config,
                training=training_config,
                logging=LoggingConfig(
                    backend="wandb",
                    wandb=WandbConfig(
                        project="NCA-notebook",
                        group=f"single-config-{task_picker.value}",
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
        _config_output = mo.accordion({
            "Constructed ExperimentConfig": mo.md(
                f"```json\n{json.dumps(config_to_dict(experiment_config), indent=2)}\n```"
            )
        })
    _config_output
    return


@app.cell
def _(
    data,
    data_name,
    experiment_config,
    experiment_config_error,
    model,
    model_name,
):
    augmenter = None
    augmenter_name = None
    run_name = None
    trainer = None
    trainer_context = None
    trainer_setup_error = experiment_config_error
    if trainer_setup_error is None and data is None:
        trainer_setup_error = "Load the selected data before constructing the trainer."
    if trainer_setup_error is None:
        try:
            augmenter, augmenter_name = build_data_augmenter(experiment_config.data)
            run_name = (
                f"{experiment_config.experiment.name}_{model_name}_"
                f"{data_name}_{augmenter_name}"
            )
            trainer_context = TrainerContext(
                run_name=run_name,
                storage_id=create_model_id(experiment_config),
                model_directory=os.path.join(
                    experiment_config.model_store.root,
                    experiment_config.logging.wandb.group,
                    "",
                ),
                data_augmenter=augmenter,
                observed_channels=experiment_config.data.emoji.observed_channels,
                data_channels=experiment_config.data.emoji.data_channels,
                loss_time_channel_mask=(
                    experiment_config.training.trainer.loss_time_channel_mask
                ),
                evaluation_input=evaluation_input_provenance(data),
            )
            trainer = build_trainer(
                experiment_config,
                model,
                data=data,
                context=trainer_context,
            )
        except Exception as _error:
            trainer_setup_error = str(_error)
    return run_name, trainer, trainer_context, trainer_setup_error


@app.cell(hide_code=True)
def _(run_name, trainer_setup_error):
    if trainer_setup_error is not None:
        mo.callout(trainer_setup_error, kind="danger")
    else:
        mo.callout(
            f"Trainer is ready for `{run_name}`. Optimizer, loss, logging, and "
            "checkpoint state will be resolved from `ExperimentConfig` at train time.",
            kind="success",
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Actual training

    The following cell just runs `trainer.train(train_key)`, but wrapped in some nice dynamic loss plotting. `trainer.train` owns preparation, compilation, and execution. Depending on data resolution, training iterations, and model timesteps between target images, this can be quick or slow.
    """)
    return


@app.cell(hide_code=True)
def _(trainer_setup_error):
    training_is_ready = trainer_setup_error is None
    training_button = mo.ui.run_button(
        label="Run this local typed-config training job",
        disabled=not training_is_ready,
    )
    if training_is_ready:
        _training_control = training_button
    else:
        _training_control = mo.vstack([
            mo.callout(trainer_setup_error, kind="danger"),
            training_button,
        ])
    _training_control
    return (training_button,)


@app.cell(hide_code=True)
def _(
    data_path_base,
    experiment_config,
    run_name,
    train_key,
    trainer,
    trainer_context,
    training_button,
):
    training_outcome = None
    if training_button.value:
        os.environ["DATA_PATH_BASE"] = data_path_base.value.strip()
        os.environ["MODEL_STORE_ROOT"] = experiment_config.model_store.root
        os.environ["WANDB_MODE"] = "offline"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        set_matmul_precision(experiment_config.runtime)

        _loss_history = []

        def _update_loss_plot(iteration, loss, metrics):
            del metrics
            _loss_history.append(loss)
            _iterations = np.arange(1, len(_loss_history) + 1)
            _best_losses = np.minimum.accumulate(_loss_history)
            _figure, _axis = plt.subplots(figsize=(8, 4))
            _axis.plot(_iterations, _loss_history, label="Training loss")
            _axis.plot(
                _iterations,
                _best_losses,
                linestyle="--",
                label="Best loss",
            )
            _axis.set(
                xlabel="Iteration",
                ylabel="Loss",
                title=(
                    f"Training loss — iteration {iteration + 1}/"
                    f"{experiment_config.training.loop.iterations}"
                ),
            )
            if all(_loss > 0 for _loss in _loss_history):
                _axis.set_yscale("log")
            _axis.grid(alpha=0.25)
            _axis.legend()
            _figure.tight_layout()
            mo.output.replace(_figure)
            plt.close(_figure)

        _training_result = trainer.train(
            key=train_key,
            progress_callback=_update_loss_plot,
        )
        _bundle_path = None
        if _training_result.checkpoint_path is not None:
            _bundle = publish_model_bundle(
                store_root=experiment_config.model_store.root,
                collection=(
                    experiment_config.model_store.collection
                    or experiment_config.logging.wandb.project
                ),
                model_id=trainer_context.storage_id,
                display_name=run_name,
                checkpoint_path=_training_result.checkpoint_path,
                cfg=experiment_config,
                training_result=_training_result,
                model_factory=experiment_config.model_store.model_factory,
                evaluation_input=trainer_context.evaluation_input,
            )
            _bundle.verify()
            _training_result.checkpoint_path.unlink()
            _bundle_path = str(_bundle.path)
        training_outcome = {
            "run_name": run_name,
            "bundle_path": _bundle_path,
            "best_loss": _training_result.best_loss,
        }
    return (training_outcome,)


@app.cell(hide_code=True)
def _(training_outcome):
    if training_outcome is None:
        mo.callout("No training has been started in this session.", kind="info")
    else:
        mo.callout(
            f"Training completed: `{training_outcome['run_name']}`. "
            f"Bundle: `{training_outcome['bundle_path']}`. "
            f"Best loss: `{training_outcome['best_loss']}`.",
            kind="success",
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Model registry, evaluation, and inference

    The registry table is restricted to bundles published by this notebook.
    Select one model to reload its immutable configuration and checkpoint.
    Inference reconstructs and verifies the saved data-derived initial condition
    before rendering observable and hidden state channels.
    """)
    return


@app.cell(hide_code=True)
def _():
    bundle_refresh = mo.ui.run_button(label="Refresh notebook model bundles")
    return (bundle_refresh,)


@app.cell(hide_code=True)
def _(bundle_refresh, local_store_root, training_outcome):
    _rows = []
    _store_root = Path(local_store_root.value).expanduser()
    if bundle_refresh.value or training_outcome is not None:
        _registry = ModelRegistry(_store_root)
        _registry.reindex()
        _models = _registry.models_df()
        _notebook_models = _models[
            (_models["collection"].fillna("").str.lower() == "nca-notebook")
            & _models["experiment"].fillna("").str.startswith("notebook_emoji_")
        ]
        _rows = _notebook_models.to_dict("records")
    bundle_table = mo.ui.table(
        _rows,
        selection="single",
        page_size=10,
    )
    mo.vstack([
        bundle_refresh,
        mo.md(
            "Only the `nca-notebook` collection with `notebook_emoji_*` "
            "experiment names is shown."
        ),
        bundle_table,
    ])
    return (bundle_table,)


@app.cell(hide_code=True)
def _(bundle_table, local_store_root):
    _selected = bundle_table.value
    if _selected is None or len(_selected) == 0:
        _bundle_detail = mo.md("Select a model to inspect its bundle.")
    else:
        _model_id = (
            _selected.iloc[0]["model_id"]
            if hasattr(_selected, "iloc")
            else _selected[0]["model_id"]
        )
        _bundle = ModelRegistry(
            Path(local_store_root.value).expanduser()
        ).get(str(_model_id))
        _bundle_detail = mo.md(
            "### Selected bundle\n\n"
            f"- ID: `{_bundle.id}`\n"
            f"- Dataset: `{_bundle.manifest.data.get('dataset', 'unknown')}`\n"
            f"- Task: `{_bundle.manifest.data.get('task', 'unknown')}`\n"
            f"- Path: `{_bundle.path}`"
        )
    _bundle_detail
    return


@app.cell(hide_code=True)
def _():
    inference_steps = mo.ui.number(
        1,
        512,
        value=128,
        step=1,
        label="Inference steps",
    )
    inference_seed = mo.ui.number(
        0,
        2**31 - 1,
        value=0,
        step=1,
        label="Inference seed",
    )
    inference_fps = mo.ui.slider(
        1,
        60,
        value=20,
        step=1,
        label="Video frames per second",
    )
    run_inference = mo.ui.run_button(label="Run inference and render video")
    mo.vstack([
        mo.hstack([inference_steps, inference_seed, inference_fps]),
        run_inference,
    ])
    return inference_fps, inference_seed, inference_steps, run_inference


@app.cell(hide_code=True)
def _(
    bundle_table,
    data_path_base,
    inference_fps,
    inference_seed,
    inference_steps,
    local_store_root,
    run_inference,
):
    if not run_inference.value:
        _inference_output = mo.callout(
            "Select one notebook model and run inference to generate its video.",
            kind="info",
        )
    else:
        try:
            _selected = bundle_table.value
            if _selected is None or len(_selected) != 1:
                raise ValueError("Select exactly one notebook model.")
            _model_id = (
                _selected.iloc[0]["model_id"]
                if hasattr(_selected, "iloc")
                else _selected[0]["model_id"]
            )
            _registry = ModelRegistry(
                Path(local_store_root.value).expanduser()
            )
            _bundle = _registry.get(str(_model_id))
            if (
                str(_bundle.manifest.collection).lower() != "nca-notebook"
                or not str(_bundle.manifest.experiment).startswith("notebook_emoji_")
            ):
                raise ValueError("The selected bundle was not generated by this notebook.")
            if "evaluation_input" not in _bundle.manifest:
                raise ValueError(
                    "The bundle has no evaluation-input fingerprint; retrain it "
                    "with this notebook."
                )

            _image_root = Path(data_path_base.value).expanduser() / "Emojis"
            _inference_data, _ = load_data(
                _bundle.config.data,
                impath=str(_image_root),
            )
            verify_evaluation_input(
                _inference_data,
                _bundle.manifest.evaluation_input,
            )

            _key = jax.random.PRNGKey(int(inference_seed.value))
            _inference_model = _bundle.load_model(key=_key)
            _observed_channels = int(
                _bundle.config.data.emoji.observed_channels
            )
            _input_hidden_channels = int(_inference_model.N_CHANNELS) - int(
                _bundle.config.data.emoji.data_channels
            )
            _hidden_channel_count = (
                int(_inference_model.N_CHANNELS) - _observed_channels
            )
            if _input_hidden_channels < 0 or _hidden_channel_count < 0:
                raise ValueError(
                    "Model has fewer channels than its configured input data."
                )

            _augmenter_class, _ = build_data_augmenter(_bundle.config.data)
            _inference_augmenter = _augmenter_class(
                data_true=_inference_data,
                hidden_channels=_input_hidden_channels,
            )
            _inference_augmenter.data_init(
                _bundle.config.training.trainer.sharding
            )
            _initial_state = _inference_augmenter.return_saved_data()[0][0]
            _trajectory = np.asarray(
                _inference_model.run(
                    int(inference_steps.value),
                    jnp.asarray(_initial_state),
                    key=_key,
                )
            )

            _panel_columns = 5
            if _observed_channels > _panel_columns - 1:
                raise ValueError(
                    "Video layout supports at most four observable channels."
                )
            _hidden_rows = max(
                1,
                (_hidden_channel_count + _panel_columns - 1) // _panel_columns,
            )
            _panel_rows = 1 + _hidden_rows
            _figure, _axes = plt.subplots(
                _panel_rows,
                _panel_columns,
                figsize=(8, 2.25 * _panel_rows),
                squeeze=False,
            )

            def _rgb_frame(frame):
                _observable = frame[:_observed_channels]
                _rgb = np.moveaxis(
                    _observable[: min(3, len(_observable))],
                    0,
                    -1,
                )
                if _rgb.shape[-1] < 3:
                    _rgb = np.pad(
                        _rgb,
                        ((0, 0), (0, 0), (0, 3 - _rgb.shape[-1])),
                    )
                return np.clip(_rgb, 0.0, 1.0)

            _rgb_artist = _axes[0, 0].imshow(_rgb_frame(_trajectory[0]))
            _axes[0, 0].set_title("RGB composite", fontsize=8)
            _observable_artists = []
            for _channel in range(_observed_channels):
                _artist = _axes[0, _channel + 1].imshow(
                    np.clip(_trajectory[0, _channel], 0.0, 1.0),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                )
                _axes[0, _channel + 1].set_title(
                    f"Observable {_channel}",
                    fontsize=8,
                )
                _observable_artists.append(_artist)

            _hidden_values = _trajectory[:, _observed_channels:]
            _hidden_scale = max(
                float(np.max(np.abs(_hidden_values)))
                if _hidden_values.size
                else 0.0,
                1e-6,
            )
            _hidden_artists = []
            for _hidden_channel in range(_hidden_channel_count):
                _state_channel = _observed_channels + _hidden_channel
                _hidden_row, _hidden_column = divmod(
                    _hidden_channel,
                    _panel_columns,
                )
                _artist = _axes[1 + _hidden_row, _hidden_column].imshow(
                    _trajectory[0, _state_channel],
                    cmap="coolwarm",
                    vmin=-_hidden_scale,
                    vmax=_hidden_scale,
                )
                _axes[1 + _hidden_row, _hidden_column].set_title(
                    f"Hidden {_hidden_channel}",
                    fontsize=8,
                )
                _hidden_artists.append(_artist)

            for _axis in _axes.flat:
                _axis.set_axis_off()
            _figure.tight_layout()

            def _render_frame(frame_index):
                _rgb_artist.set_data(_rgb_frame(_trajectory[frame_index]))
                for _channel, _artist in enumerate(_observable_artists):
                    _artist.set_data(
                        np.clip(
                            _trajectory[frame_index, _channel],
                            0.0,
                            1.0,
                        )
                    )
                for _hidden_channel, _artist in enumerate(_hidden_artists):
                    _artist.set_data(
                        _trajectory[
                            frame_index,
                            _observed_channels + _hidden_channel,
                        ]
                    )
                return [
                    _rgb_artist,
                    *_observable_artists,
                    *_hidden_artists,
                ]

            _movie = animation.FuncAnimation(
                _figure,
                _render_frame,
                frames=len(_trajectory),
                interval=1000 / int(inference_fps.value),
                blit=True,
            )
            with tempfile.TemporaryDirectory() as _video_directory:
                _video_path = Path(_video_directory) / "nca-inference.mp4"
                _movie.save(
                    _video_path,
                    writer=animation.FFMpegWriter(
                        fps=int(inference_fps.value)
                    ),
                    dpi=100,
                )
                _video_bytes = _video_path.read_bytes()
            plt.close(_figure)
            _inference_output = mo.vstack([
                mo.md(
                    f"Verified and rendered `{_bundle.id}` from its configured "
                    f"initial condition for {int(inference_steps.value)} steps."
                ),
                mo.video(
                    _video_bytes,
                    controls=True,
                    muted=True,
                    autoplay=True,
                    loop=True,
                    width="100%",
                    rounded=True,
                ),
            ])
        except Exception as _error:
            _inference_output = mo.callout(str(_error), kind="danger")
    _inference_output
    return


if __name__ == "__main__":
    app.run()
