# /// script
# dependencies = ["marimo", "jax", "matplotlib", "numpy", "pyyaml"]
# ///

"""NCA objectives, regularisers, the data augmenter and pool dynamics.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/nca_objectives_and_pool_dynamics.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import os
    import tempfile
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
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
    )
    from Experiments.emoji.config import EmojiDataConfig, ProbabilityScheduleConfig
    from Experiments.emoji.config_helpers import build_data_augmenter, load_data
    from Experiments.nca_training import run_training
    from NCA.model.config import ModelConfig
    from NCA.model.factory import build_model
    from NCA.trainer.config import PoolAdmissionConfig, TrainerConfig
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.data_augmenter import reinject_observations
    from NCA.trainer.objective import resolve_objective
    from NCA.trainer.pool import PoolAdmissionController


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA objectives and pool dynamics

    This follows on from `demo/nca_training_config_walkthrough.py`, which
    goes through the whole training workflow. Here the model and data are
    kept small and fixed, and we look at the parts that change what the NCA
    learns:

    1. the **loss terms and regularisers** (`cfg.loss`);
    2. the **data augmenter**, which builds the training pool
       (`NCA/trainer/data_augmenter/`, set up from `cfg.data.emoji`);
    3. **pool admission**, which decides whether a rollout is written back
       into the pool (`NCA/trainer/pool.py`, set up from
       `cfg.trainer.pool_admission`).

    Each section calls the same functions the trainer uses, rather than a
    simplified copy. The short training run at the end is optional.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. A fixed, cheap task

    Two small emoji frames: the NCA should turn the first into the second.
    Keeping this task fixed makes it easier to compare settings.
    """)
    return


@app.cell(hide_code=True)
def _():
    _repo_root_pool_lab = Path(__file__).resolve().parents[1]
    pool_lab_data_path = mo.ui.text(
        value=os.environ.get("DATA_PATH_BASE", str(_repo_root_pool_lab / "demo" / "demo_data")),
        label="DATA_PATH_BASE (must contain Emojis/)",
        full_width=True,
    )
    pool_lab_source_name = mo.ui.text(value="crab.png", label="Initial frame")
    pool_lab_target_name = mo.ui.text(value="microbe.png", label="Target frame")
    pool_lab_seed = mo.ui.number(0, 1_000_000, value=0, step=1, label="Seed")
    mo.vstack([
        pool_lab_data_path,
        mo.hstack([pool_lab_source_name, pool_lab_target_name, pool_lab_seed]),
    ])
    return (
        pool_lab_data_path,
        pool_lab_seed,
        pool_lab_source_name,
        pool_lab_target_name,
    )


@app.cell(hide_code=True)
def _(pool_lab_data_path):
    _emoji_root_pool_lab = Path(pool_lab_data_path.value).expanduser() / "Emojis"
    if _emoji_root_pool_lab.is_dir():
        _available_pool_lab = sorted(
            _path.name
            for _path in _emoji_root_pool_lab.iterdir()
            if _path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        )
        _available_output_pool_lab = mo.md(
            "Available files: " + ", ".join(f"`{_name}`" for _name in _available_pool_lab)
        )
    else:
        _available_output_pool_lab = mo.callout(
            "Set DATA_PATH_BASE to a directory containing `Emojis/`.", kind="info"
        )
    _available_output_pool_lab
    return


@app.cell
def _(pool_lab_source_name, pool_lab_target_name):
    pool_lab_emoji_config = EmojiDataConfig(
        task="sequence",
        sequence=(pool_lab_source_name.value.strip(), pool_lab_target_name.value.strip()),
        pad=(4, 4, 4, 4),
        # Augmentation is switched off for training here; section 3 explores it
        shift_amount=0,
        noise_strength=0.0,
    )
    pool_lab_data_config = DataConfig(
        dataset="emojis",
        batches=2,
        downsample=4,
        emoji=pool_lab_emoji_config,
    )
    return (pool_lab_data_config,)


@app.cell
def _(pool_lab_data_config, pool_lab_data_path):
    pool_lab_data = None
    pool_lab_data_error = None
    try:
        pool_lab_data, _pool_lab_data_name = load_data(
            pool_lab_data_config,
            impath=str(Path(pool_lab_data_path.value).expanduser() / "Emojis"),
        )
    except Exception as _error_pool_lab:
        pool_lab_data_error = str(_error_pool_lab)
    return pool_lab_data, pool_lab_data_error


@app.cell(hide_code=True)
def _(pool_lab_data, pool_lab_data_error):
    if pool_lab_data_error is not None:
        _data_preview_pool_lab = mo.callout(pool_lab_data_error, kind="danger")
    else:
        _frames_pool_lab = np.asarray(pool_lab_data)[0]
        _figure_pool_lab, _axes_pool_lab = plt.subplots(1, len(_frames_pool_lab), figsize=(6, 3))
        for _time_pool_lab, _axis_pool_lab in enumerate(np.atleast_1d(_axes_pool_lab)):
            _axis_pool_lab.imshow(
                np.clip(np.moveaxis(_frames_pool_lab[_time_pool_lab, :3], 0, -1), 0, 1)
            )
            _axis_pool_lab.set(
                title="initial condition" if _time_pool_lab == 0 else f"target {_time_pool_lab}",
                xticks=[],
                yticks=[],
            )
        _figure_pool_lab.tight_layout()
        _data_preview_pool_lab = _figure_pool_lab
    _data_preview_pool_lab
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Loss terms and regularisers

    `cfg.loss` has two parts:

    - `terms`: what the observed channels should match. Each term has a
      `type` (a name in `Common/trainer/loss_table.py`) and a `weight`. Weights
      are *relative*: the trainer takes the weighted average of the terms.
    - `regularisers`: how the NCA should get there, as a mapping from name
      (see `NCA/trainer/NCA_regulariser.py`) to coefficient \(\lambda_r\).
      Each regulariser is evaluated at every NCA step, not just at the target
      times, and coefficients are *added* to the loss.

    For a rollout of \(T\) NCA steps the trainer minimises

    \[
    \mathcal L = \operatorname{mean}(\mathcal L_\mathrm{data}) +
    \sum_r \lambda_r\,\frac{1}{T}\operatorname{mean}
    \left(\sum_{t=1}^{T} r_t\right).
    \]

    Before training, `NCA/trainer/objective.py:resolve_objective` turns the
    loss config into the loss names, the options they share and the
    regulariser coefficients. Terms whose options disagree are rejected at
    this point, before anything is compiled.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_loss_mode = mo.ui.radio(
        options={
            "L2 pixels": "l2",
            "Spectral structure": "spectral",
            "L2 + 0.25 spectral": "l2_spectral",
        },
        value="L2 pixels",
        label="Loss terms",
    )
    pool_lab_regulariser_name = mo.ui.dropdown(
        options={
            "None": "none",
            "Keep states in [0, 1] (intermediate_state)": "intermediate_state",
            "Keep hidden channels small (hidden_state_size)": "hidden_state_size",
            "Encourage contiguous observable growth (contiguous_growth)": "contiguous_growth",
            "Localise hidden activity (localised_hidden)": "localised_hidden",
        },
        value="None",
        label="Regulariser",
    )
    pool_lab_regulariser_weight = mo.ui.number(
        0.0, 10.0, value=0.1, step=0.01, label="Regulariser coefficient"
    )
    mo.vstack([
        pool_lab_loss_mode,
        mo.hstack([pool_lab_regulariser_name, pool_lab_regulariser_weight]),
    ])
    return (
        pool_lab_loss_mode,
        pool_lab_regulariser_name,
        pool_lab_regulariser_weight,
    )


@app.cell
def _(pool_lab_loss_mode, pool_lab_regulariser_name, pool_lab_regulariser_weight):
    if pool_lab_loss_mode.value == "l2_spectral":
        _terms_pool_lab = (
            PointwiseLossConfig(type="l2", weight=1.0),
            PointwiseLossConfig(type="spectral", weight=0.25),
        )
    else:
        _terms_pool_lab = (PointwiseLossConfig(type=pool_lab_loss_mode.value),)
    _regularisers_pool_lab = (
        {}
        if pool_lab_regulariser_name.value == "none"
        else {pool_lab_regulariser_name.value: float(pool_lab_regulariser_weight.value)}
    )
    pool_lab_loss_config = LossConfig(terms=_terms_pool_lab, regularisers=_regularisers_pool_lab)
    pool_lab_objective = resolve_objective(pool_lab_loss_config)
    return pool_lab_loss_config, pool_lab_objective


@app.cell(hide_code=True)
def _(pool_lab_loss_config, pool_lab_objective):
    mo.vstack([
        mo.md(
            "### `cfg.loss` as YAML\n\n"
            f"```yaml\n{yaml.safe_dump(config_to_dict(pool_lab_loss_config), sort_keys=False)}```"
        ),
        mo.md(
            "### What `resolve_objective` returns\n\n"
            f"- Loss functions: `{pool_lab_objective.names}`\n"
            f"- Relative weights: `{pool_lab_objective.arguments['component_weights']}`\n"
            f"- Regulariser coefficients: `{pool_lab_objective.regulariser_coefficients}`"
        ),
        mo.callout(
            "`l2` and `spectral` are cheap pointwise losses with no options, so they can "
            "always be combined. Losses with options (e.g. the VGG and optimal transport "
            "losses) can only be combined when their shared options agree.",
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The data augmenter and the training pool

    The trainer keeps a *pool* of states: for each trajectory and each
    transition between images, a state `x` to start from and a target `y` to
    reach after `run.t` NCA steps. After every training step it calls
    `augmenter.advance_pool(x, y, iteration, key)` with the states the NCA
    actually reached, and trains on what it returns next.

    For emoji data this is `EmojiAugmenter`
    (`NCA/trainer/data_augmenter/emoji.py`). Its options are constructor
    arguments, filled in from `cfg.data.emoji` by `build_data_augmenter`. Each
    step it:

    1. moves every trajectory on by one slot, so the state reached from
       image \(k\) becomes the start of the next transition;
    2. puts the true image back into the observed channels of half the slots
       (`reinject_observations`), keeping the hidden channels;
    3. optionally keeps the final predicted state (`terminal_carry`), rolls
       the images by a random offset (`shift_amount`), cuts out a random
       circle (`regeneration`, to learn to repair damage) and blends in noise
       (`noise_strength`).

    All the randomness comes from the `key` that is passed in, so a run can be
    repeated exactly. The functions used in each stage are in
    `data_augmenter/transforms.py` and can be called on their own.

    ### 3a. Reinjection on its own

    A toy pool with one trajectory and three slots. The observed channel is
    set to `0`, `0.5` and `1` in the three true slots; the hidden channel of
    the pool starts at `0.8`. Change the fraction and note that reinjection
    only changes the observed channel, and that slot 0 always gets the true
    initial condition.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_reinjection_fraction = mo.ui.slider(
        0.0, 1.0, value=0.5, step=0.5, label="Fraction of slots reinjected"
    )
    pool_lab_reinjection_fraction
    return (pool_lab_reinjection_fraction,)


@app.cell(hide_code=True)
def _(pool_lab_reinjection_fraction):
    # Pool [batch=1, time=3, channels=4, 5, 5]: channels 0-1 observed, 2-3 hidden
    _pool_before_pool_lab = jnp.zeros((1, 3, 4, 5, 5), dtype=jnp.float32)
    _pool_before_pool_lab = _pool_before_pool_lab.at[:, :, 2:].set(0.8)
    _truth_pool_lab = jnp.zeros_like(_pool_before_pool_lab)
    _truth_pool_lab = _truth_pool_lab.at[:, 1, :2].set(0.5)
    _truth_pool_lab = _truth_pool_lab.at[:, 2, :2].set(1.0)
    _pool_after_pool_lab = reinject_observations(
        _pool_before_pool_lab,
        _truth_pool_lab,
        observable_channels=2,
        key=jax.random.PRNGKey(17),
        fraction=float(pool_lab_reinjection_fraction.value),
    )
    _figure_reinject_pool_lab, _axes_reinject_pool_lab = plt.subplots(2, 3, figsize=(8, 5))
    for _time_reinject_pool_lab in range(3):
        for _row_pool_lab, (_channel_pool_lab, _label_pool_lab) in enumerate(
            ((0, "observed"), (2, "hidden"))
        ):
            _axis_reinject_pool_lab = _axes_reinject_pool_lab[_row_pool_lab, _time_reinject_pool_lab]
            _axis_reinject_pool_lab.imshow(
                np.asarray(_pool_after_pool_lab[0, _time_reinject_pool_lab, _channel_pool_lab]),
                vmin=0,
                vmax=1,
            )
            _axis_reinject_pool_lab.set(
                title=f"{_label_pool_lab}, slot {_time_reinject_pool_lab}", xticks=[], yticks=[]
            )
    _figure_reinject_pool_lab.suptitle("Pool after propagation and reinjection")
    _figure_reinject_pool_lab.tight_layout()
    _figure_reinject_pool_lab
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3b. The full augmenter on the emoji data

    Here an `EmojiAugmenter` is built from the data in section 1, with the
    options below, and `advance_pool` is called once. The pool the NCA would
    reach is imitated by the true pool, so the effects of shifting, damage
    and noise are easy to see. Each row is one copy of the trajectory in the
    pool (`data.batches = 2`), each column one time slot.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_shift = mo.ui.slider(0, 8, value=4, step=1, label="data.emoji.shift_amount")
    pool_lab_noise = mo.ui.slider(0.0, 0.5, value=0.05, step=0.05, label="data.emoji.noise_strength")
    pool_lab_regeneration = mo.ui.switch(value=True, label="data.emoji.regeneration.enabled")
    pool_lab_augmenter_key = mo.ui.number(0, 1000, value=0, step=1, label="Key")
    mo.hstack([pool_lab_shift, pool_lab_noise, pool_lab_regeneration, pool_lab_augmenter_key])
    return pool_lab_augmenter_key, pool_lab_noise, pool_lab_regeneration, pool_lab_shift


@app.cell
def _(
    pool_lab_augmenter_key,
    pool_lab_data,
    pool_lab_data_config,
    pool_lab_noise,
    pool_lab_regeneration,
    pool_lab_shift,
):
    mo.stop(pool_lab_data is None)
    _augmented_emoji_config = EmojiDataConfig(
        sequence=pool_lab_data_config.emoji.sequence,
        pad=pool_lab_data_config.emoji.pad,
        shift_amount=int(pool_lab_shift.value),
        noise_strength=float(pool_lab_noise.value),
        regeneration=ProbabilityScheduleConfig(
            enabled=bool(pool_lab_regeneration.value),
            initial_probability=1.0,
            final_probability=1.0,
        ),
    )
    _augmented_data_config = DataConfig(
        dataset="emojis",
        batches=pool_lab_data_config.batches,
        downsample=pool_lab_data_config.downsample,
        emoji=_augmented_emoji_config,
    )
    _augmenter, _ = build_data_augmenter(_augmented_data_config, pool_lab_data, model_channels=8)
    _key = jax.random.PRNGKey(int(pool_lab_augmenter_key.value))
    _x, _y = _augmenter.initialize_pool(_key)
    pool_lab_augmented_x, _ = _augmenter.advance_pool(_x, _y, 1, jax.random.fold_in(_key, 1))
    return (pool_lab_augmented_x,)


@app.cell(hide_code=True)
def _(pool_lab_augmented_x):
    _rows_aug_pool_lab = len(pool_lab_augmented_x)
    _columns_aug_pool_lab = pool_lab_augmented_x[0].shape[0]
    _figure_aug_pool_lab, _axes_aug_pool_lab = plt.subplots(
        _rows_aug_pool_lab,
        _columns_aug_pool_lab,
        figsize=(3 * _columns_aug_pool_lab, 3 * _rows_aug_pool_lab),
        squeeze=False,
    )
    for _batch_aug_pool_lab, _trajectory_aug_pool_lab in enumerate(pool_lab_augmented_x):
        for _time_aug_pool_lab in range(_columns_aug_pool_lab):
            _axes_aug_pool_lab[_batch_aug_pool_lab, _time_aug_pool_lab].imshow(
                np.clip(np.moveaxis(np.asarray(_trajectory_aug_pool_lab[_time_aug_pool_lab, :3]), 0, -1), 0, 1)
            )
            _axes_aug_pool_lab[_batch_aug_pool_lab, _time_aug_pool_lab].set(
                title=f"copy {_batch_aug_pool_lab}, slot {_time_aug_pool_lab}", xticks=[], yticks=[]
            )
    _figure_aug_pool_lab.suptitle("Start states x after one advance_pool call")
    _figure_aug_pool_lab.tight_layout()
    _figure_aug_pool_lab
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Pool admission decides whether the pool moves on

    After each optimisation step, the new rollout is only written back into
    the pool if its loss is close enough to both an exponential moving
    average (EMA) of earlier losses and the last admitted loss. A rejected
    rollout **does not undo the gradient step**; it only stops a bad rollout
    becoming the next starting point. This is set in
    `cfg.trainer.pool_admission` and applied by `PoolAdmissionController`
    in `NCA/trainer/runner.py`.

    Below, the same controller is run on a hand-written sequence of losses.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_loss_sequence = mo.ui.text(
        value="1.0, 0.9, 1.4, 0.8, 1.05, 0.7",
        label="Loss sequence (comma separated)",
        full_width=True,
    )
    pool_lab_relative_threshold = mo.ui.number(
        1.0, 4.0, value=1.25, step=0.05, label="pool_admission.relative_threshold"
    )
    pool_lab_previous_threshold = mo.ui.number(
        1.0, 4.0, value=1.10, step=0.05, label="pool_admission.previous_relative_threshold"
    )
    mo.vstack([
        pool_lab_loss_sequence,
        mo.hstack([pool_lab_relative_threshold, pool_lab_previous_threshold]),
    ])
    return (
        pool_lab_loss_sequence,
        pool_lab_previous_threshold,
        pool_lab_relative_threshold,
    )


@app.cell(hide_code=True)
def _(pool_lab_loss_sequence, pool_lab_previous_threshold, pool_lab_relative_threshold):
    try:
        _losses_admission_pool_lab = [
            float(_value_pool_lab.strip())
            for _value_pool_lab in pool_lab_loss_sequence.value.split(",")
            if _value_pool_lab.strip()
        ]
        if not _losses_admission_pool_lab or min(_losses_admission_pool_lab) < 0:
            raise ValueError("Use one or more non-negative losses.")
        _controller_pool_lab = PoolAdmissionController(
            PoolAdmissionConfig(
                enabled=True,
                relative_threshold=float(pool_lab_relative_threshold.value),
                previous_relative_threshold=float(pool_lab_previous_threshold.value),
                ema_decay=0.5,
                warmup=0,
            )
        )
        # The same two calls the training loop makes each iteration
        _decisions_pool_lab = []
        for _iteration_pool_lab, _loss_pool_lab in enumerate(_losses_admission_pool_lab):
            _decision_pool_lab = _controller_pool_lab.decide(_loss_pool_lab, _iteration_pool_lab)
            _controller_pool_lab.update(_decision_pool_lab, _loss_pool_lab)
            _decisions_pool_lab.append(_decision_pool_lab)

        _figure_admission_pool_lab, _axis_admission_pool_lab = plt.subplots(figsize=(8, 3))
        _steps_admission_pool_lab = np.arange(len(_losses_admission_pool_lab))
        _axis_admission_pool_lab.plot(
            _steps_admission_pool_lab, _losses_admission_pool_lab, "o-", label="rollout loss"
        )
        _axis_admission_pool_lab.plot(
            _steps_admission_pool_lab,
            [_decision.loss_reference for _decision in _decisions_pool_lab],
            "--",
            label="EMA reference",
        )
        _axis_admission_pool_lab.scatter(
            _steps_admission_pool_lab,
            _losses_admission_pool_lab,
            c=["tab:green" if _decision.admit else "tab:red" for _decision in _decisions_pool_lab],
            s=70,
            zorder=3,
        )
        _axis_admission_pool_lab.set(
            xlabel="iteration", ylabel="loss", title="green: admitted, red: rejected"
        )
        _axis_admission_pool_lab.legend()
        _axis_admission_pool_lab.grid(alpha=0.25)
        _figure_admission_pool_lab.tight_layout()
        _rows_pool_lab = [
            {
                "iteration": _iteration_pool_lab,
                "loss": round(_losses_admission_pool_lab[_iteration_pool_lab], 3),
                "admit": _decision_pool_lab.admit,
                "EMA ratio": round(_decision_pool_lab.loss_ratio, 3),
                "previous ratio": round(_decision_pool_lab.previous_loss_ratio, 3),
            }
            for _iteration_pool_lab, _decision_pool_lab in enumerate(_decisions_pool_lab)
        ]
        _admission_output_pool_lab = mo.vstack([
            _figure_admission_pool_lab,
            mo.ui.table(_rows_pool_lab, selection=None),
        ])
    except ValueError as _error_admission_pool_lab:
        _admission_output_pool_lab = mo.callout(str(_error_admission_pool_lab), kind="danger")
    _admission_output_pool_lab
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. A short training run

    This puts the objective from section 2 and an admission policy into an
    `ExperimentConfig` and trains with `run_training`, as the entrypoints do.
    Logging and model publishing are switched off, and the checkpoint goes to
    a temporary directory. It is a quick check of how the settings behave, not
    a measure of quality. The plot shows the total loss and each regulariser's
    contribution, taken from the `metrics` the training loop passes to
    `progress_callback`.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_enable_admission = mo.ui.switch(value=True, label="trainer.pool_admission.enabled")
    pool_lab_iterations = mo.ui.dropdown(
        options={
            "10 iterations": 10,
            "25 iterations": 25,
            "50 iterations": 50,
            "100 iterations": 100,
            "1000 iterations": 1000,
        },
        value="25 iterations",
        label="run.iterations",
    )
    pool_lab_run_button = mo.ui.run_button(label="Run short training")
    mo.vstack([pool_lab_enable_admission, pool_lab_iterations, pool_lab_run_button])
    return pool_lab_enable_admission, pool_lab_iterations, pool_lab_run_button


@app.cell
def _(
    pool_lab_data_config,
    pool_lab_enable_admission,
    pool_lab_iterations,
    pool_lab_loss_config,
    pool_lab_seed,
):
    pool_lab_experiment_config = ExperimentConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        seed=int(pool_lab_seed.value),
        experiment=ExperimentMetadataConfig(name="notebook_objectives_and_pool"),
        system=SystemConfig(precision="highest"),
        data=pool_lab_data_config,
        model=ModelConfig(
            family="NCA",
            channels=16,
            kernel_str=("ID", "LAP", "GRAD"),
            fire_rate=0.5,
            padding="CIRCULAR",
        ),
        run=RunConfig(
            t=32,
            iterations=int(pool_lab_iterations.value),
            checkpoint_warmup=0,
            write_images=False,
            write_videos=False,
        ),
        trainer=TrainerConfig(
            loop_autodiff="lax",
            log_every=max(1, int(pool_lab_iterations.value) // 2),
            pool_admission=PoolAdmissionConfig(
                enabled=bool(pool_lab_enable_admission.value),
                relative_threshold=1.05,
                previous_relative_threshold=1.05,
                ema_decay=0.95,
                warmup=10,
            ),
        ),
        optimiser=OptimiserConfig(
            learn_rate=0.001,
            warmup_steps=0,
            schedule=ScheduleConfig(type="cosine", final_factor=0.2),
        ),
        loss=pool_lab_loss_config,
        logging=LoggingConfig(
            backend="none",
            wandb=WandbConfig(project="NCA-notebook", group="objectives-and-pool"),
        ),
        model_store=ModelStoreConfig(enabled=False),
    )
    return (pool_lab_experiment_config,)


@app.cell
def _(pool_lab_data, pool_lab_experiment_config, pool_lab_run_button):
    mo.stop(
        not pool_lab_run_button.value or pool_lab_data is None,
        mo.callout("Choose the settings above, then press the button.", kind="info"),
    )
    jax.config.update(
        "jax_default_matmul_precision", pool_lab_experiment_config.system.precision
    )
    _model_key_pool_lab, _train_key_pool_lab = jax.random.split(
        jax.random.PRNGKey(pool_lab_experiment_config.seed)
    )
    _model_pool_lab, _ = build_model(pool_lab_experiment_config.model, key=_model_key_pool_lab)
    _augmenter_pool_lab, _ = build_data_augmenter(
        pool_lab_experiment_config.data, pool_lab_data, _model_pool_lab.N_CHANNELS
    )
    _context_pool_lab = TrainerContext(
        run_name="objectives_and_pool",
        model_directory=tempfile.mkdtemp(prefix="nca-demo-"),
        data_augmenter=_augmenter_pool_lab,
        observed_channels=pool_lab_experiment_config.data.emoji.observed_channels,
        data_channels=pool_lab_experiment_config.data.emoji.data_channels,
    )

    _regularisers_pool_lab = tuple(pool_lab_experiment_config.loss.regularisers)
    _history_pool_lab = {"loss": [], "admit": [], **{_name: [] for _name in _regularisers_pool_lab}}

    def _plot_history_pool_lab(title):
        _figure_pool_lab, _axis_pool_lab = plt.subplots(figsize=(8, 3))
        _floor_pool_lab = np.finfo(float).tiny
        for _name_pool_lab in ("loss", *_regularisers_pool_lab):
            _axis_pool_lab.plot(
                np.maximum(_history_pool_lab[_name_pool_lab], _floor_pool_lab),
                label="total loss" if _name_pool_lab == "loss" else _name_pool_lab,
            )
        _axis_pool_lab.set(xlabel="iteration", ylabel="loss", yscale="log", title=title)
        _axis_pool_lab.grid(alpha=0.25)
        _axis_pool_lab.legend()
        _figure_pool_lab.tight_layout()
        return _figure_pool_lab

    def _record_pool_lab(iteration, loss, metrics):
        _history_pool_lab["loss"].append(float(loss))
        _history_pool_lab["admit"].append(int(metrics["pool/admit"]))
        for _name_pool_lab in _regularisers_pool_lab:
            _history_pool_lab[_name_pool_lab].append(float(metrics[_name_pool_lab]))
        _live_figure_pool_lab = _plot_history_pool_lab(f"iteration {iteration + 1}")
        mo.output.replace(_live_figure_pool_lab)
        plt.close(_live_figure_pool_lab)

    _result_pool_lab = run_training(
        pool_lab_experiment_config,
        model=_model_pool_lab,
        data=pool_lab_data,
        context=_context_pool_lab,
        key=_train_key_pool_lab,
        progress_callback=_record_pool_lab,
    )
    _admitted_pool_lab = sum(_history_pool_lab["admit"])
    mo.vstack([
        _plot_history_pool_lab("training loss"),
        mo.md(
            f"Best loss: `{_result_pool_lab.best_loss:.4g}`. Rollouts admitted to the pool: "
            f"`{_admitted_pool_lab}`; rejected: `{len(_history_pool_lab['admit']) - _admitted_pool_lab}`."
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    - Loss **terms** say *what* the observed pattern should match.
    - **Regularisers** say *how* the NCA should get there, and are applied at
      every step.
    - The **augmenter** builds each new pool from the NCA's own states mixed
      with the true images; it does not start from scratch each step.
    - **Pool admission** guards what goes back into the pool, not the
      gradient update.

    For a real experiment, start from `Experiments/emoji/conf/base_config.yaml`,
    change one of these settings at a time in a sweep file, and run it on the
    cluster (see the first walkthrough and the README).
    """)
    return


if __name__ == "__main__":
    app.run()
