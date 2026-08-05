# /// script
# dependencies = ["marimo", "jax", "matplotlib", "numpy"]
# ///

"""A compact laboratory for NCA objectives, regularisers, and pool dynamics.

Run from the repository root with:

    marimo edit demo/nca_objectives_and_pool_dynamics.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import json
    import os
    import sys
    from pathlib import Path

    _repository_root = Path(__file__).resolve().parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))

    import jax
    import jax.numpy as jnp
    import marimo as mo
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
    from Experiments.emoji.config import EmojiDataConfig, ProbabilityScheduleConfig
    from Experiments.emoji.config_helpers import build_data_augmenter, load_data
    from NCA.model.config import ModelConfig
    from NCA.trainer.config import PoolAdmissionConfig, TrainerConfig
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.data_augmenter import reinject_observations
    from NCA.trainer.objective import resolve_objective
    from NCA.trainer.pool import PoolAdmissionController
    from NCA.trainer.trainer import build_trainer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA objectives and pool dynamics

    This is a follow-on to `nca_training_config_walkthrough.py`. It keeps the
    model and data deliberately small, and focuses on what changes the
    optimisation problem and the recurrent training pool.

    The notebook uses the active typed-config trainer, rather than a tutorial
    reimplementation. Training is optional and intentionally short.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. A fixed, cheap baseline

    Select two small emoji frames. Unlike the first notebook, there is no data
    exploration here: keeping this task fixed makes comparisons meaningful.
    """)
    return


@app.cell(hide_code=True)
def _():
    _repo_root_pool_lab = Path(__file__).resolve().parents[1]
    pool_lab_data_path = mo.ui.text(
        value=os.environ.get(
            "DATA_PATH_BASE", str(_repo_root_pool_lab / "demo" / "demo_data")
        ),
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
    pool_lab_data_config = DataConfig(
        dataset="emojis",
        batches=2,
        preprocessing=PreprocessingConfig(downsample=4),
        augmentation=EmojiDataConfig(
            task="sequence",
            sequence=(pool_lab_source_name.value.strip(), pool_lab_target_name.value.strip()),
            target_repeats=1,
            pad=(4, 4, 4, 4),
            shift_amount=0,
            noise_strength=0.0,
            regenerate=False,
            terminal_carry=ProbabilityScheduleConfig(),
            regeneration=ProbabilityScheduleConfig(),
        ),
    )
    return (pool_lab_data_config,)


@app.cell
def _(pool_lab_data_config, pool_lab_data_path):
    pool_lab_data = None
    pool_lab_data_error = None
    try:
        _emoji_root_load_pool_lab = Path(pool_lab_data_path.value).expanduser() / "Emojis"
        pool_lab_data, _pool_lab_data_name = load_data(
            pool_lab_data_config, impath=str(_emoji_root_load_pool_lab)
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
        _axes_pool_lab = np.atleast_1d(_axes_pool_lab)
        for _time_pool_lab, _axis_pool_lab in enumerate(_axes_pool_lab):
            _axis_pool_lab.imshow(np.clip(np.moveaxis(_frames_pool_lab[_time_pool_lab, :3], 0, -1), 0, 1))
            if _time_pool_lab==0:
                _axis_pool_lab.set(title=f"initial condition", xticks=[], yticks=[])
            else:
                _axis_pool_lab.set(title=f"target time {_time_pool_lab}", xticks=[], yticks=[])
        _figure_pool_lab.tight_layout()
        _data_preview_pool_lab = _figure_pool_lab
    _data_preview_pool_lab
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Loss terms and regularisers

    For a rollout of length \(T\), the trainer minimises

    \[
    \mathcal L = \operatorname{mean}(\mathcal L_\mathrm{data}) +
    \sum_r \lambda_r\,\frac{1}{T}\operatorname{mean}
    \left(\sum_{t=1}^{T} r_t\right).
    \]

    Loss-term weights are *relative*: the trainer forms their weighted average.
    Regulariser coefficients \(\lambda_r\) are additive. Each active
    regulariser is evaluated at every NCA update, not just at the endpoint.
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
        label="Data objective",
    )
    pool_lab_regulariser_name = mo.ui.dropdown(
        options={
            "None": "none",
            "Keep states in [0, 1]": "intermediate_state",
            "Keep hidden channels small": "hidden_state_size",
            "Encourage contiguous observable growth": "contiguous_growth",
            "Localise hidden activity": "localised_hidden",
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
def _(
    pool_lab_loss_mode,
    pool_lab_regulariser_name,
    pool_lab_regulariser_weight,
):
    if pool_lab_loss_mode.value == "l2_spectral":
        _terms_pool_lab = (
            PointwiseLossConfig(type="l2", weight=1.0),
            PointwiseLossConfig(type="spectral", weight=0.25),
        )
    else:
        _terms_pool_lab = (PointwiseLossConfig(type=pool_lab_loss_mode.value),)
    pool_lab_regularisers = (
        {}
        if pool_lab_regulariser_name.value == "none"
        else {pool_lab_regulariser_name.value: float(pool_lab_regulariser_weight.value)}
    )
    pool_lab_loss_config = LossConfig(
        terms=_terms_pool_lab, regularisers=pool_lab_regularisers
    )
    pool_lab_objective = resolve_objective(pool_lab_loss_config)
    return pool_lab_loss_config, pool_lab_objective


@app.cell(hide_code=True)
def _(pool_lab_loss_config, pool_lab_objective):
    mo.vstack([
        mo.md("### Resolved active objective"),
        mo.md(f"```json\n{json.dumps(config_to_dict(pool_lab_loss_config), indent=2)}\n```"),
        mo.md(
            f"Loss functions: `{pool_lab_objective.names}`  \\n+"
            f"Relative weights: `{pool_lab_objective.arguments['component_weights']}`  \\n+"
            f"Regulariser coefficients: `{pool_lab_objective.regulariser_coefficients}`"
        ),
        mo.callout(
            "`l2` and `spectral` are inexpensive, compatible pointwise terms. "
            "More specialised terms can require shared auxiliary arguments; use them "
            "only when those arguments agree.",
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Reinjection is a concrete pool transformation

    The default basic augmenter does not replace a bad pool with fresh starts.
    It propagates each trajectory by one slot, restores the first observation,
    reinjects observed channels at a fraction of later slots, retains hidden
    channels, then adds small noise. The following is a direct call to the same
    `reinject_observations` primitive used by that augmenter.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_reinjection_fraction = mo.ui.slider(
        0.0, 1.0, value=0.5, step=0.25, label="Observable reinjection fraction"
    )
    pool_lab_reinjection_fraction
    return (pool_lab_reinjection_fraction,)


@app.cell(hide_code=True)
def _(pool_lab_reinjection_fraction):
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
        _axes_reinject_pool_lab[0, _time_reinject_pool_lab].imshow(
            np.asarray(_pool_after_pool_lab[0, _time_reinject_pool_lab, 0]), vmin=0, vmax=1
        )
        _axes_reinject_pool_lab[0, _time_reinject_pool_lab].set(
            title=f"observed, slot {_time_reinject_pool_lab}", xticks=[], yticks=[]
        )
        _axes_reinject_pool_lab[1, _time_reinject_pool_lab].imshow(
            np.asarray(_pool_after_pool_lab[0, _time_reinject_pool_lab, 2]), vmin=0, vmax=1
        )
        _axes_reinject_pool_lab[1, _time_reinject_pool_lab].set(
            title=f"hidden, slot {_time_reinject_pool_lab}", xticks=[], yticks=[]
        )
    _figure_reinject_pool_lab.suptitle("Pool after propagation and reinjection")
    _figure_reinject_pool_lab.tight_layout()
    mo.vstack([
        mo.md(
            "Here observed channels are deliberately `0`, `0.5`, and `1` in the source "
            "trajectory; hidden channels begin at `0.8`. Re-run with different fractions "
            "to see that reinjection changes only observed channels."
        ),
        _figure_reinject_pool_lab,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Admission decides whether the pool advances

    After an optimisation step, a candidate rollout is admitted only if its
    scalar loss is sufficiently close to both the exponential moving average
    (EMA) and the previous admitted loss. Rejection **does not undo the model
    update**; it prevents that rollout becoming the next recurrent pool state.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_loss_sequence = mo.ui.text(
        value="1.0, 0.9, 1.4, 0.8, 1.05, 0.7",
        label="Illustrative loss sequence (comma separated)",
        full_width=True,
    )
    pool_lab_relative_threshold = mo.ui.number(
        1.0, 4.0, value=1.25, step=0.05, label="EMA relative threshold"
    )
    pool_lab_previous_threshold = mo.ui.number(
        1.0, 4.0, value=1.10, step=0.05, label="Previous-loss threshold"
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
def _(
    pool_lab_loss_sequence,
    pool_lab_previous_threshold,
    pool_lab_relative_threshold,
):
    try:
        _losses_admission_pool_lab = [
            float(_value_pool_lab.strip())
            for _value_pool_lab in pool_lab_loss_sequence.value.split(",")
            if _value_pool_lab.strip()
        ]
        if not _losses_admission_pool_lab or min(_losses_admission_pool_lab) < 0:
            raise ValueError("Use one or more non-negative losses.")
        _admission_config_pool_lab = PoolAdmissionConfig(
            enabled=True,
            relative_threshold=float(pool_lab_relative_threshold.value),
            previous_relative_threshold=float(pool_lab_previous_threshold.value),
            ema_decay=0.5,
            warmup=0,
        )
        _controller_pool_lab = PoolAdmissionController(_admission_config_pool_lab, default_warmup=0)
        _decisions_pool_lab = []
        for _iteration_pool_lab, _loss_pool_lab in enumerate(_losses_admission_pool_lab):
            _decision_pool_lab = _controller_pool_lab.decide(_loss_pool_lab, _iteration_pool_lab)
            _decisions_pool_lab.append(_decision_pool_lab)
            _controller_pool_lab.update(_decision_pool_lab, _loss_pool_lab)
        _figure_admission_pool_lab, _axis_admission_pool_lab = plt.subplots(figsize=(8, 3))
        _steps_admission_pool_lab = np.arange(len(_losses_admission_pool_lab))
        _axis_admission_pool_lab.plot(_steps_admission_pool_lab, _losses_admission_pool_lab, "o-", label="candidate loss")
        _axis_admission_pool_lab.plot(
            _steps_admission_pool_lab,
            [_decision.loss_reference for _decision in _decisions_pool_lab],
            "--", label="EMA reference",
        )
        for _step_pool_lab, _decision_pool_lab in enumerate(_decisions_pool_lab):
            _axis_admission_pool_lab.scatter(
                _step_pool_lab,
                _losses_admission_pool_lab[_step_pool_lab],
                color="tab:green" if _decision_pool_lab.admit else "tab:red",
                s=70,
                zorder=3,
            )
        _axis_admission_pool_lab.set(xlabel="iteration", ylabel="loss")
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
    ## 5. Short integrated run

    This final experiment uses the selected objective and admission policy.
    It is a diagnostic run, not a quality benchmark. The callback records data
    plus regularisation loss, and the admission metrics produced by the active
    trainer.
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_lab_enable_admission = mo.ui.switch(value=True, label="Enable pool admission")
    pool_lab_iterations = mo.ui.dropdown(
        options={"10 iterations": 10, "25 iterations": 25, "50 iterations": 50, "100 iterations": 100, "1000 iterations": 1000},
        value="25 iterations",
        label="Iterations",
    )
    pool_lab_run_button = mo.ui.run_button(label="Run short diagnostic training")
    mo.vstack([pool_lab_enable_admission, pool_lab_iterations, pool_lab_run_button])
    return pool_lab_enable_admission, pool_lab_iterations, pool_lab_run_button


@app.cell
def _(
    pool_lab_enable_admission,
    pool_lab_iterations,
    pool_lab_loss_config,
    pool_lab_seed,
):
    pool_lab_model_config = ModelConfig(
        family="NCA",
        channels=16,
        activation="relu",
        kernel_str=("ID", "LAP", "GRAD"),
        fire_rate=0.5,
        padding="CIRCULAR",
    )
    _model_key_pool_lab = jax.random.PRNGKey(int(pool_lab_seed.value))
    pool_lab_model, _pool_lab_model_name = build_model(pool_lab_model_config, key=_model_key_pool_lab)
    pool_lab_train_key = jax.random.fold_in(_model_key_pool_lab, 1)
    pool_lab_training_config = TrainingConfig(
        loop=TrainingLoopConfig(t=32, iterations=int(pool_lab_iterations.value), write_images=False),
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
        optimizer=OptimizerConfig(
            learn_rate=0.001,
            warmup_steps=0,
            schedule=ScheduleConfig(type="cosine", final_factor=0.2),
        ),
        loss=pool_lab_loss_config,
        checkpoint=CheckpointConfig(warmup=0),
    )
    return (
        pool_lab_model,
        pool_lab_model_config,
        pool_lab_train_key,
        pool_lab_training_config,
    )


@app.cell(hide_code=True)
def _(
    pool_lab_data,
    pool_lab_data_config,
    pool_lab_data_error,
    pool_lab_model,
    pool_lab_model_config,
    pool_lab_run_button,
    pool_lab_train_key,
    pool_lab_training_config,
):
    pool_lab_training_output = None
    if pool_lab_run_button.value:
        if pool_lab_data_error is not None:
            pool_lab_training_output = mo.callout(pool_lab_data_error, kind="danger")
        else:
            try:
                _augmenter_class_pool_lab, _augmenter_name_pool_lab = build_data_augmenter(pool_lab_data_config)
                _experiment_pool_lab = ExperimentConfig(
                    schema_version=1,
                    seed=0,
                    experiment=ExperimentMetadataConfig(name="notebook_nca_objectives_pool"),
                    runtime=RuntimeConfig(precision="highest"),
                    data=pool_lab_data_config,
                    model=pool_lab_model_config,
                    training=pool_lab_training_config,
                    logging=LoggingConfig(
                        backend="none",
                        wandb=WandbConfig(
                            project="NCA-notebook",
                            group="objectives-and-pool-lab",
                        ),
                    ),
                    model_store=ModelStoreConfig(
                        enabled=False,
                        root=str(Path("models") / "local" / "notebook-objectives"),
                        collection="NCA-notebook",
                    ),
                )
                _context_pool_lab = TrainerContext(
                    run_name="objectives_pool_diagnostic",
                    model_directory=_experiment_pool_lab.model_store.root,
                    data_augmenter=_augmenter_class_pool_lab,
                    observed_channels=_experiment_pool_lab.data.emoji.observed_channels,
                    data_channels=_experiment_pool_lab.data.emoji.data_channels,
                )
                _trainer_pool_lab = build_trainer(
                    _experiment_pool_lab, pool_lab_model, pool_lab_data, _context_pool_lab
                )
                _history_pool_lab = {"loss": [], "admit": [], "reject": []}
                for _regulariser_pool_lab in _experiment_pool_lab.training.loss.regularisers:
                    _history_pool_lab[_regulariser_pool_lab] = []

                def _make_training_plot_pool_lab(iteration=None):
                    _figure_pool_lab, _axis_pool_lab = plt.subplots(figsize=(8, 3))
                    _plot_floor_pool_lab = np.finfo(float).tiny
                    _axis_pool_lab.plot(
                        np.maximum(_history_pool_lab["loss"], _plot_floor_pool_lab),
                        label="total loss",
                    )
                    for _regulariser_pool_lab in _experiment_pool_lab.training.loss.regularisers:
                        _axis_pool_lab.plot(
                            np.maximum(
                                _history_pool_lab[_regulariser_pool_lab],
                                _plot_floor_pool_lab,
                            ),
                            label=_regulariser_pool_lab,
                        )
                    _title_pool_lab = "Diagnostic training loss"
                    if iteration is not None:
                        _title_pool_lab += f" — iteration {iteration + 1}"
                    _axis_pool_lab.set(
                        xlabel="iteration",
                        ylabel="loss",
                        yscale="log",
                        title=_title_pool_lab,
                    )
                    _axis_pool_lab.grid(alpha=0.25)
                    _axis_pool_lab.legend()
                    _figure_pool_lab.tight_layout()
                    return _figure_pool_lab

                def _record_pool_lab(_iteration_pool_lab, _loss_pool_lab, _metrics_pool_lab):
                    _history_pool_lab["loss"].append(float(_loss_pool_lab))
                    _history_pool_lab["admit"].append(float(_metrics_pool_lab["pool/admit"]))
                    _history_pool_lab["reject"].append(float(_metrics_pool_lab["pool/reject"]))
                    for _regulariser_pool_lab in _experiment_pool_lab.training.loss.regularisers:
                        _history_pool_lab[_regulariser_pool_lab].append(
                            float(_metrics_pool_lab[_regulariser_pool_lab])
                        )
                    _live_figure_pool_lab = _make_training_plot_pool_lab(
                        _iteration_pool_lab
                    )
                    mo.output.replace(_live_figure_pool_lab)
                    plt.close(_live_figure_pool_lab)

                os.environ["WANDB_MODE"] = "offline"
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                set_matmul_precision(_experiment_pool_lab.runtime)
                _result_pool_lab = _trainer_pool_lab.train(
                    key=pool_lab_train_key, progress_callback=_record_pool_lab
                )
                _figure_training_pool_lab = _make_training_plot_pool_lab()
                _admitted_pool_lab = int(sum(_history_pool_lab["admit"]))
                _rejected_pool_lab = int(sum(_history_pool_lab["reject"]))
                pool_lab_training_output = mo.vstack([
                    _figure_training_pool_lab,
                    mo.md(
                        f"Best loss: `{_result_pool_lab.best_loss:.4g}`. "
                        f"Pool updates admitted: `{_admitted_pool_lab}`; rejected: `{_rejected_pool_lab}`."
                    ),
                ])
            except Exception as _training_error_pool_lab:
                pool_lab_training_output = mo.callout(str(_training_error_pool_lab), kind="danger")
    if pool_lab_training_output is None:
        pool_lab_training_output = mo.callout(
            "Configure the objective, then press the button to run the short diagnostic.",
            kind="info",
        )
    pool_lab_training_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Takeaways

    - Use loss terms to specify *what the visible pattern should match*.
    - Use regularisers to specify *how the NCA should realise that match*.
    - Reinjection preserves a mixture of predicted recurrent state and known
      observations; it is not a full reset.
    - Admission is a guard on the pool transition, not on the optimiser update.

    For a real experiment, start from the first walkthrough’s full
    configuration workflow, increase the budget remotely, and change one of
    these mechanisms at a time.
    """)
    return


if __name__ == "__main__":
    app.run()
