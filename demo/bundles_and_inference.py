# /// script
# dependencies = ["marimo", "jax", "equinox", "matplotlib", "numpy", "pandas", "pyyaml", "omegaconf"]
# ///

"""Tutorial: find trained models in the model store, load them, and run them.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/bundles_and_inference.py

Uses the shared model store (MODEL_STORE_ROOT) if set, else the local store
that demo/nca_training_config_walkthrough.py trains into. Running a model
here works for emoji bundles; other tasks point to their own notebooks.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from omegaconf import OmegaConf

    from Experiments.emoji.config_helpers import build_data_augmenter, load_data
    from Experiments.model_registry import (
        ModelRegistry,
        load_selection,
        selection_document,
        verify_evaluation_input,
    )
    from NCA.trainer.intervention import (
        rollout_model,
        rollout_model_with_blocked_channel,
    )

    REPO_ROOT = Path(__file__).resolve().parents[1]

    def ticked_rows(table):
        """The rows ticked in a ``mo.ui.table``, as a list of dicts."""
        value = table.value
        if value is None:
            return []
        if hasattr(value, "to_dict"):  # a pandas dataframe
            return value.to_dict("records")
        return list(value)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Trained models: bundles, the registry and inference

    - Every training run saves its best model as a **bundle**: `model.eqx` (weights), `config.yaml` (the full config), `manifest.yaml` (checksums, git commit, data fingerprint, results)
    - Bundles live in the **model store** (`MODEL_STORE_ROOT`) under `bundles/<collection>/<experiment>/`, and are never edited
    - The **registry** (`Experiments/model_registry.py`) indexes the store into `registry.sqlite` and returns tables (pandas dataframes)
    - This notebook: search the store, save a **selection** file, load a bundle, run it, and knock a channel out
    - No models yet? Run `demo/nca_training_config_walkthrough.py` first; it trains one into the local store
    - Reference: `docs/model_registry.md`
    """)
    return


@app.cell
def _():
    _default_store = os.environ.get(
        "MODEL_STORE_ROOT", str(REPO_ROOT / "models" / "local" / "notebook")
    )
    store_root = mo.ui.text(value=_default_store, label="Model store", full_width=True)
    reindex_button = mo.ui.run_button(label="Rebuild the index (after copying new bundles)")
    mo.vstack([store_root, reindex_button])
    return reindex_button, store_root


@app.cell
def _(reindex_button, store_root):
    _root = Path(store_root.value).expanduser()
    mo.stop(
        not (_root / "bundles").is_dir(),
        mo.callout(f"No `bundles/` folder in `{_root}`. Train a model first, or set the store path.", kind="warn"),
    )
    registry = ModelRegistry(_root)
    if reindex_button.value:
        registry.reindex()  # the same as: python -m Experiments.model_registry --root <store> reindex
    models = registry.models_df()
    mo.md(f"`{len(models)}` bundles in the index. Columns: " + ", ".join(f"`{_c}`" for _c in models.columns))
    return models, registry


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Search

    - `models` is a pandas dataframe; filter it with `models.query(...)`, e.g. `family == 'gNCA' and dataset == 'micropatterns_260726'`
    - Every setting that differed within a sweep is also a W&B tag (`registry.wandb_tags_df()`), e.g. `model.channels:48`
    - `Experiments/model_registry_explorer.py` is the full search tool; this is the same idea in a few lines
    """)
    return


@app.cell
def _():
    query = mo.ui.text(value="status == 'complete'", label="Filter (pandas query)", full_width=True)
    query
    return (query,)


@app.cell
def _(models, query):
    try:
        found = models.query(query.value) if query.value.strip() else models
        _message = mo.md(f"`{len(found)}` models match. Tick some to select them.")
    except Exception as _error:
        found = models.iloc[0:0]
        _message = mo.callout(f"Filter error: {_error}", kind="danger")
    _columns = ["model_id", "experiment", "family", "dataset", "task", "best_loss", "created_at"]
    model_table = mo.ui.table(found[_columns].to_dict("records"), selection="multi", page_size=8)
    mo.vstack([_message, model_table])
    return (model_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Save a selection

    - A **selection** is a YAML list of model IDs; evaluation and figure notebooks read one to know which models to use
    - Keep the selections behind a result with the project (e.g. `Experiments/micropatterns/figures/selections/`, in git), so the figure can be traced back to its models
    - `selection_document(records)` makes the file; `load_selection(path)` reads it back
    """)
    return


@app.cell
def _():
    selection_path = mo.ui.text(
        value=str(REPO_ROOT / "output" / "tutorial_selection.yaml"),
        label="Selection file",
        full_width=True,
    )
    save_button = mo.ui.run_button(label="Save the ticked models")
    mo.vstack([selection_path, save_button])
    return save_button, selection_path


@app.cell
def _(model_table, save_button, selection_path):
    mo.stop(not save_button.value, mo.md("Tick models above, then save."))
    _records = [
        {"model_id": _row["model_id"], "experiment": _row["experiment"]}
        for _row in ticked_rows(model_table)
    ]
    mo.stop(not _records, mo.callout("No models ticked.", kind="warn"))
    _path = Path(selection_path.value).expanduser()
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(yaml.safe_dump(selection_document(_records), sort_keys=False))
    _loaded = load_selection(_path)
    mo.md(
        f"Saved `{_path}`; read back {len(_loaded)} models: "
        + ", ".join(f"`{_record['model_id']}`" for _record in _loaded)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Load a bundle

    - `registry.get(model_id)` opens a bundle; `bundle.config` is the full typed config it was trained with
    - `bundle.load_model(key=...)` rebuilds the model from that config and checks the weights' checksum
    - Choose one model below (the first ticked one is used)
    """)
    return


@app.cell
def _(model_table, registry):
    _ticked = ticked_rows(model_table)
    mo.stop(not _ticked, mo.md("Tick a model in the table above."))
    bundle = registry.get(_ticked[0]["model_id"])
    model = bundle.load_model(key=jax.random.PRNGKey(0))
    _cfg = bundle.config
    mo.md(
        f"""
    - Bundle `{bundle.id}`
    - Folder: `{bundle.path}`
    - Data: `{_cfg.data.dataset}`; model: `{_cfg.model.family}` with {_cfg.model.channels} channels, kernels `{list(_cfg.model.kernel_str)}`
    - Trained for {_cfg.run.iterations} iterations, `run.t = {_cfg.run.t}` steps between snapshots
    - Code: git commit `{OmegaConf.select(bundle.manifest, "provenance.git.commit") or "not recorded"}`
    """
    )
    return bundle, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Run it from its own initial condition

    - Bundles don't contain data; reload it with the task's `load_data(bundle.config.data)`
    - `verify_evaluation_input` checks it against the fingerprint saved at training time; it fails if the data or its cleaning changed
    - The task's augmenter adds the hidden channels, exactly as in training
    - This part is written for **emoji** bundles. Micropattern models need colony masks and cleaning; use `Experiments/micropatterns/nodal_ko_eval.py` or `figures/data.py`
    """)
    return


@app.cell
def _(bundle, model):
    mo.stop(
        bundle.config.data.dataset != "emojis",
        mo.callout(
            f"This bundle uses `{bundle.config.data.dataset}` data; pick an emoji model to run it here.",
            kind="info",
        ),
    )
    _data_root = Path(os.environ.get("DATA_PATH_BASE", REPO_ROOT / "demo" / "demo_data"))
    _data, _ = load_data(bundle.config.data, impath=str(_data_root / "Emojis"))
    verify_evaluation_input(_data, bundle.manifest.evaluation_input)
    _augmenter, _ = build_data_augmenter(
        bundle.config.data, bundle.config.run.iterations, _data, model.N_CHANNELS
    )
    # [trajectories][time] -> the first trajectory's initial state [channels, H, W]
    initial_state = jnp.asarray(_augmenter.return_saved_data()[0][0])
    observed_channels = int(bundle.config.data.emoji.observed_channels)
    mo.md(f"Initial state `{initial_state.shape}`; the first {observed_channels} channels are observed (RGBA).")
    return initial_state, observed_channels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Knock a channel out

    - A **knockout** hides one channel from the model's perception from a chosen step on; the channel itself is still updated
    - For micropattern models this is how Nodal inhibition is imitated (channel `NODAL`, from 0 h or 24 h)
    - `rollout_model_with_blocked_channel` does it; compare with the normal `rollout_model`
    - Hidden channels are the interesting ones to knock out: they carry the model's internal signals
    """)
    return


@app.cell
def _(model, observed_channels):
    total_steps = mo.ui.slider(16, 256, step=16, value=128, label="Steps")
    blocked_channel = mo.ui.dropdown(
        options={
            (f"{_c} (observed)" if _c < observed_channels else f"{_c} (hidden)"): _c
            for _c in range(model.N_CHANNELS)
        },
        value=f"{observed_channels} (hidden)",
        label="Channel to knock out",
    )
    knockout_step = mo.ui.slider(0, 128, step=8, value=32, label="Knock out from step")
    mo.hstack([total_steps, blocked_channel, knockout_step])
    return blocked_channel, knockout_step, total_steps


@app.cell
def _(blocked_channel, initial_state, knockout_step, model, total_steps):
    _key = jax.random.PRNGKey(0)

    def _no_boundary(state):
        return state

    normal = np.asarray(rollout_model(model, initial_state, _no_boundary, _key, int(total_steps.value)))
    knocked_out = np.asarray(
        rollout_model_with_blocked_channel(
            model,
            initial_state,
            _no_boundary,
            _key,
            int(total_steps.value),
            int(blocked_channel.value),
            int(knockout_step.value),
        )
    )
    return knocked_out, normal


@app.cell
def _(knocked_out, normal):
    _frames = np.linspace(0, len(normal) - 1, 5).astype(int)
    _fig, _axes = plt.subplots(2, len(_frames), figsize=(2 * len(_frames), 4.4))

    def _rgb(state):
        return np.clip(np.moveaxis(state[:4], 0, -1), 0, 1)

    for _column, _frame in enumerate(_frames):
        _axes[0, _column].imshow(_rgb(normal[_frame]))
        _axes[0, _column].set_title(f"step {_frame}")
        _axes[1, _column].imshow(_rgb(knocked_out[_frame]))
    _axes[0, 0].set_ylabel("normal")
    _axes[1, 0].set_ylabel("knockout")
    for _axis in _axes.flat:
        _axis.set_xticks([])
        _axis.set_yticks([])
    _fig.tight_layout()
    _difference = float(np.abs(normal[-1, :4] - knocked_out[-1, :4]).mean())
    mo.vstack([_fig, mo.md(f"Mean difference of the observed channels at the end: `{_difference:.4f}`")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where next

    - Search and compare many models: `marimo edit Experiments/model_registry_explorer.py`
    - Micropattern knockout evaluation: `Experiments/micropatterns/nodal_ko_eval.py` (`docs/workflow_micropatterns.md`, step 8)
    - Record an evaluation result next to a model: `record_evaluation` (`docs/model_registry.md`)
    - From the command line: `python -m Experiments.model_registry list` and `... show <model-id>`
    """)
    return


if __name__ == "__main__":
    app.run()
