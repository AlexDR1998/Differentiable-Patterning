# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
#   "jax",
#   "equinox",
#   "omegaconf",
# ]
# ///

"""A read-only marimo explorer for a local model registry.

Run from the repository root with:

    marimo run Experiments/model_registry_explorer.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sqlite3
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, mo, os, pd, sqlite3


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Model registry explorer

    Search the read-only SQLite catalogue, then select a model to inspect its
    indexed metadata. Rebuild the index separately with
    `python -m Experiments.model_registry reindex` when needed.
    """)
    return


@app.cell
def _(Path, mo, os):
    _repository_root = Path(__file__).resolve().parents[1]
    _default_store = os.environ.get("MODEL_STORE_ROOT", str(_repository_root / "models"))
    store_root = mo.ui.text(_default_store, label="Model store")
    store_root
    return (store_root,)


@app.cell
def _(Path, pd, sqlite3, store_root):
    _store_path = Path(store_root.value).expanduser()
    database_path = _store_path / "registry.sqlite"
    if not database_path.is_file():
        database_error = (
            f"No registry.sqlite found at {database_path}. Build it with "
            f"`python -m Experiments.model_registry --root {_store_path} reindex`."
        )
    else:
        try:
            with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
                pd.read_sql_query("SELECT 1 FROM models LIMIT 1", _connection)
            database_error = None
        except sqlite3.Error as _error:
            database_error = f"Could not read {database_path}: {_error}"
    return database_error, database_path


@app.cell(hide_code=True)
def _(mo):
    search_text = mo.ui.text(
        placeholder="Model ID, slug, collection, experiment, alias, tag, or dataset",
        label="Search",
        full_width=True,
    )
    family_filter = mo.ui.text(placeholder="e.g. NCA", label="Family")
    dataset_filter = mo.ui.text(placeholder="e.g. emojis", label="Dataset")
    mo.hstack([search_text, family_filter, dataset_filter], widths=[3, 1, 1])
    return dataset_filter, family_filter, search_text


@app.cell(hide_code=True)
def _(
    database_error,
    database_path,
    dataset_filter,
    family_filter,
    pd,
    search_text,
    sqlite3,
):
    _columns = [
        "model_id", "alias", "slug", "family", "dataset", "task", "collection",
        "experiment", "status", "best_loss", "best_iteration", "seed", "created_at",
        "tags",
    ]
    if database_error:
        results = pd.DataFrame(columns=_columns)
    else:
        _clauses = []
        _parameters = []
        _query = search_text.value.strip()
        if _query:
            _searchable_metadata = (
                "COALESCE(m.model_id, '') || ' ' || COALESCE(m.slug, '') || ' ' || "
                "COALESCE(m.collection, '') || ' ' || COALESCE(m.experiment, '') || ' ' || "
                "COALESCE(m.family, '') || ' ' || COALESCE(m.dataset, '') || ' ' || "
                "COALESCE(m.task, '') || ' ' || COALESCE(a.alias, '') || ' ' || "
                "COALESCE(a.notes, '') || ' ' || COALESCE(t.tags, '')"
            )
            _clauses.append(f"LOWER({_searchable_metadata}) LIKE LOWER(?)")
            _parameters.append(f"%{_query}%")
        if family_filter.value.strip():
            _clauses.append("LOWER(m.family) = LOWER(?)")
            _parameters.append(family_filter.value.strip())
        if dataset_filter.value.strip():
            _clauses.append("LOWER(m.dataset) = LOWER(?)")
            _parameters.append(dataset_filter.value.strip())

        _where = f" WHERE {' AND '.join(_clauses)}" if _clauses else ""
        _sql = f"""
            WITH tags AS (
                SELECT model_id, GROUP_CONCAT(tag, ', ') AS tags
                FROM model_tags
                GROUP BY model_id
            )
            SELECT
                m.model_id, a.alias, m.slug, m.family, m.dataset, m.task,
                m.collection, m.experiment, m.status, m.best_loss,
                m.best_iteration, m.seed, m.created_at, t.tags
            FROM models AS m
            LEFT JOIN model_annotations AS a ON a.model_id = m.model_id
            LEFT JOIN tags AS t ON t.model_id = m.model_id
            {_where}
            ORDER BY m.created_at DESC
            LIMIT 500
        """
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
            results = pd.read_sql_query(_sql, _connection, params=_parameters)
    return (results,)


@app.cell(hide_code=True)
def _(database_error, mo, results):
    if database_error:
        _message = mo.callout(database_error, kind="danger")
    else:
        _message = mo.md(f"**{len(results)} matching models** (showing up to 500)")
    _message
    return


@app.cell
def _(mo, results):
    results_table = mo.ui.table(
        results,
        selection="multi",
        page_size=15,
        show_data_types=False,
        freeze_columns_left=["model_id"],
        wrapped_columns=["slug", "tags"],
    )
    results_table
    return (results_table,)


@app.cell(hide_code=True)
def _(database_path, mo, pd, results_table, sqlite3):
    _selected = results_table.value
    if _selected is None or len(_selected) == 0:
        _detail = mo.md("Select a model to view its indexed metadata.")
    else:
        _model_id = _selected.iloc[0]["model_id"]
        _sql = """
            WITH tags AS (
                SELECT model_id, GROUP_CONCAT(tag, ', ') AS tags
                FROM model_tags
                GROUP BY model_id
            )
            SELECT m.*, a.alias, a.notes, t.tags
            FROM models AS m
            LEFT JOIN model_annotations AS a ON a.model_id = m.model_id
            LEFT JOIN tags AS t ON t.model_id = m.model_id
            WHERE m.model_id = ?
        """
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
            _metadata = pd.read_sql_query(_sql, _connection, params=[_model_id]).transpose()
        _metadata.columns = ["Value"]
        _detail = mo.vstack([
            mo.md("## Selected model metadata"),
            mo.ui.table(_metadata, selection=None),
        ])
    _detail
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Side-by-side rollout

    Select up to eight models above. **Verified bundle input** reloads each
    model's configured data source and checks its saved input fingerprint.
    **Specific `.npy` input** accepts a `C × H × W` array (or a batched array,
    from which the first item is used); it is intentionally not verified.
    """)
    return


@app.cell
def _(mo):
    evaluation_mode = mo.ui.dropdown(
        ["Verified bundle input", "Specific .npy input"],
        value="Verified bundle input",
        label="Input source",
    )
    initial_condition_path = mo.ui.text(
        placeholder="/path/to/initial_condition.npy",
        label="Specific input",
        full_width=True,
    )
    rollout_steps = mo.ui.number(0, 512, value=128, step=1, label="Rollout steps")
    rollout_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    run_comparison = mo.ui.run_button(label="Run selected models")
    mo.vstack([
        mo.hstack([evaluation_mode, rollout_steps, rollout_seed]),
        initial_condition_path,
        run_comparison,
    ])
    return (
        evaluation_mode,
        initial_condition_path,
        rollout_seed,
        rollout_steps,
        run_comparison,
    )


@app.cell(hide_code=True)
def _(
    Path,
    evaluation_mode,
    initial_condition_path,
    mo,
    results_table,
    rollout_seed,
    rollout_steps,
    run_comparison,
    store_root,
):
    if not run_comparison.value:
        _comparison = mo.md("Choose models and click **Run selected models** to start a rollout.")
    else:
        try:
            import jax.numpy as jnp
            import jax.random as jr
            import matplotlib.pyplot as plt
            import numpy as np

            from Common.model.boundary import hard_boundary, model_boundary, no_boundary
            from NCA.registry import ModelRegistry, verify_evaluation_input

            _selected = results_table.value
            if _selected is None or len(_selected) == 0:
                raise ValueError("Select at least one model from the search results.")
            if len(_selected) > 8:
                raise ValueError("Select at most eight models for one comparison.")

            _registry = ModelRegistry(Path(store_root.value).expanduser())
            _bundles = [_registry.get(_model_id) for _model_id in _selected["model_id"]]

            def _load_verified_input(_bundle):
                _cfg = _bundle.config
                if _cfg.data.dataset == "emojis":
                    from Experiments.emoji.config_helpers import load_data

                    _data, _ = load_data(_cfg)
                    _boundary = None
                elif str(_cfg.data.dataset).startswith("micropatterns"):
                    from Experiments.micropatterns.config_helpers import load_data

                    _data, _, _, _boundary, _, _ = load_data(_cfg)
                else:
                    raise ValueError(
                        f"No local evaluation loader is registered for dataset {_cfg.data.dataset!r}."
                    )
                if "evaluation_input" not in _bundle.manifest:
                    raise ValueError(
                        f"{_bundle.id} has no evaluation-input fingerprint; republish it or use a specific input."
                    )
                verify_evaluation_input(
                    _data,
                    _bundle.manifest.evaluation_input,
                    boundary_mask=_boundary,
                )
                _initial = np.asarray(_data)[0, 0]
                _boundary = None if _boundary is None else np.asarray(_boundary)[0]
                return _initial, _boundary

            if evaluation_mode.value == "Verified bundle input":
                _inputs = [_load_verified_input(_bundle) for _bundle in _bundles]
                _fingerprint = _bundles[0].manifest.evaluation_input.initial_state.sha256
                _boundary_fingerprint = _bundles[0].manifest.evaluation_input.get(
                    "boundary_mask", None
                )
                if any(
                    _bundle.manifest.evaluation_input.initial_state.sha256 != _fingerprint
                    for _bundle in _bundles[1:]
                ) or any(
                    _bundle.manifest.evaluation_input.get("boundary_mask", None)
                    != _boundary_fingerprint
                    for _bundle in _bundles[1:]
                ):
                    raise ValueError(
                        "Selected models have different verified inputs; "
                        "compare them with a specific input instead."
                    )
            else:
                _input_path = Path(initial_condition_path.value).expanduser()
                if not _input_path.is_file():
                    raise ValueError("Specific input must be an existing .npy file.")
                _initial = np.load(_input_path, allow_pickle=False)
                if _initial.ndim == 4:
                    _initial = _initial[0]
                if _initial.ndim != 3:
                    raise ValueError("Specific input must have shape C x H x W or B x C x H x W.")
                _inputs = [(_initial, None)] * len(_bundles)

            _final_states = []
            for _index, (_bundle, (_initial, _boundary)) in enumerate(zip(_bundles, _inputs)):
                _model = _bundle.load_model(key=jr.PRNGKey(int(rollout_seed.value)))
                _channels = int(_model.N_CHANNELS)
                if _initial.shape[0] > _channels:
                    raise ValueError(
                        f"{_bundle.id} has {_channels} state channels but the input has "
                        f"{_initial.shape[0]}."
                    )
                _state = jnp.pad(
                    jnp.asarray(_initial),
                    ((0, _channels - _initial.shape[0]), (0, 0), (0, 0)),
                )
                if _boundary is None:
                    _callback = no_boundary()
                else:
                    _boundary_state = jnp.asarray(_boundary)
                    if _bundle.config.trainer.get("boundary_mode", "soft") == "hard":
                        _callback = hard_boundary(_boundary_state)
                    else:
                        _callback = model_boundary(_boundary_state)
                _trajectory = _model.run(
                    int(rollout_steps.value),
                    _state,
                    callback=_callback,
                    key=jr.PRNGKey(int(rollout_seed.value) + _index),
                )
                _final_states.append(np.asarray(_trajectory[-1]))

            _figure, _axes = plt.subplots(1, len(_final_states), figsize=(4 * len(_final_states), 4))
            _axes = np.atleast_1d(_axes)
            for _axis, _bundle, _state in zip(_axes, _bundles, _final_states):
                _image = np.moveaxis(_state[: min(3, _state.shape[0])], 0, -1)
                if _image.shape[-1] == 1:
                    _axis.imshow(_image[..., 0], cmap="viridis")
                else:
                    _axis.imshow(np.pad(_image, ((0, 0), (0, 0), (0, 3 - _image.shape[-1]))))
                _axis.set_title(str(_bundle.manifest.slug), fontsize=8)
                _axis.axis("off")
            _figure.tight_layout()
            _comparison = mo.vstack([
                mo.md(f"Compared {len(_bundles)} models after {int(rollout_steps.value)} steps."),
                _figure,
            ])
        except Exception as _error:
            _comparison = mo.callout(str(_error), kind="danger")
    _comparison
    return


if __name__ == "__main__":
    app.run()
