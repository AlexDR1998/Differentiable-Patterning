# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
#   "jax",
#   "equinox",
#   "omegaconf",
#   "python-dotenv",
# ]
# ///

"""A read-only marimo explorer for a local model registry.

Run from the repository root with:

    marimo run Experiments/model_registry_explorer.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path
    _repository_root = Path(__file__).resolve().parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))
    import os
    import sqlite3
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv(_repository_root / ".env", override=False)
        # try:
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.model.boundary import hard_boundary, model_boundary, no_boundary
    from NCA.registry import ModelRegistry, verify_evaluation_input


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Model registry explorer

    Search the read-only SQLite catalogue, then select a model to inspect its
    indexed metadata. Rebuild the index separately with
    `python -m Experiments.model_registry reindex` when needed.
    """)
    return


@app.cell(hide_code=True)
def _():
    _repository_root = Path(__file__).resolve().parents[1]
    _default_store = os.environ.get("MODEL_STORE_ROOT", str(_repository_root / "models"))
    _default_data_root = os.environ.get("DATA_PATH_BASE", "")
    store_root = mo.ui.text(_default_store, label="Model store")
    data_root = mo.ui.text(
        _default_data_root,
        label="Data root",
        placeholder="Directory containing 260726_nca_dataset or Emojis",
        full_width=True,
    )
    mo.vstack([store_root, data_root])
    return data_root, store_root


@app.cell(hide_code=True)
def _(store_root):
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
                _table_rows = _connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
                _tables = {_row[0] for _row in _table_rows}
                _required_tables = {
                    "models", "evaluations", "model_annotations", "model_tags",
                    "model_wandb_tags",
                }
                _model_columns = {
                    _row[1] for _row in _connection.execute("PRAGMA table_info(models)")
                }
                _required_model_columns = {"model_id", "config_id", "display_name"}
                _missing_schema = sorted(
                    (_required_tables - _tables)
                    | (_required_model_columns - _model_columns)
                )
            if _missing_schema:
                database_error = (
                    f"The registry at {database_path} uses an outdated schema "
                    f"(missing: {', '.join(_missing_schema)}). Rebuild it with "
                    f"`python -m Experiments.model_registry --root {_store_path} reindex`."
                )
            else:
                database_error = None
        except (sqlite3.Error, OSError) as _error:
            database_error = f"Could not read {database_path}: {_error}"
    return database_error, database_path


@app.cell(hide_code=True)
def _():
    search_text = mo.ui.text(
        placeholder="Model ID, name, collection, experiment, tag, or dataset",
        label="Search",
        full_width=True,
    )
    family_filter = mo.ui.text(placeholder="e.g. NCA", label="Family")
    dataset_filter = mo.ui.text(placeholder="e.g. emojis", label="Dataset")
    mo.hstack([search_text, family_filter, dataset_filter], widths=[3, 1, 1])
    return dataset_filter, family_filter, search_text


@app.cell(hide_code=True)
def _(database_error, database_path):
    if database_error:
        _wandb_tag_options = []
        _numeric_tag_options = []
    else:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
            _wandb_tag_rows = pd.read_sql_query(
                "SELECT DISTINCT tag FROM model_wandb_tags ORDER BY tag", _connection
            )
        _wandb_tag_options = _wandb_tag_rows["tag"].tolist()
        _tag_values = {}
        for _tag in _wandb_tag_options:
            _tag_key, _separator, _tag_value = _tag.partition(":")
            if _separator:
                _tag_values.setdefault(_tag_key, []).append(_tag_value)
        _numeric_tag_options = []
        for _tag_key, _values in _tag_values.items():
            try:
                [float(_value) for _value in _values]
            except ValueError:
                continue
            _numeric_tag_options.append(_tag_key)
    wandb_tag_filter = mo.ui.multiselect(
        options=_wandb_tag_options,
        label="Configuration/W&B tags (match all)",
        full_width=True,
    )
    numeric_tag_sort = mo.ui.dropdown(
        options=["No sort", "Created at", *_numeric_tag_options],
        value="No sort",
        label="Sort by numerical tag",
        full_width=True,
    )
    numeric_tag_sort_direction = mo.ui.dropdown(
        options=["Ascending", "Descending"],
        value="Ascending",
        label="Direction",
    )
    mo.vstack([
        wandb_tag_filter,
        mo.hstack([numeric_tag_sort, numeric_tag_sort_direction], widths=[3, 1]),
    ])
    return numeric_tag_sort, numeric_tag_sort_direction, wandb_tag_filter


@app.cell(hide_code=True)
def _(
    database_error,
    database_path,
    dataset_filter,
    family_filter,
    numeric_tag_sort,
    numeric_tag_sort_direction,
    search_text,
    wandb_tag_filter,
):
    _columns = [
        "model_id", "alias", "display_name", "family", "dataset", "task", "collection",
        "experiment", "status", "best_loss", "best_iteration", "seed", "created_at",
        "annotation_tags", "wandb_tags",
    ]
    if database_error:
        results = pd.DataFrame(columns=_columns)
    else:
        _clauses = []
        _parameters = []
        _query = search_text.value.strip()
        if _query:
            _searchable_metadata = (
                "COALESCE(m.model_id, '') || ' ' || COALESCE(m.config_id, '') || ' ' || "
                "COALESCE(m.display_name, '') || ' ' || COALESCE(m.slug, '') || ' ' || "
                "COALESCE(m.collection, '') || ' ' || COALESCE(m.experiment, '') || ' ' || "
                "COALESCE(m.family, '') || ' ' || COALESCE(m.dataset, '') || ' ' || "
                "COALESCE(m.task, '') || ' ' || COALESCE(a.alias, '') || ' ' || "
                "COALESCE(a.notes, '') || ' ' || COALESCE(t.tags, '') || ' ' || "
                "COALESCE(wt.tags, '')"
            )
            _clauses.append(f"LOWER({_searchable_metadata}) LIKE LOWER(?)")
            _parameters.append(f"%{_query}%")
        if family_filter.value.strip():
            _clauses.append("LOWER(m.family) = LOWER(?)")
            _parameters.append(family_filter.value.strip())
        if dataset_filter.value.strip():
            _clauses.append("LOWER(m.dataset) = LOWER(?)")
            _parameters.append(dataset_filter.value.strip())
        for _wandb_tag in wandb_tag_filter.value:
            _clauses.append(
                "EXISTS (SELECT 1 FROM model_wandb_tags AS wf "
                "WHERE wf.model_id = m.model_id AND wf.tag = ?)"
            )
            _parameters.append(_wandb_tag)

        _where = f" WHERE {' AND '.join(_clauses)}" if _clauses else ""
        _sort_key = numeric_tag_sort.value
        if _sort_key in {"No sort", "Created at"}:
            _numeric_sort_cte = ""
            _numeric_sort_join = ""
            _numeric_sort_select = ""
            _order_clause = (
                "" if _sort_key == "No sort" else "ORDER BY m.created_at DESC"
            )
            _query_parameters = _parameters
        else:
            _direction = (
                "ASC" if numeric_tag_sort_direction.value == "Ascending" else "DESC"
            )
            _numeric_sort_cte = """, numeric_sort AS (
                SELECT model_id,
                       MAX(CAST(SUBSTR(tag, INSTR(tag, ':') + 1) AS REAL)) AS value
                FROM model_wandb_tags
                WHERE SUBSTR(tag, 1, INSTR(tag, ':') - 1) = ?
                GROUP BY model_id
            )"""
            _numeric_sort_join = (
                "LEFT JOIN numeric_sort AS ns ON ns.model_id = m.model_id"
            )
            _numeric_sort_select = ", ns.value AS numeric_tag_value"
            _order_clause = (
                f"ORDER BY ns.value IS NULL, ns.value {_direction}, m.created_at DESC"
            )
            _query_parameters = [_sort_key, *_parameters]
        _sql = f"""
            WITH tags AS (
                SELECT model_id, GROUP_CONCAT(tag, ', ') AS tags
                FROM model_tags
                GROUP BY model_id
            ), wandb_tags AS (
                SELECT model_id, GROUP_CONCAT(tag, ', ') AS tags
                FROM model_wandb_tags
                GROUP BY model_id
            ){_numeric_sort_cte}
            SELECT
                m.model_id, a.alias, m.display_name, m.family, m.dataset, m.task,
                m.collection, m.experiment, m.status, m.best_loss,
                m.best_iteration, m.seed, m.created_at,
                t.tags AS annotation_tags, wt.tags AS wandb_tags
                {_numeric_sort_select}
            FROM models AS m
            LEFT JOIN model_annotations AS a ON a.model_id = m.model_id
            LEFT JOIN tags AS t ON t.model_id = m.model_id
            LEFT JOIN wandb_tags AS wt ON wt.model_id = m.model_id
            {_numeric_sort_join}
            {_where}
            {_order_clause}
            LIMIT 500
        """
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
            results = pd.read_sql_query(_sql, _connection, params=_query_parameters)
        if _sort_key not in {"No sort", "Created at"}:
            results = results.rename(
                columns={"numeric_tag_value": f"sort: {_sort_key}"}
            )
    return (results,)


@app.cell(hide_code=True)
def _(database_error, results):
    if database_error:
        _message = mo.callout(database_error, kind="danger")
    else:
        _message = mo.md(f"**{len(results)} matching models** (showing up to 500)")
    _message
    return


@app.cell(hide_code=True)
def _(results):
    results_table = mo.ui.table(
        results,
        selection="multi",
        page_size=15,
        show_data_types=False,
        freeze_columns_left=["model_id","experiment"],
        wrapped_columns=[],
    )
    results_table
    return (results_table,)


@app.cell(hide_code=True)
def _(database_path, results_table):
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
            ), wandb_tags AS (
                SELECT model_id, GROUP_CONCAT(tag, ', ') AS tags
                FROM model_wandb_tags
                GROUP BY model_id
            )
            SELECT m.*, a.alias, a.notes, t.tags AS annotation_tags,
                   wt.tags AS wandb_tags
            FROM models AS m
            LEFT JOIN model_annotations AS a ON a.model_id = m.model_id
            LEFT JOIN tags AS t ON t.model_id = m.model_id
            LEFT JOIN wandb_tags AS wt ON wt.model_id = m.model_id
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
def _():
    mo.md("""
    ## Side-by-side rollout

    Select up to eight models above. **Verified bundle input** reloads each
    model's configured data source and checks its saved input fingerprint.
    **Specific `.npy` input** accepts a `C × H × W` array (or a batched array,
    from which the first item is used); it is intentionally not verified.
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
    _timepoint_options = ["0h", "12h", "24h", "36h", "48h"]
    _channel_options = [
        "DAPI", "LMBR", "TBXT", "SOX17", "SOX2", "FOXA2",
        "CER1", "LEFTY", "NODAL", "LEF1", "SMAD23",
    ]
    timepoint_filter = mo.ui.multiselect(
        options=_timepoint_options,
        value=_timepoint_options,
        label="Displayed timepoints",
    )
    channel_filter = mo.ui.multiselect(
        options=_channel_options,
        value=_channel_options,
        label="Displayed channels",
    )
    mo.hstack([timepoint_filter, channel_filter])
    return channel_filter, timepoint_filter


@app.cell(hide_code=True)
def _(
    data_root,
    evaluation_mode,
    initial_condition_path,
    results_table,
    rollout_seed,
    rollout_steps,
    run_comparison,
    store_root,
):
    if not run_comparison.value:
        comparison_rollouts = None
        _status = mo.md("Choose models and click **Run selected models** to start a rollout.")
    else:


        _selected = results_table.value
        if _selected is None or len(_selected) == 0:
            raise ValueError("Select at least one model from the search results.")
        if len(_selected) > 8:
            raise ValueError("Select at most eight models for one comparison.")

        _registry = ModelRegistry(Path(store_root.value).expanduser())
        _bundles = [_registry.get(_model_id) for _model_id in _selected["model_id"]]

        def _load_verified_input(_bundle):
            _cfg = _bundle.config
            _data_root = Path(data_root.value).expanduser()
            if not data_root.value.strip():
                raise ValueError(
                    "Set Data root to DATA_PATH_BASE before using verified bundle input."
                )
            if _cfg.data.dataset == "emojis":
                from Experiments.emoji.config_helpers import load_data

                _data_path = _data_root / "Emojis"
                _data, _ = load_data(_cfg.data, impath=str(_data_path))
                _boundary = None
                _channel_names = ()
            elif str(_cfg.data.dataset).startswith("micropatterns"):
                from Experiments.micropatterns.config_helpers import load_data

                if _cfg.data.dataset == "micropatterns_260726":
                    _data_path = _data_root / "260726_nca_dataset"
                else:
                    _data_path = _data_root / "Timecourse_seperate_colonies"
                if not _data_path.is_dir():
                    raise ValueError(
                        f"Dataset {_cfg.data.dataset!r} is not available at "
                        f"{_data_path}. Set Data root to the directory containing "
                        f"{_data_path.name}."
                    )
                _data, _, _channel_names, _boundary, _, _ = load_data(
                    _cfg.data, impath=str(_data_path)
                )
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
            return _initial, _boundary, tuple(_channel_names)

        if evaluation_mode.value == "Verified bundle input":
            _inputs = [_load_verified_input(_bundle) for _bundle in _bundles]
            _fingerprint = _bundles[0].manifest.evaluation_input.initial_state.sha256
            _boundary_fingerprint = _bundles[0].manifest.evaluation_input.get(
                "boundary_mask", None
            )
            _inputs_differ = any(
                _bundle.manifest.evaluation_input.initial_state.sha256 != _fingerprint
                for _bundle in _bundles[1:]
            ) or any(
                _bundle.manifest.evaluation_input.get("boundary_mask", None)
                != _boundary_fingerprint
                for _bundle in _bundles[1:]
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
            _inputs = [(_initial, None, ())] * len(_bundles)
            _inputs_differ = False

        _rollouts = []
        for _index, (_bundle, (_initial, _boundary, _channel_names)) in enumerate(
            zip(_bundles, _inputs)
        ):
            _model = _bundle.load_model(
                key=jr.PRNGKey(int(rollout_seed.value)),
                implementation="portable",
            )
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
            if str(_bundle.config.data.dataset).startswith("micropatterns"):
                _time_labels = tuple(
                    f"{int(_hour)}h"
                    for _hour in _bundle.config.data.micropattern.timesteps
                )
            else:
                _time_labels = ("0h", "12h", "24h", "36h", "48h")
            _steps_per_observation = int(_bundle.config.run.t)
            _frame_indices = tuple(
                _time_index * _steps_per_observation
                for _time_index in range(len(_time_labels))
            )
            _total_steps = max(int(rollout_steps.value), _frame_indices[-1])
            _trajectory = _model.run(
                _total_steps,
                _state,
                callback=_callback,
                key=jr.PRNGKey(int(rollout_seed.value) + _index),
            )
            _observed_channels = int(_initial.shape[0])
            _names = list(_channel_names[:_observed_channels])
            _names.extend(
                f"Channel {_channel_index + 1}"
                for _channel_index in range(len(_names), _observed_channels)
            )
            _frames = np.asarray(_trajectory)[list(_frame_indices), :_observed_channels]
            _rollouts.append((_bundle, _frames, _time_labels, tuple(_names)))

        comparison_rollouts = tuple(_rollouts)
        _input_note = (
            " Each model used its own individually verified input."
            if _inputs_differ
            else ""
        )
        _status = mo.md(
            f"Ran {len(_bundles)} model(s). Use the display filters to compare subsets."
            f"{_input_note}"
        )
    _status
    return (comparison_rollouts,)


@app.cell(hide_code=True)
def _(channel_filter, comparison_rollouts, timepoint_filter):
    if comparison_rollouts is None:
        _comparison = mo.md("")
    elif not timepoint_filter.value or not channel_filter.value:
        _comparison = mo.callout(
            "Select at least one timepoint and one channel.", kind="info"
        )
    else:
        _figures = []
        for _bundle, _frames, _time_labels, _channel_names in comparison_rollouts:
            _time_indices = [
                _index for _index, _label in enumerate(_time_labels)
                if _label in timepoint_filter.value
            ]
            _channel_indices = [
                _index for _index, _name in enumerate(_channel_names)
                if _name.rsplit("/", 1)[-1] in channel_filter.value
            ]
            if not _time_indices or not _channel_indices:
                continue
            _selected_times = tuple(_time_labels[_index] for _index in _time_indices)
            _selected_channels = tuple(
                _channel_names[_index] for _index in _channel_indices
            )
            _selected_frames = _frames[np.ix_(_time_indices, _channel_indices)]
            _row_count = len(_selected_channels)
            _column_count = len(_selected_times)
            _figure, _axes = plt.subplots(
                _row_count,
                _column_count,
                figsize=(2.2 * _column_count, 2.2 * _row_count),
                squeeze=False,
            )
            for _channel_index, _channel_name in enumerate(_selected_channels):
                _channel_frames = _selected_frames[:, _channel_index]
                _vmin = float(np.nanmin(_channel_frames))
                _vmax = float(np.nanmax(_channel_frames))
                if _vmax <= _vmin:
                    _vmax = _vmin + 1.0
                for _time_index, _time_label in enumerate(_selected_times):
                    _axis = _axes[_channel_index, _time_index]
                    _axis.imshow(
                        _channel_frames[_time_index],
                        cmap="gray",
                        vmin=_vmin,
                        vmax=_vmax,
                    )
                    if _channel_index == 0:
                        _axis.set_title(_time_label)
                    if _time_index == 0:
                        _axis.set_ylabel(_channel_name, rotation=0, ha="right", va="center")
                    _axis.set_xticks([])
                    _axis.set_yticks([])
            # _figure.suptitle(str(_bundle.manifest.slug), fontsize=10)
            _figure.tight_layout()
            _figures.append(_figure)
        if _figures:
            _comparison = mo.hstack(_figures)
        else:
            _comparison = mo.callout(
                "None of the selected models contain the chosen channels or timepoints.",
                kind="info",
            )
    _comparison
    return


if __name__ == "__main__":
    app.run()
