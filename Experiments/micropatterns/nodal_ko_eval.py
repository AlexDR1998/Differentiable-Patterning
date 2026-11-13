# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
#   "plotly",
#   "jax",
#   "equinox",
#   "omegaconf",
#   "python-dotenv",
#   "tqdm",
#   "opencv-python",
#   "cmapy",
#   "einops",
# ]
# ///

"""Evaluate an exported model-registry selection on NODAL-KO conditions.

Run from the repository root with:

    marimo run Experiments/micropatterns/nodal_ko_eval.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    from dataclasses import replace
    import json
    import os
    import sys
    import time
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[2]

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from matplotlib.colors import to_rgb
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    from Common.dataloader.micropattern_projections import PROJECTIONS
    from Experiments.model_registry import ModelRegistry, load_selection, open_model_bundle
    from NCA.trainer.intervention import (
        rollout_model,
        rollout_model_sampled,
        rollout_model_with_blocked_channel,
        rollout_model_with_blocked_channel_sampled,
        rollout_model_with_blocked_channel_pattern_counts,
    )
    from Common.dataloader.cell_type_fit import radial_rings
    from Common.dataloader.cell_type_shares import (
        colony_label_shares,
        pattern_counts,
        rule_stains,
        stain_counts_from_patterns,
    )
    from Common.dataloader.cell_types import (
        OTHER,
        SEVERAL,
        CellTypeRules,
        label_names,
        load_cell_type_rules,
        marker_values,
        pixel_label_shares,
        sample_pixel_labels,
        save_cell_type_rules,
    )
    from Common.model.boundary import build_boundary_callback, no_boundary
    from Common.save_to_video import save_to_video_rgb
    # Cell placement of the density sweep (tuned in synthetic_textured_ic_explorer.py)
    from Experiments.micropatterns.texture_settings import DEFAULT_TEXTURE_SETTINGS_FILE, load_cell_settings
    # Density × KO sweep (run here or as a remote job), and the data loading
    # it shares with this notebook
    from Experiments.micropatterns.density_ko_sweep import (
        DensityKoSettings,
        actual_densities,
        capped_densities,
        format_density,
        job_configs,
        load_condition_data,
        load_density_ko_results,
        load_training_histogram_bins,
        model_pattern_counts,
        record_label,
        run_density_ko_sweep,
        save_density_ko_results,
    )
    from Experiments.notebook_jobs import repository_path, write_job_manifest

    def repository_relative_path(path):
        """``path`` relative to the repository root when it lies inside it."""
        path = Path(path).resolve()
        return str(path.relative_to(repository_root)) if path.is_relative_to(repository_root) else str(path)
    from Experiments.micropatterns.rendering import (
        render_cmy_composite,
        render_fate_cmy,
    )
    # Evaluation helpers (rollouts under a knockout, cell-type shares)
    from Experiments.micropatterns.ko_evaluation import (
        DENSITY_AXIS_LABELS,
        DENSITY_COLUMNS,
        DENSITY_UNITS,
        REFERENCE_LABELS,
        REFERENCE_LINESTYLES,
        REFERENCE_MARKERS,
        add_density_axes,
        cell_edges,
        density_column,
        draw_reference_strip,
        density_conversions,
        density_rows,
        data_label_shares,
        evaluate_condition,
        ko_time_label,
        ko_time_ticks,
        main_density_unit,
        make_render_initial_condition,
        marker_channel_indices,
        measured_density,
        model_label_shares,
        model_option_label,
        model_stain_counts,
        parse_one_based_indices,
        read_model_settings,
        reference_legend_handles,
        rules_projection,
        uncertainty_band,
    )
    # Shared figure style and colony drawing
    from Experiments.micropatterns.figures.data import cell_type_rgba, masked_rgba
    from Experiments.micropatterns.figures.style import (
        PROJECTION_COLOURS,
        SERIES,
        STYLE_FILE,
        cell_type_colour,
    )
    from matplotlib.patches import Patch

    load_dotenv(repository_root / ".env", override=False)

    # Cell types come from a rule file exported by data_cleaning.py.
    CELL_TYPE_DIRECTORY = repository_root / "Experiments/micropatterns/conf/cell_types"


    # Display order of the conditions
    CONDITION_LABELS = {"baseline": "Baseline", "ko_24h": "KO 24h", "ko_0h": "KO 0h"}

    # Standard matplotlib colour maps offered for the heatmaps
    COLOUR_MAPS = (
        "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
        "gray", "Greys", "Blues", "Reds", "YlGnBu", "coolwarm", "RdBu_r",
    )


@app.cell(hide_code=True)
def _():
    plt.style.use(["default", STYLE_FILE])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # NODAL knockout model evaluation

    Load a YAML selection exported by `model_registry_explorer.py`, then run
    those models against matching baseline, 0h NODAL-KO, or 24h NODAL-KO
    measurements. Knockout predictions block the model from reading its
    recurrent NODAL channel from the selected intervention time onward.

    The reusable `evaluation_results` value produced below contains the full
    predictions, experimental targets, registry details, and saved notes for
    custom analysis and figure construction.
    """)
    return


@app.cell(hide_code=True)
def _():
    _backend = jax.default_backend()
    _devices = ", ".join(
        f"{_device.platform}: {_device.device_kind}" for _device in jax.devices()
    )
    _kind = "success" if _backend == "gpu" else "warn"
    mo.callout(
        f"JAX backend: **{_backend}** · Devices: {_devices}. "
        "The first rollout for each model architecture includes JIT compilation.",
        kind=_kind,
    )
    return


@app.cell(hide_code=True)
def _():
    _default_store = os.environ.get(
        "MODEL_STORE_ROOT", str(repository_root / "models")
    )
    _default_data = os.environ.get("DATA_PATH_BASE", "")
    selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    store_root = mo.ui.text(
        _default_store,
        label="Model store",
        full_width=True,
    )
    data_root = mo.ui.text(
        _default_data,
        label="Data root",
        placeholder="Directory containing 260726_nca_dataset",
        full_width=True,
    )
    mo.vstack([selection_path, store_root, data_root])
    return data_root, selection_path, store_root


@app.cell(hide_code=True)
def _(selection_path, store_root):
    _path = Path(selection_path.value).expanduser()
    if not selection_path.value.strip():
        loaded_records = ()
        _load_status = mo.md("Choose an exported model selection YAML file.")
    elif not _path.is_file():
        loaded_records = ()
        _load_status = mo.callout(f"Selection file not found: `{_path}`", kind="danger")
    else:
        _records = load_selection(_path)
        # Projection and downsample come from each bundle's config.
        _settings = read_model_settings(
            ModelRegistry(Path(store_root.value).expanduser()),
            [_record["model_id"] for _record in _records],
        )
        loaded_records = tuple(
            {**_record, **_settings[_record["model_id"]]} for _record in _records
        )
        _load_status = mo.md("")
    _load_status
    return (loaded_records,)


@app.cell(hide_code=True)
def _(loaded_records):
    _present_projections = sorted(
        {_record["projection"] for _record in loaded_records},
        key=lambda _name: PROJECTIONS.index(_name) if _name in PROJECTIONS else len(PROJECTIONS),
    )
    selection_projections = mo.ui.multiselect(
        options=_present_projections,
        value=_present_projections,
        label="Projections",
    )
    return (selection_projections,)


@app.cell(hide_code=True)
def _(loaded_records, selection_projections):
    # Every section below only offers the models of the chosen projections.
    selection_records = tuple(
        _record for _record in loaded_records
        if _record["projection"] in selection_projections.value
    )
    if not loaded_records:
        _preview = mo.md("")
    else:
        _columns = [
            "model_id", "alias", "display_name", "notes", "projection",
            "downsample", "family", "experiment", "best_loss", "seed",
        ]
        _preview_data = pd.DataFrame(selection_records).reindex(columns=_columns)
        _projection_counts = pd.DataFrame(loaded_records).groupby(
            ["projection", "downsample"]
        ).size().rename("models").reset_index()
        _preview = mo.vstack([
            mo.md(f"## Selected models ({len(selection_records)} of {len(loaded_records)})"),
            mo.hstack([
                selection_projections,
                mo.ui.table(_projection_counts, selection=None, page_size=12),
            ], justify="start"),
            mo.ui.table(_preview_data, selection=None, page_size=10),
        ])
    _preview
    return (selection_records,)


@app.cell(hide_code=True)
def _():
    evaluation_conditions = mo.ui.multiselect(
        options=["baseline", "ko_0h", "ko_24h"],
        value=["baseline", "ko_0h", "ko_24h"],
        label="Evaluation conditions",
    )
    evaluation_replicate = mo.ui.number(
        1,
        999,
        value=1,
        step=1,
        label="Replicate (one-based)",
    )
    evaluation_seed = mo.ui.number(
        0,
        2**31 - 1,
        value=0,
        step=1,
        label="Rollout seed",
    )
    run_evaluation = mo.ui.run_button(label="Run NODAL-KO evaluation")
    mo.hstack([
        evaluation_conditions,
        evaluation_replicate,
        evaluation_seed,
        run_evaluation,
    ])
    return (
        evaluation_conditions,
        evaluation_replicate,
        evaluation_seed,
        run_evaluation,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    evaluation_conditions,
    evaluation_replicate,
    evaluation_seed,
    run_evaluation,
    selection_records,
    store_root,
):
    if not run_evaluation.value:
        evaluation_results = None
        evaluation_timings = pd.DataFrame()
        _evaluation_status = mo.md(
            "Choose a selection and conditions, then run the evaluation."
        )
    else:
        if not selection_records:
            raise ValueError("Choose a non-empty exported registry selection.")
        if not evaluation_conditions.value:
            raise ValueError("Select at least one evaluation condition.")
        if not data_root.value.strip():
            raise ValueError("Set Data root before loading NODAL-KO data.")

        _timing_rows = []
        _model_load_start = time.perf_counter()
        _registry = ModelRegistry(Path(store_root.value).expanduser())
        _bundles = [
            _registry.get(_record["model_id"]) for _record in selection_records
        ]
        _models = [
            _bundle.load_model(
                key=jr.PRNGKey(int(evaluation_seed.value)),
                implementation="portable",
            )
            for _bundle in tqdm(_bundles, desc="Loading models", unit="model")
        ]
        _timing_rows.append({
            "phase": "model loading",
            "model_id": "all",
            "condition": "all",
            "cache_hit": False,
            "seconds": time.perf_counter() - _model_load_start,
        })
        _root = Path(data_root.value).expanduser()
        _histogram_cache = {}
        _condition_data_cache = {}
        _results = []
        for _model_index, (_model, _bundle, _record) in enumerate(tqdm(
            zip(_models, _bundles, selection_records),
            total=len(_models),
            desc="Evaluating models",
            unit="model",
        )):
            _label = record_label(_record)
            for _condition_index, _condition in enumerate(
                evaluation_conditions.value
            ):
                _effective_data_config = replace(
                    _bundle.config.data,
                    knockout=replace(
                        _bundle.config.data.knockout,
                        curriculum=(_condition,),
                    ),
                )
                _data_cache_key = (
                    repr(_effective_data_config),
                    str(_root.resolve()),
                    _condition,
                    int(evaluation_replicate.value),
                )
                _data_cache_hit = _data_cache_key in _condition_data_cache
                _data_load_start = time.perf_counter()
                if not _data_cache_hit:
                    _baseline_data_config = replace(
                        _bundle.config.data,
                        knockout=replace(
                            _bundle.config.data.knockout,
                            curriculum=("baseline",),
                        ),
                    )
                    _histogram_cache_key = (
                        repr(_baseline_data_config),
                        str(_root.resolve()),
                    )
                    if _histogram_cache_key not in _histogram_cache:
                        _histogram_cache[_histogram_cache_key] = (
                            load_training_histogram_bins(_bundle, _root)
                        )
                    _condition_data_cache[_data_cache_key] = load_condition_data(
                        _bundle,
                        _root,
                        _condition,
                        int(evaluation_replicate.value),
                        _histogram_cache[_histogram_cache_key],
                    )
                _condition_data = _condition_data_cache[_data_cache_key]
                _timing_rows.append({
                    "phase": "data loading",
                    "model_id": _bundle.id,
                    "condition": _condition,
                    "cache_hit": _data_cache_hit,
                    "seconds": time.perf_counter() - _data_load_start,
                })
                _rollout_start = time.perf_counter()
                _prediction, _time_labels = evaluate_condition(
                    _model,
                    _bundle,
                    _condition_data,
                    _condition,
                    int(evaluation_seed.value)
                    + _model_index * len(evaluation_conditions.value)
                    + _condition_index,
                )
                _timing_rows.append({
                    "phase": "compiled rollout",
                    "model_id": _bundle.id,
                    "condition": _condition,
                    "cache_hit": False,
                    "seconds": time.perf_counter() - _rollout_start,
                })
                _results.append({
                    "model_id": _bundle.id,
                    "label": str(_label),
                    "notes": _record.get("notes"),
                    "projection": _record["projection"],
                    "downsample": _record["downsample"],
                    "registry_record": _record,
                    "condition": _condition,
                    "replicate": int(evaluation_replicate.value),
                    "prediction": _prediction,
                    "target": _condition_data["target"],
                    "channel_names": _condition_data["channel_names"],
                    "time_labels": _time_labels,
                    "boundary": np.asarray(
                        _condition_data["boundary"]
                    ).squeeze().astype(bool),
                })
        evaluation_results = tuple(_results)
        evaluation_timings = pd.DataFrame(_timing_rows)
        _evaluation_status = mo.callout(
            f"Completed {len(evaluation_results)} model-condition evaluations.",
            kind="success",
        )
    _evaluation_status
    return evaluation_results, evaluation_timings


@app.cell(hide_code=True)
def _(evaluation_timings):
    if evaluation_timings.empty:
        _timing_view = mo.md("")
    else:
        _timing_view = mo.vstack([
            mo.md("## Evaluation timings"),
            mo.ui.table(evaluation_timings, selection=None, page_size=25),
        ])
    _timing_view
    return


@app.cell(hide_code=True)
def _(evaluation_results):
    if evaluation_results is None:
        evaluation_metrics = pd.DataFrame()
        _metrics_view = mo.md("")
    else:
        _metric_rows = []
        for _result in evaluation_results:
            _target = _result["target"]
            _prediction = _result["prediction"]
            _time_count = min(_prediction.shape[0], _target.shape[0])
            _channel_count = min(_prediction.shape[1], _target.shape[1])
            _squared_error = (
                _prediction[:_time_count, :_channel_count]
                - _target[:_time_count, :_channel_count]
            ) ** 2
            for _time_index in range(_time_count):
                _metric_rows.append({
                    "model_id": _result["model_id"],
                    "label": _result["label"],
                    "projection": _result["projection"],
                    "downsample": _result["downsample"],
                    "condition": _result["condition"],
                    "replicate": _result["replicate"],
                    "time": _result["time_labels"][_time_index],
                    "mse": float(np.nanmean(_squared_error[_time_index])),
                })
        evaluation_metrics = pd.DataFrame(_metric_rows)
        # Each projection is compared against its own (separately normalised) data.
        _projection_metrics = (
            evaluation_metrics.groupby(
                ["projection", "downsample", "condition", "time"], sort=False
            )["mse"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        _metrics_view = mo.vstack([
            mo.md("## Starter metrics"),
            mo.md("- Mean MSE per projection (each against its own data)"),
            mo.ui.table(_projection_metrics, selection=None, page_size=15),
            mo.ui.table(evaluation_metrics, selection=None, page_size=15),
        ])
    _metrics_view
    return


@app.cell(hide_code=True)
def _(evaluation_results):
    if evaluation_results is None:
        _model_options = {}
        _projection_options = []
        _group_options = []
        _channel_options = []
        _time_options = []
    else:
        _model_options = {
            model_option_label(_result["registry_record"]): _result["model_id"]
            for _result in evaluation_results
        }
        _projection_options = list(dict.fromkeys(
            _result["projection"] for _result in evaluation_results
        ))
        _group_options = list(dict.fromkeys(
            _name.split("/", 1)[0]
            for _result in evaluation_results
            for _name in _result["channel_names"]
            if "/" in _name
        ))
        _channel_options = list(dict.fromkeys(
            _name.rsplit("/", 1)[-1]
            for _result in evaluation_results
            for _name in _result["channel_names"]
        ))
        _time_options = list(dict.fromkeys(
            _time
            for _result in evaluation_results
            for _time in _result["time_labels"]
        ))
    display_models = mo.ui.multiselect(
        options=_model_options,
        value=list(_model_options),
        label="Displayed models",
        full_width=True,
    )
    display_projections = mo.ui.multiselect(
        options=_projection_options,
        value=_projection_options,
        label="Displayed projections",
    )
    display_channels = mo.ui.multiselect(
        options=_channel_options,
        value=_channel_options[:1],
        label="Displayed channels",
        full_width=True,
    )
    display_groups = mo.ui.multiselect(
        options=_group_options,
        value=_group_options[:1],
        label="Displayed experiment groups",
        full_width=True,
    )
    channel_selection_mode = mo.ui.dropdown(
        {
            "Experiment groups": "groups",
            "Individual channels": "channels",
        },
        value="Experiment groups",
        label="Channel selection",
        full_width=True,
    )
    display_times = mo.ui.multiselect(
        options=_time_options,
        value=_time_options,
        label="Displayed timepoints",
        full_width=True,
    )
    display_source = mo.ui.dropdown(
        {
            "Both side-by-side": "both",
            "True data only": "target",
            "Model predictions only": "prediction",
        },
        value="Both side-by-side",
        label="Rendered data",
        full_width=True,
    )
    mo.vstack([
        mo.hstack([display_projections, display_models], widths=[1, 4]),
        mo.hstack([channel_selection_mode, display_groups, display_channels]),
        mo.hstack([display_times, display_source]),
    ])
    return (
        channel_selection_mode,
        display_channels,
        display_groups,
        display_models,
        display_projections,
        display_source,
        display_times,
    )


@app.cell(hide_code=True)
def _(
    channel_selection_mode,
    display_channels,
    display_groups,
    display_models,
    display_projections,
    display_source,
    display_times,
    evaluation_results,
):
    if (
        evaluation_results is None
        or not display_models.value
        or not display_times.value
        or (
            channel_selection_mode.value == "groups"
            and not display_groups.value
        )
        or (
            channel_selection_mode.value == "channels"
            and not display_channels.value
        )
    ):
        _figure_view = mo.md("")
    else:
        _figures = []
        for _result in evaluation_results:
            if (
                _result["model_id"] not in display_models.value
                or _result["projection"] not in display_projections.value
            ):
                continue
            if display_source.value == "target":
                _source_specs = (("True", _result["target"]),)
            elif display_source.value == "prediction":
                _source_specs = (("Prediction", _result["prediction"]),)
            else:
                _source_specs = (
                    ("True", _result["target"]),
                    ("Prediction", _result["prediction"]),
                )
            _max_channels = min(_array.shape[1] for _, _array in _source_specs)
            _max_times = min(_array.shape[0] for _, _array in _source_specs)
            _channel_indices = [
                _index
                for _index, _name in enumerate(_result["channel_names"])
                if (
                    (
                        channel_selection_mode.value == "groups"
                        and "/" in _name
                        and _name.split("/", 1)[0] in display_groups.value
                    )
                    or (
                        channel_selection_mode.value == "channels"
                        and _name.rsplit("/", 1)[-1]
                        in display_channels.value
                    )
                )
                and _index < _max_channels
            ]
            _time_indices = [
                _index
                for _index, _time in enumerate(_result["time_labels"])
                if _time in display_times.value
            ]
            if not _channel_indices or not _time_indices:
                continue
            _time_indices = [
                _index for _index in _time_indices
                if _index < _max_times
            ]
            if not _time_indices:
                continue
            _row_count = len(_channel_indices)
            _source_count = len(_source_specs)
            _column_count = len(_time_indices) * _source_count
            _tile_rows = []
            _row_labels = []
            _column_labels = []
            for _channel_offset, _channel_index in enumerate(_channel_indices):
                _channel_name = _result["channel_names"][
                    _channel_index
                ]
                if channel_selection_mode.value == "channels":
                    _channel_name = _channel_name.rsplit("/", 1)[-1]
                _combined = np.concatenate([
                    _array[_time_indices, _channel_index].reshape(-1)
                    for _, _array in _source_specs
                ])
                _vmin = float(np.nanmin(_combined))
                _vmax = float(np.nanmax(_combined))
                if _vmax <= _vmin:
                    _vmax = _vmin + 1.0
                _row_tiles = []
                for _time_offset, _time_index in enumerate(_time_indices):
                    for _source_offset, (_source_label, _array) in enumerate(
                        _source_specs
                    ):
                        _row_tiles.append(
                            np.clip(
                                (_array[_time_index, _channel_index] - _vmin)
                                / (_vmax - _vmin),
                                0.0,
                                1.0,
                            )
                        )
                        if _channel_offset == 0:
                            _title = _result["time_labels"][_time_index]
                            if _source_count > 1:
                                _title += f" · {_source_label}"
                            _column_labels.append(_title)
                _tile_rows.append(np.concatenate(_row_tiles, axis=1))
                _row_labels.append(_channel_name)
            _mosaic = np.concatenate(_tile_rows, axis=0)
            _tile_height, _tile_width = _row_tiles[0].shape
            _figure, _axis = plt.subplots(
                figsize=(min(20, max(6, 1.45 * _column_count)), max(3, 1.4 * _row_count))
            )
            _axis.imshow(_mosaic, cmap="gray", vmin=0.0, vmax=1.0)
            _axis.set_xticks(
                (np.arange(_column_count) + 0.5) * _tile_width - 0.5,
                _column_labels,
                rotation=0,
                ha="center",
            )
            _axis.set_yticks(
                (np.arange(_row_count) + 0.5) * _tile_height - 0.5,
                _row_labels,
            )
            _axis.set_title(
                f"{_result['label']} · {_result['projection']} "
                f"ds{_result['downsample']} · {_result['condition']}"
            )
            _axis.tick_params(length=0)
            _figure.tight_layout()
            _figures.append(_figure)
        _figure_view = mo.vstack(_figures) if _figures else mo.md(
            "The selected channel is not available in these results."
        )
    _figure_view
    return


@app.cell(hide_code=True)
def _(get_cell_type_files_version):
    # Listed again after a file is saved in the threshold tuning section.
    get_cell_type_files_version()
    _cell_type_files = sorted(CELL_TYPE_DIRECTORY.glob("*.yaml"))
    _file_options = {
        f"{_path.name} (tuned on {rules_projection(load_cell_type_rules(_path))})": str(_path)
        for _path in _cell_type_files
    }
    # Default: the newest file tuned on the projection, else the newest file.
    _default_files = {
        _projection: (
            [_option for _option in _file_options if _option.endswith(f"(tuned on {_projection})")]
            or list(_file_options)
            or [None]
        )[-1]
        for _projection in PROJECTIONS
    }
    # Thresholds depend on the projection, so each one has its own file.
    cell_type_files = mo.ui.dictionary({
        _projection: mo.ui.dropdown(
            options=_file_options,
            value=_default_files[_projection],
            label=f"Cell type file for {_projection}",
            full_width=True,
        )
        for _projection in PROJECTIONS
    })
    cell_type_rings = mo.ui.number(
        1, 40, value=12, step=1, label="Rings for the share estimate"
    )
    mo.vstack([
        mo.md(
            "## Cell types\n\n"
            "Thresholds, cell-type rules and stains are read from a file "
            "exported by `data_cleaning.py` (Export cell types). Because SOX2 "
            "and FOXA2 are imaged in different stains, cell-type shares are "
            "estimated per radial ring by combining the stains (see "
            "`Common/dataloader/cell_type_shares.py`), for the model output "
            "as well as the data, and the rings are then averaged by area. "
            "Pixels matching more than one cell type are counted as *Several*."
        ),
        mo.md("- Each projection uses its own file; models and data use the file of their projection"),
        mo.hstack([cell_type_files, cell_type_rings]),
    ])
    return cell_type_files, cell_type_rings


@app.cell(hide_code=True)
def _(cell_type_files):
    mo.stop(
        any(_path is None for _path in cell_type_files.value.values()),
        mo.callout(f"No cell type files found in `{CELL_TYPE_DIRECTORY}`.", kind="warn"),
    )
    cell_type_rules_by_projection = {
        _projection: load_cell_type_rules(_path)
        for _projection, _path in cell_type_files.value.items()
    }
    # Plots and tables share one list of cell types across projections.
    _label_sets = {
        _projection: tuple(label_names(_rules))
        for _projection, _rules in cell_type_rules_by_projection.items()
    }
    mo.stop(
        len(set(_label_sets.values())) > 1,
        mo.callout(
            "The cell type files must define the same cell types: "
            + "; ".join(f"{_p}: {', '.join(_l)}" for _p, _l in _label_sets.items()),
            kind="danger",
        ),
    )
    cell_type_labels = next(iter(_label_sets.values()))

    _rule_tabs = {}
    for _projection, _rules in cell_type_rules_by_projection.items():
        _rule_rows = [
            {
                "cell_type": _cell_type,
                "clause": _index + 1,
                "weight": _weight,
                **{_marker: _clause.get(_marker, "any") for _marker in _rules.markers},
            }
            for _cell_type, _clauses in _rules.rules.items()
            for _index, (_clause, _weight) in enumerate(
                zip(_clauses, _rules.weights_of(_cell_type))
            )
        ]
        _mismatch = (
            [mo.callout(
                f"This file was tuned on the {rules_projection(_rules)} projection.",
                kind="warn",
            )]
            if rules_projection(_rules) != _projection else []
        )
        _rule_tabs[_projection] = mo.vstack([
            *_mismatch,
            mo.md(
                f"Tuned at {_rules.hour}h. Thresholds: "
                + ", ".join(
                    f"{_marker} {_value:g}"
                    for _marker, _value in _rules.thresholds.items()
                )
                + ". Stains: "
                + "; ".join(
                    f"{_stain} ({', '.join(_channels)})"
                    for _stain, _channels in rule_stains(_rules).items()
                )
                + ". The thresholds only hold for images pre-processed with the "
                "file's `preprocessing` settings."
            ),
            mo.ui.table(_rule_rows, selection=None),
        ])
    mo.ui.tabs(_rule_tabs)
    return cell_type_labels, cell_type_rules_by_projection


@app.cell(hide_code=True)
def _():
    # Defaults as in data_cleaning.py (Cell type prevalence), with
    # cell_fate_s2/SOX17 excluded for data quality, so SOX17 comes from stain 1.
    _markers = ("SOX17", "SOX2", "TBXT", "FOXA2")
    DEFAULT_TUNING_RULES = CellTypeRules(
        channels={
            "SOX17": "cell_fate_s1/SOX17",
            "SOX2": "cell_fate_s1/SOX2",
            "TBXT": "cell_fate_s2/TBXT",
            "FOXA2": "cell_fate_s2/FOXA2",
        },
        thresholds={_marker: 0.3 for _marker in _markers},
        rules={
            "Notochord": [{"SOX17": "low", "SOX2": "any", "TBXT": "high", "FOXA2": "high"}],
            "Endoderm": [{"SOX17": "high", "SOX2": "any", "TBXT": "any", "FOXA2": "any"}],
            "Mesoderm": [
                {"SOX17": "low", "SOX2": "any", "TBXT": "high", "FOXA2": "low"},
                {"SOX17": "any", "SOX2": "high", "TBXT": "high", "FOXA2": "any"},
            ],
        },
        # Only used for the true-data estimates in the other sections.
        stains={
            "cell_fate_s1": {
                "SOX17": "cell_fate_s1/SOX17",
                "SOX2": "cell_fate_s1/SOX2",
                "TBXT": "cell_fate_s1/TBXT",
            },
            "cell_fate_s2": {"TBXT": "cell_fate_s2/TBXT", "FOXA2": "cell_fate_s2/FOXA2"},
        },
    )
    # A loaded file (or a reset) replaces the rules the controls start from.
    get_tuning_rules, set_tuning_rules = mo.state(DEFAULT_TUNING_RULES)
    get_cell_type_files_version, set_cell_type_files_version = mo.state(0)
    return (
        DEFAULT_TUNING_RULES,
        get_cell_type_files_version,
        get_tuning_rules,
        set_cell_type_files_version,
        set_tuning_rules,
    )


@app.cell(hide_code=True)
def _(get_tuning_rules):
    _rules = get_tuning_rules()
    _states = {"High": "high", "Low": "low", "Indifferent": "any"}
    _state_labels = {_value: _label for _label, _value in _states.items()}
    tuning_thresholds = mo.ui.dictionary({
        _marker: mo.ui.slider(
            0.0,
            1.0,
            step=0.01,
            value=round(min(max(_rules.thresholds[_marker], 0.0), 1.0), 2),
            show_value=True,
            label=_marker,
        )
        for _marker in _rules.markers
    })
    # {cell type: {clause number: {marker: dropdown}}}
    tuning_rules = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            str(_i): mo.ui.dictionary({
                _marker: mo.ui.dropdown(
                    options=_states, value=_state_labels[_clause.get(_marker, "any")]
                )
                for _marker in _rules.markers
            })
            for _i, _clause in enumerate(_clauses)
        })
        for _cell_type, _clauses in _rules.rules.items()
    })
    tuning_weights = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            str(_i): mo.ui.slider(0.0, 1.0, step=0.05, value=_weight, show_value=True)
            for _i, _weight in enumerate(_rules.weights_of(_cell_type))
        })
        for _cell_type in _rules.cell_types
    })
    return tuning_rules, tuning_thresholds, tuning_weights


@app.cell(hide_code=True)
def _(evaluation_results):
    # Models are shown by a short key (M1, M2, ...) in the plots.
    _model_ids = list(dict.fromkeys(_result["model_id"] for _result in (evaluation_results or ())))
    tuning_model_keys = {_model_id: f"M{_index + 1}" for _index, _model_id in enumerate(_model_ids)}
    _model_options = {
        f"{tuning_model_keys[_result['model_id']]} · {model_option_label(_result['registry_record'])}": _result["model_id"]
        for _result in (evaluation_results or ())
    }
    _time_options = list(dict.fromkeys(
        _time for _result in (evaluation_results or ()) for _time in _result["time_labels"]
    ))
    tuning_models = mo.ui.multiselect(
        options=_model_options,
        value=list(_model_options),
        label="Models (one column each)",
        full_width=True,
    )
    tuning_time = mo.ui.dropdown(
        options=_time_options,
        value=_time_options[-1] if _time_options else None,
        label="Timepoint",
    )
    tuning_show_markers = mo.ui.checkbox(
        value=False, label="Show each marker after thresholding (model and true data)"
    )
    tuning_path = mo.ui.text(
        value="Experiments/micropatterns/conf/cell_types/260726_cell_types_nca.yaml",
        label="Threshold file (relative to the repository root)",
        full_width=True,
    )
    save_tuning = mo.ui.run_button(label="Save thresholds and rules")
    load_tuning = mo.ui.run_button(label="Load thresholds and rules")
    reset_tuning = mo.ui.run_button(label="Reset to defaults")
    return (
        load_tuning,
        reset_tuning,
        save_tuning,
        tuning_model_keys,
        tuning_models,
        tuning_path,
        tuning_show_markers,
        tuning_time,
    )


@app.cell(hide_code=True)
def _(
    evaluation_results,
    get_tuning_rules,
    tuning_models,
    tuning_rules,
    tuning_thresholds,
    tuning_time,
    tuning_weights,
):
    _base = get_tuning_rules()
    tuning_selected = [
        _result for _result in (evaluation_results or ())
        if _result["model_id"] in tuning_models.value
    ]
    _names = list(dict.fromkeys(
        _name for _result in tuning_selected for _name in _result["channel_names"]
    ))
    _projections = sorted({_result["projection"] for _result in tuning_selected})

    def _channel(marker, channel):
        """The file's channel if the rollouts have it, else one of that marker."""
        if not _names or channel in _names:
            return channel
        return next((_n for _n in _names if _n.rsplit("/", 1)[-1] == marker), marker)

    _time = (tuning_time.value or "").rstrip("h")
    tuned_cell_type_rules = CellTypeRules(
        channels={_m: _channel(_m, _c) for _m, _c in _base.channels.items()},
        thresholds={_m: float(_t) for _m, _t in tuning_thresholds.value.items()},
        rules={
            _cell_type: [_clauses[_key] for _key in sorted(_clauses, key=int)]
            for _cell_type, _clauses in tuning_rules.value.items()
        },
        hour=int(_time) if _time.isdigit() else _base.hour,
        preprocessing={
            "projection": _projections[0] if len(_projections) == 1 else rules_projection(_base),
            "tuned_on": "NCA rollouts (every marker in every pixel)",
        },
        stains=_base.stains,
        clause_weights={
            _cell_type: [_weights[_key] for _key in sorted(_weights, key=int)]
            for _cell_type, _weights in tuning_weights.value.items()
        },
    )
    tuning_mixed_projections = len(_projections) > 1
    return tuned_cell_type_rules, tuning_mixed_projections, tuning_selected


@app.cell(hide_code=True)
def _(
    get_tuning_rules,
    load_tuning,
    reset_tuning,
    save_tuning,
    tuning_models,
    tuning_path,
    tuning_rules,
    tuning_show_markers,
    tuning_thresholds,
    tuning_time,
    tuning_weights,
):
    _markers = get_tuning_rules().markers
    _rule_grid = [
        mo.hstack(
            [mo.md("**cell type**")] + [mo.md(f"**{_m}**") for _m in _markers] + [mo.md("**weight**")],
            widths="equal",
        )
    ]
    for _cell_type, _clauses in tuning_rules.value.items():
        for _key in sorted(_clauses, key=int):
            _rule_grid.append(mo.hstack(
                [mo.md(_cell_type if _key == "0" else f"*or* {_cell_type}")]
                + [tuning_rules[_cell_type][_key][_m] for _m in _markers]
                + [tuning_weights[_cell_type][_key]],
                widths="equal",
            ))
    mo.vstack([
        mo.md(
            "## Cell-type threshold tuning on NCA rollouts\n\n"
            "- Uses the rollouts of **Run NODAL-KO evaluation**\n"
            "- Every marker is read from every model pixel, so no stain combining\n"
            "- True data is only shown one marker at a time\n"
            "- Defaults as in `data_cleaning.py`; saved files appear in the "
            "cell type file lists above"
        ),
        mo.hstack([tuning_models, tuning_time, tuning_show_markers], widths=[4, 1, 1]),
        mo.md("**Thresholds**"),
        mo.hstack([tuning_thresholds[_m] for _m in _markers], justify="start"),
        mo.md("**Rules** (rows of one cell type are joined by *or*; weight = how much a row counts)"),
        *_rule_grid,
        tuning_path,
        mo.hstack([save_tuning, load_tuning, reset_tuning], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(
    tuned_cell_type_rules,
    tuning_mixed_projections,
    tuning_model_keys,
    tuning_selected,
    tuning_show_markers,
    tuning_time,
):
    _rules = tuned_cell_type_rules
    _labels = label_names(_rules)
    _colours = {_label: cell_type_colour(_label) for _label in _labels}
    _models = list(dict.fromkeys(_result["model_id"] for _result in tuning_selected))
    _conditions = [
        _c for _c in CONDITION_LABELS if any(_r["condition"] == _c for _r in tuning_selected)
    ]
    mo.stop(
        not _models or tuning_time.value is None,
        mo.md("Run the evaluation above, then choose models."),
    )
    _by_key = {(_r["model_id"], _r["condition"]): _r for _r in tuning_selected}

    def _frame(result, source="prediction"):
        """Index of the chosen timepoint in ``result[source]``, or None."""
        if tuning_time.value not in result["time_labels"]:
            return None
        index = result["time_labels"].index(tuning_time.value)
        return index if index < result[source].shape[0] else None

    def _grid(rows, columns):
        figure, axes = plt.subplots(
            len(rows), len(columns),
            figsize=(1.5 * len(columns) + 0.8, 1.5 * len(rows) + 0.6),
            squeeze=False,
        )
        for _r, _row in enumerate(rows):
            for _c, _column in enumerate(columns):
                _axis = axes[_r, _c]
                _axis.set_xticks([])
                _axis.set_yticks([])
                for _spine in _axis.spines.values():
                    _spine.set_visible(False)
                if _r == 0:
                    _axis.set_title(_column, fontsize=9)
                if _c == 0:
                    _axis.set_ylabel(_row, fontsize=9)
        return figure, axes

    # Cell types: one row per condition, one column per model.
    _figure, _axes = _grid(
        [CONDITION_LABELS[_c] for _c in _conditions],
        [tuning_model_keys[_m] for _m in _models],
    )
    _prevalence_rows = []
    for _row, _condition in enumerate(_conditions):
        for _column, _model_id in enumerate(_models):
            _result = _by_key.get((_model_id, _condition))
            _index = None if _result is None else _frame(_result)
            if _index is None:
                continue
            _colony = _result["boundary"]
            _values = marker_values(
                _result["prediction"][_index], _result["channel_names"], _rules, axis=0
            )
            _shares = pixel_label_shares(_values, _rules, _colony)
            _axes[_row, _column].imshow(cell_type_rgba(_shares, _colours, _colony))
            _prevalence_rows.append({
                "model": tuning_model_keys[_model_id],
                "condition": CONDITION_LABELS[_condition],
                **{_label: float(_shares[_label][_colony].mean()) for _label in _labels},
            })
    _figure.legend(
        handles=[Patch(color=to_rgb(_colours[_label]), label=_label) for _label in _labels],
        loc="lower center",
        ncol=len(_labels),
        fontsize=8,
        frameon=False,
    )
    _figure.tight_layout(rect=(0, 0.1, 1, 1))

    _key_rows = [
        {
            "key": tuning_model_keys[_m],
            "model": record_label(_r["registry_record"]),
            "projection": _r["projection"],
            "downsample": _r["downsample"],
            "model_id": _m,
        }
        for _m, _r in {_r["model_id"]: _r for _r in tuning_selected}.items()
    ]
    _prevalence = pd.DataFrame(_prevalence_rows).round(3)
    _views = [
        *(
            [mo.callout(
                "The models use different projections; thresholds usually differ between them.",
                kind="warn",
            )]
            if tuning_mixed_projections else []
        ),
        mo.md(f"**Estimated cell types at {tuning_time.value}** (colours blended by clause weights)"),
        mo.hstack([_figure, mo.ui.table(_key_rows, selection=None)], justify="start"),
        mo.md("**Share of the colony**"),
        mo.ui.table(_prevalence, selection=None),
    ]

    if tuning_show_markers.value:
        # Per marker: true data (first model of each condition) and each model.
        _marker_tabs = {}
        _histograms = {_m: {"Model": [], "True": []} for _m in _rules.markers}
        for _marker in _rules.markers:
            _threshold = _rules.thresholds[_marker]
            _figure, _axes = _grid(
                [CONDITION_LABELS[_c] for _c in _conditions],
                ["True"] + [tuning_model_keys[_m] for _m in _models],
            )
            for _row, _condition in enumerate(_conditions):
                _sources = [
                    ("target", next(
                        (_by_key[(_m, _condition)] for _m in _models if (_m, _condition) in _by_key),
                        None,
                    ))
                ] + [("prediction", _by_key.get((_m, _condition))) for _m in _models]
                for _column, (_source, _result) in enumerate(_sources):
                    _index = None if _result is None else _frame(_result, _source)
                    if _index is None:
                        continue
                    _colony = _result["boundary"]
                    _image = marker_values(
                        _result[_source][_index], _result["channel_names"], _rules, axis=0
                    )[_marker]
                    _histograms[_marker]["True" if _source == "target" else "Model"].append(
                        _image[_colony]
                    )
                    _high = np.repeat((_image > _threshold)[..., None], 3, axis=-1).astype(float)
                    # Light grey: low, dark: high
                    _axes[_row, _column].imshow(masked_rgba(0.88 - 0.73 * _high, _colony))
            _figure.suptitle(
                f"{_rules.channels[_marker]} > {_threshold:g} (dark)", fontsize=9
            )
            _figure.tight_layout()
            _marker_tabs[_marker] = _figure
        # Marker values inside the colony, with the threshold marked.
        _figure, _axes = plt.subplots(
            1, len(_rules.markers), figsize=(3.0 * len(_rules.markers), 2.4), squeeze=False
        )
        for _axis, _marker in zip(_axes[0], _rules.markers):
            for _source, _colour in (("Model", "black"), ("True", "#1f77b4")):
                if _histograms[_marker][_source]:
                    _axis.hist(
                        np.concatenate(_histograms[_marker][_source]),
                        bins=50,
                        range=(0.0, 1.0),
                        density=True,
                        histtype="step",
                        color=_colour,
                        label=_source,
                    )
            _axis.axvline(_rules.thresholds[_marker], color="red", linestyle="--")
            _axis.set_title(_marker, fontsize=9)
            _axis.set_yticks([])
        _axes[0, 0].legend(fontsize=8)
        _figure.tight_layout()
        _views += [
            mo.md("**Each marker after thresholding**"),
            mo.ui.tabs(_marker_tabs),
            mo.md("**Marker values inside the colony** (red: threshold)"),
            _figure,
        ]
    mo.vstack(_views)
    return


@app.cell(hide_code=True)
def _(
    save_tuning,
    set_cell_type_files_version,
    tuned_cell_type_rules,
    tuning_path,
):
    mo.stop(not save_tuning.value)
    _path = Path(tuning_path.value).expanduser()
    _path = _path if _path.is_absolute() else repository_root / _path
    save_cell_type_rules(
        _path, tuned_cell_type_rules, made_with="Experiments/micropatterns/nodal_ko_eval.py"
    )
    set_cell_type_files_version(lambda _version: _version + 1)
    mo.md(f"Saved **{len(tuned_cell_type_rules.rules)}** cell types to `{_path}`.")
    return


@app.cell(hide_code=True)
def _(
    DEFAULT_TUNING_RULES,
    load_tuning,
    reset_tuning,
    set_tuning_rules,
    tuning_path,
):
    mo.stop(not (load_tuning.value or reset_tuning.value))
    if reset_tuning.value:
        set_tuning_rules(DEFAULT_TUNING_RULES)
        _message = mo.md("Thresholds and rules reset to the defaults.")
    else:
        _path = Path(tuning_path.value).expanduser()
        _path = _path if _path.is_absolute() else repository_root / _path
        _loaded = load_cell_type_rules(_path)
        set_tuning_rules(_loaded)
        _message = mo.md(
            f"Loaded **{len(_loaded.rules)}** cell types from `{_path}` "
            f"(tuned on {rules_projection(_loaded)})."
        )
    _message
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fine NODAL-KO timing sweep (model ensembles)

    Select one or more models and sweep the time at which recurrent NODAL is blocked.
    The default one-hour spacing creates a denser set of predictions than the
    measured 0h and 24h knockout conditions. Each rollout retains only the
    requested endpoint (48h by default), keeping the memory cost small.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(selection_records):
    _fine_model_options = {
        model_option_label(_record): _record["model_id"]
        for _record in selection_records
    }
    fine_ko_models = mo.ui.multiselect(
        options=_fine_model_options,
        value=list(_fine_model_options),
        label="Models",
        full_width=True,
    )
    fine_ko_start = mo.ui.number(0, 48, value=0, step=0.5, label="First KO time (h)")
    fine_ko_stop = mo.ui.number(0, 48, value=48, step=0.5, label="Last KO time (h)")
    fine_ko_interval = mo.ui.number(0.5, 24, value=1, step=0.5, label="Spacing (h)")
    fine_ko_endpoint = mo.ui.number(1, 48, value=48, step=1, label="Snapshot time (h)")
    fine_ko_replicates = mo.ui.text(
        "1",
        label="Replicates (for example: 1,2,4-6)",
        full_width=True,
    )
    fine_ko_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Rollout seed")
    fine_duplicate_runs = mo.ui.number(
        1,
        100,
        value=1,
        step=1,
        label="Duplicate runs",
    )
    fine_include_true_data = mo.ui.checkbox(
        value=True,
        label="Include true KO 0h, KO 24h, and baseline prevalence (per projection)",
    )
    run_fine_ko = mo.ui.run_button(label="Run fine KO timing sweep")
    mo.vstack([
        fine_ko_models,
        mo.hstack([fine_ko_start, fine_ko_stop, fine_ko_interval, fine_ko_endpoint]),
        mo.hstack([
            fine_ko_replicates,
            fine_ko_seed,
            fine_duplicate_runs,
            fine_include_true_data,
            run_fine_ko,
        ]),
    ])
    return (
        fine_duplicate_runs,
        fine_include_true_data,
        fine_ko_endpoint,
        fine_ko_interval,
        fine_ko_models,
        fine_ko_replicates,
        fine_ko_seed,
        fine_ko_start,
        fine_ko_stop,
        run_fine_ko,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    fine_duplicate_runs,
    fine_include_true_data,
    fine_ko_endpoint,
    fine_ko_interval,
    fine_ko_models,
    fine_ko_replicates,
    fine_ko_seed,
    fine_ko_start,
    fine_ko_stop,
    run_fine_ko,
    selection_records,
    store_root,
):
    if not run_fine_ko.value:
        fine_ko_results = None
        _fine_ko_status = mo.md("Configure the sweep, then run it.")
    else:
        if not fine_ko_models.value:
            raise ValueError("Choose at least one exported model from the selection.")
        if not data_root.value.strip():
            raise ValueError("Set Data root before running the timing sweep.")
        if fine_ko_stop.value < fine_ko_start.value:
            raise ValueError("Last KO time must be at or after first KO time.")
        if fine_ko_stop.value > fine_ko_endpoint.value:
            raise ValueError("KO times cannot be later than the snapshot time.")
        if fine_include_true_data.value and fine_ko_endpoint.value != 48:
            raise ValueError(
                "True KO and baseline prevalence overlays currently require a 48h snapshot."
            )

        _fine_registry = ModelRegistry(Path(store_root.value).expanduser())
        _fine_root = Path(data_root.value).expanduser()
        _fine_requested_hours = np.arange(
            float(fine_ko_start.value),
            float(fine_ko_stop.value) + 0.5 * float(fine_ko_interval.value),
            float(fine_ko_interval.value),
        )
        _fine_records_by_id = {
            _record["model_id"]: _record for _record in selection_records
        }
        _fine_replicates = parse_one_based_indices(fine_ko_replicates.value)
        _fine_jobs = [
            (_fine_model_index, _fine_model_id, _fine_replicate)
            for _fine_model_index, _fine_model_id in enumerate(fine_ko_models.value)
            for _fine_replicate in _fine_replicates
        ]
        _fine_model_cache = {}
        _fine_bundle_cache = {}
        _fine_bins_cache = {}
        # True data is loaded once per projection and replicate, with the
        # first model of that projection.
        _fine_true_loaded = set()
        _fine_model_results = []
        _fine_total_trajectories = (
            len(_fine_jobs)
            * int(fine_duplicate_runs.value)
            * len(_fine_requested_hours)
        )
        _fine_progress = tqdm(
            total=_fine_total_trajectories,
            desc="Running KO trajectories",
            unit="trajectory",
        )
        for _fine_job_index, (
            _fine_model_index,
            _fine_model_id,
            _fine_replicate,
        ) in enumerate(_fine_jobs):
            _fine_record = _fine_records_by_id[_fine_model_id]
            if _fine_model_id not in _fine_bundle_cache:
                _fine_bundle_cache[_fine_model_id] = _fine_registry.get(
                    _fine_model_id
                )
                _fine_model_cache[_fine_model_id] = _fine_bundle_cache[
                    _fine_model_id
                ].load_model(
                    key=jr.PRNGKey(int(fine_ko_seed.value)),
                    implementation="portable",
                )
                _fine_bins_cache[_fine_model_id] = load_training_histogram_bins(
                    _fine_bundle_cache[_fine_model_id], _fine_root
                )
            _fine_bundle = _fine_bundle_cache[_fine_model_id]
            _fine_model = _fine_model_cache[_fine_model_id]
            _fine_bins = _fine_bins_cache[_fine_model_id]
            _fine_data = load_condition_data(
                _fine_bundle,
                _fine_root,
                "baseline",
                _fine_replicate,
                _fine_bins,
            )
            _fine_projection = _fine_record["projection"]
            _fine_true_snapshots = None
            if (
                fine_include_true_data.value
                and (_fine_projection, _fine_replicate) not in _fine_true_loaded
            ):
                _fine_true_loaded.add((_fine_projection, _fine_replicate))
                _fine_measurement_hours = tuple(
                    float(_hour)
                    for _hour in _fine_bundle.config.data.micropattern.timesteps
                )
                if float(fine_ko_endpoint.value) not in _fine_measurement_hours:
                    raise ValueError(
                        f"No true measurement is available at {fine_ko_endpoint.value:g}h. "
                        f"Available times: {_fine_measurement_hours}."
                    )
                _fine_measurement_index = _fine_measurement_hours.index(
                    float(fine_ko_endpoint.value)
                )
                _fine_true_ko_0 = load_condition_data(
                    _fine_bundle,
                    _fine_root,
                    "ko_0h",
                    _fine_replicate,
                    _fine_bins,
                )
                _fine_true_ko_24 = load_condition_data(
                    _fine_bundle,
                    _fine_root,
                    "ko_24h",
                    _fine_replicate,
                    _fine_bins,
                )
                _fine_true_snapshots = {
                    0.0: _fine_true_ko_0["target"][_fine_measurement_index],
                    24.0: _fine_true_ko_24["target"][_fine_measurement_index],
                    float(fine_ko_endpoint.value): _fine_data["target"][
                        _fine_measurement_index
                    ],
                }
            _fine_initial = _fine_data["initial_state"]
            _fine_channel_count = int(_fine_model.N_CHANNELS)
            _fine_state = jnp.pad(
                jnp.asarray(_fine_initial),
                ((0, _fine_channel_count - _fine_initial.shape[0]), (0, 0), (0, 0)),
            )
            _fine_steps_per_12h = int(_fine_bundle.config.run.t)
            _fine_total_steps = int(round(
                float(fine_ko_endpoint.value) * _fine_steps_per_12h / 12.0
            ))
            _fine_observation_steps = jnp.asarray([_fine_total_steps], dtype=jnp.int32)
            _fine_boundary_mode = _fine_bundle.config.trainer.boundary_mode
            _fine_curriculum = tuple(
                _fine_bundle.config.data.knockout.curriculum or ("baseline",)
            )
            _fine_label = record_label(_fine_record)
            for _fine_run_index in range(int(fine_duplicate_runs.value)):
                _fine_run_seed = (
                    int(fine_ko_seed.value)
                    + _fine_job_index * int(fine_duplicate_runs.value)
                    + _fine_run_index
                )
                _fine_snapshots = []
                _fine_realised_hours = []
                for _fine_hour in _fine_requested_hours:
                    _fine_knockout_step = int(round(
                        float(_fine_hour) * _fine_steps_per_12h / 12.0
                    ))
                    _fine_selected = rollout_model_with_blocked_channel_sampled(
                        _fine_model,
                        _fine_state,
                        jnp.asarray(_fine_data["boundary"]),
                        _fine_boundary_mode,
                        jr.PRNGKey(_fine_run_seed),
                        _fine_total_steps,
                        _fine_data["nodal_channel"],
                        jnp.asarray(_fine_knockout_step, dtype=jnp.int32),
                        _fine_observation_steps,
                    )
                    _fine_snapshots.append(
                        np.asarray(_fine_selected)[
                            0, :len(_fine_data["state_channel_names"])
                        ]
                    )
                    _fine_progress.update(1)
                    _fine_realised_hours.append(
                        12.0 * _fine_knockout_step / _fine_steps_per_12h
                    )
                _fine_model_results.append({
                    "model_id": _fine_bundle.id,
                    "label": _fine_label,
                    "run_index": _fine_run_index,
                    "seed": _fine_run_seed,
                    "replicate": _fine_replicate,
                    "projection": _fine_projection,
                    "downsample": _fine_record["downsample"],
                    "groups": {
                        "projection": _fine_projection,
                        "projection_resolution": (
                            f"{_fine_projection} · ds{_fine_record['downsample']}"
                        ),
                        "projection_curriculum": (
                            f"{_fine_projection} · "
                            + " + ".join(map(str, _fine_curriculum))
                        ),
                        "curriculum": " + ".join(map(str, _fine_curriculum)),
                        "experiment": str(
                            _fine_record.get("experiment") or "Unlabelled"
                        ),
                        "family": str(
                            _fine_record.get("family") or "Unlabelled"
                        ),
                        "notes": str(_fine_record.get("notes") or "Unlabelled"),
                        "alias": str(_fine_record.get("alias") or "Unlabelled"),
                        "individual": _fine_label,
                        "all": "All selected models",
                    },
                    "requested_ko_hours": np.asarray(_fine_requested_hours),
                    "realised_ko_hours": np.asarray(_fine_realised_hours),
                    "snapshot_hour": float(fine_ko_endpoint.value),
                    "snapshots": np.stack(_fine_snapshots),
                    "channel_names": _fine_data["state_channel_names"],
                    "boundary": np.asarray(
                        _fine_data["boundary"]
                    ).squeeze().astype(bool),
                    "true_snapshots": (
                        _fine_true_snapshots if _fine_run_index == 0 else None
                    ),
                    "true_channel_names": (
                        _fine_data["channel_names"]
                        if _fine_true_snapshots is not None and _fine_run_index == 0
                        else None
                    ),
                })
        _fine_progress.close()
        fine_ko_results = tuple(_fine_model_results)
        _fine_ko_status = mo.callout(
            f"Completed {_fine_total_trajectories} knockout trajectories for "
            f"{len(fine_ko_models.value)} models × "
            f"{len(_fine_replicates)} replicates × "
            f"{int(fine_duplicate_runs.value)} seeded runs.",
            kind="success",
        )
    _fine_ko_status
    return (fine_ko_results,)


@app.cell(hide_code=True)
def _(fine_group_by, fine_ko_results):
    _fine_available_groups = list(dict.fromkeys(
        _result["groups"][fine_group_by.value]
        for _result in (fine_ko_results or ())
    ))
    fine_display_groups = mo.ui.multiselect(
        options=_fine_available_groups,
        value=_fine_available_groups,
        label="Displayed model groups",
        full_width=True,
    )
    fine_display_groups
    return (fine_display_groups,)


@app.cell(hide_code=True)
def _(cell_type_labels, fine_ko_results):
    _fine_channel_options = (
        [
            _name for _name in fine_ko_results[0]["channel_names"]
            if all(_name in _result["channel_names"] for _result in fine_ko_results)
        ] if fine_ko_results else []
    )
    fine_snapshot_channels = mo.ui.multiselect(
        options=_fine_channel_options,
        value=[
            _name for _name in ("TBXT", "SOX17", "SOX2", "FOXA2")
            if _name in _fine_channel_options
        ],
        label="Snapshot channels",
        full_width=True,
    )
    fine_snapshot_stride = mo.ui.number(
        1,
        48,
        value=1,
        step=1,
        label="Display every Nth swept KO time",
    )
    fine_group_by = mo.ui.dropdown(
        {
            "Projection": "projection",
            "Projection and resolution": "projection_resolution",
            "Projection and curriculum": "projection_curriculum",
            "Training curriculum": "curriculum",
            "Experiment": "experiment",
            "Family": "family",
            "Notes": "notes",
            "Alias": "alias",
            "Each model separately": "individual",
            "All models together": "all",
        },
        value="Projection",
        label="Group replicate models by",
        full_width=True,
    )
    _fine_true_projection_options = list(dict.fromkeys(
        _result["projection"]
        for _result in (fine_ko_results or ())
        if _result["true_snapshots"] is not None
    ))
    fine_true_projections = mo.ui.multiselect(
        options=_fine_true_projection_options,
        value=_fine_true_projection_options,
        label="True-data projections",
    )
    fine_uncertainty = mo.ui.dropdown(
        {
            "Standard deviation": "std",
            "Standard error of mean": "sem",
            "95% percentile interval": "percentile",
            "No uncertainty band": "none",
        },
        value="Standard deviation",
        label="Uncertainty",
        full_width=True,
    )
    fine_display_cell_types = mo.ui.multiselect(
        options=list(cell_type_labels),
        # Pixels matching several or no cell types are hidden unless chosen
        value=[_label for _label in cell_type_labels if _label not in (SEVERAL, OTHER)],
        label="Displayed cell types",
        full_width=True,
    )
    fine_scale_cell_maps_by_lmbr = mo.ui.checkbox(
        value=False,
        label="Scale cell-type snapshot brightness by LMBR",
    )
    fine_resolve_several = mo.ui.slider(
        0.0,
        1.0,
        value=0.0,
        step=0.05,
        show_value=True,
        label="Resolve multiple matches to the most likely type",
    )
    fine_cell_map_mode = mo.ui.dropdown(
        {"Mix colours by share": "mixed", "Sample one label per pixel": "sampled"},
        value="Mix colours by share",
        label="Snapshot colouring",
    )
    mo.vstack([
        mo.hstack([fine_snapshot_channels, fine_snapshot_stride]),
        fine_display_cell_types,
        mo.hstack([fine_resolve_several, fine_cell_map_mode]),
        mo.md(
            "- Resolve: a multiple match goes to the most likely type if it "
            "leads the next by at least 1 − value (0: keep *Several*, "
            "1: always). Used by every prevalence plot.\n"
            "- Sample: one label per pixel drawn from its shares (fixed seed)."
        ),
        fine_scale_cell_maps_by_lmbr,
        mo.hstack([fine_group_by, fine_uncertainty, fine_true_projections]),
    ])
    return (
        fine_cell_map_mode,
        fine_display_cell_types,
        fine_group_by,
        fine_resolve_several,
        fine_scale_cell_maps_by_lmbr,
        fine_snapshot_channels,
        fine_snapshot_stride,
        fine_true_projections,
        fine_uncertainty,
    )


@app.cell(hide_code=True)
def _(
    cell_type_labels,
    cell_type_rings,
    cell_type_rules_by_projection,
    fine_display_cell_types,
    fine_display_groups,
    fine_group_by,
    fine_ko_results,
    fine_resolve_several,
    fine_true_projections,
    fine_uncertainty,
):
    _fine_labels = cell_type_labels
    _fine_rings = int(cell_type_rings.value)
    # Markers each result needs, from the cell type file of its projection.
    _fine_missing = sorted({
        _marker
        for _result in (fine_ko_results or ())
        for _marker in cell_type_rules_by_projection[_result["projection"]].markers
        if _marker not in _result["channel_names"]
    })
    if fine_ko_results is None:
        fine_cell_type_prevalence = pd.DataFrame()
        fine_true_cell_type_prevalence = pd.DataFrame()
        fine_prevalence_export = pd.DataFrame()
        fine_prevalence_summary = pd.DataFrame()
        fine_cell_type_maps = ()
        fine_prevalence_figure = None
        _fine_prevalence_view = mo.md("")
    elif _fine_missing:
        fine_cell_type_prevalence = pd.DataFrame()
        fine_true_cell_type_prevalence = pd.DataFrame()
        fine_prevalence_export = pd.DataFrame()
        fine_prevalence_summary = pd.DataFrame()
        fine_cell_type_maps = ()
        fine_prevalence_figure = None
        _fine_prevalence_view = mo.callout(
            "Cell-type prevalence requires: " + ", ".join(_fine_missing),
            kind="warn",
        )
    else:
        _fine_true_sources = [
            _result for _result in fine_ko_results
            if _result["true_snapshots"] is not None
        ]
        _fine_prevalence_rows = []
        _fine_map_results = []
        for _fine_result in fine_ko_results:
            _fine_names_for_fates = _fine_result["channel_names"]
            _fine_fate_images = _fine_result["snapshots"]
            _fine_colony = _fine_result["boundary"]
            _fine_rules = cell_type_rules_by_projection[_fine_result["projection"]]
            _fine_result_maps = []
            for _fine_offset, _fine_hour in enumerate(
                _fine_result["requested_ko_hours"]
            ):
                _fine_image = _fine_fate_images[_fine_offset]
                # Per-pixel label shares for the maps, with the clause
                # weights used for the prevalence below.
                _fine_result_maps.append(pixel_label_shares(
                    marker_values(
                        _fine_image, _fine_names_for_fates, _fine_rules, axis=0
                    ),
                    _fine_rules,
                    _fine_colony,
                    resolve_several=fine_resolve_several.value,
                ))
                _fine_shares = model_label_shares(
                    _fine_image,
                    _fine_names_for_fates,
                    _fine_rules,
                    _fine_colony,
                    _fine_rings,
                    resolve_several=fine_resolve_several.value,
                )
                for _fine_cell_type, _fine_share in zip(_fine_labels, _fine_shares):
                    _fine_prevalence_rows.append({
                        "model_id": _fine_result["model_id"],
                        "model_label": _fine_result["label"],
                        "projection": _fine_result["projection"],
                        "downsample": _fine_result["downsample"],
                        "run_index": _fine_result.get("run_index", 0),
                        "seed": _fine_result.get("seed"),
                        "replicate": _fine_result.get("replicate"),
                        "group": _fine_result["groups"][fine_group_by.value],
                        "knockout_hour": float(_fine_hour),
                        "realised_knockout_hour": float(
                            _fine_result["realised_ko_hours"][_fine_offset]
                        ),
                        "cell_type": _fine_cell_type,
                        "relative_prevalence": float(_fine_share),
                    })
            _fine_map_results.append({
                "model_id": _fine_result["model_id"],
                "label": _fine_result["label"],
                "projection": _fine_result["projection"],
                "run_index": _fine_result.get("run_index", 0),
                "seed": _fine_result.get("seed"),
                "replicate": _fine_result.get("replicate"),
                "group": _fine_result["groups"][fine_group_by.value],
                "requested_ko_hours": np.asarray(
                    _fine_result["requested_ko_hours"]
                ),
                "boundary": np.asarray(_fine_colony),
                "cell_shares": tuple(_fine_result_maps),
                "lmbr": np.asarray(
                    _fine_fate_images[:, _fine_names_for_fates.index("LMBR")]
                ),
            })
        fine_cell_type_maps = tuple(_fine_map_results)
        fine_cell_type_prevalence = pd.DataFrame(_fine_prevalence_rows)
        _fine_export_index = [
            "model_id",
            "model_label",
            "projection",
            "downsample",
            "replicate",
            "seed",
            "group",
            "knockout_hour",
            "realised_knockout_hour",
        ]
        fine_prevalence_export = (
            fine_cell_type_prevalence.pivot(
                index=_fine_export_index,
                columns="cell_type",
                values="relative_prevalence",
            )
            .rename(columns=lambda cell_type: f"{cell_type.lower()}_prevalence")
            .reset_index()
            .rename(columns={
                "model_id": "model_realisation_id",
                "model_label": "model_realisation_label",
                "replicate": "initial_condition_replicate",
                "seed": "random_seed",
                "knockout_hour": "requested_knockout_hour",
            })
        )
        _fine_true_rows = []
        for _fine_true_source in _fine_true_sources:
            _fine_true_labels = {
                0.0: "True KO 0h",
                24.0: "True KO 24h",
                _fine_true_source["snapshot_hour"]: "True baseline",
            }
            for _fine_true_hour, _fine_true_image in (
                _fine_true_source["true_snapshots"].items()
            ):
                _fine_true_shares = data_label_shares(
                    _fine_true_image,
                    _fine_true_source["true_channel_names"],
                    cell_type_rules_by_projection[_fine_true_source["projection"]],
                    _fine_true_source["boundary"],
                    _fine_rings,
                    resolve_several=fine_resolve_several.value,
                )
                for _fine_cell_type, _fine_share in zip(_fine_labels, _fine_true_shares):
                    _fine_true_rows.append({
                        "projection": _fine_true_source["projection"],
                        "replicate": _fine_true_source["replicate"],
                        "knockout_hour": float(_fine_true_hour),
                        "condition": _fine_true_labels[_fine_true_hour],
                        "cell_type": _fine_cell_type,
                        "relative_prevalence": float(_fine_share),
                    })
        fine_true_cell_type_prevalence = pd.DataFrame(_fine_true_rows)
        _fine_summary_rows = []
        for (_fine_group_label, _fine_cell_type, _fine_hour), _fine_values in (
            fine_cell_type_prevalence.groupby(["group", "cell_type", "knockout_hour"])
        ):
            _fine_samples = _fine_values["relative_prevalence"].to_numpy()
            _fine_mean = float(np.mean(_fine_samples))
            if fine_uncertainty.value == "std":
                _fine_error = float(np.std(_fine_samples, ddof=1)) if len(_fine_samples) > 1 else 0.0
                _fine_lower, _fine_upper = _fine_mean - _fine_error, _fine_mean + _fine_error
            elif fine_uncertainty.value == "sem":
                _fine_error = (
                    float(np.std(_fine_samples, ddof=1) / np.sqrt(len(_fine_samples)))
                    if len(_fine_samples) > 1 else 0.0
                )
                _fine_lower, _fine_upper = _fine_mean - _fine_error, _fine_mean + _fine_error
            elif fine_uncertainty.value == "percentile":
                _fine_lower, _fine_upper = np.percentile(_fine_samples, [2.5, 97.5])
            else:
                _fine_lower = _fine_upper = _fine_mean
            _fine_summary_rows.append({
                "group": _fine_group_label,
                "cell_type": _fine_cell_type,
                "knockout_hour": float(_fine_hour),
                "mean_prevalence": _fine_mean,
                "lower": max(0.0, float(_fine_lower)),
                "upper": min(1.0, float(_fine_upper)),
                "sample_count": int(len(_fine_samples)),
            })
        fine_prevalence_summary = pd.DataFrame(_fine_summary_rows)
        fine_prevalence_figure, _fine_prevalence_axis = plt.subplots(
            figsize=(11, 5), dpi=150
        )
        _fine_group_styles = ("-", "--", ":", "-.")
        _fine_group_names = list(dict.fromkeys(fine_prevalence_summary["group"]))
        _fine_style_by_group = {
            _fine_group_name: _fine_group_styles[
                _fine_group_index % len(_fine_group_styles)
            ]
            for _fine_group_index, _fine_group_name in enumerate(_fine_group_names)
        }
        _fine_display_summary = fine_prevalence_summary[
            fine_prevalence_summary["cell_type"].isin(fine_display_cell_types.value)
            & fine_prevalence_summary["group"].isin(fine_display_groups.value)
        ]
        # The group goes in the legend only when several groups are shown
        _fine_several_groups = _fine_display_summary["group"].nunique() > 1
        for (_fine_group_label, _fine_cell_type), _fine_curve in (
            _fine_display_summary.groupby(["group", "cell_type"], sort=False)
        ):
            _fine_curve = _fine_curve.sort_values("knockout_hour")
            _fine_line = _fine_prevalence_axis.plot(
                _fine_curve["knockout_hour"],
                _fine_curve["mean_prevalence"],
                _fine_style_by_group[_fine_group_label],
                color=cell_type_colour(_fine_cell_type),
                label=(
                    f"{_fine_group_label} · {_fine_cell_type}" if _fine_several_groups
                    else _fine_cell_type
                ),
            )
            if (
                fine_uncertainty.value != "none"
                and _fine_curve["sample_count"].max() > 1
            ):
                _fine_prevalence_axis.fill_between(
                    _fine_curve["knockout_hour"],
                    _fine_curve["lower"],
                    _fine_curve["upper"],
                    color=_fine_line[0].get_color(),
                    alpha=0.18,
                )
        _fine_true_display = (
            fine_true_cell_type_prevalence[
                fine_true_cell_type_prevalence["cell_type"].isin(
                    fine_display_cell_types.value
                )
                & fine_true_cell_type_prevalence["projection"].isin(
                    fine_true_projections.value
                )
            ]
            if not fine_true_cell_type_prevalence.empty
            else fine_true_cell_type_prevalence
        )
        if not _fine_true_display.empty:
            _fine_true_markers = {
                "True KO 0h": "X",
                "True KO 24h": "D",
                "True baseline": "o",
            }
            # With several projections, the marker edge shows the projection.
            _fine_true_shown = list(dict.fromkeys(_fine_true_display["projection"]))
            _fine_several_true = len(_fine_true_shown) > 1
            for _, _fine_true_row in _fine_true_display.iterrows():
                _fine_prevalence_axis.scatter(
                    _fine_true_row["knockout_hour"],
                    _fine_true_row["relative_prevalence"],
                    color=cell_type_colour(_fine_true_row["cell_type"]),
                    marker=_fine_true_markers[_fine_true_row["condition"]],
                    s=65,
                    edgecolor=(
                        PROJECTION_COLOURS.get(_fine_true_row["projection"], "black")
                        if _fine_several_true else "black"
                    ),
                    linewidth=1.2 if _fine_several_true else 0.5,
                    zorder=5,
                    label="_nolegend_",
                )
            for _fine_true_condition, _fine_true_marker in _fine_true_markers.items():
                _fine_prevalence_axis.scatter(
                    [],
                    [],
                    color="gray",
                    marker=_fine_true_marker,
                    s=65,
                    edgecolor="black",
                    linewidth=0.5,
                    label=_fine_true_condition,
                )
            if _fine_several_true:
                for _fine_true_projection in _fine_true_shown:
                    _fine_prevalence_axis.scatter(
                        [],
                        [],
                        color="white",
                        marker="s",
                        s=65,
                        edgecolor=PROJECTION_COLOURS.get(_fine_true_projection, "black"),
                        linewidth=1.2,
                        label=f"True data: {_fine_true_projection}",
                    )
        _fine_ko_ticks, _fine_ko_tick_labels = ko_time_ticks(
            fine_ko_results[0]["snapshot_hour"]
        )
        _fine_prevalence_axis.set(
            xlabel="NODAL knockout time (h)",
            ylabel="Fraction of colony pixels",
            xticks=_fine_ko_ticks,
            xticklabels=_fine_ko_tick_labels,
            title=(
                f"Predicted cell-type prevalence at "
                f"{fine_ko_results[0]['snapshot_hour']:g}h"
            ),
        )
        if _fine_display_summary.empty and _fine_true_display.empty:
            _fine_prevalence_axis.text(
                0.5,
                0.5,
                "Select at least one model group or true-data cell type",
                ha="center",
                va="center",
                transform=_fine_prevalence_axis.transAxes,
            )
        else:
            _fine_prevalence_axis.legend(fontsize="small", ncol=2)
        fine_prevalence_figure.tight_layout()
        _fine_prevalence_view = mo.vstack([
            fine_prevalence_figure,
            mo.md(
                "Bands show the selected uncertainty across distinct models, "
                "initial-condition replicates, and seeded runs in each group. "
                "Groups with one total sample have no uncertainty band. Each true "
                "data replicate is plotted separately. Model and true prevalences "
                "are both estimated from the stains of the cell type file (see "
                "Cell types above)."
            ),
            mo.ui.table(fine_prevalence_summary, selection=None, page_size=15),
            mo.ui.table(fine_prevalence_export, selection=None, page_size=15),
            mo.ui.table(
                fine_true_cell_type_prevalence, selection=None, page_size=15
            ),
        ])
    _fine_prevalence_view
    plt.show()
    return (
        fine_cell_type_maps,
        fine_prevalence_export,
        fine_prevalence_figure,
        fine_prevalence_summary,
        fine_true_cell_type_prevalence,
    )


@app.cell(hide_code=True)
def _(
    fine_prevalence_export,
    fine_prevalence_summary,
    fine_true_cell_type_prevalence,
):
    _fine_export_ready = not fine_prevalence_export.empty
    _fine_export_status = (
        "Downloads are ready."
        if _fine_export_ready
        else "Run the fine NODAL-KO timing sweep to enable these downloads."
    )
    mo.vstack([
        mo.md("### CSV exports"),
        mo.md(_fine_export_status),
        mo.hstack([
            mo.download(
                lambda: fine_prevalence_export.to_csv(index=False).encode("utf-8"),
                filename="nodal_ko_cell_type_prevalence.csv",
                mimetype="text/csv",
                disabled=not _fine_export_ready,
                label="Download model prevalence CSV",
            ),
            mo.download(
                lambda: fine_prevalence_summary.to_csv(index=False).encode("utf-8"),
                filename="nodal_ko_cell_type_prevalence_summary.csv",
                mimetype="text/csv",
                disabled=fine_prevalence_summary.empty,
                label="Download grouped summary CSV",
            ),
            mo.download(
                lambda: fine_true_cell_type_prevalence.to_csv(index=False).encode(
                    "utf-8"
                ),
                filename="nodal_ko_true_cell_type_prevalence.csv",
                mimetype="text/csv",
                disabled=fine_true_cell_type_prevalence.empty,
                label="Download true-data prevalence CSV",
            ),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _():
    save_overview_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_overview_panels = mo.ui.run_button(label="Save for overview figure")
    mo.vstack([
        mo.md(
            "### Overview figure\n\n"
            "- Saves the prevalence curves (CSV, plus PNG/PDF/optional SVG as shown) and the "
            "cell-type snapshots (shares of the first shown model run, every Nth KO "
            "time as displayed) to `figures/overview/`\n"
            "- Read by `overview_figure.py`"
        ),
        mo.hstack([save_overview_svg, save_overview_panels], justify="start"),
    ])
    return save_overview_panels, save_overview_svg


@app.cell(hide_code=True)
def _(
    fine_cell_map_export,
    fine_cell_map_figures,
    fine_prevalence_export,
    fine_prevalence_figure,
    fine_prevalence_summary,
    fine_true_cell_type_prevalence,
    save_overview_panels,
    save_overview_svg,
):
    mo.stop(not save_overview_panels.value)
    mo.stop(
        fine_prevalence_summary.empty,
        mo.callout("Run the fine NODAL-KO timing sweep first.", kind="warn"),
    )
    _directory = repository_root / "figures" / "overview"
    _directory.mkdir(parents=True, exist_ok=True)
    _saved = []
    for _name, _table in (
        ("nodal_ko_prevalence_summary.csv", fine_prevalence_summary),
        ("nodal_ko_true_prevalence.csv", fine_true_cell_type_prevalence),
        ("nodal_ko_prevalence_samples.csv", fine_prevalence_export),
    ):
        _table.to_csv(_directory / _name, index=False)
        _saved.append(_name)
    _svg = ("svg",) if save_overview_svg.value else ()
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        if fine_prevalence_figure is not None:
            for _suffix in ("png", "pdf") + _svg:
                fine_prevalence_figure.savefig(
                    _directory / f"nodal_ko_prevalence.{_suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white",
                )
                _saved.append(f"nodal_ko_prevalence.{_suffix}")
        for _index, _figure in enumerate(fine_cell_map_figures):
            for _suffix in ("png",) + _svg:
                _figure.savefig(
                    _directory / f"nodal_ko_cell_maps_{_index}.{_suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white",
                )
                _saved.append(f"nodal_ko_cell_maps_{_index}.{_suffix}")
    if fine_cell_map_export is not None:
        np.savez_compressed(_directory / "nodal_ko_cell_maps.npz", **fine_cell_map_export)
        _saved.append("nodal_ko_cell_maps.npz")
    mo.md(f"Saved to `{_directory}`:\n\n" + "\n".join(f"- `{_n}`" for _n in _saved))
    return


@app.cell(hide_code=True)
def _(
    fine_cell_map_mode,
    fine_cell_type_maps,
    fine_display_cell_types,
    fine_display_groups,
    fine_ko_results,
    fine_scale_cell_maps_by_lmbr,
    fine_snapshot_stride,
):
    _fine_map_rng = np.random.default_rng(0)
    fine_cell_map_figures = []
    # Per-pixel cell-type shares of the first shown model run, for overview_figure.py
    fine_cell_map_export = None
    if fine_ko_results is None or not fine_cell_type_maps:
        _fine_cell_map_view = mo.md("")
    else:
        _fine_map_colors = {
            _label: np.asarray(to_rgb(cell_type_colour(_label)))
            for _label in fine_display_cell_types.value
        }
        _fine_selected_map_types = tuple(fine_display_cell_types.value)
        _fine_visible_map_results = [
            _fine_map_result
            for _fine_map_result in fine_cell_type_maps
            if _fine_map_result["group"] in fine_display_groups.value
        ]
        for _fine_map_result in _fine_visible_map_results:
            _fine_map_indices = list(range(
                0,
                len(_fine_map_result["cell_shares"]),
                int(fine_snapshot_stride.value),
            ))
            if _fine_map_indices[-1] != len(_fine_map_result["cell_shares"]) - 1:
                _fine_map_indices.append(len(_fine_map_result["cell_shares"]) - 1)
            _fine_rgb_maps = []
            _fine_export_shares = []
            for _fine_map_index in _fine_map_indices:
                _fine_rgb_map = np.zeros(
                    (*_fine_map_result["boundary"].shape, 3), dtype=float
                )
                # Colours are mixed by each pixel's label shares, or one
                # label per pixel is drawn from them.
                _fine_pixel_shares = _fine_map_result["cell_shares"][_fine_map_index]
                if fine_cell_map_mode.value == "sampled":
                    _fine_pixel_shares = sample_pixel_labels(
                        _fine_pixel_shares, _fine_map_rng
                    )
                _fine_export_shares.append(_fine_pixel_shares)
                for _fine_map_cell_type in _fine_selected_map_types:
                    _fine_rgb_map += (
                        _fine_pixel_shares[_fine_map_cell_type][..., None]
                        * _fine_map_colors[_fine_map_cell_type]
                    )
                if fine_scale_cell_maps_by_lmbr.value:
                    _fine_lmbr_brightness = np.clip(
                        _fine_map_result["lmbr"][_fine_map_index], 0.0, 1.0
                    )
                    _fine_rgb_map *= _fine_lmbr_brightness[..., None]
                _fine_rgb_maps.append(_fine_rgb_map)
            _fine_cell_map_figure, _fine_cell_map_axes = plt.subplots(
                1,
                len(_fine_rgb_maps),
                figsize=(min(22, max(5, 2.0 * len(_fine_rgb_maps))), 2.6),
                squeeze=False,
            )
            for _fine_cell_map_axis, _fine_rgb_map, _fine_map_index in zip(
                _fine_cell_map_axes[0], _fine_rgb_maps, _fine_map_indices
            ):
                _fine_cell_map_axis.imshow(_fine_rgb_map, vmin=0.0, vmax=1.0)
                _fine_cell_map_axis.set_title(ko_time_label(
                    _fine_map_result["requested_ko_hours"][_fine_map_index],
                    fine_ko_results[0]["snapshot_hour"],
                    prefix="KO ",
                ))
                _fine_cell_map_axis.set_axis_off()
            # _fine_cell_map_figure.suptitle(
            #     f"{_fine_map_result['label']} · cell identity at "
            #     f"{fine_ko_results[0]['snapshot_hour']:g}h"
            # )
            _fine_cell_map_figure.suptitle(
                f"{_fine_map_result['projection']} · cell identity at "
                f"{fine_ko_results[0]['snapshot_hour']:g}h"
            )
            _fine_cell_map_figure.tight_layout()
            fine_cell_map_figures.append(_fine_cell_map_figure)
            if fine_cell_map_export is None:
                _fine_export_labels = list(_fine_export_shares[0])
                fine_cell_map_export = {
                    "hours": np.asarray(_fine_map_result["requested_ko_hours"])[_fine_map_indices],
                    # [KO time, cell type, H, W]
                    "shares": np.stack([
                        np.stack([np.asarray(_shares[_label]) for _label in _fine_export_labels])
                        for _shares in _fine_export_shares
                    ]).astype(np.float16),
                    "cell_types": np.asarray(_fine_export_labels),
                    "mask": np.asarray(_fine_map_result["boundary"]),
                    "snapshot_hour": np.asarray(fine_ko_results[0]["snapshot_hour"]),
                    "label": np.asarray(f"{_fine_map_result['projection']} · {_fine_map_result['label']}"),
                }

        _fine_map_legend = " · ".join(
            f"<span style='color: rgb({int(255 * _fine_map_colors[_fine_type][0])}, "
            f"{int(255 * _fine_map_colors[_fine_type][1])}, "
            f"{int(255 * _fine_map_colors[_fine_type][2])})'>■</span> {_fine_type}"
            for _fine_type in _fine_selected_map_types
        )
        if not _fine_visible_map_results:
            _fine_cell_map_view = mo.md("Select at least one model group.")
        elif not _fine_selected_map_types:
            _fine_cell_map_view = mo.md("Select at least one displayed cell type.")
        else:
            _fine_cell_map_view = mo.vstack([
                mo.md("### Final cell-type snapshots"),
                mo.md(
                    "Brightness is scaled by the LMBR nuclear marker."
                    if fine_scale_cell_maps_by_lmbr.value
                    else (
                        "One label per pixel, drawn from its cell-type shares."
                        if fine_cell_map_mode.value == "sampled"
                        else "Colours are mixed by each pixel's cell-type shares."
                    )
                ),
                mo.md(_fine_map_legend),
                *fine_cell_map_figures,
            ])
    _fine_cell_map_view
    return fine_cell_map_export, fine_cell_map_figures


@app.cell(hide_code=True)
def _(
    fine_display_groups,
    fine_group_by,
    fine_ko_results,
    fine_snapshot_channels,
    fine_snapshot_stride,
):
    if fine_ko_results is None or not fine_snapshot_channels.value:
        _fine_snapshot_view = mo.md("")
    else:
        _fine_snapshot_figures = []
        for _fine_result in fine_ko_results:
            if _fine_result["groups"][fine_group_by.value] not in fine_display_groups.value:
                continue
            _fine_names = _fine_result["channel_names"]
            _fine_indices = [
                _fine_names.index(_name) for _name in fine_snapshot_channels.value
            ]
            _fine_all_images = _fine_result["snapshots"]
            _fine_display_indices = list(range(
                0,
                _fine_all_images.shape[0],
                int(fine_snapshot_stride.value),
            ))
            if _fine_display_indices[-1] != _fine_all_images.shape[0] - 1:
                _fine_display_indices.append(_fine_all_images.shape[0] - 1)
            _fine_images = _fine_all_images[_fine_display_indices]
            _fine_rows = len(_fine_indices)
            _fine_columns = _fine_images.shape[0]
            _fine_tile_rows = []
            for _fine_row, _fine_index in enumerate(_fine_indices):
                _fine_values = _fine_images[:, _fine_index]
                _fine_vmin = float(np.nanmin(_fine_values))
                _fine_vmax = float(np.nanmax(_fine_values))
                if _fine_vmax <= _fine_vmin:
                    _fine_vmax = _fine_vmin + 1.0
                _fine_normalised_tiles = []
                for _fine_column in range(_fine_columns):
                    _fine_normalised_tiles.append(
                        np.clip(
                            (_fine_values[_fine_column] - _fine_vmin)
                            / (_fine_vmax - _fine_vmin),
                            0.0,
                            1.0,
                        )
                    )
                _fine_tile_rows.append(
                    np.concatenate(_fine_normalised_tiles, axis=1)
                )
            _fine_mosaic = np.concatenate(_fine_tile_rows, axis=0)
            _fine_tile_height, _fine_tile_width = _fine_normalised_tiles[0].shape
            _fine_figure, _fine_axis = plt.subplots(
                figsize=(min(22, max(8, 0.75 * _fine_columns)), max(3, 1.5 * _fine_rows))
            )
            _fine_axis.imshow(_fine_mosaic, cmap="gray", vmin=0.0, vmax=1.0)
            _fine_tick_stride = max(1, int(np.ceil(_fine_columns / 12)))
            _fine_tick_indices = list(range(0, _fine_columns, _fine_tick_stride))
            if _fine_tick_indices[-1] != _fine_columns - 1:
                _fine_tick_indices.append(_fine_columns - 1)
            _fine_axis.set_xticks(
                (np.asarray(_fine_tick_indices) + 0.5) * _fine_tile_width - 0.5,
                [
                    ko_time_label(
                        _fine_result["requested_ko_hours"][_index],
                        _fine_result["snapshot_hour"],
                        prefix="KO ",
                    )
                    for _index in (
                        _fine_display_indices[_display_index]
                        for _display_index in _fine_tick_indices
                    )
                ],
                rotation=0,
                ha="center",
            )
            _fine_axis.set_yticks(
                (np.arange(_fine_rows) + 0.5) * _fine_tile_height - 0.5,
                [_fine_names[_index] for _index in _fine_indices],
            )
            # _fine_axis.set_title(
            #     f"{_fine_result['label']} · predicted at "
            #     f"{_fine_result['snapshot_hour']:g}h"
            # )
            _fine_axis.set_title(
                f"{_fine_result['projection']} · predicted at "
                f"{_fine_result['snapshot_hour']:g}h"
            )
            _fine_axis.tick_params(length=0)
            _fine_figure.tight_layout()
            _fine_snapshot_figures.append(_fine_figure)
        _fine_snapshot_view = (
            mo.vstack(_fine_snapshot_figures)
            if _fine_snapshot_figures
            else mo.md("Select at least one model group.")
        )
    _fine_snapshot_view
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cell density × NODAL-KO time

    - Circular colony (mean area of the chosen replicates' colonies) with a synthetic 0h state
    - Whole cells cut out of each replicate's 0h colony and placed at random, as in inverse design; the density sets how many are kept
    - Each run places its cells once: every density and KO time of a run uses the same cells and rollout seed
    - Bands: spread over models, replicates and duplicate runs in each group
    - Cell types as above, with *Resolve multiple matches* from the fine KO sweep
    """)
    return


@app.cell(hide_code=True)
def _():
    dko_density_unit = mo.ui.dropdown(
        options={
            "Cells per mm²": "per_mm2",
            "Total cells in the colony": "per_colony",
            "Pixel coverage": "share",
        },
        value="Cells per mm²",
        label="Sweep density as",
    )
    return (dko_density_unit,)


@app.cell(hide_code=True)
def _(dko_density_unit):
    # Cells: nuclei counted in the raw 0h images set the scale (cell_counts.py)
    _low, _high, _start, _stop, _step = {
        "per_mm2": (0, 50000, 1000, 8000, 50),
        "per_colony": (0, 20000, 200, 1600, 10),
        "share": (0.02, 0.98, 0.2, 0.8, 0.01),
    }[dko_density_unit.value]
    dko_density_start = mo.ui.number(_low, _high, value=_start, step=_step, label="Lowest density")
    dko_density_stop = mo.ui.number(_low, _high, value=_stop, step=_step, label="Highest density")
    return dko_density_start, dko_density_stop


@app.cell(hide_code=True)
def _(selection_records):
    # Created apart from the layout below: a cell that shows the density
    # controls reruns when they change, which would reset these choices
    _dko_model_options = {
        model_option_label(_record): _record["model_id"]
        for _record in selection_records
    }
    dko_models = mo.ui.multiselect(
        options=_dko_model_options,
        value=list(_dko_model_options),
        label="Models",
        full_width=True,
    )
    dko_density_count = mo.ui.number(2, 50, value=7, step=1, label="Densities")
    dko_ko_start = mo.ui.number(0, 48, value=0, step=0.5, label="First KO time (h)")
    dko_ko_stop = mo.ui.number(0, 48, value=48, step=0.5, label="Last KO time (h)")
    dko_ko_interval = mo.ui.number(0.5, 24, value=6, step=0.5, label="KO spacing (h)")
    dko_endpoint = mo.ui.number(1, 48, value=48, step=1, label="Snapshot time (h)")
    dko_replicates = mo.ui.text("1", label="Replicates (for example: 1,2,4-6)", full_width=True)
    dko_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Rollout seed")
    dko_duplicate_runs = mo.ui.number(1, 100, value=3, step=1, label="Duplicate runs")
    # Cell placement for each projection (cells sections; tuned in synthetic_textured_ic_explorer.py)
    dko_texture_file = mo.ui.text(
        value=repository_relative_path(DEFAULT_TEXTURE_SETTINGS_FILE),
        label="Texture settings file",
        full_width=True,
    )
    # Rollouts sent to the GPU together (lower it if memory runs out)
    dko_max_batch = mo.ui.number(1, 4096, value=64, step=1, label="Rollouts per GPU batch")
    # References rolled out beside the sweep, at the same KO times and seeds
    dko_include_true_data = mo.ui.switch(value=False, label="True data initial condition")
    dko_include_measured = mo.ui.switch(value=False, label="Synthetic at the measured density")
    run_dko_sweep = mo.ui.run_button(label="Run density × KO sweep")
    return (
        dko_density_count,
        dko_duplicate_runs,
        dko_endpoint,
        dko_include_measured,
        dko_include_true_data,
        dko_ko_interval,
        dko_ko_start,
        dko_ko_stop,
        dko_max_batch,
        dko_models,
        dko_replicates,
        dko_seed,
        dko_texture_file,
        run_dko_sweep,
    )


@app.cell(hide_code=True)
def _(
    dko_density_count,
    dko_density_start,
    dko_density_stop,
    dko_density_unit,
    dko_duplicate_runs,
    dko_endpoint,
    dko_include_measured,
    dko_include_true_data,
    dko_ko_interval,
    dko_ko_start,
    dko_ko_stop,
    dko_max_batch,
    dko_models,
    dko_replicates,
    dko_seed,
    dko_texture_file,
    run_dko_sweep,
):
    mo.vstack([
        dko_models,
        mo.hstack([dko_density_unit, dko_density_start, dko_density_stop, dko_density_count]),
        mo.hstack([dko_ko_start, dko_ko_stop, dko_ko_interval, dko_endpoint]),
        mo.accordion({
            "Cell placement": mo.vstack([
                dko_texture_file,
                mo.md("- The `cells` settings of each model's projection; tune them in `synthetic_textured_ic_explorer.py`"),
            ]),
            "Compute": dko_max_batch,
            "References": mo.vstack([
                mo.md(
                    "- True data: the 0h control colony of each replicate above, in its own outline\n"
                    "- Synthetic at the measured density: the sweep's cells at the measured density "
                    "(mean over the replicates)\n"
                    "- Both use the same KO times and run seeds as the sweep"
                ),
                mo.hstack([dko_include_true_data, dko_include_measured], justify="start"),
            ]),
        }),
        mo.hstack([dko_replicates, dko_seed, dko_duplicate_runs, run_dko_sweep]),
    ])
    return


@app.cell(hide_code=True)
def _():
    # The latest results, from a run or a loaded file
    get_dko_results, set_dko_results = mo.state(None)
    return get_dko_results, set_dko_results


@app.cell(hide_code=True)
def _(
    cell_type_files,
    cell_type_rings,
    data_root,
    dko_density_count,
    dko_density_start,
    dko_density_stop,
    dko_density_unit,
    dko_duplicate_runs,
    dko_endpoint,
    dko_include_measured,
    dko_include_true_data,
    dko_ko_interval,
    dko_ko_start,
    dko_ko_stop,
    dko_max_batch,
    dko_models,
    dko_replicates,
    dko_seed,
    dko_texture_file,
    selection_records,
    store_root,
):
    # Everything the sweep reads, for a run here or a remote job
    dko_settings = DensityKoSettings(
        data_root=data_root.value.strip(),
        store_root=store_root.value.strip(),
        cell_type_files={
            _projection: repository_relative_path(_path)
            for _projection, _path in cell_type_files.value.items() if _path is not None
        },
        rings=int(cell_type_rings.value),
        density_unit=dko_density_unit.value,
        density_start=float(dko_density_start.value),
        density_stop=float(dko_density_stop.value),
        density_count=int(dko_density_count.value),
        ko_start=float(dko_ko_start.value),
        ko_stop=float(dko_ko_stop.value),
        ko_interval=float(dko_ko_interval.value),
        snapshot_hour=float(dko_endpoint.value),
        replicates=parse_one_based_indices(dko_replicates.value),
        seed=int(dko_seed.value),
        duplicate_runs=int(dko_duplicate_runs.value),
        cells={
            _projection: load_cell_settings(repository_path(dko_texture_file.value.strip()), _projection).to_dict()
            for _projection, _path in cell_type_files.value.items() if _path is not None
        },
        max_batch=int(dko_max_batch.value),
        include_true_data=bool(dko_include_true_data.value),
        include_measured_density=bool(dko_include_measured.value),
    )
    _records = {_record["model_id"]: _record for _record in selection_records}
    dko_records = tuple(_records[_model_id] for _model_id in dko_models.value)
    return dko_records, dko_settings


@app.cell(hide_code=True)
def _(dko_records, dko_settings, run_dko_sweep, set_dko_results):
    if not run_dko_sweep.value:
        _dko_status = mo.md("Configure the sweep, then run it here, export it as a remote job, or load saved results below.")
    else:
        if not dko_records:
            raise ValueError("Choose at least one exported model from the selection.")
        with tqdm(
            total=dko_settings.trajectories(len(dko_records)),
            desc="Running density × KO trajectories",
            unit="trajectory",
        ) as _dko_progress:
            _dko_results, _dko_warnings = run_density_ko_sweep(
                dko_settings, dko_records, progress=_dko_progress.update
            )
        set_dko_results(_dko_results)
        _dko_status = mo.vstack([
            mo.callout(
                f"Completed {len(dko_settings.densities())} densities × "
                f"{len(dko_settings.ko_hours())} KO times for "
                f"{len(dko_records)} models × {len(dko_settings.replicates)} replicates × "
                f"{dko_settings.duplicate_runs} runs.",
                kind="success",
            ),
            *(
                [mo.callout(
                    "Even the most cell layers (10) cannot hold the highest density, "
                    "so it is cut to what they hold (or nuclei could not be counted):\n\n"
                    + "\n".join(f"- {_warning}" for _warning in _dko_warnings),
                    kind="warn",
                )]
                if _dko_warnings else []
            ),
        ])
    _dko_status
    return


@app.cell(hide_code=True)
def _():
    dko_job_name = mo.ui.text("density_ko", label="Job name", full_width=True)
    dko_job_split = mo.ui.dropdown(
        options={"One per model": "model", "One per model and replicate": "replicate"},
        value="One per model",
        label="Parts (one pod each)",
    )
    dko_job_output = mo.ui.text(
        "/workspace/writeable/output/nodal_ko_eval",
        label="Results folder on the cluster (the job name is added)",
        full_width=True,
    )
    export_dko_job = mo.ui.run_button(label="Export remote job")
    mo.vstack([
        mo.md(
            "### Run the sweep remotely\n\n"
            "- Writes a job manifest with the settings above and the chosen models\n"
            "- Each part writes `part_<i>.npz`; load the copied-back folder below\n"
            "- See `docs/remote_notebook_jobs.md`"
        ),
        mo.hstack([dko_job_name, dko_job_split]),
        dko_job_output,
        export_dko_job,
    ])
    return dko_job_name, dko_job_output, dko_job_split, export_dko_job


@app.cell(hide_code=True)
def _(
    dko_job_name,
    dko_job_output,
    dko_job_split,
    dko_records,
    dko_settings,
    export_dko_job,
):
    if not export_dko_job.value:
        _dko_job_status = mo.md("")
    elif not dko_records:
        _dko_job_status = mo.callout("Choose at least one model.", kind="warn")
    else:
        dko_settings.check()
        _name = dko_job_name.value.strip() or "density_ko"
        _output = f"{dko_job_output.value.strip().rstrip('/')}/{_name}"
        # Next to the training manifests, so a git checkout on the cluster has it
        _manifest = write_job_manifest(
            repository_root / "Experiments/micropatterns/conf/generated/density_ko" / _name / "manifest.yaml",
            f"density_ko_{_name}",
            job_configs(dko_settings, dko_records, _output, split=dko_job_split.value),
        )
        _relative = _manifest.relative_to(repository_root)
        _local = f"output/nodal_ko_eval/{_name}"
        _dko_job_status = mo.callout(
            mo.md(
                f"Wrote {_relative}: {len(dko_records)} models ("
                + ", ".join(
                    f"{_count} {_projection}"
                    for _projection, _count in sorted(
                        pd.Series([_record["projection"] for _record in dko_records]).value_counts().items()
                    )
                )
                + "). On the cluster, from the repository root:\n\n"
                f"```bash\nbash launch/launch_batch_multi_job.sh "
                f"Experiments/micropatterns/density_ko_sweep.py {_relative} <workers> H200\n```\n\n"
                f"Then copy `{_output}/` back to `{_local}/` "
                f"(`bash helper_scripts/copy_models_from_cluster.sh --port <port>`) "
                f"and load that folder below. "
                f"Test one part locally with `python Experiments/micropatterns/density_ko_sweep.py "
                f"--manifest {_relative} --index 0`."
            ),
            kind="success",
        )
    _dko_job_status
    return


@app.cell(hide_code=True)
def _():
    dko_results_path = mo.ui.text(
        "output/nodal_ko_eval/density_ko_results.npz",
        label="Results file, or folder of a remote job's parts (relative to the repository root)",
        full_width=True,
    )
    save_dko_results = mo.ui.run_button(label="Save results")
    load_dko_results = mo.ui.run_button(label="Load results")
    dko_overwrite = mo.ui.checkbox(value=False, label="Overwrite an existing file")
    mo.vstack([
        mo.md(
            "### Save or load density × KO results\n\n"
            "- Saves the pattern counts of every run, so the plots below can be "
            "remade without rerunning the sweep\n"
            "- Cell types are recomputed on loading; the file records the "
            "thresholds and rings they need"
        ),
        dko_results_path,
        mo.hstack([save_dko_results, dko_overwrite, load_dko_results], justify="start"),
    ])
    return dko_overwrite, dko_results_path, load_dko_results, save_dko_results


@app.cell(hide_code=True)
def _(
    dko_overwrite,
    dko_results_path,
    get_dko_results,
    load_dko_results,
    save_dko_results,
    set_dko_results,
):
    _path = Path(dko_results_path.value.strip()).expanduser()
    if not _path.is_absolute():
        _path = repository_root / _path
    if save_dko_results.value:
        if get_dko_results() is None:
            _dko_file_status = mo.callout("Run the sweep (or load results) first.", kind="warn")
        elif _path.exists() and not dko_overwrite.value:
            _dko_file_status = mo.callout(
                f"{_path} exists; choose another name or tick Overwrite.", kind="warn"
            )
        else:
            save_density_ko_results(_path, get_dko_results())
            _dko_file_status = mo.callout(f"Saved to {_path}", kind="success")
    elif load_dko_results.value:
        if not _path.exists():
            _dko_file_status = mo.callout(f"No file or folder at {_path}", kind="danger")
        else:
            set_dko_results(load_density_ko_results(_path))
            _dko_file_status = mo.callout(f"Loaded {_path}", kind="success")
    else:
        _dko_file_status = mo.md("")
    _dko_file_status
    return


@app.cell(hide_code=True)
def _(get_dko_results):
    # Which of the loaded (or run) results are shown below
    _dko_loaded_projections = sorted({_result["projection"] for _result in get_dko_results() or ()})
    dko_projections = mo.ui.multiselect(
        options=_dko_loaded_projections,
        value=_dko_loaded_projections,
        label="Projections shown",
    )
    dko_projections
    return (dko_projections,)


@app.cell(hide_code=True)
def _(
    cell_type_rings,
    cell_type_rules_by_projection,
    dko_projections,
    get_dko_results,
):
    _all_results = tuple(
        _result for _result in get_dko_results() or ()
        if _result["projection"] in dko_projections.value
    )
    # Results need a cell type file of their projection
    _no_rules = sorted({
        _result["projection"] for _result in _all_results
        if _result["projection"] not in cell_type_rules_by_projection
    })
    _usable = tuple(
        _result for _result in _all_results
        if _result["projection"] in cell_type_rules_by_projection
    )
    # Sweep results; the references (true data, synthetic at the measured density) apart
    dko_results = tuple(_result for _result in _usable if _result["source"] == "sweep") or None
    dko_references = tuple(_result for _result in _usable if _result["source"] != "sweep")

    def _thresholds(_rules):
        return {_marker: float(_rules.thresholds[_marker]) for _marker in _rules.markers}

    # Saved counts only hold for the thresholds and rings they were made with
    _mismatches = sorted({
        f"{_result['label']} ({_result['projection']})"
        for _result in dko_results or ()
        if _result.get("rings") != int(cell_type_rings.value)
        or _result.get("cell_type_thresholds")
        != _thresholds(cell_type_rules_by_projection[_result["projection"]])
    })
    _warnings = []
    if len({_result["density_unit"] for _result in dko_results or ()}) > 1:
        _warnings.append(mo.callout(
            "These results sweep the density in different units, so they are not plotted together "
            "correctly; load or run them separately.",
            kind="warn",
        ))
    if _no_rules:
        _warnings.append(mo.callout(
            "Not shown, as no cell type file is chosen for their projection: "
            + ", ".join(_no_rules),
            kind="warn",
        ))
    if _mismatches:
        _warnings.append(mo.callout(
            "These results were made with other cell type thresholds or rings than "
            "the ones chosen above, so their cell types may not match: "
            + ", ".join(_mismatches),
            kind="warn",
        ))
    mo.vstack(_warnings)
    return dko_references, dko_results


@app.cell(hide_code=True)
def _(cell_type_labels, dko_results):
    _dko_densities = [] if not dko_results else [float(_d) for _d in dko_results[0]["densities"]]
    # Mean density actually used at each requested density (cut at what the cell layers hold)
    _dko_actual = (
        [] if not dko_results
        else list(np.mean([actual_densities(_result) for _result in dko_results], axis=0))
    )
    _dko_unit = "share" if not dko_results else dko_results[0]["density_unit"]
    _dko_ko_hours = [] if not dko_results else [float(_h) for _h in dko_results[0]["requested_ko_hours"]]
    _dko_snapshot_hour = 48.0 if not dko_results else dko_results[0]["snapshot_hour"]
    _dko_measured = (
        float(np.mean([_result["measured_density"] for _result in dko_results]))
        if dko_results else 0.5
    )
    dko_group_by = mo.ui.dropdown(
        {
            "Projection": "projection",
            "Projection and resolution": "projection_resolution",
            "Training curriculum": "curriculum",
            "Experiment": "experiment",
            "Family": "family",
            "Notes": "notes",
            "Each model separately": "individual",
            "All models together": "all",
        },
        value="Projection",
        label="Group replicate models by",
    )
    dko_uncertainty = mo.ui.dropdown(
        {
            "Standard deviation": "std",
            "Standard error of mean": "sem",
            "95% percentile interval": "percentile",
            "No uncertainty band": "none",
        },
        value="Standard deviation",
        label="Uncertainty",
    )
    dko_display_cell_types = mo.ui.multiselect(
        options=list(cell_type_labels),
        # Pixels matching several or no cell types are hidden unless chosen
        value=[_label for _label in cell_type_labels if _label not in (SEVERAL, OTHER)],
        label="Displayed cell types",
        full_width=True,
    )
    # Slices default to the measured density and the KO time nearest 24h
    def _density_option(_index):
        return (
            f"{format_density(_dko_actual[_index], _dko_unit)} "
            f"(requested {format_density(_dko_densities[_index], _dko_unit)})"
        )

    # Values are the requested densities, which label the sweep's levels
    dko_fixed_density = mo.ui.dropdown(
        options={_density_option(_i): _d for _i, _d in enumerate(_dko_densities)},
        value=(
            _density_option(int(np.argmin(np.abs(np.asarray(_dko_actual) - _dko_measured))))
            if _dko_densities else None
        ),
        label="Fixed density (KO sweep)",
    )
    dko_fixed_ko = mo.ui.dropdown(
        options={ko_time_label(_h, _dko_snapshot_hour): _h for _h in _dko_ko_hours},
        value=(
            ko_time_label(min(_dko_ko_hours, key=lambda _h: abs(_h - 24.0)), _dko_snapshot_hour)
            if _dko_ko_hours else None
        ),
        label="Fixed KO time (density sweep)",
    )
    mo.vstack([
        dko_display_cell_types,
        mo.hstack([dko_group_by, dko_uncertainty, dko_fixed_ko, dko_fixed_density], justify="start"),
    ])
    return (
        dko_display_cell_types,
        dko_fixed_density,
        dko_fixed_ko,
        dko_group_by,
        dko_uncertainty,
    )


@app.cell(hide_code=True)
def _(dko_references, dko_results):
    dko_show_measured = mo.ui.switch(value=True, label="Show measured density")
    # References present in the results (true data, synthetic at the measured density), shown on request
    _present = [_source for _source in REFERENCE_LABELS if any(_r["source"] == _source for _r in dko_references)]
    dko_show_references = mo.ui.multiselect(
        options={REFERENCE_LABELS[_source]: _source for _source in _present},
        value=[REFERENCE_LABELS[_source] for _source in _present],
        label="References",
    )
    # Density measures shown on the density axes; starts with the sweep's own
    _sweep_unit = dko_results[0]["density_unit"] if dko_results else "per_colony"
    dko_density_axes = mo.ui.multiselect(
        options={_label: _unit for _unit, _label in DENSITY_AXIS_LABELS.items()},
        value=[DENSITY_AXIS_LABELS[_sweep_unit]],
        label="Density axes",
    )
    # Cell-type prevalence as a share of the colony's pixels or of its cells
    # (each 0h cell counted once; needs cells packed without overlap)
    dko_prevalence_basis = mo.ui.dropdown(
        options={"Share of colony pixels": "pixels", "Share of cells": "cells"},
        value="Share of colony pixels",
        label="Prevalence as",
    )
    mo.vstack([
        mo.hstack([dko_prevalence_basis, dko_show_measured, dko_density_axes], justify="start"),
        mo.hstack([
            dko_show_references,
            mo.md(
                "- References: true data (every replicate, each in its own outline), and synthetic colonies "
                "at the measured density\n"
                "- True data has no cell counts, so it is shown by pixels only"
            ),
        ], justify="start") if _present else mo.md(""),
    ])
    return dko_density_axes, dko_prevalence_basis, dko_show_measured, dko_show_references


@app.cell(hide_code=True)
def _(
    cell_type_labels,
    cell_type_rules_by_projection,
    dko_group_by,
    dko_prevalence_basis,
    dko_references,
    dko_results,
    dko_uncertainty,
    fine_resolve_several,
):
    # Shares from the stored pattern counts, so the controls above need no rerun.
    # By cells: counts where every 0h cell counts once, spread over its pixels.
    _by_cells = dko_prevalence_basis.value == "cells"
    _counts_key = "cell_pattern_counts" if _by_cells else "pattern_counts"
    dko_prevalence_label = "Fraction of cells" if _by_cells else "Fraction of colony pixels"
    _no_cell_counts = sorted({
        f"{_result['label']} ({_result['projection']})"
        for _result in dko_results or () if _result.get(_counts_key) is None
    })
    def _prevalence_rows(_results):
        """One row per run, density, KO time and cell type of ``_results``."""
        _rows = []
        for _result in _results:
            if _result.get(_counts_key) is None:
                continue
            _rules = cell_type_rules_by_projection[_result["projection"]]
            _actual = actual_densities(_result)
            _capped = capped_densities(_result)
            for _density_index, _density in enumerate(_result["densities"]):
                for _ko_index, _ko_hour in enumerate(_result["requested_ko_hours"]):
                    _shares = colony_label_shares(
                        model_stain_counts(
                            _result[_counts_key][_density_index, _ko_index], _rules
                        ),
                        _rules,
                        resolve_several=fine_resolve_several.value,
                    )
                    for _cell_type, _share in zip(cell_type_labels, _shares):
                        _rows.append({
                            "model_id": _result["model_id"],
                            "model_label": _result["label"],
                            "projection": _result["projection"],
                            "downsample": _result["downsample"],
                            "replicate": _result["replicate"],
                            "seed": _result["seed"],
                            # sweep, true_data or measured_density
                            "source": _result["source"],
                            "group": _result["groups"][dko_group_by.value],
                            # Requested; actual_density is what was used (in density_unit)
                            "density": float(_density),
                            "actual_density": float(_actual[_density_index]),
                            "capped": bool(_capped[_density_index]),
                            "density_unit": _result["density_unit"],
                            "covered_share": float(_result["covered_shares"][_density_index]),
                            "cells_per_mm2": (
                                float(_result["cells_per_mm2"][_density_index])
                                if _result["cells_per_mm2"] is not None else float("nan")
                            ),
                            "cells_per_colony": (
                                float(_result["cells_per_colony"][_density_index])
                                if _result["cells_per_colony"] is not None else float("nan")
                            ),
                            "measured_density": _result["measured_density"],
                            "max_density": _result["max_density"],
                            "knockout_hour": float(_ko_hour),
                            "realised_knockout_hour": float(_result["realised_ko_hours"][_ko_index]),
                            "cell_type": _cell_type,
                            "relative_prevalence": float(_share),
                        })
        return pd.DataFrame(_rows)

    def _summary(_frame, _keys):
        """Mean and band of the prevalence per ``_keys``, with every density measure."""
        _summary_rows = []
        if _frame.empty:
            return pd.DataFrame()
        for _key_values, _values in _frame.groupby(_keys, sort=False):
            _mean, _lower, _upper = uncertainty_band(
                _values["relative_prevalence"], dko_uncertainty.value
            )
            _summary_rows.append({
                **dict(zip(_keys, _key_values)),
                # References: one density each (it can differ a little between models)
                **({} if "density" in _keys else {"density": float(_values["density"].mean())}),
                # Mean and range of the densities used by the runs at this requested density
                "actual_density": float(_values["actual_density"].mean()),
                "actual_density_min": float(_values["actual_density"].min()),
                "actual_density_max": float(_values["actual_density"].max()),
                "capped_share": float(_values["capped"].mean()),
                # Every density measure, for the extra density axes
                **{_column: float(_values[_column].mean()) for _column in DENSITY_COLUMNS.values()},
                "mean_prevalence": _mean,
                "lower": _lower,
                "upper": _upper,
                "sample_count": int(len(_values)),
            })
        return pd.DataFrame(_summary_rows)

    dko_prevalence = _prevalence_rows(dko_results or ())
    dko_summary = _summary(dko_prevalence, ["group", "cell_type", "density", "knockout_hour"])
    # References: true data, and synthetic colonies at the measured density
    dko_reference_prevalence = _prevalence_rows(dko_references)
    dko_reference_summary = _summary(dko_reference_prevalence, ["group", "source", "cell_type", "knockout_hour"])
    (
        mo.callout(
            "No cell counts (cells placed in overlapping layers, or older results), so not shown "
            "by cells: " + ", ".join(_no_cell_counts),
            kind="warn",
        )
        if _by_cells and _no_cell_counts else mo.md("")
    )
    return (
        dko_prevalence,
        dko_prevalence_label,
        dko_reference_prevalence,
        dko_reference_summary,
        dko_summary,
    )


@app.cell(hide_code=True)
def _():
    # Colour scale of the synthetic initial states below
    dko_initial_scale = mo.ui.dropdown(
        options={"Percentiles of these states": "percentile", "Fixed values": "fixed"},
        value="Percentiles of these states",
        label="Colour scale",
    )
    dko_initial_percentiles = mo.ui.range_slider(0, 100, step=0.5, value=[1, 99], label="Percentiles")
    dko_initial_values = mo.ui.range_slider(0.0, 1.0, step=0.01, value=[0.0, 1.0], label="Values")
    dko_initial_cmap = mo.ui.dropdown(COLOUR_MAPS, value="magma", label="Colour map")
    return dko_initial_cmap, dko_initial_percentiles, dko_initial_scale, dko_initial_values


@app.cell(hide_code=True)
def _(
    dko_initial_cmap,
    dko_initial_percentiles,
    dko_initial_scale,
    dko_initial_values,
    dko_references,
    dko_results,
    dko_show_references,
):
    # Synthetic initial states of the first run, SOX2 inside the colony; then the references' (when shown)
    if not dko_results:
        _dko_initial_view = mo.md("")
    else:
        _first = dko_results[0]
        _channel = (
            _first["state_channel_names"].index("SOX2")
            if "SOX2" in _first["state_channel_names"] else 0
        )
        _unit = _first["density_unit"]
        # (state, colony, title) of each panel
        _panels = [
            (_state, _first["colony"], f"{format_density(_density, _unit)} · share {_share:.2f}")
            for _density, _share, _state in zip(
                actual_densities(_first), _first["covered_shares"], _first["initial_states"]
            )
        ]
        for _source in dko_show_references.value:
            # The reference of the shown run's replicate, else another of the model's
            _candidates = [
                _r for _r in dko_references
                if _r["source"] == _source and _r["model_id"] == _first["model_id"] and _r["initial_states"] is not None
            ]
            _reference = next(
                (_r for _r in _candidates if _r["replicate"] == _first["replicate"]),
                _candidates[0] if _candidates else None,
            )
            if _reference is not None:
                _panels.append((
                    _reference["initial_states"][0], _reference["colony"],
                    f"{REFERENCE_LABELS[_source]} · replicate {_reference['replicate']}\n"
                    f"{format_density(actual_densities(_reference)[0], _unit)} · "
                    f"share {_reference['covered_shares'][0]:.2f}",
                ))
        # A grid of at most 5 snapshots per row, so each stays readable
        _columns = min(5, len(_panels))
        _rows = int(np.ceil(len(_panels) / _columns))
        _figure, _axes = plt.subplots(
            _rows, _columns, figsize=(3.2 * _columns, 3.4 * _rows), dpi=150, squeeze=False
        )
        for _axis in _axes.flat[len(_panels):]:
            _axis.axis("off")
        _cmap = plt.get_cmap(dko_initial_cmap.value).copy()
        _cmap.set_bad(color="#d9d9d9")
        # One colour scale for every density: percentiles of the colony pixels of all of them, or fixed values
        if dko_initial_scale.value == "percentile":
            _vmin, _vmax = np.percentile(
                _first["initial_states"][:, _channel][:, _first["colony"]], dko_initial_percentiles.value
            )
        else:
            _vmin, _vmax = dko_initial_values.value
        _vmax = max(float(_vmax), float(_vmin) + 1e-6)
        for _axis, (_state, _colony, _title) in zip(_axes.flat, _panels):
            _axis.imshow(
                np.where(_colony, _state[_channel], np.nan),
                cmap=_cmap, vmin=_vmin, vmax=_vmax, origin="lower",
            )
            _axis.set_title(_title, fontsize=9)
            _axis.set_xticks([])
            _axis.set_yticks([])
        _figure.suptitle(
            f"Synthetic 0h {_first['state_channel_names'][_channel]} · "
            f"replicate {_first['replicate']} · seed {_first['seed']} · "
            f"colour scale {_vmin:.3g}–{_vmax:.3g}",
            fontsize=10,
        )
        _figure.tight_layout()
        _dko_initial_view = mo.vstack([
            mo.hstack([
                dko_initial_scale,
                dko_initial_percentiles if dko_initial_scale.value == "percentile" else dko_initial_values,
                dko_initial_cmap,
            ], justify="start"),
            _figure,
            mo.md(
                f"- Sweep: {DENSITY_UNITS[_first['density_unit']]}; "
                + (
                    "cells packed without overlap; the colony holds at most "
                    if _first.get("placement") == "tiled" else "overlapping layers; they hold at most "
                )
                + f"{format_density(_first['max_density'], _first['density_unit'])}\n"
                f"- Measured in the replicate's 0h colony (estimates of the true values): "
                + (
                    f"{_first['measured_cells_per_mm2'] * _first['colony_mm2']:.0f} cells in a colony "
                    f"of this size, {_first['measured_cells_per_mm2']:.0f} cells/mm², "
                    if "colony_mm2" in _first else ""
                )
                + f"pixel coverage {_first['measured_covered_share']:.2f}"
                + (
                    f"\n- Pieces placed: {_first.get('piece_source', 'segments')}, "
                    f"{_first['cells_per_segment']:.2f} cells each"
                    if "cells_per_segment" in _first else ""
                )
            ),
        ])
        # _figure.show()
    _dko_initial_view
    return


@app.cell(hide_code=True)
def _(
    dko_display_cell_types,
    dko_density_axes,
    dko_fixed_density,
    dko_fixed_ko,
    dko_prevalence_label,
    dko_reference_summary,
    dko_results,
    dko_show_measured,
    dko_show_references,
    dko_summary,
    dko_uncertainty,
):
    # References shown: their summary rows of the displayed cell types
    _references = (
        dko_reference_summary[
            dko_reference_summary["source"].isin(dko_show_references.value)
            & dko_reference_summary["cell_type"].isin(dko_display_cell_types.value)
        ]
        if not dko_reference_summary.empty else pd.DataFrame()
    )

    def _plot_references(_axis, _x_column=None):
        """References at the fixed KO time as points at their density (``_x_column``),
        or against KO time as lines (``_x_column`` None)."""
        if _references.empty:
            return
        for (_group, _source, _cell_type), _rows in _references.groupby(["group", "source", "cell_type"], sort=False):
            _colour = cell_type_colour(_cell_type)
            if _x_column is None:
                _rows = _rows.sort_values("knockout_hour")
                _axis.plot(
                    _rows["knockout_hour"], _rows["mean_prevalence"], color=_colour,
                    linestyle=REFERENCE_LINESTYLES[_source], marker=REFERENCE_MARKERS[_source], markersize=4,
                )
            else:
                _rows = _rows[np.isclose(_rows["knockout_hour"], dko_fixed_ko.value)]
                _mean = _rows["mean_prevalence"].to_numpy()
                _axis.errorbar(
                    _rows[_x_column].to_numpy(), _mean,
                    yerr=[_mean - _rows["lower"].to_numpy(), _rows["upper"].to_numpy() - _mean],
                    color=_colour, marker=REFERENCE_MARKERS[_source], markersize=8, linestyle="none", capsize=2,
                )

    def _plot_slice(_axis, _fixed_column, _fixed_value, _x_column):
        # Mean ± band against _x_column, at _fixed_column == _fixed_value
        _styles = ("-", "--", ":", "-.")
        _shown = dko_summary[
            np.isclose(dko_summary[_fixed_column], _fixed_value)
            & dko_summary["cell_type"].isin(dko_display_cell_types.value)
        ]
        _groups = list(dict.fromkeys(dko_summary["group"]))
        # The group goes in the legend only when several groups are shown
        _several_groups = _shown["group"].nunique() > 1
        for (_group, _cell_type), _curve in _shown.groupby(["group", "cell_type"], sort=False):
            _curve = _curve.sort_values(_x_column)
            _line = _axis.plot(
                _curve[_x_column], _curve["mean_prevalence"],
                _styles[_groups.index(_group) % len(_styles)],
                marker="o", markersize=3,
                color=cell_type_colour(_cell_type),
                label=f"{_group} · {_cell_type}" if _several_groups else _cell_type,
            )
            if dko_uncertainty.value != "none" and _curve["sample_count"].max() > 1:
                _axis.fill_between(
                    _curve[_x_column], _curve["lower"], _curve["upper"],
                    color=_line[0].get_color(), alpha=0.18,
                )
        _axis.set(ylabel=dko_prevalence_label, ylim=(0.0, None))

    if dko_summary.empty or dko_fixed_ko.value is None or dko_fixed_density.value is None:
        _dko_slice_view = mo.md("")
    else:
        _snapshot_hour = dko_results[0]["snapshot_hour"]
        _unit = dko_results[0]["density_unit"]
        # Drawn linear in the main chosen density measure; the others are extra axes
        _main = main_density_unit(dko_summary, _unit, dko_density_axes.value)
        _column = density_column(_main, _unit)
        _measured = measured_density(dko_results, _main)
        _figure, (_density_axis, _ko_axis) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
        _plot_slice(_density_axis, "knockout_hour", dko_fixed_ko.value, _column)
        _plot_references(_density_axis, _column)
        if dko_show_measured.value and _measured is not None:
            _density_axis.axvline(_measured, color="gray", linestyle="--", linewidth=1)
            _density_axis.text(
                _measured, 0.98, " measured", color="gray", fontsize=8, va="top",
                transform=_density_axis.get_xaxis_transform(),
            )
        _density_axis.set(
            title=f"Density sweep · KO {ko_time_label(dko_fixed_ko.value, _snapshot_hour)}",
        )
        add_density_axes(
            _density_axis, "x", _main,
            density_conversions(
                dko_summary[np.isclose(dko_summary["knockout_hour"], dko_fixed_ko.value)],
                _unit, dko_density_axes.value, _main,
            ),
            gap=0.2,
        )
        _plot_slice(_ko_axis, "density", dko_fixed_density.value, "knockout_hour")
        _plot_references(_ko_axis)
        _ticks, _tick_labels = ko_time_ticks(_snapshot_hour)
        _ko_axis.set(
            xlabel="NODAL knockout time (h)",
            xticks=_ticks,
            xticklabels=_tick_labels,
            title=(
                "KO sweep · density "
                + format_density(
                    float(dko_summary.loc[
                        np.isclose(dko_summary["density"], dko_fixed_density.value), _column
                    ].mean()),
                    _main,
                )
            ),
        )
        _handles, _labels = _ko_axis.get_legend_handles_labels()
        _reference_handles = reference_legend_handles(
            [_source for _source in REFERENCE_LABELS if not _references.empty and _source in set(_references["source"])]
        )
        _ko_axis.legend(
            _handles + _reference_handles, _labels + [_h.get_label() for _h in _reference_handles],
            fontsize="small", ncol=2,
        )
        _figure.suptitle(f"Predicted cell-type prevalence at {_snapshot_hour:g}h")
        _figure.tight_layout()
        _dko_slice_view = _figure
    _dko_slice_view
    return


@app.cell(hide_code=True)
def _():
    save_dko_slice = mo.ui.run_button(label="Save KO slice for overview figure")
    mo.vstack([
        mo.md(
            "- Saves the KO sweep at the fixed density above (default: nearest the measured "
            "density) to `figures/overview/nodal_ko_density_slice_samples.csv`, one row per run "
            "and KO time\n"
            "- `overview_figure.py` uses it for the KO-time curve in place of the fine KO sweep"
        ),
        save_dko_slice,
    ])
    return (save_dko_slice,)


@app.cell(hide_code=True)
def _(dko_fixed_density, dko_prevalence, dko_prevalence_basis, save_dko_slice):
    mo.stop(not save_dko_slice.value)
    mo.stop(
        dko_prevalence.empty or dko_fixed_density.value is None,
        mo.callout("Run or load the density × KO sweep first.", kind="warn"),
    )
    _slice = dko_prevalence[np.isclose(dko_prevalence["density"], dko_fixed_density.value)]
    # Same layout as the fine KO sweep export (nodal_ko_prevalence_samples.csv),
    # plus the density of each run
    _index = [
        "model_id", "model_label", "projection", "downsample", "replicate", "seed", "group",
        "knockout_hour", "realised_knockout_hour",
        "density_unit", "density", "actual_density", "measured_density",
    ]
    _export = (
        _slice.pivot(index=_index, columns="cell_type", values="relative_prevalence")
        .rename(columns=lambda cell_type: f"{cell_type.lower()}_prevalence")
        .reset_index()
        .rename(columns={
            "model_id": "model_realisation_id",
            "model_label": "model_realisation_label",
            "replicate": "initial_condition_replicate",
            "seed": "random_seed",
            "knockout_hour": "requested_knockout_hour",
            "density": "requested_density",
        })
        .assign(prevalence_basis=dko_prevalence_basis.value)
    )
    _path = repository_root / "figures" / "overview" / "nodal_ko_density_slice_samples.csv"
    _path.parent.mkdir(parents=True, exist_ok=True)
    _export.to_csv(_path, index=False)
    mo.callout(
        f"Saved {len(_export)} rows ({_export['requested_knockout_hour'].nunique()} KO times) to `{_path}`",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(dko_results, dko_summary):
    _dko_heatmap_groups = list(dict.fromkeys(dko_summary["group"])) if not dko_summary.empty else []
    dko_heatmap_group = mo.ui.dropdown(
        options=_dko_heatmap_groups,
        value=next(iter(_dko_heatmap_groups), None),
        label="Heatmap group",
    )
    dko_heatmap_scale = mo.ui.dropdown(
        {"0 to 1 for every cell type": "shared", "Each cell type's own range": "own"},
        value="0 to 1 for every cell type",
        label="Colour scale",
    )
    dko_heatmap_cmap = mo.ui.dropdown(COLOUR_MAPS, value="viridis", label="Colour map")
    # Density measures of the heatmaps (and surfaces); rows are linear in the main one
    _sweep_unit = dko_results[0]["density_unit"] if dko_results else "per_colony"
    dko_heatmap_density_axes = mo.ui.multiselect(
        options={_label: _unit for _unit, _label in DENSITY_AXIS_LABELS.items()},
        value=[DENSITY_AXIS_LABELS[_sweep_unit]],
        label="Density axes",
    )
    mo.hstack(
        [dko_heatmap_group, dko_heatmap_scale, dko_heatmap_cmap, dko_heatmap_density_axes], justify="start"
    )
    return dko_heatmap_cmap, dko_heatmap_density_axes, dko_heatmap_group, dko_heatmap_scale


@app.cell(hide_code=True)
def _(
    dko_display_cell_types,
    dko_fixed_density,
    dko_fixed_ko,
    dko_heatmap_cmap,
    dko_heatmap_density_axes,
    dko_heatmap_group,
    dko_heatmap_scale,
    dko_prevalence_label,
    dko_reference_summary,
    dko_results,
    dko_show_measured,
    dko_show_references,
    dko_summary,
):
    # Grids of the heatmaps, also drawn as surfaces below
    dko_heatmap_grids = None
    if dko_summary.empty or dko_heatmap_group.value is None or not dko_display_cell_types.value:
        _dko_heatmap_view = mo.md("")
    else:
        _snapshot_hour = dko_results[0]["snapshot_hour"]
        _shown = dko_summary[dko_summary["group"] == dko_heatmap_group.value]
        _cell_types = [
            _cell_type for _cell_type in dko_display_cell_types.value
            if _cell_type in set(_shown["cell_type"])
        ]
        _unit = dko_results[0]["density_unit"]
        # Rows are linear in the main chosen density measure; the others are extra axes
        _main = main_density_unit(_shown, _unit, dko_heatmap_density_axes.value)
        _measured = measured_density(dko_results, _main)
        _conversions = density_conversions(_shown, _unit, dko_heatmap_density_axes.value, _main)
        # References of this group (true data, synthetic at the measured density), as a strip above each map
        _references = (
            dko_reference_summary[
                (dko_reference_summary["group"] == dko_heatmap_group.value)
                & dko_reference_summary["source"].isin(dko_show_references.value)
            ]
            if not dko_reference_summary.empty else pd.DataFrame()
        )
        _sources = [_s for _s in REFERENCE_LABELS if not _references.empty and _s in set(_references["source"])]
        # Each extra density axis takes about 1.2 in left of its heatmap
        _width = (4.6 + 1.2 * len(_conversions)) * len(_cell_types)
        if _sources:
            _figure, _all_axes = plt.subplots(
                2, len(_cell_types), figsize=(_width, 4.4 + 0.3 * len(_sources)), dpi=150, squeeze=False,
                sharex="col", height_ratios=[0.3 * len(_sources), 3.6], layout="constrained",
            )
            _strips, _axes = _all_axes[:1], _all_axes[1:]
        else:
            _figure, _axes = plt.subplots(1, len(_cell_types), figsize=(_width, 4.2), dpi=150, squeeze=False)
            _strips = None
        # Rows sit at the density each requested density actually used, on a
        # linear axis, so rows can differ in height. Requested densities cut
        # down to (about) the same value by the cell layers share one row.
        _levels = _shown.groupby("density")[density_column(_main, _unit)].mean().sort_index()
        _row_values, _row_of = density_rows(_levels)

        dko_heatmap_grids = {
            "group": dko_heatmap_group.value,
            "snapshot_hour": _snapshot_hour,
            "unit": _main,
            "measured": _measured,
            "value_label": dko_prevalence_label,
            "densities": _row_values,
            # The rows in the extra density measures, for the surfaces' hover text
            "extra_densities": {
                _extra: _forward(_row_values) for _extra, (_forward, _) in _conversions.items()
            },
            "grids": {},
            # {source: {"density": in the main measure, "values": {cell type: [KO times]}}}
            "references": {},
        }
        for _column, (_axis, _cell_type) in enumerate(zip(_axes[0], _cell_types)):
            _grid = _shown[_shown["cell_type"] == _cell_type].pivot(
                index="density", columns="knockout_hour", values="mean_prevalence"
            ).sort_index().sort_index(axis=1)
            _grid = _grid.groupby(_grid.index.map(_row_of)).mean().sort_index()
            dko_heatmap_grids["ko_hours"] = _grid.columns.to_numpy()
            dko_heatmap_grids["grids"][_cell_type] = _grid.to_numpy()
            _image = _axis.pcolormesh(
                cell_edges(_grid.columns.to_numpy()), cell_edges(_row_values), _grid.to_numpy(),
                shading="flat", cmap=dko_heatmap_cmap.value,
                **({"vmin": 0.0, "vmax": 1.0} if dko_heatmap_scale.value == "shared" else {}),
            )
            # Where the two slices above cut the map
            _axis.axvline(dko_fixed_ko.value, color="white", linewidth=0.8, linestyle=":")
            _fixed_level = _levels.index[np.argmin(np.abs(_levels.index.to_numpy() - dko_fixed_density.value))]
            _axis.axhline(
                _row_values[_row_of[_fixed_level]], color="white", linewidth=0.8, linestyle=":",
            )
            if dko_show_measured.value and _measured is not None:
                _axis.axhline(_measured, color="white", linewidth=0.8, linestyle="--")
            _ticks, _tick_labels = ko_time_ticks(_snapshot_hour)
            _axis.set(
                xlabel="NODAL knockout time (h)",
                xticks=_ticks,
                xticklabels=_tick_labels,
                title=_cell_type,
            )
            add_density_axes(_axis, "y", _main, _conversions, gap=0.3)
            if _strips is not None:
                # The references' prevalence at the same KO times, on the same colour scale
                _ko = _grid.columns.to_numpy()
                _strip_rows = []
                for _source in _sources:
                    _of = _references[(_references["source"] == _source) & (_references["cell_type"] == _cell_type)]
                    _strip_rows.append(_of.set_index("knockout_hour")["mean_prevalence"].reindex(_ko).to_numpy())
                    _entry = dko_heatmap_grids["references"].setdefault(
                        _source, {"density": float(_of[density_column(_main, _unit)].mean()), "values": {}},
                    )
                    _entry["values"][_cell_type] = _strip_rows[-1]
                draw_reference_strip(_strips[0, _column], _ko, _strip_rows, _sources, _image)
                _strips[0, _column].set_title(_cell_type)
                _axis.set_title("")
                plt.setp(_strips[0, _column].get_xticklabels(), visible=False)
                _figure.colorbar(_image, ax=[_strips[0, _column], _axis], label=dko_prevalence_label)
            else:
                _figure.colorbar(_image, ax=_axis, label=dko_prevalence_label)
        _figure.suptitle(
            f"{dko_heatmap_group.value} · mean prevalence at {_snapshot_hour:g}h "
            + ("(dotted: slices above, dashed: measured density)" if dko_show_measured.value
               else "(dotted: slices above)")
            + ("; strip: references" if _sources else "")
        )
        if _strips is None:
            _figure.tight_layout()
        _dko_heatmap_view = _figure
    _dko_heatmap_view
    return (dko_heatmap_grids,)


@app.cell(hide_code=True)
def _(dko_heatmap_grids, dko_show_measured):
    # The heatmaps as translucent surfaces: drag to rotate, click a legend entry to hide it
    if dko_heatmap_grids is None:
        _dko_surface_view = mo.md("")
    else:
        _ko = dko_heatmap_grids["ko_hours"]
        _densities = dko_heatmap_grids["densities"]
        _surface_figure = go.Figure()
        # Extra density measures of each row, shown on hover
        _extras = dko_heatmap_grids["extra_densities"]
        _extra_hover = "".join(
            f"<br>{DENSITY_AXIS_LABELS[_extra]} %{{customdata[{_index}]:.4g}}"
            for _index, _extra in enumerate(_extras)
        )
        for _cell_type, _grid in dko_heatmap_grids["grids"].items():
            _colour = cell_type_colour(_cell_type)
            _surface_figure.add_trace(go.Surface(
                x=_ko, y=_densities, z=_grid,
                customdata=(
                    np.stack([np.broadcast_to(_values[:, None], _grid.shape) for _values in _extras.values()], axis=-1)
                    if _extras else None
                ),
                name=_cell_type,
                showlegend=True,
                showscale=False,
                opacity=0.55,
                colorscale=[[0.0, _colour], [1.0, _colour]],
                hovertemplate=(
                    f"{_cell_type}<br>KO %{{x:g}} h<br>{DENSITY_AXIS_LABELS[dko_heatmap_grids['unit']]} %{{y:.4g}}{_extra_hover}<br>"
                    "prevalence %{z:.3f}<extra></extra>"
                ),
            ))
        # References: a line over KO time at each reference's density
        for _source, _reference in dko_heatmap_grids["references"].items():
            for _cell_type, _values in _reference["values"].items():
                _surface_figure.add_trace(go.Scatter3d(
                    x=dko_heatmap_grids["ko_hours"], y=np.full(len(_values), _reference["density"]), z=_values,
                    mode="lines+markers",
                    line={"color": cell_type_colour(_cell_type), "width": 5,
                          "dash": "dash" if _source == "true_data" else "dot"},
                    marker={"size": 3, "symbol": "diamond" if _source == "true_data" else "square"},
                    name=f"{REFERENCE_LABELS[_source]} · {_cell_type}",
                ))
        _measured = dko_heatmap_grids["measured"]
        if dko_show_measured.value and _measured is not None and _densities.min() <= _measured <= _densities.max():
            # Measured density, along the floor
            _surface_figure.add_trace(go.Scatter3d(
                x=[_ko.min(), _ko.max()], y=[_measured, _measured], z=[0.0, 0.0],
                mode="lines", line={"color": "gray", "dash": "dash", "width": 4},
                name="Measured density",
            ))
        _surface_figure.update_layout(
            title=(
                f"{dko_heatmap_grids['group']} · mean prevalence at "
                f"{dko_heatmap_grids['snapshot_hour']:g}h"
            ),
            scene={
                "xaxis_title": "NODAL knockout time (h)",
                "yaxis_title": DENSITY_AXIS_LABELS[dko_heatmap_grids["unit"]],
                "zaxis_title": dko_heatmap_grids["value_label"],
                "zaxis": {"range": [0.0, 1.0]},
            },
            legend={"title": "Click to show or hide"},
            # Keeps the camera where the user left it when the plot is redrawn
            uirevision="dko_surfaces",
            height=650,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        _dko_surface_view = _surface_figure
    _dko_surface_view
    return


@app.cell(hide_code=True)
def _(dko_results):
    # Measured channels of the final state: mean over the chosen rings
    _with_means = [_result for _result in dko_results or () if _result.get("channel_means") is not None]
    _names = list(dict.fromkeys(_name for _result in _with_means for _name in _result["state_channel_names"]))
    _rings = max([int(_result["rings"]) for _result in _with_means] or [1])
    dko_channels = mo.ui.multiselect(options=_names, value=_names, label="Channels", full_width=True)
    dko_ring_range = mo.ui.range_slider(
        1, max(_rings, 2), value=[1, _rings], step=1, label="Rings (1: centre)", show_value=True
    )
    mo.vstack([
        mo.md(
            "### Measured channels\n\n"
            "- Mean of each channel of the final state over the chosen rings, against density and KO time\n"
            "- Uses the fixed KO time, fixed density, grouping and heatmap group above"
        ),
        (
            mo.callout("These results have no channel means (run before they were saved).", kind="warn")
            if dko_results and not _with_means else mo.hstack([dko_channels, dko_ring_range], justify="start")
        ),
    ])
    return dko_channels, dko_ring_range


@app.cell(hide_code=True)
def _(
    dko_channels,
    dko_group_by,
    dko_references,
    dko_results,
    dko_ring_range,
    dko_uncertainty,
):
    _low, _high = (int(_value) for _value in dko_ring_range.value)

    def _channel_values(_results):
        """Mean of each chosen channel over the chosen rings, per run, density and KO time."""
        _frames = []
        for _result in _results:
            _means = _result.get("channel_means")
            if _means is None:
                continue
            # Weight each chosen ring by its pixels, so the mean is over the colony area
            _rings = int(_result["rings"])
            _ring_of_pixel = radial_rings(_result["colony"], _rings)
            _area = np.bincount(_ring_of_pixel[_ring_of_pixel >= 0], minlength=_rings).astype(float)
            _area[: _low - 1] = 0.0
            _area[_high:] = 0.0
            _weights = np.where(np.isfinite(_means), _area[None, None, :, None], 0.0)
            _values = np.nansum(_means * _weights, axis=2) / np.maximum(_weights.sum(axis=2), 1e-12)
            _names = list(_result["state_channel_names"])
            _density_index, _ko_index, _channel_index = np.meshgrid(
                np.arange(_values.shape[0]), np.arange(_values.shape[1]), np.arange(_values.shape[2]),
                indexing="ij",
            )
            _frame = pd.DataFrame({
                "density": np.asarray(_result["densities"], dtype=float)[_density_index.ravel()],
                "actual_density": actual_densities(_result)[_density_index.ravel()],
                "knockout_hour": np.asarray(_result["requested_ko_hours"], dtype=float)[_ko_index.ravel()],
                "channel": np.asarray(_names)[_channel_index.ravel()],
                "value": _values.ravel(),
            })
            for _unit, _column in DENSITY_COLUMNS.items():
                _measure = _result.get({"share": "covered_shares"}.get(_unit, _column))
                _frame[_column] = (
                    np.asarray(_measure, dtype=float)[_density_index.ravel()] if _measure is not None else np.nan
                )
            _frame["group"] = _result["groups"][dko_group_by.value]
            _frame["source"] = _result["source"]
            _frame["model_label"] = _result["label"]
            _frame["replicate"] = _result["replicate"]
            _frame["seed"] = _result["seed"]
            _frames.append(_frame[_frame["channel"].isin(dko_channels.value)])
        return pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()

    def _channel_summary(_frame, _keys):
        _summary_rows = []
        if _frame.empty:
            return pd.DataFrame()
        for _key_values, _rows in _frame.groupby(_keys, sort=False):
            _mean, _lower, _upper = uncertainty_band(_rows["value"], dko_uncertainty.value, clip=False)
            _summary_rows.append({
                **dict(zip(_keys, _key_values)),
                # References: one density each (it can differ a little between models)
                **({} if "density" in _keys else {"density": float(_rows["density"].mean())}),
                "actual_density": float(_rows["actual_density"].mean()),
                **{_column: float(_rows[_column].mean()) for _column in DENSITY_COLUMNS.values()},
                "mean_value": _mean,
                "lower": _lower,
                "upper": _upper,
                "sample_count": int(len(_rows)),
            })
        return pd.DataFrame(_summary_rows)

    dko_channel_values = _channel_values(dko_results or ())
    dko_channel_summary = _channel_summary(dko_channel_values, ["group", "channel", "density", "knockout_hour"])
    # References: true data, and synthetic colonies at the measured density
    dko_reference_channel_summary = _channel_summary(
        _channel_values(dko_references), ["group", "source", "channel", "knockout_hour"],
    )
    return dko_channel_summary, dko_channel_values, dko_reference_channel_summary


@app.cell(hide_code=True)
def _(
    dko_channel_summary,
    dko_channel_values,
    dko_channels,
    dko_density_axes,
    dko_fixed_density,
    dko_fixed_ko,
    dko_reference_channel_summary,
    dko_results,
    dko_show_references,
    dko_uncertainty,
):
    # One row per channel: density sweep at the fixed KO time, KO sweep at the fixed density
    if dko_channel_summary.empty or dko_fixed_ko.value is None or dko_fixed_density.value is None:
        _dko_channel_view = mo.md("")
    else:
        _channels = [_c for _c in dko_channels.value if _c in set(dko_channel_summary["channel"])]
        _groups = list(dict.fromkeys(dko_channel_summary["group"]))
        _unit = dko_results[0]["density_unit"]
        _snapshot_hour = dko_results[0]["snapshot_hour"]
        # Drawn linear in the main chosen density measure; the others are extra axes
        _main = main_density_unit(dko_channel_summary, _unit, dko_density_axes.value)
        _column = density_column(_main, _unit)
        _conversions = density_conversions(
            dko_channel_summary[np.isclose(dko_channel_summary["knockout_hour"], dko_fixed_ko.value)],
            _unit, dko_density_axes.value, _main,
        )
        _figure, _axes = plt.subplots(
            len(_channels), 2, figsize=(13, (3.0 + 0.6 * len(_conversions)) * len(_channels)),
            dpi=130, squeeze=False,
        )

        def _plot(_axis, _rows, _x_column):
            for _index, (_group, _curve) in enumerate(_rows.groupby("group", sort=False)):
                _curve = _curve.sort_values(_x_column)
                _colour = SERIES[_groups.index(_group) % len(SERIES)]
                _axis.plot(
                    _curve[_x_column], _curve["mean_value"], marker="o", markersize=3,
                    color=_colour, label=_group,
                )
                if dko_uncertainty.value != "none" and _curve["sample_count"].max() > 1:
                    _axis.fill_between(
                        _curve[_x_column], _curve["lower"], _curve["upper"], color=_colour, alpha=0.18,
                    )

        # References shown: points at their density (left) and lines over KO time (right)
        _references = (
            dko_reference_channel_summary[dko_reference_channel_summary["source"].isin(dko_show_references.value)]
            if not dko_reference_channel_summary.empty else pd.DataFrame()
        )

        def _plot_references(_density_axis, _ko_axis, _channel):
            if _references.empty:
                return
            _of = _references[_references["channel"] == _channel]
            for (_group, _source), _rows in _of.groupby(["group", "source"], sort=False):
                _colour = SERIES[_groups.index(_group) % len(SERIES)] if _group in _groups else "0.35"
                _rows = _rows.sort_values("knockout_hour")
                _ko_axis.plot(
                    _rows["knockout_hour"], _rows["mean_value"], color=_colour,
                    linestyle=REFERENCE_LINESTYLES[_source], marker=REFERENCE_MARKERS[_source], markersize=4,
                )
                _at = _rows[np.isclose(_rows["knockout_hour"], dko_fixed_ko.value)]
                _mean = _at["mean_value"].to_numpy()
                _density_axis.errorbar(
                    _at[_column].to_numpy(), _mean,
                    yerr=[_mean - _at["lower"].to_numpy(), _at["upper"].to_numpy() - _mean],
                    color=_colour, marker=REFERENCE_MARKERS[_source], markersize=8, linestyle="none", capsize=2,
                )

        for _row, _channel in enumerate(_channels):
            _shown = dko_channel_summary[dko_channel_summary["channel"] == _channel]
            _density_axis, _ko_axis = _axes[_row]
            _plot(_density_axis, _shown[np.isclose(_shown["knockout_hour"], dko_fixed_ko.value)], _column)
            _plot_references(_density_axis, _ko_axis, _channel)
            _density_axis.set(
                ylabel=f"{_channel} (mean)",
                title=f"KO {ko_time_label(dko_fixed_ko.value, _snapshot_hour)}" if _row == 0 else None,
            )
            add_density_axes(_density_axis, "x", _main, _conversions, gap=0.25)
            _plot(_ko_axis, _shown[np.isclose(_shown["density"], dko_fixed_density.value)], "knockout_hour")
            _ticks, _tick_labels = ko_time_ticks(_snapshot_hour)
            _ko_axis.set(
                xlabel="NODAL knockout time (h)", xticks=_ticks, xticklabels=_tick_labels,
                title=(
                    "density " + format_density(float(_shown.loc[
                        np.isclose(_shown["density"], dko_fixed_density.value), _column
                    ].mean()), _main)
                ) if _row == 0 else None,
            )
        _reference_handles = reference_legend_handles(
            [_source for _source in REFERENCE_LABELS if not _references.empty and _source in set(_references["source"])]
        )
        if len(_groups) > 1 or _reference_handles:
            _handles, _labels = _axes[0, 1].get_legend_handles_labels()
            _axes[0, 1].legend(
                (_handles if len(_groups) > 1 else []) + _reference_handles,
                (_labels if len(_groups) > 1 else []) + [_h.get_label() for _h in _reference_handles],
                fontsize="small",
            )
        _figure.tight_layout()
        _dko_channel_view = mo.vstack([
            _figure,
            mo.hstack([
                mo.download(
                    lambda: dko_channel_values.to_csv(index=False).encode("utf-8"),
                    filename="nodal_ko_density_channels.csv", mimetype="text/csv",
                    label="Download per-run channel CSV",
                ),
                mo.download(
                    lambda: dko_channel_summary.to_csv(index=False).encode("utf-8"),
                    filename="nodal_ko_density_channels_summary.csv", mimetype="text/csv",
                    label="Download grouped channel CSV",
                ),
            ], justify="start"),
        ])
    _dko_channel_view
    return


@app.cell(hide_code=True)
def _():
    # Colour map of the channel heatmaps below
    dko_channel_cmap = mo.ui.dropdown(COLOUR_MAPS, value="viridis", label="Colour map")
    return (dko_channel_cmap,)


@app.cell(hide_code=True)
def _(
    dko_channel_cmap,
    dko_channel_summary,
    dko_channels,
    dko_fixed_density,
    dko_fixed_ko,
    dko_heatmap_density_axes,
    dko_heatmap_group,
    dko_reference_channel_summary,
    dko_results,
    dko_show_measured,
    dko_show_references,
):
    # Heatmap of each channel for the heatmap group, on the densities used (linear axis)
    if dko_channel_summary.empty or dko_heatmap_group.value is None:
        _dko_channel_heatmaps = mo.md("")
    else:
        _shown = dko_channel_summary[dko_channel_summary["group"] == dko_heatmap_group.value]
        _channels = [_c for _c in dko_channels.value if _c in set(_shown["channel"])]
        _unit = dko_results[0]["density_unit"]
        # Rows are linear in the main chosen density measure; the others are extra axes
        _main = main_density_unit(_shown, _unit, dko_heatmap_density_axes.value)
        _levels = _shown.groupby("density")[density_column(_main, _unit)].mean().sort_index()
        _row_values, _row_of = density_rows(_levels)
        _snapshot_hour = dko_results[0]["snapshot_hour"]
        _measured = measured_density(dko_results, _main)
        _fixed_level = _levels.index[np.argmin(np.abs(_levels.index.to_numpy() - dko_fixed_density.value))]
        _conversions = density_conversions(_shown, _unit, dko_heatmap_density_axes.value, _main)
        _columns = min(4, max(len(_channels), 1))
        _rows = -(-len(_channels) // _columns)
        # References of this group, as a strip above each map
        _references = (
            dko_reference_channel_summary[
                (dko_reference_channel_summary["group"] == dko_heatmap_group.value)
                & dko_reference_channel_summary["source"].isin(dko_show_references.value)
            ]
            if not dko_reference_channel_summary.empty else pd.DataFrame()
        )
        _sources = [_s for _s in REFERENCE_LABELS if not _references.empty and _s in set(_references["source"])]
        # Each extra density axis takes about 1.2 in left of its heatmap
        _width = (4.4 + 1.2 * len(_conversions)) * _columns
        if _sources:
            _figure, _all_axes = plt.subplots(
                2 * _rows, _columns, figsize=(_width, (4.0 + 0.3 * len(_sources)) * _rows), dpi=130,
                squeeze=False, height_ratios=[0.3 * len(_sources), 3.4] * _rows, layout="constrained",
            )
            _strips, _axes = _all_axes[0::2], _all_axes[1::2]
        else:
            _figure, _axes = plt.subplots(_rows, _columns, figsize=(_width, 3.8 * _rows), dpi=130, squeeze=False)
            _strips = None
        for _index, (_axis, _channel) in enumerate(zip(_axes.flat, _channels)):
            _grid = _shown[_shown["channel"] == _channel].pivot(
                index="density", columns="knockout_hour", values="mean_value"
            ).sort_index().sort_index(axis=1)
            _grid = _grid.groupby(_grid.index.map(_row_of)).mean().sort_index()
            _image = _axis.pcolormesh(
                cell_edges(_grid.columns.to_numpy()), cell_edges(_row_values), _grid.to_numpy(),
                shading="flat", cmap=dko_channel_cmap.value,
            )
            _axis.axvline(dko_fixed_ko.value, color="white", linewidth=0.8, linestyle=":")
            _axis.axhline(_row_values[_row_of[_fixed_level]], color="white", linewidth=0.8, linestyle=":")
            if dko_show_measured.value and _measured is not None:
                _axis.axhline(_measured, color="white", linewidth=0.8, linestyle="--")
            _ticks, _tick_labels = ko_time_ticks(_snapshot_hour)
            _axis.set(
                xlabel="NODAL knockout time (h)",
                xticks=_ticks, xticklabels=_tick_labels, title=_channel,
            )
            add_density_axes(_axis, "y", _main, _conversions, gap=0.3)
            if _strips is not None:
                # The references' channel means at the same KO times, on the same colour scale
                _strip = _strips.flat[_index]
                _ko = _grid.columns.to_numpy()
                draw_reference_strip(
                    _strip, _ko,
                    [
                        _references[(_references["source"] == _source) & (_references["channel"] == _channel)]
                        .set_index("knockout_hour")["mean_value"].reindex(_ko).to_numpy()
                        for _source in _sources
                    ],
                    _sources, _image,
                )
                _strip.set_xlim(_axis.get_xlim())
                _strip.set_xticks([])
                _strip.set_title(_channel)
                _axis.set_title("")
                _figure.colorbar(_image, ax=[_strip, _axis], label="Mean value")
            else:
                _figure.colorbar(_image, ax=_axis, label="Mean value")
        for _axis in list(_axes.flat)[len(_channels):]:
            _axis.set_visible(False)
        for _strip in (list(_strips.flat)[len(_channels):] if _strips is not None else []):
            _strip.set_visible(False)
        _figure.suptitle(
            f"{dko_heatmap_group.value} · channel means at {_snapshot_hour:g}h" + ("; strip: references" if _sources else "")
        )
        if _strips is None:
            _figure.tight_layout()
        _dko_channel_heatmaps = mo.vstack([dko_channel_cmap, _figure])
    _dko_channel_heatmaps
    return


@app.cell(hide_code=True)
def _(dko_prevalence, dko_summary):
    mo.vstack([
        mo.md("### Density × KO CSV exports"),
        mo.hstack([
            mo.download(
                lambda: dko_prevalence.to_csv(index=False).encode("utf-8"),
                filename="nodal_ko_density_prevalence.csv",
                mimetype="text/csv",
                disabled=dko_prevalence.empty,
                label="Download per-run prevalence CSV",
            ),
            mo.download(
                lambda: dko_summary.to_csv(index=False).encode("utf-8"),
                filename="nodal_ko_density_prevalence_summary.csv",
                mimetype="text/csv",
                disabled=dko_summary.empty,
                label="Download grouped summary CSV",
            ),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cell-fate dynamics across intervention and measurement time

    Sweep NODAL-KO time while measuring the cell-type shares (from the cell
    type file above) at every NCA update. Only the number of pixels of each
    marker pattern in each ring is kept per step, rather than full image
    trajectories. After the sweep, use the
    selectors below to explore model and cell-type heatmaps without rerunning.
    """)
    return


@app.cell(hide_code=True)
def _(selection_records):
    _dynamics_model_options = {
        model_option_label(_record): _record["model_id"]
        for _record in selection_records
    }
    dynamics_models = mo.ui.multiselect(
        options=_dynamics_model_options,
        value=list(_dynamics_model_options),
        label="Models",
        full_width=True,
    )
    dynamics_ko_start = mo.ui.number(
        0, 48, value=0, step=0.5, label="First KO time (h)"
    )
    dynamics_ko_stop = mo.ui.number(
        0, 48, value=48, step=0.5, label="Last KO time (h)"
    )
    dynamics_ko_interval = mo.ui.number(
        0.5, 24, value=1, step=0.5, label="KO spacing (h)"
    )
    dynamics_endpoint = mo.ui.number(
        1, 48, value=48, step=1, label="Final measurement time (h)"
    )
    dynamics_replicate = mo.ui.number(
        1, 999, value=1, step=1, label="Replicate"
    )
    dynamics_seed = mo.ui.number(
        0, 2**31 - 1, value=0, step=1, label="Rollout seed"
    )
    run_dynamics_sweep = mo.ui.run_button(label="Run prevalence dynamics sweep")
    mo.vstack([
        dynamics_models,
        mo.hstack([
            dynamics_ko_start,
            dynamics_ko_stop,
            dynamics_ko_interval,
            dynamics_endpoint,
        ]),
        mo.hstack([dynamics_replicate, dynamics_seed, run_dynamics_sweep]),
    ])
    return (
        dynamics_endpoint,
        dynamics_ko_interval,
        dynamics_ko_start,
        dynamics_ko_stop,
        dynamics_models,
        dynamics_replicate,
        dynamics_seed,
        run_dynamics_sweep,
    )


@app.cell(hide_code=True)
def _(
    cell_type_files,
    cell_type_labels,
    cell_type_rings,
    cell_type_rules_by_projection,
    data_root,
    dynamics_endpoint,
    dynamics_ko_interval,
    dynamics_ko_start,
    dynamics_ko_stop,
    dynamics_models,
    dynamics_replicate,
    dynamics_seed,
    fine_resolve_several,
    run_dynamics_sweep,
    selection_records,
    store_root,
):
    if not run_dynamics_sweep.value:
        temporal_prevalence_results = None
        _dynamics_status = mo.md("Configure the sweep, then run it.")
    else:
        if not dynamics_models.value:
            raise ValueError("Choose at least one model.")
        if not data_root.value.strip():
            raise ValueError("Set Data root before running the dynamics sweep.")
        if dynamics_ko_stop.value < dynamics_ko_start.value:
            raise ValueError("Last KO time must be at or after first KO time.")
        if dynamics_ko_stop.value > dynamics_endpoint.value:
            raise ValueError("KO times cannot be later than the final measurement time.")

        _dynamics_registry = ModelRegistry(Path(store_root.value).expanduser())
        _dynamics_root = Path(data_root.value).expanduser()
        _dynamics_records = {
            _record["model_id"]: _record for _record in selection_records
        }
        _dynamics_requested_ko = np.arange(
            float(dynamics_ko_start.value),
            float(dynamics_ko_stop.value)
            + 0.5 * float(dynamics_ko_interval.value),
            float(dynamics_ko_interval.value),
        )
        _dynamics_cell_types = cell_type_labels
        _dynamics_rings = int(cell_type_rings.value)
        _dynamics_results = []
        for _dynamics_model_id in tqdm(
            dynamics_models.value,
            desc="Running prevalence dynamics models",
            unit="model",
        ):
            _dynamics_record = _dynamics_records[_dynamics_model_id]
            _dynamics_projection = _dynamics_record["projection"]
            _dynamics_rules = cell_type_rules_by_projection[_dynamics_projection]
            _dynamics_bundle = _dynamics_registry.get(_dynamics_model_id)
            _dynamics_model = _dynamics_bundle.load_model(
                key=jr.PRNGKey(int(dynamics_seed.value)),
                implementation="portable",
            )
            _dynamics_bins = load_training_histogram_bins(
                _dynamics_bundle, _dynamics_root
            )
            _dynamics_data = load_condition_data(
                _dynamics_bundle,
                _dynamics_root,
                "baseline",
                int(dynamics_replicate.value),
                _dynamics_bins,
            )
            _dynamics_thresholds = [
                _dynamics_rules.thresholds[_dynamics_marker]
                for _dynamics_marker in _dynamics_rules.markers
            ]
            _dynamics_colony = (
                np.asarray(_dynamics_data["boundary"]).squeeze().astype(bool)
            )
            _dynamics_initial = _dynamics_data["initial_state"]
            _dynamics_model_channels = int(_dynamics_model.N_CHANNELS)
            _dynamics_state = jnp.pad(
                jnp.asarray(_dynamics_initial),
                (
                    (0, _dynamics_model_channels - _dynamics_initial.shape[0]),
                    (0, 0),
                    (0, 0),
                ),
            )
            _dynamics_steps_per_12h = int(_dynamics_bundle.config.run.t)
            _dynamics_total_steps = int(round(
                float(dynamics_endpoint.value) * _dynamics_steps_per_12h / 12.0
            ))
            _dynamics_fate_channels = jnp.asarray(marker_channel_indices(
                _dynamics_data["state_channel_names"], _dynamics_rules
            ))
            _dynamics_matrices = []
            _dynamics_realised_ko = []
            for _dynamics_ko_hour in tqdm(
                _dynamics_requested_ko,
                desc=f"KO times [{_dynamics_bundle.id[-8:]}]",
                unit="time",
                leave=False,
            ):
                _dynamics_ko_step = int(round(
                    float(_dynamics_ko_hour) * _dynamics_steps_per_12h / 12.0
                ))
                _dynamics_counts = (
                    rollout_model_with_blocked_channel_pattern_counts(
                        _dynamics_model,
                        _dynamics_state,
                        jnp.asarray(_dynamics_data["boundary"]),
                        _dynamics_bundle.config.trainer.boundary_mode,
                        jr.PRNGKey(int(dynamics_seed.value)),
                        _dynamics_total_steps,
                        _dynamics_data["nodal_channel"],
                        jnp.asarray(_dynamics_ko_step, dtype=jnp.int32),
                        _dynamics_fate_channels,
                        jnp.asarray(_dynamics_thresholds),
                        jnp.asarray(radial_rings(_dynamics_colony, _dynamics_rings)),
                        _dynamics_rings,
                    )
                )
                # Shares at every measurement step, estimated as in the fine sweep.
                _dynamics_matrices.append(np.stack([
                    colony_label_shares(
                        model_stain_counts(_dynamics_step_counts, _dynamics_rules),
                        _dynamics_rules,
                        resolve_several=fine_resolve_several.value,
                    )
                    for _dynamics_step_counts in np.asarray(_dynamics_counts)
                ]))
                _dynamics_realised_ko.append(
                    12.0 * _dynamics_ko_step / _dynamics_steps_per_12h
                )
            _dynamics_results.append({
                "model_id": _dynamics_bundle.id,
                "label": record_label(_dynamics_record),
                "option_label": model_option_label(_dynamics_record),
                "projection": _dynamics_projection,
                "downsample": _dynamics_record["downsample"],
                "initial_condition_replicate": int(dynamics_replicate.value),
                "random_seed": int(dynamics_seed.value),
                "cell_types": _dynamics_cell_types,
                "requested_ko_hours": np.asarray(_dynamics_requested_ko),
                "realised_ko_hours": np.asarray(_dynamics_realised_ko),
                "measurement_hours": (
                    np.arange(_dynamics_total_steps + 1)
                    * 12.0 / _dynamics_steps_per_12h
                ),
                "prevalence": np.stack(_dynamics_matrices),
                "cell_type_file": cell_type_files.value[_dynamics_projection],
            })
        temporal_prevalence_results = tuple(_dynamics_results)
        _dynamics_status = mo.callout(
            f"Completed {len(_dynamics_requested_ko)} KO times for "
            f"{len(temporal_prevalence_results)} models.",
            kind="success",
        )
    _dynamics_status
    return (temporal_prevalence_results,)


@app.cell(hide_code=True)
def _(temporal_prevalence_results):
    _dynamics_export_rows = []
    for _result in temporal_prevalence_results or ():
        for _ko_index, _requested_ko_hour in enumerate(
            _result["requested_ko_hours"]
        ):
            for _measurement_index, _measurement_hour in enumerate(
                _result["measurement_hours"]
            ):
                _dynamics_export_rows.append({
                    "model_realisation_id": _result["model_id"],
                    "model_realisation_label": _result["label"],
                    "projection": _result["projection"],
                    "downsample": _result["downsample"],
                    "initial_condition_replicate": _result[
                        "initial_condition_replicate"
                    ],
                    "random_seed": _result["random_seed"],
                    "requested_knockout_hour": float(_requested_ko_hour),
                    "realised_knockout_hour": float(
                        _result["realised_ko_hours"][_ko_index]
                    ),
                    "measurement_hour": float(_measurement_hour),
                    **{
                        f"{_cell_type.lower()}_prevalence": float(
                            _result["prevalence"][
                                _ko_index, _measurement_index, _cell_index
                            ]
                        )
                        for _cell_index, _cell_type in enumerate(
                            _result["cell_types"]
                        )
                    },
                })
    dynamics_prevalence_export = pd.DataFrame(_dynamics_export_rows)
    _dynamics_export_ready = not dynamics_prevalence_export.empty
    _dynamics_download = mo.download(
            lambda: dynamics_prevalence_export.to_csv(index=False).encode("utf-8"),
            filename="nodal_ko_cell_type_prevalence_dynamics.csv",
            mimetype="text/csv",
            disabled=not _dynamics_export_ready,
            label="Download prevalence dynamics CSV",
    )
    mo.vstack([
        mo.md("### Dynamics CSV export"),
        mo.md(
            "Download is ready."
            if _dynamics_export_ready
            else "Run the prevalence dynamics sweep to enable this download."
        ),
        _dynamics_download,
    ])
    return


@app.cell(hide_code=True)
def _(temporal_prevalence_results):
    _heatmap_model_options = {
        _result["option_label"]: _result["model_id"]
        for _result in (temporal_prevalence_results or ())
    }
    dynamics_heatmap_model = mo.ui.dropdown(
        options=_heatmap_model_options,
        value=next(iter(_heatmap_model_options), None),
        label="Heatmap model",
        full_width=True,
    )
    _heatmap_cell_options = (
        list(temporal_prevalence_results[0]["cell_types"])
        if temporal_prevalence_results else []
    )
    dynamics_heatmap_cell_type = mo.ui.dropdown(
        options=_heatmap_cell_options,
        value=next(iter(_heatmap_cell_options), None),
        label="Heatmap cell type",
        full_width=True,
    )
    dynamics_heatmap_cmap = mo.ui.dropdown(COLOUR_MAPS, value="viridis", label="Colour map")
    mo.hstack([dynamics_heatmap_model, dynamics_heatmap_cell_type, dynamics_heatmap_cmap])
    return dynamics_heatmap_cell_type, dynamics_heatmap_cmap, dynamics_heatmap_model


@app.cell(hide_code=True)
def _(
    dynamics_heatmap_cell_type,
    dynamics_heatmap_cmap,
    dynamics_heatmap_model,
    temporal_prevalence_results,
):
    if (
        temporal_prevalence_results is None
        or dynamics_heatmap_model.value is None
        or dynamics_heatmap_cell_type.value is None
    ):
        _dynamics_heatmap_view = mo.md("")
    else:
        _heatmap_result = next(
            _result for _result in temporal_prevalence_results
            if _result["model_id"] == dynamics_heatmap_model.value
        )
        _heatmap_cell_index = _heatmap_result["cell_types"].index(
            dynamics_heatmap_cell_type.value
        )
        _heatmap_values = _heatmap_result["prevalence"][:, :, _heatmap_cell_index].T
        _heatmap_ko_hours = _heatmap_result["requested_ko_hours"]
        _heatmap_measurement_hours = _heatmap_result["measurement_hours"]
        _heatmap_figure, _heatmap_axis = plt.subplots(figsize=(10, 6), dpi=150)
        _heatmap_image = _heatmap_axis.imshow(
            _heatmap_values,
            origin="lower",
            aspect="auto",
            cmap=dynamics_heatmap_cmap.value,
            vmin=0.0,
            vmax=1.0,
            extent=(
                float(_heatmap_ko_hours[0]),
                float(_heatmap_ko_hours[-1]),
                float(_heatmap_measurement_hours[0]),
                float(_heatmap_measurement_hours[-1]),
            ),
        )
        _heatmap_axis.set(
            xlabel="NODAL knockout time (h)",
            ylabel="Measurement time (h)",
            title=(
                f"{_heatmap_result['label']} · {_heatmap_result['projection']}\n"
                f"{dynamics_heatmap_cell_type.value} area fraction"
            ),
            yticks=[0, 12, 24, 36, 48],
        )
        # A KO at the last measurement changes none of them: the baseline.
        _heatmap_axis.set_xticks(
            *ko_time_ticks(_heatmap_measurement_hours[-1])
        )
        _heatmap_figure.colorbar(
            _heatmap_image, ax=_heatmap_axis, label="Fraction of colony pixels"
        )
        _heatmap_figure.tight_layout()
        _dynamics_heatmap_view = _heatmap_figure
    _dynamics_heatmap_view
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Composite CMY snapshots and video

    Render a full-resolution trajectory from one selected model and initial
    condition as three horizontal CMY panels: SOX2–TBXT–SOX17,
    CER1–LEFTY2–NODAL, and FOXA2–LEF1–SMAD23. The shaped conditions resample
    the measured 0h channel distributions inside the selected mask.
    """)
    return


@app.cell(hide_code=True)
def _(selection_records):
    _render_model_options = {
        model_option_label(_record): _record["model_id"]
        for _record in selection_records
    }
    composite_model = mo.ui.dropdown(
        options=_render_model_options,
        value=next(iter(_render_model_options), None),
        label="Model",
        full_width=True,
    )
    composite_condition = mo.ui.dropdown(
        options={
            "Baseline": "baseline",
            "NODAL KO at 0h": "ko_0h",
            "NODAL KO at 24h": "ko_24h",
        },
        value="Baseline",
        label="Intervention",
    )
    composite_shape = mo.ui.dropdown(
        options={
            "Measured colony": "colony",
            "Circle": "circle",
            "Triangle": "triangle",
            "Ellipse": "ellipse",
            "Full field": "full",
        },
        value="Measured colony",
        label="Initial-condition shape",
    )
    composite_radius = mo.ui.slider(
        0.25,
        1.0,
        value=0.8,
        step=0.05,
        label="Shape scale",
        show_value=True,
    )
    composite_replicate = mo.ui.number(
        1, 999, value=1, step=1, label="Source replicate (one-based)"
    )
    composite_seed = mo.ui.number(
        0, 2**31 - 1, value=42, step=1, label="Rollout seed"
    )
    composite_fps = mo.ui.number(1, 60, value=30, step=1, label="Video FPS")
    composite_duration = mo.ui.number(
        1, 120, value=20, step=1, label="Video duration (seconds)"
    )
    composite_scale = mo.ui.number(
        1, 12, value=4, step=1, label="Video pixel scale"
    )
    composite_output = mo.ui.text(
        str(repository_root / "output" / "nodal_ko_videos"),
        label="Video output directory",
        full_width=True,
    )
    run_composite_render = mo.ui.run_button(label="Render snapshots and MP4")
    mo.vstack([
        composite_model,
        mo.hstack([composite_condition, composite_shape, composite_radius]),
        mo.hstack([composite_replicate, composite_seed]),
        mo.hstack([composite_fps, composite_duration, composite_scale]),
        composite_output,
        run_composite_render,
    ])
    return (
        composite_condition,
        composite_duration,
        composite_fps,
        composite_model,
        composite_output,
        composite_radius,
        composite_replicate,
        composite_scale,
        composite_seed,
        composite_shape,
        run_composite_render,
    )


@app.cell(hide_code=True)
def _(
    composite_condition,
    composite_duration,
    composite_fps,
    composite_model,
    composite_output,
    composite_radius,
    composite_replicate,
    composite_scale,
    composite_seed,
    composite_shape,
    data_root,
    run_composite_render,
    store_root,
):
    if not run_composite_render.value:
        composite_render_result = None
        _composite_render_view = mo.md(
            "Choose a model and initial condition, then render the trajectory."
        )
    else:
        if composite_model.value is None:
            raise ValueError("Choose a model from a non-empty registry selection.")
        if not data_root.value.strip():
            raise ValueError("Set Data root before rendering a trajectory.")

        _composite_registry = ModelRegistry(Path(store_root.value).expanduser())
        _composite_bundle = _composite_registry.get(composite_model.value)
        _composite_key = jr.PRNGKey(int(composite_seed.value))
        _composite_model_object = _composite_bundle.load_model(
            key=_composite_key,
            implementation="portable",
        )
        _composite_data_root = Path(data_root.value).expanduser()
        _composite_bins = load_training_histogram_bins(
            _composite_bundle, _composite_data_root
        )
        _composite_data = load_condition_data(
            _composite_bundle,
            _composite_data_root,
            "baseline",
            int(composite_replicate.value),
            _composite_bins,
        )
        _composite_initial, _composite_boundary, _composite_mask = (
            make_render_initial_condition(
                _composite_data["initial_state"],
                _composite_data["boundary"],
                composite_shape.value,
                float(composite_radius.value),
                jr.fold_in(_composite_key, 1),
            )
        )
        _composite_channel_count = int(_composite_model_object.N_CHANNELS)
        if _composite_initial.shape[0] > _composite_channel_count:
            raise ValueError("The selected model has fewer channels than its input data.")
        _composite_state = jnp.pad(
            _composite_initial,
            ((0, _composite_channel_count - _composite_initial.shape[0]), (0, 0), (0, 0)),
        )
        _composite_steps_per_observation = int(_composite_bundle.config.run.t)
        _composite_hours = tuple(
            float(_hour)
            for _hour in _composite_bundle.config.data.micropattern.timesteps
        )
        _composite_total_steps = (
            (len(_composite_hours) - 1) * _composite_steps_per_observation
        )
        _composite_boundary_mode = _composite_bundle.config.trainer.boundary_mode
        if _composite_boundary_mode == "none":
            _composite_boundary_callback = no_boundary()
        else:
            _composite_boundary_callback = build_boundary_callback(
                _composite_boundary_mode, _composite_boundary
            )

        if composite_condition.value == "baseline":
            _composite_states = rollout_model(
                _composite_model_object,
                _composite_state,
                _composite_boundary_callback,
                jr.fold_in(_composite_key, 2),
                _composite_total_steps,
            )
        else:
            _composite_ko_hour = 0 if composite_condition.value == "ko_0h" else 24
            _composite_ko_step = (
                _composite_ko_hour // 12
            ) * _composite_steps_per_observation
            _composite_states = rollout_model_with_blocked_channel(
                _composite_model_object,
                _composite_state,
                _composite_boundary_callback,
                jr.fold_in(_composite_key, 2),
                _composite_total_steps,
                _composite_data["nodal_channel"],
                jnp.asarray(_composite_ko_step, dtype=jnp.int32),
            )
        _composite_rgb = render_cmy_composite(
            np.asarray(_composite_states),
            _composite_data["state_channel_names"],
        )
        _composite_output_path = Path(composite_output.value).expanduser()
        _composite_output_path.mkdir(parents=True, exist_ok=True)
        _composite_safe_id = "".join(
            _character if _character.isalnum() or _character in "-_" else "_"
            for _character in composite_model.value
        )
        _composite_filename = _composite_output_path / (
            f"{_composite_safe_id}_{composite_condition.value}_"
            f"{composite_shape.value}_cmy.mp4"
        )
        save_to_video_rgb(
            _composite_rgb,
            str(_composite_filename),
            fps=int(composite_fps.value),
            duration=float(composite_duration.value),
            SCALE_UP=int(composite_scale.value),
        )
        _composite_snapshot_indices = [
            _index * _composite_steps_per_observation
            for _index in range(len(_composite_hours))
        ]
        _composite_figure, _composite_axes = plt.subplots(
            1,
            len(_composite_snapshot_indices),
            figsize=(min(24, 5 * len(_composite_snapshot_indices)), 3),
            squeeze=False,
        )
        for _composite_axis, _composite_index in zip(
            _composite_axes[0],
            _composite_snapshot_indices,
        ):
            _composite_axis.imshow(_composite_rgb[_composite_index])
            _composite_axis.set_axis_off()
        _composite_figure.subplots_adjust(0, 0, 1, 1, wspace=0.02)
        composite_render_result = {
            "model_id": composite_model.value,
            "condition": composite_condition.value,
            "shape": composite_shape.value,
            "mask": _composite_mask,
            "states": np.asarray(_composite_states),
            "rgb_frames": _composite_rgb,
            "snapshot_hours": _composite_hours,
            "video_path": str(_composite_filename),
        }
        _composite_render_view = mo.vstack([
            mo.callout(
                f"Saved `{_composite_filename}` ({len(_composite_rgb)} frames).",
                kind="success",
            ),
            _composite_figure,
        ])
    _composite_render_view
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generative trajectory ensemble

    Render nine independently seeded predictions from one model as a 3×3 grid.
    Every tile is a SOX2–TBXT–SOX17 CMY composite at the selected developmental
    time. Initial marker intensities are independently resampled within the
    measured colony for every trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(selection_records):
    _generative_model_options = {
        model_option_label(_record): _record["model_id"]
        for _record in selection_records
    }
    generative_model = mo.ui.dropdown(
        options=_generative_model_options,
        value=next(iter(_generative_model_options), None),
        label="Model",
        full_width=True,
    )
    generative_time = mo.ui.number(
        1,
        48,
        value=48,
        step=1,
        label="Rendered developmental time (h)",
    )
    generative_seed = mo.ui.number(
        0,
        2**31 - 10,
        value=42,
        step=1,
        label="First random seed",
    )
    generative_replicate = mo.ui.number(
        1,
        999,
        value=1,
        step=1,
        label="First source replicate (one-based)",
    )
    generative_vary_replicates = mo.ui.checkbox(
        value=False,
        label="Use nine consecutive source replicates",
    )
    generative_renormalise_channels = mo.ui.checkbox(
        value=False,
        label="Renormalise each CMY channel to [0, 1]",
    )
    run_generative_grid = mo.ui.run_button(label="Render 3×3 prediction grid")
    mo.vstack([
        generative_model,
        mo.hstack([generative_time, generative_seed]),
        mo.hstack([generative_replicate, generative_vary_replicates]),
        generative_renormalise_channels,
        run_generative_grid,
    ])
    return (
        generative_model,
        generative_renormalise_channels,
        generative_replicate,
        generative_seed,
        generative_time,
        generative_vary_replicates,
        run_generative_grid,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    generative_model,
    generative_renormalise_channels,
    generative_replicate,
    generative_seed,
    generative_time,
    generative_vary_replicates,
    run_generative_grid,
    store_root,
):
    if not run_generative_grid.value:
        generative_grid_result = None
        _generative_grid_view = mo.md(
            "Choose a model and render nine generative predictions."
        )
    else:
        if generative_model.value is None:
            raise ValueError("Choose a model from a non-empty registry selection.")
        if not data_root.value.strip():
            raise ValueError("Set Data root before rendering predictions.")

        _generative_registry = ModelRegistry(Path(store_root.value).expanduser())
        _generative_bundle = _generative_registry.get(generative_model.value)
        _generative_first_key = jr.PRNGKey(int(generative_seed.value))
        _generative_model_object = _generative_bundle.load_model(
            key=_generative_first_key,
            implementation="portable",
        )
        _generative_data_root = Path(data_root.value).expanduser()
        _generative_bins = load_training_histogram_bins(
            _generative_bundle, _generative_data_root
        )
        _generative_steps_per_12h = int(_generative_bundle.config.run.t)
        _generative_total_steps = int(round(
            float(generative_time.value) * _generative_steps_per_12h / 12.0
        ))
        _generative_observation_steps = jnp.asarray(
            [_generative_total_steps], dtype=jnp.int32
        )
        _generative_boundary_mode = _generative_bundle.config.trainer.boundary_mode
        _generative_data_cache = {}
        _generative_images = []
        _generative_trajectories = []
        for _generative_index in tqdm(
            range(9), desc="Rendering generative grid", unit="trajectory"
        ):
            _generative_source_replicate = int(generative_replicate.value) + (
                _generative_index if generative_vary_replicates.value else 0
            )
            if _generative_source_replicate not in _generative_data_cache:
                _generative_data_cache[_generative_source_replicate] = (
                    load_condition_data(
                        _generative_bundle,
                        _generative_data_root,
                        "baseline",
                        _generative_source_replicate,
                        _generative_bins,
                    )
                )
            _generative_data = _generative_data_cache[
                _generative_source_replicate
            ]
            _generative_key = jr.PRNGKey(
                int(generative_seed.value) + _generative_index
            )
            _generative_initial, _generative_boundary, _ = (
                make_render_initial_condition(
                    _generative_data["initial_state"],
                    _generative_data["boundary"],
                    "colony",
                    1.0,
                    jr.fold_in(_generative_key, 1),
                )
            )
            _generative_channel_count = int(_generative_model_object.N_CHANNELS)
            if _generative_initial.shape[0] > _generative_channel_count:
                raise ValueError(
                    "The selected model has fewer channels than its input data."
                )
            _generative_state = jnp.pad(
                _generative_initial,
                (
                    (0, _generative_channel_count - _generative_initial.shape[0]),
                    (0, 0),
                    (0, 0),
                ),
            )
            _generative_selected = rollout_model_sampled(
                _generative_model_object,
                _generative_state,
                _generative_boundary,
                _generative_boundary_mode,
                jr.fold_in(_generative_key, 2),
                _generative_total_steps,
                _generative_observation_steps,
            )
            _generative_selected_array = np.asarray(_generative_selected)
            _generative_images.append(render_fate_cmy(
                _generative_selected_array,
                _generative_data["state_channel_names"],
                renormalise=generative_renormalise_channels.value,
            )[0])
            _generative_trajectories.append({
                "seed": int(generative_seed.value) + _generative_index,
                "replicate": _generative_source_replicate,
                "state": _generative_selected_array[0],
            })

        _generative_figure, _generative_axes = plt.subplots(
            3, 3, figsize=(9, 9), squeeze=False
        )
        for _generative_axis, _generative_image in zip(
            _generative_axes.reshape(-1), _generative_images
        ):
            _generative_axis.imshow(_generative_image, vmin=0.0, vmax=1.0)
            _generative_axis.set_axis_off()
        _generative_figure.subplots_adjust(
            left=0.0,
            right=1.0,
            bottom=0.0,
            top=1.0,
            wspace=0.02,
            hspace=0.02,
        )
        generative_grid_result = {
            "model_id": generative_model.value,
            "time_hour": float(generative_time.value),
            "channels_renormalised": generative_renormalise_channels.value,
            "images": np.stack(_generative_images),
            "trajectories": tuple(_generative_trajectories),
            "figure": _generative_figure,
        }
        _generative_grid_view = _generative_figure
    _generative_grid_view
    return


if __name__ == "__main__":
    app.run()
