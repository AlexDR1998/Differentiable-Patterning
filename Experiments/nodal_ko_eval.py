# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
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

    marimo run Experiments/nodal_ko_eval.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    from dataclasses import replace
    import os
    import sys
    import time
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    from NCA.registry import ModelRegistry
    from NCA.trainer.intervention import (
        rollout_model,
        rollout_model_sampled,
        rollout_model_with_blocked_channel,
        rollout_model_with_blocked_channel_sampled,
        rollout_model_with_blocked_channel_prevalence,
    )
    from Common.model.boundary import hard_boundary, model_boundary, no_boundary
    from Common.save_to_video import save_to_video_rgb

    load_dotenv(repository_root / ".env", override=False)

    FATE_MARKERS = ("TBXT", "SOX17", "SOX2", "FOXA2")
    CELL_TYPES = ("Notochord", "Endoderm", "Mesoderm")
    DEFAULT_FATE_RULES = {
        "Notochord": {"TBXT": "high", "SOX17": "low", "SOX2": "low", "FOXA2": "high"},
        "Endoderm": {"TBXT": "any", "SOX17": "high", "SOX2": "any", "FOXA2": "any"},
        "Mesoderm": {"TBXT": "high", "SOX17": "low", "SOX2": "high", "FOXA2": "low"},
    }

    def fate_rule_mask(high_by_marker, rule):
        """Return the pixels satisfying a configurable marker-state rule."""
        conditions = [
            high_by_marker[marker] if state == "high" else ~high_by_marker[marker]
            for marker, state in rule.items()
            if state != "any"
        ]
        if not conditions:
            return np.ones_like(next(iter(high_by_marker.values())), dtype=bool)
        return np.logical_and.reduce(conditions)

    def fate_rule_warnings(rules):
        """Describe exact or potentially overlapping categorical definitions."""
        warnings = []
        for left_index, left in enumerate(CELL_TYPES):
            for right in CELL_TYPES[left_index + 1:]:
                left_rule, right_rule = rules[left], rules[right]
                if left_rule == right_rule:
                    warnings.append(f"{left} and {right} have duplicate definitions.")
                    continue
                conflicts = any(
                    left_rule[marker] != "any"
                    and right_rule[marker] != "any"
                    and left_rule[marker] != right_rule[marker]
                    for marker in FATE_MARKERS
                )
                if not conflicts:
                    warnings.append(
                        f"{left} and {right} can classify the same cells; summed "
                        "prevalence will not be normalised."
                    )
        return warnings

    def fate_rule_matrix(rules):
        encoding = {"low": -1, "any": 0, "high": 1}
        return np.asarray([
            [encoding[rules[cell_type][marker]] for marker in FATE_MARKERS]
            for cell_type in CELL_TYPES
        ], dtype=np.int8)

    def load_training_histogram_bins(bundle, root):
        """Fit scaling once from the model's configured baseline train split."""

        from Experiments.micropatterns.config_helpers import load_data

        baseline_config = replace(
            bundle.config.data,
            intervention=replace(
                bundle.config.data.intervention,
                curriculum=("baseline",),
            ),
        )
        train_replicates = baseline_config.micropattern.get(
            "train_replicates", None
        )
        if train_replicates is None:
            train_replicates = tuple(range(1, baseline_config.batches + 1))
        baseline_data = load_data(
            baseline_config,
            impath=str(root / "260726_nca_dataset"),
            replicate_indices=tuple(index - 1 for index in train_replicates),
            pool_copies_override=1,
        )
        return baseline_data[1]["histogram_bins"]

    def load_condition_data(bundle, root, condition, replicate, histogram_bins):
        """Load one matching experimental condition using training scaling."""

        if bundle.config.data.dataset != "micropatterns_260726":
            raise ValueError(
                f"{bundle.id} uses {bundle.config.data.dataset!r}; NODAL-KO evaluation "
                "requires 'micropatterns_260726'."
            )

        from Experiments.micropatterns.config_helpers import load_data

        data_config = replace(
            bundle.config.data,
            intervention=replace(
                bundle.config.data.intervention,
                curriculum=(condition,),
            ),
        )
        data_path = root / "260726_nca_dataset"
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"NODAL-KO dataset not found at {data_path}. Set Data root to the "
                "directory containing 260726_nca_dataset."
            )

        loaded_data = load_data(
            data_config,
            impath=str(data_path),
            replicate_indices=(replicate - 1,),
            histogram_bins=histogram_bins,
            pool_copies_override=1,
        )
        data, aux, channel_names, boundary, _, _ = loaded_data
        channel_schema = aux.get("channel_schema")
        if channel_schema is None or "NODAL" not in channel_schema.state_channels:
            raise ValueError(f"{bundle.id} has no NODAL channel in its data schema.")

        initial_measurements = np.asarray(data)[0, 0]
        initial_state = initial_measurements[
            np.asarray(channel_schema.primary_measurements)
        ]
        output_channels = tuple(channel_schema.target_to_state)
        displayed_count = len(output_channels)
        names = list(channel_names[:displayed_count])
        names.extend(
            f"Channel {index + 1}"
            for index in range(len(names), displayed_count)
        )
        return {
            "initial_state": initial_state,
            "boundary": np.asarray(boundary)[0],
            "target": np.asarray(data)[0],
            "target_state": np.asarray(data)[0][
                :, np.asarray(channel_schema.primary_measurements)
            ],
            "output_channels": output_channels,
            "channel_names": tuple(names),
            "nodal_channel": channel_schema.state_channels.index("NODAL"),
            "state_channel_names": tuple(channel_schema.state_channels),
        }

    def evaluate_condition(model, bundle, condition_data, condition, seed):
        """Roll out one model, blocking recurrent NODAL reads when requested."""

        initial = condition_data["initial_state"]
        channels = int(model.N_CHANNELS)
        if initial.shape[0] > channels:
            raise ValueError(
                f"{bundle.id} has {channels} state channels but its input has "
                f"{initial.shape[0]}."
            )
        state = jnp.pad(
            jnp.asarray(initial),
            ((0, channels - initial.shape[0]), (0, 0), (0, 0)),
        )

        boundary_state = jnp.asarray(condition_data["boundary"])
        boundary_mode = bundle.config.trainer.get("boundary_mode", "soft")

        time_labels = tuple(
            f"{int(hour)}h" for hour in bundle.config.data.micropattern.timesteps
        )
        steps_per_observation = int(bundle.config.run.t)
        frame_indices = tuple(
            time_index * steps_per_observation
            for time_index in range(len(time_labels))
        )
        total_steps = frame_indices[-1]
        observation_steps = jnp.asarray(frame_indices, dtype=jnp.int32)
        rollout_key = jr.PRNGKey(seed)

        if condition == "baseline":
            selected_states = rollout_model_sampled(
                model,
                state,
                boundary_state,
                boundary_mode,
                rollout_key,
                total_steps,
                observation_steps,
            )
        else:
            knockout_hour = 0 if condition == "ko_0h" else 24
            knockout_step = (knockout_hour // 12) * steps_per_observation
            selected_states = rollout_model_with_blocked_channel_sampled(
                model,
                state,
                boundary_state,
                boundary_mode,
                rollout_key,
                total_steps,
                condition_data["nodal_channel"],
                jnp.asarray(knockout_step, dtype=jnp.int32),
                observation_steps,
            )

        prediction = np.asarray(selected_states)[
            :, condition_data["output_channels"]
        ]
        return prediction, time_labels

    def make_render_initial_condition(initial, boundary, shape, radius, key):
        """Resample the measured 0h state inside a requested colony shape."""

        source_mask = np.asarray(boundary).squeeze().astype(bool)
        height, width = source_mask.shape
        y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        if shape == "colony":
            render_mask = source_mask
        elif shape == "circle":
            render_mask = x**2 + y**2 <= float(radius) ** 2
        elif shape == "ellipse":
            render_mask = (x / float(radius)) ** 2 + (y / (0.62 * float(radius))) ** 2 <= 1.0
        elif shape == "triangle":
            scaled_y = y / float(radius)
            scaled_x = x / float(radius)
            render_mask = (
                (scaled_y >= -0.72)
                & (scaled_y <= 0.85)
                & (np.abs(scaled_x) <= (0.85 - scaled_y) / 1.57)
            )
        elif shape == "full":
            render_mask = np.ones((height, width), dtype=bool)
        else:
            raise ValueError(f"Unknown render shape {shape!r}.")

        source_values = jnp.asarray(initial)[:, source_mask]
        if source_values.shape[1] == 0:
            raise ValueError("The source colony mask contains no pixels.")
        sample_indices = jr.randint(
            key,
            (height, width),
            0,
            source_values.shape[1],
        )
        sampled = source_values[:, sample_indices.reshape(-1)]
        sampled = sampled.reshape(initial.shape[0], height, width)
        shaped_initial = jnp.where(jnp.asarray(render_mask)[None], sampled, 0.0)
        shaped_boundary = jnp.broadcast_to(
            jnp.asarray(render_mask, dtype=jnp.asarray(boundary).dtype),
            jnp.asarray(boundary).shape,
        )
        return shaped_initial, shaped_boundary, render_mask

    def render_cmy_composite(states, state_channel_names):
        """Build the thesis-style row of three CMY composite panels."""

        triplets = (
            ("SOX2", "TBXT", "SOX17"),
            ("CER1", "LEFTY", "NODAL"),
            ("FOXA2", "LEF1", "SMAD23"),
        )
        missing = sorted(
            {name for triplet in triplets for name in triplet}
            .difference(state_channel_names)
        )
        if missing:
            raise ValueError(
                "Three-panel CMY rendering requires state channels: "
                + ", ".join(missing)
            )

        panels = []
        for cyan_name, magenta_name, yellow_name in triplets:
            cyan, magenta, yellow = (
                np.clip(states[:, state_channel_names.index(name)], 0.0, 1.0)
                for name in (cyan_name, magenta_name, yellow_name)
            )
            panels.append(np.stack(
                (
                    0.5 * (magenta + yellow),
                    0.5 * (cyan + yellow),
                    0.5 * (cyan + magenta),
                ),
                axis=-1,
            ))
        return np.concatenate(panels, axis=2)

    def render_fate_cmy(states, state_channel_names, renormalise=False):
        """Render SOX2 cyan, TBXT magenta, and SOX17 yellow."""

        channels = [
            np.clip(states[:, state_channel_names.index(name)], 0.0, 1.0)
            for name in ("SOX2", "TBXT", "SOX17")
        ]
        if renormalise:
            normalised_channels = []
            for channel in channels:
                channel_min = np.min(channel, axis=(-2, -1), keepdims=True)
                channel_max = np.max(channel, axis=(-2, -1), keepdims=True)
                channel_range = channel_max - channel_min
                normalised_channels.append(np.where(
                    channel_range > 0,
                    (channel - channel_min) / np.maximum(channel_range, 1e-12),
                    0.0,
                ))
            channels = normalised_channels
        cyan, magenta, yellow = channels
        return np.stack(
            (
                0.5 * (magenta + yellow),
                0.5 * (cyan + yellow),
                0.5 * (cyan + magenta),
            ),
            axis=-1,
        )

    def parse_one_based_indices(specification):
        """Parse comma-separated indices and inclusive ranges such as 1,3-5."""

        indices = []
        for part in specification.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, stop_text = part.split("-", 1)
                start, stop = int(start_text), int(stop_text)
                if stop < start:
                    raise ValueError(f"Invalid descending replicate range {part!r}.")
                indices.extend(range(start, stop + 1))
            else:
                indices.append(int(part))
        indices = list(dict.fromkeys(indices))
        if not indices or any(index < 1 for index in indices):
            raise ValueError("Replicates must contain positive one-based indices.")
        return tuple(indices)


@app.cell(hide_code=True)
def _():
    import matplotlib.style
    matplotlib.style.use(
        "default"
    )
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
def _(selection_path):
    _path = Path(selection_path.value).expanduser()
    if not selection_path.value.strip():
        selection_records = ()
        _preview = mo.md("Choose an exported model selection YAML file.")
    elif not _path.is_file():
        selection_records = ()
        _preview = mo.callout(f"Selection file not found: `{_path}`", kind="danger")
    else:
        _document = OmegaConf.to_container(OmegaConf.load(_path), resolve=False)
        if not isinstance(_document, dict) or _document.get("schema_version") != 1:
            raise ValueError("Expected a model registry export with schema_version: 1")
        selection_records = tuple(_document.get("models", ()))
        _columns = [
            "model_id", "alias", "display_name", "notes", "family",
            "experiment", "best_loss", "seed",
        ]
        _preview_data = pd.DataFrame(selection_records).reindex(columns=_columns)
        _preview = mo.vstack([
            mo.md(f"## Selected models ({len(selection_records)})"),
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
            _label = (
                _record.get("notes")
                or _record.get("alias")
                or _record.get("display_name")
                or _bundle.id
            )
            for _condition_index, _condition in enumerate(
                evaluation_conditions.value
            ):
                _effective_data_config = replace(
                    _bundle.config.data,
                    intervention=replace(
                        _bundle.config.data.intervention,
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
                        intervention=replace(
                            _bundle.config.data.intervention,
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
                    "registry_record": _record,
                    "condition": _condition,
                    "replicate": int(evaluation_replicate.value),
                    "prediction": _prediction,
                    "target": _condition_data["target"],
                    "channel_names": _condition_data["channel_names"],
                    "time_labels": _time_labels,
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
                    "condition": _result["condition"],
                    "replicate": _result["replicate"],
                    "time": _result["time_labels"][_time_index],
                    "mse": float(np.nanmean(_squared_error[_time_index])),
                })
        evaluation_metrics = pd.DataFrame(_metric_rows)
        _metrics_view = mo.vstack([
            mo.md("## Starter metrics"),
            mo.ui.table(evaluation_metrics, selection=None, page_size=15),
        ])
    _metrics_view
    return


@app.cell(hide_code=True)
def _(evaluation_results):
    if evaluation_results is None:
        _model_options = {}
        _group_options = []
        _channel_options = []
        _time_options = []
    else:
        _models_by_id = {
            _result["model_id"]: _result["label"]
            for _result in evaluation_results
        }
        _model_options = {
            f"{_label} [{_model_id[-8:]}]": _model_id
            for _model_id, _label in _models_by_id.items()
        }
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
        display_models,
        mo.hstack([channel_selection_mode, display_groups, display_channels]),
        mo.hstack([display_times, display_source]),
    ])
    return (
        channel_selection_mode,
        display_channels,
        display_groups,
        display_models,
        display_source,
        display_times,
    )


@app.cell(hide_code=True)
def _(
    channel_selection_mode,
    display_channels,
    display_groups,
    display_models,
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
            if _result["model_id"] not in display_models.value:
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
            _axis.set_title(f"{_result['label']} · {_result['condition']}")
            _axis.tick_params(length=0)
            _figure.tight_layout()
            _figures.append(_figure)
        _figure_view = mo.vstack(_figures) if _figures else mo.md(
            "The selected channel is not available in these results."
        )
    _figure_view
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


@app.cell(hide_code=True)
def _(selection_records):
    _fine_model_options = {
        str(
            _record.get("notes")
            or _record.get("alias")
            or _record.get("display_name")
            or _record["model_id"]
        ) + f" [{_record['model_id'][-8:]}]": _record["model_id"]
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
        label="Include true KO 0h, KO 24h, and baseline prevalence",
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
            _fine_true_snapshots = None
            if _fine_model_index == 0 and fine_include_true_data.value:
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
            _fine_boundary_mode = _fine_bundle.config.trainer.get("boundary_mode", "soft")
            _fine_curriculum = tuple(
                _fine_bundle.config.data.intervention.curriculum or ("baseline",)
            )
            _fine_label = str(
                _fine_record.get("notes")
                or _fine_record.get("alias")
                or _fine_record.get("display_name")
                or _fine_bundle.id
            )
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
                    "groups": {
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
                        _fine_true_snapshots
                        if _fine_model_index == 0 and _fine_run_index == 0
                        else None
                    ),
                    "true_channel_names": (
                        _fine_data["channel_names"]
                        if _fine_model_index == 0 and _fine_run_index == 0
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
def _(fine_ko_results):
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
    fine_channel_thresholds = mo.ui.dictionary({
        _marker: mo.ui.slider(
            0.05,
            0.95,
            value=0.3,
            step=0.05,
            label=f"{_marker} threshold fraction",
            full_width=True,
        )
        for _marker in ("TBXT", "SOX17", "SOX2", "FOXA2")
    })
    fine_channel_percentiles = mo.ui.dictionary({
        _marker: mo.ui.slider(
            90.0,
            100.0,
            value=99.0,
            step=0.5,
            label=f"{_marker} reference percentile",
            full_width=True,
        )
        for _marker in ("TBXT", "SOX17", "SOX2", "FOXA2")
    })
    fine_cell_type_rules = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            _marker: mo.ui.dropdown(
                {"High": "high", "Low": "low", "Irrespective": "any"},
                value={"high": "High", "low": "Low", "any": "Irrespective"}[
                    DEFAULT_FATE_RULES[_cell_type][_marker]
                ],
                label=_marker,
                full_width=True,
            )
            for _marker in FATE_MARKERS
        })
        for _cell_type in CELL_TYPES
    })
    fine_group_by = mo.ui.dropdown(
        {
            "Training curriculum": "curriculum",
            "Experiment": "experiment",
            "Family": "family",
            "Notes": "notes",
            "Alias": "alias",
            "Each model separately": "individual",
            "All models together": "all",
        },
        value="Training curriculum",
        label="Group replicate models by",
        full_width=True,
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
        options=[*CELL_TYPES, "Other"],
        value=list(CELL_TYPES),
        label="Displayed cell types",
        full_width=True,
    )
    fine_scale_cell_maps_by_lmbr = mo.ui.checkbox(
        value=False,
        label="Scale cell-type snapshot brightness by LMBR",
    )
    mo.vstack([
        mo.hstack([fine_snapshot_channels, fine_snapshot_stride]),
        mo.md("### Per-marker expression thresholds"),
        fine_channel_thresholds,
        mo.md("### Per-marker reference percentiles"),
        fine_channel_percentiles,
        mo.md("### Cell-type marker selections"),
        fine_cell_type_rules,
        fine_display_cell_types,
        fine_scale_cell_maps_by_lmbr,
        mo.hstack([fine_group_by, fine_uncertainty]),
    ])
    return (
        fine_channel_percentiles,
        fine_channel_thresholds,
        fine_cell_type_rules,
        fine_display_cell_types,
        fine_group_by,
        fine_scale_cell_maps_by_lmbr,
        fine_snapshot_channels,
        fine_snapshot_stride,
        fine_uncertainty,
    )


@app.cell(hide_code=True)
def _(fine_cell_type_rules):
    _fine_rule_warnings = fate_rule_warnings(fine_cell_type_rules.value)
    if _fine_rule_warnings:
        mo.callout("\n\n".join(_fine_rule_warnings), kind="warn")
    else:
        mo.callout("Cell-type definitions are mutually exclusive.", kind="success")
    return


@app.cell(hide_code=True)
def _(
    fine_cell_type_rules,
    fine_channel_percentiles,
    fine_channel_thresholds,
    fine_display_cell_types,
    fine_display_groups,
    fine_group_by,
    fine_ko_results,
    fine_uncertainty,
):
    _fine_required_markers = ("TBXT", "SOX17", "SOX2", "FOXA2")
    _fine_rules = fine_cell_type_rules.value
    if fine_ko_results is None:
        fine_cell_type_prevalence = pd.DataFrame()
        fine_true_cell_type_prevalence = pd.DataFrame()
        fine_prevalence_export = pd.DataFrame()
        fine_prevalence_summary = pd.DataFrame()
        fine_cell_type_maps = ()
        _fine_prevalence_view = mo.md("")
    elif not all(
        _marker in _result["channel_names"]
        for _result in fine_ko_results
        for _marker in _fine_required_markers
    ):
        fine_cell_type_prevalence = pd.DataFrame()
        fine_true_cell_type_prevalence = pd.DataFrame()
        fine_prevalence_export = pd.DataFrame()
        fine_prevalence_summary = pd.DataFrame()
        fine_cell_type_maps = ()
        _fine_missing = sorted(
            set(_fine_required_markers).difference(
                *[set(_result["channel_names"]) for _result in fine_ko_results]
            )
        )
        _fine_prevalence_view = mo.callout(
            "Cell-type prevalence requires: " + ", ".join(_fine_missing),
            kind="warn",
        )
    else:
        _fine_true_sources = [
            _result for _result in fine_ko_results
            if _result["true_snapshots"] is not None
        ]
        _fine_true_reference_names = {
            "TBXT": "cell_fate_s2/TBXT",
            "SOX17": "cell_fate_s2/SOX17",
            "SOX2": "cell_fate_s1/SOX2",
            "FOXA2": "cell_fate_s2/FOXA2",
        }
        _fine_thresholds = {}
        for _marker in _fine_required_markers:
            if _fine_true_sources:
                _fine_reference_values = np.concatenate([
                    _source["true_snapshots"][_source["snapshot_hour"]][
                        _source["true_channel_names"].index(
                            _fine_true_reference_names[_marker]
                        )
                    ][_source["boundary"]]
                    for _source in _fine_true_sources
                ])
            else:
                _fine_reference_values = np.concatenate([
                    _result["snapshots"][
                        :,
                        _result["channel_names"].index(_marker),
                        _result["boundary"],
                    ].reshape(-1)
                    for _result in fine_ko_results
                ])
            _fine_reference_intensity = (
                float(np.nanmax(_fine_reference_values))
                if fine_channel_percentiles.value[_marker] == 100.0
                else float(np.nanpercentile(
                    _fine_reference_values,
                    fine_channel_percentiles.value[_marker],
                ))
            )
            _fine_thresholds[_marker] = (
                float(fine_channel_thresholds.value[_marker])
                * _fine_reference_intensity
            )
        _fine_prevalence_rows = []
        _fine_map_results = []
        for _fine_result in fine_ko_results:
            _fine_names_for_fates = _fine_result["channel_names"]
            _fine_fate_images = _fine_result["snapshots"]
            _fine_colony = _fine_result["boundary"]
            _fine_result_maps = []
            for _fine_offset, _fine_hour in enumerate(
                _fine_result["requested_ko_hours"]
            ):
                _fine_high = {
                    _marker: _fine_fate_images[
                        _fine_offset, _fine_names_for_fates.index(_marker)
                    ] > _fine_thresholds[_marker]
                    for _marker in _fine_required_markers
                }
                _fine_cell_masks = {
                    _cell_type: fate_rule_mask(_fine_high, _fine_rules[_cell_type])
                    for _cell_type in CELL_TYPES
                }
                _fine_cell_masks["Other"] = ~np.logical_or.reduce(
                    tuple(_fine_cell_masks.values())
                )
                _fine_result_maps.append({
                    _fine_cell_type: np.asarray(_fine_cell_mask & _fine_colony)
                    for _fine_cell_type, _fine_cell_mask in _fine_cell_masks.items()
                })
                for _fine_cell_type, _fine_cell_mask in _fine_cell_masks.items():
                    _fine_prevalence_rows.append({
                        "model_id": _fine_result["model_id"],
                        "model_label": _fine_result["label"],
                        "run_index": _fine_result.get("run_index", 0),
                        "seed": _fine_result.get("seed"),
                        "replicate": _fine_result.get("replicate"),
                        "group": _fine_result["groups"][fine_group_by.value],
                        "knockout_hour": float(_fine_hour),
                        "realised_knockout_hour": float(
                            _fine_result["realised_ko_hours"][_fine_offset]
                        ),
                        "cell_type": _fine_cell_type,
                        "relative_prevalence": float(
                            np.mean(_fine_cell_mask[_fine_colony])
                        ),
                    })
            _fine_map_results.append({
                "model_id": _fine_result["model_id"],
                "label": _fine_result["label"],
                "run_index": _fine_result.get("run_index", 0),
                "seed": _fine_result.get("seed"),
                "replicate": _fine_result.get("replicate"),
                "group": _fine_result["groups"][fine_group_by.value],
                "requested_ko_hours": np.asarray(
                    _fine_result["requested_ko_hours"]
                ),
                "boundary": np.asarray(_fine_colony),
                "cell_masks": tuple(_fine_result_maps),
                "lmbr": np.asarray(
                    _fine_fate_images[:, _fine_names_for_fates.index("LMBR")]
                ),
            })
        fine_cell_type_maps = tuple(_fine_map_results)
        fine_cell_type_prevalence = pd.DataFrame(_fine_prevalence_rows)
        _fine_export_index = [
            "model_id",
            "model_label",
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
        if _fine_true_sources:
            _fine_true_s1_thresholds = {}
            for _fine_true_s1_marker in ("TBXT", "SOX2", "SOX17"):
                _fine_true_s1_values = np.concatenate([
                    _source["true_snapshots"][_source["snapshot_hour"]][
                        _source["true_channel_names"].index(
                            f"cell_fate_s1/{_fine_true_s1_marker}"
                        )
                    ][_source["boundary"]]
                    for _source in _fine_true_sources
                ])
                _fine_true_s1_reference = (
                    float(np.nanmax(_fine_true_s1_values))
                    if fine_channel_percentiles.value[_fine_true_s1_marker] == 100.0
                    else float(np.nanpercentile(
                        _fine_true_s1_values,
                        fine_channel_percentiles.value[_fine_true_s1_marker],
                    ))
                )
                _fine_true_s1_thresholds[_fine_true_s1_marker] = (
                    float(fine_channel_thresholds.value[_fine_true_s1_marker])
                    * _fine_true_s1_reference
                )
            for _fine_true_source in _fine_true_sources:
                _fine_true_names = _fine_true_source["true_channel_names"]
                _fine_true_colony = _fine_true_source["boundary"]
                _fine_true_labels = {
                    0.0: "True KO 0h",
                    24.0: "True KO 24h",
                    _fine_true_source["snapshot_hour"]: "True baseline",
                }
                for _fine_true_hour, _fine_true_image in (
                    _fine_true_source["true_snapshots"].items()
                ):
                    _fine_true_high = {
                        _marker: _fine_true_image[
                            _fine_true_names.index(f"cell_fate_s2/{_marker}")
                        ] > _fine_thresholds[_marker]
                        for _marker in ("TBXT", "SOX17", "FOXA2")
                    }
                    _fine_true_s1_high = {
                        _marker: _fine_true_image[
                            _fine_true_names.index(f"cell_fate_s1/{_marker}")
                        ] > _fine_true_s1_thresholds[_marker]
                        for _marker in ("TBXT", "SOX2", "SOX17")
                    }
                    # Never combine pixels from the separately stained S1 and S2
                    # panels. Use whichever panel measures more of the selected
                    # rule, and explicitly report any omitted marker selections.
                    _fine_true_panels = (
                        ("cell_fate_s2", _fine_true_high, {"TBXT", "SOX17", "FOXA2"}),
                        ("cell_fate_s1", _fine_true_s1_high, {"TBXT", "SOX17", "SOX2"}),
                    )
                    _fine_true_masks = {}
                    _fine_true_definitions = {}
                    for _cell_type in CELL_TYPES:
                        _fine_selected = {
                            marker for marker, state in _fine_rules[_cell_type].items()
                            if state != "any"
                        }
                        _fine_panel_name, _fine_panel_high, _fine_available = max(
                            _fine_true_panels,
                            key=lambda panel: len(_fine_selected & panel[2]),
                        )
                        _fine_panel_rule = {
                            marker: state
                            for marker, state in _fine_rules[_cell_type].items()
                            if marker in _fine_available
                        }
                        _fine_omitted = sorted(_fine_selected - _fine_available)
                        _fine_true_masks[_cell_type] = fate_rule_mask(
                            _fine_panel_high, _fine_panel_rule
                        )
                        _fine_true_definitions[_cell_type] = (
                            f"{_fine_panel_name}: "
                            + ", ".join(
                                f"{marker}={state}"
                                for marker, state in _fine_panel_rule.items()
                                if state != "any"
                            )
                            + (f"; omitted {', '.join(_fine_omitted)}" if _fine_omitted else "")
                        )
                    for _fine_cell_type, _fine_true_mask in (
                        _fine_true_masks.items()
                    ):
                        _fine_true_rows.append({
                            "replicate": _fine_true_source["replicate"],
                            "knockout_hour": float(_fine_true_hour),
                            "condition": _fine_true_labels[_fine_true_hour],
                            "cell_type": _fine_cell_type,
                            "measured_definition": _fine_true_definitions[
                                _fine_cell_type
                            ],
                            "relative_prevalence": float(
                                np.mean(_fine_true_mask[_fine_true_colony])
                            ),
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
        _fine_prevalence_figure, _fine_prevalence_axis = plt.subplots(
            figsize=(11, 5), dpi=150
        )
        _fine_cell_colors = {
            "Notochord": "#d627a0",
            "Endoderm": "#17becf",
            "Mesoderm": "#2ca02c",
            "Other": "#7f7f7f",
        }
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
        for (_fine_group_label, _fine_cell_type), _fine_curve in (
            _fine_display_summary.groupby(["group", "cell_type"], sort=False)
        ):
            _fine_curve = _fine_curve.sort_values("knockout_hour")
            _fine_line = _fine_prevalence_axis.plot(
                _fine_curve["knockout_hour"],
                _fine_curve["mean_prevalence"],
                _fine_style_by_group[_fine_group_label],
                color=_fine_cell_colors[_fine_cell_type],
                label=f"{_fine_group_label} · {_fine_cell_type}",
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
            for _fine_true_condition, _fine_true_group in (
                _fine_true_display.groupby("condition", sort=False)
            ):
                for _, _fine_true_row in _fine_true_group.iterrows():
                    _fine_prevalence_axis.scatter(
                        _fine_true_row["knockout_hour"],
                        _fine_true_row["relative_prevalence"],
                        color=_fine_cell_colors[_fine_true_row["cell_type"]],
                        marker=_fine_true_markers[_fine_true_condition],
                        s=65,
                        edgecolor="black",
                        linewidth=0.5,
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
        _fine_prevalence_axis.set(
            xlabel="NODAL knockout time (h)",
            ylabel="Fraction of colony pixels",
            xticks=[0, 12, 24, 36, 48],
            xticklabels=["0h", "12h", "24h", "36h", "48h"],
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
        _fine_prevalence_figure.tight_layout()
        _fine_prevalence_view = mo.vstack([
            _fine_prevalence_figure,
            mo.md(
                "Bands show the selected uncertainty across distinct models, "
                "initial-condition replicates, and seeded runs in each group. "
                "Groups with one total sample have no uncertainty band. Each true "
                "data replicate is plotted separately. Because SOX2 and FOXA2 are "
                "stained in separate true-data panels, each true prevalence uses "
                "the panel satisfying the most selected markers. Its measured "
                "definition column explicitly lists any omitted selections."
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
def _(
    fine_cell_type_maps,
    fine_display_cell_types,
    fine_display_groups,
    fine_ko_results,
    fine_scale_cell_maps_by_lmbr,
    fine_snapshot_stride,
):
    if fine_ko_results is None or not fine_cell_type_maps:
        _fine_cell_map_view = mo.md("")
    else:
        _fine_map_colors = {
            "Notochord": np.asarray(
                [214, 39, 160], dtype=float
            ) / 255.0,
            "Endoderm": np.asarray(
                [23, 190, 207], dtype=float
            ) / 255.0,
            "Mesoderm": np.asarray(
                [44, 160, 44], dtype=float
            ) / 255.0,
            "Other": np.asarray([127, 127, 127], dtype=float) / 255.0,
        }
        _fine_selected_map_types = tuple(fine_display_cell_types.value)
        _fine_visible_map_results = [
            _fine_map_result
            for _fine_map_result in fine_cell_type_maps
            if _fine_map_result["group"] in fine_display_groups.value
        ]
        _fine_cell_map_figures = []
        for _fine_map_result in _fine_visible_map_results:
            _fine_map_indices = list(range(
                0,
                len(_fine_map_result["cell_masks"]),
                int(fine_snapshot_stride.value),
            ))
            if _fine_map_indices[-1] != len(_fine_map_result["cell_masks"]) - 1:
                _fine_map_indices.append(len(_fine_map_result["cell_masks"]) - 1)
            _fine_rgb_maps = []
            for _fine_map_index in _fine_map_indices:
                _fine_rgb_map = np.zeros(
                    (*_fine_map_result["boundary"].shape, 3), dtype=float
                )
                for _fine_map_cell_type in _fine_selected_map_types:
                    _fine_rgb_map[
                        _fine_map_result["cell_masks"][_fine_map_index][
                            _fine_map_cell_type
                        ]
                    ] = _fine_map_colors[_fine_map_cell_type]
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
                _fine_cell_map_axis.set_title(
                    f"KO {_fine_map_result['requested_ko_hours'][_fine_map_index]:g}h"
                )
                _fine_cell_map_axis.set_axis_off()
            _fine_cell_map_figure.suptitle(
                f"{_fine_map_result['label']} · cell identity at "
                f"{fine_ko_results[0]['snapshot_hour']:g}h"
            )
            _fine_cell_map_figure.tight_layout()
            _fine_cell_map_figures.append(_fine_cell_map_figure)

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
                    else "Cell types use uniform categorical brightness."
                ),
                mo.md(_fine_map_legend),
                *_fine_cell_map_figures,
            ])
    _fine_cell_map_view
    return


@app.cell(hide_code=True)
def _(fine_ko_results, fine_snapshot_channels, fine_snapshot_stride):
    if fine_ko_results is None or not fine_snapshot_channels.value:
        _fine_snapshot_view = mo.md("")
    else:
        _fine_snapshot_figures = []
        for _fine_result in fine_ko_results:
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
                    f"KO {_fine_result['requested_ko_hours'][_index]:g}h"
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
            _fine_axis.set_title(
                f"{_fine_result['label']} · predicted at "
                f"{_fine_result['snapshot_hour']:g}h"
            )
            _fine_axis.tick_params(length=0)
            _fine_figure.tight_layout()
            _fine_snapshot_figures.append(_fine_figure)
        _fine_snapshot_view = mo.vstack(_fine_snapshot_figures)
    _fine_snapshot_view
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cell-fate dynamics across intervention and measurement time

    Sweep NODAL-KO time while measuring the mutually exclusive cell-fate area
    fractions at every NCA update. Only four scalar prevalences are retained
    per step, rather than full image trajectories. After the sweep, use the
    selectors below to explore model and cell-type heatmaps without rerunning.
    """)
    return


@app.cell(hide_code=True)
def _(selection_records):
    _dynamics_model_options = {
        str(
            _record.get("notes")
            or _record.get("alias")
            or _record.get("display_name")
            or _record["model_id"]
        ) + f" [{_record['model_id'][-8:]}]": _record["model_id"]
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
    dynamics_channel_thresholds = mo.ui.dictionary({
        _marker: mo.ui.slider(
            0.05,
            0.95,
            value=0.3,
            step=0.05,
            label=f"{_marker} threshold fraction",
            full_width=True,
        )
        for _marker in ("TBXT", "SOX17", "SOX2", "FOXA2")
    })
    dynamics_channel_percentiles = mo.ui.dictionary({
        _marker: mo.ui.slider(
            90.0,
            100.0,
            value=99.0,
            step=0.5,
            label=f"{_marker} reference percentile",
            full_width=True,
        )
        for _marker in ("TBXT", "SOX17", "SOX2", "FOXA2")
    })
    dynamics_cell_type_rules = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            _marker: mo.ui.dropdown(
                {"High": "high", "Low": "low", "Irrespective": "any"},
                value={"high": "High", "low": "Low", "any": "Irrespective"}[
                    DEFAULT_FATE_RULES[_cell_type][_marker]
                ],
                label=_marker,
                full_width=True,
            )
            for _marker in FATE_MARKERS
        })
        for _cell_type in CELL_TYPES
    })
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
        mo.md("### Per-marker expression thresholds"),
        dynamics_channel_thresholds,
        mo.md("### Per-marker reference percentiles"),
        dynamics_channel_percentiles,
        mo.md("### Cell-type marker selections"),
        dynamics_cell_type_rules,
    ])
    return (
        dynamics_channel_percentiles,
        dynamics_channel_thresholds,
        dynamics_cell_type_rules,
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
def _(dynamics_cell_type_rules):
    _dynamics_rule_warnings = fate_rule_warnings(dynamics_cell_type_rules.value)
    if _dynamics_rule_warnings:
        mo.callout("\n\n".join(_dynamics_rule_warnings), kind="warn")
    else:
        mo.callout("Cell-type definitions are mutually exclusive.", kind="success")
    return


@app.cell(hide_code=True)
def _(
    data_root,
    dynamics_channel_percentiles,
    dynamics_channel_thresholds,
    dynamics_cell_type_rules,
    dynamics_endpoint,
    dynamics_ko_interval,
    dynamics_ko_start,
    dynamics_ko_stop,
    dynamics_models,
    dynamics_replicate,
    dynamics_seed,
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
        _dynamics_cell_types = (*CELL_TYPES, "Other")
        _dynamics_fate_rules = fate_rule_matrix(dynamics_cell_type_rules.value)
        _dynamics_reference_names = {
            "TBXT": "cell_fate_s2/TBXT",
            "SOX17": "cell_fate_s2/SOX17",
            "SOX2": "cell_fate_s1/SOX2",
            "FOXA2": "cell_fate_s2/FOXA2",
        }
        _dynamics_results = []
        for _dynamics_model_id in tqdm(
            dynamics_models.value,
            desc="Running prevalence dynamics models",
            unit="model",
        ):
            _dynamics_record = _dynamics_records[_dynamics_model_id]
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
            _dynamics_hours = tuple(
                float(_hour)
                for _hour in _dynamics_bundle.config.data.micropattern.timesteps
            )
            if float(dynamics_endpoint.value) not in _dynamics_hours:
                raise ValueError(
                    f"No true baseline reference is available at "
                    f"{dynamics_endpoint.value:g}h for {_dynamics_bundle.id}."
                )
            _dynamics_reference = _dynamics_data["target"][
                _dynamics_hours.index(float(dynamics_endpoint.value))
            ]
            _dynamics_marker_order = ("TBXT", "SOX17", "SOX2", "FOXA2")
            _dynamics_thresholds = []
            for _dynamics_marker in _dynamics_marker_order:
                _dynamics_values = _dynamics_reference[
                    _dynamics_data["channel_names"].index(
                        _dynamics_reference_names[_dynamics_marker]
                    )
                ][np.asarray(_dynamics_data["boundary"]).squeeze().astype(bool)]
                _dynamics_reference_intensity = (
                    float(np.nanmax(_dynamics_values))
                    if dynamics_channel_percentiles.value[_dynamics_marker] == 100.0
                    else float(np.nanpercentile(
                        _dynamics_values,
                        dynamics_channel_percentiles.value[_dynamics_marker],
                    ))
                )
                _dynamics_thresholds.append(
                    float(dynamics_channel_thresholds.value[_dynamics_marker])
                    * _dynamics_reference_intensity
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
            _dynamics_fate_channels = jnp.asarray([
                _dynamics_data["state_channel_names"].index(_marker)
                for _marker in _dynamics_marker_order
            ])
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
                _dynamics_prevalence = (
                    rollout_model_with_blocked_channel_prevalence(
                        _dynamics_model,
                        _dynamics_state,
                        jnp.asarray(_dynamics_data["boundary"]),
                        _dynamics_bundle.config.trainer.get("boundary_mode", "soft"),
                        jr.PRNGKey(int(dynamics_seed.value)),
                        _dynamics_total_steps,
                        _dynamics_data["nodal_channel"],
                        jnp.asarray(_dynamics_ko_step, dtype=jnp.int32),
                        _dynamics_fate_channels,
                        jnp.asarray(_dynamics_thresholds),
                        jnp.asarray(_dynamics_fate_rules),
                        jnp.asarray(_dynamics_data["boundary"]).squeeze().astype(bool),
                    )
                )
                _dynamics_matrices.append(np.asarray(_dynamics_prevalence))
                _dynamics_realised_ko.append(
                    12.0 * _dynamics_ko_step / _dynamics_steps_per_12h
                )
            _dynamics_label = str(
                _dynamics_record.get("notes")
                or _dynamics_record.get("alias")
                or _dynamics_record.get("display_name")
                or _dynamics_bundle.id
            )
            _dynamics_results.append({
                "model_id": _dynamics_bundle.id,
                "label": _dynamics_label,
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
                "thresholds": dict(zip(
                    _dynamics_marker_order, _dynamics_thresholds
                )),
                "cell_type_rules": dynamics_cell_type_rules.value,
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
    return (dynamics_prevalence_export,)


@app.cell(hide_code=True)
def _(temporal_prevalence_results):
    _heatmap_model_options = {
        f"{_result['label']} [{_result['model_id'][-8:]}]": _result["model_id"]
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
    mo.hstack([dynamics_heatmap_model, dynamics_heatmap_cell_type])
    return dynamics_heatmap_cell_type, dynamics_heatmap_model


@app.cell(hide_code=True)
def _(
    dynamics_heatmap_cell_type,
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
            cmap="viridis",
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
                f"{_heatmap_result['label']}\n"
                f"{dynamics_heatmap_cell_type.value} area fraction"
            ),
            xticks=[0, 12, 24, 36, 48],
            yticks=[0, 12, 24, 36, 48],
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
        str(
            _record.get("notes")
            or _record.get("alias")
            or _record.get("display_name")
            or _record["model_id"]
        ): _record["model_id"]
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
        _composite_boundary_mode = _composite_bundle.config.trainer.get(
            "boundary_mode", "soft"
        )
        if _composite_boundary_mode == "soft":
            _composite_boundary_callback = model_boundary(_composite_boundary)
        elif _composite_boundary_mode == "hard":
            _composite_boundary_callback = hard_boundary(_composite_boundary)
        elif _composite_boundary_mode == "none":
            _composite_boundary_callback = no_boundary()
        else:
            raise ValueError(f"Unknown boundary mode {_composite_boundary_mode!r}.")

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
        str(
            _record.get("notes")
            or _record.get("alias")
            or _record.get("display_name")
            or _record["model_id"]
        ): _record["model_id"]
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
        _generative_boundary_mode = _generative_bundle.config.trainer.get(
            "boundary_mode", "soft"
        )
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
