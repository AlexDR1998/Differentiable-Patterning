"""Interactive visual and statistical inspection of repository datasets.

Run with:
    marimo edit Experiments/dataset_explorer.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.dataloader.emoji import load_emoji_sequence
    from Common.dataloader.micropattern import (
        load_micropattern_260726,
        load_micropattern_circle_4ch_individual,
        load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
    )
    from Common.dataloader.micropattern_schemas import (
        DEFAULT_260726_HISTOGRAM_PERCENTILES,
        DEFAULT_260726_INITIAL_INTENSITY_SCALES,
        MICROPATTERN_260726_SCHEMA,
    )
    from Common.dataloader.preprocessing import ProcessingStep
    from Common.dataloader.texture import load_textures

    return (
        DEFAULT_260726_HISTOGRAM_PERCENTILES,
        DEFAULT_260726_INITIAL_INTENSITY_SCALES,
        MICROPATTERN_260726_SCHEMA,
        ProcessingStep,
        load_emoji_sequence,
        load_micropattern_260726,
        load_micropattern_circle_4ch_individual,
        load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
        load_textures,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(DEFAULT_260726_HISTOGRAM_PERCENTILES, ProcessingStep, mo):
    dataset_kind = mo.ui.dropdown(
        options={
            "260726 multichannel micropattern": "micropattern_260726",
            "Legacy grouped/knockout micropattern": "micropattern_grouped",
            "Legacy four-channel micropattern": "micropattern_4ch",
            "Emoji sequence": "emoji",
            "Texture sequence": "texture",
        },
        value="260726 multichannel micropattern",
        label="Dataset",
    )
    root = mo.ui.text(
        value="../Data/260726_nca_dataset",
        label="Dataset root",
        full_width=True,
    )
    filenames = mo.ui.text(
        value="alien_monster.png,microbe.png",
        label="Image filenames (emoji/texture, comma separated)",
        full_width=True,
    )
    timesteps = mo.ui.text(value="0,12,24,36,48", label="Timesteps (hours)")
    downsample = mo.ui.slider(1, 16, value=4, label="Downsample")
    batches = mo.ui.slider(1, 8, value=1, label="Replicates/batches")
    conditions = mo.ui.multiselect(
        options={"Control": "ctrl", "Nodal knockout at 0h": "sl0", "Nodal knockout at 24h": "sl24"},
        value=["Control"],
        label="260726 conditions",
    )
    experiment_groups = mo.ui.multiselect(
        options={
            "Cell fate (stain 1)": "cell_fate_s1",
            "Cell fate (stain 2)": "cell_fate_s2",
            "RNA expression": "rna_expression",
            "Protein response": "protein_response",
        },
        value=[
            "Cell fate (stain 1)",
            "Cell fate (stain 2)",
            "RNA expression",
            "Protein response",
        ],
        label="260726 experiment groups",
    )
    substitute_preperturbation = mo.ui.checkbox(
        value=True,
        label="Use control measurements before knockout",
    )
    knockout = mo.ui.dropdown(
        options={"Baseline": "baseline", "Knockout at 0h": "ko0", "Knockout at 24h": "ko24"},
        value="Baseline",
        label="Legacy condition",
    )
    processing = mo.ui.multiselect(
        options={step.value: step.value for step in ProcessingStep},
        value=["map_to_0_1", "downsample"],
        label="Ordered preprocessing steps (legacy loaders)",
    )
    align = mo.ui.checkbox(value=True, label="Align 260726 images")
    # Defaults match micropattern training (data.micropattern.histogram_percentiles).
    percentile_low = mo.ui.number(
        0.0, 99.0, value=DEFAULT_260726_HISTOGRAM_PERCENTILES[0], step=0.1,
        label="Histogram low percentile",
    )
    percentile_high = mo.ui.number(
        1.0, 100.0, value=DEFAULT_260726_HISTOGRAM_PERCENTILES[1], step=0.05,
        label="Histogram high percentile",
    )
    load_button = mo.ui.run_button(label="Load dataset")
    _controls = mo.vstack(
        [
            mo.hstack([dataset_kind, downsample, batches, knockout]),
            mo.hstack([conditions, substitute_preperturbation]),
            experiment_groups,
            root,
            filenames,
            mo.hstack([timesteps, align, percentile_low, percentile_high]),
            processing,
            load_button,
        ]
    )
    _controls
    return (
        align,
        batches,
        conditions,
        dataset_kind,
        downsample,
        experiment_groups,
        filenames,
        knockout,
        load_button,
        percentile_high,
        percentile_low,
        processing,
        root,
        substitute_preperturbation,
        timesteps,
    )


@app.cell(hide_code=True)
def _(DEFAULT_260726_INITIAL_INTENSITY_SCALES, MICROPATTERN_260726_SCHEMA, mo):
    initial_intensity_scales = mo.ui.dictionary(
        {
            _name: mo.ui.number(
                0.0,
                5.0,
                value=DEFAULT_260726_INITIAL_INTENSITY_SCALES.get(_name, 1.0),
                step=0.005,
                label=_name,
            )
            for _name in MICROPATTERN_260726_SCHEMA.measurement_names
        }
    )
    mo.vstack(
        [
            mo.md(
                "### 0h intensity corrections (260726 only)\n\n"
                "Some 0h images are brighter than later timepoints because of "
                "imaging artifacts, before cells express the marker. Each factor "
                "multiplies the raw 0h intensities of that channel, for every "
                "condition, before the shared per-channel normalisation. A value "
                "of 1 leaves the channel unchanged. The corrected 0h images also "
                "count towards the percentile bins, so changing one factor can "
                "shift that channel's later timepoints slightly. Compare the "
                "result in **0h intensity check** below."
            ),
            initial_intensity_scales,
        ]
    )
    return (initial_intensity_scales,)


@app.cell(hide_code=True)
def _(
    align,
    batches,
    conditions,
    dataset_kind,
    downsample,
    experiment_groups,
    filenames,
    initial_intensity_scales,
    knockout,
    load_button,
    load_emoji_sequence,
    load_micropattern_260726,
    load_micropattern_circle_4ch_individual,
    load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
    load_textures,
    percentile_high,
    percentile_low,
    processing,
    root,
    substitute_preperturbation,
    timesteps,
):
    load_button
    _selected_times = tuple(int(value.strip()) for value in timesteps.value.split(",") if value.strip())
    _selected_files = tuple(value.strip() for value in filenames.value.split(",") if value.strip())
    _ordered_processing = tuple(processing.value)
    if dataset_kind.value == "micropattern_260726":
        _selected_conditions = tuple(conditions.value)
        if not _selected_conditions:
            raise ValueError("Select at least one 260726 condition")
        loaded = load_micropattern_260726(
            root.value,
            conditions=_selected_conditions,
            timesteps=_selected_times,
            downsample=downsample.value,
            replicate_count=batches.value,
            experiment_groups=tuple(experiment_groups.value) or None,
            substitute_preperturbation=substitute_preperturbation.value,
            align=align.value,
            hist_eqs=(percentile_low.value, percentile_high.value),
            intensity_factors={
                _name: {0: _value}
                for _name, _value in initial_intensity_scales.value.items()
                if _value != 1.0
            },
        )
    elif dataset_kind.value == "micropattern_grouped":
        _ko_time = {"baseline": None, "ko0": 0, "ko24": 24}[knockout.value]
        loaded = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=root.value,
            DOWNSAMPLE=downsample.value,
            BATCHES=batches.value,
            TIMESTEPS=_selected_times,
            FILTER_KN_TIME=_ko_time,
            HIST_EQS=(percentile_low.value, percentile_high.value),
            PROCESSING_MODES=_ordered_processing,
        )
    elif dataset_kind.value == "micropattern_4ch":
        loaded = load_micropattern_circle_4ch_individual(
            impath=root.value,
            DOWNSAMPLE=downsample.value,
            BATCHES=batches.value,
            TIMESTEPS=_selected_times,
            HIST_EQS=(percentile_low.value, percentile_high.value),
            PROCESSING_MODES=_ordered_processing,
        )
    elif dataset_kind.value == "emoji":
        loaded = load_emoji_sequence(_selected_files, root.value, downsample.value, True)
    else:
        loaded = load_textures(_selected_files, root.value, downsample.value, True)
    return (loaded,)


@app.cell(hide_code=True)
def _(data, loaded, mo, names, np, plt):
    # Mean normalised intensity inside the colony boundary, averaged over the
    # measured replicates, so each channel's 0h level can be compared with
    # its later timepoints on the same scale.
    _aux = getattr(loaded, "aux", {})
    _hours = tuple(_aux.get("timesteps", ()))
    _measured = _aux.get("measurement_mask")
    _boundary = getattr(loaded, "boundary_mask", None)
    if not _hours or _measured is None or _boundary is None or len(_hours) < 2:
        _intensity_view = mo.md(
            "### 0h intensity check\n\n"
            "Needs a 260726 micropattern dataset with at least two timesteps."
        )
    else:
        _measured = np.asarray(_measured)
        _inside = np.asarray(_boundary)[:, 0]
        _pixel_sums = (data * _inside[:, None, None]).sum(axis=(-2, -1))
        _pixel_means = _pixel_sums / np.maximum(_inside.sum(axis=(-2, -1)), 1)[:, None, None]
        _pixel_means = np.where(_measured, _pixel_means, np.nan)
        _replicate_counts = _measured.sum(axis=0)
        _channel_means = np.where(
            _replicate_counts > 0,
            np.nansum(_pixel_means, axis=0) / np.maximum(_replicate_counts, 1),
            np.nan,
        )
        _applied = _aux.get("intensity_factors", {})
        _rows = []
        for _channel, _name in enumerate(names):
            _start, _next = _channel_means[0, _channel], _channel_means[1, _channel]
            _rows.append(
                {
                    "channel": _name,
                    "0h scale": _applied.get(_name, {}).get(0, 1.0),
                    "0h mean": round(float(_start), 4),
                    f"{_hours[1]}h mean": round(float(_next), 4),
                    f"0h / {_hours[1]}h": (
                        round(float(_start / _next), 3) if _next > 0 else None
                    ),
                    f"0h brighter than {_hours[1]}h": bool(_start > _next),
                }
            )
        _columns = min(4, len(names))
        _plot_rows = int(np.ceil(len(names) / _columns))
        _figure, _axes = plt.subplots(
            _plot_rows,
            _columns,
            figsize=(3.2 * _columns, 2.4 * _plot_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for _channel, _axis in enumerate(_axes.flat):
            if _channel >= len(names):
                _axis.axis("off")
                continue
            _axis.plot(_hours, _channel_means[:, _channel], color="black", marker="o")
            for _replicate in range(_pixel_means.shape[0]):
                _axis.plot(
                    _hours,
                    _pixel_means[_replicate, :, _channel],
                    color="grey",
                    alpha=0.4,
                    linewidth=0.8,
                )
            _axis.set_title(names[_channel], fontsize=8)
            _axis.set_xticks(_hours)
        for _axis in _axes[-1]:
            _axis.set_xlabel("hours")
        for _axis in _axes[:, 0]:
            _axis.set_ylabel("mean intensity")
        _figure.tight_layout()
        _intensity_view = mo.vstack(
            [
                mo.md(
                    "### 0h intensity check\n\n"
                    "Mean normalised intensity inside the boundary mask for each "
                    "channel. Black is the mean over measured replicates and grey "
                    "lines are single replicates. Unlike the tiled images, these "
                    "values are on one fixed scale per channel. Tune the 0h "
                    "corrections above until 0h sits where the experimentalists "
                    "expect it. Copy the final values from the `0h scale` column."
                ),
                mo.ui.table(_rows, selection=None),
                _figure,
            ]
        )
    _intensity_view
    return


@app.cell(hide_code=True)
def _(mo):
    fate_threshold_mode = mo.ui.dropdown(
        options={
            "Relative to reference percentile": "relative",
            "Absolute value (0–1)": "absolute",
        },
        value="Relative to reference percentile",
        label="Threshold mode",
        full_width=True,
    )
    return (fate_threshold_mode,)


@app.cell(hide_code=True)
def _(data, fate_threshold_mode, loaded, mo):
    _default_fate_rules = {
        "Notochord": {
            "SOX17": "low",
            "SOX2": "low",
            "TBXT": "high",
            "FOXA2": "high",
        },
        "Endoderm": {
            "SOX17": "high",
            "SOX2": "any",
            "TBXT": "any",
            "FOXA2": "any",
        },
        "Mesoderm": {
            "SOX17": "low",
            "SOX2": "high",
            "TBXT": "high",
            "FOXA2": "low",
        },
    }
    fate_threshold_fractions = mo.ui.dictionary({
        _marker: mo.ui.slider(
            0.0 if fate_threshold_mode.value == "absolute" else 0.05,
            1.0 if fate_threshold_mode.value == "absolute" else 0.95,
            value=0.3,
            step=0.01 if fate_threshold_mode.value == "absolute" else 0.05,
            label=(
                f"{_marker} absolute threshold"
                if fate_threshold_mode.value == "absolute"
                else f"{_marker} threshold fraction"
            ),
            full_width=True,
        )
        for _marker in ("SOX17", "SOX2", "TBXT", "FOXA2")
    })
    fate_reference_percentiles = mo.ui.dictionary({
        _marker: mo.ui.slider(
            90.0,
            100.0,
            value=99.0,
            step=0.5,
            label=f"{_marker} reference percentile",
            full_width=True,
        )
        for _marker in ("SOX17", "SOX2", "TBXT", "FOXA2")
    })
    _fate_aux = getattr(loaded, "aux", {})
    _fate_conditions = tuple(_fate_aux.get("batch_conditions", ()))
    _fate_replicates = tuple(_fate_aux.get("batch_replicates", ()))
    _fate_batch_options = {
        (
            f"{_index}: {_fate_conditions[_index]}, "
            f"replicate {_fate_replicates[_index]}"
            if _index < len(_fate_conditions) and _index < len(_fate_replicates)
            else f"Batch {_index}"
        ): _index
        for _index in range(data.shape[0])
    }
    fate_batch_indices = mo.ui.multiselect(
        options=_fate_batch_options,
        value=[next(iter(_fate_batch_options))],
        label="Replicates to display",
        full_width=True,
    )
    _fate_timesteps = tuple(_fate_aux.get("timesteps", ()))
    _fate_time_options = {
        (
            f"{_fate_timesteps[_index]}h"
            if _index < len(_fate_timesteps)
            else f"Time index {_index}"
        ): _index
        for _index in range(data.shape[1])
    }
    fate_time_indices = mo.ui.multiselect(
        options=_fate_time_options,
        value=[next(reversed(_fate_time_options))],
        label="Timepoints to display",
        full_width=True,
    )
    fate_cell_type_rules = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            _marker: mo.ui.dropdown(
                options={"High": "high", "Low": "low", "Indifferent": "any"},
                value={"high": "High", "low": "Low", "any": "Indifferent"}[
                    _default_fate_rules[_cell_type][_marker]
                ],
                label=_marker,
                full_width=True,
            )
            for _marker in ("SOX17", "SOX2", "TBXT", "FOXA2")
        })
        for _cell_type in ("Notochord", "Endoderm", "Mesoderm")
    })
    _threshold_explanation = (
        "Each marker is classified directly against its selected absolute "
        "0–1 intensity threshold."
        if fate_threshold_mode.value == "absolute"
        else "Each expression cutoff is the selected fraction of that marker's "
        "reference percentile across available final-timestep replicates, so "
        "the same absolute cutoffs are used at every displayed timepoint."
    )
    _threshold_controls = (
        fate_threshold_fractions
        if fate_threshold_mode.value == "absolute"
        else mo.hstack(
            [fate_threshold_fractions, fate_reference_percentiles],
            widths="equal",
        )
    )
    mo.vstack(
        [
            mo.md(
                "## Cell-fate threshold explorer\n\n"
                + _threshold_explanation
            ),
            fate_threshold_mode,
            mo.hstack([fate_batch_indices, fate_time_indices], widths="equal"),
            _threshold_controls,
            mo.md("### Cell-type lineage definitions"),
            fate_cell_type_rules,
        ]
    )
    return (
        fate_batch_indices,
        fate_cell_type_rules,
        fate_reference_percentiles,
        fate_threshold_fractions,
        fate_time_indices,
    )


@app.cell(hide_code=True)
def _(mo):
    automatic_threshold_tolerance = mo.ui.slider(
        0.0,
        0.25,
        value=0.02,
        step=0.005,
        label="Maximum active pixel fraction still considered low",
        full_width=True,
    )
    automatic_threshold_grid_size = mo.ui.slider(
        101,
        2001,
        value=501,
        step=100,
        label="Threshold search grid size",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            "## Automatic baseline threshold selection\n\n"
            "Find shared absolute SOX17 and FOXA2 thresholds across baseline "
            "replicates. SOX17 is expected low at 0, 12, and 24h and high at "
            "36 and 48h. FOXA2 is expected low at 0 and 12h and high at 24, "
            "36, and 48h. A timepoint is low at or below the tolerated active "
            "pixel fraction and high above it. TBXT and SOX2 remain "
            "user-defined above."
        ),
        mo.hstack(
            [automatic_threshold_tolerance, automatic_threshold_grid_size],
            widths="equal",
        ),
    ])
    return automatic_threshold_grid_size, automatic_threshold_tolerance


@app.cell(hide_code=True)
def _(
    automatic_threshold_grid_size,
    automatic_threshold_tolerance,
    data,
    loaded,
    mo,
    names,
    np,
    plt,
):
    _auto_expectations = {
        "SOX17": {0: "low", 12: "low", 24: "low", 36: "high", 48: "high"},
        "FOXA2": {0: "low", 12: "low", 24: "high", 36: "high", 48: "high"},
    }
    _auto_preferred = {
        "SOX17": "cell_fate_s2/SOX17",
        "FOXA2": "cell_fate_s2/FOXA2",
    }
    _auto_channels = {
        _marker: (
            _auto_preferred[_marker]
            if _auto_preferred[_marker] in names
            else _marker if _marker in names else None
        )
        for _marker in _auto_expectations
    }
    _auto_aux = getattr(loaded, "aux", {})
    _auto_times = tuple(_auto_aux.get("timesteps", range(data.shape[1])))
    _auto_conditions = tuple(_auto_aux.get("batch_conditions", ()))
    _auto_replicates = tuple(_auto_aux.get("batch_replicates", ()))
    _auto_baseline_batches = [
        _batch for _batch in range(data.shape[0])
        if not _auto_conditions or _auto_conditions[_batch] == "ctrl"
    ]
    _auto_missing_markers = [
        _marker for _marker, _channel in _auto_channels.items()
        if _channel is None
    ]
    _auto_missing_times = sorted({0, 12, 24, 36, 48}.difference(_auto_times))
    if _auto_missing_markers:
        _auto_view = mo.callout(
            "Automatic selection requires " + ", ".join(_auto_missing_markers),
            kind="warn",
        )
    elif not _auto_baseline_batches:
        _auto_view = mo.callout(
            "No baseline (`ctrl`) replicates are loaded.", kind="warn"
        )
    elif _auto_missing_times:
        _auto_view = mo.callout(
            "Load all required timepoints: "
            + ", ".join(f"{_time}h" for _time in _auto_missing_times),
            kind="warn",
        )
    else:
        _auto_boundary = getattr(loaded, "boundary_mask", None)
        _auto_measurement_mask = getattr(loaded, "measurement_mask", None)
        _auto_candidates = np.linspace(
            0.0, 1.0, int(automatic_threshold_grid_size.value)
        )
        _auto_tolerance = float(automatic_threshold_tolerance.value)
        _auto_records = {}
        _auto_shared = {}
        _auto_per_replicate = {}
        _auto_threshold_rows = []
        _auto_audit_rows = []

        def _auto_curve(_values):
            _values = np.sort(np.asarray(_values)[np.isfinite(_values)])
            if not _values.size:
                return np.full(_auto_candidates.shape, np.nan)
            return 1.0 - np.searchsorted(
                _values, _auto_candidates, side="right"
            ) / _values.size

        def _auto_choose(_records):
            _margins = np.stack([
                _auto_tolerance - _record["curve"]
                if _record["expected"] == "low"
                else _record["curve"] - _auto_tolerance
                for _record in _records
            ])
            _violations = np.maximum(-_margins, 0.0)
            _batches = sorted({_record["batch"] for _record in _records})
            _batch_losses = np.stack([
                np.nanmean(
                    _violations[
                        [_index for _index, _record in enumerate(_records)
                         if _record["batch"] == _batch]
                    ],
                    axis=0,
                )
                for _batch in _batches
            ])
            # Balance overall fit with the worst-fitting replicate so a single
            # trajectory cannot be hidden by the others.
            _objective = np.nanmean(_violations, axis=0) + np.nanmax(
                _batch_losses, axis=0
            )
            _best = np.nanmin(_objective)
            _ties = np.flatnonzero(np.isclose(_objective, _best, atol=1e-12))
            _tie_margins = np.nanmean(_margins[:, _ties], axis=0)
            _index = int(_ties[np.nanargmax(_tie_margins)])
            return _index, float(_objective[_index])

        for _marker, _expectations in _auto_expectations.items():
            _channel_index = names.index(_auto_channels[_marker])
            _marker_records = []
            for _batch in _auto_baseline_batches:
                _boundary = (
                    np.asarray(_auto_boundary)[_batch, 0].astype(bool)
                    if _auto_boundary is not None
                    else np.ones(data.shape[-2:], dtype=bool)
                )
                for _time_index, _time in enumerate(_auto_times):
                    if _time not in _expectations:
                        continue
                    if (
                        _auto_measurement_mask is not None
                        and not bool(np.asarray(_auto_measurement_mask)[
                            _batch, _time_index, _channel_index
                        ])
                    ):
                        continue
                    _values = np.asarray(
                        data[_batch, _time_index, _channel_index]
                    )[_boundary]
                    _marker_records.append({
                        "batch": _batch,
                        "replicate": (
                            _auto_replicates[_batch]
                            if _batch < len(_auto_replicates) else _batch
                        ),
                        "time": _time,
                        "expected": _expectations[_time],
                        "values": _values[np.isfinite(_values)],
                        "curve": _auto_curve(_values),
                    })
            _auto_records[_marker] = _marker_records
            _shared_index, _shared_loss = _auto_choose(_marker_records)
            _shared_threshold = float(_auto_candidates[_shared_index])
            _auto_shared[_marker] = (_shared_threshold, _shared_index)
            _auto_threshold_rows.append({
                "marker": _marker,
                "scope": "shared",
                "replicate": "all baseline",
                "threshold": _shared_threshold,
                "objective": _shared_loss,
            })
            _auto_per_replicate[_marker] = {}
            for _batch in _auto_baseline_batches:
                _batch_records = [
                    _record for _record in _marker_records
                    if _record["batch"] == _batch
                ]
                _index, _loss = _auto_choose(_batch_records)
                _threshold = float(_auto_candidates[_index])
                _auto_per_replicate[_marker][_batch] = _threshold
                _auto_threshold_rows.append({
                    "marker": _marker,
                    "scope": "replicate",
                    "replicate": _batch_records[0]["replicate"],
                    "threshold": _threshold,
                    "objective": _loss,
                })
            for _record in _marker_records:
                _fraction = float(_record["curve"][_shared_index])
                _passes = (
                    _fraction <= _auto_tolerance
                    if _record["expected"] == "low"
                    else _fraction > _auto_tolerance
                )
                _auto_audit_rows.append({
                    "marker": _marker,
                    "replicate": _record["replicate"],
                    "time (h)": _record["time"],
                    "expected": _record["expected"],
                    "active fraction": _fraction,
                    "passes": _passes,
                })

        _auto_hist_figure, _auto_hist_axes = plt.subplots(
            1, 2, figsize=(14, 4.5), squeeze=False
        )
        _auto_curve_figure, _auto_curve_axes = plt.subplots(
            1, 2, figsize=(14, 4.5), squeeze=False
        )
        for _column, _marker in enumerate(_auto_expectations):
            _hist_axis = _auto_hist_axes[0, _column]
            _curve_axis = _auto_curve_axes[0, _column]
            for _expected, _color in (("low", "tab:blue"), ("high", "tab:orange")):
                _values = [
                    _record["values"] for _record in _auto_records[_marker]
                    if _record["expected"] == _expected
                ]
                _hist_axis.hist(
                    np.concatenate(_values),
                    bins=np.linspace(0.0, 1.0, 81),
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    color=_color,
                    label=f"Expected {_expected}",
                )
            _threshold, _threshold_index = _auto_shared[_marker]
            _hist_axis.axvline(
                _threshold, color="black", linewidth=2.0,
                label=f"Shared {_threshold:.3f}"
            )
            for _replicate_threshold in _auto_per_replicate[_marker].values():
                _hist_axis.axvline(
                    _replicate_threshold,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                )
            _hist_axis.set(
                title=f"{_marker} intensity distributions",
                xlabel="intensity",
                ylabel="density",
                xlim=(0.0, 1.0),
            )
            _hist_axis.legend(fontsize="small")
            for _batch in _auto_baseline_batches:
                _batch_records = sorted(
                    (_record for _record in _auto_records[_marker]
                     if _record["batch"] == _batch),
                    key=lambda _record: _record["time"],
                )
                _curve_axis.plot(
                    [_record["time"] for _record in _batch_records],
                    [_record["curve"][_threshold_index]
                     for _record in _batch_records],
                    marker="o",
                    label=f"Replicate {_batch_records[0]['replicate']}",
                )
            _curve_axis.axhline(
                _auto_tolerance,
                color="black",
                linestyle="--",
                label=f"Tolerance {_auto_tolerance:.3f}",
            )
            _curve_axis.set(
                title=f"{_marker} above shared threshold",
                xlabel="time (h)",
                ylabel="active colony fraction",
                ylim=(-0.01, 1.01),
            )
            _curve_axis.grid(alpha=0.2)
            _curve_axis.legend(fontsize="small")
        _auto_hist_figure.tight_layout()
        _auto_curve_figure.tight_layout()
        _auto_view = mo.vstack([
            mo.callout(
                "Solid histogram slices are shared optima; dashed gray slices "
                "are replicate-specific optima. The objective combines average "
                "constraint violation with the worst replicate's violation.",
                kind="info",
            ),
            mo.ui.table(_auto_threshold_rows, selection=None),
            _auto_hist_figure,
            _auto_curve_figure,
            mo.md("### Shared-threshold constraint audit"),
            mo.ui.table(
                _auto_audit_rows,
                selection=None,
                pagination=True,
                page_size=20,
            ),
        ])
    _auto_view
    return


@app.cell
def _(data, mo, names, np, plt):
    _channel_values = data.transpose(2, 0, 1, 3, 4).reshape(data.shape[2], -1)
    _stats = [
        {
            "channel": name,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "p01": float(np.nanpercentile(values, 1)),
            "p50": float(np.nanpercentile(values, 50)),
            "p99": float(np.nanpercentile(values, 99)),
        }
        for name, values in zip(names, _channel_values)
    ]
    _figure, _axis = plt.subplots(figsize=(max(7, len(names) * 0.6), 3.5))
    _axis.boxplot([values[np.isfinite(values)] for values in _channel_values], showfliers=False)
    _axis.set_xticks(range(1, len(names) + 1), names, rotation=60, ha="right")
    _axis.set_ylabel("value")
    _figure.tight_layout()
    mo.vstack([mo.md("## Channel statistics"), mo.ui.table(_stats), _figure])
    return


@app.cell(column=1, hide_code=True)
def _(loaded, mo, np):
    data = np.asarray(loaded.data)
    names = tuple(getattr(loaded, "channel_names", ()))
    if not names:
        names = tuple(f"channel {index}" for index in range(data.shape[2]))
    _summary = {
        "shape [B,T,C,X,Y]": tuple(data.shape),
        "dtype": str(data.dtype),
        "minimum": float(np.nanmin(data)),
        "maximum": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "standard deviation": float(np.nanstd(data)),
        "finite fraction": float(np.isfinite(data).mean()),
    }
    _aux = getattr(loaded, "aux", {})
    _batch_conditions = tuple(_aux.get("batch_conditions", ()))
    _batch_replicates = tuple(_aux.get("batch_replicates", ()))
    _batch_options = {
        (
            f"{index}: {condition}, replicate {replicate}"
            if index < len(_batch_conditions) and index < len(_batch_replicates)
            else f"Batch {index}"
        ): index
        for index, (condition, replicate) in enumerate(
            zip(
                _batch_conditions or ("",) * data.shape[0],
                _batch_replicates or ("",) * data.shape[0],
            )
        )
    }
    batch_index = mo.ui.dropdown(
        options=_batch_options,
        value=next(iter(_batch_options)),
        label="Batch / condition",
    )
    time_indices = mo.ui.multiselect(
        options={str(index): index for index in range(data.shape[1])},
        value=["0","1","2","3","4"],
        label="Timesteps to tile",
    )
    channel_indices = mo.ui.multiselect(
        options={name: index for index, name in enumerate(names)},
        value=[names[0],names[1],names[2],names[3]],
        label="Channels to tile",
    )
    zero_to_nan = mo.ui.checkbox(value=False, label="Discard zero values (set to NaN)")
    mo.vstack(
        [
            mo.md("## Loaded data"),
            mo.ui.table([_summary]),
            mo.hstack([batch_index, time_indices, channel_indices, zero_to_nan]),
        ]
    )
    return batch_index, channel_indices, data, names, time_indices, zero_to_nan


@app.cell(hide_code=True)
def _(
    batch_index,
    channel_indices,
    data,
    loaded,
    names,
    np,
    plt,
    time_indices,
    zero_to_nan,
):
    # Multiselect values are strings for the timestep control and channel
    # names for the channel control. Preserve the option order in the plot.
    _selected_times = [int(value) for value in time_indices.value]
    _selected_channels = [
        int(value) if isinstance(value, (int, np.integer)) else names.index(value)
        for value in channel_indices.value
    ]
    _selected_times = _selected_times or [0]
    _selected_channels = _selected_channels or [0]
    _n_tiles = len(_selected_times) * len(_selected_channels)
    _figure, _axes = plt.subplots(
        len(_selected_channels),
        len(_selected_times),
        figsize=(3.2 * len(_selected_times), 3.0 * len(_selected_channels)),
        squeeze=False,
        layout="constrained",
    )
    _histogram_figure, _histogram_axis = plt.subplots(figsize=(8, 4))
    # One colour range per channel, taken over every batch and timestep, so
    # brightness can be compared across time (and batches) within a row.
    _ranges = {}
    for _channel in _selected_channels:
        _values = data[:, :, _channel]
        _values = _values[np.isfinite(_values)]
        if zero_to_nan.value:
            _values = _values[_values != 0]
        _ranges[_channel] = (
            (float(_values.min()), float(_values.max())) if _values.size else (0.0, 1.0)
        )
    for _column, _time in enumerate(_selected_times):
        for _row, _channel in enumerate(_selected_channels):
            _image = data[batch_index.value, _time, _channel]
            if zero_to_nan.value:
                _image = np.where(_image == 0, np.nan, _image)
            _finite = _image[np.isfinite(_image)]
            _shown = _axes[_row, _column].imshow(
                _image,
                cmap="grey",
                vmin=_ranges[_channel][0],
                vmax=_ranges[_channel][1],
            )
            if _column == len(_selected_times) - 1:
                _figure.colorbar(_shown, ax=_axes[_row, :].tolist(), shrink=0.8)
            _axes[_row, _column].set_title(f"t={_time}, {names[_channel]}")
            _axes[_row, _column].axis("off")
            if _finite.size:
                _histogram_axis.hist(
                    _finite.ravel(),
                    bins=80,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"t={_time}, {names[_channel]}",
                )
    _boundary = getattr(loaded, "boundary_mask", None)
    if _boundary is not None:
        _boundary_image = np.asarray(_boundary)[batch_index.value, 0]
        for _axis in _axes.flat:
            _axis.contour(_boundary_image, levels=[0.5], colors="black", linewidths=0.7)
    _histogram_axis.set_title("Overlaid intensity distributions")
    _histogram_axis.set_xlabel("intensity")
    _histogram_axis.set_ylabel("density")
    if _n_tiles > 1:
        _histogram_axis.legend(fontsize="small", ncol=2)
    _histogram_axis.grid(alpha=0.2)
    _histogram_figure.tight_layout()
    # mo.vstack(
    #     [
    #         mo.md(
    #             "## Tiled monochrome images\n\n"
    #             "Each row (channel) uses one colour range, set by that channel's "
    #             "minimum and maximum over all batches and timesteps."
    #         ),
    #         _figure,
    #         _histogram_figure,
    #     ]
    # )
    _figure
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(batch_index, loaded, mo, np, plt):
    _mask = getattr(loaded, "measurement_mask", None)
    _aux = getattr(loaded, "aux", {})
    _groups = tuple(_aux.get("selected_experiment_groups", ()))
    _group_mask = _aux.get("group_mask")
    _source_conditions = _aux.get("source_conditions")
    _substituted = _aux.get("is_substituted")
    _source_files = _aux.get("source_files")
    _times = tuple(_aux.get("timesteps", ()))

    if _mask is None or not _groups:
        _provenance_view = mo.md(
            "## Measurement availability and provenance\n\n"
            "This dataset does not expose modern micropattern provenance metadata."
        )
    else:
        _mask_array = np.asarray(_mask)[batch_index.value]
        _group_mask_array = np.asarray(_group_mask)[batch_index.value]
        _rows = []
        for _time_index, _time in enumerate(_times):
            for _group_index, _group in enumerate(_groups):
                _available = bool(_group_mask_array[_time_index, _group_index])
                _rows.append(
                    {
                        "time (h)": _time,
                        "experiment group": _group,
                        "available": _available,
                        "source condition": str(_source_conditions[batch_index.value, _time_index, _group_index]) if _available else "—",
                        "control substituted": bool(_substituted[batch_index.value, _time_index, _group_index]) if _available else False,
                        "source file": str(_source_files[batch_index.value, _time_index, _group_index]) if _available else "—",
                    }
                )
        _availability_figure, _availability_axis = plt.subplots(
            figsize=(max(7, _mask_array.shape[1] * 0.55), 3.5)
        )
        _availability_axis.imshow(_mask_array, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        _availability_axis.set_yticks(range(len(_times)), [f"{time}h" for time in _times])
        _availability_axis.set_xlabel("measurement channel")
        _availability_axis.set_ylabel("time")
        _availability_axis.set_title("Available measurements for selected condition/replicate")
        _availability_figure.tight_layout()
        _provenance_view = mo.vstack(
            [
                mo.md("## Measurement availability and knockout provenance"),
                _availability_figure,
                mo.ui.table(_rows, selection=None, pagination=True, page_size=12),
            ]
        )
    _provenance_view
    return


@app.cell(hide_code=True)
def _(
    data,
    fate_reference_percentiles,
    fate_threshold_fractions,
    fate_threshold_mode,
    loaded,
    names,
    np,
):
    # Marker channels and the absolute cutoffs shared by the cell-fate views.
    fate_markers = ("SOX17", "SOX2", "TBXT", "FOXA2")
    _preferred_channels = {
        "SOX17": "cell_fate_s2/SOX17",
        "SOX2": "cell_fate_s1/SOX2",
        "TBXT": "cell_fate_s2/TBXT",
        "FOXA2": "cell_fate_s2/FOXA2",
    }
    fate_channels = {
        _marker: (
            _preferred_channels[_marker]
            if _preferred_channels[_marker] in names
            else _marker if _marker in names else None
        )
        for _marker in fate_markers
    }
    _reference_time = data.shape[1] - 1
    _boundary = getattr(loaded, "boundary_mask", None)
    _measurement_mask = getattr(loaded, "measurement_mask", None)
    fate_absolute_thresholds = {}
    for _marker in fate_markers:
        if fate_channels[_marker] is None:
            continue
        if fate_threshold_mode.value == "absolute":
            fate_absolute_thresholds[_marker] = float(
                fate_threshold_fractions.value[_marker]
            )
            continue
        _channel_index = names.index(fate_channels[_marker])
        _reference_values = []
        for _batch in range(data.shape[0]):
            if (
                _measurement_mask is not None
                and not bool(np.asarray(_measurement_mask)[
                    _batch, _reference_time, _channel_index
                ])
            ):
                continue
            _batch_boundary = (
                np.asarray(_boundary)[_batch, 0].astype(bool)
                if _boundary is not None
                else np.ones(data.shape[-2:], dtype=bool)
            )
            _values = data[_batch, _reference_time, _channel_index][
                _batch_boundary
            ]
            _reference_values.append(_values[np.isfinite(_values)])
        _pooled_values = (
            np.concatenate(_reference_values)
            if _reference_values
            else np.asarray([], dtype=float)
        )
        _reference_percentile = float(
            fate_reference_percentiles.value[_marker]
        )
        _reference_intensity = (
            float(np.nanmax(_pooled_values))
            if _pooled_values.size and _reference_percentile == 100.0
            else float(np.nanpercentile(_pooled_values, _reference_percentile))
            if _pooled_values.size
            else np.nan
        )
        fate_absolute_thresholds[_marker] = (
            float(fate_threshold_fractions.value[_marker])
            * _reference_intensity
        )
    return fate_absolute_thresholds, fate_channels, fate_markers


@app.cell(hide_code=True)
def _(
    data,
    fate_absolute_thresholds,
    fate_batch_indices,
    fate_cell_type_rules,
    fate_channels,
    fate_markers,
    fate_threshold_mode,
    fate_time_indices,
    loaded,
    mo,
    names,
    np,
    plt,
):
    _missing_fate_markers = [
        _marker for _marker, _channel in fate_channels.items()
        if _channel is None
    ]
    if _missing_fate_markers:
        _fate_view = mo.callout(
            "Cell-fate estimation requires channels: "
            + ", ".join(_missing_fate_markers)
            + ". Select the relevant cell-fate experiment groups and reload.",
            kind="warn",
        )
    else:
        _reference_time = data.shape[1] - 1
        _boundary = getattr(loaded, "boundary_mask", None)
        _cell_colors = {
            "Notochord": np.asarray([214, 39, 160], dtype=float) / 255.0,
            "Endoderm": np.asarray([23, 190, 207], dtype=float) / 255.0,
            "Mesoderm": np.asarray([44, 160, 44], dtype=float) / 255.0,
            "Other": np.asarray([127, 127, 127], dtype=float) / 255.0,
        }
        _legend = " · ".join(
            f"<span style='color: rgb({int(255 * _color[0])}, "
            f"{int(255 * _color[1])}, {int(255 * _color[2])})'>■</span> "
            f"{_cell_type}"
            for _cell_type, _color in _cell_colors.items()
        )
        _selected_batches = [int(_value) for _value in fate_batch_indices.value]
        _selected_times = [int(_value) for _value in fate_time_indices.value]
        _fate_rules = fate_cell_type_rules.value
        if not _selected_batches or not _selected_times:
            _fate_view = mo.callout(
                "Select at least one replicate and one timepoint to display.",
                kind="warn",
            )
        else:
            _row_specs = [
                (_batch, _time)
                for _batch in _selected_batches
                for _time in _selected_times
            ]
            _fate_figure, _fate_axes = plt.subplots(
                len(_row_specs),
                5,
                figsize=(20, 5 * len(_row_specs)),
                squeeze=False,
            )
            _prevalence_rows = []
            _fate_aux = getattr(loaded, "aux", {})
            _conditions = tuple(_fate_aux.get("batch_conditions", ()))
            _replicates = tuple(_fate_aux.get("batch_replicates", ()))
            _timesteps = tuple(_fate_aux.get("timesteps", ()))
            for _row, (_batch, _time) in enumerate(_row_specs):
                _selected_boundary = (
                    np.asarray(_boundary)[_batch, 0].astype(bool)
                    if _boundary is not None
                    else np.ones(data.shape[-2:], dtype=bool)
                )
                _selected_images = {
                    _marker: data[
                        _batch,
                        _time,
                        names.index(fate_channels[_marker]),
                    ]
                    for _marker in fate_markers
                }
                _high = {
                    _marker: (
                        _selected_images[_marker]
                        > fate_absolute_thresholds[_marker]
                    )
                    for _marker in fate_markers
                }
                _cell_masks = {}
                for _cell_type, _rule in _fate_rules.items():
                    _rule_conditions = [
                        _high[_marker] if _state == "high" else ~_high[_marker]
                        for _marker, _state in _rule.items()
                        if _state != "any"
                    ]
                    _cell_masks[_cell_type] = (
                        np.logical_and.reduce(_rule_conditions)
                        if _rule_conditions
                        else np.ones_like(next(iter(_high.values())), dtype=bool)
                    )
                _cell_masks["Other"] = ~np.logical_or.reduce(
                    tuple(_cell_masks.values())
                )
                _cell_map = np.zeros((*_selected_boundary.shape, 3), dtype=float)
                for _cell_type, _cell_mask in _cell_masks.items():
                    _cell_map[_cell_mask & _selected_boundary] = _cell_colors[
                        _cell_type
                    ]

                for _column, _marker in enumerate(fate_markers):
                    _axis = _fate_axes[_row, _column]
                    _threshold_mask = _high[_marker] & _selected_boundary
                    _shown_image = np.where(
                        _threshold_mask, _selected_images[_marker], 0.0
                    )
                    _axis.imshow(_shown_image, cmap="gray", vmin=0.0, vmax=1.0)
                    if _row == 0:
                        _axis.set_title(
                            f"{_marker}\ncutoff={fate_absolute_thresholds[_marker]:.3g}"
                        )
                    _axis.set_axis_off()
                _fate_axes[_row, 4].imshow(_cell_map, vmin=0.0, vmax=1.0)
                if _row == 0:
                    _fate_axes[_row, 4].set_title("Estimated cell type")
                _fate_axes[_row, 4].set_axis_off()
                _row_label = (
                    f"Batch {_batch}: {_conditions[_batch]}, "
                    f"replicate {_replicates[_batch]}, "
                    + (
                        f"{_timesteps[_time]}h"
                        if _time < len(_timesteps)
                        else f"time index {_time}"
                    )
                    if _batch < len(_conditions) and _batch < len(_replicates)
                    else (
                        f"Batch {_batch}, {_timesteps[_time]}h"
                        if _time < len(_timesteps)
                        else f"Batch {_batch}, time index {_time}"
                    )
                )
                _fate_axes[_row, 0].text(
                    -0.08,
                    0.5,
                    _row_label,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=_fate_axes[_row, 0].transAxes,
                )
                _colony_pixels = int(np.sum(_selected_boundary))
                for _cell_type, _cell_mask in _cell_masks.items():
                    _prevalence_rows.append({
                        "batch": _batch,
                        "condition": (
                            _conditions[_batch]
                            if _batch < len(_conditions) else "—"
                        ),
                        "replicate": (
                            _replicates[_batch]
                            if _batch < len(_replicates) else "—"
                        ),
                        "time": (
                            _timesteps[_time]
                            if _time < len(_timesteps) else _time
                        ),
                        "cell type": _cell_type,
                        "fraction of colony": (
                            float(np.mean(_cell_mask[_selected_boundary]))
                            if _colony_pixels else np.nan
                        ),
                    })
            _threshold_basis = (
                "absolute 0–1 cutoffs"
                if fate_threshold_mode.value == "absolute"
                else f"cutoffs referenced to time index {_reference_time}"
            )
            _fate_figure.suptitle(
                f"Selected timepoints; {_threshold_basis}", y=1.0
            )
            _fate_figure.tight_layout()
            _lineage_types = tuple(_fate_rules)
            _overlapping_pairs = []
            for _left_index, _left_type in enumerate(_lineage_types):
                for _right_type in _lineage_types[_left_index + 1:]:
                    _rules_conflict = any(
                        _fate_rules[_left_type][_marker] != "any"
                        and _fate_rules[_right_type][_marker] != "any"
                        and _fate_rules[_left_type][_marker]
                        != _fate_rules[_right_type][_marker]
                        for _marker in fate_markers
                    )
                    if not _rules_conflict:
                        _overlapping_pairs.append(
                            f"{_left_type} / {_right_type}"
                        )
            _overlap_view = (
                mo.callout(
                    "These lineage definitions can overlap: "
                    + ", ".join(_overlapping_pairs)
                    + ". Overlapping pixels use the colour of the later lineage.",
                    kind="warn",
                )
                if _overlapping_pairs
                else mo.callout(
                    "The selected lineage definitions are mutually exclusive.",
                    kind="success",
                )
            )
            _rule_summary = "; ".join(
                f"{_cell_type}: "
                + ", ".join(
                    f"{_marker}={_state}"
                    for _marker, _state in _rule.items()
                )
                for _cell_type, _rule in _fate_rules.items()
            )
            _fate_view = mo.vstack(
                [
                    _overlap_view,
                    mo.md(_legend),
                    _fate_figure,
                    mo.ui.table(_prevalence_rows, selection=None),
                    mo.md(
                        f"Current rules — {_rule_summary}. Marker panels show "
                        "only above-threshold intensity; all other pixels are zero."
                    ),
                ]
            )
    _fate_view
    # plt.show()
    return


@app.cell(hide_code=True)
def _(
    data,
    fate_absolute_thresholds,
    fate_channels,
    fate_markers,
    loaded,
    mo,
    names,
    np,
    plt,
):
    _available_markers = [
        _marker for _marker in fate_markers if fate_channels[_marker] is not None
    ]
    if not _available_markers:
        _count_view = mo.callout(
            "No cell-fate marker channels are loaded.", kind="warn"
        )
    else:
        _boundary = getattr(loaded, "boundary_mask", None)
        _measurement_mask = getattr(loaded, "measurement_mask", None)
        _aux = getattr(loaded, "aux", {})
        _conditions = tuple(_aux.get("batch_conditions", ()))
        _replicates = tuple(_aux.get("batch_replicates", ()))
        _timesteps = tuple(_aux.get("timesteps", ()))
        _times = (
            np.asarray(_timesteps, dtype=float)
            if len(_timesteps) == data.shape[1]
            else np.arange(data.shape[1], dtype=float)
        )
        _time_label = (
            "time (h)" if len(_timesteps) == data.shape[1] else "time index"
        )
        _batch_labels = [
            f"{_conditions[_batch]}, rep {_replicates[_batch]}"
            if _batch < len(_conditions) and _batch < len(_replicates)
            else f"Batch {_batch}"
            for _batch in range(data.shape[0])
        ]
        _condition_names = sorted(set(_conditions)) if _conditions else [None]
        _condition_colors = {
            _condition: plt.get_cmap("tab10")(_index % 10)
            for _index, _condition in enumerate(_condition_names)
        }
        _time_colors = plt.get_cmap("viridis")(
            np.linspace(0.0, 1.0, data.shape[1])
        )
        _bins = np.linspace(0.0, 1.0, 101)

        _count_figure, _count_axes = plt.subplots(
            1, len(_available_markers),
            figsize=(5 * len(_available_markers), 4), squeeze=False,
        )
        _hist_figure, _hist_axes = plt.subplots(
            1, len(_available_markers),
            figsize=(5 * len(_available_markers), 4), squeeze=False,
        )
        for _column, _marker in enumerate(_available_markers):
            _channel_index = names.index(fate_channels[_marker])
            _threshold = fate_absolute_thresholds[_marker]
            # Colony pixel values per [batch][time]; None where not measured.
            _values = [
                [
                    None
                    if _measurement_mask is not None
                    and not bool(np.asarray(_measurement_mask)[
                        _batch, _time, _channel_index
                    ])
                    else np.asarray(data[_batch, _time, _channel_index])[
                        np.asarray(_boundary)[_batch, 0].astype(bool)
                        if _boundary is not None
                        else np.ones(data.shape[-2:], dtype=bool)
                    ]
                    for _time in range(data.shape[1])
                ]
                for _batch in range(data.shape[0])
            ]

            _count_axis = _count_axes[0, _column]
            for _batch in range(data.shape[0]):
                _counts = [
                    np.nan if _pixels is None
                    else float(np.sum(_pixels > _threshold))
                    for _pixels in _values[_batch]
                ]
                _condition = _conditions[_batch] if _conditions else None
                _count_axis.plot(
                    _times, _counts, marker="o", alpha=0.8,
                    color=_condition_colors[_condition],
                    label=_batch_labels[_batch],
                )
            _count_axis.set(
                title=f"{_marker} pixels above {_threshold:.3g}",
                xlabel=_time_label,
                ylabel="number of high pixels",
            )
            _count_axis.grid(alpha=0.2)

            _hist_axis = _hist_axes[0, _column]
            for _time in range(data.shape[1]):
                _pooled = [
                    _pixels for _pixels in
                    (_values[_batch][_time] for _batch in range(data.shape[0]))
                    if _pixels is not None
                ]
                if not _pooled:
                    continue
                _pooled = np.concatenate(_pooled)
                _hist_axis.hist(
                    _pooled[np.isfinite(_pooled)],
                    bins=_bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=_time_colors[_time],
                    label=(
                        f"{_timesteps[_time]}h"
                        if _time < len(_timesteps) else f"t={_time}"
                    ),
                )
            _hist_axis.axvline(
                _threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"threshold {_threshold:.3g}",
            )
            _hist_axis.set(
                title=f"{_marker} colony intensities",
                xlabel="intensity",
                ylabel="density",
                xlim=(0.0, 1.0),
                yscale="log",
            )
            _hist_axis.legend(fontsize="small")
        _count_axes[0, -1].legend(
            fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1.0)
        )
        _count_figure.tight_layout()
        _hist_figure.tight_layout()
        _count_view = mo.vstack([
            mo.md(
                "### Above-threshold pixel counts\n\n"
                "Number of colony pixels above each marker's current cutoff, "
                "for every loaded replicate and timepoint (coloured by "
                "condition). Histograms pool colony pixels over replicates at "
                "each timepoint; the dashed line is the cutoff."
            ),
            _count_figure,
            _hist_figure,
        ])
    _count_view
    return


if __name__ == "__main__":
    app.run()
