"""Interactive visual and statistical inspection of repository datasets.

Run with:
    marimo edit Common/dataloader/inspect_datasets.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.dataloader.emoji import load_emoji_sequence
    from Common.dataloader.micropattern import (
        load_micropattern_260726,
        load_micropattern_circle_4ch_individual,
        load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
    )
    from Common.dataloader.preprocessing import ProcessingStep
    from Common.dataloader.texture import load_textures

    return (
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
def _(ProcessingStep, mo):
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
    percentile_low = mo.ui.number(0.0, 99.0, value=0.5, step=0.1, label="Histogram low percentile")
    percentile_high = mo.ui.number(1.0, 100.0, value=99.95, step=0.05, label="Histogram high percentile")
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
def _(
    align,
    batches,
    conditions,
    dataset_kind,
    downsample,
    experiment_groups,
    filenames,
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
        value=["0"],
        label="Timesteps to tile",
    )
    channel_indices = mo.ui.multiselect(
        options={name: index for index, name in enumerate(names)},
        value=[names[0]],
        label="Channels to tile",
    )
    zero_to_nan = mo.ui.checkbox(value=True, label="Discard zero values (set to NaN)")
    mo.vstack(
        [
            mo.md("## Loaded data"),
            mo.ui.table([_summary]),
            mo.hstack([batch_index, time_indices, channel_indices, zero_to_nan]),
        ]
    )
    return batch_index, channel_indices, data, names, time_indices, zero_to_nan


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


@app.cell
def _(
    batch_index,
    channel_indices,
    data,
    loaded,
    mo,
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
    )
    _histogram_figure, _histogram_axis = plt.subplots(figsize=(8, 4))
    for _column, _time in enumerate(_selected_times):
        for _row, _channel in enumerate(_selected_channels):
            _image = data[batch_index.value, _time, _channel]
            if zero_to_nan.value:
                _image = np.where(_image == 0, np.nan, _image)
            _finite = _image[np.isfinite(_image)]
            _axes[_row, _column].imshow(_image, cmap="viridis")
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
    _figure.tight_layout()
    _histogram_figure.tight_layout()
    mo.vstack([mo.md("## Tiled monochrome images"), _figure, _histogram_figure])
    return


@app.cell(hide_code=True)
def _(data, loaded, mo):
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
            0.05,
            0.95,
            value=0.3,
            step=0.05,
            label=f"{_marker} threshold fraction",
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
    mo.vstack(
        [
            mo.md(
                "## Cell-fate threshold explorer\n\n"
                "Each expression cutoff is the selected fraction of that marker's "
                "reference percentile across available final-timestep replicates, "
                "so the same absolute cutoffs are used at every displayed timepoint."
            ),
            mo.hstack([fate_batch_indices, fate_time_indices], widths="equal"),
            mo.hstack(
                [fate_threshold_fractions, fate_reference_percentiles],
                widths="equal",
            ),
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
def _(
    data,
    fate_batch_indices,
    fate_cell_type_rules,
    fate_reference_percentiles,
    fate_threshold_fractions,
    fate_time_indices,
    loaded,
    mo,
    names,
    np,
    plt,
):
    _fate_markers = ("SOX17", "SOX2", "TBXT", "FOXA2")
    _preferred_channels = {
        "SOX17": "cell_fate_s2/SOX17",
        "SOX2": "cell_fate_s1/SOX2",
        "TBXT": "cell_fate_s2/TBXT",
        "FOXA2": "cell_fate_s2/FOXA2",
    }
    _fate_channels = {
        _marker: (
            _preferred_channels[_marker]
            if _preferred_channels[_marker] in names
            else _marker if _marker in names else None
        )
        for _marker in _fate_markers
    }
    _missing_fate_markers = [
        _marker for _marker, _channel in _fate_channels.items()
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
        _measurement_mask = getattr(loaded, "measurement_mask", None)
        _absolute_thresholds = {}
        for _marker in _fate_markers:
            _channel_index = names.index(_fate_channels[_marker])
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
            _absolute_thresholds[_marker] = (
                float(fate_threshold_fractions.value[_marker])
                * _reference_intensity
            )

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
                        names.index(_fate_channels[_marker]),
                    ]
                    for _marker in _fate_markers
                }
                _high = {
                    _marker: (
                        _selected_images[_marker] > _absolute_thresholds[_marker]
                    )
                    for _marker in _fate_markers
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

                for _column, _marker in enumerate(_fate_markers):
                    _axis = _fate_axes[_row, _column]
                    _shown_image = np.where(
                        _selected_boundary, _selected_images[_marker], np.nan
                    )
                    _axis.imshow(_shown_image, cmap="gray")
                    _threshold_mask = _high[_marker] & _selected_boundary
                    if np.any(_threshold_mask) and not np.all(_threshold_mask):
                        _axis.contour(
                            _threshold_mask,
                            levels=[0.5],
                            colors="cyan",
                            linewidths=0.6,
                        )
                    if _row == 0:
                        _axis.set_title(
                            f"{_marker}\ncutoff={_absolute_thresholds[_marker]:.3g}"
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
            _fate_figure.suptitle(
                f"Selected timepoints; cutoffs referenced to time index "
                f"{_reference_time}",
                y=1.0,
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
                        for _marker in _fate_markers
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
                        f"Current rules — {_rule_summary}. Cyan contours mark "
                        "pixels above each displayed cutoff."
                    ),
                ]
            )
    _fate_view
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


if __name__ == "__main__":
    app.run()
