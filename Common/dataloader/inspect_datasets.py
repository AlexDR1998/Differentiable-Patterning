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
