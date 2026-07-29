"""Interactive visual and statistical inspection of repository datasets.

Run with:
    marimo edit Common/dataloader/inspect_datasets.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import ast
    from pathlib import Path
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
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


@app.cell
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
    controls = mo.vstack(
        [
            mo.hstack([dataset_kind, downsample, batches, knockout]),
            root,
            filenames,
            mo.hstack([timesteps, align, percentile_low, percentile_high]),
            processing,
            load_button,
        ]
    )
    controls
    return (
        align,
        batches,
        dataset_kind,
        downsample,
        filenames,
        knockout,
        load_button,
        percentile_high,
        percentile_low,
        processing,
        root,
        timesteps,
    )


@app.cell
def _(
    align,
    batches,
    dataset_kind,
    downsample,
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
    timesteps,
):
    load_button
    selected_times = tuple(int(value.strip()) for value in timesteps.value.split(",") if value.strip())
    selected_files = tuple(value.strip() for value in filenames.value.split(",") if value.strip())
    ordered_processing = tuple(processing.value)
    if dataset_kind.value == "micropattern_260726":
        loaded = load_micropattern_260726(
            root.value,
            timesteps=selected_times,
            downsample=downsample.value,
            replicate_count=batches.value,
            align=align.value,
            hist_eqs=(percentile_low.value, percentile_high.value),
        )
    elif dataset_kind.value == "micropattern_grouped":
        ko_time = {"baseline": None, "ko0": 0, "ko24": 24}[knockout.value]
        loaded = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=root.value,
            DOWNSAMPLE=downsample.value,
            BATCHES=batches.value,
            TIMESTEPS=selected_times,
            FILTER_KN_TIME=ko_time,
            HIST_EQS=(percentile_low.value, percentile_high.value),
            PROCESSING_MODES=ordered_processing,
        )
    elif dataset_kind.value == "micropattern_4ch":
        loaded = load_micropattern_circle_4ch_individual(
            impath=root.value,
            DOWNSAMPLE=downsample.value,
            BATCHES=batches.value,
            TIMESTEPS=selected_times,
            HIST_EQS=(percentile_low.value, percentile_high.value),
            PROCESSING_MODES=ordered_processing,
        )
    elif dataset_kind.value == "emoji":
        loaded = load_emoji_sequence(selected_files, root.value, downsample.value, True)
    else:
        loaded = load_textures(selected_files, root.value, downsample.value, True)
    return (loaded,)


@app.cell
def _(loaded, mo, np):
    data = np.asarray(loaded.data)
    names = tuple(getattr(loaded, "channel_names", ()))
    if not names:
        names = tuple(f"channel {index}" for index in range(data.shape[2]))
    summary = {
        "shape [B,T,C,X,Y]": tuple(data.shape),
        "dtype": str(data.dtype),
        "minimum": float(np.nanmin(data)),
        "maximum": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "standard deviation": float(np.nanstd(data)),
        "finite fraction": float(np.isfinite(data).mean()),
    }
    batch_index = mo.ui.slider(0, data.shape[0] - 1, value=0, label="Batch")
    time_index = mo.ui.slider(0, data.shape[1] - 1, value=0, label="Time")
    channel_index = mo.ui.slider(0, data.shape[2] - 1, value=0, label="Channel")
    zero_to_nan = mo.ui.checkbox(value=True, label="Discard zero values (set to NaN)")
    mo.vstack([mo.md("## Loaded data"), mo.ui.table([summary]), mo.hstack([batch_index, time_index, channel_index,zero_to_nan])])
    return batch_index, channel_index, data, names, time_index, zero_to_nan


@app.cell
def _(
    batch_index,
    channel_index,
    data,
    loaded,
    mo,
    names,
    np,
    plt,
    time_index,
    zero_to_nan,
):
    image = data[batch_index.value, time_index.value, channel_index.value]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    if zero_to_nan.value:
        image = np.where(image == 0, np.nan, image)
    finite = image[np.isfinite(image)]
    axes[0].imshow(image, cmap="viridis")
    axes[0].set_title(names[channel_index.value])
    axes[0].axis("off")
    axes[1].hist(finite.ravel(), bins=80)
    axes[1].set_title("Pixel distribution")
    axes[1].set_xlabel("value")
    boundary = getattr(loaded, "boundary_mask", None)
    if boundary is not None:
        axes[0].contour(np.asarray(boundary)[batch_index.value, 0], levels=[0.5], colors="white", linewidths=0.7)
    figure.tight_layout()
    mo.vstack([mo.md("## Image and histogram"), figure])
    return


@app.cell
def _(data, mo, names, np, plt):
    channel_values = data.transpose(2, 0, 1, 3, 4).reshape(data.shape[2], -1)
    stats = [
        {
            "channel": name,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "p01": float(np.nanpercentile(values, 1)),
            "p50": float(np.nanpercentile(values, 50)),
            "p99": float(np.nanpercentile(values, 99)),
        }
        for name, values in zip(names, channel_values)
    ]
    _figure, axis = plt.subplots(figsize=(max(7, len(names) * 0.6), 3.5))
    axis.boxplot([values[np.isfinite(values)] for values in channel_values], showfliers=False)
    axis.set_xticks(range(1, len(names) + 1), names, rotation=60, ha="right")
    axis.set_ylabel("value")
    _figure.tight_layout()
    mo.vstack([mo.md("## Channel statistics"), mo.ui.table(stats), _figure])
    return


if __name__ == "__main__":
    app.run()
