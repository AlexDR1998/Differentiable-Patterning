# /// script
# dependencies = [
#   "marimo",
#   "anywidget",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pyyaml",
#   "scipy",
#   "scikit-image",
# ]
# ///

"""Ground truth for the 0 h SOX2/LMBR nucleus segmentations, and their scores.

Run from the repository root with:

    marimo edit Experiments/micropatterns/nucleus_segmentation_explorer.py

The workflow is in ``nucleus_segmentation.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    from Experiments.micropatterns.box_picker import PointPicker, rgb_data_url
    from Experiments.micropatterns.nucleus_segmentation import (
        annotation_crops,
        crop_id,
        projection_of,
        read_annotations,
        save_annotations,
        score_labels,
        segmentation_root,
        stored_labels,
        stored_methods,
    )

    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env", override=False)
    # Typical nucleus diameter (µm, from the segmented 0 h nuclei); a found nucleus matches a click within half of it
    NUCLEUS_UM = 12.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Nucleus segmentation: ground truth and scores

    - Click the centre of every nucleus in each crop (SOX2 green, LMBR magenta); click a mark to remove it
    - Clicks are saved as you go, so you can stop and come back
    - Each stored segmentation is scored against the clicks: a nucleus found within half a nucleus of a click matches it
    """)
    return


@app.cell(hide_code=True)
def _():
    data_root = mo.ui.text(
        str(Path(os.environ.get("DATA_PATH_BASE", "../Data")) / "260726_nca_dataset"),
        label="260726 dataset folder",
        full_width=True,
    )
    data_root
    return (data_root,)


@app.cell(hide_code=True)
def _(data_root):
    dataset = Path(data_root.value).expanduser()
    crops = annotation_crops(dataset)
    inputs_folder = segmentation_root(dataset) / "inputs"
    mo.stop(
        not inputs_folder.exists(),
        mo.callout("No exported inputs: run nucleus_segmentation.export_inputs first.", kind="warn"),
    )
    # Clicks of every crop, kept here and saved on each change
    clicks = read_annotations()

    @mo.lru_cache(maxsize=16)
    def load_input(relative):
        from Experiments.micropatterns.nucleus_segmentation import image_name

        with np.load(inputs_folder / f"{image_name(relative)}.npz") as stored:
            return stored["sox2"], stored["lmbr"], stored["colony"], float(stored["pixel_um"])

    def composite(relative, crop):
        """SOX2 green and LMBR magenta of a crop, each scaled inside the colony."""
        sox2, lmbr, colony, _ = load_input(relative)
        box = (slice(crop["top"], crop["top"] + crop["size"]), slice(crop["left"], crop["left"] + crop["size"]))
        scaled = []
        for channel in (sox2, lmbr):
            low, high = np.percentile(channel[colony], [1.0, 99.8])
            scaled.append(np.clip((channel[box] - low) / max(high - low, 1e-6), 0.0, 1.0))
        return np.stack([scaled[1], scaled[0], scaled[1]], axis=-1)

    return clicks, composite, crops, dataset, load_input


@app.cell(hide_code=True)
def _(clicks, crops):
    _options = {
        f"{_i + 1}. {projection_of(_c['image'])} · {Path(_c['image']).stem} ({len(clicks.get(crop_id(_c), []))} marked)": _i
        for _i, _c in enumerate(crops)
    }
    crop_choice = mo.ui.dropdown(_options, value=next(iter(_options)), label="Crop")
    crop_choice
    return (crop_choice,)


@app.cell(hide_code=True)
def _(clicks, composite, crop_choice, crops):
    crop = crops[int(crop_choice.value)]
    # Clicks are stored in image pixels; the widget works in crop pixels
    _start = [[_r - crop["top"], _c - crop["left"]] for _r, _c in clicks.get(crop_id(crop), [])]
    picker = mo.ui.anywidget(PointPicker(
        background=rgb_data_url(composite(crop["image"], crop)), points=_start, display=720,
    ))
    mo.vstack([
        mo.md(f"- `{crop['image']}`, {crop['size']} px square from row {crop['top']}, column {crop['left']}"),
        picker,
    ])
    return crop, picker


@app.cell(hide_code=True)
def _(clicks, crop, crops, picker):
    # Saved on every click
    clicks[crop_id(crop)] = [[_r + crop["top"], _c + crop["left"]] for _r, _c in picker.value["points"]]
    _path = save_annotations(crops, clicks)
    mo.md(
        f"- Saved {sum(len(_p) for _p in clicks.values())} clicks over {len(clicks)} crops to "
        f"`{_path.relative_to(repository_root)}`"
    )
    return


@app.cell(hide_code=True)
def _():
    refresh = mo.ui.button(label="Rescan stored segmentations and scores", value=0, on_click=lambda _n: _n + 1)
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(clicks, crops, dataset, load_input, picker, refresh):
    # Re-scored after clicks (picker) and on request (refresh)
    _ = picker.value, refresh.value
    methods = stored_methods(dataset)
    _rows = []
    for _method in methods:
        for _crop in crops:
            _clicked = clicks.get(crop_id(_crop), [])
            _labels = stored_labels(dataset, _method, _crop["image"])
            if not _clicked or _labels is None:
                continue
            _pixel_um = load_input(_crop["image"])[3]
            _rows.append({
                "method": _method,
                "crop": crop_id(_crop),
                "projection": projection_of(_crop["image"]),
                **score_labels(_labels, _crop, _clicked, 0.5 * NUCLEUS_UM / _pixel_um),
            })
    scores = pd.DataFrame(_rows)
    if scores.empty:
        _view = mo.callout("No crop has clicks and stored labels yet.", kind="info")
    else:
        _total = scores.groupby("method")[["clicked", "found", "matched", "extra", "missed"]].sum()
        _total["precision"] = _total["matched"] / _total["found"].clip(lower=1)
        _total["recall"] = _total["matched"] / _total["clicked"].clip(lower=1)
        _total["f1"] = 2 * _total["matched"] / (2 * _total["matched"] + _total["extra"] + _total["missed"]).clip(lower=1)
        _total["count_ratio"] = _total["found"] / _total["clicked"].clip(lower=1)
        _view = mo.vstack([
            mo.md("### Scores (all clicked crops)\n\n- count ratio: nuclei found over nuclei clicked (1 is right)"),
            mo.ui.table(_total.sort_values("f1", ascending=False).round(3).reset_index(), selection=None),
            mo.accordion({"Per crop": mo.ui.table(scores.round(3), selection=None, page_size=12)}),
        ])
    _view
    return (methods,)


@app.cell(hide_code=True)
def _(methods):
    mo.stop(not methods)
    method_choice = mo.ui.dropdown(methods, value=methods[0], label="Show segmentation")
    method_choice
    return (method_choice,)


@app.cell(hide_code=True)
def _(clicks, composite, crop, dataset, method_choice):
    from skimage.segmentation import find_boundaries

    from Experiments.micropatterns.nucleus_segmentation import label_centres

    # The chosen segmentation on this crop, beside the clicks
    _labels = stored_labels(dataset, method_choice.value, crop["image"])
    mo.stop(_labels is None, mo.callout(f"{method_choice.value} has no labels for this image.", kind="warn"))
    _size = crop["size"]
    _crop_labels = _labels[crop["top"] : crop["top"] + _size, crop["left"] : crop["left"] + _size]
    _rgb = composite(crop["image"], crop)
    _outlined = _rgb.copy()
    _outlined[find_boundaries(_crop_labels, mode="inner")] = (1.0, 1.0, 1.0)
    _clicked = np.asarray(clicks.get(crop_id(crop), []), dtype=float).reshape(-1, 2) - [crop["top"], crop["left"]]
    _centres = label_centres(_crop_labels)
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 6))
    for _axis, _image, _title in zip(_axes, (_rgb, _outlined), ("Clicks", f"{method_choice.value}: {len(_centres)} nuclei")):
        _axis.imshow(_image, interpolation="nearest")
        _axis.set_title(_title, fontsize=10)
        _axis.set_xticks([])
        _axis.set_yticks([])
    _axes[0].scatter(_clicked[:, 1], _clicked[:, 0], s=14, c="yellow", edgecolors="black", linewidths=0.5)
    _axes[1].scatter(_centres[:, 1], _centres[:, 0], s=10, c="cyan", edgecolors="black", linewidths=0.4)
    _figure.tight_layout()
    _figure
    return


if __name__ == "__main__":
    app.run()
