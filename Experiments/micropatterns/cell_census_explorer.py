# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
#   "omegaconf",
#   "python-dotenv",
#   "tqdm",
#   "scikit-image",
# ]
# ///

"""Cell numbers and cell types of the measured colonies over time, by nuclei.

Run from the repository root with:

    marimo edit Experiments/micropatterns/cell_census_explorer.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from tqdm.auto import tqdm

    from Common.dataloader.cell_types import OTHER, SEVERAL, label_names, load_cell_type_rules
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Experiments.micropatterns.cell_census import (
        census_label_shares,
        colony_census,
        load_census,
        save_census,
    )
    from Experiments.micropatterns.cell_counts import CELL_EXPANSION_UM
    from Experiments.micropatterns.figures.style import cell_type_colour
    from Experiments.micropatterns.ko_evaluation import parse_one_based_indices, rules_projection
    from Experiments.model_registry import ModelRegistry

    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env", override=False)
    CELL_TYPE_DIRECTORY = repository_root / "Experiments/micropatterns/conf/cell_types"
    CONDITIONS = {"Baseline": "baseline", "KO 24h": "ko_24h", "KO 0h": "ko_0h"}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cell census of the measured colonies

    - Nuclei (LMBR) counted in the raw images of every replicate and time, grown into cells
    - Cell types estimated per ring as elsewhere, once counting every cell once and once counting pixels
    - Images follow the chosen model's data settings: a slice-trained model gives the slices
    - Checks: does the cell number change over time, and does the share of cells differ from the share of pixels?
    """)
    return


@app.cell(hide_code=True)
def _():
    _cell_type_files = {
        f"{_path.name} (tuned on {rules_projection(load_cell_type_rules(_path))})": str(_path)
        for _path in sorted(CELL_TYPE_DIRECTORY.glob("*.yaml"))
    }
    _slice_files = [_label for _label in _cell_type_files if _label.endswith("(tuned on slice)")]
    store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(repository_root / "models")), label="Model store", full_width=True
    )
    data_root = mo.ui.text(
        os.environ.get("DATA_PATH_BASE", ""), label="Data root (contains 260726_nca_dataset)", full_width=True
    )
    model_id = mo.ui.text("", label="Model ID (sets the images and their scaling)", full_width=True)
    cell_type_file = mo.ui.dropdown(
        options=_cell_type_files,
        value=(_slice_files or list(_cell_type_files) or [None])[-1],
        label="Cell type file",
        full_width=True,
    )
    conditions = mo.ui.multiselect(options=CONDITIONS, value=["Baseline"], label="Conditions")
    replicates = mo.ui.text("1-4", label="Replicates (for example: 1,2,4-6)")
    rings = mo.ui.number(1, 40, value=12, step=1, label="Rings")
    expansion = mo.ui.number(0.0, 10.0, value=CELL_EXPANSION_UM, step=0.5, label="Cell reach beyond nucleus (µm)")
    run_census = mo.ui.run_button(label="Run census")
    mo.vstack([
        store_root,
        data_root,
        model_id,
        cell_type_file,
        mo.hstack([conditions, replicates, rings, expansion], justify="start"),
        run_census,
    ])
    return (
        cell_type_file,
        conditions,
        data_root,
        expansion,
        model_id,
        replicates,
        rings,
        run_census,
        store_root,
    )


@app.cell(hide_code=True)
def _():
    # The latest census, from a run or a loaded file
    get_census, set_census = mo.state(None)
    return get_census, set_census


@app.cell(hide_code=True)
def _(
    cell_type_file,
    conditions,
    data_root,
    expansion,
    model_id,
    replicates,
    rings,
    run_census,
    set_census,
    store_root,
):
    if not run_census.value:
        _status = mo.md("Set the model and data, then run the census (or load a saved one below).")
    else:
        if not model_id.value.strip() or not data_root.value.strip():
            raise ValueError("Set the model ID and the data root.")
        _bundle = ModelRegistry(Path(store_root.value).expanduser()).get(model_id.value.strip())
        _projection = _bundle.config.data.micropattern.projection
        _rules = load_cell_type_rules(cell_type_file.value)
        _replicates = parse_one_based_indices(replicates.value)
        _hours = len(_bundle.config.data.micropattern.timesteps)
        _rows = []
        with tqdm(total=len(conditions.value) * len(_replicates) * _hours * 4, desc="Census", unit="image") as _bar:
            for _condition in conditions.value:
                _rows.extend(colony_census(
                    _bundle, data_root.value, _condition, _replicates, _rules, int(rings.value),
                    expansion_um=float(expansion.value), cache_dir=cache_dir_from_environment(),
                    progress=_bar.update,
                ))
        _settings = {
            "model_id": _bundle.id,
            "projection": _projection,
            "cell_type_file": str(Path(cell_type_file.value).relative_to(repository_root)),
            "rings": int(rings.value),
            "expansion_um": float(expansion.value),
        }
        set_census((_rows, _settings))
        _status = mo.vstack([
            mo.callout(f"Counted {len(_rows)} panel images ({_projection} projection).", kind="success"),
            *(
                [mo.callout(f"This model was trained on the {_projection} projection, not slices.", kind="warn")]
                if _projection != "slice" else []
            ),
        ])
    _status
    return


@app.cell(hide_code=True)
def _():
    census_path = mo.ui.text(
        "output/cell_census/census.npz", label="Census file (relative to the repository root)", full_width=True
    )
    save_button = mo.ui.run_button(label="Save census")
    load_button = mo.ui.run_button(label="Load census")
    mo.hstack([census_path, save_button, load_button], justify="start")
    return census_path, load_button, save_button


@app.cell(hide_code=True)
def _(census_path, get_census, load_button, save_button, set_census):
    _path = Path(census_path.value.strip()).expanduser()
    _path = _path if _path.is_absolute() else repository_root / _path
    if save_button.value:
        if get_census() is None:
            _file_status = mo.callout("Run or load a census first.", kind="warn")
        else:
            save_census(_path, *get_census())
            _file_status = mo.callout(f"Saved to {_path}", kind="success")
    elif load_button.value:
        set_census(load_census(_path))
        _file_status = mo.callout(f"Loaded {_path}", kind="success")
    else:
        _file_status = mo.md("")
    _file_status
    return


@app.cell(hide_code=True)
def _(get_census):
    _census = get_census()
    census_rows, census_settings = _census if _census is not None else ([], {})
    # Shares need the cell type file the counts were made with (its thresholds)
    census_rules = (
        load_cell_type_rules(repository_root / census_settings["cell_type_file"]) if census_settings else None
    )
    census_counts = pd.DataFrame([
        {key: _row[key] for key in ("replicate", "condition", "hour", "panel", "nuclei", "colony_mm2")}
        for _row in census_rows
    ])
    if not census_counts.empty:
        census_counts["nuclei_per_mm2"] = census_counts["nuclei"] / census_counts["colony_mm2"]
    mo.md(
        f"- Census of model `{census_settings['model_id']}` ({census_settings['projection']}), "
        f"cell types from `{census_settings['cell_type_file']}`"
        if census_settings else ""
    )
    return census_counts, census_rows, census_rules


@app.cell(hide_code=True)
def _(census_rules):
    _labels = list(label_names(census_rules)) if census_rules is not None else []
    shown_cell_types = mo.ui.multiselect(
        options=_labels,
        value=[_label for _label in _labels if _label not in (SEVERAL, OTHER)],
        label="Cell types shown",
    )
    resolve_several = mo.ui.slider(0.0, 1.0, value=0.0, step=0.05, label="Resolve multiple matches", show_value=True)
    mo.hstack([shown_cell_types, resolve_several], justify="start")
    return resolve_several, shown_cell_types


@app.cell(hide_code=True)
def _(census_rows, census_rules, resolve_several):
    census_shares = pd.DataFrame(
        census_label_shares(census_rows, census_rules, resolve_several=resolve_several.value)
        if census_rules is not None else []
    )
    return (census_shares,)


@app.cell(hide_code=True)
def _(census_counts):
    # Cells per slice and per mm² over time (dots: each replicate and panel)
    if census_counts.empty:
        _view = mo.md("")
    else:
        _figure, _axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
        for _condition, _rows in census_counts.groupby("condition", sort=False):
            for _axis, _column in zip(_axes, ("nuclei", "nuclei_per_mm2")):
                _axis.scatter(_rows["hour"], _rows[_column], s=10, alpha=0.4)
                _mean = _rows.groupby("hour")[_column].mean()
                _axis.plot(
                    _mean.index, _mean.to_numpy(), marker="o",
                    label={_key: _label for _label, _key in CONDITIONS.items()}.get(_condition, _condition),
                )
        _axes[0].set(xlabel="Time (h)", ylabel="Nuclei in the colony")
        _axes[1].set(xlabel="Time (h)", ylabel="Nuclei per mm²")
        _axes[0].legend()
        _figure.tight_layout()
        _view = _figure
    _view
    return


@app.cell(hide_code=True)
def _(census_shares, shown_cell_types):
    # Share of cells (solid) and of pixels (dashed) per cell type, mean ± std over replicates
    if census_shares.empty:
        _view = mo.md("")
    else:
        _shown = census_shares[census_shares["cell_type"].isin(shown_cell_types.value)]
        _conditions = list(dict.fromkeys(_shown["condition"]))
        _figure, _axes = plt.subplots(
            2, len(_conditions), figsize=(5.5 * len(_conditions), 7), dpi=130, squeeze=False
        )
        for _column, _condition in enumerate(_conditions):
            for _cell_type, _rows in _shown[_shown["condition"] == _condition].groupby("cell_type", sort=False):
                _summary = _rows.groupby("hour")[["share_cells", "share_pixels", "area_per_cell_um2"]].agg(["mean", "std"])
                _hours = _summary.index.to_numpy()
                _colour = cell_type_colour(_cell_type)
                for _measure, _style in (("share_cells", "-"), ("share_pixels", "--")):
                    _axes[0, _column].plot(
                        _hours, _summary[(_measure, "mean")], _style, marker="o", color=_colour,
                        label=f"{_cell_type} ({'cells' if _measure == 'share_cells' else 'pixels'})",
                    )
                    _axes[0, _column].fill_between(
                        _hours, _summary[(_measure, "mean")] - _summary[(_measure, "std")].fillna(0),
                        _summary[(_measure, "mean")] + _summary[(_measure, "std")].fillna(0),
                        color=_colour, alpha=0.12,
                    )
                _axes[1, _column].errorbar(
                    _hours, _summary[("area_per_cell_um2", "mean")], yerr=_summary[("area_per_cell_um2", "std")],
                    marker="o", color=_colour, capsize=3, label=_cell_type,
                )
            _title = {_key: _label for _label, _key in CONDITIONS.items()}.get(_condition, _condition)
            _axes[0, _column].set(title=_title, xlabel="Time (h)", ylabel="Share", ylim=(0.0, None))
            _axes[1, _column].set(xlabel="Time (h)", ylabel="Colony area per cell (µm²)")
        _axes[0, 0].legend(fontsize="small")
        _axes[1, 0].legend(fontsize="small")
        _figure.tight_layout()
        _view = _figure
    _view
    return


@app.cell(hide_code=True)
def _(census_counts, census_shares):
    mo.vstack([
        mo.ui.table(census_shares, selection=None, page_size=10) if not census_shares.empty else mo.md(""),
        mo.hstack([
            mo.download(
                lambda: census_counts.to_csv(index=False).encode("utf-8"),
                filename="cell_census_counts.csv", mimetype="text/csv",
                disabled=census_counts.empty, label="Download nuclei counts CSV",
            ),
            mo.download(
                lambda: census_shares.to_csv(index=False).encode("utf-8"),
                filename="cell_census_shares.csv", mimetype="text/csv",
                disabled=census_shares.empty, label="Download cell-type shares CSV",
            ),
        ], justify="start"),
    ])
    return


if __name__ == "__main__":
    app.run()
