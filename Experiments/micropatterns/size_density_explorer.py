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
# ]
# ///

"""Colony size × cell density sweep of model-registry NCAs, with radial kymographs.

Run from the repository root with:

    marimo edit Experiments/micropatterns/size_density_explorer.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[2]

    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from tqdm.auto import tqdm

    from Common.dataloader.cell_type_shares import colony_label_shares
    from Common.dataloader.cell_types import OTHER, SEVERAL, label_names, load_cell_type_rules
    from Common.dataloader.micropattern_projections import PROJECTIONS
    from Experiments.model_registry import ModelRegistry, load_selection
    # Cell placement shared with the density × KO sweep (synthetic_ic.py)
    from Experiments.micropatterns.density_ko_sweep import format_density, prepare_model_cells
    from Experiments.micropatterns.ko_evaluation import (
        REFERENCE_LABELS,
        REFERENCE_LINESTYLES,
        REFERENCE_MARKERS,
        cell_edges,
        draw_reference_strip,
        reference_legend_handles,
        model_option_label,
        model_stain_counts,
        parse_one_based_indices,
        read_model_settings,
        rules_projection,
        uncertainty_band,
    )
    # The sweep (run here or as a remote job) and its analysis helpers
    from Experiments.micropatterns.size_density_sweep import (
        SizeDensitySettings,
        front_positions,
        front_speed,
        job_configs,
        load_size_density_results,
        mean_kymograph,
        profile_positions,
        run_size_density_sweep,
        save_size_density_results,
        size_initial_states,
        sweep_geometry,
    )
    from Experiments.micropatterns.texture_settings import DEFAULT_TEXTURE_SETTINGS_FILE, load_cell_settings
    from Experiments.micropatterns.figures.style import (
        FIGURE_CHANNELS,
        GROUP_LABELS,
        STYLE_FILE,
        cell_type_colour,
        legend_options,
        marker_colour,
    )
    # Colony discs with one sector per experiment group, as in the paper figures
    from Experiments.micropatterns.figures.drawing import (
        composite,
        draw_sectors,
        marker_legend_entries,
        rows_to_columns,
        sliced_rgba,
    )
    from Experiments.notebook_jobs import repository_path, write_job_manifest

    load_dotenv(repository_root / ".env", override=False)

    CELL_TYPE_DIRECTORY = repository_root / "Experiments/micropatterns/conf/cell_types"
    DENSITY_LABELS = {"per_mm2": "Cells per mm²", "share": "Covered share"}
    AXIS_LABELS = {
        "edge_um": "Distance from edge (µm)",
        "centre_um": "Distance from centre (µm)",
        "relative": "Distance from centre (fraction of radius)",
    }

    def repository_relative_path(path):
        """``path`` relative to the repository root when it lies inside it."""
        path = Path(path).resolve()
        return str(path.relative_to(repository_root)) if path.is_relative_to(repository_root) else str(path)

    def parse_numbers(text):
        """Numbers in a comma or space separated list."""
        return tuple(float(_value) for _value in text.replace(",", " ").split())

    def edge_to_axis(edge_um, radius_um, axis):
        """Distance from the edge (µm) on another profile axis."""
        return {
            "edge_um": edge_um,
            "centre_um": radius_um - edge_um,
            "relative": (radius_um - edge_um) / radius_um,
        }[axis]


@app.cell(hide_code=True)
def _():
    plt.style.use(["default", STYLE_FILE])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Colony size × cell density

    - Circular colonies of several diameters, filled with whole measured cells at a range of densities (`synthetic_ic.py`, as in the density × KO sweep of `nodal_ko_eval.py`)
    - Every size is drawn at the model's own µm per pixel, on one grid per model
    - Radial profiles over time (kymographs) to look for wavefronts
    - The models only saw ~500 µm colonies: other sizes are extrapolation
    """)
    return


@app.cell(hide_code=True)
def _():
    _backend = jax.default_backend()
    _devices = ", ".join(f"{_device.platform}: {_device.device_kind}" for _device in jax.devices())
    mo.callout(
        f"JAX backend: **{_backend}** · Devices: {_devices}",
        kind="success" if _backend == "gpu" else "warn",
    )
    return


@app.cell(hide_code=True)
def _():
    selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(repository_root / "models")),
        label="Model store",
        full_width=True,
    )
    data_root = mo.ui.text(
        os.environ.get("DATA_PATH_BASE", ""),
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
        # Projection and downsample come from each bundle's config
        _settings = read_model_settings(
            ModelRegistry(Path(store_root.value).expanduser()),
            [_record["model_id"] for _record in _records],
        )
        loaded_records = tuple({**_record, **_settings[_record["model_id"]]} for _record in _records)
        _load_status = mo.md("")
    _load_status
    return (loaded_records,)


@app.cell(hide_code=True)
def _(loaded_records):
    _present = sorted(
        {_record["projection"] for _record in loaded_records},
        key=lambda _name: PROJECTIONS.index(_name) if _name in PROJECTIONS else len(PROJECTIONS),
    )
    selection_projections = mo.ui.multiselect(options=_present, value=_present, label="Projections")
    return (selection_projections,)


@app.cell(hide_code=True)
def _(loaded_records, selection_projections):
    selection_records = tuple(
        _record for _record in loaded_records if _record["projection"] in selection_projections.value
    )
    if not loaded_records:
        _preview = mo.md("")
    else:
        _columns = ["model_id", "alias", "notes", "projection", "downsample", "family", "experiment", "best_loss"]
        _preview = mo.vstack([
            mo.md(f"## Selected models ({len(selection_records)} of {len(loaded_records)})"),
            selection_projections,
            mo.ui.table(pd.DataFrame(selection_records).reindex(columns=_columns), selection=None, page_size=10),
        ])
    _preview
    return (selection_records,)


@app.cell(hide_code=True)
def _():
    _files = sorted(CELL_TYPE_DIRECTORY.glob("*.yaml"))
    _options = {
        f"{_path.name} (tuned on {rules_projection(load_cell_type_rules(_path))})": str(_path)
        for _path in _files
    }
    # Default: the newest file tuned on the projection, else the newest file
    _defaults = {
        _projection: (
            [_o for _o in _options if _o.endswith(f"(tuned on {_projection})")] or list(_options) or [None]
        )[-1]
        for _projection in PROJECTIONS
    }
    cell_type_files = mo.ui.dictionary({
        _projection: mo.ui.dropdown(
            options=_options, value=_defaults[_projection], label=f"Cell type file for {_projection}", full_width=True,
        )
        for _projection in PROJECTIONS
    })
    cell_type_rings = mo.ui.number(1, 40, value=12, step=1, label="Rings for the colony-wide shares")
    mo.vstack([
        mo.md("## Cell types\n\n- Rule files exported by `data_cleaning.py`; one per projection"),
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
        _projection: load_cell_type_rules(_path) for _projection, _path in cell_type_files.value.items()
    }
    _label_sets = {tuple(label_names(_rules)) for _rules in cell_type_rules_by_projection.values()}
    mo.stop(
        len(_label_sets) > 1,
        mo.callout("The cell type files must define the same cell types.", kind="danger"),
    )
    cell_type_labels = next(iter(_label_sets))
    return cell_type_labels, cell_type_rules_by_projection


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Sweep

    - Diameters in µm; densities as real cells per mm² (the same at every size) or covered share
    - Baseline rollouts always; optional NODAL knockouts from the listed hours
    - Each run places its cells once per size: every density and condition of a run shares the rollout seed
    """)
    return


@app.cell(hide_code=True)
def _():
    sd_density_unit = mo.ui.dropdown(
        options={_label: _unit for _unit, _label in DENSITY_LABELS.items()},
        value=DENSITY_LABELS["per_mm2"],
        label="Sweep density as",
    )
    return (sd_density_unit,)


@app.cell(hide_code=True)
def _(sd_density_unit):
    _low, _high, _start, _stop, _step = {
        "per_mm2": (0, 50000, 1000, 8000, 50),
        "share": (0.02, 0.98, 0.2, 0.8, 0.01),
    }[sd_density_unit.value]
    sd_density_start = mo.ui.number(_low, _high, value=_start, step=_step, label="Lowest density")
    sd_density_stop = mo.ui.number(_low, _high, value=_stop, step=_step, label="Highest density")
    return sd_density_start, sd_density_stop


@app.cell(hide_code=True)
def _(selection_records):
    # Created apart from the layout below, so changing the density unit keeps these choices
    _model_options = {model_option_label(_record): _record["model_id"] for _record in selection_records}
    sd_models = mo.ui.multiselect(
        options=_model_options, value=list(_model_options), label="Models", full_width=True,
    )
    sd_diameters = mo.ui.text("200, 300, 400, 500, 600, 800, 1000", label="Colony diameters (µm)", full_width=True)
    sd_density_count = mo.ui.number(1, 50, value=5, step=1, label="Densities")
    sd_ko_hours = mo.ui.text("", label="NODAL KO hours (empty: baseline only)", placeholder="0, 24")
    sd_snapshot = mo.ui.number(1, 96, value=48, step=1, label="End time (h)")
    sd_record_every = mo.ui.number(0.25, 12, value=1, step=0.25, label="Profile every (h)")
    sd_bin_um = mo.ui.number(1, 100, value=10, step=1, label="Profile bin (µm)")
    sd_replicates = mo.ui.text("1", label="Replicates (for example: 1,2,4-6)")
    sd_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    sd_duplicate_runs = mo.ui.number(1, 100, value=2, step=1, label="Duplicate runs")
    sd_texture_file = mo.ui.text(
        value=repository_relative_path(DEFAULT_TEXTURE_SETTINGS_FILE), label="Texture settings file", full_width=True,
    )
    sd_max_batch = mo.ui.number(1, 4096, value=32, step=1, label="Rollouts per GPU batch")
    # References rolled out beside the sweep, with the same conditions and seeds
    sd_include_true_data = mo.ui.switch(value=False, label="True data initial condition")
    sd_include_measured = mo.ui.switch(value=False, label="Synthetic at the measured density")
    return (
        sd_bin_um,
        sd_density_count,
        sd_diameters,
        sd_duplicate_runs,
        sd_include_measured,
        sd_include_true_data,
        sd_ko_hours,
        sd_max_batch,
        sd_models,
        sd_record_every,
        sd_replicates,
        sd_seed,
        sd_snapshot,
        sd_texture_file,
    )


@app.cell(hide_code=True)
def _(
    sd_bin_um,
    sd_density_count,
    sd_density_start,
    sd_density_stop,
    sd_density_unit,
    sd_diameters,
    sd_duplicate_runs,
    sd_include_measured,
    sd_include_true_data,
    sd_ko_hours,
    sd_max_batch,
    sd_models,
    sd_record_every,
    sd_replicates,
    sd_seed,
    sd_snapshot,
    sd_texture_file,
):
    mo.vstack([
        sd_models,
        sd_diameters,
        mo.hstack([sd_density_unit, sd_density_start, sd_density_stop, sd_density_count]),
        mo.hstack([sd_ko_hours, sd_snapshot, sd_record_every, sd_bin_um]),
        mo.hstack([sd_replicates, sd_seed, sd_duplicate_runs]),
        mo.accordion({
            "Cell placement": mo.vstack([
                sd_texture_file,
                mo.md("- The `cells` settings of each model's projection; tune them in `synthetic_textured_ic_explorer.py`"),
            ]),
            "Compute": sd_max_batch,
            "References": mo.vstack([
                mo.md(
                    "- True data: the 0h control colony of each replicate above, in its own outline and size\n"
                    "- Synthetic at the measured density: every size at the measured density "
                    "(mean over the replicates)\n"
                    "- Both use the same conditions and run seeds as the sweep"
                ),
                mo.hstack([sd_include_true_data, sd_include_measured], justify="start"),
            ]),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(
    cell_type_files,
    cell_type_rings,
    data_root,
    sd_bin_um,
    sd_density_count,
    sd_density_start,
    sd_density_stop,
    sd_density_unit,
    sd_diameters,
    sd_duplicate_runs,
    sd_include_measured,
    sd_include_true_data,
    sd_ko_hours,
    sd_max_batch,
    sd_models,
    sd_record_every,
    sd_replicates,
    sd_seed,
    sd_snapshot,
    sd_texture_file,
    selection_records,
    store_root,
):
    # Everything the sweep reads, for a run here or a remote job
    _projections = [_p for _p, _path in cell_type_files.value.items() if _path is not None]
    sd_settings = SizeDensitySettings(
        data_root=data_root.value.strip(),
        store_root=store_root.value.strip(),
        cell_type_files={_p: repository_relative_path(cell_type_files.value[_p]) for _p in _projections},
        cells={
            _p: load_cell_settings(repository_path(sd_texture_file.value.strip()), _p).to_dict()
            for _p in _projections
        },
        diameters_um=parse_numbers(sd_diameters.value),
        rings=int(cell_type_rings.value),
        density_unit=sd_density_unit.value,
        density_start=float(sd_density_start.value),
        density_stop=float(sd_density_stop.value),
        density_count=int(sd_density_count.value),
        ko_hours=parse_numbers(sd_ko_hours.value),
        snapshot_hour=float(sd_snapshot.value),
        record_every_hours=float(sd_record_every.value),
        profile_bin_um=float(sd_bin_um.value),
        replicates=parse_one_based_indices(sd_replicates.value),
        seed=int(sd_seed.value),
        duplicate_runs=int(sd_duplicate_runs.value),
        max_batch=int(sd_max_batch.value),
        include_true_data=bool(sd_include_true_data.value),
        include_measured_density=bool(sd_include_measured.value),
    )
    _by_id = {_record["model_id"]: _record for _record in selection_records}
    sd_records = tuple(_by_id[_model_id] for _model_id in sd_models.value)
    _cells = pd.DataFrame([
        {"projection": _p, **{_k: _v[_k] for _k in ("placement", "pieces", "segmentation", "density_panels")}}
        for _p, _v in sd_settings.cells.items()
    ])
    mo.vstack([
        mo.md(
            f"- {sd_settings.trajectories(len(sd_records))} rollouts: {len(sd_records)} models × "
            f"{len(sd_settings.replicates)} replicates × {sd_settings.duplicate_runs} runs × "
            f"{len(sd_settings.diameters_um)} sizes × {len(sd_settings.densities())} densities × "
            f"{len(sd_settings.conditions())} conditions"
        ),
        mo.ui.table(_cells, selection=None),
    ])
    return sd_records, sd_settings


@app.cell(hide_code=True)
def _():
    # The latest results, from a run or a loaded file
    get_sd_results, set_sd_results = mo.state(None)
    return get_sd_results, set_sd_results


@app.cell(hide_code=True)
def _(sd_records):
    sd_preview_model = mo.ui.dropdown(
        options={model_option_label(_record): _record["model_id"] for _record in sd_records},
        value=model_option_label(sd_records[0]) if sd_records else None,
        label="Model",
    )
    sd_run_preview = mo.ui.run_button(label="Preview initial states")
    mo.vstack([
        mo.md(
            "### Preview\n\n"
            "- Initial states of the first replicate at every size, lowest and highest density (no rollout)\n"
            "- Loads the model's data and cells; no GPU needed"
        ),
        mo.hstack([sd_preview_model, sd_run_preview], justify="start"),
    ])
    return sd_preview_model, sd_run_preview


@app.cell(hide_code=True)
def _(sd_preview_model, sd_records, sd_run_preview, sd_settings):
    if not sd_run_preview.value or sd_preview_model.value is None:
        _preview_view = mo.md("")
    else:
        sd_settings.check()
        _record = {_r["model_id"]: _r for _r in sd_records}[sd_preview_model.value]
        _bundle = ModelRegistry(Path(sd_settings.store_root).expanduser()).get(_record["model_id"])
        _prepared = prepare_model_cells(
            _bundle, sd_settings, _record["projection"], Path(sd_settings.data_root).expanduser()
        )
        _geometry = sweep_geometry(_prepared, sd_settings)
        _names = list(_prepared["names"])
        _channel = _names.index("SOX2") if "SOX2" in _names else 0
        _unit = sd_settings.density_unit
        _densities = sd_settings.densities()[[0, -1]]
        _figure, _axes = plt.subplots(
            2, len(_geometry["masks"]), figsize=(2.0 * len(_geometry["masks"]), 4.3), dpi=150, squeeze=False,
        )
        _rows = []
        _cmap = plt.get_cmap("magma").copy()
        _cmap.set_bad("#d9d9d9")
        for _size, (_diameter, _mask) in enumerate(zip(sd_settings.diameters_um, _geometry["masks"])):
            _states = size_initial_states(_prepared, 0, _mask, _densities, _unit, sd_settings.seed)
            for _row, _density in enumerate(_densities):
                _axis = _axes[_row, _size]
                _axis.imshow(np.where(_mask, _states["initial_states"][_row, _channel], np.nan), cmap=_cmap)
                _axis.set_title(f"{_diameter:g} µm · {int(_states['cells'][_row])} cells", fontsize=7)
                _axis.set_axis_off()
                _rows.append({
                    "diameter_um": _diameter,
                    "requested": format_density(_density, _unit),
                    "colony_mm2": round(_states["mask_mm2"], 4),
                    "cells": int(_states["cells"][_row]),
                    "cells_per_mm2": round(_states["cells"][_row] / _states["mask_mm2"]),
                    "covered_share": round(float(_states["shares"][_row]), 3),
                    "max_density": format_density(_states["max_density"], _unit),
                    "squeeze": round(float(_states["squeeze"]), 2),
                })
        _figure.suptitle(f"{_names[_channel]} at 0h · rows: lowest and highest density", fontsize=9)
        _figure.tight_layout()
        _measured = _prepared["replicates"][0]["cell_counts"]["cells_per_mm2"]
        _preview_view = mo.vstack([
            mo.md(
                f"- Grid {_geometry['shape'][0]}×{_geometry['shape'][1]} px at {_prepared['pixel_um']:.2f} µm/px; "
                f"training colony {_geometry['training_diameter_um']:.0f} µm; "
                f"measured density {_measured:.0f} cells/mm²"
            ),
            _figure,
            mo.ui.table(pd.DataFrame(_rows), selection=None),
            *[mo.callout(_w, kind="warn") for _w in _prepared["warnings"]],
        ])
    _preview_view
    return


@app.cell(hide_code=True)
def _():
    sd_run = mo.ui.run_button(label="Run size × density sweep here")
    sd_run
    return (sd_run,)


@app.cell(hide_code=True)
def _(sd_records, sd_run, sd_settings, set_sd_results):
    if not sd_run.value:
        _run_status = mo.md("- Run here, export as a remote job, or load saved results below")
    else:
        if not sd_records:
            raise ValueError("Choose at least one model.")
        with tqdm(total=sd_settings.trajectories(len(sd_records)), desc="Size × density", unit="trajectory") as _bar:
            _results, _warnings = run_size_density_sweep(sd_settings, sd_records, progress=_bar.update)
        set_sd_results(_results)
        _run_status = mo.vstack([
            mo.callout(f"Completed {len(_results)} runs.", kind="success"),
            *([mo.callout("\n".join(f"- {_w}" for _w in _warnings), kind="warn")] if _warnings else []),
        ])
    _run_status
    return


@app.cell(hide_code=True)
def _():
    sd_job_name = mo.ui.text("size_density", label="Job name", full_width=True)
    sd_job_split = mo.ui.dropdown(
        options={
            "One per model": "model",
            "One per model and replicate": "replicate",
            "One per model, replicate and size": "size",
        },
        value="One per model",
        label="Parts (one pod each)",
    )
    sd_job_output = mo.ui.text(
        "/workspace/writeable/output/size_density",
        label="Results folder on the cluster (the job name is added)",
        full_width=True,
    )
    sd_export_job = mo.ui.run_button(label="Export remote job")
    mo.vstack([
        mo.md(
            "### Run the sweep remotely\n\n"
            "- Writes a job manifest with the settings above and the chosen models\n"
            "- Each part writes `part_<i>.npz`; load the copied-back folder below\n"
            "- The pods need the stored nucleus segmentation under the data root\n"
            "- See `docs/remote_notebook_jobs.md`"
        ),
        mo.hstack([sd_job_name, sd_job_split]),
        sd_job_output,
        sd_export_job,
    ])
    return sd_export_job, sd_job_name, sd_job_output, sd_job_split


@app.cell(hide_code=True)
def _(sd_export_job, sd_job_name, sd_job_output, sd_job_split, sd_records, sd_settings):
    if not sd_export_job.value:
        _job_status = mo.md("")
    elif not sd_records:
        _job_status = mo.callout("Choose at least one model.", kind="warn")
    else:
        sd_settings.check()
        _name = sd_job_name.value.strip() or "size_density"
        _output = f"{sd_job_output.value.strip().rstrip('/')}/{_name}"
        _configs = job_configs(sd_settings, sd_records, _output, split=sd_job_split.value)
        # Next to the training manifests, so a git checkout on the cluster has it
        _manifest = write_job_manifest(
            repository_root / "Experiments/micropatterns/conf/generated/size_density" / _name / "manifest.yaml",
            f"size_density_{_name}",
            _configs,
        )
        _relative = _manifest.relative_to(repository_root)
        _job_status = mo.callout(
            mo.md(
                f"Wrote {_relative} ({len(_configs)} parts). On the cluster, from the repository root:\n\n"
                f"```bash\nbash launch/launch_batch_multi_job.sh "
                f"Experiments/micropatterns/size_density_sweep.py {_relative} <workers> H200\n```\n\n"
                f"Then copy `{_output}/` back to `output/size_density/{_name}/` and load that folder below. "
                f"Test one part locally with `python Experiments/micropatterns/size_density_sweep.py "
                f"--manifest {_relative} --index 0`."
            ),
            kind="success",
        )
    _job_status
    return


@app.cell(hide_code=True)
def _():
    sd_results_path = mo.ui.text(
        "output/size_density/size_density_results.npz",
        label="Results file, or folder of a remote job's parts (relative to the repository root)",
        full_width=True,
    )
    sd_save = mo.ui.run_button(label="Save results")
    sd_load = mo.ui.run_button(label="Load results")
    sd_overwrite = mo.ui.checkbox(value=False, label="Overwrite an existing file")
    mo.vstack([
        mo.md("### Save or load results"),
        sd_results_path,
        mo.hstack([sd_save, sd_overwrite, sd_load], justify="start"),
    ])
    return sd_load, sd_overwrite, sd_results_path, sd_save


@app.cell(hide_code=True)
def _(get_sd_results, sd_load, sd_overwrite, sd_results_path, sd_save, set_sd_results):
    _path = Path(sd_results_path.value.strip()).expanduser()
    if not _path.is_absolute():
        _path = repository_root / _path
    if sd_save.value:
        if get_sd_results() is None:
            _file_status = mo.callout("Run the sweep (or load results) first.", kind="warn")
        elif _path.exists() and not sd_overwrite.value:
            _file_status = mo.callout(f"{_path} exists; choose another name or tick Overwrite.", kind="warn")
        else:
            save_size_density_results(_path, get_sd_results())
            _file_status = mo.callout(f"Saved to {_path}", kind="success")
    elif sd_load.value:
        if not _path.exists():
            _file_status = mo.callout(f"No file or folder at {_path}", kind="danger")
        else:
            set_sd_results(load_size_density_results(_path))
            _file_status = mo.callout(f"Loaded {_path}", kind="success")
    else:
        _file_status = mo.md("")
    _file_status
    return


@app.cell(hide_code=True)
def _(cell_type_rings, cell_type_rules_by_projection, get_sd_results):
    _all = tuple(get_sd_results() or ())
    _usable = tuple(_r for _r in _all if _r["projection"] in cell_type_rules_by_projection)
    # Sweep results; the references (true data, synthetic at the measured density) apart
    sd_results = tuple(_r for _r in _usable if _r["source"] == "sweep") or None
    sd_references = tuple(_r for _r in _usable if _r["source"] != "sweep")

    def _thresholds(_rules):
        return {_marker: float(_rules.thresholds[_marker]) for _marker in _rules.markers}

    # Saved counts only hold for the thresholds and rings they were made with
    _mismatches = sorted({
        f"{_r['label']} ({_r['projection']})" for _r in sd_results or ()
        if _r["rings"] != int(cell_type_rings.value)
        or _r["cell_type_thresholds"] != _thresholds(cell_type_rules_by_projection[_r["projection"]])
    })
    _warnings = []
    if len({_r["density_unit"] for _r in sd_results or ()}) > 1:
        _warnings.append(mo.callout("These results sweep the density in different units; load them separately.", kind="warn"))
    if len({tuple(_r["diameters_um"]) for _r in sd_results or ()}) > 1:
        _warnings.append(mo.callout("These results have different sizes; load them separately.", kind="warn"))
    if _mismatches:
        _warnings.append(mo.callout(
            "Made with other cell type thresholds or rings than chosen above: " + ", ".join(_mismatches),
            kind="warn",
        ))
    # Short model names for figures (full names are too long); the table is the key
    _first_runs = {_r["model_id"]: _r for _r in reversed(sd_results or ())}
    _model_ids = list(dict.fromkeys(_r["model_id"] for _r in sd_results or ()))
    sd_model_names = {_model_id: f"M{_i + 1}" for _i, _model_id in enumerate(_model_ids)}
    _key = pd.DataFrame([
        {
            "Model": sd_model_names[_model_id],
            "Projection": _first_runs[_model_id]["projection"],
            "Full name": _first_runs[_model_id]["label"],
            "Model id": _model_id,
        }
        for _model_id in _model_ids
    ])
    mo.vstack(
        [mo.md("## Results\n\n- Figures name models M1, M2, …; see the key"), mo.ui.table(_key, selection=None), *_warnings]
        if sd_results else []
    )
    return sd_model_names, sd_references, sd_results


@app.cell(hide_code=True)
def _(sd_group_by, sd_model_names):
    def sd_group_of(_result):
        """The group of a result; models on their own go by their short name."""
        if sd_group_by.value == "individual":
            return sd_model_names[_result["model_id"]]
        return _result["groups"][sd_group_by.value]

    return (sd_group_of,)


@app.cell(hide_code=True)
def _(cell_type_labels, sd_references, sd_results):
    _conditions = list(sd_results[0]["conditions"]) if sd_results else ["baseline"]
    _densities = [float(_d) for _d in sd_results[0]["densities"]] if sd_results else []
    _unit = sd_results[0]["density_unit"] if sd_results else "per_mm2"
    sd_group_by = mo.ui.dropdown(
        {
            "Projection": "projection",
            "Projection and resolution": "projection_resolution",
            "Training curriculum": "curriculum",
            "Experiment": "experiment",
            "Notes": "notes",
            "Each model separately": "individual",
            "All models together": "all",
        },
        value="Projection",
        label="Group models by",
    )
    sd_uncertainty = mo.ui.dropdown(
        {"Standard deviation": "std", "Standard error of mean": "sem", "95% percentile interval": "percentile", "None": "none"},
        value="Standard deviation",
        label="Uncertainty",
    )
    sd_condition = mo.ui.dropdown(_conditions, value=_conditions[0], label="Condition")
    sd_display_cell_types = mo.ui.multiselect(
        options=list(cell_type_labels),
        value=[_label for _label in cell_type_labels if _label not in (SEVERAL, OTHER)],
        label="Displayed cell types",
        full_width=True,
    )
    sd_resolve = mo.ui.slider(0.0, 1.0, value=0.0, step=0.05, label="Resolve multiple matches")
    sd_basis = mo.ui.dropdown(
        {"Share of colony pixels": "pattern_counts", "Share of cells": "cell_pattern_counts"},
        value="Share of colony pixels",
        label="Prevalence as",
    )
    sd_fixed_density = mo.ui.dropdown(
        {format_density(_d, _unit): _d for _d in _densities},
        value=format_density(_densities[len(_densities) // 2], _unit) if _densities else None,
        label=f"Fixed density ({DENSITY_LABELS[_unit]}, requested)",
    )
    # References present in the results, shown on request
    _present = [_source for _source in REFERENCE_LABELS if any(_r["source"] == _source for _r in sd_references)]
    sd_show_references = mo.ui.multiselect(
        options={REFERENCE_LABELS[_source]: _source for _source in _present},
        value=[REFERENCE_LABELS[_source] for _source in _present],
        label="References",
    )
    (
        mo.vstack([
            sd_display_cell_types,
            mo.hstack([sd_group_by, sd_uncertainty, sd_condition, sd_fixed_density], justify="start"),
            mo.hstack([sd_basis, sd_resolve], justify="start"),
            mo.hstack([
                sd_show_references,
                mo.md(
                    "- True data: every replicate's colony in its own outline and size (pooled at their mean "
                    "size); synthetic: every size at the measured density\n"
                    "- True data has no cell counts, so it is shown by pixels only"
                ),
            ], justify="start") if _present else mo.md(""),
        ])
        if sd_results else mo.md("")
    )
    return (
        sd_basis,
        sd_condition,
        sd_display_cell_types,
        sd_fixed_density,
        sd_group_by,
        sd_resolve,
        sd_show_references,
        sd_uncertainty,
    )


@app.cell(hide_code=True)
def _(
    cell_type_labels,
    cell_type_rules_by_projection,
    sd_basis,
    sd_group_of,
    sd_model_names,
    sd_references,
    sd_resolve,
    sd_results,
    sd_uncertainty,
):
    # Colony-wide shares at the end time, from the stored ring counts
    def _share_rows(_results):
        _rows = []
        for _result in _results:
            _counts = _result[sd_basis.value]
            if _counts is None:
                continue
            _rules = cell_type_rules_by_projection[_result["projection"]]
            for _s, _diameter in enumerate(_result["diameters_um"]):
                for _d, _density in enumerate(_result["densities"]):
                    for _k, _condition in enumerate(_result["conditions"]):
                        _shares = colony_label_shares(
                            model_stain_counts(_counts[_s, _d, _k], _rules), _rules, resolve_several=sd_resolve.value,
                        )
                        for _cell_type, _share in zip(cell_type_labels, _shares):
                            _rows.append({
                                "model_id": _result["model_id"],
                                "model": sd_model_names.get(_result["model_id"], _result["model_id"]),
                                "model_label": _result["label"],
                                "projection": _result["projection"],
                                "replicate": _result["replicate"],
                                "seed": _result["seed"],
                                # sweep, true_data or measured_density
                                "source": _result["source"],
                                "group": sd_group_of(_result),
                                "diameter_um": float(_diameter),
                                "density": float(_density),
                                "cells_per_mm2": float(_result["cells_per_mm2"][_s, _d]),
                                "covered_share": float(_result["covered_shares"][_s, _d]),
                                "measured_density": float(_result["measured_density"]),
                                "training_diameter_um": float(_result["training_diameter_um"]),
                                "condition": _condition,
                                "cell_type": _cell_type,
                                "share": float(_share),
                            })
        return pd.DataFrame(_rows)

    def _summarise(_frame, _keys):
        _summary = []
        if _frame.empty:
            return pd.DataFrame()
        for _key_values, _values in _frame.groupby(_keys, sort=False):
            _mean, _lower, _upper = uncertainty_band(_values["share"], sd_uncertainty.value)
            _summary.append({
                **dict(zip(_keys, _key_values)),
                # References: one density each (it can differ a little between models)
                **({} if "density" in _keys else {"density": float(_values["density"].mean())}),
                "cells_per_mm2": float(_values["cells_per_mm2"].mean()),
                "covered_share": float(_values["covered_share"].mean()),
                "mean_share": _mean, "lower": _lower, "upper": _upper, "samples": len(_values),
            })
        return pd.DataFrame(_summary)

    sd_shares = _share_rows(sd_results or ())
    sd_summary = _summarise(sd_shares, ["group", "cell_type", "diameter_um", "density", "condition"])
    # References: true data (one size), and every size at the measured density
    sd_reference_shares = _share_rows(sd_references)
    if not sd_reference_shares.empty:
        # The replicates' true colonies differ a little in size: pooled at their mean size
        _true = sd_reference_shares["source"] == "true_data"
        sd_reference_shares.loc[_true, "diameter_um"] = sd_reference_shares.loc[_true, "diameter_um"].mean()
    sd_reference_summary = _summarise(
        sd_reference_shares, ["group", "source", "cell_type", "diameter_um", "condition"],
    )
    return sd_reference_shares, sd_reference_summary, sd_shares, sd_summary


@app.cell(hide_code=True)
def _(sd_summary):
    _groups = list(dict.fromkeys(sd_summary["group"])) if not sd_summary.empty else []
    sd_group = mo.ui.dropdown(_groups, value=_groups[0] if _groups else None, label="Group shown")
    sd_heatmap_scale = mo.ui.dropdown(
        {"0 to 1 for every cell type": "shared", "Each cell type's own range": "own"},
        value="0 to 1 for every cell type",
        label="Colour scale",
    )
    mo.hstack([sd_group, sd_heatmap_scale], justify="start") if _groups else mo.md("")
    return sd_group, sd_heatmap_scale


@app.cell(hide_code=True)
def _(
    sd_condition,
    sd_display_cell_types,
    sd_group,
    sd_heatmap_scale,
    sd_reference_summary,
    sd_results,
    sd_shares,
    sd_show_references,
    sd_summary,
):
    if sd_summary.empty or sd_group.value is None or not sd_display_cell_types.value:
        _heatmap_view = mo.md("")
    else:
        _shown = sd_summary[(sd_summary["group"] == sd_group.value) & (sd_summary["condition"] == sd_condition.value)]
        _unit = sd_results[0]["density_unit"]
        _x_column = "cells_per_mm2" if _unit == "per_mm2" else "covered_share"
        _cell_types = [_c for _c in sd_display_cell_types.value if _c in set(_shown["cell_type"])]
        # References of this group and condition, as a strip right of each map
        _references = (
            sd_reference_summary[
                (sd_reference_summary["group"] == sd_group.value)
                & (sd_reference_summary["condition"] == sd_condition.value)
                & sd_reference_summary["source"].isin(sd_show_references.value)
            ]
            if not sd_reference_summary.empty else pd.DataFrame()
        )
        _sources = [_s for _s in REFERENCE_LABELS if not _references.empty and _s in set(_references["source"])]
        if _sources:
            _figure, _all_axes = plt.subplots(
                1, 2 * len(_cell_types), figsize=((4.0 + 0.45 * len(_sources)) * len(_cell_types), 3.8), dpi=150,
                squeeze=False, width_ratios=[3.2, 0.35 * len(_sources)] * len(_cell_types), layout="constrained",
            )
            _axes, _strips = _all_axes[0, 0::2], _all_axes[0, 1::2]
        else:
            _figure, _axes = plt.subplots(1, len(_cell_types), figsize=(4.0 * len(_cell_types), 3.6), dpi=150, squeeze=False)
            _axes, _strips = _axes[0], None
        # Columns at the density each requested density used (mean over sizes and runs)
        _x = _shown.groupby("density")[_x_column].mean().sort_index().to_numpy()
        _diameters = np.sort(_shown["diameter_um"].unique())
        _training = float(sd_shares["training_diameter_um"].mean())
        _measured = float(sd_shares["measured_density"].mean())
        for _index, (_axis, _cell_type) in enumerate(zip(_axes, _cell_types)):
            _grid = _shown[_shown["cell_type"] == _cell_type].pivot(
                index="diameter_um", columns="density", values="mean_share"
            ).sort_index().sort_index(axis=1)
            _image = _axis.pcolormesh(
                cell_edges(_x), cell_edges(_diameters), _grid.to_numpy(), shading="flat", cmap="viridis",
                **({"vmin": 0.0, "vmax": 1.0} if sd_heatmap_scale.value == "shared" else {}),
            )
            _axis.axhline(_training, color="white", linewidth=0.8, linestyle="--")
            _axis.axvline(_measured, color="white", linewidth=0.8, linestyle=":")
            _axis.set(xlabel=f"{DENSITY_LABELS[_unit]} (used)", ylabel="Colony diameter (µm)", title=_cell_type)
            if _strips is not None:
                # Each reference's share at each size; the true colony (one size) in the row nearest its size
                _strip_rows = []
                for _source in _sources:
                    _values = np.full(len(_diameters), np.nan)
                    _of = _references[(_references["source"] == _source) & (_references["cell_type"] == _cell_type)]
                    for _diameter, _value in zip(_of["diameter_um"], _of["mean_share"]):
                        _values[int(np.argmin(np.abs(_diameters - _diameter)))] = _value
                    _strip_rows.append(_values)
                draw_reference_strip(_strips[_index], _diameters, _strip_rows, _sources, _image, vertical=True)
                _strips[_index].set_ylim(_axis.get_ylim())
                _strips[_index].set_yticks([])
                _figure.colorbar(_image, ax=[_axis, _strips[_index]], label="Share")
            else:
                _figure.colorbar(_image, ax=_axis, label="Share")
        _figure.suptitle(
            f"{sd_group.value} · {sd_condition.value} · mean share at {sd_results[0]['snapshot_hour']:g}h "
            "(dashed: training size, dotted: measured density)"
            + ("\nstrip: references (true data in the row nearest its size)" if _sources else "")
        )
        if _strips is None:
            _figure.tight_layout()
        _heatmap_view = _figure
    _heatmap_view
    return


@app.cell(hide_code=True)
def _(cell_type_labels):
    sd_slice_cell_type = mo.ui.dropdown(list(cell_type_labels), value=cell_type_labels[0], label="Cell type (right)")
    sd_slice_cell_type
    return (sd_slice_cell_type,)


@app.cell(hide_code=True)
def _(
    sd_condition,
    sd_display_cell_types,
    sd_fixed_density,
    sd_group,
    sd_reference_summary,
    sd_results,
    sd_shares,
    sd_show_references,
    sd_slice_cell_type,
    sd_summary,
):
    if sd_summary.empty or sd_group.value is None or sd_fixed_density.value is None:
        _slice_view = mo.md("")
    else:
        _shown = sd_summary[(sd_summary["group"] == sd_group.value) & (sd_summary["condition"] == sd_condition.value)]
        _unit = sd_results[0]["density_unit"]
        _x_column = "cells_per_mm2" if _unit == "per_mm2" else "covered_share"
        _figure, (_left, _right) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
        _at_density = _shown[np.isclose(_shown["density"], sd_fixed_density.value)]
        for _cell_type in sd_display_cell_types.value:
            _line = _at_density[_at_density["cell_type"] == _cell_type].sort_values("diameter_um")
            _colour = cell_type_colour(_cell_type)
            _left.plot(_line["diameter_um"], _line["mean_share"], marker="o", color=_colour, label=_cell_type)
            _left.fill_between(_line["diameter_um"], _line["lower"], _line["upper"], color=_colour, alpha=0.2, linewidth=0)
        # References: every size at the measured density (lines), the true colony (a point at its size)
        _references = (
            sd_reference_summary[
                (sd_reference_summary["group"] == sd_group.value)
                & (sd_reference_summary["condition"] == sd_condition.value)
                & sd_reference_summary["source"].isin(sd_show_references.value)
            ]
            if not sd_reference_summary.empty else pd.DataFrame()
        )
        _sources = [_s for _s in REFERENCE_LABELS if not _references.empty and _s in set(_references["source"])]
        for _source in _sources:
            for _cell_type in sd_display_cell_types.value:
                _line = _references[
                    (_references["source"] == _source) & (_references["cell_type"] == _cell_type)
                ].sort_values("diameter_um")
                _left.plot(
                    _line["diameter_um"].to_numpy(), _line["mean_share"].to_numpy(), color=cell_type_colour(_cell_type),
                    linestyle=REFERENCE_LINESTYLES[_source], marker=REFERENCE_MARKERS[_source],
                    markersize=8 if _source == "true_data" else 4,
                )
        _left.set(
            xlabel="Colony diameter (µm)", ylabel="Share",
            title=f"At {format_density(sd_fixed_density.value, _unit)} {DENSITY_LABELS[_unit].lower()}",
        )
        _handles, _labels = _left.get_legend_handles_labels()
        _reference_handles = reference_legend_handles(_sources)
        _left.legend(
            _handles + _reference_handles, _labels + [_h.get_label() for _h in _reference_handles], fontsize=7,
        )
        _of_type = _shown[_shown["cell_type"] == sd_slice_cell_type.value]
        _diameters = np.sort(_of_type["diameter_um"].unique())
        _colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(_diameters)))
        for _diameter, _colour in zip(_diameters, _colours):
            _line = _of_type[_of_type["diameter_um"] == _diameter].sort_values("density")
            _right.plot(_line[_x_column], _line["mean_share"], marker="o", color=_colour, label=f"{_diameter:g} µm")
            _right.fill_between(_line[_x_column], _line["lower"], _line["upper"], color=_colour, alpha=0.15, linewidth=0)
        # References at their own density: each size at the measured density, the true colony
        for _source in _sources:
            _of = _references[
                (_references["source"] == _source) & (_references["cell_type"] == sd_slice_cell_type.value)
            ]
            for _diameter, _x, _value in zip(_of["diameter_um"], _of[_x_column], _of["mean_share"]):
                _nearest = int(np.argmin(np.abs(_diameters - _diameter))) if len(_diameters) else 0
                _right.plot(
                    [_x], [_value], linestyle="none", marker=REFERENCE_MARKERS[_source],
                    markersize=9 if _source == "true_data" else 6,
                    color="black" if _source == "true_data" else _colours[_nearest],
                )
        _right.axvline(float(sd_shares["measured_density"].mean()), color="grey", linewidth=0.8, linestyle=":")
        _right.set(xlabel=f"{DENSITY_LABELS[_unit]} (used)", ylabel="Share", title=sd_slice_cell_type.value)
        _right.legend(fontsize=7)
        _left.axvline(float(sd_shares["training_diameter_um"].mean()), color="grey", linewidth=0.8, linestyle="--")
        _figure.suptitle(f"{sd_group.value} · {sd_condition.value}")
        _figure.tight_layout()
        _slice_view = _figure
    _slice_view
    return


@app.cell(hide_code=True)
def _(sd_model_names, sd_results):
    # Images are kept for the first run of each job part only
    _with_images = [
        (f"{sd_model_names[_r['model_id']]} · replicate {_r['replicate']} · run {_r['run_index']}", _i)
        for _i, _r in enumerate(sd_results or ()) if _r["final_states"] is not None
    ]
    sd_image_result = mo.ui.dropdown(
        dict(_with_images), value=_with_images[0][0] if _with_images else None, label="Run",
    )
    sd_image_stage = mo.ui.dropdown(["End", "Initial (0h)"], value="End", label="State")
    _first = sd_results[0] if sd_results else None
    _names = set(_first["state_channel_names"]) if _first else set()
    # Experiment groups whose figure markers the model has, one sector each
    _sector_groups = {
        GROUP_LABELS[_g]: _g for _g, _markers in FIGURE_CHANNELS.items() if _names & set(_markers)
    }
    sd_image_sectors = mo.ui.multiselect(_sector_groups, value=list(_sector_groups), label="Sectors")
    _diameters = [float(_d) for _d in _first["diameters_um"]] if _first else []
    sd_image_sizes = mo.ui.multiselect(
        {f"{_d:g} µm": _d for _d in _diameters}, value=[f"{_d:g} µm" for _d in _diameters], label="Sizes",
    )
    _unit = _first["density_unit"] if _first else "per_mm2"
    _densities = [float(_d) for _d in _first["densities"]] if _first else []
    sd_image_densities = mo.ui.multiselect(
        {format_density(_d, _unit): _d for _d in _densities},
        value=[format_density(_d, _unit) for _d in _densities],
        label=f"Densities ({DENSITY_LABELS[_unit]}, requested)",
    )
    sd_image_transpose = mo.ui.switch(value=False, label="Sizes as columns")
    sd_image_fill = mo.ui.switch(value=False, label="Same display size")
    sd_image_scale_bar = mo.ui.switch(value=True, label="Scale bars")
    (
        mo.vstack([
            mo.md(
                "### Colonies\n\n"
                "- One sector per experiment group, showing its figure markers\n"
                "- Drawn to scale, or each colony enlarged to fill its panel (same display size; use the scale bars)\n"
                "- Rows are sizes and columns densities, or the other way round"
            ),
            mo.hstack([sd_image_result, sd_image_stage, sd_image_sectors], justify="start"),
            mo.hstack([sd_image_sizes, sd_image_densities], justify="start"),
            mo.hstack([sd_image_transpose, sd_image_fill, sd_image_scale_bar], justify="start"),
        ])
        if _with_images else mo.md("")
    )
    return (
        sd_image_densities,
        sd_image_fill,
        sd_image_result,
        sd_image_scale_bar,
        sd_image_sectors,
        sd_image_sizes,
        sd_image_stage,
        sd_image_transpose,
    )


@app.cell(hide_code=True)
def _(
    sd_condition,
    sd_image_densities,
    sd_image_fill,
    sd_image_result,
    sd_image_scale_bar,
    sd_image_sectors,
    sd_image_sizes,
    sd_image_stage,
    sd_image_transpose,
    sd_model_names,
    sd_references,
    sd_results,
    sd_show_references,
):
    def _scale_bar(_axis, _half_px, _pixel_um):
        """Black scale bar in the lower left of a panel spanning ±``_half_px`` pixels,
        about a quarter of its width, rounded to a readable length."""
        _target_um = 0.25 * 2.0 * _half_px * _pixel_um
        _lengths = [_l for _l in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if _l <= _target_um]
        _length_um = _lengths[-1] if _lengths else _target_um
        _length_px = _length_um / _pixel_um
        _x, _y = -0.92 * _half_px, -0.92 * _half_px
        _axis.plot([_x, _x + _length_px], [_y, _y], color="black", linewidth=1.5, solid_capstyle="butt")
        _axis.text(_x + _length_px / 2, _y + 0.04 * _half_px, f"{_length_um:g} µm", ha="center", va="bottom", fontsize=6)

    def _sector_disc(_states, _names, _groups, _mask):
        """RGBA of ``_states`` [C, H, W] on the whole grid (so sizes stay to scale), one sector per group."""
        _rgbas = []
        for _group in _groups:
            _colours = [marker_colour(_n) if _n in FIGURE_CHANNELS[_group] else None for _n in _names]
            _rgb = composite(np.clip(_states, 0.0, 1.0), _colours)
            _rgbas.append(np.concatenate([_rgb, _mask[..., None].astype(float)], axis=-1))
        return sliced_rgba(_rgbas)

    _groups = [_g for _g in FIGURE_CHANNELS if _g in sd_image_sectors.value]
    _sizes_shown = [float(_d) for _d in sd_image_sizes.value]
    _densities_shown = [float(_d) for _d in sd_image_densities.value]
    if sd_image_result.value is None or not (_groups and _sizes_shown and _densities_shown):
        _image_view = mo.md("- Choose at least one sector, size and density") if sd_image_result.value is not None else mo.md("")
    else:
        _result = sd_results[sd_image_result.value]
        _names = list(_result["state_channel_names"])
        _k = list(_result["conditions"]).index(sd_condition.value)
        _diameters = np.asarray(_result["diameters_um"], dtype=float)
        _densities = np.asarray(_result["densities"], dtype=float)
        # Indices of the chosen sizes and densities, in sweep order
        _size_index = [_s for _s, _d in enumerate(_diameters) if np.any(np.isclose(_d, _sizes_shown))]
        _density_index = [_i for _i, _d in enumerate(_densities) if np.any(np.isclose(_d, _densities_shown))]
        _unit = _result["density_unit"]
        # References of this model with images, shown as extra densities: every size at the
        # measured density, and the true colony in the row of the nearest shown size
        def _reference_image(_source):
            """The reference with images of this model, of the shown run's replicate where there is one."""
            _candidates = [
                _r for _r in sd_references
                if _r["source"] == _source and _r["model_id"] == _result["model_id"] and _r["final_states"] is not None
            ]
            return next(
                (_r for _r in _candidates if _r["replicate"] == _result["replicate"]),
                _candidates[0] if _candidates else None,
            )

        _reference_of = {_source: _reference_image(_source) for _source in sd_show_references.value}
        _reference_of = {_source: _r for _source, _r in _reference_of.items() if _r is not None}
        _true_size = (
            min(_size_index, key=lambda _s: abs(_diameters[_s] - float(_reference_of["true_data"]["diameters_um"][0])))
            if "true_data" in _reference_of else None
        )
        _density_index = list(_density_index) + [_source for _source in REFERENCE_LABELS if _source in _reference_of]
        _rows, _columns = (_density_index, _size_index) if sd_image_transpose.value else (_size_index, _density_index)
        _figure, _axes = plt.subplots(
            len(_rows), len(_columns), figsize=(1.8 * len(_columns) + 1.6, 1.8 * len(_rows) + 0.4),
            dpi=150, squeeze=False,
        )
        _height, _width = np.shape(_result["colony_masks"])[-2:]
        _extent = (-_width / 2, _width / 2, -_height / 2, _height / 2)

        def _size_label(_s):
            return f"{_diameters[_s]:g} µm"

        def _density_label(_d):
            if isinstance(_d, str):
                return REFERENCE_LABELS[_d].replace(" at ", "\nat ")
            return f"{format_density(_densities[_d], _unit)}"

        def _panel(_s, _d):
            """(state, mask) of size ``_s`` at density ``_d`` (an index, or a reference); None if there is none."""
            _source = _d if isinstance(_d, str) else None
            _of = _result if _source is None else _reference_of[_source]
            if _source == "true_data":
                if _s != _true_size:
                    return None
                _s_of, _d_of = 0, 0
            else:
                _s_of, _d_of = _s, (0 if _source else _d)
            _states = (
                _of["final_states"][_s_of, _d_of, _k] if sd_image_stage.value == "End"
                else _of["initial_states"][_s_of, _d_of, : len(_names)]
            )
            return _states, np.asarray(_of["colony_masks"][_s_of], dtype=bool)

        for _row, _a in enumerate(_rows):
            for _column, _b in enumerate(_columns):
                _s, _d = (_b, _a) if sd_image_transpose.value else (_a, _b)
                _axis = _axes[_row, _column]
                _axis.set_xticks([])
                _axis.set_yticks([])
                for _spine in _axis.spines.values():
                    _spine.set_visible(False)
                if _column == 0:
                    _axis.set_ylabel(_density_label(_a) if sd_image_transpose.value else _size_label(_a), fontsize=7)
                if _row == 0:
                    _axis.set_title(_size_label(_b) if sd_image_transpose.value else _density_label(_b), fontsize=7, loc="center")
                _found = _panel(_s, _d)
                if _found is None:
                    continue
                _states, _mask = _found
                _axis.imshow(_sector_disc(np.asarray(_states, dtype=np.float32), _names, _groups, _mask),
                             extent=_extent, interpolation="nearest")
                _radius = float(np.sqrt(_mask.sum() / np.pi))
                draw_sectors(_axis, (0.0, 0.0), _radius, [False] * len(_groups))
                # Same display size: zoom to the colony (a small margin for the scale bar); else the whole grid
                _half = 1.1 * _radius if sd_image_fill.value else _width / 2
                _axis.set_xlim(-_half, _half)
                _axis.set_ylim(-_half, _half)
                if sd_image_scale_bar.value:
                    _scale_bar(_axis, _half, float(_result["pixel_um"]))
        # Legend: each group's sector and marker colours
        _legend_names = [f"{_g}/{_m}" for _g in _groups for _m in FIGURE_CHANNELS[_g]]
        _handles, _labels, _ncol = rows_to_columns(marker_legend_entries(_legend_names, _groups, sectors=_groups))
        _figure.legend(_handles, _labels, ncol=_ncol, loc="center left", bbox_to_anchor=(1.0, 0.5), **legend_options())
        _figure.suptitle(
            f"{sd_model_names[_result['model_id']]} · replicate {_result['replicate']} · "
            f"{sd_condition.value} · {sd_image_stage.value} · "
            + ("rows: density, columns: size" if sd_image_transpose.value else "rows: size, columns: density")
            + f" ({DENSITY_LABELS[_unit].lower()}, requested)",
            fontsize=9,
        )
        _figure.tight_layout()
        _image_view = _figure
    _image_view
    return


@app.cell(hide_code=True)
def _(cell_type_labels, sd_results):
    _channels = list(sd_results[0]["state_channel_names"]) if sd_results else []
    # Any channel mean or cell type share, for the kymographs and the fronts
    sd_quantity_options = {
        **{f"{_c} (channel mean)": _c for _c in _channels},
        **{f"{_l} (share)": _l for _l in cell_type_labels},
    }
    sd_quantity = mo.ui.dropdown(
        options=sd_quantity_options,
        value=next(iter(sd_quantity_options), None),
        label="Kymograph quantity",
    )
    sd_axis = mo.ui.dropdown(
        {_label: _axis for _axis, _label in AXIS_LABELS.items()},
        value=AXIS_LABELS["edge_um"],
        label="Position",
    )
    sd_level_fraction = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="Front level (fraction of range)")
    (
        mo.vstack([
            mo.md(
                "## Kymographs and wavefronts\n\n"
                "- Radially averaged quantity against time; one model, its replicates and runs pooled\n"
                "- Front: where the profile first crosses the level, reading in from the edge\n"
                "- Level: the chosen fraction of each quantity's range over every shown model, size, density and time"
            ),
            mo.hstack([sd_quantity, sd_axis, sd_level_fraction], justify="start"),
        ])
        if sd_results else mo.md("")
    )
    return sd_axis, sd_level_fraction, sd_quantity, sd_quantity_options


@app.cell(hide_code=True)
def _(cell_type_rules_by_projection, sd_group_of, sd_model_names, sd_references, sd_resolve, sd_results):
    def sd_kymograph_entries(_quantity, _source="sweep"):
        """Mean kymograph ``[sizes, densities, conditions, times, bins]`` of
        ``_quantity`` for each model, of the sweep or of one reference."""
        _results = (sd_results or ()) if _source == "sweep" else [_r for _r in sd_references if _r["source"] == _source]
        _entries = {}
        for _model_id in dict.fromkeys(_r["model_id"] for _r in _results):
            _runs = [_r for _r in _results if _r["model_id"] == _model_id]
            _entries[_model_id] = {
                "values": mean_kymograph(
                    _runs, _quantity, cell_type_rules_by_projection[_runs[0]["projection"]], sd_resolve.value,
                ),
                "result": _runs[0],
                "name": sd_model_names.get(_model_id, _model_id),
                "group": sd_group_of(_runs[0]),
            }
        return _entries

    def sd_front_level(_entries, _fraction):
        """``(level, low, high)``: the level is ``_fraction`` of the range of the
        kymographs ``_entries`` over every model, size, density and time."""
        _values = np.concatenate([_k["values"].ravel() for _k in _entries.values()]) if _entries else np.zeros(1)
        _low, _high = float(np.nanmin(_values)), float(np.nanmax(_values))
        return _low + float(_fraction) * (_high - _low), _low, _high

    return sd_front_level, sd_kymograph_entries


@app.cell(hide_code=True)
def _(sd_front_level, sd_kymograph_entries, sd_level_fraction, sd_quantity, sd_show_references):
    sd_kymographs = sd_kymograph_entries(sd_quantity.value) if sd_quantity.value is not None else {}
    # References shown: {source: {model id: entry}}; their fronts use the sweep's level
    sd_reference_kymographs = {
        _source: sd_kymograph_entries(sd_quantity.value, _source)
        for _source in sd_show_references.value if sd_quantity.value is not None
    }
    sd_level, _low, _high = sd_front_level(sd_kymographs, sd_level_fraction.value)
    mo.md(f"- Front level {sd_level:.3g} (range {_low:.3g} to {_high:.3g})") if sd_kymographs else mo.md("")
    return sd_kymographs, sd_level, sd_reference_kymographs


@app.cell(hide_code=True)
def _(sd_kymographs):
    _first = next(iter(sd_kymographs.values()), None)
    sd_kymo_model = mo.ui.dropdown(
        {_k["name"]: _id for _id, _k in sd_kymographs.items()},
        value=_first["name"] if _first else None,
        label="Model",
    )
    sd_kymo_layout = mo.ui.dropdown(
        {"Every size at one density": "sizes", "Every density at one size": "densities"},
        value="Every size at one density",
        label="Panels",
    )
    _diameters = [float(_d) for _d in _first["result"]["diameters_um"]] if _first else []
    sd_kymo_size = mo.ui.dropdown(
        {f"{_d:g} µm": _d for _d in _diameters},
        value=f"{_diameters[len(_diameters) // 2]:g} µm" if _diameters else None,
        label="Fixed size",
    )
    sd_kymo_shared = mo.ui.switch(value=True, label="One colour scale")
    mo.hstack([sd_kymo_model, sd_kymo_layout, sd_kymo_size, sd_kymo_shared], justify="start") if _first else mo.md("")
    return sd_kymo_layout, sd_kymo_model, sd_kymo_shared, sd_kymo_size


@app.cell(hide_code=True)
def _(
    sd_axis,
    sd_condition,
    sd_fixed_density,
    sd_kymo_layout,
    sd_kymo_model,
    sd_kymo_shared,
    sd_kymo_size,
    sd_kymographs,
    sd_level,
    sd_quantity,
    sd_reference_kymographs,
):
    if sd_kymo_model.value is None or sd_fixed_density.value is None:
        _kymograph_view = mo.md("")
    else:
        _entry = sd_kymographs[sd_kymo_model.value]
        _result, _values = _entry["result"], _entry["values"]
        _k = list(_result["conditions"]).index(sd_condition.value)
        _diameters = list(np.asarray(_result["diameters_um"], dtype=float))
        _densities = list(np.asarray(_result["densities"], dtype=float))
        _d_fixed = int(np.argmin(np.abs(np.asarray(_densities) - sd_fixed_density.value)))
        _s_fixed = int(np.argmin(np.abs(np.asarray(_diameters) - sd_kymo_size.value)))
        # (entry, size index, density index, title) of each panel
        _panels = [
            (_entry, _s, _d, f"{_diameters[_s]:g} µm · {_densities[_d]:g}")
            for _s, _d in (
                [(_s, _d_fixed) for _s in range(len(_diameters))] if sd_kymo_layout.value == "sizes"
                else [(_s_fixed, _d) for _d in range(len(_densities))]
            )
        ]
        # References of this model: the fixed size at the measured density, and the true colony
        for _source, _entries in sd_reference_kymographs.items():
            _reference = _entries.get(sd_kymo_model.value)
            if _reference is None:
                continue
            _ref_diameters = np.asarray(_reference["result"]["diameters_um"], dtype=float)
            _ref_s = int(np.argmin(np.abs(_ref_diameters - sd_kymo_size.value))) if _source != "true_data" else 0
            _panels.append((
                _reference, _ref_s, 0,
                f"{REFERENCE_LABELS[_source].replace(' at ', chr(10) + 'at ')} · {_ref_diameters[_ref_s]:.0f} µm",
            ))
        _hours = np.asarray(_result["profile_hours"], dtype=float)
        _shown = np.concatenate([_p[0]["values"][_p[1], _p[2], _k].ravel() for _p in _panels])
        _scale = {"vmin": float(np.nanmin(_shown)), "vmax": float(np.nanmax(_shown))} if sd_kymo_shared.value else {}
        _figure, _axes = plt.subplots(
            1, len(_panels), figsize=(2.4 * len(_panels) + 1.0, 3.6), dpi=150, sharey=True, squeeze=False,
        )
        for _axis, (_panel_entry, _s, _d, _title) in zip(_axes[0], _panels):
            _z = _panel_entry["values"][_s, _d, _k]
            _keep = np.isfinite(_z).any(axis=0)
            _edge = profile_positions(_panel_entry["result"], _s, "edge_um")[_keep]
            _radius = float(_panel_entry["result"]["diameters_um"][_s]) / 2.0
            _x = edge_to_axis(_edge, _radius, sd_axis.value)
            _order = np.argsort(_x)
            _x, _z = _x[_order], _z[:, _keep][:, _order]
            _mesh = _axis.pcolormesh(
                cell_edges(_x), cell_edges(_hours), np.ma.masked_invalid(_z), shading="flat", cmap="magma", **_scale,
            )
            if np.nanmin(_z) < sd_level < np.nanmax(_z):
                _axis.contour(_x, _hours, np.ma.masked_invalid(_z), levels=[sd_level], colors="white", linewidths=0.7)
            # Fronts are found reading in from the edge, then drawn on the chosen axis
            _fronts = front_positions(_z, _edge[_order], sd_level)
            _axis.plot(edge_to_axis(_fronts, _radius, sd_axis.value), _hours, color="#56B4E9", linewidth=1.0, linestyle="--")
            _axis.set_title(_title, fontsize=8)
            _axis.set_xlabel(AXIS_LABELS[sd_axis.value], fontsize=7)
        _axes[0, 0].set_ylabel("Time (h)")
        _figure.colorbar(_mesh, ax=_axes[0].tolist(), label=sd_quantity.value)
        _figure.suptitle(
            f"{_entry['name']} · {sd_condition.value} · white: level {sd_level:.3g}, blue dashed: front from the edge",
            fontsize=9,
        )
        _kymograph_view = _figure
    _kymograph_view
    return


@app.cell(hide_code=True)
def _(
    sd_axis,
    sd_condition,
    sd_fixed_density,
    sd_kymo_model,
    sd_kymographs,
    sd_quantity,
    sd_reference_kymographs,
):
    # Radial profiles at the end time, one line per size
    if sd_kymo_model.value is None or sd_fixed_density.value is None:
        _profile_view = mo.md("")
    else:
        _entry = sd_kymographs[sd_kymo_model.value]
        _result = _entry["result"]
        _k = list(_result["conditions"]).index(sd_condition.value)
        _d = int(np.argmin(np.abs(np.asarray(_result["densities"], dtype=float) - sd_fixed_density.value)))
        _diameters = np.asarray(_result["diameters_um"], dtype=float)
        _figure, _axis = plt.subplots(figsize=(5.5, 3.4), dpi=150)
        _colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(_diameters)))
        for _s, (_diameter, _colour) in enumerate(zip(_diameters, _colours)):
            _z = _entry["values"][_s, _d, _k, -1]
            _x = profile_positions(_result, _s, sd_axis.value)
            _keep = np.isfinite(_z)
            _order = np.argsort(_x[_keep])
            _axis.plot(_x[_keep][_order], _z[_keep][_order], color=_colour, label=f"{_diameter:g} µm")
        # References: every size at the measured density (in its size's colour), the true colony (black)
        for _source, _entries in sd_reference_kymographs.items():
            _reference = _entries.get(sd_kymo_model.value)
            if _reference is None:
                continue
            for _s, _diameter in enumerate(np.asarray(_reference["result"]["diameters_um"], dtype=float)):
                _z = _reference["values"][_s, 0, _k, -1]
                _x = profile_positions(_reference["result"], _s, sd_axis.value)
                _keep = np.isfinite(_z)
                _order = np.argsort(_x[_keep])
                _nearest = int(np.argmin(np.abs(_diameters - _diameter)))
                _axis.plot(
                    _x[_keep][_order], _z[_keep][_order], linestyle=REFERENCE_LINESTYLES[_source],
                    color="black" if _source == "true_data" else _colours[_nearest],
                )
        _handles, _labels = _axis.get_legend_handles_labels()
        _reference_handles = reference_legend_handles(
            [_source for _source in REFERENCE_LABELS if sd_reference_kymographs.get(_source, {}).get(sd_kymo_model.value)]
        )
        _axis.set(
            xlabel=AXIS_LABELS[sd_axis.value], ylabel=sd_quantity.value,
            title=f"{_entry['name']} · {sd_condition.value} · {_result['snapshot_hour']:g}h · density {_result['densities'][_d]:g}",
        )
        _axis.legend(_handles + _reference_handles, _labels + [_h.get_label() for _h in _reference_handles], fontsize=7)
        _figure.tight_layout()
        _profile_view = _figure
    _profile_view
    return


@app.cell(hide_code=True)
def _(sd_quantity_options, sd_results):
    _end = float(sd_results[0]["snapshot_hour"]) if sd_results else 48.0
    sd_window = mo.ui.range_slider(0.0, _end, value=[0.0, _end], step=1.0, label="Speed fit window (h)", full_width=True)
    sd_front_axis = mo.ui.dropdown(
        {"Distance from edge (µm)": "edge_um", "Fraction of radius from edge": "relative"},
        value="Distance from edge (µm)",
        label="Front position as",
    )
    # Fronts of any channel or cell type share, one row of plots each
    sd_front_quantities = mo.ui.multiselect(
        options=sd_quantity_options,
        value=list(sd_quantity_options)[:1],
        label="Front quantities",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            "### Wavefronts\n\n"
            "- Fronts that start at the edge and move at a similar µm/h in every size suggest a travelling front\n"
            "- Fronts that settle at a fixed fraction of the radius suggest a pattern that scales with the colony\n"
            "- Each quantity's level is the front level fraction of its own range"
        ),
        sd_front_quantities,
        mo.hstack([sd_window, sd_front_axis]),
    ]) if sd_results else mo.md("")
    return sd_front_axis, sd_front_quantities, sd_window


@app.cell(hide_code=True)
def _(
    sd_front_level,
    sd_front_quantities,
    sd_kymograph_entries,
    sd_level_fraction,
    sd_show_references,
    sd_window,
):
    # Front of every quantity, model, size, density and condition; and its speed in the window
    _positions, _speeds, _levels = [], [], []

    def _add_fronts(_entries, _quantity, _level, _source):
        for _model_id, _entry in _entries.items():
            _result = _entry["result"]
            _hours = np.asarray(_result["profile_hours"], dtype=float)
            for _s, _diameter in enumerate(np.asarray(_result["diameters_um"], dtype=float)):
                _edge = profile_positions(_result, _s, "edge_um")
                _radius = _diameter / 2.0
                for _d, _density in enumerate(np.asarray(_result["densities"], dtype=float)):
                    for _k, _condition in enumerate(_result["conditions"]):
                        _fronts = front_positions(_entry["values"][_s, _d, _k], _edge, _level)
                        _common = {
                            "model_id": _model_id, "model": _entry["name"], "model_label": _result["label"],
                            # sweep, true_data or measured_density
                            "source": _source,
                            "group": _entry["group"], "quantity": _quantity, "level": _level,
                            "diameter_um": _diameter, "density": _density, "condition": _condition,
                        }
                        _positions.extend(
                            {**_common, "hour": _h, "front_from_edge_um": _f, "front_relative": _f / _radius}
                            for _h, _f in zip(_hours, _fronts)
                        )
                        _speed = front_speed(_hours, _fronts, tuple(sd_window.value))
                        _speeds.append({
                            **_common,
                            "speed_um_per_h": _speed,
                            "relative_speed_per_h": _speed / _radius,
                            "fronts_found": int(np.isfinite(_fronts).sum()),
                        })

    for _quantity in sd_front_quantities.value:
        _entries = sd_kymograph_entries(_quantity)
        _level, _, _ = sd_front_level(_entries, sd_level_fraction.value)
        _levels.append(f"{_quantity} {_level:.3g}")
        _add_fronts(_entries, _quantity, _level, "sweep")
        # References: fronts at the sweep's level
        for _source in sd_show_references.value:
            _add_fronts(sd_kymograph_entries(_quantity, _source), _quantity, _level, _source)
    sd_front_positions = pd.DataFrame(_positions)
    sd_front_speeds = pd.DataFrame(_speeds)
    mo.md("- Front levels: " + ", ".join(_levels)) if _levels else mo.md("")
    return sd_front_positions, sd_front_speeds


@app.cell(hide_code=True)
def _(
    sd_condition,
    sd_fixed_density,
    sd_front_axis,
    sd_front_positions,
    sd_front_quantities,
    sd_front_speeds,
    sd_group,
    sd_results,
    sd_uncertainty,
):
    if sd_front_positions.empty or sd_group.value is None or sd_fixed_density.value is None:
        _front_view = mo.md("")
    else:
        _column = "front_from_edge_um" if sd_front_axis.value == "edge_um" else "front_relative"
        _speed_column = "speed_um_per_h" if sd_front_axis.value == "edge_um" else "relative_speed_per_h"
        _quantities = [_q for _q in sd_front_quantities.value if _q in set(sd_front_positions["quantity"])]
        _training = float(np.mean([_r["training_diameter_um"] for _r in sd_results]))
        # One row per quantity: front over time (left), speed against size (right)
        _figure, _axes = plt.subplots(
            len(_quantities), 2, figsize=(10, 3.8 * len(_quantities)), dpi=150, squeeze=False,
        )
        _sources = [_s for _s in REFERENCE_LABELS if _s in set(sd_front_positions["source"])]
        for (_left, _right), _quantity in zip(_axes, _quantities):
            _of = sd_front_positions[
                (sd_front_positions["quantity"] == _quantity)
                & (sd_front_positions["group"] == sd_group.value)
                & (sd_front_positions["condition"] == sd_condition.value)
            ]
            _shown = _of[(_of["source"] == "sweep") & np.isclose(_of["density"], sd_fixed_density.value)]
            _diameters = np.sort(_shown["diameter_um"].unique())
            _colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(_diameters)))
            for _diameter, _colour in zip(_diameters, _colours):
                _rows = []
                for _hour, _values in _shown[_shown["diameter_um"] == _diameter].groupby("hour"):
                    _values = _values[_column].dropna()
                    if len(_values):
                        _rows.append((_hour, *uncertainty_band(_values, sd_uncertainty.value, clip=False)))
                if _rows:
                    _hour, _mean, _lower, _upper = np.asarray(_rows).T
                    _left.plot(_hour, _mean, color=_colour, label=f"{_diameter:g} µm")
                    _left.fill_between(_hour, _lower, _upper, color=_colour, alpha=0.15, linewidth=0)
            # References: each size at the measured density (its size's colour), the true colony (black)
            for _source in _sources:
                _reference = _of[_of["source"] == _source]
                for _diameter, _rows in _reference.groupby("diameter_um"):
                    _mean = _rows.groupby("hour")[_column].mean()
                    _nearest = int(np.argmin(np.abs(_diameters - _diameter))) if len(_diameters) else 0
                    _left.plot(
                        _mean.index.to_numpy(), _mean.to_numpy(), linestyle=REFERENCE_LINESTYLES[_source],
                        color="black" if _source == "true_data" or not len(_diameters) else _colours[_nearest],
                    )
            _left.set(
                xlabel="Time (h)",
                ylabel="Front, distance from edge (µm)" if _column == "front_from_edge_um" else "Front, fraction of radius from edge",
                title=f"{_quantity} · density {sd_fixed_density.value:g}",
            )
            if _left.has_data():
                _handles, _labels = _left.get_legend_handles_labels()
                _reference_handles = reference_legend_handles(_sources)
                _left.legend(
                    _handles + _reference_handles, _labels + [_h.get_label() for _h in _reference_handles], fontsize=7,
                )
            _all_speeds = sd_front_speeds[
                (sd_front_speeds["quantity"] == _quantity)
                & (sd_front_speeds["group"] == sd_group.value)
                & (sd_front_speeds["condition"] == sd_condition.value)
            ]
            _speeds = _all_speeds[_all_speeds["source"] == "sweep"]
            _densities = np.sort(_speeds["density"].unique())
            _colours = plt.get_cmap("plasma")(np.linspace(0, 0.85, len(_densities)))
            for _density, _colour in zip(_densities, _colours):
                _line = _speeds[_speeds["density"] == _density].groupby("diameter_um")[_speed_column].agg(["mean", "std"])
                _right.errorbar(_line.index, _line["mean"], yerr=_line["std"], color=_colour, marker="o", capsize=2, label=f"{_density:g}")
            # References: every size at the measured density, and the true colony at its size
            for _source in _sources:
                _line = _all_speeds[_all_speeds["source"] == _source].groupby("diameter_um")[_speed_column].agg(["mean", "std"])
                _right.errorbar(
                    _line.index.to_numpy(), _line["mean"].to_numpy(), yerr=_line["std"].fillna(0.0).to_numpy(),
                    color="black", marker=REFERENCE_MARKERS[_source], linestyle=REFERENCE_LINESTYLES[_source],
                    markersize=9 if _source == "true_data" else 5, capsize=2, label=REFERENCE_LABELS[_source],
                )
            _right.axhline(0.0, color="grey", linewidth=0.6)
            _right.axvline(_training, color="grey", linewidth=0.8, linestyle="--")
            _right.set(
                xlabel="Colony diameter (µm)",
                ylabel="Front speed inward (µm/h)" if _speed_column == "speed_um_per_h" else "Front speed inward (radii/h)",
                title=f"{_quantity} · speed in the fit window (dashed: training size)",
            )
            _right.legend(title="Density", fontsize=7)
        _figure.suptitle(f"{sd_group.value} · {sd_condition.value}")
        _figure.tight_layout()
        _front_view = _figure
    _front_view
    return


@app.cell(hide_code=True)
def _(sd_front_positions, sd_front_speeds, sd_shares):
    def _download(_table, _name, _label):
        return mo.download(data=_table.to_csv(index=False).encode(), filename=_name, mimetype="text/csv", label=_label)

    (
        mo.hstack([
            _download(sd_shares, "size_density_shares.csv", "Cell-type shares"),
            _download(sd_front_positions, "size_density_fronts.csv", "Front positions"),
            _download(sd_front_speeds, "size_density_front_speeds.csv", "Front speeds"),
        ], justify="start")
        if not sd_shares.empty else mo.md("")
    )
    return


if __name__ == "__main__":
    app.run()
