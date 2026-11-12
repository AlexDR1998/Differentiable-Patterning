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
        cell_edges,
        model_option_label,
        model_stain_counts,
        parse_one_based_indices,
        read_model_settings,
        rules_projection,
        uncertainty_band,
    )
    from Experiments.micropatterns.rendering import render_fate_cmy
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
    from Experiments.micropatterns.figures.style import STYLE_FILE, cell_type_colour
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
    return (
        sd_bin_um,
        sd_density_count,
        sd_diameters,
        sd_duplicate_runs,
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
    sd_results = tuple(_r for _r in _all if _r["projection"] in cell_type_rules_by_projection) or None

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
    mo.vstack([mo.md("## Results"), *_warnings] if sd_results else [])
    return (sd_results,)


@app.cell(hide_code=True)
def _(cell_type_labels, sd_results):
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
    (
        mo.vstack([
            sd_display_cell_types,
            mo.hstack([sd_group_by, sd_uncertainty, sd_condition, sd_fixed_density], justify="start"),
            mo.hstack([sd_basis, sd_resolve], justify="start"),
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
        sd_uncertainty,
    )


@app.cell(hide_code=True)
def _(cell_type_labels, cell_type_rules_by_projection, sd_basis, sd_group_by, sd_resolve, sd_results, sd_uncertainty):
    # Colony-wide shares at the end time, from the stored ring counts
    _rows = []
    for _result in sd_results or ():
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
                            "model_label": _result["label"],
                            "projection": _result["projection"],
                            "replicate": _result["replicate"],
                            "seed": _result["seed"],
                            "group": _result["groups"][sd_group_by.value],
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
    sd_shares = pd.DataFrame(_rows)
    _summary = []
    if not sd_shares.empty:
        for (_group, _cell_type, _diameter, _density, _condition), _values in sd_shares.groupby(
            ["group", "cell_type", "diameter_um", "density", "condition"], sort=False
        ):
            _mean, _lower, _upper = uncertainty_band(_values["share"], sd_uncertainty.value)
            _summary.append({
                "group": _group, "cell_type": _cell_type, "diameter_um": _diameter, "density": _density,
                "condition": _condition, "cells_per_mm2": float(_values["cells_per_mm2"].mean()),
                "covered_share": float(_values["covered_share"].mean()),
                "mean_share": _mean, "lower": _lower, "upper": _upper, "samples": len(_values),
            })
    sd_summary = pd.DataFrame(_summary)
    return sd_shares, sd_summary


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
def _(sd_condition, sd_display_cell_types, sd_group, sd_heatmap_scale, sd_results, sd_shares, sd_summary):
    if sd_summary.empty or sd_group.value is None or not sd_display_cell_types.value:
        _heatmap_view = mo.md("")
    else:
        _shown = sd_summary[(sd_summary["group"] == sd_group.value) & (sd_summary["condition"] == sd_condition.value)]
        _unit = sd_results[0]["density_unit"]
        _x_column = "cells_per_mm2" if _unit == "per_mm2" else "covered_share"
        _cell_types = [_c for _c in sd_display_cell_types.value if _c in set(_shown["cell_type"])]
        _figure, _axes = plt.subplots(1, len(_cell_types), figsize=(4.0 * len(_cell_types), 3.6), dpi=150, squeeze=False)
        # Columns at the density each requested density used (mean over sizes and runs)
        _x = _shown.groupby("density")[_x_column].mean().sort_index().to_numpy()
        _diameters = np.sort(_shown["diameter_um"].unique())
        _training = float(sd_shares["training_diameter_um"].mean())
        _measured = float(sd_shares["measured_density"].mean())
        for _axis, _cell_type in zip(_axes[0], _cell_types):
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
            _figure.colorbar(_image, ax=_axis, label="Share")
        _figure.suptitle(
            f"{sd_group.value} · {sd_condition.value} · mean share at {sd_results[0]['snapshot_hour']:g}h "
            "(dashed: training size, dotted: measured density)"
        )
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
    sd_results,
    sd_shares,
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
        _left.set(
            xlabel="Colony diameter (µm)", ylabel="Share",
            title=f"At {format_density(sd_fixed_density.value, _unit)} {DENSITY_LABELS[_unit].lower()}",
        )
        _left.legend(fontsize=7)
        _of_type = _shown[_shown["cell_type"] == sd_slice_cell_type.value]
        _diameters = np.sort(_of_type["diameter_um"].unique())
        _colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(_diameters)))
        for _diameter, _colour in zip(_diameters, _colours):
            _line = _of_type[_of_type["diameter_um"] == _diameter].sort_values("density")
            _right.plot(_line[_x_column], _line["mean_share"], marker="o", color=_colour, label=f"{_diameter:g} µm")
            _right.fill_between(_line[_x_column], _line["lower"], _line["upper"], color=_colour, alpha=0.15, linewidth=0)
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
def _(sd_results):
    # Images are kept for the first run of each job part only
    _with_images = [
        (f"{_r['label']} · replicate {_r['replicate']} · run {_r['run_index']}", _i)
        for _i, _r in enumerate(sd_results or ()) if _r["final_states"] is not None
    ]
    sd_image_result = mo.ui.dropdown(
        dict(_with_images), value=_with_images[0][0] if _with_images else None, label="Run",
    )
    sd_image_stage = mo.ui.dropdown(["End", "Initial (0h)"], value="End", label="State")
    (
        mo.vstack([
            mo.md("### Colonies\n\n- SOX2 cyan, TBXT magenta, SOX17 yellow; every colony drawn to scale"),
            mo.hstack([sd_image_result, sd_image_stage], justify="start"),
        ])
        if _with_images else mo.md("")
    )
    return sd_image_result, sd_image_stage


@app.cell(hide_code=True)
def _(sd_condition, sd_image_result, sd_image_stage, sd_results):
    if sd_image_result.value is None:
        _image_view = mo.md("")
    else:
        _result = sd_results[sd_image_result.value]
        _names = list(_result["state_channel_names"])
        _k = list(_result["conditions"]).index(sd_condition.value)
        _sizes, _densities = len(_result["diameters_um"]), len(_result["densities"])
        _figure, _axes = plt.subplots(
            _sizes, _densities, figsize=(1.7 * _densities, 1.7 * _sizes), dpi=150, squeeze=False,
        )
        for _s, _mask in enumerate(_result["colony_masks"]):
            _states = (
                _result["final_states"][_s, :, _k] if sd_image_stage.value == "End"
                else _result["initial_states"][_s, :, : len(_names)]
            )
            _rgb = render_fate_cmy(np.asarray(_states, dtype=np.float32), _names)
            for _d in range(_densities):
                _axis = _axes[_s, _d]
                _axis.imshow(np.where(_mask[..., None], _rgb[_d], 1.0))
                _axis.set_xticks([])
                _axis.set_yticks([])
                if _d == 0:
                    _axis.set_ylabel(f"{_result['diameters_um'][_s]:g} µm", fontsize=7)
                if _s == 0:
                    _axis.set_title(
                        f"{_result['cells_per_mm2'][-1, _d]:.0f}/mm²" if _result["density_unit"] == "per_mm2"
                        else f"{_result['covered_shares'][-1, _d]:.2f}",
                        fontsize=7,
                    )
        _figure.suptitle(f"{_result['label']} · {sd_condition.value} · {sd_image_stage.value}", fontsize=9)
        _figure.tight_layout()
        _image_view = _figure
    _image_view
    return


@app.cell(hide_code=True)
def _(cell_type_labels, sd_results):
    _channels = list(sd_results[0]["state_channel_names"]) if sd_results else []
    sd_quantity = mo.ui.dropdown(
        options={**{f"{_c} (channel mean)": _c for _c in _channels}, **{f"{_l} (share)": _l for _l in cell_type_labels}},
        value=f"{_channels[0]} (channel mean)" if _channels else None,
        label="Quantity",
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
                "- Level: the chosen fraction of the quantity's range over every shown model, size, density and time"
            ),
            mo.hstack([sd_quantity, sd_axis, sd_level_fraction], justify="start"),
        ])
        if sd_results else mo.md("")
    )
    return sd_axis, sd_level_fraction, sd_quantity


@app.cell(hide_code=True)
def _(cell_type_rules_by_projection, sd_group_by, sd_level_fraction, sd_quantity, sd_resolve, sd_results):
    # Mean kymograph [sizes, densities, conditions, times, bins] of each model
    sd_kymographs = {}
    for _model_id in dict.fromkeys(_r["model_id"] for _r in sd_results or ()):
        _runs = [_r for _r in sd_results if _r["model_id"] == _model_id]
        sd_kymographs[_model_id] = {
            "values": mean_kymograph(
                _runs, sd_quantity.value, cell_type_rules_by_projection[_runs[0]["projection"]], sd_resolve.value,
            ),
            "result": _runs[0],
            "label": _runs[0]["label"],
            "group": _runs[0]["groups"][sd_group_by.value],
        }
    _values = np.concatenate([_k["values"].ravel() for _k in sd_kymographs.values()]) if sd_kymographs else np.zeros(1)
    _low, _high = float(np.nanmin(_values)), float(np.nanmax(_values))
    sd_level = _low + float(sd_level_fraction.value) * (_high - _low)
    mo.md(f"- Front level {sd_level:.3g} (range {_low:.3g} to {_high:.3g})") if sd_kymographs else mo.md("")
    return sd_kymographs, sd_level


@app.cell(hide_code=True)
def _(sd_kymographs):
    _first = next(iter(sd_kymographs.values()), None)
    sd_kymo_model = mo.ui.dropdown(
        {_k["label"]: _id for _id, _k in sd_kymographs.items()},
        value=_first["label"] if _first else None,
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
        _panels = (
            [(_s, _d_fixed) for _s in range(len(_diameters))] if sd_kymo_layout.value == "sizes"
            else [(_s_fixed, _d) for _d in range(len(_densities))]
        )
        _hours = np.asarray(_result["profile_hours"], dtype=float)
        _shown = _values[:, :, _k]
        _scale = {"vmin": float(np.nanmin(_shown)), "vmax": float(np.nanmax(_shown))} if sd_kymo_shared.value else {}
        _figure, _axes = plt.subplots(
            1, len(_panels), figsize=(2.4 * len(_panels) + 1.0, 3.4), dpi=150, sharey=True, squeeze=False,
        )
        for _axis, (_s, _d) in zip(_axes[0], _panels):
            _z = _values[_s, _d, _k]
            _keep = np.isfinite(_z).any(axis=0)
            _edge = profile_positions(_result, _s, "edge_um")[_keep]
            _radius = _diameters[_s] / 2.0
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
            _axis.set_title(f"{_diameters[_s]:g} µm · {_densities[_d]:g}", fontsize=8)
            _axis.set_xlabel(AXIS_LABELS[sd_axis.value], fontsize=7)
        _axes[0, 0].set_ylabel("Time (h)")
        _figure.colorbar(_mesh, ax=_axes[0].tolist(), label=sd_quantity.value)
        _figure.suptitle(
            f"{_entry['label']} · {sd_condition.value} · white: level {sd_level:.3g}, blue dashed: front from the edge",
            fontsize=9,
        )
        _kymograph_view = _figure
    _kymograph_view
    return


@app.cell(hide_code=True)
def _(sd_axis, sd_condition, sd_fixed_density, sd_kymo_model, sd_kymographs, sd_quantity):
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
        _axis.set(
            xlabel=AXIS_LABELS[sd_axis.value], ylabel=sd_quantity.value,
            title=f"{_entry['label']} · {sd_condition.value} · {_result['snapshot_hour']:g}h · density {_result['densities'][_d]:g}",
        )
        _axis.legend(fontsize=7)
        _figure.tight_layout()
        _profile_view = _figure
    _profile_view
    return


@app.cell(hide_code=True)
def _(sd_results):
    _end = float(sd_results[0]["snapshot_hour"]) if sd_results else 48.0
    sd_window = mo.ui.range_slider(0.0, _end, value=[0.0, _end], step=1.0, label="Speed fit window (h)", full_width=True)
    sd_front_axis = mo.ui.dropdown(
        {"Distance from edge (µm)": "edge_um", "Fraction of radius from edge": "relative"},
        value="Distance from edge (µm)",
        label="Front position as",
    )
    mo.vstack([
        mo.md(
            "### Wavefronts\n\n"
            "- Fronts that start at the edge and move at a similar µm/h in every size suggest a travelling front\n"
            "- Fronts that settle at a fixed fraction of the radius suggest a pattern that scales with the colony"
        ),
        mo.hstack([sd_window, sd_front_axis]),
    ]) if sd_results else mo.md("")
    return sd_front_axis, sd_window


@app.cell(hide_code=True)
def _(sd_kymographs, sd_level, sd_quantity, sd_window):
    # Front of every model, size, density and condition; and its speed in the window
    _positions, _speeds = [], []
    for _model_id, _entry in sd_kymographs.items():
        _result = _entry["result"]
        _hours = np.asarray(_result["profile_hours"], dtype=float)
        for _s, _diameter in enumerate(np.asarray(_result["diameters_um"], dtype=float)):
            _edge = profile_positions(_result, _s, "edge_um")
            _radius = _diameter / 2.0
            for _d, _density in enumerate(np.asarray(_result["densities"], dtype=float)):
                for _k, _condition in enumerate(_result["conditions"]):
                    _fronts = front_positions(_entry["values"][_s, _d, _k], _edge, sd_level)
                    _common = {
                        "model_id": _model_id, "model_label": _entry["label"], "group": _entry["group"],
                        "quantity": sd_quantity.value, "level": sd_level, "diameter_um": _diameter,
                        "density": _density, "condition": _condition,
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
    sd_front_positions = pd.DataFrame(_positions)
    sd_front_speeds = pd.DataFrame(_speeds)
    return sd_front_positions, sd_front_speeds


@app.cell(hide_code=True)
def _(
    sd_condition,
    sd_fixed_density,
    sd_front_axis,
    sd_front_positions,
    sd_front_speeds,
    sd_group,
    sd_quantity,
    sd_results,
    sd_uncertainty,
):
    if sd_front_positions.empty or sd_group.value is None or sd_fixed_density.value is None:
        _front_view = mo.md("")
    else:
        _column = "front_from_edge_um" if sd_front_axis.value == "edge_um" else "front_relative"
        _speed_column = "speed_um_per_h" if sd_front_axis.value == "edge_um" else "relative_speed_per_h"
        _shown = sd_front_positions[
            (sd_front_positions["group"] == sd_group.value)
            & (sd_front_positions["condition"] == sd_condition.value)
            & np.isclose(sd_front_positions["density"], sd_fixed_density.value)
        ]
        _figure, (_left, _right) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
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
        _left.set(
            xlabel="Time (h)",
            ylabel="Front, distance from edge (µm)" if _column == "front_from_edge_um" else "Front, fraction of radius from edge",
            title=f"Density {sd_fixed_density.value:g}",
        )
        _left.legend(fontsize=7)
        _speeds = sd_front_speeds[
            (sd_front_speeds["group"] == sd_group.value) & (sd_front_speeds["condition"] == sd_condition.value)
        ]
        _densities = np.sort(_speeds["density"].unique())
        _colours = plt.get_cmap("plasma")(np.linspace(0, 0.85, len(_densities)))
        for _density, _colour in zip(_densities, _colours):
            _line = _speeds[_speeds["density"] == _density].groupby("diameter_um")[_speed_column].agg(["mean", "std"])
            _right.errorbar(_line.index, _line["mean"], yerr=_line["std"], color=_colour, marker="o", capsize=2, label=f"{_density:g}")
        _right.axhline(0.0, color="grey", linewidth=0.6)
        _right.axvline(float(np.mean([_r["training_diameter_um"] for _r in sd_results])), color="grey", linewidth=0.8, linestyle="--")
        _right.set(
            xlabel="Colony diameter (µm)",
            ylabel="Front speed inward (µm/h)" if _speed_column == "speed_um_per_h" else "Front speed inward (radii/h)",
            title="Speed in the fit window (dashed: training size)",
        )
        _right.legend(title="Density", fontsize=7)
        _figure.suptitle(f"{sd_group.value} · {sd_condition.value} · {sd_quantity.value}")
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
