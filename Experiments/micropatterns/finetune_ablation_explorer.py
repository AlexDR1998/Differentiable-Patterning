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

"""Fine-tune vs joint training ablation: evaluate the models and plot the results.

Run from the repository root with:

    marimo edit Experiments/micropatterns/finetune_ablation_explorer.py

The sweeps are described in docs/finetune_ablation.md.
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

    from Experiments.micropatterns.figures.style import (
        CMAP,
        SERIES,
        WIDTH,
        paper_style,
    )
    from Experiments.micropatterns.finetune_ablation import (
        AblationEvaluationSettings,
        CONDITIONS,
        evaluate_models,
        job_configs,
        load_results,
        results_table,
        save_results,
    )
    from Experiments.model_registry import load_selection
    from Experiments.notebook_jobs import repository_path, write_job_manifest

    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env")
    METRICS = {
        "radial_mse": "Ring profile error",
        "pixel_mse": "Pixel error",
        "share_error": "Cell type share error (last time)",
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fine-tune vs joint training ablation

    - Every model has 12000 iterations in total. A share **f** of them is baseline-only pretraining (the parent); the rest trains on all conditions.
    - **s** is the share of the loss weight on the baseline condition while training on all conditions.
    - Every model is scored the same way on its held-out replicates (`Experiments/micropatterns/finetune_ablation.py`), so the arms can be compared.
    - Sweeps and workflow: `docs/finetune_ablation.md`.
    """)
    return


@app.cell(hide_code=True)
def _():
    abl_selection = mo.ui.text(
        "models/finetune_ablation_v1.yaml",
        label="Selection of the ablation models (from model_registry_explorer.py)",
        full_width=True,
    )
    abl_store = mo.ui.text(
        os.getenv("MODEL_STORE_ROOT", "models"), label="Model store", full_width=True
    )
    abl_data = mo.ui.text(
        os.getenv("DATA_PATH_BASE", ""),
        label="Data root (holds 260726_nca_dataset)",
        full_width=True,
    )
    abl_seeds = mo.ui.number(start=1, stop=8, value=2, label="Rollouts per replicate")
    abl_rules = mo.ui.text(
        "",
        label="Cell type rule file for share errors (optional; must not need left-out channels)",
        full_width=True,
    )
    mo.vstack([
        mo.md("## 1. Models and settings"),
        abl_selection,
        abl_store,
        abl_data,
        abl_rules,
        abl_seeds,
    ])
    return abl_data, abl_rules, abl_seeds, abl_selection, abl_store


@app.cell(hide_code=True)
def _(abl_data, abl_rules, abl_seeds, abl_selection, abl_store):
    abl_model_ids = []
    abl_settings = AblationEvaluationSettings(
        data_root=abl_data.value.strip(),
        store_root=abl_store.value.strip(),
        seeds=tuple(range(int(abl_seeds.value))),
        cell_type_rules=abl_rules.value.strip() or None,
    )
    _path = repository_path(abl_selection.value.strip())
    if _path.is_file():
        abl_model_ids = [_record["model_id"] for _record in load_selection(_path)]
        _status = mo.md(f"{len(abl_model_ids)} models in the selection.")
    else:
        _status = mo.callout(
            "No selection file yet. Pick the ablation models in "
            "`Experiments/model_registry_explorer.py` and export a selection, "
            "or skip to section 3 to load results.",
            kind="warn",
        )
    _status
    return abl_model_ids, abl_settings


@app.cell(hide_code=True)
def _():
    abl_run_here = mo.ui.run_button(label="Evaluate here")
    abl_job_name = mo.ui.text("finetune_ablation_v1", label="Job name")
    abl_job_output = mo.ui.text(
        "/workspace/writeable/output/finetune_ablation",
        label="Results folder on the cluster (the job name is added)",
        full_width=True,
    )
    abl_export = mo.ui.run_button(label="Export remote job")
    mo.vstack([
        mo.md(
            "## 2. Evaluate\n\n"
            "- **Here**: fine for a few models (tens of seconds per small model and condition)\n"
            "- **Remote job**: one pod per model; see `docs/remote_notebook_jobs.md`"
        ),
        abl_run_here,
        mo.hstack([abl_job_name]),
        abl_job_output,
        abl_export,
    ])
    return abl_export, abl_job_name, abl_job_output, abl_run_here


@app.cell(hide_code=True)
def _(abl_model_ids, abl_run_here, abl_settings):
    mo.stop(not abl_run_here.value)
    mo.stop(not abl_model_ids, mo.callout("Choose a selection first.", kind="warn"))
    _steps = len(abl_model_ids) * len(abl_settings.conditions) * 2 * len(abl_settings.seeds)
    with mo.status.progress_bar(total=_steps, title="Evaluating") as _bar:
        abl_new_results = evaluate_models(
            abl_settings, abl_model_ids, progress=lambda _n: _bar.update(_n)
        )
    mo.md(f"Evaluated {len(abl_model_ids)} models. Save them in section 3.")
    return (abl_new_results,)


@app.cell(hide_code=True)
def _(abl_export, abl_job_name, abl_job_output, abl_model_ids, abl_settings):
    mo.stop(not abl_export.value)
    mo.stop(not abl_model_ids, mo.callout("Choose a selection first.", kind="warn"))
    _name = abl_job_name.value.strip() or "finetune_ablation"
    _output = f"{abl_job_output.value.strip().rstrip('/')}/{_name}"
    _configs = job_configs(abl_settings, abl_model_ids, _output)
    # Next to the training manifests, so a git checkout on the cluster has it
    _manifest = write_job_manifest(
        repository_root / "Experiments/micropatterns/conf/generated/finetune_ablation_eval" / _name / "manifest.yaml",
        f"finetune_ablation_eval_{_name}",
        _configs,
    )
    _relative = _manifest.relative_to(repository_root)
    mo.callout(
        mo.md(
            f"Wrote {_relative} ({len(_configs)} parts). On the cluster, from the repository root:\n\n"
            f"```bash\nbash launch/launch_batch_multi_job.sh "
            f"Experiments/micropatterns/finetune_ablation.py {_relative} <workers> H200\n```\n\n"
            f"Then copy `{_output}/` back to `output/finetune_ablation/{_name}/` and load that folder below. "
            f"Test one part locally with `python Experiments/micropatterns/finetune_ablation.py "
            f"--manifest {_relative} --index 0`."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    abl_results_path = mo.ui.text(
        "output/finetune_ablation/finetune_ablation_v1",
        label="Results file (.npz), or folder of a remote job's parts",
        full_width=True,
    )
    abl_save = mo.ui.run_button(label="Save results evaluated here")
    abl_load = mo.ui.run_button(label="Load results")
    mo.vstack([mo.md("## 3. Results"), abl_results_path, mo.hstack([abl_save, abl_load])])
    return abl_load, abl_results_path, abl_save


@app.cell(hide_code=True)
def _(abl_new_results, abl_results_path, abl_save):
    mo.stop(not abl_save.value)
    _path = repository_path(abl_results_path.value.strip())
    if _path.suffix != ".npz":
        _path = _path / "results.npz"
    save_results(_path, abl_new_results)
    mo.md(f"Saved to `{_path}`.")
    return


@app.cell(hide_code=True)
def _(abl_load, abl_results_path):
    mo.stop(not abl_load.value)
    _path = repository_path(abl_results_path.value.strip())
    mo.stop(not _path.exists(), mo.callout(f"`{_path}` does not exist.", kind="warn"))
    abl_table = results_table(load_results(_path))
    mo.md(f"{abl_table['model_id'].nunique()} models, {len(abl_table)} rows.")
    return (abl_table,)


@app.cell(hide_code=True)
def _(abl_table):
    _metrics = {label: name for name, label in METRICS.items() if name in abl_table}
    abl_metric = mo.ui.dropdown(_metrics, value=next(iter(_metrics)), label="Score")
    _times = ["all", *sorted(abl_table["time"].unique())]
    abl_time = mo.ui.dropdown(_times, value="all", label="Time point")
    mo.hstack([abl_metric, abl_time])
    return abl_metric, abl_time


@app.cell(hide_code=True)
def _(abl_metric, abl_table, abl_time):
    _rows = abl_table if abl_time.value == "all" else abl_table[abl_table["time"] == abl_time.value]
    # One score per model and condition first, so every model counts once
    abl_per_model = (
        _rows.groupby(
            ["model_id", "condition", "pretrain_fraction", "baseline_share"], as_index=False
        )[abl_metric.value]
        .mean()
        .dropna()
    )
    abl_per_model["f"] = abl_per_model["pretrain_fraction"].round(2)
    abl_per_model["s"] = abl_per_model["baseline_share"].round(2)
    abl_cells = abl_per_model.groupby(["condition", "f", "s"])[abl_metric.value].agg(["mean", "std", "count"])
    return abl_cells, abl_per_model


@app.cell(hide_code=True)
def _(abl_cells, abl_metric):
    _conditions = [_c for _c in CONDITIONS if _c in abl_cells.index.get_level_values(0)]
    _f = sorted(abl_cells.index.get_level_values(1).unique())
    _s = sorted(abl_cells.index.get_level_values(2).unique())
    with paper_style():
        _fig, _axes = plt.subplots(
            1, len(_conditions), figsize=(WIDTH["double"], WIDTH["double"] / 3.2), squeeze=False
        )
        for _ax, _condition in zip(_axes[0], _conditions):
            _grid = (
                abl_cells.loc[_condition]["mean"].unstack("s").reindex(index=_f, columns=_s)
            )
            _image = _ax.imshow(_grid.to_numpy(), cmap=CMAP["heatmap"], aspect="auto", origin="lower")
            _ax.set_xticks(range(len(_s)), [f"{_v:.2f}" for _v in _s])
            _ax.set_yticks(range(len(_f)), [f"{_v:.2f}" for _v in _f])
            _ax.set_xlabel("Baseline share s")
            _ax.set_ylabel("Pretraining share f")
            _ax.set_title(_condition)
            _fig.colorbar(_image, ax=_ax, label=METRICS[abl_metric.value])
        _fig.tight_layout()
    mo.vstack([
        mo.md(
            "### Mean score per arm\n\n"
            "- Lower is better. Empty cells are arms not in the sweeps\n"
            "- f = 0: trained from scratch on all conditions; f = 1, s = 1: baseline only"
        ),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(abl_metric, abl_per_model):
    _conditions = [_c for _c in CONDITIONS if _c in set(abl_per_model["condition"])]
    _shares = sorted(abl_per_model["s"].unique())
    with paper_style():
        _fig, _axes = plt.subplots(
            1, len(_conditions), figsize=(WIDTH["double"], WIDTH["double"] / 3.2),
            sharey=True, squeeze=False,
        )
        for _ax, _condition in zip(_axes[0], _conditions):
            _rows = abl_per_model[abl_per_model["condition"] == _condition]
            for _index, _share in enumerate(_shares):
                _points = _rows[_rows["s"] == _share]
                _colour = SERIES[_index % len(SERIES)]
                _ax.scatter(_points["f"], _points[abl_metric.value], s=6, color=_colour, alpha=0.5)
                _means = _points.groupby("f")[abl_metric.value].mean()
                _ax.plot(_means.index, _means.to_numpy(), color=_colour, label=f"s = {_share:.2f}")
            _ax.set_xlabel("Pretraining share f")
            _ax.set_title(_condition)
        _axes[0, 0].set_ylabel(METRICS[abl_metric.value])
        _axes[0, -1].legend(frameon=False)
        _fig.tight_layout()
    mo.vstack([
        mo.md("### Every model\n\n- Dots: one model (replicate split × repeat); lines: mean per baseline share"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(abl_cells):
    mo.vstack([mo.md("### Table"), abl_cells.reset_index()])
    return


if __name__ == "__main__":
    app.run()
