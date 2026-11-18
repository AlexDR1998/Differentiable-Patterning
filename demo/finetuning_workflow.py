# /// script
# dependencies = ["marimo", "jax", "equinox", "matplotlib", "numpy", "pandas", "pyyaml", "omegaconf", "python-dotenv"]
# ///

"""Tutorial: train a micropattern baseline, fine-tune it on new experiments, and evaluate.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/finetuning_workflow.py

Nothing is trained here: training runs on the EIDF cluster. Sections 1-4
run on a laptop in seconds with only the files in this repository. Section 5
needs the model store (MODEL_STORE_ROOT) and the micropattern data
(DATA_PATH_BASE), and takes a minute or so on a GPU.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml
    from dotenv import load_dotenv

    from Experiments.config import load_experiment_config
    from Experiments.config_workflow import generate_manifest, load_yaml
    from Experiments.micropatterns.finetune import (
        check_finetune_sweep,
        check_parent_compatible,
        finetune_sweep,
    )
    from Experiments.model_registry import ModelRegistry, load_selection

    REPO_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(REPO_ROOT / ".env")
    CONF = REPO_ROOT / "demo" / "conf" / "finetuning"
    BASE_CONFIG = REPO_ROOT / "Experiments" / "micropatterns" / "conf" / "base_config.yaml"
    # Everything this notebook writes goes here (ignored by git)
    OUTPUT = REPO_ROOT / "output" / "tutorial_finetuning"

    def relative(path):
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()

    def show_yaml(path):
        return mo.md(f"`{relative(path)}`\n\n```yaml\n{Path(path).read_text()}\n```")

    def run_table(configs):
        """One row per run with the settings that matter here."""
        return pd.DataFrame([
            {
                "run": index,
                "parent": config.initialization.model_id,
                "train replicates": config.data.micropattern.train_replicates,
                "held out": config.data.micropattern.validation_replicates,
                "downsample": config.data.downsample,
                "run.t": config.run.t,
                "curriculum": config.data.knockout.curriculum or ("baseline",),
                "iterations": config.run.iterations,
            }
            for index, config in enumerate(configs)
        ])

    class PretendParent:
        """Stands in for a trained bundle when no model store is available."""

        def __init__(self, model_id, config):
            self.id = model_id
            self.config = config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Baseline training, fine-tuning and evaluation

    - Some experiments give only a little data: Nodal knockouts (KO), and soon colonies seeded at different cell densities
    - Training on them alone wastes what the large wild-type (**baseline**) data already teaches. Instead:
        1. train a **baseline** model on the control colonies
        2. **fine-tune** it: a new run that starts from the baseline model's weights (`initialization.model_id`) and trains on the new conditions
        3. **evaluate** the baseline and the fine-tuned model the same way, on replicates neither saw
    - What carries over: only the weights. The optimiser, learning-rate warmup and loss schedules start again
    - The child must read the data **exactly** as its parent did: same model, resolution, `run.t`, projection, cleaning, normalisation and training replicates. Otherwise the weights don't mean the same thing. The run stops if any of these differ
    - Is fine-tuning better than training on everything at once? That is the ablation in `docs/finetune_ablation.md`
    - This is the most fiddly workflow in the repository. Each section is one step, with the commands for the cluster
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The baseline sweep

    - An ordinary sweep file: the settings that differ from `Experiments/micropatterns/conf/base_config.yaml`
    - `data.knockout.curriculum: [null]` means control (baseline) colonies only
    - The `groups:` block makes one run per **replicate split**: train on replicates 1-2 and hold out 3-4, and the other way round. Each fine-tune later uses its parent's split, so the held-out replicates stay unseen
    """)
    return


@app.cell
def _():
    show_yaml(CONF / "baseline_v1.yaml")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - A **manifest** lists every run of a sweep with its full config. Make it from **one** sweep file (never a folder: that rewrites every manifest in it)
    - On the real workflow the manifest goes into `Experiments/micropatterns/conf/generated/` and is committed, as the record of how the models were trained. Here it goes into `output/` instead
    """)
    return


@app.cell
def _():
    baseline_button = mo.ui.run_button(label="Make the baseline manifest")
    baseline_button
    return (baseline_button,)


@app.cell
def _(baseline_button):
    mo.stop(not baseline_button.value)
    _sweep = load_yaml(CONF / "baseline_v1.yaml")
    baseline_manifest = generate_manifest(
        load_yaml(BASE_CONFIG), _sweep, OUTPUT / _sweep["experiment_name"]
    )
    # The same check as docs/workflow_micropatterns.md step 3
    baseline_configs = [load_experiment_config(_entry["config"]) for _entry in baseline_manifest["configs"]]
    _path = relative(OUTPUT / _sweep["experiment_name"] / "manifest.yaml")
    mo.vstack([
        mo.md(f"All {len(baseline_configs)} configs are valid. Written to `{_path}`."),
        run_table(baseline_configs),
        mo.md(
            "On the cluster (`docs/cluster_eidf.md`), after committing the sweep file and its "
            "manifest and updating the NFS checkout:\n\n"
            "```bash\n"
            "python Experiments/generate_configs.py demo/conf/finetuning/baseline_v1.yaml \\\n"
            "    Experiments/micropatterns/conf/base_config.yaml Experiments/micropatterns/conf/generated\n"
            "bash launch/launch_batch_multi_job.sh Experiments/run_config.py \\\n"
            "    Experiments/micropatterns/conf/generated/tutorial_finetune_baseline_v1/manifest.yaml 2 H200\n"
            "```"
        ),
    ])
    return (baseline_configs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Bring the models back and choose the parents

    - Copy the finished bundles from the cluster and rebuild the index:
      `helper_scripts/copy_models_from_cluster.sh --port <port>`
    - Open `marimo edit Experiments/model_registry_explorer.py`, tick the baseline models to fine-tune, and export a **selection** (a YAML list of model IDs). Keep it in git, e.g. under `Experiments/micropatterns/figures/selections/`
    - `demo/conf/finetuning/selection_example.yaml` is one such file. It names a real ds32 sum baseline in the shared store
    - No store on this machine? Choose **pretend parents** below: the baseline manifest's configs stand in for trained models, so sections 3-4 still work
    - Press the button in section 1 first; the cells below wait for it
    """)
    return


@app.cell
def _():
    show_yaml(CONF / "selection_example.yaml")
    return


@app.cell
def _():
    parent_source = mo.ui.radio(
        {
            "Pretend parents (the baseline manifest above)": "pretend",
            "A selection file and the model store": "store",
        },
        value="Pretend parents (the baseline manifest above)",
        label="Parents",
    )
    selection_path = mo.ui.text(
        relative(CONF / "selection_example.yaml"), label="Selection", full_width=True
    )
    store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(REPO_ROOT / "models")), label="Model store", full_width=True
    )
    mo.vstack([parent_source, selection_path, store_root])
    return parent_source, selection_path, store_root


@app.cell
def _(parent_source, selection_path, store_root):
    if parent_source.value == "store":
        _root = Path(store_root.value).expanduser()
        mo.stop(
            not (_root / "bundles").is_dir(),
            mo.callout(f"No `bundles/` folder in `{_root}`. Use pretend parents, or set the store.", kind="warn"),
        )
        _records = load_selection(REPO_ROOT / selection_path.value)
        registry = ModelRegistry(_root)
        try:
            parents = [registry.get(_record["model_id"]) for _record in _records]
        except Exception as _error:
            mo.stop(True, mo.callout(f"A selected model is not in this store: {_error}", kind="warn"))
    else:
        registry = None
        parents = None
    return parents, registry


@app.cell
def _(baseline_configs, parents):
    # Pretend parents need the baseline manifest from section 1
    chosen_parents = parents or [
        PretendParent(f"pretend-parent-{_index}", _config)
        for _index, _config in enumerate(baseline_configs)
    ]
    mo.vstack([mo.md(f"{len(chosen_parents)} parent(s):"), run_table([_p.config for _p in chosen_parents]).assign(model_id=[_p.id for _p in chosen_parents])])
    return (chosen_parents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Write the fine-tune sweep

    - Typing parent IDs into a sweep by hand, along with each parent's cleaning, resolution and split, is where most fine-tuning mistakes come from
    - Instead, a **template** holds only what is new: the conditions (`data.knockout.curriculum`), the training length and the loss. It sets `initialization.model_id: [FROM_PARENTS]`
    - `Experiments/micropatterns/finetune.py` adds one row per parent to a zipped `groups:` block, copying everything the child must share with that parent. It then expands the sweep and checks every run against its parent
    - Command line: `python -m Experiments.micropatterns.finetune <template> --selection <selection> --out <sweep.yaml>`
    """)
    return


@app.cell
def _():
    show_yaml(CONF / "finetune_template_v1.yaml")
    return


@app.cell
def _():
    write_button = mo.ui.run_button(label="Write and check the fine-tune sweep")
    write_button
    return (write_button,)


@app.cell
def _(chosen_parents, write_button):
    mo.stop(not write_button.value)
    finetune = finetune_sweep(load_yaml(CONF / "finetune_template_v1.yaml"), chosen_parents)
    _count = check_finetune_sweep(finetune, BASE_CONFIG, chosen_parents)
    _path = OUTPUT / "tutorial_finetune_ko_v1.yaml"
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(yaml.safe_dump(finetune, sort_keys=False))
    # Show the parent block with long values shortened
    _block = finetune["groups"][-1]
    _rows = [
        {"setting": _key, "per parent": ", ".join(str(_v)[:40] for _v in _values)}
        for _key, _values in _block.items()
    ]
    with tempfile.TemporaryDirectory() as _folder:
        _manifest = generate_manifest(load_yaml(BASE_CONFIG), finetune, Path(_folder))
    finetune_configs = [load_experiment_config(_entry["config"]) for _entry in _manifest["configs"]]
    mo.vstack([
        mo.md(f"Every one of the {_count} runs fits its parent. Written to `{relative(_path)}`. The parent block:"),
        pd.DataFrame(_rows),
        mo.md("The runs:"),
        run_table(finetune_configs),
        mo.md(
            "On the real workflow: write the sweep into `Experiments/micropatterns/conf/experiments/`, "
            "commit it, generate its manifest from that one file, and launch it like the baseline."
        ),
    ])
    return (finetune_configs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What the check catches

    - `check_parent_compatible(parent_config, child_config)` lists every difference that makes the parent's weights meaningless for the child. `Experiments/micropatterns/train.py` calls it before loading any data, and stops the run if the list is not empty
    - Below, the first fine-tune run is changed the way people often get it wrong: a different replicate split, a different `run.t`, and a missing stripe filter
    """)
    return


@app.cell
def _(chosen_parents, finetune_configs):
    _child = finetune_configs[0]
    _parent = next(_p for _p in chosen_parents if _p.id == _child.initialization.model_id)
    _micropattern = _child.data.micropattern
    _wrong = replace(
        _child,
        run=replace(_child.run, t=2 * _child.run.t),
        data=replace(
            _child.data,
            micropattern=replace(
                _micropattern,
                train_replicates=_micropattern.validation_replicates,
                validation_replicates=_micropattern.train_replicates,
                cleaning=replace(_micropattern.cleaning, horizontal_stripe_widths={}, stripe_block=0),
            ),
        ),
    )
    mo.vstack([
        mo.md(f"As written: `{check_parent_compatible(_parent.config, _child)}`"),
        mo.md("Changed:\n\n" + "\n".join(f"- {_problem}" for _problem in check_parent_compatible(_parent.config, _wrong))),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. The curriculum: which data each batch holds

    - `data.knockout.curriculum` lists the conditions in each training batch, e.g. `[baseline, ko_0h, ko_24h]` (names in `Experiments/micropatterns/config_helpers.py:CURRICULUM_CONDITIONS`). Each entry brings `data.batches` colonies
    - Keep `baseline` in the fine-tune curriculum, or the model may forget it
    - Predictions are only matched to targets, and replicates only swapped, within one condition (one **condition id** per batch item)
    - `data.knockout.curriculum_weights` sets each condition's share of the training loss, e.g. `{baseline: 4, ko_0h: 1, ko_24h: 1}`. Weights are cheaper than repeating an entry in the curriculum: the loss compares every ordering of the colonies of one condition, so repeats get slow quickly
    - New experiments (e.g. densities) become new curriculum names; see `docs/finetune_ablation.md`
    """)
    return


@app.cell
def _():
    from Experiments.micropatterns.config_helpers import curriculum_condition_ids

    _curriculum = ("baseline", "ko_0h", "ko_24h")
    _weights = {"baseline": 4.0, "ko_0h": 1.0, "ko_24h": 1.0}
    _ids = curriculum_condition_ids(_curriculum, 2)
    pd.DataFrame({
        "batch item": range(len(_ids)),
        "condition": [_curriculum[_i] for _i in _ids],
        "condition id": _ids,
        "loss weight": [_weights[_curriculum[_i]] for _i in _ids],
    })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Evaluate the parent and the child

    - Each model is rolled out on every condition from the 0h image of each **held-out** replicate, and scored against the measured images (`Experiments/micropatterns/finetune_ablation.py`):
        - **ring profile error**: squared error of the channel averages in rings from the centre to the edge
        - **pixel error**: squared error per pixel inside the colony
    - The same scores for every model, so a parent and its children (or the arms of the ablation) can be compared. W&B validation losses can't be: they depend on each run's curriculum and weights
    - The defaults are a real parent (the example selection) and one of its fine-tunes from `nca_nodal_ko_projections_finetune_v3_sum`. Needs the model store and the data; about a minute on a GPU
    - Deeper analysis (cell types, knockout timing, density): `Experiments/micropatterns/nodal_ko_eval.py`
    """)
    return


@app.cell
def _():
    eval_parent = mo.ui.text("20261027T143308Z-cfgb505b3517573-a92c347a", label="Parent", full_width=True)
    eval_child = mo.ui.text("20261103T130746Z-cfg25ddc91060ef-ed10f82d", label="Fine-tuned child", full_width=True)
    eval_data = mo.ui.text(os.environ.get("DATA_PATH_BASE", ""), label="Data root (holds 260726_nca_dataset)", full_width=True)
    eval_button = mo.ui.run_button(label="Evaluate both")
    mo.vstack([eval_parent, eval_child, eval_data, eval_button])
    return eval_button, eval_child, eval_data, eval_parent


@app.cell
def _(eval_button, eval_child, eval_data, eval_parent, store_root):
    mo.stop(not eval_button.value)
    from Experiments.micropatterns.finetune_ablation import (
        AblationEvaluationSettings,
        evaluate_models,
        results_table,
    )

    _data = Path(eval_data.value).expanduser()
    mo.stop(
        not (_data / "260726_nca_dataset").is_dir(),
        mo.callout(f"No `260726_nca_dataset` in `{_data}`. Set the data root.", kind="warn"),
    )
    _settings = AblationEvaluationSettings(
        data_root=str(_data), store_root=store_root.value, seeds=(0,)
    )
    _ids = [eval_parent.value.strip(), eval_child.value.strip()]
    with mo.status.spinner("Rolling out both models on every condition"):
        evaluation = results_table(evaluate_models(_settings, _ids))
    evaluation["model"] = np.where(evaluation["model_id"] == _ids[0], "parent", "child")
    return (evaluation,)


@app.cell
def _(evaluation):
    _summary = evaluation.groupby(["condition", "model"])["radial_mse"].mean().unstack("model")
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _summary[["parent", "child"]].plot.bar(ax=_ax, rot=0)
    _ax.set_ylabel("Ring profile error")
    _ax.set_xlabel("")
    _fig.tight_layout()
    mo.vstack([
        mo.md("Mean over held-out replicates and time points (lower is better):"),
        _summary,
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pitfalls

    - **Manifests in `conf/generated/` are records** of how models were trained. Don't regenerate or edit them. To change a sweep that has run, bump `_vN` in both `experiment_name` and `output_subdir`
    - **Generate from one sweep file**, never a folder
    - **Templates are not sweeps**: they live in `conf/experiments/finetune_templates/` (or `demo/conf/`) so folder tools skip them. Generating one by mistake stops at `FROM_PARENTS`
    - **Write the fine-tune sweep again** when the parents change; don't edit its `groups:` block by hand
    - **Bundles are never edited.** The parent link is the child's `initialization.model_id`
    - **"Pretrained model ... does not fit this run"**: the message names each differing setting. Usually the base config changed after the parent was trained; take that setting from the parent
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where next

    - The full micropattern workflow and troubleshooting: `docs/workflow_micropatterns.md`
    - Is fine-tuning better than joint training? `docs/finetune_ablation.md` and `Experiments/micropatterns/finetune_ablation_explorer.py`
    - Knockout analysis: `Experiments/micropatterns/nodal_ko_eval.py`
    - Running notebook work on the cluster: `docs/remote_notebook_jobs.md`
    """)
    return


if __name__ == "__main__":
    app.run()
