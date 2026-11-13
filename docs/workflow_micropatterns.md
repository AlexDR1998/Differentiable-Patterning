# Workflow: from a sweep to a figure (micropatterns)

The path a real micropattern experiment takes, step by step. Other tasks
(emoji, snowmelt, ...) follow steps 1–6 in the same way with their own
`Experiments/<task>/` folder.

```text
1 sweep file ─► 2 manifest ─► 3 check ─► 4 sync to EIDF ─► 5 launch ─► 6 watch on W&B
                                                                          │
9 figure ◄── 8 evaluate (nodal_ko_eval.py) ◄── 7 copy bundles back, pick models
```

Before starting:
- set up your machine ([../README.md](../README.md));
- get EIDF access ([cluster_eidf.md](cluster_eidf.md));
- know roughly what the settings mean ([concepts.md](concepts.md),
  [losses.md](losses.md)).

## 1. Write a sweep file

Sweep files are in `Experiments/micropatterns/conf/experiments/`. They
change settings of `Experiments/micropatterns/conf/base_config.yaml`, using
the same names (`run.t` is `run: t:` in the base config). Start by copying
the sweep closest to what you want. `micropattern_projections_training_sweep.yaml`
trains the wild-type paper models, and `nca_nodal_ko_projections_finetune.yaml`
fine-tunes them on knockouts.

```yaml
experiment_name: nca_my_question_v1      # also the W&B group and the bundle folder
entrypoint: Experiments.micropatterns.train:run
output_subdir: generated/nca_my_question_v1
seed_mode: index                         # each run gets its own seed (its index)

grid:                     # every combination of these lists is one run
  run.repeat: [0, 1, 2]   # repeats of the same settings
  model.family: [gNCA]
  model.channels: [32, 48]
  logging.wandb.project: [nca_micropattern_my_question]

groups:                   # these lists are zipped together, not combined
  - data.downsample: [8, 16, 32]
    run.t: [256, 128, 64]
```

This gives 3 × 2 × 3 = 18 runs.

**Versions.** To change a sweep after it has run, don't edit it in place.
Bump the version in both `experiment_name` and `output_subdir`
(`_v1` → `_v2`), and say what changed in a comment at the top. The bundles,
W&B groups and tracked manifests of the old version keep their name, so the
models in a figure can always be traced back to their settings.

## 2. Generate the manifest

```bash
python Experiments/generate_configs.py \
    Experiments/micropatterns/conf/experiments/nca_my_question_v1.yaml \
    Experiments/micropatterns/conf/base_config.yaml \
    Experiments/micropatterns/conf/generated
```

This writes `Experiments/micropatterns/conf/generated/nca_my_question_v1/manifest.yaml`,
holding the full config of every run.

> **Trap:** always give **one sweep file**. If you give the folder
> (`Experiments/micropatterns`), every sweep in it is regenerated, and the
> tracked manifests of earlier experiments are overwritten with whatever
> their sweep files say *now*. Those manifests are often the only record of
> how models in the registry were trained. If that happens, restore them with
> `git checkout -- Experiments/*/conf/generated`.

## 3. Check the manifest

Configs are checked strictly (unknown keys, wrong types, impossible values)
when they are loaded. Do that on your laptop rather than finding out on the
cluster:

```bash
python -c "
import sys
from omegaconf import OmegaConf
from Experiments.config import load_experiment_config
manifest = OmegaConf.load(sys.argv[1])
for entry in manifest.configs:
    load_experiment_config(entry.config)
print('All', len(manifest.configs), 'configs are valid')
" Experiments/micropatterns/conf/generated/nca_my_question_v1/manifest.yaml
```

This takes about a second and needs no GPU. A typo in a setting name shows
up here as an error naming the setting.

## 4. Sync the code to EIDF

Pods run the code in your NFS checkout on EIDF. They do not use your laptop's
copy. Commit the sweep file and the generated manifest, and update the
checkout on EIDF (see [cluster_eidf.md](cluster_eidf.md#where-things-are)).
Manifests are tracked in git on purpose: they record exactly what ran.

## 5. Launch

On the EIDF VM, from the repository root:

```bash
bash launch/launch_batch_multi_job.sh Experiments/run_config.py \
    Experiments/micropatterns/conf/generated/nca_my_question_v1/manifest.yaml 6 H200
```

This runs 6 runs at once on H200 GPUs. To estimate how long it will take,
look at earlier runs of a similar sweep on W&B.

## 6. Watch

- **W&B:** the project from `logging.wandb.project`, grouped by experiment
  name. It shows losses, validation, and images and videos of rollouts.
- **Kubernetes:** `kubectl get pods`, `kubectl logs -f <pod>`
  ([cluster_eidf.md](cluster_eidf.md#watching-jobs)).

Each run saves its best checkpoint as a **bundle** in the shared model store
on EIDF when it finishes (`model_store.enabled: true` in the base config).

## 7. Copy models back and pick the ones to use

```bash
helper_scripts/copy_models_from_cluster.sh --port <port>
```

This copies the new bundles into `models/` (or `MODEL_STORE_ROOT`) and
rebuilds the registry index. Then:

```bash
marimo edit Experiments/model_registry_explorer.py
```

1. Filter by experiment, W&B tags, loss and so on.
2. Tick the models you want.
3. Download the **selection** (`model_registry_selection.yaml`).

The download goes to your browser's download folder. Give it a clear name and
keep it with the project, since figures and evaluations are made from it. We
suggest `Experiments/micropatterns/figures/selections/<name>.yaml`, committed
to git, so each figure can be traced back to its models.

To use the models in your own code, see
[model_registry.md](model_registry.md) and the tutorial
`demo/bundles_and_inference.py`.

## 8. Evaluate

`marimo edit Experiments/micropatterns/nodal_ko_eval.py` loads a selection
and rolls the models out under the control and knockout conditions. It
compares cell-type prevalence with the data and sweeps knockout time and
cell density.
- **Heavy sections:** some sections take hours locally. They can be exported
  as cluster jobs ("Export remote job"), and the results loaded back. See
  [remote_notebook_jobs.md](remote_notebook_jobs.md).
- **Exports:** buttons such as "Save for overview figure" write CSV and npz
  files into `figures/overview/`, which the figure notebooks read.

To record a scalar evaluation result next to the model, use
`record_evaluation` ([model_registry.md](model_registry.md#recording-evaluations)).

## 9. Make the figure

The figure notebooks are in `Experiments/micropatterns/figures/`
([figures.md](figures.md)).
- Each one takes a selection file and the data and model folders from `.env`.
- Some also read the exports from step 8. `overview_figure.py` needs
  `nodal_ko_eval.py` to have saved its data into `figures/overview/` first.
- Figures are saved to `figures/<folder>/` as PDF and PNG. The `figures/`
  folder is not in git.

---

## Worked example: fine-tuning on Nodal knockouts

Knockout models start from a trained wild-type model (the **parent**) and are
trained further on control plus knockout data.
`Experiments/micropatterns/conf/experiments/nca_nodal_ko_projections_finetune.yaml`
does this:

```yaml
data.knockout.channel: [Nodal]
data.knockout.curriculum:
  - [baseline, ko_0h, ko_24h]   # control, Nodal inhibited from 0 h and from 24 h

groups:
  - initialization.model_id:     # one parent per row
      - 20261027T212939Z-cfg48df3cfa8558-ac5a0126
      - ...
    data.downsample: [8, 16, 32] # must match each parent
    data.micropattern.train_replicates: [[3, 4], [1, 2], [3, 4]]  # parent's split
```

- **Parent:** `initialization.model_id` names the parent bundle, which pods
  load from the shared store. Find IDs in `model_registry_explorer.py`.
- **Matching settings:** the run stops early if the parent's projection, data
  cleaning or model settings differ from the fine-tuning config.
- **Same replicate split:** keep the parent's train/validation split, so
  validation colonies stay unseen.
- **Paper models:** the sweep file is now at `v3_sum`. Older versions
  survive as tracked manifests in `conf/generated/`
  (`nca_nodal_ko_projections_finetune/` holds v2).
  **TODO(handover):** which experiment versions the paper's knockout models
  come from.

## Things that go wrong

| Symptom | Likely cause |
| --- | --- |
| Error on conversion naming a setting | Typo in a dotted key, or a setting that moved. Check `docs/configuration.md` and the dataclasses. |
| `DATA_PATH_BASE must be set` | `.env` missing locally, or running outside a pod without the variable. |
| Pod `Pending` for a long time | No free GPUs of that type; try another type, or wait. `kubectl describe pod` shows why. |
| Pod `OOMKilled` or CUDA out of memory | Batch, resolution or `run.t` too large. Try `trainer.loop_autodiff: checkpointed` (slower, less memory) or a smaller batch. |
| Run stops: parent settings differ | The fine-tuning config does not match the parent bundle (projection, cleaning, channels). |
| `verify_evaluation_input` fails | The data or cleaning changed since the model was trained. |
| A notebook is very slow the first time | JAX compiles each new shape once. The second run is fast. |

More pitfalls: [tips_and_pitfalls.md](tips_and_pitfalls.md).
