# Fine-tuning vs joint training

Experiments with little data (Nodal knockouts now, colonies seeded at
different cell densities soon) can be learned in two ways:

- **staged**: train a baseline model on the control colonies, then fine-tune
  it on all conditions (`initialization.model_id`);
- **joint**: train one model on all conditions from the start.

This ablation asks which works better, and what happens in between. The
tutorial `demo/finetuning_workflow.py` walks through the staged workflow
itself.

## Design

Every model gets the same budget of **12000 iterations** in total (parent plus
fine-tune), and every iteration costs the same: each batch always holds
`[baseline, ko_0h, ko_24h]`. Two settings vary.

| Setting | Meaning | Values |
| --- | --- | --- |
| **f**, pretraining share | Share of the budget spent training on baseline only (the parent). The rest trains on all conditions. | 0 (joint), 0.25, 0.5, 0.75, 0.9; and 1 (baseline only, as a reference) |
| **s**, baseline share | Share of the loss weight on the baseline condition while training on all conditions (`data.knockout.curriculum_weights`, knockout weights 1 each). | 0 (`baseline: 0`), 1/3 (`baseline: 1`), 2/3 (`baseline: 4`) |

The other settings are fixed: sum projection with the stripe filter, ds16,
`run.t` 128, gNCA 48 channels, and the loss of
`nca_nodal_ko_projections_finetune.yaml`. Both replicate splits and 2 repeats
are used.

- The loss weights replace repeating entries in the curriculum. The
  multi-target loss compares every ordering of the colonies of one
  condition, so repeating an entry gets slow very quickly (8 colonies have
  40,320 orderings).
- Baseline-only pretraining also loads the knockout conditions, with weight
  0. That keeps the cost per iteration equal, and the parent reads the data
  exactly as its children will.
- The fine-tune stage starts a new optimiser, warmup and loss schedule (the
  schedules are fractions of each stage's own iterations). That is what
  fine-tuning means here.

## Running it

Sweep files are in `Experiments/micropatterns/conf/experiments/`. The
manifests of the first two are in `conf/generated/`.

| Step | File | Runs |
| --- | --- | --- |
| 1. Parents: baseline only for 3000, 6000, 9000, 10800 iterations | `nca_finetune_ablation_pretrain_v1.yaml` | 16 |
| 1. Joint from scratch, 12000 iterations, at each s, plus the baseline-only reference | `nca_finetune_ablation_scratch_v1.yaml` | 16 |
| 2. Fine-tunes of every parent at each s, for 12000 minus the parent's iterations | written from `finetune_templates/nca_finetune_ablation_finetune_v1.yaml` | 48 |
| 3. Evaluation | `Experiments/micropatterns/finetune_ablation_explorer.py` | one pod per model |

1. Launch steps 1 together (`docs/cluster_eidf.md`):
   ```bash
   bash launch/launch_batch_multi_job.sh Experiments/run_config.py \
       Experiments/micropatterns/conf/generated/nca_finetune_ablation_pretrain_v1/manifest.yaml 8 H200
   bash launch/launch_batch_multi_job.sh Experiments/run_config.py \
       Experiments/micropatterns/conf/generated/nca_finetune_ablation_scratch_v1/manifest.yaml 8 H200
   ```
2. Copy the parents back, select all 16 in `model_registry_explorer.py`, and
   write the fine-tune sweep (it checks every run against its parent):
   ```bash
   python -m Experiments.micropatterns.finetune \
       Experiments/micropatterns/conf/experiments/finetune_templates/nca_finetune_ablation_finetune_v1.yaml \
       --selection <selection of the 16 parents> --total-iterations 12000 \
       --out Experiments/micropatterns/conf/experiments/nca_finetune_ablation_finetune_v1.yaml
   python Experiments/generate_configs.py \
       Experiments/micropatterns/conf/experiments/nca_finetune_ablation_finetune_v1.yaml \
       Experiments/micropatterns/conf/base_config.yaml Experiments/micropatterns/conf/generated
   ```
   Commit both, then launch the manifest as in step 1.
3. Select all 80 models (parents, joint models and fine-tunes), open
   `finetune_ablation_explorer.py`, export the evaluation job, launch it with
   `Experiments/micropatterns/finetune_ablation.py`, copy the results back
   and load them.

## Evaluation

W&B validation losses depend on each run's curriculum and weights, so they
can't be compared across arms. `Experiments/micropatterns/finetune_ablation.py`
scores every model the same way. Each model is rolled out on every condition
from the 0 h image of each of its **held-out** replicates, two rollouts each.
It is scored against the measured images on:

- `radial_mse`: ring-averaged profiles, centre to edge;
- `pixel_mse`: per pixel inside the colony;
- `share_error` (optional): cell type shares at 48 h. This needs a rule file
  without left-out channels; the current ones still use
  `cell_fate_s2/SOX17`.

Every result row records the model's f (`pretrain_fraction`, read from its
parent) and s (`baseline_share`). The explorer shows heatmaps of f × s per
condition, and every model as a point.

## The 250K density data

`$DATA_PATH_BASE/density_nodal_ko_finetune/` (README inside; lab log 554):

- 12 images: `{MAX,SUM,SLICE}_KO{24,28,32,36}.tif`, plated at 250K cells
  instead of 150K. Each is one colony, fixed at 48 h, with Nodal inhibited
  from the hour in the file name.
- Each holds one 4-channel cell-fate panel: SOX17, TBXT, FOXA2, LMBR, in that
  order. These are the markers of `cell_fate_s2`, in a different order. Current
  models leave out `cell_fate_s2/SOX17`, so that channel goes unused.
- There are no 0 h images, no controls, and no RNA or protein panels. More
  images may come; the plan does not rely on them.
- The pixel size is 0.581 µm (260726: 0.596 µm), so the images need
  resampling to the 260726 scale.

### What is in place

- **Knockouts part-way through a slot.** Training and validation block NODAL
  from the first step at or after the knockout hour
  (`NCA/trainer/intervention.py:nodal_read_block_mask` with `step`), so KO28
  and KO32 differ from KO24 inside the 24-36 h slot. Knockouts on a 12 h
  boundary block exactly as before. `ko_evaluation.evaluate_condition` takes
  any `knockout_hour`, with the same rule (`knockout_step`).
- **Condition ids, loss weights per condition, the fine-tune sweep writer
  and parent checks** (above).
- **Sparse targets.** The measurement mask leaves unmeasured channels and
  times out of the loss and of reinjection. Here only 48 h of the
  `cell_fate_s2` panel is measured.

### First: a held-out test, no training

Before any fine-tuning, the 250K colonies test the density × KO sweep of
`nodal_ko_eval.py` ("Held-out test: 250K plating", after the sweep's
heatmaps). Module: `Experiments/micropatterns/density_test_data.py`.

- Each colony is read, put in `cell_fate_s2` channel order, resampled to the
  260726 pixel size, cleaned with a sweep model's settings, and centred on a
  fitted circle. It is placed at 250/150 times the measured 150K density
  and at its knockout hour.
- Cell types use only the clauses SOX17, TBXT and FOXA2 can check, for the
  data and the model alike. Markers whose threshold was tuned on another
  panel (e.g. SOX17 from `cell_fate_s1` in the slice rules) are listed.
- **Tune the cell type rules and normalisation first** in
  `data_cleaning.py` ("250K colonies beside the 150K colonies", after "Cell
  types at 48h"). It shows the 250K colonies next to chosen 150K control,
  KO 0 h and KO 24 h colonies at 48 h, with the notebook's live cleaning,
  percentile and threshold controls and the shared cell type colours.
  The 250K colonies have their own brightness factors (per channel and
  colony), hot pixel thresholds and background radii there, with a preview
  of each cleaning step. These settings are not saved yet: note the values
  you settle on, so the fine-tuning loader can use them.
- **Thresholds for 150K and 250K together** (`nodal_ko_eval.py`, after the
  held-out heatmaps). The sweep keeps only counts made with the cell type
  file's thresholds, so the models are rolled out again at the points with
  data (150K: baseline, KO 0 h, KO 24 h; 250/150 × 150K: the 250K KO times)
  and their 48 h states kept. One set of threshold sliders then scores model,
  150K and 250K alike, with the mean NCA − data difference for each density.
  Tuned thresholds can be saved as a new cell type file; rerun the sweep with
  it to redraw the heatmaps.
- **Normalisation** is a choice, since there is no 0 h image: joint over
  the four knockout times, per image, or the joint range matched to the 150K
  KO 24 h colonies at 48 h. Compare them on the images, the cell type maps
  and the shares before choosing one for fine-tuning.
- The sweep needs cells per mm² or per colony, and KO times that include
  28 and 32 h (every 2 or 4 h); otherwise predictions are interpolated.

### Still to do

1. **Loader.** Read the folder; reorder the channels onto the
   `cell_fate_s2/*` measurement names; resample to the 260726 pixel size;
   clean (hot pixels, background, the stripe filter for sum TBXT) and centre
   (alignment entries for the new images, from `data_cleaning.py`).
   **Normalisation (decision needed):** each colony is normalised on its own
   today, which with one image per knockout time would stretch every time to
   the same range. The plan is to normalise the four images of each
   projection together.
2. **Conditions.** Turn `CURRICULUM_CONDITIONS` into a small table of
   (source, knockout hour, density, initial state) and add `d250_ko24`,
   `d250_ko28`, `d250_ko32` and `d250_ko36`.
3. **Synthetic initial states.** No 250K 0 h image exists, so each 250K
   training item starts from a state built at load time with
   `Experiments/micropatterns/synthetic_ic.py:cell_states`. Its cells come from
   the training replicates' 150K 0 h images
   (`density_ko_sweep.prepare_model_cells`), at **250/150 times** their
   measured cells per mm² (or per colony). Several copies with different
   seeds give more than one item per condition. The pods need the stored
   nucleus segmentation.
4. **No density input to the model.** The density is visible in the initial
   state.
5. **Evaluation.** With one colony per knockout time there are no held-out
   replicates. Hold out one knockout time instead (train on three, score the
   48 h panel of the fourth, four folds), with several synthetic initial
   states per score.
6. **Experiments, per projection (max, sum, slice).** Fine-tune the 150K
   knockout model on `[baseline, ko_0h, ko_24h, d250_ko*]`, varying the loss
   weight of the 250K items, against joint training from scratch on the same
   data. Each projection has its own parents; the parent check enforces this.
   The f × s grid above stays for the 150K knockout data; four images are too
   few for it.
