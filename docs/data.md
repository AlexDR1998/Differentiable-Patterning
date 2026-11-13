# Data

Datasets are not in git. They live in one folder, `DATA_PATH_BASE` (set in
`.env`, see `.env.example`). On the EIDF cluster that folder is
`/workspace/writeable/Data/` inside every pod (on the `$USER-ws1` volume; see
[cluster_eidf.md](cluster_eidf.md)). Only a few example emojis are included
in the repository (`demo/demo_data/Emojis/`), for the tutorials.

```text
$DATA_PATH_BASE/
  260726_nca_dataset/          current micropattern data (data.dataset: micropatterns_260726)
  260726_nca_segmentation/     stored Cellpose nucleus segmentation of it
  Emojis/                      emoji PNGs (data.dataset: emojis)
  snowmelt/v1/, snowmelt/v2/   snowmelt maps (data.dataset: snowmelt)
  Timecourse_seperate_colonies/  older micropattern data (data.dataset: micropatterns)
  cache/                       optional, local only: cleaned 260726 images (DATA_CACHE_DIR)
```

PDE and signalling data are simulated when a run starts, so they need no files.

Every loader returns `[batch, time, channels, height, width]`; see
[concepts.md](concepts.md#data-as-trajectories).

## Micropatterns: the 260726 dataset

Human stem-cell colonies grown on circular micropatterns, fixed and stained
at 0, 12, 24, 36 and 48 h. Each image is a different colony, so a
"trajectory" is built from different colonies at each time (a
**replicate**). The dataset's own description, with links to the lab
notebook entries, is `260726_nca_dataset/NCA_FULL_README.md`, and there is a
README in each subfolder.

**Conditions** (`data.knockout.curriculum` picks which are loaded):
- `ctrl`: control, no perturbation;
- `sl0`: Nodal signalling inhibited from 0 h;
- `sl24`: Nodal signalling inhibited from 24 h.

These are `baseline`, `ko_0h` and `ko_24h` in the config.

**Staining panels** (`data.micropattern.experiment_groups`): each colony was
stained for only a few markers, so the data has four panels, each from
different colonies:

| Panel | Folder | Channels |
| --- | --- | --- |
| `cell_fate_s1` | `cell_fate_markers/<condition>_s1_<projection>` | SOX17, SOX2, TBXT, LMBR |
| `cell_fate_s2` | `cell_fate_markers/<condition>_s2_<projection>` | SOX17, FOXA2, TBXT, LMBR |
| `rna_expression` | `signalling/rna_expression/<condition>` | CER1, LEFTY, NODAL (RNA FISH) |
| `protein_response` | `signalling/protein_response/<condition>_<projection>` | LEF1, LMBR, SMAD23 |

So there are 14 **measured channels**, and 10 different markers:
LMBR, TBXT, SOX17, SOX2, FOXA2, CER1, LEFTY, NODAL, LEF1 and SMAD23.
- The NCA state has one channel per marker: these 10 observed channels come
  first, then the hidden channels.
- Losses compare each measured channel with the matching state channel, panel
  by panel. A marker measured in two panels (e.g. TBXT) is compared twice.
- `data.micropattern.excluded_measurements` drops measured channels. The base
  config drops `cell_fate_s2/SOX17`, leaving 13.

All of this is written down in
`Common/dataloader/micropattern_schemas.py:MICROPATTERN_260726_SCHEMA`; code
should look channels up there by name rather than by number.

**Projections** (`data.micropattern.projection`): the colonies were imaged in
3D, and the 3D stack is turned into 2D as a `max` projection, a `sum`
projection (folders end in `_sums`) or a single `slice` (`_slices`). The RNA
panel was imaged in 2D only.

**Cleaning.** Raw images are cleaned when they are loaded: intensity
corrections, hot pixels, stripes from tile stitching, background, colony
alignment and rescaling to [0, 1]. The settings are tuned in the notebook
`Experiments/micropatterns/data_cleaning.py` and copied into
`data.micropattern.cleaning`. Each step is explained in
[configuration.md](configuration.md#260726-micropattern-pre-processing).
- The colony centres come from
  `Experiments/micropatterns/conf/alignment/260726_alignment.yaml`.
- Images flagged as bad are listed in `Experiments/micropatterns/conf/quality/`.
- Cell-type thresholds come from the files in
  `Experiments/micropatterns/conf/cell_types/`.
- Both are tracked in git, because training depends on them.

**Resolution** (`data.downsample`): the images are block-averaged by this
factor. The paper models use 8, 16 and 32, with `run.t` scaled to match
(`run.t = 128 × 16 / downsample`).

**Nucleus segmentation.** `260726_nca_segmentation/` holds Cellpose nucleus
labels of the 0 h images. They are used to count cells and to build synthetic
initial conditions of a chosen cell density (`Experiments/micropatterns/nucleus_segmentation.py`,
`size_density_explorer.py`). Cellpose needs its own environment
(`requirements/requirements_segmentation.txt`); the stored labels don't.

**Local cache.** Cleaning the images takes a while. With `DATA_CACHE_DIR` set,
local notebooks keep the cleaned images on disk and reuse them
(`Common/dataloader/disk_cache.py`). Delete the cache folder after changing
the cleaning code.

## Older micropattern data

`data.dataset: micropatterns` reads `Timecourse_seperate_colonies/` with
`Common/dataloader/micropattern.py`: 4-channel colonies, and a 9-channel
Nodal knockout set. It is kept for the thesis models and old bundles. New
work should use `micropatterns_260726`. This loader joins paths as plain
strings, so `DATA_PATH_BASE` must end in `/`.

## Emojis

`Emojis/` holds RGBA PNG images (for example from an emoji set). A sweep
names the files to use (`data.emoji.sequence` or `data.emoji.pairs`). The
`sequence` task grows one image after another; `multi_attractor` grows each
target from its own starting patch. See the tutorial
`demo/nca_training_config_walkthrough.py`.

## Snowmelt

Satellite maps of snow cover in a catchment, as dates through a melt season.
`v1` has 5 dates (May–July) and `v2` has 10 dates (April–September).
- **Target:** the NCA reproduces the dynamic channel `SCA` (snow-covered
  area).
- **Static inputs:** fixed maps such as `DEM` (elevation) are written into
  the last state channels every step.

The settings are in `Experiments/snowmelt/config.py`, and the loader is
`Common/dataloader/snowmelt.py`. The root folder is `data.snowmelt.root`,
else `SNOWMELT_DATA_ROOT`, else `$DATA_PATH_BASE/snowmelt`.

## PDEs and signalling tasks

- **PDEs:** `data.dataset: pde` simulates one of the PDEs in
  `PDE/catalogue.py` when the run starts. See
  [configuration.md](configuration.md#pde-data) and `demo/pde_patterns_demo.py`.
- **Signalling tasks:** `data.dataset: signal_tasks` builds procedural tasks
  for the signalling NCAs. See
  [configuration.md](configuration.md#signalling-task-data).

## Checking data against a trained model

A bundle does not contain its training data. Instead it stores fingerprints
(hashes and shapes) of the initial conditions. Before evaluating a model,
reload its data with its own config and call `verify_evaluation_input`. This
fails if the data or the cleaning has changed since training. See
[model_registry.md](model_registry.md#checking-evaluation-inputs).
