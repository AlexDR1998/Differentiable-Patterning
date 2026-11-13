# Notebooks

All notebooks are [marimo](https://marimo.io) notebooks: plain `.py` files.
Open one with `marimo edit <file>` from the repository root. Most need some
of the following, set in `.env` or typed into the notebook:
- **data:** `DATA_PATH_BASE`;
- **store:** the model store, `MODEL_STORE_ROOT`;
- **selection:** a selection file exported from `model_registry_explorer.py`.

Status:
- **maintained:** used recently and expected to work with the current code;
- **legacy:** kept for reference; it needs old model files or data layouts
  and may not run.

## Tutorials (`demo/`)

See [demo/README.md](../demo/README.md) for the order. They need only the
files in the repository.

## Models

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `Experiments/model_registry_explorer.py` | Search, filter and compare trained bundles; export a **selection** of models. | store | maintained |

## Micropatterns (`Experiments/micropatterns/`)

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `data_cleaning.py` | Tune the cleaning of the 260726 images; exports the cleaning settings, alignment and cell-type files. Very large. | data | maintained |
| `nodal_ko_eval.py` | Evaluate a selection under Nodal knockouts: cell-type prevalence, knockout time and density sweeps; exports data for `overview_figure.py`. | data, store, selection | maintained |
| `size_density_explorer.py` | Sweep colony size × cell density; radial kymographs. | data, store, selection | maintained |
| `ablation_comparison.py` | Compare the ablation sweep runs from their W&B logs and by running the bundles. | W&B, store, data | maintained |
| `augmenter_explorer.py` | See what the micropattern augmenter does to the training data. | data | maintained |
| `boundary_mask_explorer.py` | Boundary masks and initial conditions for colonies. | data | maintained |
| `drawn_mask_explorer.py` | Draw a colony shape and run a trained NCA on it. | data, store | maintained |
| `synthetic_textured_ic_explorer.py` | Build synthetic initial conditions from measured 0 h colonies. | data | maintained |
| `cell_census_explorer.py` | Cell numbers and types of the measured colonies over time. | data | maintained |
| `nucleus_segmentation_explorer.py` | Ground truth and scores for the nucleus segmentation. | data | maintained |
| `figures/*.py` | Paper figures; see [figures.md](figures.md). | varies | maintained |
| `thesis_chapter_2_figures.py` | Thesis figures. Loads old `.eqx` files by file name. | old models | legacy |
| `nodal_knockout_dev.py` | Early development notebook for knockouts. | old data | legacy |

## Emoji (`Experiments/emoji/`)

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `multi_attractor_explorer.py` | Explore emoji NCAs trained to grow several targets. | store | maintained |
| `adaptive_firing_explorer.py` | Explore NCAs with a learned firing rate (aNCA). | store | maintained |
| `thesis_chapter_1_figures.py` | Thesis figures. Loads old `.eqx` files by file name. | old models | legacy |

## Impulses (`Experiments/impulse/`)

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `impulse_explorer.py` | Look at optimised impulses on an emoji NCA. | store | maintained |
| `impulse_sweep_explorer.py` | Compare impulse sweep runs. | store | maintained |

## Snowmelt (`Experiments/snowmelt/`)

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `snowmelt_explorer.py` | Explore the snowmelt dataset. | data | maintained |
| `snowmelt_sweep_explorer.py` | Compare the snowmelt models of each sweep. | data, store | maintained |

## Other

| Notebook | What it is for | Needs | Status |
| --- | --- | --- | --- |
| `Experiments/dataset_explorer.py` | Look at any dataset in the repository. Partly uses the older loaders. | data | maintained |
| `WebDemo/marimo_app.py` | Web page for the WebGL viewer. Unfinished. | exported model | legacy |

## Writing a notebook

- Each global name can be defined in only one cell. Prefix cell-local names
  with `_`.
- Keep slow work behind a run button.
- Move heavy loops into a plain module, so they can also run on the cluster
  ([remote_notebook_jobs.md](remote_notebook_jobs.md)).
- Figure notebooks take colours and fonts from
  `Experiments/micropatterns/figures/style.py`.
- `marimo check <file>` finds structural mistakes without running the
  notebook.
