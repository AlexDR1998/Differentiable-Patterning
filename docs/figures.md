# Micropattern paper figures

All figures for the micropattern paper share one style. The code lives in
`Experiments/micropatterns/figures/`:

| File | Contents |
| --- | --- |
| `style.py` | The style config: page widths, font sizes per text role, line widths, colours (page, data/model/accent, line series, cell types, channels, groups, conditions), colour maps, labels, and helpers (`paper_style`, `new_figure`, `label_panel`, `legend_options`, `save_figure`). |
| `paper.mplstyle` | The matplotlib settings that `paper_style()` applies. |
| `drawing.py` | Shared drawing: colony discs and sectors, cards, arrows, time axes, the 3D stack, the trajectory figure, and `stack_panels` for stacked schematic panels. |
| `data.py` | Shared data loading: data configs, `load_dataset`/`load_colonies`, model rollouts, pickle caching, cell-type images. |
| `toy.py` | The toy patterning used by the method schematics. |
| `*.py` notebooks | One marimo notebook per figure (see below). |

## Making a figure

```python
from Experiments.micropatterns.figures.style import WIDTH, label_panel, paper_style, save_figure

with paper_style():
    fig = plt.figure(figsize=(WIDTH["single"], 2.0))
    ax = fig.add_subplot()
    ...
    label_panel(ax, "a", "Title", "subtitle")
save_figure(fig, "my_figure", "schematics")  # figures/schematics/my_figure.pdf and .png
```

- Draw inside `with paper_style():` so fonts, line widths and the colour cycle match.
- Size figures with `WIDTH["single"]` (89 mm) or `WIDTH["double"]` (183 mm).
- Don't write colours, font sizes or colour maps in notebooks; take them from `style.py`. To change the look of every figure, change `style.py`.
- Notebooks keep only widgets that choose content (models, groups, panels, zoom), not colours or fonts.
- Fluorescence channels use `MARKER_COLOURS`, shown per group as in `FIGURE_CHANNELS` (`group_colours`).
- The line colours and cell-type colours were checked for colour-blind separation.

## Text roles

| Role | Size | Weight |
| --- | --- | --- |
| Panel letter | 8 pt | bold |
| Panel title | 7 pt | bold |
| Subtitle | 7 pt | regular, grey |
| Axis label | 7 pt | regular |
| Ticks, legend, annotations | 6 pt | regular |
| Smallest allowed | 5 pt | |

`paper_style("talk")` scales every size for slide figures (`overview_figure.py`).

## Figure notebooks

| Notebook | Figure | Saved to |
| --- | --- | --- |
| `imaging_schematic.py` | 3D stack → slice, max and sum projections | `figures/schematics/` |
| `projection_resolution_figure.py` | NCA trajectories per projection and per resolution | `figures/schematics/` |
| `nodal_ko_schematic.py` | True data and NCA trajectories under NODAL knockout (drawing code in `ko_trajectory.py`) | `figures/schematics/` |
| `project_overview.py` | Project overview schematic (imaging, training, perturbations) | `figures/schematics/` |
| `nca_training_schematic.py` | NCA update step and training schematic | `figures/schematics/` |
| `loss_schematic.py` | Loss functions schematic | `figures/schematics/` |
| `overview_figure.py` | Results overview (slide-sized by default) | `figures/overview/` |

## What each figure needs

The notebooks read the data folder (`DATA_PATH_BASE`) and the model store
(`MODEL_STORE_ROOT`) from `.env`. Models are chosen with a **selection** file
exported from `Experiments/model_registry_explorer.py`
([workflow_micropatterns.md](workflow_micropatterns.md#7-copy-models-back-and-pick-the-ones-to-use)).

| Notebook | Data | Models | Other inputs |
| --- | --- | --- | --- |
| `imaging_schematic.py` | yes | | one 3D stack (`MICROPATTERN_3D_STACK`) |
| `loss_schematic.py` | yes | | |
| `nca_training_schematic.py` | | | toy data only |
| `project_overview.py` | yes | | |
| `nodal_ko_schematic.py` | yes | selection | |
| `projection_resolution_figure.py` | yes | two selections (per projection, per resolution) | |
| `overview_figure.py` | yes | selection | exports from `nodal_ko_eval.py`, saved into `figures/overview/`: **run `nodal_ko_eval.py` first** |

Slow results (rollouts) are cached as pickles in `figures/cache/` and
`figures/overview/cache/`. Delete them after changing a model or the data.

TODO(handover): which selection file each final paper figure was made with.

Figures also made outside this folder:

- `nodal_ko_eval.py`: evaluation notebook. It also draws prevalence against knockout time and cell-type maps, and exports the data that `overview_figure.py` reads.
- `data_cleaning.py`: data processing and QC. Section 6 draws the cleaning steps figure.
- `ablation_comparison.py`: ablation figures (W&B runs and bundle inference).
