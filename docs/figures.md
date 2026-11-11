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
save_figure(fig, "my_figure", "results")  # figures/results/my_figure.pdf and .png
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
| `nodal_ko_schematic.py` | NODAL knockout timeline with real colonies | `figures/schematics/` |
| `project_overview.py` | Project overview schematic (imaging, training, perturbations) | `figures/schematics/` |
| `nca_training_schematic.py` | NCA update step and training schematic | `figures/schematics/` |
| `loss_schematic.py` | Loss functions schematic | `figures/schematics/` |
| `overview_figure.py` | Results overview (slide-sized by default) | `figures/overview/` |

Figures also made outside this folder:

- `nodal_ko_eval.py`: evaluation notebook. It also draws the trajectory/branch figure, prevalence against knockout time and cell-type maps, and exports the data that `overview_figure.py` reads.
- `data_cleaning.py`: data processing and QC. Section 6 draws the cleaning steps figure.
- `ablation_comparison.py`: ablation figures (W&B runs and bundle inference).
