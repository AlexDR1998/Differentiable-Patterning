# Micropattern geometry optimisation

Finds the colony shape for which a trained micropattern NCA makes the most
of one cell type (e.g. notochord). The NCA is frozen and only the shape changes.

## Pieces

| Piece | Where |
| --- | --- |
| Shape | `NCA/inverse_design/geometry.py`: `FourierRadialGeometry`, r(θ) = R exp(Σ aₖ cos kθ + bₖ sin kθ), rescaled so the area stays πR². With `fixed_radius` only the shape is learned. |
| Textures | `Experiments/micropatterns/texture_settings.py`: synthesis settings for each projection, in `Experiments/micropatterns/conf/textures/260726_texture_settings.yaml`. Tune them in `boundary_mask_explorer.py` (pick the projection, adjust, then save). A run reads the section for its model's projection (`inverse_design.texture.file`; null uses `texture.*`). Patch sizes are saved with the downsample they were tuned at and are rescaled to the model's. The settings used are written to `summary.yaml` under `texture`. |
| Mask → NCA | `initial_state.build_initial_state` (texture × mask, and the mask channel for soft boundaries) and `rollout.rollout_final` (mask re-imposed after every step). |
| Objective | `objectives.CellTypeShareObjective.from_rules`: the target's share of the colony, using the rules of a cell type file (`Common/dataloader/cell_types.py`). Each marker threshold becomes a sigmoid of width `temperature`. "Or" clauses, clause weights, "Several" and "Other" count as in `pixel_label_shares`, and match it exactly with `hard=True`. |
| Cell density | `NCA/inverse_design/density.py` (with `inverse_design.density.mode` fixed or optimise): the share of the colony covered by cells. Textures are made from training colonies whose gaps between cells are filled (`Experiments/micropatterns/reference.py:fill_reference_gaps`, with cells told from gaps by the generic markers, LMBR and SOX2). Each rollout then gets gaps from a random field with the training colonies' cell/gap spectrum, cut so that the chosen share is covered. Cells keep their brightness. With `optimise` the density is learned together with the shape (`geometry.optimise: false` learns only the density). `boundary_mask_explorer.py` compares both texture models. |
| Optimiser | `optimise.optimise_geometry`: a batch of initial textures with training-strength noise, adam with gradient clipping, and a mask edge that sharpens over the run. |
| Evaluation | `optimise.evaluate_binary`: rollouts on the binary mask (as in training), scored with the hard rules. These are the scores to report. The best geometry is chosen by them. |
| Logging | `Experiments/inverse_design/wandb_log.py`: each evaluation logs the binary score, its gain over the circle, the colony share of every label (`CellTypes/*`) and panels of mask, fates, cell type map (colours in `CellTypes/legend`) and target share. At the end, a video of `video_seeds` rollouts of the best geometry (`optimise.binary_trajectories`, same random numbers as the evaluations). |
| Experiment | `Experiments/inverse_design/`: `optimise.py:run`, `config.py`, `records.py`, `wandb_log.py`, `conf/`. |

## A run

1. Load the bundle (`checkpoint.model_id`), the cell type file, and the 0 h
   training colonies (`Experiments/micropatterns/reference.py`).
2. Check that the rules were tuned on the model's projection.
3. Synthesise `texture.bank_size` 0 h textures on a grid `geometry.grid_scale`
   times the training grid. Training colonies nearly fill their grid, so
   the extra space leaves room for new shapes.
4. Score the circle of the training colonies' area, and same-area ellipses,
   with the same random numbers as every later evaluation.
5. Optimise, evaluating every `log_every` iterations.
6. Save a record in `<store>/inverse_design/<experiment>/<id>/`: the best
   geometry, the config, and `summary.yaml` with the baseline and optimised
   shares of every label. `records.load_geometry` reads the geometry back.

```bash
python Experiments/generate_configs.py Experiments/inverse_design \
    Experiments/inverse_design/conf/base_config.yaml Experiments/inverse_design/conf/generated
```

## Caveats

- The models only saw binary circles of one size. Read optimised shapes as
  hypotheses, and compare against `Baseline/circle` and the ellipses, not
  against the soft training score.
- The models only saw the cell densities of the training colonies. Densities far from `Density/initial` are, like new shapes, hypotheses.
- Gaps are at least a pixel wide. At coarse downsampling real gaps may be smaller than a pixel, so check the textures in `boundary_mask_explorer.py` first.
- `Binary/border_fraction` above 0 means the colony reaches the grid edge,
  which wraps round under circular padding.
- The thresholds hold only for the projection and hour they were tuned on.
