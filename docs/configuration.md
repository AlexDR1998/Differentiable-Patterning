# Experiment configuration

Experiment YAML files (a base config plus sweep overrides) are merged with
OmegaConf and then converted into frozen dataclasses by
`Experiments.config.load_experiment_config`. Only these dataclasses are
passed into `Common/` and `NCA/`.

The dataclasses use the same section and field names as the YAML files, so
`run.t` in a sweep file is `cfg.run.t` in code:

```text
ExperimentConfig                 (Experiments/config.py)
├── experiment                   name, stability_mode
├── system: SystemConfig         precision, gpu, xla_flags
├── data: DataConfig             dataset, batches, downsample
│   ├── emoji | micropattern | snowmelt | pde   (whichever matches data.dataset)
│   └── knockout: KnockoutConfig          curriculum, channel, mode, time
├── model: ModelConfig           (NCA/model/config.py)
├── run: RunConfig               t, iterations, checkpoint_warmup, ...
├── trainer: TrainerConfig       (NCA/trainer/config.py), incl. pool_admission
├── optimiser: OptimiserConfig   (Common/trainer/config.py)
├── loss: LossConfig             terms, regularisers, schedule_label
├── logging: LoggingConfig
├── model_store: ModelStoreConfig
├── initialization               model_id (fine-tuning parent)
└── labels                       descriptive W&B metadata only
```

Impulse optimisation has its own `ImpulseExperimentConfig`, which reuses the
same `system`, `data` and `logging` sections. It has no `model` section: the
trained model is loaded from the registry by `checkpoint.model_id`, and
`data: null` (the default) means the data the model was trained on.

Conversion is strict: unknown fields and unsupported model families fail
before JAX starts. Every field has a value after conversion, so read fields as
attributes (`cfg.run.t`), never with `.get(key, default)`. To add an option,
add a field (with a default) to the right dataclass.

Two warmups exist: `run.checkpoint_warmup` (the best checkpoint is only saved
after this many iterations) and `trainer.pool_admission.warmup` (pool
admission starts after this many). A null pool admission warmup means the same
as `run.checkpoint_warmup`.

## W&B tags

Each run gets a `key:value` W&B tag for every setting in
`logging.wandb.tag_keys` (a key also covers everything below it, e.g.
`data.micropattern.cleaning`), plus any tags in `logging.wandb.tags`.
`Experiments/generate_configs.py` sets `tag_keys` to the settings that differ
between runs of the sweep, so tags show what distinguishes each run. A sweep
can set its own `logging.wandb.tag_keys` in its grid. With `tag_keys: null`
(a single config, or an old manifest) every setting is tagged. The model
registry always keeps the full tags.

## Older configs

Current files have `schema_version: 7`. Older configs are translated when read
by `upgrade_legacy_config`, the only place old spellings are handled. Each
upgrade fills in the values the code used before, so old runs and bundles load
the same data they were trained on:

- **7** added `data.snowmelt.version` (`v1` or `v2`, the dataset folder under
  `data.snowmelt.root`). Older snowmelt configs get `v1` and, if they used
  the default, the old static channels `[DEM, INCIDENCE]`.
- **6** replaced `data.micropattern.initial_intensity_scales` (0h only) with
  `intensity_factors` (any channel and timestep) and added
  `data.micropattern.cleaning`. Older configs keep cleaning off.
- **5** added `data.micropattern.histogram_percentiles` and
  `initial_intensity_scales`. Older configs get the previously hard-coded
  `[0.5, 99.95]` and `cell_fate_s2/FOXA2: 0.075`.
- **3** and **2** differ only by the unused `trainer.sharding` and
  `trainer.backend` options, which are dropped.
- **1** came in two layouts:
  - sweep files and manifests: top-level `knockout`, `run.warmup`, flat
    `trainer.pool_admission_*` keys, `run.filename_mode` and
    `data.emoji.regenerate`;
  - saved bundles: `runtime`, `training.{loop, trainer, optimizer, loss,
    checkpoint}`, `data.preprocessing` and `data.{augmentation, intervention}`.

`data.emoji.regenerate` in old sweep files never actually turned regeneration
off (the base config's `regeneration.enabled` won); old manifests reproduce
this. Current sweep files set `data.emoji.regeneration.enabled` directly.

## 260726 micropattern pre-processing

For `data.dataset: micropatterns_260726`, the raw images are cleaned before
training as tuned in `Experiments/micropatterns/data_cleaning.py`. That
notebook prints the matching `data.micropattern` block under **Training
settings** and exports the alignment file. The steps are described in
`Common/dataloader/micropattern_cleaning.py`:

1. `intensity_factors` multiply the raw intensities of a channel at a
   timestep (hours), for every condition, to correct imaging artifacts.
2. Hot pixels (isolated bright specks) are replaced by the mean of their
   neighbours, per channel (`cleaning.hot_pixel_thresholds`, 0 = off).
3. A rolling-ball background is subtracted per channel
   (`cleaning.background_radii` in full-resolution pixels, 0 = off).
4. The images are block-averaged to downsampling 4 (the notebook's), or to
   the training downsampling if it is not a multiple of 4.
5. Each colony is centred with the centres in `cleaning.alignment_file`, and
   masked with a disk of the pattern radius times `cleaning.mask_radius_scale`.
6. Each channel is rescaled linearly to [0, 1] between a low and a high
   percentile (`histogram_percentiles`, or `cleaning.channel_percentiles` per
   channel), taken over all timesteps of a trajectory (inside the mask with
   `percentiles_inside_mask`), so brightness can still be compared over time.
   `cleaning.normalisation` sets how trajectories share these bounds:
   `replicate_mean` (each trajectory's bounds, averaged), `per_replicate` (own
   bounds) or `pooled` (all pixels together). With
   `knockouts_use_control_bounds`, knockouts take their bounds from the
   control data, so a knockout's change in overall intensity is kept.
7. Pixels outside the mask are zeroed and the images are block-averaged to
   the training downsampling.

With shared bounds, validation replicates reuse the bounds of the training
replicates; with `per_replicate`, each replicate sets its own.

The alignment file is computed once in the notebook (an automatic
fixed-radius fit, then manual corrections) and lists the colony centre of
every image, keyed by its path relative to the dataset root. A relative
`alignment_file` is resolved against the repository root.

```yaml
data:
  micropattern:
    histogram_percentiles: [20.0, 99.5]
    intensity_factors:
      cell_fate_s1/SOX17: {0: 0.3}
      cell_fate_s2/SOX17: {0: 0.3}
      cell_fate_s2/FOXA2: {0: 0.02}
    cleaning:
      enabled: true
      hot_pixel_window: 5
      hot_pixel_thresholds: {cell_fate_s1/SOX17: 30.0, ...}
      background_shrink: 4
      background_radii: {cell_fate_s1/SOX17: 50, ...}
      alignment_file: Experiments/micropatterns/conf/alignment/260726_alignment.yaml
      mask_radius_scale: 1.0
      normalisation: replicate_mean
      channel_percentiles: {}
      percentiles_inside_mask: true
      knockouts_use_control_bounds: false
```

Images can also be flagged as low quality in the notebook (**Manual
centring**) and exported as a quality flags file (**Quality flags**), set as
`data.micropattern.quality_flags_file` (relative to the repository root;
null for none). Each flag is one image file, so one experiment group at one
timestep of one replicate. Training treats a flagged image as not measured:
its channels are off in the measurement mask, so the loss and reinjection
skip them, and it does not count towards the normalisation bounds. The
replicate's other groups and timesteps are kept. Because the NCA pool is
seeded from every stored timestep, the flagged slot is filled with the mean
of the same group and timestep over the other replicates of the condition in
that load. A flagged control image is also excluded where knockout
trajectories reuse it before the knockout.

Cell-type thresholds tuned in the notebook (**Cell types at 48h**) can be
exported as a cell type file (**Export and load cell types**, by default
`Experiments/micropatterns/conf/cell_types/260726_cell_types.yaml`). It holds
the channel and threshold of each marker, the high/low/any rule of each cell
type, and the `data.micropattern` settings the thresholds were tuned with,
since they only hold for images normalised that way. Training does not read
it; other code loads it with `Common.dataloader.cell_types.load_cell_type_rules`
and labels normalised images with `classify_cell_types` (`marker_values`
picks the markers out of a channel stack by measurement or marker name).

With `cleaning.enabled: false` (the case for configs saved before schema
version 6), each channel is clipped to percentiles pooled over all loaded
images at full resolution, and each image is centred on a circle fitted to
its own colony; validation replicates and knockout conditions reuse the bins
of the control training replicates. The legacy `micropatterns` dataset
ignores all of these settings.

## PDE data

`data.dataset: pde` trains on trajectories simulated at the start of each run
(`Experiments/pde/train.py`, settings in `Experiments/pde/config.py`):

- `data.pde.model` names a PDE in `PDE/catalogue.py`. `parameters` overrides
  its defaults; keys must be that model's parameter names.
- `data.batches` trajectories start from `initial_condition` (seeded by
  `data.pde.seed`, not the model seed) and are sampled at `frames` evenly
  spaced times in `[0, t_end]`. `run.t` is the number of NCA steps per frame.
- `observed_channels` picks the PDE channels the NCA reproduces; the other
  PDE channels stay hidden from it.
- Leave `parameters` empty in base configs: sweeps are merged into it, so a
  sweep can add keys but not remove them.
- `conf/experiments/nca_pde_benchmark.yaml` holds working `dx`, `dt`,
  `t_end` and initial conditions for every PDE.

## Loss weight schedules

Any loss term's `weight` can be multiplied by a schedule, and so can each
named component of the `multi_target` loss. Fractions are of the whole
training run.

```yaml
loss:
  schedule_label: cos_macro
  terms:
    - type: multi_target
      weight: 1.0
      multi_target_weights:
        texture: 1.0
        radial: 1.0
        channel_mean: 1.0
        correlation: 1.0
      multi_target_schedules:
        texture:
          type: cosine
          initial_factor: 0.05
          final_factor: 1.0
          start_fraction: 0.30
          end_fraction: 0.75
        radial:
          type: cosine
          initial_factor: 1.0
          final_factor: 0.25
          start_fraction: 0.30
          end_fraction: 0.75
```

Schedule types are `constant`, `linear` and `cosine`. Effective weights are
logged under `loss_weight/*` and unweighted multi-target components under
`loss_component_raw/*`. The best checkpoint is only chosen after the last
weight change, so losses under different weightings aren't compared.
`schedule_label`, if set, is added to model and log names as `_ls<label>`.
