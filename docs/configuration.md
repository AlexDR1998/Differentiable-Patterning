# Typed experiment configuration

Experiment YAML files are composed with OmegaConf in `Experiments/` (base
config plus sweep overrides) and then converted straight away into frozen
dataclasses by `Experiments.config.load_experiment_config`. OmegaConf objects
must not be passed into `Common/` or `NCA/`.

The dataclasses use the same section and field names as the YAML files, so
`run.t` in a sweep file is `cfg.run.t` in code:

```text
ExperimentConfig                 (Experiments/config.py)
├── experiment                   name, stability_mode
├── system: SystemConfig         precision, gpu, xla_flags
├── data: DataConfig             dataset, batches, downsample
│   ├── emoji | micropattern | snowmelt   (whichever matches data.dataset)
│   └── knockout: KnockoutConfig          curriculum, channel, mode, time
├── model: ModelConfig           (NCA/model/config.py)
├── run: RunConfig               t, iterations, checkpoint_warmup, ...
├── trainer: TrainerConfig       (NCA/trainer/config.py), incl. pool_admission
├── optimiser: OptimizerConfig   (Common/trainer/config.py)
├── loss: LossConfig             terms, regularisers, schedule_label
├── logging: LoggingConfig
├── model_store: ModelStoreConfig
├── initialization               model_id (fine-tuning parent)
└── labels                       descriptive W&B metadata only
```

Impulse optimisation has its own `ImpulseExperimentConfig`, which reuses the
same `system`, `data` and `model` sections.

Conversion is strict: unknown fields and unsupported model families fail
before JAX starts. Every field has a value after conversion, so read fields as
attributes (`cfg.run.t`), never with `.get(key, default)`. To add an option,
add a field (with a default) to the right dataclass.

Two warmups exist: `run.checkpoint_warmup` (the best checkpoint is only saved
after this many iterations) and `trainer.pool_admission.warmup` (pool
admission starts after this many). A null pool admission warmup means the same
as `run.checkpoint_warmup`.

## Older configs

Current files have `schema_version: 2`. Configs with `schema_version: 1` are
translated as they are read by `upgrade_legacy_config`, the only place old
spellings are handled. Version 1 came in two layouts:

- sweep files and manifests: top-level `knockout`, `run.warmup`, flat
  `trainer.pool_admission_*` keys, `run.filename_mode` and
  `data.emoji.regenerate`;
- saved model bundles: `runtime`, `training.{loop, trainer, optimizer, loss,
  checkpoint}`, `data.preprocessing` and `data.{augmentation, intervention}`.

Saved bundles are never edited, so they keep loading through this function.
Old manifests also still run. Note that `data.emoji.regenerate` in old sweep
files never actually switched regeneration off (the base config's
`regeneration.enabled` always took precedence); old manifests reproduce that,
and current sweep files set `data.emoji.regeneration.enabled` directly.

## Loss weight schedules

Any loss term can multiply its configured `weight` by an optional schedule.
The `multi_target` loss also supports schedules for its named internal
components. Fractions refer to the complete configured training run.

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

Supported schedule types are `constant`, `linear`, and `cosine`. Effective
weights and raw, unweighted multi-target components are logged under
`loss_weight/*` and `loss_component_raw/*`, respectively. Best-checkpoint
selection begins only after the final weight transition so that losses from
different objective phases are not compared directly. When provided,
`schedule_label` is appended to model and logging names as `_ls<label>`.
