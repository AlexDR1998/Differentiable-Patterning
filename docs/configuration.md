# Typed experiment configuration

Supported workflows resolve YAML interpolation and sweep overrides in
`Experiments/`, then immediately convert the result to frozen dataclasses in
`Experiments.config`. OmegaConf objects must not be passed into `Common/` or
`NCA/` builders.

The stable training schema is:

```text
ExperimentConfig
├── RuntimeConfig
├── DataConfig
│   ├── PreprocessingConfig
│   ├── EmojiDataConfig | MicropatternDataConfig
│   └── KnockoutConfig
├── ModelConfig
├── TrainingConfig
│   ├── TrainingLoopConfig
│   ├── TrainerConfig
│   ├── OptimizerConfig
│   ├── LossConfig
│   └── CheckpointConfig
├── LoggingConfig
└── ModelStoreConfig
```

Impulse optimisation has its own `ImpulseExperimentConfig`, while reusing the
same runtime, data, and reconstructable model types.

Configuration conversion is strict: unknown fields and unsupported model
families fail before JAX is initialised. Augmentation schedules are data-owned
configuration. Their current iteration is supplied by the training loop, and
fractional schedules receive the configured total iteration count explicitly.

Model bundles contain the resolved dataclass representation with
`schema_version`. This is a clean-break schema; regenerate manifests and retrain
models created before this configuration system.

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
