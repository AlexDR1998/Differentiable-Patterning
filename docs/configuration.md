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
