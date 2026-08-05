# NCA trainer architecture

Active NCA experiments use one immutable `ExperimentConfig`. Experiment
entrypoints are responsible only for loading domain data, building a model and
augmenter, and deriving a `TrainerContext`.

```text
emoji/micropattern entrypoint
    -> TrainerContext + ExperimentConfig
    -> build_trainer
    -> NcaTrainer.train
    -> TrainingResult
    -> optional model publication
```

## Configuration and runtime values

User choices must be represented by frozen config dataclasses. Do not add
individual constructor or `train` arguments for new training options.

`TrainerContext` is reserved for runtime values derived from loaded data:
boundaries, channel schemas, masks, display names, output paths and provenance.
Models, arrays and PRNG keys are runtime values and do not belong in config.

## Focused components

- `objective.py` resolves typed loss terms and regularisers once.
- `pool.py` owns recurrent-pool admission state and decisions.
- `checkpointing.py` owns best-checkpoint policy.
- `instrumentation.py` contains optional profiling and timing.
- `training_execution.py` defines standard execution behavior.
- `sycl_execution.py` contains two-tile execution behavior.
- `trainer.py` is the public backend-selection boundary.

The compiled numerical step performs rollout, objective differentiation and
the optimizer update. Python-side logging, profiling, pool admission and
checkpoint decisions remain outside differentiated code.

`NCA/trainer_old` is a read-only historical fallback and must not be imported
by active code.

