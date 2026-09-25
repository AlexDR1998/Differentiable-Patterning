# NCA trainer

Experiment entrypoints (`Experiments/<domain>/train.py`) load the data, build
the model and data augmenter, and fill in a `TrainerContext`. Everything after
that is shared:

```text
Experiments/<domain>/train.py:run(cfg)
  -> Experiments/nca_training.py:run_training   (adds W&B tags, publishes the bundle)
     -> NCA/trainer/trainer.py:build_trainer -> NcaTrainer
        -> NcaTrainer.train
           1. preparation.prepare_training   -> PreparedTraining
           2. NcaTrainer.setup_logging
           3. step.build_train_step          -> jitted train_step
           4. validation.build_validation_evaluator (only with validation data)
           5. runner.run_loop                -> TrainingResult
```

## Where things live

| File | What it does |
| --- | --- |
| `trainer.py` | `NcaTrainer`: keeps the config sections it uses (`run_config`, `trainer_config`, `optimiser_config`, `loss_config`, `logging_config`, `knockout_config`) and what is derived from the data: the augmenter, boundary callbacks, loss masks, channel counts. Chooses the logger. |
| `preparation.py` | `prepare_training` reads the trainer and returns a frozen `PreparedTraining`: optimiser, interval schedule, loss functions and cached target features, regularisers, initial pool. It does not change the trainer. |
| `step.py` | Everything traced by JAX: `batch_model` (applies the model to every batch, blocking NODAL for knockouts), `run_nca_steps` (the rollout scan), `batch_loss`, `multi_target_losses`, and `build_train_step` (rollout, loss, gradient, optimiser update). Also `TrainState` and `StepOutput`. |
| `runner.py` | `run_loop`: the Python loop around the compiled step. Pool admission, validation, logging and best-checkpoint saving (`BestCheckpoint`). |
| `validation.py` | `ValidationEvaluator`: rollout and loss on held-out replicates, with their own boundary callbacks and masks. |
| `objective.py` | `resolve_objective`: turns the typed loss config into loss names, shared loss arguments and regulariser weights. |
| `pool.py` | Pool admission decisions (whether a rollout is written back into the training pool). |
| `loss_schedule.py`, `interval_schedule.py` | Loss-weight schedules, and steps per time slot for non-uniform observation times. |
| `optimizer.py`, `NCA_regulariser.py`, `intervention.py`, `instrumentation.py` | Optimiser and learning-rate schedule, regularisers, NODAL knockout helpers, optional profiling. |

## Rules

- User choices belong in the frozen config dataclasses. Don't add constructor
  or `train()` arguments for new options.
- `TrainerContext` holds only values derived from loaded data or the runtime:
  boundaries, channel schemas, masks, names, paths, provenance and W&B tags.
- The compiled step covers the rollout, loss, gradient and optimiser update.
  Logging, pool admission and checkpoint decisions stay in Python.
- Data augmenters follow `data_augmenter/protocols.py:NCAAugmenterProtocol`.
  All randomness must come from the supplied key.

`tests/unit/test_trainer_runs.py` runs the whole loop for a few iterations on a
tiny synthetic problem, so a broken trainer shows up in `pytest tests/unit`.
