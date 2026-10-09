# Model registry

With `model_store.enabled: true`, training saves the best checkpoint as a model
bundle. W&B holds the training logs; bundles are what you load models from
afterwards. Bundles are never edited once written.

The store root is `model_store.root` from the config, overridden by the
`MODEL_STORE_ROOT` environment variable (the cluster launchers set it):

```text
$MODEL_STORE_ROOT/
  bundles/<collection>/<experiment>/<slug>--<model-id>/
    model.eqx
    config.yaml
    manifest.yaml
  evaluations/<evaluator>/<evaluation-id>/
    manifest.yaml
  registry.sqlite
```

Training jobs only write their own bundle folder. `registry.sqlite` is an index
you rebuild after copying models locally:

```bash
python -m Experiments.model_registry reindex
python -m Experiments.model_registry list
python -m Experiments.model_registry show <model-id>
```

The CLI uses `--root` if given, then `MODEL_STORE_ROOT`, then `models/` in the
repository.

## Loading models

The registry returns pandas dataframes, which work directly in marimo:

```python
from Experiments.model_registry import ModelRegistry

registry = ModelRegistry.from_env()
models = registry.models_df()
evaluations = registry.evaluations_df()
tags = registry.tags_df()

selected = models.query("family == 'NCA' and status == 'complete'")
bundle = registry.get(selected.iloc[0].model_id)
model = bundle.load_model()
cfg = bundle.config  # typed ExperimentConfig
```

Each bundle records the resolved config, model factory, git state, package
versions and a checkpoint checksum. `bundle.load_model()` rebuilds the model
from its saved `ModelConfig` through that factory, checks the checksum, then
loads the weights.

Aliases, tags and notes can be changed, and are stored outside the bundle:

```python
registry.annotate(
    bundle.id,
    alias="emoji-baseline",
    tags=["baseline", "paper-figure"],
    notes="Stable after local damage tests.",
)
```

### Retired model families

Bundles saved as `NCA_sycl` or `NCA_fast` load as `NCA`, and `gNCA_sycl` as
`gNCA`; the weights have the same layout. `implementation="recorded"` does not
work for these, since the old code is gone.

`gNCA`, `nNCA` and `gnNCA` are now `NCA` with `GATED=True` and/or
`PARAMETER_NOISE_LEVEL` set. These are static fields, so they are not saved in
`model.eqx` and old bundles load unchanged.

## Recording evaluations

Evaluation results are saved separately from the bundles:

```python
from Experiments.model_registry import record_evaluation

record_evaluation(
    store_root=registry.root,
    model_id=bundle.id,
    evaluator="damage_recovery_v1",
    dataset="held_out_emojis_v2",
    seed=0,
    parameters={"steps": 128, "damage_radius": 6},
    metrics={"final_l2": 0.018, "recovery_time": 42},
)
registry.reindex()
```

Keep only scalar metrics in the manifest. Save large arrays next to it (e.g.
as `.npz`).

## Checking evaluation inputs

Bundles don't store the training data. Instead `manifest.evaluation_input`
holds hashes, shapes and dtypes of the initial condition `data[:, 0]` (and the
boundary mask, if used). Before evaluating, reload the data as described by
`bundle.config` and check it still matches:

```python
from Experiments.model_registry import verify_evaluation_input

bundle = registry.get(model_id)
data, boundary_mask = ...  # reload with the domain's loader
verify_evaluation_input(
    data,
    bundle.manifest.evaluation_input,
    boundary_mask=boundary_mask,
)
```

This fails if the data or pre-processing has changed since training.
