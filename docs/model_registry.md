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
  impulses/<experiment>/<impulse-id>/
    intervention.eqx
    config.yaml
    manifest.yaml
    states.npz
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

## Impulse records

Each impulse run (`Experiments/impulse/optimise.py`) saves its best
intervention as a record in the store it loaded the model from
(`checkpoint.store_root`, then `MODEL_STORE_ROOT`, then `models/`). Records,
like bundles, are written to a hidden folder and renamed into place, and are
never edited. The record id (timestamp, config hash, random part) is also the
W&B run name.

`manifest.yaml` holds:

- `model`: the bundle id, its checkpoint checksum and path, so the impulse
  links to the exact weights it was optimised for
- `conditions`: pair source, source and target indices, and their pattern
  names (`data.emoji.pairs[i].target`)
- `results`: switch rate, evaluation losses (with and without the impulse) and
  impulse size
- `artifact`: the intervention checksum and the state shape, so the
  intervention can be rebuilt without the model or data
- `provenance`: config hash, git commit, package versions, W&B project, group,
  run id and tags

`reindex` adds the tables `impulses` (one row per record) and
`impulse_wandb_tags` (every setting as an untruncated `key:value` tag, like
`model_wandb_tags`):

```python
registry = ModelRegistry.from_env()
impulses = registry.impulses_df()
with_models = impulses.merge(registry.models_df(), on="model_id", suffixes=("", "_model"))

record = registry.get_impulse(impulses.iloc[0].impulse_id)
intervention = record.load_intervention()   # checks the checksum
states = record.load_states()               # states.npz as a dict
bundle = record.verify_model(registry)      # the linked bundle, checked against the impulse's checksum
```

Filtering by the tags of both the model and the impulse, in SQL:

```sql
SELECT i.* FROM impulses i
WHERE EXISTS (SELECT 1 FROM model_wandb_tags t
              WHERE t.model_id = i.model_id AND t.tag = 'data.emoji.task:multi_attractor')
  AND EXISTS (SELECT 1 FROM impulse_wandb_tags t
              WHERE t.impulse_id = i.impulse_id AND t.tag = 'impulse.intervention.patch_size:3');
```

`Experiments/impulse/impulse_explorer.py` reads records from the store. Runs
saved before records existed (loose `.eqx`/`.json`/`.npz` files in one
folder) are read only by `impulse_sweep_explorer.py`.

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
