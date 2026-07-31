# Local model registry

Config-driven NCA training publishes the best checkpoint as an immutable model
bundle when `model_store.enabled` is true. W&B remains the training log; these
bundles are the offline, reproducible inference artifacts.

Local runs use `model_store.root` from the resolved config. Cluster launchers
set `MODEL_STORE_ROOT` at runtime so manifests remain portable across storage
mounts. The runtime value takes precedence over the config default.

The store has the following layout:

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

Training jobs only create their own bundle directories. They do not write the
shared SQLite file. Rebuild that disposable index after copying models locally:

```bash
python -m Experiments.model_registry reindex
python -m Experiments.model_registry list
python -m Experiments.model_registry show <model-id>
```

## Python and marimo

The registry exposes ordinary pandas dataframes suitable for marimo tables,
filters, and SQL-backed analysis:

```python
from Common.model_registry import ModelRegistry

registry = ModelRegistry.from_env()
models = registry.models_df()
evaluations = registry.evaluations_df()
tags = registry.tags_df()

selected = models.query("family == 'NCA' and status == 'complete'")
bundle = registry.get(selected.iloc[0].model_id)
model = bundle.load_model()
cfg = bundle.config
```

Aliases, tags, and notes are deliberately mutable and live outside immutable
model bundles:

```python
registry.annotate(
    bundle.id,
    alias="emoji-baseline",
    tags=["baseline", "paper-figure"],
    notes="Stable after local damage tests.",
)
```

The saved config is the resolved training config and must not be changed.
Notebook-specific rollout data and parameters should remain separate inputs.

## Recording evaluations

Evaluation summaries are immutable artifacts separate from model bundles:

```python
from Common.model_registry import record_evaluation

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

Store large trajectories beside the evaluation manifest as `.npz` or another
appropriate format. Keep only scalar summary metrics in the manifest and SQL
index.

## Recovery contract

An Equinox leaf checkpoint requires a matching model structure. Each bundle
therefore records the resolved config, model factory, Git state, package
versions, and checkpoint checksum. `bundle.load_model()` reconstructs the model
through that factory, verifies the checksum, and then loads the leaves.
