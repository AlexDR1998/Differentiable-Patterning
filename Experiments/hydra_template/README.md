# Hydra Template

This folder is a high-level starter for a two-step Hydra workflow around NCA training:

1. Generate a manifest with fully materialized configs.
2. Run a single config selected by index, usually from a scheduler.

The baseline config is intentionally dataset-agnostic but includes practical sections for:

- `data`
- `model`
- `optimiser`
- `loss`
- `trainer`
- `run`
- `logging`

## Generate

```bash
python Experiments/hydra_template/generate_configs.py
```

The default sweep uses:

- `Experiments/hydra_template/conf/config.yaml`
- `Experiments/hydra_template/conf/experiments/example_sweep.yaml`

You can override either path:

```bash
python Experiments/hydra_template/generate_configs.py \
  --base-config Experiments/hydra_template/conf/config.yaml \
  --sweep-file Experiments/hydra_template/conf/experiments/example_sweep.yaml \
  --output-dir Experiments/hydra_template/generated/example_template
```

## Run One Config

Run directly from the manifest:

```bash
python Experiments/hydra_template/run_config.py \
  --manifest Experiments/hydra_template/generated/example_template/manifest.yaml \
  --index 0
```

Or run one concrete config file:

```bash
python Experiments/hydra_template/run_config.py \
  --config-file Experiments/hydra_template/generated/example_template/config_0000.yaml \
  --entrypoint example_experiment:run
```

## Scheduler Mapping

If your scheduler sets `JOB_COMPLETION_INDEX`, the runner will use it automatically when `--index` is omitted.

For example:

```bash
python Experiments/hydra_template/run_config.py \
  --manifest Experiments/hydra_template/generated/example_template/manifest.yaml
```
