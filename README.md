# Differentiable Patterning

Research code for training neural cellular automata (NCA) on spatial
patterning tasks: emoji images, stem-cell micropatterns, snowmelt maps and
simulated pattern-forming PDEs. Built on JAX, Equinox and Optax.

This is active research code, so APIs and configs change. Full training runs
are meant for remote GPUs.

Earlier work on trainable neural PDEs and agent-based models is in the git
history.

## Layout

| Path | Contents |
| --- | --- |
| `Common/` | Model base class, spatial operators, data loaders, losses, config dataclasses. |
| `NCA/` | NCA models (`NCA/model/`) and the trainer (`NCA/trainer/`). |
| `PDE/` | Pattern-forming PDEs (Gray-Scott, Cahn-Hilliard, Keller-Segel, ...) and the solver that turns them into NCA training data. |
| `Experiments/` | YAML-configured entry points for each domain, the model registry and marimo explorers (`*_explorer.py`). |
| `launch/` | Scripts that run a sweep on Kubernetes, Slurm or one local GPU. |
| `demo/` | Walkthrough notebooks. |
| `docs/` | Notes on configuration, the trainer and model bundles. |
| `WebDemo/` | WebGL viewer for exported NCA models. |
| `tests/` | Unit tests, plus a GPU memory check in `tests/hardware/`. |

## Installation

```bash
pip install -r requirements/requirements_cpu.txt   # or requirements_gpu.txt
# or
conda env create -f requirements/env_cpu.yml       # or env_gpu.yml
pip install -e .
```

JAX GPU installs depend on the driver; treat the GPU requirements as a
starting point. The cluster launchers set `PYTHONPATH` instead of installing.

## Getting started

1. `pytest tests/unit` checks the install (a few minutes on CPU, including a
   tiny training run).
2. An experiment is one YAML config, e.g.
   `Experiments/emoji/conf/base_config.yaml`. It is converted to typed
   dataclasses ([configuration](docs/configuration.md)), and
   `Experiments/<domain>/train.py` loads the data, builds the model and
   augmenter and hands them to the trainer ([NCA trainer](docs/nca_trainer.md)).
3. Suggested reading order: `NCA/model/NCA_model.py` (the update rule),
   `NCA/trainer/data_augmenter/` (how the training pool is refreshed),
   `Common/trainer/loss_table.py` (the losses), then `NCA/trainer/step.py` and
   `NCA/trainer/runner.py` (the training loop).
4. Trained models are saved as bundles ([model registry](docs/model_registry.md)).
   `marimo edit Experiments/model_registry_explorer.py` searches and compares
   them.

## Running sweeps

Write a sweep file in `Experiments/<domain>/conf/experiments/`, turn it into a
manifest with `Experiments/generate_configs.py`, then run it:

```bash
# Kubernetes (fills launch/run.tpl.yml)
bash launch/launch_batch_multi_job.sh Experiments/run_config.py <manifest> <workers> <gpu>
# Slurm array
bash launch/launch_batch_slurm.sh Experiments/run_config.py <manifest>
# One local GPU, entries in turn
python launch/launch_local_sweep.py <manifest>
```

The cluster settings (accounts, partitions, storage paths) are specific to our
clusters; adapt them before use. Training logs go to W&B, and with
`model_store.enabled` each run also saves a model bundle.

## Tests

```bash
pytest tests/unit                          # CPU
python Experiments/pipeline_smoke_test.py  # GPU, end to end
```

The smoke test runs 12 very short training runs (emoji, micropatterns, Nodal
knockout fine-tuning, snowmelt, PDEs) from the `smoke_*.yaml` sweep files.
Each must finish and publish a bundle to the `pipeline-smoke` collection. The
fine-tuning sweep is skipped until a trained model ID is set in its YAML file.
`--dry-run` only writes the manifests and prints the Kubernetes commands.

`tests/hardware/jax_gpu_mem_test.py` is a standalone GPU check.

## Web demo

See [WebDemo/README.md](WebDemo/README.md).

## Citation

Please contact the author for the right citation for a given model or result.
