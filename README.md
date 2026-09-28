# Differentiable Patterning

Differentiable Patterning is a research codebase for learning and analysing
self-organising spatiotemporal systems with differentiable programming. It
contains neural cellular automata (NCA) and experiment workflows for
image-based, biological micropatterning and geospatial (snowmelt) tasks.

This is active research software rather than a polished general-purpose
package. APIs, model families, and experiment configurations may change as the
research develops. Full training workloads are intended for remote
accelerator/cluster hardware.

## Current scope

The actively maintained workflows focus on:

- Neural cellular automata for learning, generating, and analysing spatial
  patterns.
- Config-driven experiments for emoji, micropattern, snowmelt and
  impulse-optimisation tasks.
- Reproducible local model bundles and visualisation/export tooling.

Earlier work on differentiable PDEs, agent-based models and archived experiment
scripts has been removed from the working tree. It remains available in the git
history.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Common/` | Shared JAX/Equinox model base, spatial operators, data loaders, losses, and configuration support. |
| `NCA/` | Neural cellular automaton models (`NCA/model/`) and the trainer (`NCA/trainer/`). |
| `Experiments/` | YAML-configured experiment entry points for each domain, the model registry, and marimo explorers (`*_explorer.py`). |
| `launch/` | Scripts that run a sweep on Kubernetes, Slurm or one local GPU. |
| `demo/` | Small examples and walkthroughs for understanding the project. |
| `docs/` | Maintained documentation for configuration, training, and model bundles. |
| `WebDemo/` | Static WebGL visualisation and export tooling for compatible NCA models. |
| `tests/` | Unit tests, plus a GPU memory check in `tests/hardware/`. |

Generated checkpoints, experiment outputs, W&B logs, figures, and videos are
local research artifacts rather than source code.

## Installation

Create an environment appropriate for the available hardware:

```bash
pip install -r requirements/requirements_cpu.txt
# or
pip install -r requirements/requirements_gpu.txt
```

Conda environments are also provided:

```bash
conda env create -f requirements/env_cpu.yml
# or
conda env create -f requirements/env_gpu.yml
```

JAX installations can be hardware- and driver-specific. The supplied GPU
requirements are a starting point; use the installation guidance appropriate
for the target accelerator platform.

Then install the repository itself, so `Common`, `NCA` and `Experiments` can
be imported from any script or notebook:

```bash
pip install -e .
```

(The cluster launchers set `PYTHONPATH` to the repository instead.)

## Start here

1. **Check the install:** `pytest tests/unit` runs in a few minutes on a CPU,
   including a tiny end-to-end training run.
2. **Read how a run is put together:** an experiment is one YAML config,
   for example `Experiments/emoji/conf/base_config.yaml`. It is converted to
   typed dataclasses ([configuration](docs/configuration.md)), and the entry
   point `Experiments/<domain>/train.py` loads the data, builds the model and
   augmenter, and hands them to the trainer
   ([NCA trainer](docs/nca_trainer.md)).
3. **Look at the code in this order:** `NCA/model/NCA_model.py` (the update
   rule), `NCA/trainer/data_augmenter/` (how the training pool is refreshed),
   `Common/trainer/loss_table.py` (the available losses), then
   `NCA/trainer/step.py` and `NCA/trainer/runner.py` (the training loop).
4. **Run a sweep:** write a sweep file in `conf/experiments/`, turn it into a
   manifest with `Experiments/generate_configs.py`, and run it remotely with
   the scripts in `launch/` (see below).
5. **Inspect results:** trained models are saved as bundles
   ([model registry](docs/model_registry.md)).
   `marimo edit Experiments/model_registry_explorer.py` searches and compares
   them; the other `*_explorer.py` notebooks look at datasets, augmenters and
   evaluations.

## Experiments and reproducibility

Active experiments are defined in `Experiments/` using YAML configuration
(read with OmegaConf). Configurations are resolved and converted to typed, immutable
dataclasses before model construction. Experiment-specific data loading,
cluster launch settings, and expected compute requirements are intentionally
kept close to the relevant experiment configuration.

When enabled, NCA training publishes a versioned local model bundle containing
the checkpoint, resolved configuration, provenance, and checksums. W&B is used
for training logs. See the [model registry documentation](docs/model_registry.md)
for bundle layout, recovery, and inspection commands.

The scripts in `launch/` run every entry of a manifest:

- `bash launch/launch_batch_multi_job.sh Experiments/run_config.py <manifest> <workers> <gpu>`
  submits Kubernetes jobs from `launch/run.tpl.yml`, each pod running
  `launch/launch_inside_pod.sh`;
- `bash launch/launch_batch_slurm.sh Experiments/run_config.py <manifest>`
  submits a Slurm array that runs `launch/launch_slurm.sh`;
- `python launch/launch_local_sweep.py <manifest>` runs the entries one after
  another on a local GPU.

They share `launch/manifest.sh` for reading a manifest. The cluster settings
(accounts, partitions, storage paths) are specific to our clusters; review and
adapt them before submitting work.

## Tests

Run the lightweight unit test suite with:

```bash
pytest tests/unit
```

To check that full training still works end to end, run the smoke sweeps on a
GPU machine (10 very short runs across emoji, micropatterns, Nodal knockout
fine-tuning and snowmelt):

```bash
python Experiments/pipeline_smoke_test.py
```

Each run must finish and publish a model bundle, which goes to the normal
model store under the `pipeline-smoke` collection. The sweeps are the
`smoke_*.yaml` files in each `Experiments/<domain>/conf/experiments/` folder.
The fine-tuning sweep needs the ID of an existing trained model set in its YAML
file, and is skipped until then. `--dry-run` writes the manifests and prints
the Kubernetes commands to run every smoke run as its own pod.

`tests/hardware/jax_gpu_mem_test.py` is a standalone GPU check, not part of the
unit suite.

## Web demo

`WebDemo/` contains a static WebGL viewer for exported compatible NCA models,
alongside a marimo analysis page. See the [WebDemo README](WebDemo/README.md)
for export and serving instructions.

## Research use

If you build on this work, please contact the repository author for the most
appropriate citation for the relevant model, experiment, or result. A formal
citation and licence can be added here when they are available.
