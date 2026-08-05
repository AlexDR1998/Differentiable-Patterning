# Differentiable Patterning

Differentiable Patterning is a research codebase for learning and analysing
self-organising spatiotemporal systems with differentiable programming. It
contains neural cellular automata (NCA), differentiable partial differential
equation (PDE) models, and experiment workflows for image-based and biological
micropatterning tasks.

This is active research software rather than a polished general-purpose
package. APIs, model families, and experiment configurations may change as the
research develops. Full training workloads are intended for remote
accelerator/cluster hardware.

## Current scope

The actively maintained workflows focus on:

- Neural cellular automata for learning, generating, and analysing spatial
  patterns.
- Differentiable PDE solvers and parameterised reaction--diffusion-style
  models.
- Config-driven experiments for emoji, micropattern, and impulse-optimisation
  tasks.
- Reproducible local model bundles and visualisation/export tooling.

`ABM/` and `Experiments/archive/` contain exploratory or historical work. They
may be useful as research references, but are not the best starting point for
new development.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Common/` | Shared JAX/Equinox models, spatial operators, data utilities, losses, and configuration support. |
| `NCA/` | Neural cellular automaton models, training components, analysis, and registry support. |
| `PDE/` | Differentiable PDE solvers, fixed models, and PDE training utilities. |
| `Experiments/` | Hydra-configured experiment entry points and definitions. |
| `demo/` | Small examples and walkthroughs for understanding the project. |
| `docs/` | Maintained documentation for configuration, training, and model bundles. |
| `WebDemo/` | Static WebGL visualisation and export tooling for compatible NCA models. |
| `tests/` | Unit, integration, and hardware-specific checks. |

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

## Start here

For an overview of the current NCA workflow, begin with:

- [`demo/nca_training_config_walkthrough.py`](demo/nca_training_config_walkthrough.py)
- [`demo/nca_objectives_and_pool_dynamics.py`](demo/nca_objectives_and_pool_dynamics.py)
- [Typed experiment configuration](docs/configuration.md)
- [NCA trainer architecture](docs/nca_trainer.md)
- [Local model registry](docs/model_registry.md)

The `demo/` directory also contains PDE demonstrations and notebooks. These
are intended for exploration; do not treat a training script as a lightweight
smoke test.

## Experiments and reproducibility

Active experiments are defined in `Experiments/` using Hydra YAML
configuration. Configurations are resolved and converted to typed, immutable
dataclasses before model construction. Experiment-specific data loading,
cluster launch settings, and expected compute requirements are intentionally
kept close to the relevant experiment configuration.

When enabled, NCA training publishes a versioned local model bundle containing
the checkpoint, resolved configuration, provenance, and checksums. W&B is used
for training logs. See the [model registry documentation](docs/model_registry.md)
for bundle layout, recovery, and inspection commands.

Cluster/container launch files, including `launch_slurm.sh`,
`launch_batch_slurm.sh`, and `run.tpl.yml`, are provided for remote workflows.
Review and adapt these to the target environment before submitting work.

## Tests

Run the lightweight unit test suite with:

```bash
pytest tests/unit
```

Integration and hardware tests may require GPU, SYCL, datasets, model assets,
or cluster-specific configuration. They are not expected to run in every local
development environment.

## Web demo

`WebDemo/` contains a static WebGL viewer for exported compatible NCA models,
alongside a marimo analysis page. See the [WebDemo README](WebDemo/README.md)
for export and serving instructions.

## Research use

If you build on this work, please contact the repository author for the most
appropriate citation for the relevant model, experiment, or result. A formal
citation and licence can be added here when they are available.
