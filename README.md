# Differentiable Patterning

Research code for training neural cellular automata (NCA) on spatial
patterning tasks: stem-cell micropatterns, emoji images, snowmelt maps and
simulated pattern-forming PDEs. Built on JAX, Equinox and Optax.

This is research code, so APIs and configs change. Full training runs are
meant for remote GPUs (the EIDF cluster); notebooks and small tests run on a
laptop CPU.

Earlier work on trainable neural PDEs and agent-based models is in the git
history.

## New here?

1. **Set up** (below), about 15 minutes.
2. **Read** [docs/concepts.md](docs/concepts.md): what an NCA is and the words
   used throughout the code.
3. **Work through the tutorials** in [demo/](demo/README.md). They run on a
   laptop CPU with the example data in the repository.
4. **Run a real experiment** with
   [docs/workflow_micropatterns.md](docs/workflow_micropatterns.md), which
   covers the whole path from a sweep file to a paper figure.

All documentation is listed in [docs/README.md](docs/README.md).

## Setting up

You need Python 3.13 (conda or venv) and git.

```bash
git clone <repository url> && cd Differentiable-Patterning

# Pick one: CPU (laptop) or NVIDIA GPU
conda env create -f requirements/env_cpu.yml && conda activate jax_cpu
# conda env create -f requirements/env_gpu.yml && conda activate jax_cuda
# (or, without conda: pip install -r requirements/requirements_cpu.txt)

pip install -e .            # makes Common, NCA, PDE and Experiments importable
cp .env.example .env        # then edit the paths in .env
```

Then check it works:

```bash
pytest tests/unit -x -q                         # a few minutes on CPU
marimo edit demo/what_is_an_nca.py              # the first tutorial
```

Notes:

- **`.env`** holds your data and model folders and your W&B key. The
  tutorials work without it. Each variable is explained in
  [.env.example](.env.example).
- **Weights & Biases (W&B)** stores training logs. Run `wandb login` once, or
  set `logging.backend: none` in a config to train without it.
- **GPU vs CPU**: on a machine with a GPU, `JAX_PLATFORMS=cpu` forces JAX onto
  the CPU, which is handy for quick checks. `python -c "import jax;
  print(jax.devices())"` shows what JAX can see.
- **Videos** in some notebooks need `ffmpeg` on your system
  (`conda install ffmpeg` or your package manager).
- **Data** is not in git. Only a few example emojis are included
  (`demo/demo_data/`). See [docs/data.md](docs/data.md).

## Layout

| Path | Contents |
| --- | --- |
| `NCA/` | NCA models (`NCA/model/`) and the trainer (`NCA/trainer/`). |
| `Common/` | Shared parts: model base class, spatial operators (Laplacian, gradient, ...), data loaders, losses, W&B logging. |
| `PDE/` | Pattern-forming PDEs (Gray-Scott, Cahn-Hilliard, Keller-Segel, ...) and the solver that turns them into NCA training data. |
| `Experiments/` | One folder per task (`micropatterns/`, `emoji/`, `snowmelt/`, `pde/`, `signalling/`, `impulse/`, `inverse_design/`), each with a `train.py` and YAML configs in `conf/`. Also the model registry and the marimo notebooks used to explore results ([docs/notebooks.md](docs/notebooks.md)). |
| `Experiments/micropatterns/figures/` | Paper figure notebooks ([docs/figures.md](docs/figures.md)). |
| `launch/` | Scripts that run a sweep on the EIDF Kubernetes cluster ([docs/cluster_eidf.md](docs/cluster_eidf.md)) or on one local GPU. |
| `demo/` | Tutorials ([demo/README.md](demo/README.md)). |
| `docs/` | Documentation ([docs/README.md](docs/README.md)). |
| `helper_scripts/` | Copying models back from the cluster, and image labelling. |
| `WebDemo/` | WebGL viewer for exported NCA models. |
| `tests/` | Unit tests (`tests/unit`, CPU), plus a GPU check in `tests/hardware/`. |
| `requirements/` | Install files (see above). |

## Running experiments

An experiment is a YAML config (`Experiments/<task>/conf/base_config.yaml`).
A **sweep file** in `conf/experiments/` lists the settings to vary; it is
turned into a **manifest** (one full config per run), and the manifest is run
on the cluster:

```bash
# 1. Sweep file -> manifest (give ONE sweep file, not the folder)
python Experiments/generate_configs.py \
    Experiments/micropatterns/conf/experiments/my_sweep.yaml \
    Experiments/micropatterns/conf/base_config.yaml \
    Experiments/micropatterns/conf/generated

# 2. Run it on EIDF (Kubernetes): <workers> parallel jobs on H200 GPUs
bash launch/launch_batch_multi_job.sh Experiments/run_config.py \
    Experiments/micropatterns/conf/generated/<experiment>/manifest.yaml <workers> H200

# Or run the entries one after another on a local GPU
python launch/launch_local_sweep.py <manifest>
```

Training logs go to W&B, and each run saves a **model bundle** that can be
loaded later ([docs/model_registry.md](docs/model_registry.md)). The full
workflow, including the traps, is in
[docs/workflow_micropatterns.md](docs/workflow_micropatterns.md).

## Tests

```bash
pytest tests/unit                          # CPU, a few minutes
python Experiments/pipeline_smoke_test.py  # GPU, end to end
```

The smoke test runs very short training runs for every task (emoji,
micropatterns, Nodal knockout fine-tuning, snowmelt, PDEs, signalling,
impulses and inverse design) from the `smoke_*.yaml` sweep files. Each training
run must finish and publish a bundle to the `pipeline-smoke` collection (the
impulse and inverse-design runs use a trained model instead). A sweep is
skipped while its YAML still contains `REPLACE_WITH_PARENT_MODEL_ID`. The
fine-tuning and impulse smoke sweeps name parent models that exist only in the
shared model store. `--dry-run` only writes the manifests and prints the
Kubernetes commands. Set `SMOKE_VERSION` in the script to a label for the
change being tested; it becomes part of the W&B group.

`tests/hardware/jax_gpu_mem_test.py` is a standalone GPU check.

## Using an AI coding assistant

[CLAUDE.md](CLAUDE.md) (also read through [AGENTS.md](AGENTS.md)) gives AI
coding assistants such as Claude Code the rules of this repository. It is also
a compact technical summary for people.

## Citation

Please contact the author for the right citation for a given model or result.
