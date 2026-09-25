# CLAUDE.md

Research code for differentiable patterning: neural cellular automata (NCA), built on JAX, Equinox and Optax. Earlier PDE and agent-based work lives only in git history. See also `AGENTS.md`, `README.md` and `docs/`.

## Ground rules

- **Never run experiments or training scripts locally.** Training needs remote GPUs (Slurm / Kubernetes). Propose edits only. A human checks them and runs them remotely.
- **No git commits or PRs.** Leave local diffs for review.
- Safe to run locally: `pytest tests/unit` (CPU, lightweight). `tests/hardware` needs a GPU.
- Don't commit or rely on `.env`, `wandb/`, `models/`, `output/`, `Videos/`, `figures/`, `logs/` or machine-specific paths.
- Dependencies live in `requirements/` (`requirements_{cpu,gpu}.txt`, `env_{cpu,gpu}.yml`), not in the repo root as `AGENTS.md` says.

## Layout

| Path | Contents |
| --- | --- |
| `Common/` | Shared code. `model/` has `AbstractModel` (eqx.Module base with save/load/partition), `spatial_operators.Ops` (ID/LAP/GRAD/DIFF/AV kernels, padding), boundaries and KAN layers. `dataloader/` has emoji, micropattern, snowmelt and texture loaders plus channel schemas. `trainer/` has losses (`loss*.py`), typed loss configs (`config.py`), abstract augmenters and loggers. |
| `NCA/model/` | NCA variants: `NCA`, gated `gNCA`, noise `nNCA`/`gnNCA`, `FastKaNCA`, hierarchical `HNCA`, multiscale, attention, wavelet and others. |
| `NCA/trainer/` | The active modular trainer (see below). |
| `NCA/inverse_design/`, `NCA/trainer/impulse/` | Optimisation of trained NCAs for micropattern geometry and impulses/initial state. |
| `NCA/registry.py` | Local model-bundle registry (`ModelRegistry`, `create_model_id`, `record_evaluation`, `verify_evaluation_input`). |
| `Experiments/` | Config-driven entrypoints for each domain: `emoji/`, `micropatterns/`, `snowmelt/`, `impulse/`. |
| `demo/`, `notebooks/`, `WebDemo/` | Walkthroughs (marimo), Jupyter notebooks and a static WebGL viewer with an exporter. |

## Experiment workflow

1. Each domain has `Experiments/<domain>/conf/base_config.yaml` plus sweep files in `conf/experiments/*.yaml`. A sweep file has `experiment_name`, `entrypoint` (such as `Experiments.emoji.train:run`), `output_subdir`, `seed_mode` and a `grid:` of dotted overrides expanded as a Cartesian product. Group and condition blocks are handled in `Experiments/config_workflow.py`.
2. `python Experiments/generate_configs.py <experiments_dir> <baseline_yaml_or_dir> <output_dir>` writes a manifest into `conf/generated/<name>/`.
3. `Experiments/run_config.py --manifest M --index i` (or `--config-file F --entrypoint mod:fn`) does the following:
   - resolves OmegaConf
   - applies environment overrides (`MODEL_STORE_ROOT`, XLA flags)
   - converts the config to typed frozen dataclasses with `Experiments.config.load_experiment_config`
   - calls the entrypoint `run(cfg)`
4. The remote launchers wrap step 3:
   - `launch_batch_slurm.sh SCRIPT MANIFEST` submits a Slurm array that runs `launch_slurm.sh`.
   - `run.tpl.yml` with `launch_inside_pod.sh` runs indexed Kubernetes jobs.
   - `launch_local_sweep.py` runs entries one after another on one GPU.

Config rules (`docs/configuration.md`):
- Legacy top-level YAML sections `run`, `trainer`, `optimiser` and `loss` map into `training.loop`, `training.trainer`, `training.optimizer` and `training.loss`. So grid keys like `run.t` and `optimiser.learn_rate` are valid. `ExperimentConfig` exposes `.run` and `.optimiser` aliases.
- Conversion is strict: unknown keys and unknown model families fail early. A new option needs a field in the right dataclass. The main ones are in `Experiments/config.py`, `NCA/model/config.py`, `NCA/trainer/config.py` and `Common/trainer/config.py`.
- Never pass OmegaConf objects into `Common/` or `NCA/`. Pass only typed configs.
- Model families are built in `Experiments/config_helpers.py:build_model`. A new family goes there and into `build_model_config_string`. Archived `*_sycl` families map to portable ones through `PORTABLE_MODEL_FAMILIES`.

## NCA trainer architecture (`docs/nca_trainer.md`)

The flow is: entrypoint → `TrainerContext` + `ExperimentConfig` → `Experiments/nca_training.py:run_training` → `build_trainer` → `NcaTrainer.train` → `TrainingResult` → optional bundle publication.

- User choices belong in frozen config dataclasses. Don't add new constructor or `train()` kwargs for options.
- `TrainerContext` (`NCA/trainer/context.py`) holds only runtime values derived from data: boundaries, channel schemas, masks, names, paths and provenance.
- Responsibilities are split across files:
  - `objective.py`: resolves loss terms and regularisers
  - `pool.py`: pool admission
  - `checkpointing.py`: best-checkpoint policy
  - `step.py` and `runner.py`: compiled step and loop
  - `loss_schedule.py`: loss-weight schedules
  - `instrumentation.py`: profiling
  - `trainer.py`: public boundary and logger selection
- The compiled step covers rollout, gradient and optimiser update. Logging, pool admission and checkpoint decisions stay in Python, outside differentiated code.
- Data augmenters follow `NCA/trainer/data_augmenter/protocols.py:NCAAugmenterProtocol` (`data_init`, `initialize_pool`, `advance_pool(x, y, i, key)`, `return_saved_data`). All randomness must come from the supplied key. Trajectory data is batched as `[batch, time, channels, H, W]`, and `data[:, 0]` is the initial condition.

## Model registry (`docs/model_registry.md`)

When `model_store.enabled` is set, training publishes an immutable bundle to `bundles/<collection>/<experiment>/<slug>--<id>/`. It contains `model.eqx`, `config.yaml` and `manifest.yaml`. The SQLite index is rebuilt with `python -m Experiments.model_registry reindex`, and `list` and `show <id>` inspect it. `bundle.load_model()` rebuilds the model from its saved `ModelConfig` through the recorded factory and checks the checksum. Saved configs must never be edited.

## Code conventions

- Models and trainers must stay `equinox.Module` and PyTree compatible. Avoid mutable state or Python side effects that break `jit`, `vmap` or `grad`. Use `eqx.filter_*` and `eqx.partition`/`combine` (see `AbstractModel.partition`).
- Classes use CamelCase. Files and helpers use snake_case, grouped by prefix (`data_augmenter_*`, `update_*`, `*_trainer.py`, `loss_*.py`).
- New code uses 4-space indentation and explicit imports. Many older model and trainer files use tabs. Match the file you are editing and don't reindent whole files.
- **marimo notebooks** (`*.py` with `@app.cell`, found in `demo/`, `Experiments/*/*_explorer.py`, `thesis_chapter_*_figures.py` and `WebDemo/marimo_app.py`): a global name can be defined in only one cell. Prefix cell-local names with `_`. `marimo_utils.py` holds shared helpers.
- Tests are in `tests/unit/test_*.py` (pytest). Add or extend unit tests for config conversion, losses, augmenters and trainer components when you change them.


## Style and aesthetic
- This is a research codebase for academic use, not production software. Clarity, simplicity and ease-of-understanding (especially to potentially new 3rd parties) is crucial, so where possible, avoid unnecessarily complex design patterns
- For any documentation or comments, avoid overly verbose software engineering terms
- In general a compositional / functional programming approach is preferred over object inheretence, but this is not strict - instead prioritise using the design pattern that most cleanly matches the problem definition
