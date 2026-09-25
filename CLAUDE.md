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
| `NCA/model/` | `NCA` (with `GATED` and `PARAMETER_NOISE_LEVEL` options; the families `gNCA`, `nNCA` and `gnNCA` are these options switched on), KAN-based `FastKaNCA` and hierarchical `HNCA`. |
| `NCA/trainer/` | The active modular trainer (see below). |
| `NCA/inverse_design/`, `NCA/trainer/impulse/` | Optimisation of trained NCAs for micropattern geometry and impulses/initial state (the learned perturbation module is `impulse/perturbation.py`). |
| `Experiments/` | Config-driven entrypoints for each domain: `emoji/`, `micropatterns/`, `snowmelt/`, `impulse/`. `model_registry.py` is the local model-bundle registry and its CLI (`ModelRegistry`, `create_model_id`, `record_evaluation`, `verify_evaluation_input`). |
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
- The typed config uses the same names as the YAML files: `cfg.system`, `cfg.data` (with `data.emoji|micropattern|snowmelt` and `data.knockout`), `cfg.model`, `cfg.run`, `cfg.trainer`, `cfg.optimiser`, `cfg.loss`, ... So the grid key `run.t` is `cfg.run.t`. Read fields as attributes; configs have no `.get()`.
- Current files have `schema_version: 3`. Older configs (old sweeps, manifests and older model bundles) are translated only in `Experiments/config.py:upgrade_legacy_config`. Don't add compatibility code anywhere else.
- Conversion is strict: unknown keys and unknown model families fail early. A new option needs a field in the right dataclass. The main ones are in `Experiments/config.py`, `NCA/model/config.py`, `NCA/trainer/config.py` and `Common/trainer/config.py`.
- Never pass OmegaConf objects into `Common/` or `NCA/`. Pass only typed configs.
- Model families are built in `NCA/model/factory.py:build_model`. A new family goes into its `MODEL_FAMILIES` table and `build_model_config_string`; a small variant of `NCA` should be a constructor option plus a table entry, not a new class. New model options must be `eqx.field(static=True)` (or arrays), because every Python int/float/bool field is written to saved `.eqx` files and changing that layout breaks old bundles. Retired families (`NCA_sycl`, `NCA_fast`, `gNCA_sycl`) map to current ones through `PORTABLE_MODEL_FAMILIES`. `Experiments.config_helpers` re-exports `build_model` only because old bundles record that path as their factory.
- `NCA/` and `Common/` must not import from `Experiments/`. Values that come from the experiment config (e.g. W&B tags) are passed in through `TrainerContext`.

## NCA trainer architecture (`docs/nca_trainer.md`)

The flow is: entrypoint → `TrainerContext` + `ExperimentConfig` → `Experiments/nca_training.py:run_training` → `build_trainer` → `NcaTrainer.train` → `TrainingResult` → optional bundle publication.

- User choices belong in frozen config dataclasses. Don't add new constructor or `train()` kwargs for options.
- `TrainerContext` (`NCA/trainer/context.py`) holds only runtime values derived from data: boundaries, channel schemas, masks, names, paths and provenance.
- `NcaTrainer.train` runs `preparation.prepare_training` (returns a frozen `PreparedTraining`; never changes the trainer) → `setup_logging` → `step.build_train_step` → `runner.run_loop`. `step.py` holds everything JAX traces (`batch_model`, `run_nca_steps`, `batch_loss`); `runner.py` holds the Python loop, pool admission and `BestCheckpoint`. See `docs/nca_trainer.md` for the full file table.
- `NcaTrainer` keeps only the config sections it uses (`run_config`, `trainer_config`, `optimiser_config`, `loss_config`, `logging_config`, `knockout_config`), not the whole `ExperimentConfig`.
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
