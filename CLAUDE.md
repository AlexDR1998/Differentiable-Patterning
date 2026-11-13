# CLAUDE.md

Research code for differentiable patterning: neural cellular automata (NCA)
trained on stem-cell micropatterns, emoji, snowmelt maps and simulated PDEs,
built on JAX, Equinox and Optax. Earlier agent-based and neural-PDE work is
only in git history.

## Who you are working with

The people using this repository are **scientists, not software engineers**.
- Explain what you change and why, in plain words. Say what each change means
  for their experiments.
- Prefer small, readable changes over clever ones.
- Ask before large refactors, renames or anything that touches saved models.
- When a request is ambiguous about the science (which loss, which data,
  which models), ask rather than guess.

Concepts and vocabulary: `docs/concepts.md` (glossary at the end). All docs:
`docs/README.md`.

## Ground rules

- **Never run training, sweeps or the smoke test locally.** Training runs on
  the EIDF GPU cluster, which you cannot reach. Write the changes and the
  exact commands, and the person runs them (`docs/cluster_eidf.md`).
- **No git commits or PRs** unless asked. Leave local changes for review.
- **Keep local checks to seconds.** Run single unit test files:
  `pytest tests/unit/test_<area>.py -q`. `tests/unit/test_trainer_runs.py`
  sometimes stalls; run it on its own with `timeout 600`. Slow numerical
  checks (VGG, OT losses) belong in the smoke test, not in unit tests.
- **Don't analyse the user's data.** Build tools for looking at data
  (notebooks, plots); don't render or study data images and report findings
  unless asked.
- The repository is a package: `pip install -e .`, or `PYTHONPATH` set to the
  repo root (the launchers do this). No `sys.path` hacks or absolute paths.
- Don't commit or rely on `.env`, `wandb/`, `models/`, `output/`,
  `figures/`, `logs/` or machine-specific paths. Environment variables are
  listed in `.env.example`.
- Dependencies: `requirements/requirements_common.txt` (pinned), included by
  `requirements_{cpu,gpu}.txt` and `env_{cpu,gpu}.yml`. `pyproject.toml`
  lists none.

## Traps

- **Generate manifests from one sweep file, never a folder.**
  `python Experiments/generate_configs.py <sweep.yaml> <base_config.yaml> <conf/generated>`.
  The folder form regenerates and overwrites every tracked manifest in it,
  and those are often the only record of how registry models were trained.
- **Tracked manifests (`conf/generated/`) are records.** Don't regenerate or
  edit them. To change a sweep that has run, bump `_vN` in both
  `experiment_name` and `output_subdir`.
- **Saved bundles are immutable.** Never edit `config.yaml`, `manifest.yaml`
  or `model.eqx` in the model store.
- **Model field layout is part of the saved format.** New model options must
  be `eqx.field(static=True)` (or arrays). Adding, removing or reordering
  array fields breaks old bundles.
- **Pass explicit PRNG keys.** Many model functions silently seed from the
  clock when `key=None`.
- **marimo notebooks:** a global name can be defined in only one cell; prefix
  cell-local names with `_`. Don't edit text that people wrote in notebooks;
  any text you add should be short bullet points.

## Layout

| Path | Contents |
| --- | --- |
| `NCA/model/` | `NCA` (options `GATED`, `PARAMETER_NOISE_LEVEL`, `FIRING`: families `gNCA`, `nNCA`, `gnNCA`, `aNCA`, `gaNCA`), `SignalNCA` (`sNCA`, `gsNCA`), `HNCA`, `FastKaNCA`. `factory.py:build_model` builds any family from a `ModelConfig`. |
| `NCA/trainer/` | The trainer (see below). `data_augmenter/` has one augmenter per task, built on `base.PoolAugmenter`. `logging/` has the W&B loggers and diagnostic plots. `intervention.py` has knockout rollouts. |
| `NCA/inverse_design/`, `NCA/trainer/impulse/` | Optimising colony geometry (`docs/inverse_design.md`) and impulses (learned perturbations, `impulse/perturbation.py`) for trained NCAs. |
| `Common/model/` | `AbstractModel` (save/load/partition), `spatial_operators.Ops` (ID/LAP/GRAD/DIFF/AV kernels), boundaries, KAN layers. |
| `Common/dataloader/` | Loaders (`micropattern_260726.py` is current; `micropattern.py` is the older dataset), channel schemas (`micropattern_schemas.py`), cleaning steps, `disk_cache.py` (`DATA_CACHE_DIR`, local only). |
| `Common/trainer/` | Losses (`loss.py` generic, `loss_micropattern.py` for the micropattern channel layout, `loss_vgg.py`, `loss_ott.py`), `loss_table.py` (name → function), typed loss configs (`config.py`, `LOSS_TERM_CONFIGS`), `wandb_logger.py`. |
| `PDE/` | Pattern-forming PDEs (`model/fixed_models/`), the diffrax solver, the JAX-free catalogue (`catalogue.py`), `dataset.simulate_trajectories` → `[B, T, C, H, W]`. |
| `Experiments/<task>/` | `emoji`, `micropatterns`, `snowmelt`, `pde`, `signalling`, `impulse`, `inverse_design`: `train.py` (or `optimise.py`), `config.py`, `config_helpers.py` (`load_data`, `build_data_augmenter`), `conf/base_config.yaml`, `conf/experiments/*.yaml` sweeps, `conf/generated/` manifests. |
| `Experiments/` (top level) | `config.py` (typed `ExperimentConfig`, `upgrade_legacy_config`), `config_workflow.py` + `generate_configs.py` (sweep → manifest), `run_config.py` (runs one manifest entry), `nca_training.py`, `model_registry.py` (bundles, `ModelRegistry`, selections), `notebook_jobs.py`, `pipeline_smoke_test.py`. |
| `Experiments/micropatterns/figures/` | Paper figure notebooks with shared `style.py`; colours, fonts and sizes come only from `style.py` (`docs/figures.md`). |
| `launch/` | EIDF Kubernetes launcher (`launch_batch_multi_job.sh`, `run.tpl.yml`, `launch_inside_pod.sh`), local sweep runner; Slurm scripts are unused. |
| `demo/` | Tutorials (marimo), see `demo/README.md`. |
| `docs/` | Documentation, indexed in `docs/README.md`. |
| `WebDemo/` | WebGL viewer for exported models. |

Notebooks: `docs/notebooks.md` lists every marimo notebook and its status.
Legacy ones (`thesis_chapter_*_figures.py`, `nodal_knockout_dev.py`) need old
model files.

## Configs

The flow is: sweep YAML → manifest → `run_config.py` → OmegaConf →
`Experiments.config.load_experiment_config` → frozen dataclasses →
`entrypoint run(cfg)`. Details are in `docs/configuration.md`.

- The dataclasses use the YAML names: grid key `run.t` is `cfg.run.t`.
  Sections are `system`, `data` (with `emoji|micropattern|snowmelt|pde`,
  `knockout`), `model`, `run`, `trainer`, `optimiser`, `loss`, `logging`,
  `model_store`, `initialization`.
- Read fields as attributes; configs have no `.get()`.
- Conversion is strict, and unknown keys fail. A new option needs a field
  (with a default) in the right dataclass:
  - `Experiments/config.py`;
  - `NCA/model/config.py`;
  - `NCA/trainer/config.py`;
  - `Common/trainer/config.py`;
  - or `Experiments/<task>/config.py`.
- Current `schema_version: 10`. Old configs are translated **only** in
  `Experiments/config.py:upgrade_legacy_config`. Renaming or removing a
  setting needs a translation there and a version bump.
- Never pass OmegaConf objects into `Common/` or `NCA/`. `NCA/` and `Common/`
  must not import from `Experiments/`; values from the experiment config
  reach them through `TrainerContext`.
- `Experiments/config.py` must stay importable without JAX.
- Check a manifest locally: load each `manifest.configs[i].config` with
  `load_experiment_config` (snippet in `docs/workflow_micropatterns.md`).

## NCA trainer (`docs/nca_trainer.md`)

The call chain:
```text
Experiments/<task>/train.py:run(cfg)
  → nca_training.run_training
  → build_trainer → NcaTrainer.train → TrainingResult
  → bundle published
```

`NcaTrainer.train` runs these steps:
1. `preparation.prepare_training` (a frozen `PreparedTraining`; it never
   changes the trainer);
2. `setup_logging`;
3. `step.build_train_step`;
4. `runner.run_loop`.

Where the code goes:
- `step.py`: everything JAX traces (rollout, loss, gradient, optimiser
  update).
- `runner.py`: the Python loop, pool admission, validation, logging and
  `BestCheckpoint`. These stay outside differentiated code.
- **User choices** go in frozen config dataclasses, never as new constructor
  or `train()` arguments.
- **`TrainerContext`** holds only runtime values derived from data:
  boundaries, channel schemas, masks, names, paths, provenance and W&B tags.
- **Augmenters** follow `data_augmenter/protocols.py:NCAAugmenterProtocol`
  (`initialize_pool`, `advance_pool(x, y, i, key)`, `return_saved_data`,
  `return_observed_data`).
  - Each task builds one with `build_data_augmenter` and passes it in
    `TrainerContext.data_augmenter`.
  - Options are constructor arguments.
  - All randomness comes from the supplied key.
  - Data is `[batch, time, channels, H, W]`, and `data[:, 0]` is the initial
    condition.

## Common tasks

- **Add a loss:**
  - write the function (`Common/trainer/loss.py`, or `loss_micropattern.py`
    if it depends on the channel layout);
  - add it to `Common/trainer/loss_table.py:LOSSES` and
    `Common/trainer/config.py:LOSS_TERM_CONFIGS` (an import-time check
    compares them);
  - add a test to `tests/unit/test_loss_table.py`;
  - add a row to `docs/losses.md`.
- **Add a regulariser:**
  - write the function in `NCA/trainer/NCA_regulariser.py`;
  - add its name to `NCA/trainer/preparation.py:_regularisers`;
  - add it to `docs/losses.md`.
- **Add a model option:**
  - add a constructor argument stored as `eqx.field(static=True)`;
  - add a `ModelConfig` field;
  - pass it through in `NCA/model/factory.py`. A small variant of `NCA` is an
    option plus a `MODEL_FAMILIES` entry (and `build_model_config_string`),
    not a new class;
  - run `tests/unit/test_model_factory.py`.
- **Add a sweep:** copy the nearest file in `conf/experiments/`, give it a
  new `_v1` name, generate its manifest from that one file, and check it.
- **Add a figure:** a marimo notebook in `Experiments/micropatterns/figures/`
  using `style.py` (`paper_style`, `WIDTH`, `save_figure`); add it to
  `docs/figures.md`.
- **Run a heavy notebook section remotely:** see
  `docs/remote_notebook_jobs.md` (examples: `micropatterns/density_ko_sweep.py`,
  `size_density_sweep.py`).

## Checking a change

- **Unit tests:** in `tests/unit/test_*.py`. Add or extend tests when you
  change config conversion, losses, augmenters, models or trainer components.
  Run the relevant file(s) only.
- **Notebooks:** `marimo check <file>` after editing one.
- **Trainer, models, losses, data or configs:** after changing any of these,
  set `SMOKE_VERSION` in `Experiments/pipeline_smoke_test.py` to a label for
  the change (e.g. `new-radial-loss`). Regenerate the smoke manifests with
  `python Experiments/pipeline_smoke_test.py --dry-run`, and ask the person
  to run the smoke test on EIDF.
- **Models:** after any model change, check that old bundles still load:
  `tests/unit/test_model_factory.py`, and `bundle.load_model()` on a real
  bundle if one is available locally.

## Code style

- Clarity first: this is research code that new scientists will read.
  - Prefer simple functions and composition to class hierarchies.
  - Avoid design patterns that aren't needed.
  - Keep comments and docs short and free of software-engineering jargon.
- Models and trainers stay `equinox.Module` and PyTree compatible. Avoid
  mutable state or side effects that break `jit`, `vmap` or `grad`. Use
  `eqx.filter_*` and `eqx.partition`/`combine`.
- Classes are CamelCase; files and functions snake_case.
- New code uses 4-space indentation. Some older files use tabs: match the
  file and don't re-indent whole files.
- British spelling in new names (`optimiser`, `normalise`), except where an
  existing API uses US spelling (`initialize_pool`).
