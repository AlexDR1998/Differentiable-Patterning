# Tips and pitfalls

Things that are easy to get wrong, roughly in the order you will meet them.

## Running code

**Run from the repository root, after `pip install -e .`.** Imports such as
`from NCA.model.factory import build_model` then work from anywhere. Don't
add `sys.path` tricks to scripts.

**`DATA_PATH_BASE` should end in `/`.** The older micropattern loader joins
paths as plain text (`DATA_PATH_BASE + "Timecourse_seperate_colonies/"`).

**Training without W&B.** Set `logging.backend: none` in the config, or
`WANDB_MODE=offline` to log locally and upload later with `wandb sync`.

**Force the CPU.** `JAX_PLATFORMS=cpu python ...` or
`JAX_PLATFORMS=cpu marimo edit ...`. Useful on a laptop with a small GPU, or
to compare results.

## JAX

**Random keys.** JAX has no global random state. Every random function takes
a `key`, and the same key gives the same numbers. Split keys rather than
reusing them: `key, subkey = jax.random.split(key)`. Pass a key to model
functions explicitly: `model(x, key=key)` and `model.run(steps, x, key=key)`.
- Many model functions accept `key=None` and then make a key from the clock.
- That works, but the result can't be repeated, and nothing warns you.
- Always pass a key when the result matters.

**The first call is slow.** JAX compiles a function the first time it sees a
given input shape and setting (**jit**). Later calls are fast. Changing the
batch size, grid size or number of steps compiles again. In a notebook, keep
shapes fixed while you tweak other things.

**Arrays can't be changed in place.** Write `x = x.at[0].set(1.0)`, not
`x[0] = 1.0`.

**Python `if` on array values fails inside compiled code.** Use
`jnp.where(condition, a, b)` or `jax.lax.cond`. Python `if` on config values
(which are fixed numbers) is fine.

**Memory.** By default JAX takes 75% of the GPU memory at start-up. In
notebooks next to other GPU work, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
For long rollouts that run out of memory during training, set
`trainer.loop_autodiff: checkpointed`. It recomputes instead of storing, so
it is slower but uses much less memory.

## Configs and sweeps

**Generate manifests from one sweep file, never a whole folder.** Giving the
folder regenerates, and overwrites, every tracked manifest in it. See
[workflow_micropatterns.md](workflow_micropatterns.md#2-generate-the-manifest).

**Bump `_vN` instead of editing a sweep that has run.** Use the new version
in both `experiment_name` and `output_subdir`.

**Read config values as attributes.** Write `cfg.run.t`, not
`cfg["run"]["t"]` or `cfg.get(...)`. Every field has a value after loading.

**A new setting needs a dataclass field.** Unknown keys are rejected. Add the
field, with a default, to the right dataclass:
- `Experiments/config.py`;
- `NCA/model/config.py`;
- `NCA/trainer/config.py`;
- `Common/trainer/config.py`;
- or the task's `Experiments/<task>/config.py`.

**Old configs are translated in one place.** Configs from older code have a
lower `schema_version`. `Experiments/config.py:upgrade_legacy_config` turns
them into the current layout. If you rename or remove a setting, add the
translation there (and bump `CONFIG_SCHEMA_VERSION`), so old bundles still
load.

**Check a manifest before launching it.** See
[workflow_micropatterns.md](workflow_micropatterns.md#3-check-the-manifest).

## Models and bundles

**Never edit a saved bundle** (`config.yaml`, `manifest.yaml`, `model.eqx`).
Its checksum is checked on loading, and other records point at it. To add
notes or tags, use `registry.annotate(...)`, which stores them separately.

**Changing a model class can break old bundles.** A bundle stores only the
weights. Loading rebuilds the model from its config, then fills in the
weights in order.
- If you add, remove or reorder array fields, or change their shapes, old
  bundles no longer load.
- New options that are plain numbers, flags or strings must be marked
  `eqx.field(static=True)`. Static fields are not written to the weights file.
- `tests/unit/test_model_factory.py` checks the saved layout of the `NCA`,
  `gNCA`, `nNCA` and `gnNCA` families. Run it after any model change.

**Models are immutable.** Equinox models are frozen. To change a weight, make
a new model with `eqx.tree_at(lambda m: m.layers[0].weight, model, new_weight)`.

**Check data against a model before evaluating.** Use
`verify_evaluation_input`; see
[model_registry.md](model_registry.md#checking-evaluation-inputs).

## marimo notebooks

**Each global name can be defined in only one cell.** marimo runs cells in
dependency order, so the same name can't be assigned in two cells. Prefix
names that are only used inside one cell with `_` (`_fig`, `_ax`), and they
become private to that cell.

**Notebooks re-run cells automatically** when something they depend on
changes. Put slow steps behind a `mo.ui.run_button()` and `mo.stop(...)`, as
the existing notebooks do.

**Edit with `marimo edit file.py`.** Run without the editor with
`python file.py`. `marimo check file.py` finds structural mistakes without
running anything.

## Tests

**Unit tests:** `pytest tests/unit` (CPU, a few minutes).
- Run one file while working: `pytest tests/unit/test_loss_table.py -q`.
- `test_trainer_runs.py` sometimes stalls on a laptop. Run it on its own,
  with a time limit: `timeout 600 pytest tests/unit/test_trainer_runs.py`.

**Smoke test** (`Experiments/pipeline_smoke_test.py`): very short real runs
of every task on the cluster. Run it after changing the trainer, models,
losses, data loading or configs.
- Before running it, set `SMOKE_VERSION` in the script to a short label for
  your change (e.g. `new-radial-loss`). It becomes part of the W&B group, so
  the runs are easy to find.
- Then run `python Experiments/pipeline_smoke_test.py --dry-run` to write the
  manifests and print the launch commands.

**Slow numerical checks** of the VGG and OT losses are left to the smoke test,
so unit tests stay fast.
