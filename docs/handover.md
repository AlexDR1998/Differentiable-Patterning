# Handover notes

Where the project stood at handover (November 2026), and what is known to be
unfinished. **TODO(handover)** marks sections for the previous owner to
fill in.

## Project threads

For each thread: the question, what has been done, where the models and
results are, and what is next.

### Micropattern NCA paper

- **Question:** TODO(handover)
- **Models:** TODO(handover). Which experiments and selection files the
  paper figures use.
- **Figures:** see [figures.md](figures.md). TODO(handover): which are
  final, which still need work.
- **Open sweeps:** TODO(handover). Knockout time × density, and size ×
  density inference sweeps (`Experiments/micropatterns/conf/generated/density_ko/`,
  `size_density/`).
- **Next steps:** TODO(handover)

### Impulses

TODO(handover): critical-time sweeps, emoji attractor switching.

### Inverse design

TODO(handover); see [inverse_design.md](inverse_design.md).

### Signalling NCAs, snowmelt, PDEs

TODO(handover)

## People and accounts

- EIDF project and PI: TODO(handover)
- W&B entity and projects: TODO(handover)
- Docker Hub image `alexdr1998/jax_equinox_scratch`: TODO(handover). Who
  can push to it, and where its Dockerfile is.
- Where the raw data comes from, and who to ask about it: TODO(handover)

## Known issues in the code

These are known and not yet fixed. None of them changes current results, but
they are worth knowing before you build on the code.

- **Clock-seeded random keys.** About 13 functions
  (`NCA.__init__`, `NCA.__call__`, `NCA.run`, and the same in `HNCA`,
  `SignalNCA`, `FastKaNCA` and a few loaders) make a key from the clock when
  given `key=None`, with no warning. Always pass a key when results must be
  repeatable.
- **Loss names.** `sliced_wasserstein_full` and `spectral_wasserstein_full`
  (`Common/trainer/loss.py:wasserstein_projected`,
  `spectral_wasserstein_projected`) don't sort their projections, so they
  are not Wasserstein distances. `LPIPS_OT_SP` in `Common/trainer/loss_vgg.py`
  sorts along the channel axis after a spatial projection; check that this
  is intended.
- **Unchecked axes.** `Common/model/spatial_operators.py` has
  `TODO: check the axes are correct here` in a vector operator used only by
  the (unused) Navier-Stokes model.
- **Smoke tests depend on the shared store.**
  `smoke_micropatterns_ko_finetune.yaml` and `smoke_impulse.yaml` name parent
  model IDs that exist only in the current model store. With a new store,
  replace them with models trained by the emoji and micropattern smoke runs.
- **Some sweep files no longer match their tracked manifests.** The sweep was
  edited after it ran. The manifest in `conf/generated/` is the record of
  what actually ran. `nca_loss_cache_debug.yaml` no longer converts.
- **One manifest folder is shared by two sweeps.**
  `emoji_multi_attractor_outsource.yaml` and
  `emoji_multi_attractor_stability_data_wide_sweep.yaml` both write to
  `generated/emoji_multi_attractor_outsource/`.
- **Machine paths in manifests.** Older manifests store `sweep_file` and
  `base_config` as absolute paths on the previous owner's laptop. These are
  only records and are not used when running.

## Possible clean-up

Planned but not done, so as not to disturb work in progress:

- **Remove dead code:**
  - unused modules (`Common/trainer/custom_functions.py`, the Gray-Scott
    ensemble and Navier-Stokes PDE models);
  - unused functions in `Common/utils.py`;
  - about 190 lines of commented-out code and leftover debug prints.
- **Archive old code** to an `archive/` folder:
  - the thesis notebooks;
  - the thesis-only parts of `Common/dataloader/micropattern.py`;
  - the Slurm launchers.
- **Make `generate_configs.py` safer:** refuse to overwrite an existing
  manifest, and add a `--check` option that runs the manifest check.
- **Warn instead of silently seeding from the clock** when `key=None`.
- **Add docstrings** with array shapes to `FastKaNCA`, `PDE_solver`, the
  `NCA` class and the config dataclasses.
- **Larger refactors** that need smoke-test reruns:
  - one rollout function instead of several;
  - merge the near-identical PDE and snowmelt augmenters;
  - split the 480-line `load_micropattern_260726`;
  - consistent indentation (some older files use tabs).
