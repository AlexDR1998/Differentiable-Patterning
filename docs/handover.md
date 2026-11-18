# Handover notes

Where the project stood at handover (November 2026), and what is known to be
unfinished. **TODO(handover)** marks sections for the previous owner to
fill in.

## Project threads

For each thread: the question, what has been done, where the models and
results are, and what is next.

### Micropattern NCA paper

- **Question:** Can NCA be trained to capture pattern formation dynamics from real biological images (yes)? Can these models be useful for predicting and extrapolating new experimental settings (maybe)?
- **Models:** [Experiments/model_registr_explorer.py](Experiments/model_registr_explorer.py) is used to filter, select and bundle (as a yaml file) groups of models. It filters by any configuration tag from the training config yaml files.
- **Figures:** see [figures.md](figures.md)
- **Open sweeps:** TODO(handover). Knockout time × density, and size ×
  density inference sweeps (`Experiments/micropatterns/conf/generated/density_ko/`,
  `size_density/`).
- **Next steps:** Write paper given existing figures. Also see Inverse Design subsection

### Impulses

Work with Can. Finding impulses (bounded/local in space and time) to NCA trajectory rollouts that induce transitions between multiple stable attractor states.

 critical-time sweeps, emoji attractor switching.

### Inverse design

Gradient based optimisation of initial conditions (currently just micropattern geometry), to maximise the production of specific cell types. See [inverse_design.md](inverse_design.md).

### Signalling NCAs, snowmelt, PDEs

Some half-baked ideas in [Experiments/signalling](Experiments/signalling) - specifically introducing different biologically motivated signalling mechanisms to specific NCA channels.

Snowmelt - this will be it's own minimal git branch, so painter et al can more easily use it


## Python Environment

- For running on EIDF, you need a docker image. The existing Docker Hub image `alexdr1998/jax_equinox_scratch` currently handles all the python dependancies on EIDF for this codebase. For any major changes (i.e. new python library has a feature we really need), I suggest learning to build your own docker image.
- For running locally, the base requirements are in [requirements/requirements_common.txt](requirements/requirements_common.txt). [TODO-LLM] add pip install instructions here

## Known issues in the code

These are known and not yet fixed. None of them changes current results, but
they are worth knowing before you build on the code.

- **Clock-seeded random keys.** About 13 functions
  (`NCA.__init__`, `NCA.__call__`, `NCA.run`, and the same in `HNCA`,
  `SignalNCA`, `FastKaNCA` and a few loaders) make a key from the clock when
  given `key=None`, with no warning. Good practice with JAX code is to akways explicitely declare a random number key
- **Loss names.** `sliced_wasserstein_full` and `spectral_wasserstein_full`
  (`Common/trainer/loss.py:wasserstein_projected`,
  `spectral_wasserstein_projected`) don't sort their projections, so they
  are not Wasserstein distances. Currently unused.
- `multi_target` (`NCA/trainer/loss_multi_target.py`) works for the number of 
  data replicates we train on now, but will scale poorly for larger batch sizes.
  This is due to comparing each combination of prediction and data - we probably 
  need a smarter optimal transport style comparison before scaling to many replicates

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
