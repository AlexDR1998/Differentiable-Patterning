# Concepts

The ideas behind the code, and the words used for them. The tutorial
`demo/what_is_an_nca.py` shows the first two sections in code.

## What a neural cellular automaton is

An NCA is a grid of cells. Each cell holds a vector of numbers, its
**channels**. The whole state is an array `[channels, height, width]`.

At every **step**, each cell updates itself from what it and its immediate
neighbours hold, using the same small neural network everywhere:

1. **Perception.** Fixed convolution kernels look at each channel's
   neighbourhood (`Common/model/spatial_operators.py:Ops`):
   - `ID`: the cell's own value;
   - `LAP`: the Laplacian (how different a cell is from the average of its neighbours);
   - `GRAD`: the gradient in x and y;
   - `DIFF`: the size of the gradient;
   - `AV`: the local average.

   `model.kernel_str` picks which ones are used. With `[ID, LAP]` and 16
   channels, each cell perceives 32 numbers.
2. **Update network.** A two-layer network with 1×1 convolutions (so it acts
   on each cell separately) turns the perception into a change `dx` for every
   channel. The last layer starts at zero, so a new model changes nothing.
3. **Stochastic update.** Each cell adds `dx` with probability
   `model.fire_rate`, so cells update out of sync. With `fire_rate: 1` every
   cell updates every step.
4. **Boundary.** An optional `boundary_callback` is applied after each step,
   for example to keep cells outside the colony at zero.

The code is in `NCA/model/NCA_model.py` (`NCA.__call__` is one step,
`NCA.run(steps, x, key=key)` is a whole trajectory).

Because one step only reaches the nearest neighbours, patterns that span the
grid need many steps. Information spreads by about one cell per step.

### Observed and hidden channels

The first channels of the state are compared with data: the **observed
channels**, for example the 4 RGBA channels of an emoji or the stained
markers of a micropattern. The remaining **hidden channels** are free for the
model to use as internal signals, and are never compared with anything. The
model's `channels` counts both. With `channels: 48` and the 10 micropattern
markers observed, 38 channels are hidden.

## Model families

`model.family` picks the model, built by `NCA/model/factory.py:build_model`.

| Family | What it adds | Notes |
| --- | --- | --- |
| `NCA` | The plain model above. | |
| `gNCA` | A gated last layer: two outputs per channel combined by a gated linear unit. | Used for the micropattern paper models (`micropattern_projections_training_sweep.yaml`). |
| `nNCA`, `gnNCA` | Gaussian noise (`model.parameter_noise_level`) added to the weights at every step. | Makes models robust to small changes. |
| `aNCA`, `gaNCA` | Each cell learns its own firing probability, stored in the last channel. | `model.firing_estimator`; the `firing_energy` regulariser charges for updates. |
| `sNCA`, `gsNCA` | Signal channels that a fixed rule moves along "wires" the NCA grows (cable, axon, steady, excitable, broadcast or pointer). | `NCA/model/NCA_signalling.py`; long-range signalling tasks in `Experiments/signalling/`. |
| `HNCA` | A second, coarser NCA (the parent) coupled to the main one. | `NCA/model/NCA_hierarchical.py`. |
| `FastKaNCA` | The update network uses KAN layers (learned 1D functions on each edge) instead of a plain network. | `NCA/model/NCA_fast_KAN_model.py`; options in `model.kan`. |

`gNCA`, `nNCA`, `aNCA` and their combinations are the same `NCA` class with
options switched on.

## Data as trajectories

All training data has the shape `[batch, time, channels, height, width]`:
- `batch` trajectories (for micropatterns, replicate colonies);
- `time` snapshots, where `data[:, 0]` is the initial condition and later
  snapshots are targets;
- `channels`, the observed channels.

For micropatterns the snapshots are 0, 12, 24, 36 and 48 h.

`run.t` is the number of NCA steps between two snapshots. Training runs the
NCA for `run.t` steps from one snapshot and compares the result with the next
one.

## Training: losses, pool and augmenter

**Loss.** A loss compares the NCA's observed channels with the target
snapshot. It ranges from pixel-by-pixel (`l2`) to comparisons of texture and
radial profiles that don't need pixels to line up (`multi_target`, `vgg`,
`ott`). `loss.terms` lists them with weights. **Regularisers**
(`loss.regularisers`) are extra penalties that don't use the data, for
example keeping states in [0, 1]. Every loss and regulariser is listed in
[losses.md](losses.md).

**Pool.** The NCA has to keep working when run for a long time, not just for
one stretch of `run.t` steps. So the states the NCA reaches are kept in a
**pool** and used as the starting point of later training steps. After each
step a result is admitted back into the pool only if its loss is not much
worse than recent ones (**pool admission**, `NCA/trainer/pool.py`).

**Augmenter.** The augmenter (`NCA/trainer/data_augmenter/`) owns the data
and the pool.
- It adds the hidden channels (zeros) to the data.
- After every training step it decides what the next batch starts from: the
  pool, fresh copies of the true data (**reinjection**), noise or damage.
- Each task has its own augmenter (`EmojiAugmenter`, `MicropatternAugmenter`,
  ...).

**Training loop.** `NCA/trainer/` runs the loop:
- roll out the NCA from the pool states;
- compute the loss;
- take a gradient step with Optax;
- update the pool;
- log to W&B.

The best model is saved as a **bundle**. [nca_trainer.md](nca_trainer.md)
maps the files.

## Knockouts

A knockout (KO) imitates blocking a signal experimentally. From a chosen time
on, one channel, usually NODAL, is hidden from the NCA's perception: the NCA
can no longer read it. `data.knockout` sets which channel and when. The
knockout data (Nodal inhibited from 0 h or 24 h) trains or tests whether the
model reacts the way the cells did. The rollout code is in
`NCA/trainer/intervention.py`.

## PDEs as training data

`PDE/` simulates classic pattern-forming equations:
- Gray-Scott and Chhabra (reaction-diffusion spots and stripes);
- Cahn-Hilliard (phase separation);
- heat;
- Hillen-Painter and Keller-Segel (chemotaxis).

`PDE/dataset.py:simulate_trajectories` turns them into `[B, T, C, H, W]`
trajectories, so an NCA can be trained to copy a known mechanism. This is a
useful test of what NCAs can learn. See the tutorial
`demo/pde_patterns_demo.py`.

## Configs, sweeps and bundles

- **Config.** Every run is described by one YAML config, read into typed
  Python dataclasses (`cfg.model.channels`, `cfg.run.t`, ...). See
  [configuration.md](configuration.md).
- **Sweep.** A sweep file lists the settings to vary.
- **Manifest.** A manifest holds the full config of every run of a sweep.
  The cluster launcher runs each manifest entry as one job.
- **Bundle.** A trained model is saved as a bundle: weights, the full config,
  and a record of the code and data it came from. The **model registry**
  indexes the bundles so you can search and load them. See
  [model_registry.md](model_registry.md).

## Glossary

| Term | Meaning |
| --- | --- |
| **260726** | The name of the current micropattern dataset (`$DATA_PATH_BASE/260726_nca_dataset/`, `data.dataset: micropatterns_260726`). See [data.md](data.md). |
| **augmenter** | The object that holds the training data and the pool, and prepares each training batch. |
| **bundle** | A saved trained model: `model.eqx`, `config.yaml`, `manifest.yaml`. Never edited after it is written. |
| **channel schema** | The list of what each data channel is (marker name, staining panel), `Common/dataloader/channel_schema.py`. |
| **collection** | A group of bundles in the model store (`model_store.collection`), e.g. `pipeline-smoke`. |
| **EIDF** | The Edinburgh International Data Facility, whose Kubernetes GPU cluster runs the training. See [cluster_eidf.md](cluster_eidf.md). |
| **entrypoint** | The function a run calls, e.g. `Experiments.micropatterns.train:run`. |
| **fire rate** | The probability that a cell updates at a step. |
| **hidden channels** | State channels not compared with data. |
| **jit / tracing** | JAX compiles a function the first time it is called with a given input shape. A new shape or setting means it compiles again, which is slow. |
| **KO** | Knockout: a channel (usually NODAL) is hidden from perception from a set time on. |
| **manifest** | `manifest.yaml`, the list of full configs generated from a sweep file. |
| **marimo** | The notebook tool used here. Notebooks are plain `.py` files: `marimo edit file.py`. |
| **model store** | The folder holding bundles (`MODEL_STORE_ROOT`). |
| **NODAL** | A signalling protein; also the name of its channel in the micropattern data. |
| **observed channels** | State channels compared with data. |
| **pool** | The saved NCA states that later training steps start from. |
| **projection** | How a 3D image stack becomes 2D: `max`, `sum` or one `slice` (`data.micropattern.projection`). |
| **PVC** | A Kubernetes storage volume (persistent volume claim). On EIDF, `$USER-ws1` holds data, models and outputs. |
| **PyTree** | JAX's word for nested lists/dicts/objects of arrays. Equinox models are PyTrees. |
| **registry** | The searchable index of the model store (`registry.sqlite`, `Experiments/model_registry.py`). |
| **replicate** | One colony, i.e. one trajectory in the batch. |
| **rollout** | Running the NCA for some steps from a state. |
| **selection** | A YAML list of chosen model IDs, exported from `model_registry_explorer.py`, which figure notebooks read. |
| **static field** | A model setting stored outside the weights (`eqx.field(static=True)`). Changing which fields are static breaks loading old bundles. |
| **sweep** | A YAML file listing settings to vary (`conf/experiments/*.yaml`). |
| **W&B** | Weights & Biases, the website where training logs and plots go. |
