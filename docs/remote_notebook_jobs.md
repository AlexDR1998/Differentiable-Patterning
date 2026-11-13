# Running a notebook section on the cluster

Some notebook sections (e.g. the density × KO sweep in
`Experiments/micropatterns/nodal_ko_eval.py`) take hours on a laptop GPU.
The pattern below runs just that section on EIDF (or Slurm) with the existing
launchers, and loads the results back into the notebook.

## Pattern

- **Move the heavy loop into a plain module** next to the notebook
  (example: `Experiments/micropatterns/density_ko_sweep.py`).
  - Settings are a frozen dataclass (`DensityKoSettings`), and the work is a
    function `run_<thing>(settings, records, ...) -> results`.
  - The module does not import marimo.
- **The notebook keeps the UI and the plots.** It builds the settings from its
  widgets, and then either:
  - runs the function itself ("Run here"), or
  - writes a job manifest ("Export remote job").
- **The job manifest is an ordinary sweep manifest** (`experiment_name`, `count`,
  `configs[].config`), so `launch/` runs it unchanged.
  - Each entry is one part of the work, e.g. one model.
  - Each part writes `part_<index>.npz` into a shared folder.
  - Helpers: `Experiments/notebook_jobs.py` (`write_job_manifest`,
    `read_job_entry`, `part_path`, `part_paths`).
- **The module's `main()`** reads its entry with `read_job_entry`, which picks
  the part from `JOB_COMPLETION_INDEX` / `JOB_WORKER_*` as for training, or
  from `--index`.
  - It writes its part, and skips the part if the file already exists, so
    failed pods can simply be relaunched.
- **Save summaries, not trajectories**: counts, metrics and a few example
  frames.
- **The loader accepts a single file or a folder of parts.**

Things to keep in mind:
- The repository is mounted read-only in pods. Write results under
  `/workspace/writeable/`.
- In pods, `DATA_PATH_BASE` and `MODEL_STORE_ROOT` replace the notebook's
  data and model-store paths.
- Paths inside the repository (e.g. cell type files) are stored relative to
  the repository root.
- Manifests go in `Experiments/<domain>/conf/generated/...` (tracked by git),
  not in `output/` (ignored).
- To keep results identical to a single local run, each part needs to know
  where its seeds start (e.g. `job_offset` in the density × KO sweep).

## Density × KO sweep: steps

1. In `nodal_ko_eval.py`, set up the sweep. Then, under "Run the sweep
   remotely", choose the parts and click "Export remote job". This writes
   `Experiments/micropatterns/conf/generated/density_ko/<name>/manifest.yaml`.
2. Optional: test one part locally:
   `python Experiments/micropatterns/density_ko_sweep.py --manifest <manifest> --index 0`
3. Sync the checkout to the cluster, then run:
   `bash launch/launch_batch_multi_job.sh Experiments/micropatterns/density_ko_sweep.py <manifest> <workers> H200`
4. Copy `/workspace/writeable/output/nodal_ko_eval/<name>/` back to
   `output/nodal_ko_eval/<name>/`, and load that folder in the notebook.
   `helper_scripts/copy_models_from_cluster.sh --port <port>`
   does this: it copies everything in the cluster's `output/` along with the models.

Rollouts of all runs × densities of one KO time go to the GPU together, in
chunks of "Rollouts per GPU batch" (`max_batch`).

## Size × density sweep

`Experiments/micropatterns/size_density_explorer.py` (module `size_density_sweep.py`) sweeps the colony diameter (µm) and cell density, with optional NODAL KO hours. Initial states come from the same cell placement as the density × KO sweep (`density_ko_sweep.prepare_model_cells`). Each run keeps radial profiles over time (kymographs) as well as the end-time ring counts.

1. Under "Run the sweep remotely", export the job. This writes `Experiments/micropatterns/conf/generated/size_density/<name>/manifest.yaml`.
   - Parts can be one per model, per (model, replicate) or per (model, replicate, size).
   - Every part uses the grid of the largest colony, so split parts match a single run.
2. Launch it with:
   `bash launch/launch_batch_multi_job.sh Experiments/micropatterns/size_density_sweep.py <manifest> <workers> H200`
3. Copy `/workspace/writeable/output/size_density/<name>/` back to `output/size_density/<name>/`, then load that folder in the notebook.

The pods need the stored nucleus segmentation (`nucleus_segmentation.py`) under the data root, as the density × KO sweep does.

## References in both sweeps

Both sweeps can also roll out two references, switched on under
"References" when setting the sweep up. They use the same KO times (or
conditions) and run seeds as the sweep.

- **True data**: the 0 h control colony of each replicate in "Replicates",
  in its own outline (and, in the size sweep, its own size), with that
  replicate's run seeds. A replicate whose colony cannot be loaded is
  skipped with a warning.
- **Synthetic at the measured density**: the sweep's cells placed at the
  measured density, the mean over the sweep's replicates (in the size
  sweep, at every size).

They are saved beside the sweep's results with `source` set to
`true_data` or `measured_density` (`sweep` for the sweep itself; older files
load as `sweep`). In a split job every part averages the measured density
over all the replicates (`measured_replicates`) and runs the true data of
its own replicates (in a job split by size, in its first size part only).
The plots pool the replicates' true colonies. The plots show the references when they are
chosen under "References". True data has no cell counts, so it is shown by
pixels only.
