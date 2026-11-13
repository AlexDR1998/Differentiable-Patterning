# Running on the EIDF GPU cluster

Training runs on the EIDF (Edinburgh International Data Facility) GPU
service, a Kubernetes cluster. Each run of a manifest becomes one **pod**:
a container with one GPU that runs `Experiments/run_config.py` for one
manifest entry and then stops.

> Lines marked **TODO(handover)** are things only the previous owner knows.
> Please fill them in.

## What you need

- **An EIDF account** in the project, with access to the GPU service and
  `kubectl` set up on the project VM. Ask the project PI to add you, then
  follow EIDF's GPU service documentation for the kubeconfig.
  **TODO(handover):** project name and code, and who to ask.
- **SSH access to the project VM** through the EIDF gateway. Locally, an
  SSH alias called `eidf-1` is expected by
  `helper_scripts/copy_models_from_cluster.sh`:
  ```text
  # ~/.ssh/config     TODO(handover): check host names and user
  Host eidf-gateway
      HostName eidf-gateway.epcc.ed.ac.uk
      User <your EIDF user>
  Host eidf-1
      HostName <project VM address>
      User <your EIDF user>
      ProxyJump eidf-gateway
  ```
- **Environment variables on the VM** (e.g. in `~/.bashrc`), used by
  `launch/run.tpl.yml`:
  - `KUBE_USER_QUEUE`: your Kueue queue name;
  - `EIDF107_USER_DATA_NFS_HOST_IP`: the NFS server that holds your code.
  - `USER` is also used: job names, labels and the names below all contain
    it.
- **Two Kubernetes objects named after your user:**
  - `$USER-ws1`: a persistent volume (PVC) for data, models and outputs;
  - `$USER-wandb-login`: a secret with your W&B key:
    `kubectl create secret generic $USER-wandb-login --from-literal=WANDB_API_KEY=<key>`.
  **TODO(handover):** how the PVC was created (size, storage class), and
  whether new users get their own or share the existing one.

## Where things are

Every pod mounts:

| In the pod | What it is |
| --- | --- |
| `/workspace` (read-only) | Your checkout of this repository on the NFS share, `/user/$USER/Differentiable-Patterning/` (`WORKING_DIR`). |
| `/workspace/writeable/` | The `$USER-ws1` volume. |
| `/workspace/writeable/Data/` | `DATA_PATH_BASE`: the datasets ([data.md](data.md)). |
| `/workspace/writeable/Models/` | `MODEL_STORE_ROOT`: model bundles and impulse records. |
| `/workspace/writeable/output/` | Results of notebook jobs ([remote_notebook_jobs.md](remote_notebook_jobs.md)). |
| `/workspace/writeable/data/logs/wandb/` | Copies of the W&B run folders. |

**Important:** the code is mounted read-only, and pods use whatever is in
your NFS checkout *when the pod starts*. So push your changes there,
**including the generated manifest**, before launching.
**TODO(handover):** how the NFS checkout is updated (git pull on the VM,
rsync from a laptop, ...).

The container image is `alexdr1998/jax_equinox_scratch:latest` (Docker Hub).
It provides Python, JAX with CUDA and the other packages. The code itself
comes from `/workspace`, not from the image.
**TODO(handover):** where its Dockerfile is (Dockerfiles are ignored by
git), and how to rebuild and push it. Consider moving it to an account the
group controls.

## Launching a sweep

From the root of your checkout on the VM:

```bash
bash launch/launch_batch_multi_job.sh Experiments/run_config.py \
    Experiments/micropatterns/conf/generated/<experiment>/manifest.yaml 4 H200
```

The arguments are:
1. the Python script;
2. the manifest;
3. the number of **workers**, i.e. Kubernetes jobs;
4. the GPU type: `H200`, `H100` or `A100`;
5. optionally, the first manifest index (to restart part of a sweep).

The manifest entries are split evenly across the workers. Each worker runs
its pods one at a time (`parallelism: 1`), so `workers` is how many runs go
at once. The script fills in `launch/run.tpl.yml` once per worker, writes
`run_<i>.yml` and submits it with `kubectl create`.

The same command runs notebook jobs; only the script changes, for example
`Experiments/micropatterns/density_ko_sweep.py`.

## Watching jobs

```bash
kubectl get jobs                       # your jobs (names start with $USER-job)
kubectl get pods                       # one pod per manifest entry
kubectl logs -f <pod name>             # follow one run's output
kubectl describe pod <pod name>        # why a pod is pending or failed
kubectl delete job <job name>          # stop a worker and its pods
bash launch/multilog.sh <job name> 200 # last 200 log lines of every pod of a job
```

Training progress, losses and plots are easier to follow on W&B: the project
is `logging.wandb.project` in the config, and the group is the experiment
name.

A pod that fails is not restarted (`backoffLimitPerIndex: 0`). To re-run
part of a sweep, launch again with a start index, or generate a manifest that
holds only the failed entries.

## Getting results back

Bundles are written to `/workspace/writeable/Models/bundles/`. To copy them,
the impulse records and the notebook-job outputs to your machine, and rebuild
the local registry index:

```bash
helper_scripts/copy_models_from_cluster.sh --port <port>
```

This connects with SSH (through `eidf-1`) to a long-running pod that mounts
the `$USER-ws1` volume at `/data`, so `/data/Models/` is the same folder as
`/workspace/writeable/Models/`. `--port` is that pod's SSH port.
**TODO(handover):** how that SSH pod is started, and how to find its port.

Options:
- `--no-impulses` skips the (slow) impulse records;
- `--no-results` skips `output/`;
- `--no-reindex` skips rebuilding `models/registry.sqlite`.

## Running notebook sections on the cluster

Heavy loops in evaluation notebooks (for example the density × knockout
sweep) can run as cluster jobs and be loaded back into the notebook. See
[remote_notebook_jobs.md](remote_notebook_jobs.md).

## Other launchers

- `launch/launch_local_sweep.py <manifest>` runs the entries one after
  another on a local GPU, with no scheduler.
- `launch/launch_batch_slurm.sh` and `launch/launch_slurm.sh` ran sweeps on a
  Slurm cluster (Cambridge's Dawn) that is no longer used. Their account,
  partition and storage path are the previous owner's.
