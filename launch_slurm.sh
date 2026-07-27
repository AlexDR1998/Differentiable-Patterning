#!/usr/bin/env bash
set -eo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

# Intel's conda hooks can read unset internal variables, so enable nounset only
# after the module/conda environment is ready.
# Capture the pre-activation Python environment when diagnosing intermittent
# Conda imports across Slurm array tasks. Keep this probe non-fatal: its output
# is evidence for comparing successful and failed nodes, not a job prerequisite.
echo "PRE_SETUP_HOSTNAME=${HOSTNAME:-$(hostname)}"
echo "PRE_SETUP_PYTHONHOME=${PYTHONHOME-<unset>}"
echo "PRE_SETUP_PYTHONPATH=${PYTHONPATH-<unset>}"

PRE_SETUP_CONDA_ROOT="${CONDA_DIAGNOSTIC_ROOT:-/rds/project/rds-NQDJLHPwRqs/my_conda}"
PRE_SETUP_PYTHON="${PRE_SETUP_CONDA_ROOT}/bin/python"
PRE_SETUP_LOGGING_INIT="${PRE_SETUP_CONDA_ROOT}/lib/python3.13/logging/__init__.py"
echo "PRE_SETUP_CONDA_ROOT=${PRE_SETUP_CONDA_ROOT}"
ls -l "${PRE_SETUP_LOGGING_INIT}" 2>&1 \
    || echo "PRE_SETUP_LOGGING_INIT_RESULT=UNAVAILABLE"
if [[ -x "${PRE_SETUP_PYTHON}" ]]; then
    if "${PRE_SETUP_PYTHON}" -S -c \
        'import importlib.util, sys; print("PRE_SETUP_SYS_PATH=" + repr(sys.path)); print("PRE_SETUP_LOGGING_SPEC=" + repr(importlib.util.find_spec("logging")))'
    then
        echo "PRE_SETUP_PYTHON_RESULT=PASS"
    else
        echo "PRE_SETUP_PYTHON_RESULT=FAIL"
    fi
else
    echo "PRE_SETUP_PYTHON_RESULT=UNAVAILABLE"
fi

# Slurm submission uses --export=ALL. Do not let an inherited project or module
# PYTHONPATH shadow Python's standard library while the setup script invokes
# Conda (for example, a namespace package named ``logging``).
unset PYTHONHOME
unset PYTHONPATH
# module purge
# module load rhel9/default-dawn
# module load intelpython-conda
# module load intel-oneapi-mkl
# conda activate jax_intel_gpu

# bash ~/dawn-jax/envs/jax-setup.sh
source ~/dawn-jax/envs/jaxeqx-setup.sh

: "${PROFILE_GPU:=0}"
: "${NCA_SYCL_DIAGNOSTICS:=0}"
: "${NCA_SYCL_TRACE:=0}"
if [[ "$PROFILE_GPU" != "0" && "$PROFILE_GPU" != "1" ]]; then
    echo "PROFILE_GPU must be 0 or 1, got: $PROFILE_GPU"
    exit 1
fi
if [[ "$NCA_SYCL_DIAGNOSTICS" != "0" && "$NCA_SYCL_DIAGNOSTICS" != "1" ]]; then
    echo "NCA_SYCL_DIAGNOSTICS must be 0 or 1, got: $NCA_SYCL_DIAGNOSTICS"
    exit 1
fi
if [[ "$NCA_SYCL_TRACE" != "0" && "$NCA_SYCL_TRACE" != "1" ]]; then
    echo "NCA_SYCL_TRACE must be 0 or 1, got: $NCA_SYCL_TRACE"
    exit 1
fi

# Intel OpenXLA must see these before the first process imports JAX. They are
# enabled only for profiling jobs because tracing adds runtime overhead.
if [[ "$PROFILE_GPU" == "1" ]]; then
    export ZE_ENABLE_TRACING_LAYER=1
    export UseCyclesPerSecondTimer=1
fi
if [[ "$NCA_SYCL_TRACE" == "1" ]]; then
    export OCL_ICD_ENABLE_TRACE=1
    export ZE_LOADER_DEBUG_TRACE=1
    export NEOReadDebugKeys=1
    export PrintDebugMessages=1
fi

SYCL_BUILD_DIR="${SLURM_TMPDIR:-/tmp}/nca-sycl-${SLURM_JOB_ID}"
mkdir -p "${SYCL_BUILD_DIR}"

NCA/model/sycl/files/build_nca_sycl.sh \
    "${SYCL_BUILD_DIR}/libnca_sycl.so"

export NCA_SYCL_LIBRARY="${SYCL_BUILD_DIR}/libnca_sycl.so"

echo "SYCL_JOB_HOSTNAME=$(hostname)"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
if [[ "$NCA_SYCL_DIAGNOSTICS" == "1" ]]; then
    echo "NCA_SYCL_DIAGNOSTICS_BEGIN"
    uname -a || true
    python -m pip show jax jaxlib intel-extension-for-openxla || true
    icpx --version 2>&1 | sed -n '1,3p' || true
    sycl-ls --verbose 2>&1 || true
    ldd "$NCA_SYCL_LIBRARY" 2>&1 || true
    if command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W 2>/dev/null | grep -Ei 'level-zero|igc|intel.*(mkl|compute|opencl)|oneapi' || true
    fi
    if command -v rpm >/dev/null 2>&1; then
        rpm -qa 2>/dev/null | grep -Ei 'level-zero|igc|intel.*(mkl|compute|opencl)|oneapi' || true
    fi
    for module in i915 xe; do
        modinfo "$module" 2>/dev/null | sed -n '1,12p' || true
    done
    python - <<'PY'
from importlib import metadata
import importlib.util
import pathlib
import subprocess

for name in ("jax", "jaxlib", "intel-extension-for-openxla"):
    try:
        print(f"PACKAGE_VERSION {name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"PACKAGE_VERSION {name}=not_found")

spec = importlib.util.find_spec("jax_plugins.intel_extension_for_openxla")
if spec is not None and spec.origin:
    root = pathlib.Path(spec.origin).parent
    for library in sorted(root.rglob("*.so")):
        print(f"PLUGIN_LDD_BEGIN={library}")
        subprocess.run(["ldd", str(library)], check=False)
        print(f"PLUGIN_LDD_END={library}")
PY
    echo "NCA_SYCL_DIAGNOSTICS_END"
fi

# module purge
# module load rhel9/default-dawn
# source /usr/local/dawn/software/external/intel-oneapi/2025.2.1/setvars.sh
# if [[ -z "${ZE_FLAT_DEVICE_HIERARCHY}" ]]; then
#     export ZE_FLAT_DEVICE_HIERARCHY="FLAT"
# fi 
# source /home/rc-rich1/miniforge3/bin/activate
# conda activate jax


python - <<'PY'
from importlib import metadata
import sys
import jax
print("jax.devices(): ", jax.devices())
print("jax.local_devices(): ", jax.local_devices())
PY

set -u

: "${PY_SCRIPT:?PY_SCRIPT is not set}"
: "${MANIFEST:?MANIFEST is not set}"
: "${N_JOBS:?N_JOBS is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

ulimit -c 0

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_JOBS )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is outside manifest range 0-$((N_JOBS - 1))"
    exit 1
fi

export JOB_WORKER_INDEX=0
export JOB_WORKER_COUNT=1
export JOB_COMPLETION_INDEX="$SLURM_ARRAY_TASK_ID"

IO_ROOT="${SLURM_IO_ROOT:-/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs}"
IO_ROOT="${IO_ROOT%/}"
CODE_ROOT="${SLURM_CODE_ROOT:-$(cd "$(dirname "$PY_SCRIPT")/.." && pwd)}"
CODE_ROOT="${CODE_ROOT%/}"
ARRAY_LOG_ROOT="${SLURM_LOG_DIR:-$IO_ROOT/slurm_logs}"
ARRAY_LOG_ROOT="${ARRAY_LOG_ROOT%/}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_LOG_DIR="$ARRAY_LOG_ROOT/$ARRAY_JOB_ID"

export SLURM_ARRAY_LOG_DIR="$ARRAY_LOG_DIR"
export RUN_CONFIG_PROFILE_DIR="${RUN_CONFIG_PROFILE_DIR:-$ARRAY_LOG_DIR/${SLURM_ARRAY_TASK_ID}.profile}"
export PROFILE_GPU_DIR="${PROFILE_GPU_DIR:-$RUN_CONFIG_PROFILE_DIR}"

PVC_PATH="${PVC_PATH:-$CODE_ROOT/}"
[[ "$PVC_PATH" == */ ]] || PVC_PATH="$PVC_PATH/"
export PVC_PATH
export DATA_PATH_BASE="${DATA_PATH_BASE:-$IO_ROOT/Data/}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$IO_ROOT/Models/}"

export RUN_CONFIG_PROFILE="$PROFILE_GPU"
# The trainer captures a short warmed-up window. Do not wrap imports,
# compilation, and the entire experiment in a second profiler session.
export RUN_CONFIG_PROFILE_TRACE=0
export RUN_CONFIG_PROFILE_MEMORY=0

WANDB_TASK_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
WANDB_SCRATCH_ROOT="${WANDB_SCRATCH_ROOT:-$IO_ROOT/wandb-fast}"

export WANDB_DIR="${WANDB_DIR:-$WANDB_SCRATCH_ROOT/$WANDB_TASK_ID}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_DIR/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_DIR/artifacts}"
export WANDB_FLUSH_INTERVAL="${WANDB_FLUSH_INTERVAL:-60}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NCA_SYCL_REPORT_QUEUE_ORDERING="${NCA_SYCL_REPORT_QUEUE_ORDERING:-1}"

mkdir -p "$MODEL_SAVE_PATH" "$IO_ROOT/output" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$RUN_CONFIG_PROFILE_DIR"

echo "Running manifest index $SLURM_ARRAY_TASK_ID/$((N_JOBS - 1)): $MANIFEST"
echo "Using code root: $PVC_PATH"
echo "Using job IO root: $IO_ROOT/"
echo "Writing wandb local files to: $WANDB_DIR"
echo "GPU profiling: $PROFILE_GPU"
echo "JAX profiles: $RUN_CONFIG_PROFILE_DIR"

echo "GPU view:"
echo "  Host: $(hostname)"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "  SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
command -v sycl-ls >/dev/null 2>&1 && sycl-ls || echo "  sycl-ls: not found"

python -X faulthandler "$PY_SCRIPT" \
    --manifest "$MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID"
