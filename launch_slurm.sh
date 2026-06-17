#!/usr/bin/env bash
set -eo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

# Intel's conda activation hooks can read unset internal variables, so do not
# enable nounset until after the module/conda environment is ready.
echo "launch_slurm.sh version: ${SLURM_LAUNCH_VERSION:-unknown}"
echo "launch_slurm.sh submitted sha256: ${SLURM_LAUNCH_SHA256:-unknown}"
if command -v sha256sum >/dev/null 2>&1; then
    echo "launch_slurm.sh runtime sha256: $(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
fi

module purge
module load rhel9/default-dawn
module load intelpython-conda
module load intel-oneapi-mkl
echo "Skipping intel-oneapi-ccl; these jobs run one JAX process per array task."
conda activate jax_intel_gpu
python -m pip list | grep -E "jax|jaxlib|intel-extension-for-openxla"
python -m pip show jax jaxlib intel-extension-for-openxla intel_extension_for_openxla 2>/dev/null || true

if [[ "${SLURM_CHECK_OPENXLA_VERSION_COMPAT:-1}" == "1" ]]; then
    python - <<'PY'
from importlib import metadata
import sys


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


jax = package_version("jax")
jaxlib = package_version("jaxlib")
openxla = (
    package_version("intel-extension-for-openxla")
    or package_version("intel_extension_for_openxla")
)

compatibility = {
    "0.7.0": ("0.5.0", "0.5.0"),
    "0.6.0": ("0.4.38", "0.4.38"),
    "0.5.0": ("0.4.30", "0.4.30"),
    "0.4.0": ("0.4.26", "0.4.26"),
    "0.3.0": ("0.4.24", "0.4.24"),
}

if not openxla:
    print("Intel Extension for OpenXLA is not installed; skipping compatibility check.")
elif openxla in compatibility:
    expected_jaxlib, expected_jax = compatibility[openxla]
    if jaxlib != expected_jaxlib or jax != expected_jax:
        print("Incompatible Intel OpenXLA/JAX package versions detected.", file=sys.stderr)
        print(f"  intel-extension-for-openxla: {openxla}", file=sys.stderr)
        print(f"  installed jaxlib: {jaxlib}", file=sys.stderr)
        print(f"  installed jax: {jax}", file=sys.stderr)
        print(f"  expected jaxlib for OpenXLA {openxla}: {expected_jaxlib}", file=sys.stderr)
        print(f"  expected jax for OpenXLA {openxla}: {expected_jax}", file=sys.stderr)
        print("Refusing to continue because JAX backend discovery may segfault.", file=sys.stderr)
        print("Set SLURM_CHECK_OPENXLA_VERSION_COMPAT=0 to bypass this guard.", file=sys.stderr)
        sys.exit(2)
else:
    print(
        f"Warning: no local compatibility rule for intel-extension-for-openxla {openxla}; "
        "continuing without a version guard.",
        file=sys.stderr,
    )
PY
fi

set -u

: "${PY_SCRIPT:?PY_SCRIPT is not set}"
: "${MANIFEST:?MANIFEST is not set}"
: "${N_JOBS:?N_JOBS is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

if [[ "${SLURM_ENABLE_CORE_DUMPS:-0}" == "1" ]]; then
    ulimit -c unlimited
else
    ulimit -c 0
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_JOBS )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is outside manifest range 0-$((N_JOBS - 1))"
    exit 1
fi

# -------------------------
# Kubernetes-style env vars
# -------------------------

export JOB_WORKER_INDEX=0
export JOB_WORKER_COUNT=1
export JOB_COMPLETION_INDEX="$SLURM_ARRAY_TASK_ID"

IO_ROOT="${SLURM_IO_ROOT:-/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs}"
IO_ROOT="${IO_ROOT%/}"
CODE_ROOT="${SLURM_CODE_ROOT:-$(cd "$(dirname "$PY_SCRIPT")/.." && pwd)}"
CODE_ROOT="${CODE_ROOT%/}"

PVC_PATH="${PVC_PATH:-$CODE_ROOT/}"
[[ "$PVC_PATH" == */ ]] || PVC_PATH="$PVC_PATH/"
export PVC_PATH
export DATA_PATH_BASE="${DATA_PATH_BASE:-$IO_ROOT/Data/}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$IO_ROOT/Models/}"

export INTEL_MAX_GPU_VRAM_GB="${INTEL_MAX_GPU_VRAM_GB:-128}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.98}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-sycl}"

WANDB_TASK_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
WANDB_SCRATCH_ROOT="${WANDB_SCRATCH_ROOT:-$IO_ROOT/wandb-fast}"

export WANDB_DIR="${WANDB_DIR:-$WANDB_SCRATCH_ROOT/$WANDB_TASK_ID}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_DIR/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_DIR/artifacts}"
export WANDB_FLUSH_INTERVAL="${WANDB_FLUSH_INTERVAL:-60}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

mkdir -p "$MODEL_SAVE_PATH" "$IO_ROOT/output" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

echo "Running manifest index $SLURM_ARRAY_TASK_ID/$((N_JOBS - 1)): $MANIFEST"
echo "Using code root: $PVC_PATH"
echo "Using job IO root: $IO_ROOT/"
echo "Writing wandb local files to: $WANDB_DIR"
echo "Expected Intel GPU VRAM: ${INTEL_MAX_GPU_VRAM_GB} GB"
echo "XLA memory claim fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
python - <<'PY'
import os

vram_gb = float(os.environ["INTEL_MAX_GPU_VRAM_GB"])
fraction = float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])
print(f"Expected XLA preallocation target: {vram_gb * fraction:.1f} GB")
PY

if [[ "${SLURM_GPU_DIAGNOSTICS:-1}" == "1" ]]; then
    echo "GPU diagnostics:"
    echo "  Host: $(hostname)"
    echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
    echo "  SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
    echo "  ZE_AFFINITY_MASK: ${ZE_AFFINITY_MASK:-unset}"
    echo "  ZE_FLAT_DEVICE_HIERARCHY: ${ZE_FLAT_DEVICE_HIERARCHY:-unset}"
    echo "  ONEAPI_DEVICE_SELECTOR: ${ONEAPI_DEVICE_SELECTOR:-unset}"
    echo "  SYCL_DEVICE_FILTER: ${SYCL_DEVICE_FILTER:-unset}"
    echo "  JAX_PLATFORMS: ${JAX_PLATFORMS:-unset}"
    echo "  JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME:-unset}"
    echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: ${XLA_PYTHON_CLIENT_MEM_FRACTION:-unset}"
    command -v sycl-ls >/dev/null 2>&1 && sycl-ls || echo "  sycl-ls: not found"
    command -v ze_info >/dev/null 2>&1 && ze_info | sed -n '1,80p' || echo "  ze_info: not found"
fi

case "${SLURM_JAX_SMOKE_TEST:-0}" in
    0)
        ;;
    1|devices)
        python -X faulthandler -c 'import jax; print("JAX devices:"); print(jax.devices())'
        exit 0
        ;;
    import)
        python -X faulthandler -c 'import jax; print("Imported JAX", jax.__version__)'
        exit 0
        ;;
    cpu)
        JAX_PLATFORMS=cpu python -X faulthandler -c 'import jax; print("JAX CPU devices:"); print(jax.devices())'
        exit 0
        ;;
    *)
        echo "Unknown SLURM_JAX_SMOKE_TEST mode: $SLURM_JAX_SMOKE_TEST"
        echo "Use one of: 0, 1, devices, import, cpu"
        exit 1
        ;;
esac

if [[ "${SLURM_USE_SRUN:-0}" == "1" ]]; then
    srun python -X faulthandler "$PY_SCRIPT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID"
else
    python -X faulthandler "$PY_SCRIPT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID"
fi
