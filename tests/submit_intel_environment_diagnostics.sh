#!/usr/bin/env bash
#SBATCH --job-name=intel-env-diagnostics
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=intel-env-diagnostics-%j.out
#SBATCH --error=intel-env-diagnostics-%j.err

set -o pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

# Running this file from a login node submits it. Calling it with sbatch
# directly also works, because SLURM_JOB_ID will be set in the allocated job.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch is not available on PATH" >&2
        exit 10
    fi
    JOB_ID="$(sbatch --parsable "${SCRIPT_PATH}")"
    echo "Submitted Intel environment diagnostic job ${JOB_ID}"
    echo "Expected logs in the submission directory:"
    echo "  intel-env-diagnostics-${JOB_ID}.out"
    echo "  intel-env-diagnostics-${JOB_ID}.err"
    exit 0
fi

section() {
    echo
    echo "===== $1 ====="
}

run_optional() {
    local description="$1"
    shift
    echo "--- ${description}"
    "$@"
    local status=$?
    if [[ "${status}" -eq 0 ]]; then
        return 0
    fi
    echo "COMMAND_STATUS=${status}"
    return 0
}

section "SLURM AND HOST"
echo "DATE=$(date --iso-8601=seconds)"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-<unset>}"
run_optional "uname" uname -a
if [[ -r /etc/os-release ]]; then
    run_optional "operating system" sed -n '1,20p' /etc/os-release
fi
if command -v module >/dev/null 2>&1; then
    run_optional "loaded modules" module list
else
    echo "MODULE_COMMAND=not_found"
fi

section "MATCH TRAINING ENVIRONMENT"
JAX_SETUP_SCRIPT="${JAX_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jax-setup.sh}"
JAX_CONDA_ENV="${JAX_CONDA_ENV:-jax}"
echo "JAX_SETUP_SCRIPT=${JAX_SETUP_SCRIPT}"
echo "JAX_CONDA_ENV=${JAX_CONDA_ENV}"

if [[ -f "${JAX_SETUP_SCRIPT}" ]]; then
    run_optional "JAX cluster setup" bash "${JAX_SETUP_SCRIPT}"
else
    echo "JAX_SETUP_SCRIPT_STATUS=not_found"
fi

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    if conda activate "${JAX_CONDA_ENV}"; then
        echo "CONDA_ACTIVATE_STATUS=PASS"
    else
        echo "CONDA_ACTIVATE_STATUS=FAIL"
    fi
else
    echo "CONDA_COMMAND=not_found"
fi

# Intel/Conda activation scripts may read unset internal variables. Match the
# main cluster launcher by enabling nounset only after environment activation.
set -u

section "RELEVANT ENVIRONMENT VARIABLES"
for name in \
    CONDA_DEFAULT_ENV \
    CONDA_PREFIX \
    ONEAPI_ROOT \
    ONEAPI_DEVICE_SELECTOR \
    SYCL_DEVICE_FILTER \
    SYCL_PI_TRACE \
    ZE_AFFINITY_MASK \
    PJRT_NAMES_AND_LIBRARY_PATHS \
    JAX_PLATFORMS \
    XLA_FLAGS \
    LD_LIBRARY_PATH; do
    printf '%s=%s\n' "${name}" "${!name:-<unset>}"
done

section "PYTHON"
echo "PYTHON=$(command -v python || true)"
run_optional "python version" python --version
run_optional "pip version" python -m pip --version
run_optional "relevant packages" bash -c \
    "python -m pip list 2>/dev/null | grep -Ei '^(jax|jaxlib|equinox|intel|.*openxla|numpy|scipy)[[:space:]]' || true"

section "SYCL TOOLCHAIN"
for compiler in icpx dpcpp clang++; do
    if command -v "${compiler}" >/dev/null 2>&1; then
        echo "${compiler^^}_PATH=$(command -v "${compiler}")"
        run_optional "${compiler} version" "${compiler}" --version
    else
        echo "${compiler^^}_PATH=not_found"
    fi
done

if command -v sycl-ls >/dev/null 2>&1; then
    echo "SYCL_LS_PATH=$(command -v sycl-ls)"
    run_optional "sycl-ls default selection" sycl-ls
    run_optional "sycl-ls verbose" sycl-ls --verbose
else
    echo "SYCL_LS_PATH=not_found"
fi

section "JAX DEVICES AND INTEL OPENXLA PLUGIN"
python - <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import pathlib
import site
import subprocess
import sys


def safe_attribute(value, name):
    try:
        return getattr(value, name)
    except Exception as exc:
        return f"<error: {exc!r}>"


print(f"PYTHON_EXECUTABLE={sys.executable}")
print(f"PYTHON_VERSION={sys.version.replace(chr(10), ' ')}")

try:
    import jax
    import jaxlib

    print(f"JAX_VERSION={jax.__version__}")
    print(f"JAXLIB_VERSION={jaxlib.__version__}")
    print(f"JAX_DEFAULT_BACKEND={jax.default_backend()}")
    devices = jax.devices()
    local_devices = jax.local_devices()
    print(f"JAX_DEVICE_COUNT={len(devices)}")
    print(f"JAX_LOCAL_DEVICE_COUNT={len(local_devices)}")
    for index, device in enumerate(devices):
        print(f"JAX_DEVICE_{index}={device!r}")
        for attribute in (
            "platform",
            "device_kind",
            "id",
            "process_index",
            "slice_index",
        ):
            print(
                f"JAX_DEVICE_{index}_{attribute.upper()}="
                f"{safe_attribute(device, attribute)}"
            )
except Exception as exc:
    print(f"JAX_IMPORT_OR_DEVICE_ERROR={exc!r}")

print("RELEVANT_DISTRIBUTIONS_BEGIN")
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name", "")
    lowered = name.lower()
    if any(token in lowered for token in ("jax", "openxla", "intel")):
        print(f"DISTRIBUTION={name} VERSION={distribution.version}")
print("RELEVANT_DISTRIBUTIONS_END")

module_names = (
    "jax_plugins.intel_extension_for_openxla",
    "intel_extension_for_openxla",
    "xpu_plugin_extension",
)
for module_name in module_names:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        print(f"MODULE_SPEC_ERROR {module_name}={exc!r}")
        continue
    if spec is None:
        print(f"MODULE_NOT_FOUND={module_name}")
        continue
    print(f"MODULE_SPEC {module_name} ORIGIN={spec.origin}")
    try:
        module = importlib.import_module(module_name)
        interesting = sorted(
            name
            for name in dir(module)
            if any(token in name.lower() for token in ("custom", "ffi", "register"))
        )
        print(f"MODULE_API {module_name}={interesting}")
    except Exception as exc:
        print(f"MODULE_IMPORT_ERROR {module_name}={exc!r}")

roots = []
for value in [*site.getsitepackages(), site.getusersitepackages()]:
    path = pathlib.Path(value)
    if path.is_dir() and path not in roots:
        roots.append(path)

patterns = (
    "*pjrt*plugin*xpu*.so",
    "*xpu*plugin*.so",
    "*openxla*.so",
    "*sycl*onednn*.so",
)
shared_libraries = []
for root in roots:
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file() and path not in shared_libraries:
                shared_libraries.append(path)

print("PLUGIN_SHARED_LIBRARIES_BEGIN")
for path in shared_libraries:
    print(path)
print("PLUGIN_SHARED_LIBRARIES_END")

for path in shared_libraries:
    print(f"LDD_BEGIN={path}")
    try:
        completed = subprocess.run(
            ["ldd", str(path)],
            check=False,
            text=True,
            capture_output=True,
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
        print(f"LDD_STATUS={completed.returncode}")
    except Exception as exc:
        print(f"LDD_ERROR={exc!r}")
    print(f"LDD_END={path}")
PY
PYTHON_STATUS=$?
echo "PYTHON_DIAGNOSTIC_STATUS=${PYTHON_STATUS}"

section "DIAGNOSTIC RESULT"
if [[ "${PYTHON_STATUS}" -eq 0 ]]; then
    echo "INTEL_ENV_DIAGNOSTIC_RESULT=PASS"
else
    echo "INTEL_ENV_DIAGNOSTIC_RESULT=PARTIAL"
fi

# Do not fail the Slurm job merely because an optional diagnostic was missing;
# the log itself is the result we need for deciding the integration route.
exit 0
