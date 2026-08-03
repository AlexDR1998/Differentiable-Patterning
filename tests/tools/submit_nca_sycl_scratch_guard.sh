#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPEATS="${NCA_SYCL_SCRATCH_REPEATS:-20}"
PARALLELISM="${NCA_SYCL_SCRATCH_PARALLELISM:-4}"
LOG_ROOT="${NCA_SYCL_SCRATCH_LOG_DIR:-${PWD}/nca-sycl-scratch-logs}"
SELECTED_MODES="${NCA_SYCL_SCRATCH_MODES:-reuse per_step}"
read -r -a MODES <<< "${SELECTED_MODES}"
MODE_COUNT="${#MODES[@]}"
[[ "${MODE_COUNT}" -gt 0 ]] || { echo "No scratch modes selected" >&2; exit 2; }
TASKS=$((MODE_COUNT * REPEATS))

mkdir -p "${LOG_ROOT}"
JOB_ID="$(sbatch --parsable \
    --array="0-$((TASKS - 1))%${PARALLELISM}" \
    --output="${LOG_ROOT}/%A-%a.out" \
    --error="${LOG_ROOT}/%A-%a.err" \
    --export=ALL,NCA_SYCL_SCRATCH_REPO_ROOT="${REPO_ROOT}",NCA_SYCL_SCRATCH_REPEATS="${REPEATS}",NCA_SYCL_SCRATCH_MODES="${SELECTED_MODES}" \
    "${REPO_ROOT}/tests/tools/nca_sycl_scratch_guard.sh")"

echo "Submitted ${TASKS} tasks as job ${JOB_ID}: ${REPEATS} repeats for ${SELECTED_MODES}."
echo "Device ASan: ${NCA_SYCL_SCRATCH_ASAN:-0}; tiles per task: ${NCA_SYCL_SCRATCH_TILES:-2}."
echo "Logs: ${LOG_ROOT}/${JOB_ID}-TASK_ID.out/.err"
echo "Summarize this job with:"
echo "  python ${REPO_ROOT}/tests/tools/summarize_nca_sycl_scratch_guard.py ${LOG_ROOT} --job-id ${JOB_ID}"
