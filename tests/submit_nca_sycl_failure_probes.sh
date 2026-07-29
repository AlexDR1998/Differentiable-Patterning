#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPEATS="${NCA_SYCL_PROBE_REPEATS:-20}"
PARALLELISM="${NCA_SYCL_PROBE_PARALLELISM:-4}"
LOG_ROOT="${NCA_SYCL_PROBE_LOG_DIR:-${PWD}/nca-sycl-probe-logs}"
PROBE_COUNT=4
TASKS=$((PROBE_COUNT * REPEATS))

mkdir -p "${LOG_ROOT}"
JOB_ID="$(sbatch --parsable \
    --array="0-$((TASKS - 1))%${PARALLELISM}" \
    --output="${LOG_ROOT}/%A-%a.out" \
    --error="${LOG_ROOT}/%A-%a.err" \
    --export=ALL,NCA_SYCL_PROBE_REPO_ROOT="${REPO_ROOT}",NCA_SYCL_PROBE_REPEATS="${REPEATS}" \
    "${REPO_ROOT}/tests/nca_sycl_failure_probe.sh")"

echo "Submitted ${TASKS} tasks as job ${JOB_ID}: ${REPEATS} unconstrained repeats per probe."
echo "Logs: ${LOG_ROOT}/${JOB_ID}-TASK_ID.out/.err"
echo "Assigned hostnames are recorded in every task log."
