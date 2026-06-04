#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   kjoblogs JOB_NAME [N] [NAMESPACE]
#
# Examples:
#   ./kjoblogs my-train-job
#   ./kjoblogs my-train-job 200
#   ./kjoblogs my-train-job 200 ml
#
# Notes:
# - Assumes pods are named like: JOB_NAME_<random>
# - Prints the last N lines for each container in each matching pod.
# - If you want "currently running follow", you could add: kubectl logs -f ...

JOB_NAME="${1:-}"
TAIL_LINES="${2:-100}"
NAMESPACE="${3:-default}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "ERROR: Missing JOB_NAME"
  echo "Usage: $0 JOB_NAME [N] [NAMESPACE]"
  exit 1
fi

if ! [[ "${TAIL_LINES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: N must be an integer (got: ${TAIL_LINES})"
  exit 1
fi

# Find pods whose names start with JOB_NAME_
# (This matches your naming scheme; if you later add labels, prefer selecting by labels.)
mapfile -t PODS < <(
  kubectl get pods -n "${NAMESPACE}" \
    --no-headers \
    -o custom-columns=':metadata.name' \
  | grep -E "^${JOB_NAME}_" || true
)

if [[ "${#PODS[@]}" -eq 0 ]]; then
  echo "No pods found in namespace '${NAMESPACE}' matching: ${JOB_NAME}_*"
  exit 0
fi

echo "Namespace: ${NAMESPACE}"
echo "Job prefix: ${JOB_NAME}_"
echo "Pods found: ${#PODS[@]}"
echo "Tailing last ${TAIL_LINES} line(s) per pod (all containers)"
echo

for pod in "${PODS[@]}"; do
  echo "================================================================================"
  echo "POD: ${pod}"
  echo "--------------------------------------------------------------------------------"
  # Helpful quick context (phase/restarts/node); ignore errors if the pod disappeared.
  kubectl get pod -n "${NAMESPACE}" "${pod}" -o wide 2>/dev/null || true
  echo "--------------------------------------------------------------------------------"
  echo "LOGS (last ${TAIL_LINES} lines):"
  echo

  # --all-containers handles multi-container pods; if single container, it's fine too.
  # Add --previous if you want logs from crashed containers; here we try current first,
  # then fall back to previous if current fails (common for CrashLoopBackOff).
  if ! kubectl logs -n "${NAMESPACE}" "${pod}" --all-containers --tail="${TAIL_LINES}"; then
    echo
    echo "[info] Current logs failed (pod may have restarted). Trying --previous..."
    kubectl logs -n "${NAMESPACE}" "${pod}" --all-containers --previous --tail="${TAIL_LINES}" || true
  fi

  echo
done
