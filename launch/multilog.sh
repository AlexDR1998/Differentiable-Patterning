#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   kjoblogs JOB_NAME [N]
#
# Examples:
#   ./kjoblogs my-train-job
#   ./kjoblogs my-train-job 200
#
# Notes:
# - Does NOT pass -n/--namespace anywhere; it relies on your current kubectl context namespace.
# - Assumes pods are named like: JOB_NAME_<random>

JOB_NAME="${1:-}"
TAIL_LINES="${2:-100}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "ERROR: Missing JOB_NAME"
  echo "Usage: $0 JOB_NAME [N]"
  exit 1
fi

if ! [[ "${TAIL_LINES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: N must be an integer (got: ${TAIL_LINES})"
  exit 1
fi

mapfile -t PODS < <(
  kubectl get pods \
    --no-headers \
    -o custom-columns=':metadata.name' \
  | grep -E "^${JOB_NAME}" || true
)

if [[ "${#PODS[@]}" -eq 0 ]]; then
  echo "No pods found matching: ${JOB_NAME}* (in current kubectl namespace)"
  exit 0
fi

echo "Current kubectl context: $(kubectl config current-context 2>/dev/null || echo "<unknown>")"
echo "Tailing last ${TAIL_LINES} line(s) per pod (all containers)"
echo

for pod in "${PODS[@]}"; do
  echo "================================================================================"
  echo "POD: ${pod}"
  echo "--------------------------------------------------------------------------------"
  kubectl get pod "${pod}" -o wide 2>/dev/null || true
  echo "--------------------------------------------------------------------------------"
  echo "LOGS (last ${TAIL_LINES} lines):"
  echo

  if ! kubectl logs "${pod}" --all-containers --tail="${TAIL_LINES}"; then
    echo
    echo "[info] Current logs failed (pod may have restarted). Trying --previous..."
    kubectl logs "${pod}" --all-containers --previous --tail="${TAIL_LINES}" || true
  fi

  echo
done
