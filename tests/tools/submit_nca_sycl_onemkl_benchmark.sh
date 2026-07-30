#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NCA_SYCL_PROBES="baseline serialize_onemkl"
export NCA_SYCL_PROBE_REPEATS="${NCA_SYCL_PROBE_REPEATS:-100}"
exec "${SCRIPT_DIR}/submit_nca_sycl_failure_probes.sh"
