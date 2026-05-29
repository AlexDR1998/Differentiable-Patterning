#!/usr/bin/env bash
set -euo pipefail

: "${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}"

sleep 120
mkdir -p "$JOB_COMPLETION_INDEX"
printf 'Hello world\n' > "$JOB_COMPLETION_INDEX/hello.txt"
