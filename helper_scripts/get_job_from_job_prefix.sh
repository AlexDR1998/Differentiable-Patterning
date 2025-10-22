#!/bin/bash
# This script retrieves a single job name associated with a Kubernetes job based on the job's generateName prefix.
# It returns the most recently created job whose name starts with the given prefix (or exits non-zero).
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <generateName-prefix>" >&2
  exit 1
fi
PREFIX="$1"
namespace='eidf151ns'

job_name=$(kubectl -n "$namespace" get job --sort-by=.metadata.creationTimestamp -o custom-columns=":metadata.name" 2>/dev/null \
  | grep -F "${PREFIX}" | grep "^${PREFIX}" | tail -n 1 || true)

if [ -z "$job_name" ]; then
  echo "" >&2
  exit 1
fi

printf '%s' "$job_name"