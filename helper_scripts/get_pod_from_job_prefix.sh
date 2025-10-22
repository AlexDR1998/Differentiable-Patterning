#!/bin/bash
# This script retrieves a single pod name associated with a Kubernetes job based on the job's generateName prefix.
# It returns the most recently created pod whose name starts with the given prefix (or exits non-zero).
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <generateName-prefix>" >&2
  exit 1
fi
PREFIX="$1"
namespace='eidf151ns'

# List pods sorted by creation time and find the newest matching prefix. Print single line only.
pod_name=$(kubectl -n "$namespace" get pod --sort-by=.metadata.creationTimestamp -o custom-columns=":metadata.name" 2>/dev/null \
  | grep -F "${PREFIX}" | grep "^${PREFIX}" | tail -n 1 || true)

if [ -z "$pod_name" ]; then
  echo "" >&2
  exit 1
fi

printf '%s' "$pod_name"