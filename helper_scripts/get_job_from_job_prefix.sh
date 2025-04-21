#!/bin/bash
# This script retrieves the job name associated with a Kubernetes job based on the job's generateName prefix.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <generateName-prefix>"
  exit 1
fi
PREFIX="$1"
namespace='eidf151ns'
job_name=$(kubectl -n $namespace get job --no-headers -o custom-columns=":metadata.name" | grep "^${PREFIX}")
echo "$job_name"