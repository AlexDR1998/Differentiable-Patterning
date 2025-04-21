#!/bin/bash
# This script retrieves the pod name associated with a Kubernetes job based on the job's generateName prefix.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <generateName-prefix>"
  exit 1
fi
PREFIX="$1"
namespace='eidf151ns'
pod_name=$(kubectl -n $namespace get pod --no-headers -o custom-columns=":metadata.name" | grep "^${PREFIX}")
echo "$pod_name"