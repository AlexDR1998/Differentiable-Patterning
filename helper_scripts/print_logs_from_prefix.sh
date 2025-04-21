#!/bin/bash
PREFIX="$1"
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $PREFIX)
namespace='eidf151ns'
# Get the logs of the job
kubectl -n $namespace logs $pod_name