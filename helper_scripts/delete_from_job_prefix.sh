#!/bin/bash
PREFIX="$1"
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $PREFIX)
namespace='eidf151ns'
# Delete the job
echo "Deleting job: $job_name"
kubectl -n $namespace delete job $job_name