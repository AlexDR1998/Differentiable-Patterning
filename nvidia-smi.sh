#!/bin/bash

# Namespace
namespace='eidf151ns'
# Get run job name
default_prefix='ar-dp-job'
job_prefix="${1:-$default_prefix}"

# Get the pod name of the pod attached to the run job
#pod_name=$(kubectl -n $namespace get pod -l job-name=$job_name -o jsonpath="{.items[0].metadata.name}")
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)

# Execute nvidia-smi command
kubectl -n $namespace exec -it $pod_name -- nvidia-smi
