#!/bin/bash

# Transfer bash file
transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/transfer.sh'

# Update files
sh $transfer_file /home/eidf151/eidf151/arichardson/Differentiable-Patterning/
echo "Transfer complete"
echo "Launching run job"
# Namespace
namespace='eidf151ns'

# Run yml file
run_file="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/run_0.yml"

# Create run job
kubectl -n $namespace create -f $run_file

# Get run job pref
job_prefix='ar-dp-job'

# Get the full pod and job names
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)

# Wait for the run job to start
kubectl -n $namespace wait --for=condition=Ready pod/$pod_name --timeout=180s

# Get the logs of the run job
kubectl -n $namespace attach pod $pod_name

# Delete the run job
#kubectl -n $namespace delete job $job_name

sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/copy_logs_and_models.sh