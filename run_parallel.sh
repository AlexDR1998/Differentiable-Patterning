#!/bin/bash

# Runs a python script on N parallel GPUs using a Kubernetes job. Passes the GPU number to the python script.
SCRIPT=$1
NUM_GPUS=$2
# Get run job pref

job_prefix="ar-dp-job-${SCRIPT%%.*}"
job_prefix="${job_prefix//_/-}"
echo "Job prefix: $job_prefix"
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <relative-path-to-python-script> <num-gpus>"
  exit 1
fi

# Transfer bash file
transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/transfer.sh'

# Update files
sh $transfer_file /home/eidf151/eidf151/arichardson/Differentiable-Patterning/
echo "File transfer complete"
# Namespace
namespace='eidf151ns'


# Run yml file
run_template="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/run.tpl.yml"

# Create run job
export JOB_NAME_PREFIX="$job_prefix"
export JOB_PATH_TO_PYTHON_SCRIPT="mnt/ceph/ar-dp/${SCRIPT}"
export JOB_NUM_GPUS=$NUM_GPUS
envsubst < $run_template > /home/eidf151/eidf151/arichardson/Differentiable-Patterning/run.yml

kubectl -n $namespace create -f /home/eidf151/eidf151/arichardson/Differentiable-Patterning/run.yml


# Get the full pod and job names
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)

echo "Running ${SCRIPT} on ${NUM_GPUS} GPUs with job name ${job_name}"

# Wait for the run job to start
kubectl -n $namespace wait --for=condition=Ready pod/$pod_name --timeout=180s

# Get the logs of the run job
kubectl -n $namespace attach pod $pod_name

# Delete the run job
#kubectl -n $namespace delete job $job_name

sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/copy_logs_and_models.sh