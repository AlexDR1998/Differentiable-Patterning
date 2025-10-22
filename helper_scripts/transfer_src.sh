#!/bin/bash

# Source folder path
SOURCE_FOLDER="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/"
# SOURCE_FOLDER=$1
# Transfer yml file
transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/transfer.yml'
# transfer_file=$1'/transfer.yml'
echo "transfering files from" $SOURCE_FOLDER

# PVC mount path
PVC_MOUNT_PATH="/mnt/ceph/ar-dp/"

# Transfer job name
job_prefix="ar-dp-transfer-job"

# Namespace
namespace='eidf151ns'

kubectl -n $namespace create -f $transfer_file

# Get the full pod and job names
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)

# Wait for the transfer job to start
kubectl -n $namespace wait --for=condition=Ready pod/$pod_name --timeout=60s

kubectl -n $namespace cp "${SOURCE_FOLDER}NCA/" "$pod_name":"$PVC_MOUNT_PATH"
kubectl -n $namespace cp "${SOURCE_FOLDER}Common/" "$pod_name":"$PVC_MOUNT_PATH"
kubectl -n $namespace cp "${SOURCE_FOLDER}Experiments/" "$pod_name":"$PVC_MOUNT_PATH"


kubectl -n $namespace delete job $job_name