#!/bin/bash

# Source folder path
#SOURCE_FOLDER="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/"
SOURCE_FOLDER=$1
# Transfer yml file
#transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/transfer.yml'
transfer_file=$1'/transfer.yml'
echo "transfering files from" $SOURCE_FOLDER

# PVC mount path
PVC_MOUNT_PATH="/mnt/ceph/ar-dp/"

# Transfer job name
job_prefix="ar-dp-transfer-job"

# Namespace
namespace='eidf151ns'

# Create a temporary tar file
TEMP_TAR_FILE="/tmp/ar_folder.tar"
tar --exclude="${SOURCE_FOLDER}logs/*" -cf "$TEMP_TAR_FILE" -C "$SOURCE_FOLDER" .

echo "Temporary tar file created at $TEMP_TAR_FILE"
# Create the transfer job
kubectl -n $namespace create -f $transfer_file

# Get the full pod and job names
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)

# Wait for the transfer job to start
kubectl -n $namespace wait --for=condition=Ready pod/$pod_name --timeout=60s

# Create directory on the PVC if it doesn't exist
kubectl -n $namespace exec "$pod_name" -- mkdir -p "$PVC_MOUNT_PATH"

# Copy the tar file to the PVC
kubectl -n $namespace cp "$TEMP_TAR_FILE" "$pod_name":"$PVC_MOUNT_PATH"

# Extract the contents of the tar file on the PVC
kubectl -n $namespace exec "$pod_name" -- tar -xf "$PVC_MOUNT_PATH/ar_folder.tar" -C "$PVC_MOUNT_PATH"

# Clean up the temporary tar file
rm "$TEMP_TAR_FILE"

# Delete the tar file from the PVC
kubectl -n $namespace exec "$pod_name" -- rm "$PVC_MOUNT_PATH/ar_folder.tar"


#PVC_MOUNT_PATH_MODEL="mnt/ceph_rbd/ar-dp/models"
#PVC_MOUNT_PATH_LOG="mnt/ceph_rbd/ar-dp/logs"
#kubectl -n $namespace exec "$pod_name" -- mkdir -p "$PVC_MOUNT_PATH_MODEL"
#kubectl -n $namespace exec "$pod_name" -- mkdir -p "$PVC_MOUNT_PATH_LOG"

# Delete the transfer job
kubectl -n $namespace delete job $job_name

