#!/bin/bash

#----- Set up paths -----#

# Target folder paths
# TARGET_MODEL_DIRECTORY="home/eidf151/eidf151/arichardson/Differentiable-Patterning/models/"
# TARGET_LOG_DIRECTORY="home/eidf151/eidf151/arichardson/Differentiable-Patterning/logs/"
TARGET_MODEL_DIRECTORY="models/"
TARGET_LOG_DIRECTORY="logs/"
TARGET_OUTPUT_DIRECTORY="output/"

# Transfer yml file
transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/transfer.yml'
# File to transfer
PVC_MOUNT_PATH_MODEL="mnt/ceph/ar-dp/models"
PVC_MOUNT_PATH_LOG="mnt/ceph/ar-dp/logs"
PVC_MOUNT_PATH_OUTPUT="mnt/ceph/ar-dp/output"
# Transfer job name
job_prefix="ar-dp-transfer-job"
# Namespace
namespace='eidf151ns'






#----- Launch transfer job and wait until it starts -----#

# Start the transfer job
kubectl -n $namespace create -f $transfer_file
# Get the pod name of the pod attached to the transfer job
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)
# Wait for the transfer job to start
kubectl -n $namespace wait --for=condition=Ready pod/$pod_name --timeout=60s


#kubectl -n $namespace exec "$pod_name" -- mkdir -p "$PVC_MOUNT_PATH_MODEL"
#kubectl -n $namespace exec "$pod_name" -- mkdir -p "$PVC_MOUNT_PATH_LOG"

# Define tar files
#PVC_MODEL_TAR_FILE="/mnt/ceph_rbd/ar-dp/tmp_tar/model.tar"
#PVC_LOG_TAR_FILE="/mnt/ceph_rbd/ar-dp/tmp_tar/log.tar"
#LOCAL_MODEL_TAR_FILE="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/models/model.tar"
#LOCAL_LOG_TAR_FILE="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/logs/log.tar"


# Create tar file of transfer file on the PVC
#kubectl -n $namespace exec "$pod_name" -- mkdir "mnt/ceph_rbd/ar-dp/tmp_tar"

#kubectl -n $namespace exec "$pod_name" -- tar -cf "$PVC_MODEL_TAR_FILE" "$PVC_MOUNT_PATH_MODEL" #.
# Create tar file of transfer file on the PVC
#kubectl -n $namespace exec "$pod_name" -- tar -cf "$PVC_LOG_TAR_FILE" "$PVC_MOUNT_PATH_LOG" #.


# Copy the model file from the PVC

echo "Copying model file from PVC"
kubectl -n $namespace cp "$pod_name":"$PVC_MOUNT_PATH_MODEL" "$TARGET_MODEL_DIRECTORY"
# Copy the log file from the PVC
echo "Copying log files from PVC"
kubectl -n $namespace cp "$pod_name":"$PVC_MOUNT_PATH_LOG" "$TARGET_LOG_DIRECTORY"

echo "Copying output file from PVC"
kubectl -n $namespace cp "$pod_name":"$PVC_MOUNT_PATH_OUTPUT" "$TARGET_OUTPUT_DIRECTORY"


# Delete model tar from PVC
kubectl -n $namespace exec "$pod_name" -- sh -c "rm ${PVC_MOUNT_PATH_MODEL}/*"
# Delete log tar from PVC
kubectl -n $namespace exec "$pod_name" -- sh -c "rm -r ${PVC_MOUNT_PATH_LOG}/*"
# Delete output tar from PVC
kubectl -n $namespace exec "$pod_name" -- sh -c "rm -r ${PVC_MOUNT_PATH_OUTPUT}/*"


# Delete the transfer job
kubectl -n $namespace delete job $job_name


# Extract the contents of the tar file 
#tar -xf "$LOCAL_MODEL_TAR_FILE" "$TARGET_MODEL_DIRECTORY"
#tar -xf "$LOCAL_LOG_TAR_FILE" "$TARGET_LOG_DIRECTORY"
#rm "$LOCAL_MODEL_TAR_FILE"
#rm "$LOCAL_LOG_TAR_FILE"