#!/bin/bash
# ----------------------------------------------------------------------------------
# This script iterates over files in a subdirectory on a Kubernetes pod,
# splits each file if it exceeds a specified size (to avoid transferring huge files
# in one piece), copies the resulting split files to a local directory,
# and then recombines them to reconstruct the original file.
#
# Requirements:
#   - kubectl must be installed and configured.
#   - The pod must have 'split', 'find', and shell utilities available.
#
# Usage:
#   ./copy_and_recombine.sh <POD_NAME> <REMOTE_DIR> <LOCAL_DIR> <SIZE_LIMIT>
#
# Example:
#   ./copy_and_recombine.sh my-pod /app/data ./downloaded 100M
# ----------------------------------------------------------------------------------

# if [ "$#" -ne 4 ]; then
#     echo "Usage: $0 <POD_NAME> <REMOTE_DIR> <LOCAL_DIR> <SIZE_LIMIT>"
#     echo "Example: $0 my-pod /app/data ./downloaded 100M"
#     exit 1
# fi
job_prefix="ar-dp-transfer-job"  # The job prefix to identify the pod
namespace='eidf151ns'

transfer_file='/home/eidf151/eidf151/arichardson/Differentiable-Patterning/transfer.yml'
kubectl -n $namespace create -f $transfer_file
job_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_job_from_job_prefix.sh $job_prefix)
POD_NAME=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $job_prefix)
kubectl -n $namespace wait --for=condition=Ready pod/$POD_NAME --timeout=60s

REMOTE_DIR="/mnt/ceph/ar-dp/output"    # subdirectory in the pod to search files within
LOCAL_DIR="/home/eidf151/eidf151/arichardson/Differentiable-Patterning/output"     # local directory to store files
SIZE_LIMIT="50m"    # e.g., 100M

# Create the local directory if it does not exist
mkdir -p "$LOCAL_DIR"

echo "Listing original files in pod '$POD_NAME:$REMOTE_DIR'..."
# Use find on the pod to list regular files (non-recursively).
original_files=$(kubectl -n $namespace exec "$POD_NAME" -- sh -c "find '$REMOTE_DIR' -maxdepth 1 -type f")
if [ -z "$original_files" ]; then
  echo "No files found in $REMOTE_DIR on pod $POD_NAME"
  exit 0
fi

echo "Processing files..."
while IFS= read -r remote_file; do
    # Skip empty lines.
    [ -z "$remote_file" ] && continue

    filename=$(basename "$remote_file")
    echo "---------------------------"
    echo "Processing file: $filename"

    # Use a unique prefix for naming split files.
    # For example, if filename is "largefile.dat", the parts will be:
    #   splitted_largefile.dat.aa, splitted_largefile.dat.ab, etc.
    split_prefix="splitted_${filename}."

    echo "Splitting $filename in the pod into chunks of size $SIZE_LIMIT..."
    # Change into the remote directory and split the file.
    kubectl -n $namespace exec "$POD_NAME" -- sh -c "cd '$REMOTE_DIR' && split -b '$SIZE_LIMIT' '$filename' '$split_prefix'"
    if [ $? -ne 0 ]; then
        echo "Error splitting $filename on pod. Skipping."
        continue
    fi

    echo "Listing split files on the pod..."
    # Use find to list only regular files with names that start with the split prefix.
    remote_parts=$(kubectl -n $namespace exec "$POD_NAME" -- sh -c "find '$REMOTE_DIR' -maxdepth 1 -type f -name '${split_prefix}*'")
    if [ -z "$remote_parts" ]; then
        echo "No split parts found for $filename on the pod. Skipping."
        continue
    fi

    # Read the results into an array, one filename per element.
    # (IFS is set to newline so that file names with spaces are preserved.)
    IFS=$'\n' read -rd '' -a parts <<<"$remote_parts"

    # (Optional) Filter out any entry that might exactly match the REMOTE_DIR.
    filtered_parts=()
    for part in "${parts[@]}"; do
        if [ "$part" != "$REMOTE_DIR" ]; then
            filtered_parts+=("$part")
        fi
    done

    if [ ${#filtered_parts[@]} -eq 0 ]; then
        echo "No valid split parts found for $filename. Skipping."
        continue
    fi

    echo "Copying ${#filtered_parts[@]} split parts individually for $filename..."
    for remote_part in "${filtered_parts[@]}"; do
        echo "Copying remote file: $remote_part"
        # Copy the file from the pod. The remote file name should be an absolute path.
        kubectl -n $namespace cp "${POD_NAME}:${remote_part}" "$LOCAL_DIR/"
        if [ $? -ne 0 ]; then
            echo "Error copying $remote_part. Skipping recombination for $filename."
            continue 2
        fi
    done

    echo "Recombining parts locally into $LOCAL_DIR/$filename..."
    # We expect that the split parts have names that, when sorted lexically, are in the right order.
    # Bash globbing returns the files in lexicographic order by default.
    local_parts=( "$LOCAL_DIR"/${split_prefix}* )
    if [ ${#local_parts[@]} -eq 0 ]; then
        echo "No local split parts found for $filename. Skipping recombination."
        continue
    fi

    cat "${local_parts[@]}" > "$LOCAL_DIR/$filename"
    if [ $? -eq 0 ]; then
        echo "Successfully reassembled $filename to $LOCAL_DIR/$filename"
    else
        echo "Error reassembling $filename."
    fi

    # Clean up local split parts.
    rm -f "$LOCAL_DIR"/${split_prefix}*

    # Optionally, remove the split parts from the pod.
    echo "Cleaning up split parts on the pod..."
    kubectl -n $namespace exec "$POD_NAME" -- sh -c "cd '$REMOTE_DIR' && rm -f ${split_prefix}*"

done <<< "$original_files"

echo "All files processed."




kubectl -n $namespace delete job $job_name