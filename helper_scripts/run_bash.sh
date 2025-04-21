#takes a name prefix and a command to run - then does kubectl exec <command> on the pod with that name prefix
# Usage: run_bash.sh <name-prefix> <command>
#!/bin/bash
# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <name-prefix> <command>"
  exit 1
fi
# Assign arguments to variables
PREFIX="$1"
COMMAND="$2"
# Get the pod name associated with the job prefix
pod_name=$(sh /home/eidf151/eidf151/arichardson/Differentiable-Patterning/helper_scripts/get_pod_from_job_prefix.sh $PREFIX)

kubectl -n eidf151ns exec  $pod_name -- sh -c "$COMMAND"