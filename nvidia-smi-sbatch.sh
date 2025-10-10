#!/usr/bin/env bash
# nvidia-smi-sbatch.sh
# Usage: nvidia-smi-sbatch.sh JOBID|NAME [nvidia-smi args...]
set -euo pipefail

# brief usage
usage() {
    cat <<EOF
Usage: $0 JOBID|NAME [nvidia-smi args...]
Attach to an existing Slurm job allocation and run nvidia-smi there.
If a NAME (job name) is given, the script searches your jobs and picks the first match.
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

if ! command -v srun >/dev/null 2>&1 || ! command -v squeue >/dev/null 2>&1; then
    echo "This script requires Slurm utilities (srun, squeue)." >&2
    exit 2
fi

QUERY="$1"
shift
EXTRA_ARGS=("$@")

# determine job id
if [[ "$QUERY" =~ ^[0-9]+$ ]]; then
    JOBID="$QUERY"
else
    # look for a job with that name belonging to the current user
    JOBID=$(squeue -h -o "%i %j %u %t %N" | awk -v name="$QUERY" -v user="$USER" '$2==name && $3==user {print $1; exit}')
    if [[ -z "$JOBID" ]]; then
        # fallback: any user
        JOBID=$(squeue -h -o "%i %j %u %t %N" | awk -v name="$QUERY" '$2==name {print $1; exit}')
    fi
fi

if [[ -z "${JOBID:-}" ]]; then
    echo "Job not found: $QUERY" >&2
    exit 3
fi

echo "Attaching to job $JOBID and running: nvidia-smi ${EXTRA_ARGS[*]}"

# Run nvidia-smi inside the existing job allocation.
# --jobid runs the step on the existing job allocation.
# --ntasks=1 ensures a single task (one node/CPU). --unbuffered avoids output buffering.
# --pty ensures environment like an interactive step; drop it if your cluster disallows pty attach.
srun --jobid="$JOBID" --ntasks=1 --unbuffered --pty nvidia-smi "${EXTRA_ARGS[@]}"