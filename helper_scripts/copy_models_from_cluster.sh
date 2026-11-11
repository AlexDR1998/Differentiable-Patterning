#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 --port PORT [--no-impulses] [--no-reindex] [--no-results]"
    echo "  Copies Models/bundles and Models/impulses from EIDF into ./models."
    echo "  Also copies notebook job results (output/, e.g. the density x KO sweep"
    echo "  parts in output/nodal_ko_eval/<job>/) into ./output."
    echo "  --port is the SSH port of the EIDF pod (reached through eidf-1)."
    echo "  --no-impulses skips copying Models/impulses (slow)."
    echo "  --no-reindex skips rebuilding models/registry.sqlite after the copy."
    echo "  --no-results skips copying the notebook job results."
}

port=""
reindex=true
results=true
model_folders=(bundles impulses)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)
            [[ $# -ge 2 ]] || { usage >&2; exit 1; }
            port="$2"
            shift 2
            ;;
        --no-impulses)
            model_folders=(bundles)
            shift
            ;;
        --no-reindex)
            reindex=false
            shift
            ;;
        --no-results)
            results=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "A valid --port (1-65535) is required." >&2
    usage >&2
    exit 1
fi

eidf_ssh="ssh -J eidf-1 -p $port"
# One updating line per copy: bytes copied, overall %, speed and time left.
# --no-inc-recursive lists every file before copying, so the % is of the
# whole copy rather than of the files found so far (the listing takes a moment).
progress=(--info=progress2 --human-readable --no-inc-recursive)
# copy_folder LABEL SOURCE DESTINATION [rsync options]: SOURCE is a folder on the
# PVC (as mounted in the pod reached over SSH), DESTINATION a local folder.
copy_folder() {
    local label="$1" source="$2" destination="$3"
    shift 3
    mkdir -p "$destination"
    echo "Copying $label"
    echo "  from PVC:  root@localhost:$source"
    echo "  to local:  $(realpath "$destination")/"
    rsync -a "${progress[@]}" "$@" -e "$eidf_ssh" "root@localhost:$source" "$destination"
}

for folder in "${model_folders[@]}"; do
    copy_folder "Models/$folder" "/data/Models/$folder/" "./models/$folder/"
done

# Results of notebook jobs (docs/remote_notebook_jobs.md), written by pods to
# /workspace/writeable/output/, i.e. /data/output/ here. Parts still being
# written (hidden .tmp files) are left out.
if [[ "$results" == true ]]; then
    if $eidf_ssh root@localhost test -d /data/output; then
        copy_folder "notebook job results" "/data/output/" "./output/" --exclude '.*.tmp'
    else
        echo "No /data/output on EIDF; no notebook job results to copy."
    fi
fi

if [[ "$reindex" == true ]]; then
    echo "Reindexing the model registry"
    echo "  bundles:   $(realpath ./models)/bundles/"
    echo "  index:     $(realpath ./models)/registry.sqlite"
    start=$SECONDS
    python -m Experiments.model_registry --root ./models reindex > /dev/null
    echo "  done in $((SECONDS - start))s"
else
    echo "Skipped reindexing the model registry (--no-reindex)"
fi
