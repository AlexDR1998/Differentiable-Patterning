#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 --cluster {eidf|dawn} [--port PORT]"
    echo "  --port is required when --cluster eidf is selected."
}

cluster=""
port=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cluster)
            [[ $# -ge 2 ]] || { usage >&2; exit 1; }
            cluster="$2"
            shift 2
            ;;
        -p|--port)
            [[ $# -ge 2 ]] || { usage >&2; exit 1; }
            port="$2"
            shift 2
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

case "$cluster" in
    eidf)
        if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
            echo "A valid --port (1-65535) is required for EIDF." >&2
            usage >&2
            exit 1
        fi
        rsync -a -e "ssh -J eidf-1 -p $port" root@localhost:/data/Models/bundles/ ./models/bundles/
        ;;
    dawn)
        if [[ -n "$port" ]]; then
            echo "--port is only valid when copying from EIDF." >&2
            usage >&2
            exit 1
        fi
        rsync -r -e ssh rc-rich1@login-dawn.hpc.cam.ac.uk:/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs/Models/bundles/ ./models/bundles/
        ;;
    *)
        echo "--cluster must be either 'eidf' or 'dawn'." >&2
        usage >&2
        exit 1
        ;;
esac

python -m Experiments.model_registry --root ./models reindex
