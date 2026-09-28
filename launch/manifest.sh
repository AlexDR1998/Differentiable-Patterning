#!/usr/bin/env bash
# Read the top-level fields of a sweep manifest
# (Experiments/<domain>/conf/generated/<sweep>/manifest.yaml) without Python.
# Source this file:  source "$(dirname "${BASH_SOURCE[0]}")/manifest.sh"

# manifest_field KEY FILE: print the value of a top-level "KEY: value" line.
manifest_field() {
    local key="$1" path="$2" line value
    line="$(grep -m1 -E "^${key}[[:space:]]*:" "$path" || true)"
    [[ -n "$line" ]] || return 1
    value="${line#*:}"
    value="$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    [[ -n "$value" && "$value" != "null" ]] || return 1
    printf '%s\n' "$value"
}

# manifest_count FILE: number of runs in the manifest.
manifest_count() {
    local path="$1" count
    count="$(manifest_field count "$path" || true)"
    if [[ -z "$count" ]]; then
        count="$(grep -c -E '^-[[:space:]]*index:' "$path" || true)"
    fi
    printf '%s\n' "$count"
}

# manifest_experiment_name FILE: experiment_name, else the manifest's folder name.
manifest_experiment_name() {
    local path="$1"
    manifest_field experiment_name "$path" || basename "$(dirname "$path")"
}
