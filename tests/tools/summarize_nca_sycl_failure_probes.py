#!/usr/bin/env python3
"""Summarize probe outcomes and host recurrence from Slurm stdout/stderr."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import statistics


def field(text, name, default="unknown"):
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_directory", type=pathlib.Path)
    parser.add_argument("--job-id", help="Only include JOBID-TASKID.out files")
    args = parser.parse_args()

    records = []
    pattern = f"{args.job_id}-*.out" if args.job_id else "*.out"
    for stdout in sorted(args.log_directory.glob(pattern)):
        stderr = stdout.with_suffix(".err")
        text = stdout.read_text(errors="replace")
        error_text = stderr.read_text(errors="replace") if stderr.exists() else ""
        passed = "NCA_SYCL_FAILURE_PROBE_RESULT=PASS" in text
        records.append(
            (
                field(text, "PROBE"),
                field(text, "HOSTNAME"),
                "PASS" if passed else "CRASH_OR_FAIL",
                stdout.name,
                field(text + "\n" + error_text, "JAX_VERSION"),
                field(text, "ELAPSED_SECONDS", default=""),
            )
        )

    if not records:
        raise SystemExit(f"No .out files found in {args.log_directory}")

    counts = collections.Counter(
        (probe, outcome) for probe, _, outcome, _, _, _ in records
    )
    hosts = collections.Counter(host for _, host, _, _, _, _ in records)
    host_failures = collections.Counter(
        host for _, host, outcome, _, _, _ in records if outcome != "PASS"
    )
    print("PROBE OUTCOME COUNT")
    for key, count in sorted(counts.items()):
        print(*key, count, sep="\t")
    print("\nHOST TOTAL FAILURES")
    for host, total in sorted(hosts.items()):
        print(host, total, host_failures[host], sep="\t")
    print("\nPROBE PASS ELAPSED_MEDIAN_SECONDS ELAPSED_MIN_SECONDS ELAPSED_MAX_SECONDS")
    probes = sorted({probe for probe, _, _, _, _, _ in records})
    for probe in probes:
        elapsed = [
            float(value)
            for item_probe, _, outcome, _, _, value in records
            if item_probe == probe and outcome == "PASS" and value
        ]
        if elapsed:
            print(
                probe,
                len(elapsed),
                statistics.median(elapsed),
                min(elapsed),
                max(elapsed),
                sep="\t",
            )
    print("\nFAILED LOGS")
    for probe, host, outcome, filename, version, _ in records:
        if outcome != "PASS":
            print(probe, host, version, filename, sep="\t")


if __name__ == "__main__":
    main()
