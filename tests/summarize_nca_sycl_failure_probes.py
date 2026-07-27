#!/usr/bin/env python3
"""Summarize probe outcomes and host recurrence from Slurm stdout/stderr."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re


def field(text, name, default="unknown"):
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_directory", type=pathlib.Path)
    args = parser.parse_args()

    records = []
    for stdout in sorted(args.log_directory.glob("*.out")):
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
            )
        )

    if not records:
        raise SystemExit(f"No .out files found in {args.log_directory}")

    counts = collections.Counter((probe, outcome) for probe, _, outcome, _, _ in records)
    hosts = collections.Counter(host for _, host, _, _, _ in records)
    host_failures = collections.Counter(
        host for _, host, outcome, _, _ in records if outcome != "PASS"
    )
    print("PROBE OUTCOME COUNT")
    for key, count in sorted(counts.items()):
        print(*key, count, sep="\t")
    print("\nHOST TOTAL FAILURES")
    for host, total in sorted(hosts.items()):
        print(host, total, host_failures[host], sep="\t")
    print("\nFAILED LOGS")
    for probe, host, outcome, filename, version in records:
        if outcome != "PASS":
            print(probe, host, version, filename, sep="\t")


if __name__ == "__main__":
    main()
