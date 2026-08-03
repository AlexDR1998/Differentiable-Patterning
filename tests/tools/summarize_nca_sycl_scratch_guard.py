#!/usr/bin/env python3
"""Summarize one Slurm scratch-guard diagnostic array."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=pathlib.Path)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main():
    args = _arguments()
    outputs = sorted(args.log_root.glob(f"{args.job_id}-*.out"))
    if not outputs:
        raise SystemExit(f"No logs found for job {args.job_id} in {args.log_root}")

    summary = collections.defaultdict(collections.Counter)
    for output_path in outputs:
        text = output_path.read_text(errors="replace")
        error_path = output_path.with_suffix(".err")
        if error_path.exists():
            text += "\n" + error_path.read_text(errors="replace")
        mode_match = re.search(r"^SCRATCH_MODE=(\S+)$", text, re.MULTILINE)
        mode = mode_match.group(1) if mode_match else "unknown"
        if "NCA_SYCL_ROLLOUT_SCRATCH_GUARD_RESULT=PASS" in text:
            result = "pass"
        elif re.search(r"GUARD_CORRUPTIONS=(?!\[\])", text):
            result = "canary_failure"
        elif "differs from two one-step backward calls" in text:
            result = "gradient_failure"
        else:
            result = "crash_or_incomplete"
        summary[mode][result] += 1

    print(f"JOB_ID={args.job_id}")
    print(f"LOG_COUNT={len(outputs)}")
    for mode in sorted(summary):
        counts = summary[mode]
        print(
            f"MODE={mode} PASS={counts['pass']} "
            f"CANARY_FAILURE={counts['canary_failure']} "
            f"GRADIENT_FAILURE={counts['gradient_failure']} "
            f"CRASH_OR_INCOMPLETE={counts['crash_or_incomplete']}"
        )


if __name__ == "__main__":
    main()
