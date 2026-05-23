#!/usr/bin/env python
"""Summarize a set of per-env CSA diagnostic JSON reports into one table.

Usage::

    python experiments/scripts/csa/summarize_diagnostics.py \
        --report wall=/tmp/csa_wall/wall_report.json \
        --report pusht=/tmp/csa_pusht/pusht_report.json \
        --output /tmp/csa_summary.csv

Each ``--report`` flag is ``<env>=<path>``. The script emits a CSV with one row
per env, columns: ratio_failed_quadrant (诊断 0 primary), gap_mean (诊断 1),
gap_p90 (诊断 1), conditional_pearson (诊断 2), bucket_high_csa_failure_ratio
(诊断 3), fn_rate_avg (诊断 4). Missing fields land as ``nan`` so the table
is robust to env-specific gaps in the data.

The Go/No-Go gate decision per memo §6 is made *on this summary*, not in
any one report. The primary signal is the 诊断 0 ratio averaged across envs;
the other diagnostics provide context.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _get(d: Dict[str, Any], *keys: str, default: Any = float("nan")) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default


def summarize(env: str, report: Dict[str, Any]) -> Dict[str, Any]:
    diag0_primary = _get(report, "diagnostic_0_failure_mode", "primary", default={})
    diag1 = _get(report, "diagnostic_1_exploitation_gap", default={})
    diag2 = _get(report, "diagnostic_2_conditioned_correlation", default={})
    diag3 = _get(report, "diagnostic_3_buckets", default={})
    diag4 = _get(report, "diagnostic_4_false_negative_by_depth", default={})

    # 诊断 3: bucketed ratios per (state_familiar × CSA_unsupported).
    state_low_csa_high = _get(diag3, "state_low_csa_high", default={})
    state_high_csa_high = _get(diag3, "state_high_csa_high", default={})

    # 诊断 4: per-depth false-negative rate; take a simple mean for the summary.
    by_depth = diag4 if isinstance(diag4, dict) else {}
    fn_rates = []
    for k, v in by_depth.items():
        if isinstance(v, dict) and isinstance(v.get("false_negative_rate"), (int, float)):
            fn_rates.append(float(v["false_negative_rate"]))
    fn_avg = sum(fn_rates) / len(fn_rates) if fn_rates else float("nan")

    return {
        "env": env,
        "num_episodes": int(_get(report, "num_episodes", default=0)),
        "num_failed_episodes": int(_get(report, "num_failed_episodes", default=0)),
        "diag0_ratio_failed_quadrant": float(_get(diag0_primary, "ratio_failed", default=float("nan"))),
        "diag0_num_failed_in_quadrant": int(_get(diag0_primary, "num_failed_in_quadrant", default=0)),
        "diag1_gap_mean": float(_get(diag1, "mean_gap", default=float("nan"))),
        "diag1_gap_p90": float(_get(diag1, "p90_gap", default=float("nan"))),
        "diag1_fraction_real_worse": float(_get(diag1, "fraction_real_worse_than_pred", default=float("nan"))),
        "diag2_pearson": float(_get(diag2, "pearson", default=float("nan"))),
        "diag2_spearman": float(_get(diag2, "spearman", default=float("nan"))),
        "diag3_state_low_csa_high_failure_ratio": float(_get(state_low_csa_high, "failure_ratio", default=float("nan"))),
        "diag3_state_high_csa_high_failure_ratio": float(_get(state_high_csa_high, "failure_ratio", default=float("nan"))),
        "diag4_fn_rate_avg_across_depths": fn_avg,
        "go_no_go_verdict": str(_get(report, "go_no_go", "verdict", default="")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="env=path/to/<env>_report.json (may be repeated)",
    )
    parser.add_argument("--output", required=True, help="CSV path to write")
    args = parser.parse_args()

    rows = []
    for entry in args.report:
        env, _, path = entry.partition("=")
        env = env.strip()
        path = path.strip()
        if not env or not path:
            print(f"Skipping malformed --report entry: {entry!r}", file=sys.stderr)
            continue
        report_path = Path(path)
        if not report_path.exists():
            print(f"WARNING: report not found, writing nan row: {report_path}", file=sys.stderr)
            rows.append({"env": env})
            continue
        with report_path.open() as f:
            report = json.load(f)
        rows.append(summarize(env, report))

    if not rows:
        print("ERROR: no report rows to write", file=sys.stderr)
        sys.exit(1)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    # Pin env first for readability.
    fieldnames.remove("env")
    fieldnames.insert(0, "env")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote summary table → {output} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
