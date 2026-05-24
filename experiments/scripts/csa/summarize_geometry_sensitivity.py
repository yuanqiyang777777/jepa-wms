#!/usr/bin/env python
"""Roll up N per-env ``*_geometry_sensitivity_report.json`` files into one CSV
+ a Markdown-friendly verdict table.

The CSV row schema (one row per env x variant x quadrant_slice):

    env, env_seed, num_episodes, num_failed, variant_name, variant_pca_dim,
    variant_whiten, variant_calibration, is_primary_no_pca, quadrant_slice,
    Q1_fraction, Q2_fraction, Q3_fraction, Q4_fraction, Q1_count, ...,
    Q1_failure_rate, Q2_failure_rate, ..., Q1_mean_error, ...,
    Q2_enrichment_fail_over_succ, Q3_enrichment, Q4_enrichment,
    Q2_error_ratio_vs_Q1, dominant_failure_quadrant,
    dominant_failure_quadrant_enrichment, gate_verdict

``quadrant_slice`` is one of:
    "all_queries", "decision_steps_only", "rollout_only"

``gate_verdict`` is advisory: "Q2_enrichment_ge_2" if Q2 enrichment >= 2 AND
Q2 mean-error ratio vs Q1 >= 1.2 AND Q2 is the dominant failure quadrant.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SLICE_KEYS = {
    "all_queries": "census_all_queries",
    "decision_steps_only": "census_decision_steps_only",
    "rollout_only": "census_rollout_only",
}

QUADRANTS = (
    "Q1_familiar_familiar",
    "Q2_familiar_unfamiliar",
    "Q3_unfamiliar_familiar",
    "Q4_unfamiliar_unfamiliar",
)


def _rows_for_report(report: Dict[str, Any], env_seed: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    env = report.get("env", "")
    n_ep = report.get("num_episodes", 0)
    n_fail = report.get("num_failed_episodes", 0)
    variants = report.get("variants", {})
    for variant_name, vblock in variants.items():
        v = vblock["variant"]
        for slice_key, report_key in SLICE_KEYS.items():
            census = vblock.get(report_key, {})
            if not isinstance(census, dict) or census.get("status") == "not_computed":
                continue
            row: Dict[str, Any] = {
                "env": env,
                "env_seed": env_seed,
                "num_episodes": n_ep,
                "num_failed": n_fail,
                "variant_name": variant_name,
                "variant_pca_dim": v.get("pca_dim"),
                "variant_whiten": v.get("whiten"),
                "variant_calibration": v.get("calibration"),
                "is_primary_no_pca": v.get("is_primary_no_pca_variant", False),
                "quadrant_slice": slice_key,
            }
            per_q = census.get("per_quadrant_all_decision_steps", {})
            by_outcome = census.get("by_outcome", {})
            all_dist = by_outcome.get("all_decision_steps", {})
            enrichment = census.get("enrichment_failed_over_successful", {})
            err_ratios = census.get("mean_error_ratio_vs_Q1", {})
            for q in QUADRANTS:
                row[f"{q[:2]}_fraction"] = all_dist.get(q)
                stats = per_q.get(q, {})
                row[f"{q[:2]}_count"] = stats.get("count")
                row[f"{q[:2]}_failure_rate"] = stats.get("failure_rate")
                row[f"{q[:2]}_mean_error"] = stats.get("mean_error")
                row[f"{q[:2]}_enrichment_fail_over_succ"] = enrichment.get(q)
                row[f"{q[:2]}_error_ratio_vs_Q1"] = err_ratios.get(q)
            row["dominant_failure_quadrant"] = census.get("dominant_failure_quadrant")
            row["dominant_failure_quadrant_enrichment"] = census.get("dominant_failure_quadrant_enrichment")
            # Advisory verdict for the Q2 (CSA-key) cell.
            q2_enrich = enrichment.get("Q2_familiar_unfamiliar")
            q2_err = err_ratios.get("Q2_familiar_unfamiliar")
            dom = census.get("dominant_failure_quadrant")
            if (q2_enrich is not None and q2_enrich >= 2.0
                    and q2_err is not None and q2_err >= 1.2
                    and dom == "Q2_familiar_unfamiliar"):
                row["q2_gate_verdict"] = "Q2_supported"
            elif q2_enrich is not None and q2_enrich >= 1.5:
                row["q2_gate_verdict"] = "Q2_marginal"
            else:
                row["q2_gate_verdict"] = "Q2_unsupported"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports", nargs="+", required=True,
        help="paths to *_geometry_sensitivity_report.json files",
    )
    parser.add_argument("--output-csv", required=True, help="output CSV path")
    parser.add_argument(
        "--seed-from-filename", action="store_true",
        help="extract a 'seed' suffix from each filename (e.g. wall_seed1 -> 'seed1')",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path_str in args.reports:
        path = Path(path_str)
        with path.open("r") as f:
            report = json.load(f)
        if args.seed_from_filename:
            stem = path.stem
            env_seed = ""
            for part in stem.split("_"):
                if part.startswith("seed"):
                    env_seed = part
                    break
        else:
            env_seed = ""
        rows.extend(_rows_for_report(report, env_seed))

    if not rows:
        sys.stderr.write("ERROR: no rows produced (empty / unparseable reports?)\n")
        return 1

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    # Stable column order: env first, then env_seed, then variant, then quadrant slice + numbers
    head = ["env", "env_seed", "num_episodes", "num_failed",
            "variant_name", "variant_pca_dim", "variant_whiten",
            "variant_calibration", "is_primary_no_pca", "quadrant_slice"]
    tail = [k for k in fieldnames if k not in head]
    fieldnames = head + sorted(tail)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote geometry-sensitivity summary -> {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
