#!/usr/bin/env python
"""Roll up per-env ``*_residual_correlation_report.json`` files into:

  * a compact CSV (one row per env x variant): per-depth Pearson(r),
    Pearson(U_action), Pearson(U_SA) on the state-familiar stratum at
    depths 1-3, plus the per-env C verdict;
  * a Markdown summary anchored on the PRIMARY raw variant, with the
    overall C-pass-fail call and the "justifies B?" recommendation.

The roll-up answers the four items in the v3 §C output spec:
  1. whether C passes overall (anchored on raw_recipRMS, per-env);
  2. which envs pass / fail;
  3. whether r(q) is better than support distance;
  4. whether C is strong enough to justify proceeding to Exp B.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PRIMARY_VARIANT = "raw_recipRMS"
SENSITIVITY_VARIANT = "pca64_white_recipRMS"
PRIMARY_DEPTHS = (1, 2, 3)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:+.3f}"
    return str(v)


def _cell(per_depth: Dict[str, Any], d: int, scorer: str, stratum: str = "familiar_q50") -> Any:
    block = per_depth.get(f"depth_{d}", {}).get(stratum, {})
    if block.get("count", 0) < 2:
        return None
    return block.get(scorer, {}).get("pearson")


def _row(env_seed: str, report: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    v = report.get("variants", {}).get(variant_name, {})
    per_depth = v.get("per_depth", {})
    row: Dict[str, Any] = {
        "env_seed": env_seed,
        "env": report.get("env", ""),
        "variant": variant_name,
        "num_queries_scored": report.get("num_queries_scored"),
        "residual_size": report.get("raw_residual_size"),
        "residual_mean": report.get("residual_memory_summary", {}).get("mean_residual"),
        "residual_p95": report.get("residual_memory_summary", {}).get("p95_residual"),
    }
    for d in PRIMARY_DEPTHS:
        row[f"pearson_r_d{d}"] = _cell(per_depth, d, "r")
        row[f"pearson_u_action_d{d}"] = _cell(per_depth, d, "u_action")
        row[f"pearson_u_sa_d{d}"] = _cell(per_depth, d, "u_sa")
    rb = v.get("r_beats_summary", {})
    row["r_beats_u_action_fraction_all_depths"] = rb.get("r_beats_u_action_fraction")
    row["r_beats_u_sa_fraction_all_depths"] = rb.get("r_beats_u_sa_fraction")
    verdict = report.get("verdict", {}).get(variant_name, {})
    row["r_beats_u_action_fraction_primary_depths"] = verdict.get("r_beats_u_action_fraction")
    row["r_beats_u_sa_fraction_primary_depths"] = verdict.get("r_beats_u_sa_fraction")
    row["passes_vs_u_action"] = verdict.get("passes_vs_u_action")
    row["passes_vs_u_sa"] = verdict.get("passes_vs_u_sa")
    row["passes_c"] = verdict.get("passes_c")
    row["env_c_pass_under_raw"] = report.get("verdict", {}).get("env_c_pass_under_raw")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True,
                        help="paths to per-env *_residual_correlation_report.json files")
    parser.add_argument("--output-csv", required=True, help="output CSV path")
    parser.add_argument("--output-md", default=None,
                        help="optional Markdown summary path (default: derived from --output-csv)")
    parser.add_argument("--env-seed-from-filename", action="store_true",
                        help="extract <stem-with-_residual_correlation_report> as env_seed")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    raw_envs_pass: List[str] = []
    raw_envs_fail: List[str] = []
    primary_anchor: Dict[str, Dict[str, Any]] = {}

    for path_str in args.reports:
        path = Path(path_str)
        with path.open("r") as f:
            report = json.load(f)
        if args.env_seed_from_filename:
            env_seed = path.stem.replace("_residual_correlation_report", "")
        else:
            env_seed = report.get("env", path.stem)

        for variant_name in (PRIMARY_VARIANT, SENSITIVITY_VARIANT):
            if variant_name not in report.get("variants", {}):
                continue
            rows.append(_row(env_seed, report, variant_name))

        primary_anchor[env_seed] = report.get("verdict", {})
        if bool(report.get("verdict", {}).get("env_c_pass_under_raw")):
            raw_envs_pass.append(env_seed)
        else:
            raw_envs_fail.append(env_seed)

    if not rows:
        sys.stderr.write("ERROR: no rows produced (empty / unparseable reports?)\n")
        return 1

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (_fmt(row.get(k)) if isinstance(row.get(k), float) else row.get(k))
                             for k in fieldnames})

    md_path = Path(args.output_md) if args.output_md else csv_path.with_suffix(".md")

    n_envs = len(primary_anchor)
    n_pass = len(raw_envs_pass)
    overall_pass = n_pass >= max(3, (n_envs + 1) // 2)  # majority + at least 3 envs

    lines: List[str] = []
    lines.append("# Exp C (residual calibration) cross-env summary")
    lines.append("")
    lines.append(
        f"Anchored on PRIMARY variant **{PRIMARY_VARIANT}** (raw / no PCA). "
        f"PCA-64 (`{SENSITIVITY_VARIANT}`) shown as sensitivity check."
    )
    lines.append(
        "C-pass per env: |Pearson(r)| > |Pearson(U_action|x)| AND > |Pearson(U_SA)| on "
        "the state-familiar stratum (q=0.50), on a majority of depths 1-3."
    )
    lines.append("")
    lines.append(f"## Overall verdict (anchored on raw)")
    lines.append("")
    lines.append(f"- envs C-passes: **{n_pass}/{n_envs}**")
    if raw_envs_pass:
        lines.append(f"- pass list: {', '.join(sorted(raw_envs_pass))}")
    if raw_envs_fail:
        lines.append(f"- fail list: {', '.join(sorted(raw_envs_fail))}")
    if overall_pass:
        lines.append("")
        lines.append(
            "**C passes overall** (majority of envs, >= 3). Per v3 §5, this is one of "
            "the two Hard Go conditions; the other is Exp B (residual-based reranking "
            "improving over both vanilla and support-based reranking). **B is now "
            "justified** -- next step is to implement the 5-variant offline reranker "
            "table on the same logged top-K candidates."
        )
    else:
        lines.append("")
        lines.append(
            "**C does not pass overall.** Per v3 §5, this is a Soft-Go or No-Go case "
            "depending on Exp A. Do NOT propose an online planner change. The honest "
            "fallback is the **Failure-Mode Audit paper** (Q7): publish the diagnostic "
            "protocol + the negative C result. Exp B can still be run as a sanity check "
            "but the v3 reframing's methodological core has failed."
        )
    lines.append("")
    lines.append(f"## Per-env table (variant = {PRIMARY_VARIANT})")
    lines.append("")
    lines.append(
        "| env_seed | num_q | M_resid | r vs U_act (d=1..3) | r vs U_SA (d=1..3) | C-pass? |"
    )
    lines.append("|---|---|---|---|---|---|")
    primary_rows = [r for r in rows if r["variant"] == PRIMARY_VARIANT]
    primary_rows.sort(key=lambda r: r["env_seed"])
    for r in primary_rows:
        lines.append(
            f"| {r['env_seed']} | {r['num_queries_scored']} | {r['residual_size']} | "
            f"{_fmt(r['r_beats_u_action_fraction_primary_depths'])} | "
            f"{_fmt(r['r_beats_u_sa_fraction_primary_depths'])} | "
            f"{'PASS' if r['passes_c'] else 'fail'} |"
        )
    lines.append("")
    lines.append(f"## Per-env table (variant = {SENSITIVITY_VARIANT}, sensitivity)")
    lines.append("")
    lines.append(
        "| env_seed | num_q | M_resid | r vs U_act (d=1..3) | r vs U_SA (d=1..3) | C-pass? |"
    )
    lines.append("|---|---|---|---|---|---|")
    sens_rows = [r for r in rows if r["variant"] == SENSITIVITY_VARIANT]
    sens_rows.sort(key=lambda r: r["env_seed"])
    for r in sens_rows:
        lines.append(
            f"| {r['env_seed']} | {r['num_queries_scored']} | {r['residual_size']} | "
            f"{_fmt(r['r_beats_u_action_fraction_primary_depths'])} | "
            f"{_fmt(r['r_beats_u_sa_fraction_primary_depths'])} | "
            f"{'PASS' if r['passes_c'] else 'fail'} |"
        )
    lines.append("")
    lines.append("## What the table is NOT")
    lines.append("")
    lines.append(
        "- r(q) is NOT a calibrated uncertainty or a safety signal. It is a local "
        "average of validation residuals around q = (x, a)."
    )
    lines.append(
        "- A C-pass means r(q) is a *better* predictor of rollout error than the "
        "coverage proxy U_action|x. It does NOT mean r(q) is a *good* predictor in "
        "absolute terms; for that, inspect the raw Pearson values in the CSV."
    )
    lines.append(
        "- A C-pass alone does NOT justify online planner changes. Exp B must also "
        "pass (residual-based reranking improving selected candidate quality)."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote per-env CSV     -> {csv_path}")
    print(f"Wrote summary report  -> {md_path}")
    print(f"Overall C-pass-under-raw: {n_pass}/{n_envs} envs ({'PASS' if overall_pass else 'fail'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
