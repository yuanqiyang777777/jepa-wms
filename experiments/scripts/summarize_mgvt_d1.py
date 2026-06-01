#!/usr/bin/env python
"""Summarize MGVT-D Stage-D1 prediction-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_ORDER = ("PushT", "Wall", "PointMaze", "Metaworld")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _read_scores(skill_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(skill_root.glob("*/skill_score.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        horizons = sorted(data.get("skill_by_horizon", {}), key=lambda h: int(h))
        for horizon in horizons:
            rows.append(
                {
                    "run_id": path.parent.name,
                    "task": data.get("task"),
                    "model": data.get("model"),
                    "seed": data.get("seed"),
                    "horizon": int(horizon),
                    "skill": data.get("skill_by_horizon", {}).get(horizon),
                    "change_skill": data.get("change_skill_by_horizon", {}).get(horizon),
                    "moved_region_change_skill": data.get("moved_region_change_skill_by_horizon", {}).get(horizon),
                    "proprio_skill": data.get("proprio_skill_by_horizon", {}).get(horizon),
                    "param_count": data.get("param_count"),
                    "flops_per_forward_estimate": data.get("flops_per_forward_estimate"),
                    "train_step_time_ms": data.get("train_step_time_ms"),
                    "mamba_backend": data.get("mamba_backend") or "",
                    "deterministic_encode_max_abs_diff": data.get("deterministic_encode_max_abs_diff"),
                }
            )
    return rows


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["model"], row["horizon"])].append(row)
    out = []
    for (task, model, horizon), group in sorted(groups.items()):
        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in group if _finite(r.get(key))]

        moved = vals("moved_region_change_skill")
        skill = vals("skill")
        change = vals("change_skill")
        proprio = vals("proprio_skill")
        out.append(
            {
                "task": task,
                "model": model,
                "horizon": horizon,
                "n": len(group),
                "skill_mean": _mean(skill),
                "skill_std": _std(skill),
                "change_skill_mean": _mean(change),
                "change_skill_std": _std(change),
                "moved_region_change_skill_mean": _mean(moved),
                "moved_region_change_skill_std": _std(moved),
                "proprio_skill_mean": _mean(proprio),
                "proprio_skill_std": _std(proprio),
                "param_count_mean": round(_mean(vals("param_count")) or 0),
                "flops_per_forward_mean": round(_mean(vals("flops_per_forward_estimate")) or 0),
                "train_step_time_ms_mean": _mean(vals("train_step_time_ms")),
                "mamba_backend": ",".join(sorted({r["mamba_backend"] for r in group if r["mamba_backend"]})),
                "seeds": " ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"])),
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    def fmt(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "| Task | Model | H | n | Skill | Change | Moved change | Proprio | Params | FLOPs | Step ms | Backend | Seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        step = "n/a" if row["train_step_time_ms_mean"] is None else f"{row['train_step_time_ms_mean']:.2f}"
        lines.append(
            f"| {row['task']} | {row['model']} | {row['horizon']} | {row['n']} | "
            f"{fmt(row['skill_mean'])} | {fmt(row['change_skill_mean'])} | "
            f"{fmt(row['moved_region_change_skill_mean'])} | {fmt(row['proprio_skill_mean'])} | "
            f"{row['param_count_mean']} | {row['flops_per_forward_mean']} | {step} | "
            f"{row['mamba_backend']} | {row['seeds']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_baseline_freeze_dry_run(output: Path) -> None:
    lines = [
        "# D1 Baseline Freeze Dry Run",
        "",
        "This is a dry-run scaffold only; it is not a frozen baseline artifact.",
        "",
        "Required final fields:",
        "",
        "- Stage-2c checkpoint list and config paths.",
        "- Reproduced Stage-2c skill/change/proprio values.",
        "- Per-task moved-region mask threshold and denominator floor.",
        "- Per-task baseline moved-region mean/std.",
        "- Frozen floor_t, overall effect threshold, and harm threshold.",
        "- Task eligibility for reusable-trend cross-position tests.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", default="experiments/results/mgvt_d/d1_20260602/skill_scores")
    parser.add_argument("--output-dir", default="experiments/results/mgvt_d/d1_20260602/summary")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--baseline-freeze-dry-run", action="store_true")
    args = parser.parse_args()

    skill_root = Path(args.skill_root)
    output_dir = Path(args.output_dir)

    if args.baseline_freeze_dry_run and not args.check_only:
        _write_baseline_freeze_dry_run(output_dir / "D1_BASELINE_FREEZE_20260602_DRY_RUN.md")
    elif args.baseline_freeze_dry_run:
        print("Baseline freeze dry-run scaffold validated.")

    rows = _read_scores(skill_root) if skill_root.exists() else []
    if not rows:
        if args.check_only:
            print("No D1 skill scores found; check-only mode passed.")
            return
        raise SystemExit(f"No skill_score.json files found under {skill_root}")

    summary_rows = _group_summary(rows)
    _write_csv(rows, output_dir / "mgvt_d1_horizon_raw.csv")
    _write_csv(summary_rows, output_dir / "mgvt_d1_horizon_summary.csv")
    _write_md(summary_rows, output_dir / "mgvt_d1_horizon_summary.md")
    print(f"wrote D1 summary for {len(rows)} horizon rows to {output_dir}")


if __name__ == "__main__":
    main()
