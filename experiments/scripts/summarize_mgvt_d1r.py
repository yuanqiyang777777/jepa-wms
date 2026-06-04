#!/usr/bin/env python
"""Summarize MGVT-D STAGE-D1R smoke skill scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _variant_from_run_id(run_id: str) -> str:
    prefix = "20260604_mgvt_d1r_pusht_"
    suffix = "_seed234"
    if run_id.startswith(prefix) and run_id.endswith(suffix):
        return run_id[len(prefix) : -len(suffix)]
    return run_id


def _read_scores(skill_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(skill_root.glob("*/skill_score.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.parent.name
        horizons = sorted(data.get("skill_by_horizon", {}), key=lambda h: int(h))
        for horizon in horizons:
            rows.append(
                {
                    "run_id": run_id,
                    "variant": _variant_from_run_id(run_id),
                    "task": data.get("task"),
                    "model": data.get("model"),
                    "seed": data.get("seed"),
                    "horizon": int(horizon),
                    "skill": data.get("skill_by_horizon", {}).get(horizon),
                    "change_skill": data.get("change_skill_by_horizon", {}).get(horizon),
                    "moved_region_change_skill": data.get("moved_region_change_skill_by_horizon", {}).get(horizon),
                    "proprio_skill": data.get("proprio_skill_by_horizon", {}).get(horizon),
                    "param_count": data.get("param_count"),
                    "inference_path_param_count": data.get("inference_path_param_count"),
                    "flops_per_forward_estimate": data.get("flops_per_forward_estimate"),
                    "train_step_time_ms": data.get("train_step_time_ms"),
                }
            )
    return rows


def _h4_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["horizon"] != 4:
            continue
        out.append(row)
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
        return "n/a" if not _finite(value) else f"{float(value):.{digits}f}"

    lines = [
        "| Variant | Model | Skill@H4 | Change@H4 | Moved@H4 | Proprio@H4 | Params | Inference Params | FLOPs | Step ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['model']} | {fmt(row['skill'])} | "
            f"{fmt(row['change_skill'])} | {fmt(row['moved_region_change_skill'])} | "
            f"{fmt(row['proprio_skill'])} | {row.get('param_count') or 'n/a'} | "
            f"{row.get('inference_path_param_count') or 'n/a'} | "
            f"{row.get('flops_per_forward_estimate') or 'n/a'} | {fmt(row.get('train_step_time_ms'), 2)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    skill_root = Path(args.skill_root)
    output_dir = Path(args.output_dir)
    rows = _read_scores(skill_root) if skill_root.exists() else []
    if not rows:
        if args.check_only:
            print("No D1R skill scores found; check-only mode passed.")
            return
        raise SystemExit(f"No skill_score.json files found under {skill_root}")
    h4_rows = _h4_table(rows)
    _write_csv(rows, output_dir / "mgvt_d1r_horizon_raw.csv")
    _write_csv(h4_rows, output_dir / "mgvt_d1r_h4_table.csv")
    _write_md(h4_rows, output_dir / "mgvt_d1r_h4_table.md")
    print(f"wrote D1R summary for {len(rows)} horizon rows to {output_dir}")


if __name__ == "__main__":
    main()
