from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Iterable


TIMING_COLUMNS = [
    "run_id",
    "env",
    "backend",
    "repeat",
    "status",
    "base_config",
    "num_workers",
    "batch_size",
    "world_size",
    "profile_warmup",
    "profile_steps",
    "iterations_per_epoch",
    "dataset_size",
    "rows_used",
    "iter_mean_ms",
    "iter_median_ms",
    "iter_p95_ms",
    "gpu_mean_ms",
    "gpu_median_ms",
    "data_fetch_ms",
    "data_to_device_ms",
    "profiler_step_wall_ms",
    "epoch_estimate_min",
    "fifty_epoch_estimate_h",
    "run_dir",
]


def _read_key_values(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and re.fullmatch(r"[A-Z0-9_]+", key):
            values[key] = value.strip()
    return values


def _parse_env_backend_repeat(run_id: str, launch_values: dict[str, str]) -> tuple[str, str, str]:
    lowered = run_id.lower()
    env = "unknown"
    for candidate in ("pusht", "wall", "maze", "mw"):
        if f"_{candidate}_" in f"_{lowered}_":
            env = candidate
            break

    backend = launch_values.get("BACKEND_KIND", "unset").lower()
    if backend in {"", "unset"}:
        backend = "swm_lance" if "lance" in lowered else "raw"
    if backend == "lance":
        backend = "swm_lance"

    repeat = "unknown"
    match = re.search(r"(?:^|_)r(\d+)(?:_|$)", lowered)
    if match:
        repeat = f"r{match.group(1)}"
    return env, backend, repeat


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def _parse_profile_sections(text: str) -> dict[str, float]:
    sections: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_ ]*?)\s+(-?\d+(?:\.\d+)?)", line)
        if not match:
            continue
        name = match.group(1).strip().replace(" ", "_")
        if name in {"section"}:
            continue
        sections[f"{name}_ms"] = float(match.group(2))
    return sections


def _parse_loader_size(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"Iterations per epoch:\s*(\d+)\s*\(dataset size:\s*(\d+)\)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _load_timing_rows(csv_path: Path, warmup: int, measured_steps: int) -> tuple[list[float], list[float]]:
    with csv_path.open(newline="") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return [], []

    header = raw_rows[0]
    rows = raw_rows[1:]

    def _cell(row: list[str], idx: int) -> float | None:
        if idx < 0 or idx >= len(row) or row[idx] == "":
            return None
        try:
            return float(row[idx])
        except ValueError:
            return None

    if "iter-time(ms)" in header and "gpu-time(ms)" in header:
        iter_idx = header.index("iter-time(ms)")
        gpu_idx = header.index("gpu-time(ms)")
    else:
        # train.py always writes [epoch, itr, loss, gpu_ms, iter_ms, ...].
        # Some logger headers omit the fixed timing columns, so fall back to
        # positions 3/4 when named columns are unavailable.
        gpu_idx = 3
        iter_idx = 4

    rows = rows[warmup : warmup + measured_steps]
    iter_times = [value for row in rows if (value := _cell(row, iter_idx)) is not None]
    gpu_times = [value for row in rows if (value := _cell(row, gpu_idx)) is not None]
    return iter_times, gpu_times


def summarize_run(run_dir: str | Path, warmup: int = 30, measured_steps: int = 300) -> dict:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    launch_log = run_dir / "launch.log"
    train_csv = run_dir / "log_r0.csv"

    text = launch_log.read_text(errors="replace") if launch_log.exists() else ""
    values = _read_key_values(text)
    run_id = values.get("RUN_ID", run_id)
    env, backend, repeat = _parse_env_backend_repeat(run_id, values)
    profile_warmup = int(values.get("PROFILE_WARMUP", warmup))
    profile_steps = int(values.get("PROFILE_STEPS", measured_steps))
    iterations_per_epoch, dataset_size = _parse_loader_size(text)
    sections = _parse_profile_sections(text)

    status = "ok"
    iter_times: list[float] = []
    gpu_times: list[float] = []
    if not train_csv.exists():
        status = "missing_log_r0.csv"
    else:
        iter_times, gpu_times = _load_timing_rows(train_csv, warmup=warmup, measured_steps=measured_steps)
        if len(iter_times) < measured_steps:
            status = f"short_rows:{len(iter_times)}"

    iter_median = _median(iter_times)
    epoch_ref = dataset_size if dataset_size is not None else iterations_per_epoch
    epoch_estimate_min = (
        (epoch_ref * iter_median / 1000.0 / 60.0)
        if epoch_ref is not None and not math.isnan(iter_median)
        else math.nan
    )

    return {
        "run_id": run_id,
        "env": env,
        "backend": backend,
        "repeat": repeat,
        "status": status,
        "base_config": values.get("BASE_CONFIG", ""),
        "num_workers": values.get("NUM_WORKERS", ""),
        "batch_size": values.get("BATCH_SIZE", ""),
        "world_size": _parse_world_size(values.get("DEVICES", "")),
        "profile_warmup": profile_warmup,
        "profile_steps": profile_steps,
        "iterations_per_epoch": iterations_per_epoch or "",
        "dataset_size": dataset_size or "",
        "rows_used": len(iter_times),
        "iter_mean_ms": _mean(iter_times),
        "iter_median_ms": iter_median,
        "iter_p95_ms": _percentile(iter_times, 95.0),
        "gpu_mean_ms": _mean(gpu_times),
        "gpu_median_ms": _median(gpu_times),
        "data_fetch_ms": sections.get("data_fetch_ms", math.nan),
        "data_to_device_ms": sections.get("data_to_device_ms", math.nan),
        "profiler_step_wall_ms": sections.get("measured_step_wall_ms", math.nan),
        "epoch_estimate_min": epoch_estimate_min,
        "fifty_epoch_estimate_h": epoch_estimate_min * 50.0 / 60.0
        if not math.isnan(epoch_estimate_min)
        else math.nan,
        "run_dir": str(run_dir),
    }


def _parse_world_size(devices_value: str) -> str:
    match = re.search(r"world_size=(\d+)", devices_value)
    return match.group(1) if match else ""


def _format_float(value) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def write_csv(rows: list[dict], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in TIMING_COLUMNS})


def _group_rows(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["env"], row["backend"]), []).append(row)
    return groups


def write_markdown(rows: list[dict], output_md: str | Path) -> None:
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Time Profile 20260528",
        "",
        "Each run drops the configured warmup rows and summarizes the measured rows from `log_r0.csv`.",
        "",
        "| env | backend | runs | median iter ms | CV % | data fetch ms | epoch min | 50 epoch h | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for (env, backend), group in sorted(_group_rows(rows).items()):
        medians = [row["iter_median_ms"] for row in group if not math.isnan(row["iter_median_ms"])]
        median_of_medians = _median(medians)
        cv = (
            statistics.stdev(medians) / statistics.fmean(medians) * 100.0
            if len(medians) > 1 and statistics.fmean(medians) != 0
            else 0.0
        )
        fetches = [row["data_fetch_ms"] for row in group if not math.isnan(row["data_fetch_ms"])]
        epoch_mins = [row["epoch_estimate_min"] for row in group if not math.isnan(row["epoch_estimate_min"])]
        fifty_hours = [row["fifty_epoch_estimate_h"] for row in group if not math.isnan(row["fifty_epoch_estimate_h"])]
        status = "stable" if cv <= 5.0 and len(medians) >= 3 else "needs_review"
        if any(row["status"] != "ok" for row in group):
            status = ",".join(sorted({str(row["status"]) for row in group}))
        lines.append(
            "| {env} | {backend} | {runs} | {median} | {cv} | {fetch} | {epoch} | {fifty} | {status} |".format(
                env=env,
                backend=backend,
                runs=len(group),
                median=_format_float(median_of_medians),
                cv=_format_float(cv),
                fetch=_format_float(_mean(fetches)),
                epoch=_format_float(_mean(epoch_mins)),
                fifty=_format_float(_mean(fifty_hours)),
                status=status,
            )
        )
    output_md.write_text("\n".join(lines) + "\n")


def collect_runs(logs_root: str | Path, pattern: str, warmup: int, measured_steps: int) -> list[dict]:
    root = Path(logs_root)
    run_dirs = sorted(path for path in root.glob(pattern) if path.is_dir())
    return [summarize_run(path, warmup=warmup, measured_steps=measured_steps) for path in run_dirs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize JEPA-WM Phase-1 training timing profile runs.")
    parser.add_argument("--logs-root", required=True, help="Root containing timing run directories, usually $JEPAWM_LOGS.")
    parser.add_argument("--pattern", default="20260528_dino_wm_timing_*", help="Glob under --logs-root.")
    parser.add_argument("--output-csv", default="experiments/results/training_time_profile_20260528.csv")
    parser.add_argument("--output-md", default="experiments/results/training_time_profile_20260528.md")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measured-steps", type=int, default=300)
    args = parser.parse_args()

    rows = collect_runs(args.logs_root, args.pattern, args.warmup, args.measured_steps)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"wrote {args.output_csv} and {args.output_md} from {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
