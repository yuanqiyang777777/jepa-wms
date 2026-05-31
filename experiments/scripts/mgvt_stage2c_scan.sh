#!/bin/bash
# Run the MGVT-JEPA Stage-2c AdaLN H=4 three-seed confirmation on lab01.
#
# Default full scan: AdaLN(depth=1) x 5 Phase-1 tasks x 3 seeds.
# Smoke example:
#   QUICK_DEBUG=1 CONFIG_FILTER=wall_stage2c_pred_adaln_depth1_lance_h4 \
#   WALL_SEEDS=234 PUSHT_SEEDS="" MAZE_SEEDS="" MW_R_SEEDS="" MW_RW_SEEDS="" \
#   RUN_ROOT="$JEPAWM_LOGS/mgvt_stage2c_smoke" \
#   CKPT_ROOT="$JEPAWM_CKPT/mgvt_stage2c_smoke" bash experiments/scripts/mgvt_stage2c_scan.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/_common.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
RUN_ROOT="${RUN_ROOT:-$JEPAWM_LOGS/mgvt_stage2c_3seed}"
CKPT_ROOT="${CKPT_ROOT:-$JEPAWM_CKPT/mgvt_stage2c_3seed}"
SKILL_DEVICE="${SKILL_DEVICE:-cuda:0}"
SKILL_MAX_HORIZON="${SKILL_MAX_HORIZON:-6}"
PUSHT_SEEDS="${PUSHT_SEEDS:-234 235 236}"
WALL_SEEDS="${WALL_SEEDS:-234 235 236}"
MAZE_SEEDS="${MAZE_SEEDS:-234 235 236}"
MW_R_SEEDS="${MW_R_SEEDS:-1 2 3}"
MW_RW_SEEDS="${MW_RW_SEEDS:-1 2 3}"
TASK_FILTER="${TASK_FILTER:-}"
CONFIG_FILTER="${CONFIG_FILTER:-}"
QUICK_DEBUG="${QUICK_DEBUG:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REQUIRE_LANCE_STORES="${REQUIRE_LANCE_STORES:-1}"

DEVICES="${DEVICES//,/ }"
read -r -a DEVICE_ARRAY <<< "$DEVICES"
WORLD_SIZE="${#DEVICE_ARRAY[@]}"
if [ "$WORLD_SIZE" -lt 1 ]; then
  echo "ERROR: DEVICES resolved to an empty device list" >&2
  exit 2
fi
if [ "$WORLD_SIZE" -gt 6 ]; then
  echo "ERROR: refusing to use $WORLD_SIZE devices; max is 6" >&2
  exit 2
fi

check_gpus_free "${DEVICE_ARRAY[@]}"

if [ "$REQUIRE_LANCE_STORES" = "1" ]; then
  : "${JEPAWM_DSET_LANCE:?JEPAWM_DSET_LANCE must point to the Lance store root}"
  for uri in \
    "$JEPAWM_DSET_LANCE/PushT.lance" \
    "$JEPAWM_DSET_LANCE/Wall.lance" \
    "$JEPAWM_DSET_LANCE/PointMaze.lance" \
    "$JEPAWM_DSET_LANCE/Metaworld.lance"; do
    if [ ! -e "$uri" ]; then
      echo "ERROR: required Lance store missing: $uri" >&2
      echo "Build the store first; Stage-2c never falls back to raw data." >&2
      exit 2
    fi
  done
fi

CONFIGS=(
  "configs/vjepa_wm/mgvt_stage2c/pusht_stage2c_pred_adaln_depth1_lance_h4.yaml"
  "configs/vjepa_wm/mgvt_stage2c/wall_stage2c_pred_adaln_depth1_lance_h4.yaml"
  "configs/vjepa_wm/mgvt_stage2c/maze_stage2c_pred_adaln_depth1_lance_h4.yaml"
  "configs/vjepa_wm/mgvt_stage2c/mw_r_stage2c_pred_adaln_depth1_lance_h4.yaml"
  "configs/vjepa_wm/mgvt_stage2c/mw_rw_stage2c_pred_adaln_depth1_lance_h4.yaml"
)

mkdir -p "$RUN_ROOT" "$CKPT_ROOT" "$RUN_ROOT/skill_scores" "$RUN_ROOT/summary"

{
  echo "MGVT Stage-2c scan starting $(date -Iseconds)"
  echo "commit=$(git rev-parse HEAD)"
  echo "devices=$DEVICES"
  echo "run_root=$RUN_ROOT"
  echo "ckpt_root=$CKPT_ROOT"
  echo "pusht_seeds=$PUSHT_SEEDS"
  echo "wall_seeds=$WALL_SEEDS"
  echo "maze_seeds=$MAZE_SEEDS"
  echo "mw_r_seeds=$MW_R_SEEDS"
  echo "mw_rw_seeds=$MW_RW_SEEDS"
  echo "task_filter=$TASK_FILTER"
  echo "config_filter=$CONFIG_FILTER"
  echo "quick_debug=$QUICK_DEBUG"
  echo "skill_max_horizon=$SKILL_MAX_HORIZON"
} | tee "$RUN_ROOT/scan_start.txt"

make_config() {
  local base_config="$1"
  local config_path="$2"
  local run_dir="$3"
  local ckpt_dir="$4"
  local seed="$5"

  python - "$base_config" "$config_path" "$run_dir" "$ckpt_dir" "$seed" "$QUICK_DEBUG" "$WORLD_SIZE" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, run_dir, ckpt_dir, seed, quick_debug, world_size = sys.argv[1:8]
cfg = load_yaml(base_config)
seed = int(seed)

cfg["folder"] = run_dir
cfg["checkpoint_folder"] = ckpt_dir
cfg["tasks_per_node"] = int(world_size)
cfg.setdefault("meta", {})["seed"] = seed
cfg.setdefault("data", {})["seed"] = seed
cfg.setdefault("logging", {}).setdefault("wandb", {})["use_wandb"] = False

if quick_debug == "1":
    opt = cfg["optimization"]["transition_model"]
    opt["num_epochs"] = 2
    opt["iterations_per_epoch"] = 100
    cfg["meta"]["quick_debug"] = True

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
dump_yaml(cfg, config_path)
PY
}

health_check() {
  local run_dir="$1"
  local ckpt_dir="$2"
  local cmd_status="$3"

  python - "$run_dir" "$ckpt_dir" "$cmd_status" <<'PY'
import csv
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])
cmd_status = int(sys.argv[3])
errors = []

launch_log = run_dir / "launch.log"
latest_ckpt = ckpt_dir / "jepa-latest.pth.tar"
train_csv = run_dir / "log_r0.csv"

if cmd_status != 0:
    errors.append(f"app.main command exited non-zero: {cmd_status}")
if not latest_ckpt.exists():
    errors.append(f"missing checkpoint: {latest_ckpt}")
if not train_csv.exists():
    errors.append(f"missing train csv: {train_csv}")
else:
    with train_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        errors.append(f"train csv has no data rows: {train_csv}")
    else:
        losses = []
        for row in rows:
            raw = row.get("loss")
            if raw not in ("", None):
                try:
                    losses.append(float(raw))
                except ValueError:
                    errors.append(f"non-numeric loss value: {raw!r}")
        if not losses:
            errors.append("no loss values found in log_r0.csv")
        elif not all(math.isfinite(v) for v in losses):
            errors.append("loss contains non-finite values")

if not launch_log.exists():
    errors.append(f"missing launch log: {launch_log}")
else:
    text = launch_log.read_text(errors="replace").lower()
    for marker in ("out of memory", "cuda error"):
        if marker in text:
            errors.append(f"launch.log contains {marker!r}")
    known_loader_shutdown = (
        "exception ignored in: <function _multiprocessingdataloaderiter.__del__" in text
        or "dataloader worker" in text and "is killed by signal: aborted" in text
    )
    if "traceback" in text and (cmd_status != 0 or not known_loader_shutdown):
        errors.append("launch.log contains an unexpected traceback")

if errors:
    print("Training health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("Training health check passed")
PY
}

seed_list_for_task() {
  local task="$1"
  case "$task" in
    pusht) echo "$PUSHT_SEEDS" ;;
    wall) echo "$WALL_SEEDS" ;;
    maze) echo "$MAZE_SEEDS" ;;
    mw_r) echo "$MW_R_SEEDS" ;;
    mw_rw) echo "$MW_RW_SEEDS" ;;
    *) echo "ERROR: unknown task token: $task" >&2; return 2 ;;
  esac
}

run_one() {
  local base_config="$1"
  local seed="$2"
  local stem task
  stem="$(basename "$base_config" .yaml)"
  task="${stem%%_stage2c_*}"
  local run_id="${stem}_seed${seed}"
  local run_dir="$RUN_ROOT/$run_id"
  local ckpt_dir="$CKPT_ROOT/$run_id"
  local generated_config="$run_dir/config.yaml"
  local skill_dir="$RUN_ROOT/skill_scores/$run_id"
  local skill_json="$skill_dir/skill_score.json"

  if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
    return 0
  fi

  if [ -n "$CONFIG_FILTER" ] && [[ "$stem" != *"$CONFIG_FILTER"* ]]; then
    return 0
  fi

  if [ -f "$skill_json" ] && [ "$FORCE_RERUN" != "1" ]; then
    echo "SKIP completed run: $run_id" | tee -a "$RUN_ROOT/scan.log"
    return 0
  fi

  if [ -e "$run_dir" ] || [ -e "$ckpt_dir" ]; then
    if [ "$FORCE_RERUN" = "1" ] || [ ! -f "$skill_json" ]; then
      rm -rf "$run_dir" "$ckpt_dir" "$skill_dir"
    else
      echo "ERROR: run already exists: $run_id (set FORCE_RERUN=1 to overwrite)" >&2
      exit 2
    fi
  fi

  mkdir -p "$run_dir" "$ckpt_dir"
  make_config "$base_config" "$generated_config" "$run_dir" "$ckpt_dir" "$seed"
  cp "$0" "$run_dir/launcher.sh"
  git rev-parse HEAD > "$run_dir/git_commit.txt"
  git describe --dirty --always >> "$run_dir/git_commit.txt"

  echo "" | tee -a "$RUN_ROOT/scan.log"
  echo "=== RUN $run_id $(date -Iseconds) ===" | tee -a "$RUN_ROOT/scan.log"
  echo "config=$generated_config" | tee -a "$RUN_ROOT/scan.log"

  set +e
  python -m app.main --fname "$generated_config" --devices "${DEVICE_ARRAY[@]}" 2>&1 | tee "$run_dir/launch.log"
  cmd_status=${PIPESTATUS[0]}
  set -e

  health_check "$run_dir" "$ckpt_dir" "$cmd_status"

  python -m app.vjepa_wm.diagnostics.skill_score \
    --config "$generated_config" \
    --checkpoint "$ckpt_dir/jepa-latest.pth.tar" \
    --device "$SKILL_DEVICE" \
    --num-workers 0 \
    --max-horizon "$SKILL_MAX_HORIZON" \
    --train-log-csv "$run_dir/log_r0.csv" \
    --output "$RUN_ROOT/skill_scores/$run_id"
}

for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .yaml)"
  task="${stem%%_stage2c_*}"
  seeds="$(seed_list_for_task "$task")"
  for seed in $seeds; do
    run_one "$config" "$seed"
  done
done

python - "$RUN_ROOT/skill_scores" "$RUN_ROOT/summary" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

skill_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
summary_dir.mkdir(parents=True, exist_ok=True)

rows = []
for path in sorted(skill_root.glob("*/skill_score.json")):
    data = json.loads(path.read_text())
    horizons = sorted(data["skill_by_horizon"], key=lambda h: int(h))
    for horizon in horizons:
        rows.append({
            "run_id": path.parent.name,
            "task": data["task"],
            "model": data["model"],
            "seed": data["seed"],
            "horizon": int(horizon),
            "skill": data["skill_by_horizon"][horizon],
            "change_skill": data["change_skill_by_horizon"][horizon],
            "proprio_skill": data["proprio_skill_by_horizon"][horizon],
            "mse_model": data["mse_model_by_horizon"][horizon],
            "mse_persist": data["mse_persist_by_horizon"][horizon],
            "proprio_mse_model": data["proprio_mse_model_by_horizon"][horizon],
            "proprio_mse_persist": data["proprio_mse_persist_by_horizon"][horizon],
            "param_count": data["param_count"],
            "train_step_time_ms": data["train_step_time_ms"],
            "mamba_backend": data.get("mamba_backend") or "",
            "deterministic_encode_max_abs_diff": data["deterministic_encode_max_abs_diff"],
        })

if not rows:
    raise SystemExit("No skill_score.json files found")

raw_csv = summary_dir / "mgvt_stage2c_horizon_raw.csv"
with raw_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["task"], row["model"], row["horizon"])].append(row)

def finite_values(group, key):
    return [r[key] for r in group if r[key] is not None]

def mean(values):
    return sum(values) / len(values) if values else None

def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0

def margin(mean_value, std_value):
    return None if mean_value is None or std_value is None else mean_value - std_value

summary_rows = []
for (task, model, horizon), group in sorted(groups.items()):
    skills = finite_values(group, "skill")
    change = finite_values(group, "change_skill")
    proprio = finite_values(group, "proprio_skill")
    steps = finite_values(group, "train_step_time_ms")
    skill_mean = mean(skills)
    skill_std = stdev(skills)
    change_mean = mean(change)
    change_std = stdev(change)
    proprio_mean = mean(proprio)
    proprio_std = stdev(proprio)
    summary_rows.append({
        "task": task,
        "model": model,
        "horizon": horizon,
        "n": len(group),
        "skill_mean": skill_mean,
        "skill_std": skill_std,
        "skill_mean_minus_std": margin(skill_mean, skill_std),
        "change_skill_mean": change_mean,
        "change_skill_std": change_std,
        "change_skill_mean_minus_std": margin(change_mean, change_std),
        "proprio_skill_mean": proprio_mean,
        "proprio_skill_std": proprio_std,
        "proprio_skill_mean_minus_std": margin(proprio_mean, proprio_std),
        "param_count_mean": round(mean([r["param_count"] for r in group])),
        "train_step_time_ms_mean": mean(steps),
        "mamba_backend": ",".join(sorted({r["mamba_backend"] for r in group if r["mamba_backend"]})),
        "seeds": " ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"])),
    })

summary_csv = summary_dir / "mgvt_stage2c_horizon_summary.csv"
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

md_lines = [
    "| Task | Model | H | n | Skill mean | Skill mean-std | Change mean | Change mean-std | Proprio mean | Proprio mean-std | Params | Step ms | Mamba backend | Seeds |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
for row in summary_rows:
    def fmt(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"
    step = "n/a" if row["train_step_time_ms_mean"] is None else f"{row['train_step_time_ms_mean']:.2f}"
    md_lines.append(
        f"| {row['task']} | {row['model']} | {row['horizon']} | {row['n']} | "
        f"{fmt(row['skill_mean'])} | {fmt(row['skill_mean_minus_std'])} | "
        f"{fmt(row['change_skill_mean'])} | {fmt(row['change_skill_mean_minus_std'])} | "
        f"{fmt(row['proprio_skill_mean'])} | {fmt(row['proprio_skill_mean_minus_std'])} | "
        f"{row['param_count_mean']} | {step} | {row['mamba_backend']} | {row['seeds']} |"
    )
(summary_dir / "mgvt_stage2c_horizon_summary.md").write_text("\n".join(md_lines) + "\n")
print("\n".join(md_lines))
PY

echo "MGVT Stage-2c scan finished $(date -Iseconds)" | tee "$RUN_ROOT/scan_end.txt"
