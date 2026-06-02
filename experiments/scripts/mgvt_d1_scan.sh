#!/bin/bash
# Run MGVT-D Stage-D1 prediction-only smoke/full scans on lab01.
#
# Smoke example:
#   QUICK_DEBUG=1 TASK_FILTER=wall VARIANT_FILTER=mamba_trend_dim16 \
#   WALL_SEEDS=234 PUSHT_SEEDS="" MAZE_SEEDS="" MW_R_SEEDS="" MW_RW_SEEDS="" \
#   RUN_ROOT="$JEPAWM_LOGS/mgvt_d1_smoke_20260602" \
#   CKPT_ROOT="$JEPAWM_CKPT/mgvt_d1_smoke_20260602" bash experiments/scripts/mgvt_d1_scan.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/_common.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
RUN_ROOT="${RUN_ROOT:-$JEPAWM_LOGS/mgvt_d1_20260602}"
CKPT_ROOT="${CKPT_ROOT:-$JEPAWM_CKPT/mgvt_d1_20260602}"
SKILL_DEVICE="${SKILL_DEVICE:-cuda:0}"
SKILL_MAX_HORIZON="${SKILL_MAX_HORIZON:-6}"
PUSHT_SEEDS="${PUSHT_SEEDS:-234 235 236}"
WALL_SEEDS="${WALL_SEEDS:-234 235 236}"
MAZE_SEEDS="${MAZE_SEEDS:-234 235 236}"
MW_R_SEEDS="${MW_R_SEEDS:-1 2 3}"
MW_RW_SEEDS="${MW_RW_SEEDS:-1 2 3}"
TASK_FILTER="${TASK_FILTER:-}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
BLOCK_FILTER="${BLOCK_FILTER:-}"
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
      exit 2
    fi
  done
fi

if [ ! -d configs/vjepa_wm/mgvt_d1 ]; then
  python experiments/scripts/gen_stage_d1_configs.py
fi

mapfile -t CONFIGS < <(find configs/vjepa_wm/mgvt_d1 -maxdepth 1 -name '*.yaml' | sort)
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "ERROR: no D1 configs found; run gen_stage_d1_configs.py" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$CKPT_ROOT" "$RUN_ROOT/skill_scores" "$RUN_ROOT/probes" "$RUN_ROOT/summary"

{
  echo "MGVT-D Stage-D1 scan starting $(date -Iseconds)"
  echo "commit=$(git rev-parse HEAD)"
  echo "describe=$(git describe --dirty --always)"
  echo "devices=$DEVICES"
  echo "run_root=$RUN_ROOT"
  echo "ckpt_root=$CKPT_ROOT"
  echo "task_filter=$TASK_FILTER"
  echo "variant_filter=$VARIANT_FILTER"
  echo "block_filter=$BLOCK_FILTER"
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
cfg["evals"] = None
cfg["unroll_decode_evals"] = None

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

from src.utils.yaml_utils import load_yaml

run_dir = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])
cmd_status = int(sys.argv[3])
errors = []

latest_ckpt = ckpt_dir / "jepa-latest.pth.tar"
train_csv = run_dir / "log_r0.csv"
launch_log = run_dir / "launch.log"

if cmd_status != 0:
    errors.append(f"app.main command exited non-zero: {cmd_status}")
if not latest_ckpt.exists():
    errors.append(f"missing checkpoint: {latest_ckpt}")
if not train_csv.exists():
    errors.append(f"missing train csv: {train_csv}")
else:
    rows = list(csv.DictReader(train_csv.open(newline="")))
    losses = [float(row["loss"]) for row in rows if row.get("loss")]
    if not losses or not all(math.isfinite(v) for v in losses):
        errors.append("train csv has no finite loss values")
if not launch_log.exists():
    errors.append(f"missing launch log: {launch_log}")
else:
    raw = launch_log.read_text(errors="replace")
    low_lines = raw.lower().splitlines()
    text = raw.lower()
    for marker in ("out of memory", "cuda error"):
        if marker in text:
            errors.append(f"launch.log contains {marker!r}")
    if "traceback" in text:
        last_avg = max(
            (idx for idx, line in enumerate(low_lines) if "avg. loss" in line),
            default=-1,
        )
        final_epoch_ok = False
        try:
            cfg = load_yaml(str(run_dir / "config.yaml"))
            num_epochs = int(cfg["optimization"]["transition_model"]["num_epochs"])
            final_epoch_ok = f"epoch {num_epochs}/{num_epochs}" in text
        except Exception:
            final_epoch_ok = False

        traceback_lines = [
            idx
            for idx, line in enumerate(low_lines)
            if "traceback (most recent call last)" in line
        ]

        def is_dataloader_finalizer_traceback(idx):
            context_before = "\n".join(low_lines[max(0, idx - 3):idx])
            context_after = "\n".join(low_lines[idx:min(len(low_lines), idx + 20)])
            return (
                idx > last_avg
                and "exception ignored in:" in context_before
                and "_multiprocessingdataloaderiter.__del__" in context_before
                and "dataloader worker" in context_after
                and "aborted" in context_after
            )

        benign_teardown = (
            cmd_status == 0
            and latest_ckpt.exists()
            and final_epoch_ok
            and last_avg >= 0
            and traceback_lines
            and all(is_dataloader_finalizer_traceback(idx) for idx in traceback_lines)
        )
        if not benign_teardown:
            errors.append("launch.log contains a non-teardown 'traceback'")

if errors:
    print("Training health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("Training health check passed")
PY
}

task_from_stem() {
  local stem="$1"
  case "$stem" in
    pusht_*) echo "pusht" ;;
    wall_*) echo "wall" ;;
    maze_*) echo "maze" ;;
    mw_r_*) echo "mw_r" ;;
    mw_rw_*) echo "mw_rw" ;;
    *) echo "ERROR: cannot parse task from $stem" >&2; return 2 ;;
  esac
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
  local stem task variant block
  stem="$(basename "$base_config" .yaml)"
  task="$(task_from_stem "$stem")"
  block="$(python - "$base_config" <<'PY'
import sys
from ruamel.yaml import YAML
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = YAML(typ="safe").load(handle)
print(cfg["model"]["predictor"].get("d1_block", ""))
PY
)"
  variant="$(python - "$base_config" <<'PY'
import sys
from ruamel.yaml import YAML
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = YAML(typ="safe").load(handle)
print(cfg["model"]["predictor"].get("d1_variant", ""))
PY
)"

  if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
    return 0
  fi
  if [ -n "$VARIANT_FILTER" ] && [[ "$variant" != *"$VARIANT_FILTER"* ]]; then
    return 0
  fi
  if [ -n "$BLOCK_FILTER" ] && [ "$block" != "$BLOCK_FILTER" ]; then
    return 0
  fi

  local run_id="20260602_mgvt_${block}_${task}_${variant}_seed${seed}"
  local run_dir="$RUN_ROOT/$run_id"
  local ckpt_dir="$CKPT_ROOT/$run_id"
  local generated_config="$run_dir/config.yaml"
  local skill_dir="$RUN_ROOT/skill_scores/$run_id"
  local probe_dir="$RUN_ROOT/probes/$run_id"
  local skill_json="$skill_dir/skill_score.json"

  if [ -f "$skill_json" ] && [ "$FORCE_RERUN" != "1" ]; then
    echo "SKIP completed run: $run_id" | tee -a "$RUN_ROOT/scan.log"
    return 0
  fi
  if [ -e "$run_dir" ] || [ -e "$ckpt_dir" ]; then
    if [ "$FORCE_RERUN" = "1" ] || [ ! -f "$skill_json" ]; then
      rm -rf "$run_dir" "$ckpt_dir" "$skill_dir" "$probe_dir"
    else
      echo "ERROR: run already exists: $run_id" >&2
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
    --output "$skill_dir"

  python -m app.vjepa_wm.diagnostics.mgvt_d1_probes \
    --skill-json "$skill_json" \
    --output "$probe_dir"
}

for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .yaml)"
  task="$(task_from_stem "$stem")"
  if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
    continue
  fi
  seeds="$(seed_list_for_task "$task")"
  for seed in $seeds; do
    run_one "$config" "$seed"
  done
done

python experiments/scripts/summarize_mgvt_d1.py \
  --skill-root "$RUN_ROOT/skill_scores" \
  --output-dir "$RUN_ROOT/summary"

echo "MGVT-D Stage-D1 scan finished $(date -Iseconds)" | tee "$RUN_ROOT/scan_end.txt"
