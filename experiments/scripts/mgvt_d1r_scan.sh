#!/bin/bash
# Run MGVT-D STAGE-D1R PushT seed234 oracle-sandwich scaffold.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
RUN_ROOT="${RUN_ROOT:-$JEPAWM_LOGS/mgvt_d1r_oracle_sandwich_pusht_seed234_20260604}"
CKPT_ROOT="${CKPT_ROOT:-$JEPAWM_CKPT/mgvt_d1r_oracle_sandwich_pusht_seed234_20260604}"
SKILL_DEVICE="${SKILL_DEVICE:-cuda:0}"
SKILL_MAX_HORIZON="${SKILL_MAX_HORIZON:-4}"
QUICK_DEBUG="${QUICK_DEBUG:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
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
  if [ ! -e "$JEPAWM_DSET_LANCE/PushT.lance" ]; then
    echo "ERROR: required Lance store missing: $JEPAWM_DSET_LANCE/PushT.lance" >&2
    exit 2
  fi
fi

python experiments/scripts/gen_stage_d1r_configs.py

mkdir -p "$RUN_ROOT" "$CKPT_ROOT" "$RUN_ROOT/skill_scores" "$RUN_ROOT/probes" "$RUN_ROOT/summary"

{
  echo "MGVT-D STAGE-D1R scan starting $(date -Iseconds)"
  echo "commit=$(git rev-parse HEAD)"
  echo "describe=$(git describe --dirty --always)"
  echo "devices=$DEVICES"
  echo "run_root=$RUN_ROOT"
  echo "ckpt_root=$CKPT_ROOT"
  echo "variant_filter=$VARIANT_FILTER"
  echo "quick_debug=$QUICK_DEBUG"
  echo "skill_max_horizon=$SKILL_MAX_HORIZON"
} | tee "$RUN_ROOT/scan_start.txt"

make_config() {
  local base_config="$1"
  local config_path="$2"
  local run_dir="$3"
  local ckpt_dir="$4"
  local seed="$5"
  local pretrained="$6"

  python - "$base_config" "$config_path" "$run_dir" "$ckpt_dir" "$seed" "$QUICK_DEBUG" "$WORLD_SIZE" "$pretrained" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, run_dir, ckpt_dir, seed, quick_debug, world_size, pretrained = sys.argv[1:9]
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
if pretrained:
    cfg["meta"]["pretrained_path"] = pretrained
    cfg["meta"]["load_checkpoint"] = True
    cfg["meta"]["load_opt_scale_epoch"] = False
    cfg["meta"]["reset_epoch_on_pretrained_load"] = True

if quick_debug == "1":
    opt = cfg["optimization"]["transition_model"]
    opt["num_epochs"] = 2
    opt["iterations_per_epoch"] = 20
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
    text = launch_log.read_text(errors="replace").lower()
    for marker in ("traceback", "runtimeerror", "out of memory", "cuda error", "nan"):
        if marker in text:
            errors.append(f"launch.log contains {marker!r}")
if errors:
    print("Training health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("Training health check passed")
PY
}

config_for() {
  local variant="$1"
  local stage="$2"
  echo "configs/vjepa_wm/mgvt_d1r/pusht_d1r_${variant}_${stage}_lance_h4.yaml"
}

run_stage() {
  local variant="$1"
  local stage="$2"
  local pretrained="$3"
  local base_config
  base_config="$(config_for "$variant" "$stage")"
  if [ ! -f "$base_config" ]; then
    echo "ERROR: missing generated config: $base_config" >&2
    exit 2
  fi

  local run_id="20260604_mgvt_d1r_pusht_${variant}_${stage}_seed234"
  local run_dir="$RUN_ROOT/$run_id"
  local ckpt_dir="$CKPT_ROOT/$run_id"
  local generated_config="$run_dir/config.yaml"
  if [ -e "$run_dir" ] || [ -e "$ckpt_dir" ]; then
    if [ "$FORCE_RERUN" = "1" ]; then
      rm -rf "$run_dir" "$ckpt_dir"
    else
      echo "ERROR: run already exists: $run_id" >&2
      exit 2
    fi
  fi

  mkdir -p "$run_dir" "$ckpt_dir"
  make_config "$base_config" "$generated_config" "$run_dir" "$ckpt_dir" 234 "$pretrained"
  cp "$0" "$run_dir/launcher.sh"
  git rev-parse HEAD > "$run_dir/git_commit.txt"
  git describe --dirty --always >> "$run_dir/git_commit.txt"
  git status --short --untracked-files=all > "$run_dir/git_status_start.txt"

  echo "" | tee -a "$RUN_ROOT/scan.log"
  echo "=== RUN $run_id $(date -Iseconds) ===" | tee -a "$RUN_ROOT/scan.log"
  echo "config=$generated_config" | tee -a "$RUN_ROOT/scan.log"

  set +e
  python -m app.main --fname "$generated_config" --devices "${DEVICE_ARRAY[@]}" 2>&1 | tee "$run_dir/launch.log"
  cmd_status=${PIPESTATUS[0]}
  set -e
  health_check "$run_dir" "$ckpt_dir" "$cmd_status"
  LAST_CKPT="$ckpt_dir/jepa-latest.pth.tar"
}

score_final() {
  local variant="$1"
  local stage="$2"
  local run_id="20260604_mgvt_d1r_pusht_${variant}_${stage}_seed234"
  local run_dir="$RUN_ROOT/$run_id"
  local ckpt_dir="$CKPT_ROOT/$run_id"
  local skill_dir="$RUN_ROOT/skill_scores/$run_id"
  local probe_dir="$RUN_ROOT/probes/$run_id"
  mkdir -p "$skill_dir" "$probe_dir"
  python -m app.vjepa_wm.diagnostics.skill_score \
    --config "$run_dir/config.yaml" \
    --checkpoint "$ckpt_dir/jepa-latest.pth.tar" \
    --device "$SKILL_DEVICE" \
    --num-workers 0 \
    --max-horizon "$SKILL_MAX_HORIZON" \
    --train-log-csv "$run_dir/log_r0.csv" \
    --output "$skill_dir"
  python -m app.vjepa_wm.diagnostics.mgvt_d1_probes \
    --skill-json "$skill_dir/skill_score.json" \
    --output "$probe_dir"
}

run_single() {
  local variant="$1"
  local stage="$2"
  if [ -n "$VARIANT_FILTER" ] && [[ "$variant" != *"$VARIANT_FILTER"* ]]; then
    return 0
  fi
  run_stage "$variant" "$stage" ""
  score_final "$variant" "$stage"
}

run_staged() {
  local variant="$1"
  if [ -n "$VARIANT_FILTER" ] && [[ "$variant" != *"$VARIANT_FILTER"* ]]; then
    return 0
  fi
  local r1_ckpt r2_ckpt
  run_stage "$variant" "r1_teacher" ""
  r1_ckpt="$LAST_CKPT"
  run_stage "$variant" "r2_student" "$r1_ckpt"
  r2_ckpt="$LAST_CKPT"
  run_stage "$variant" "g_refine" "$r2_ckpt"
  score_final "$variant" "g_refine"
}

run_single "c1_adaln_param_match" "train"
run_single "c2_raw_action_guidance" "train"
run_single "t0_oracle_rh" "oracle"
run_single "s0_implicit_conditioning" "implicit"
run_staged "s1_trend_match"
run_staged "s2_trend_inv"

python experiments/scripts/summarize_mgvt_d1r.py \
  --skill-root "$RUN_ROOT/skill_scores" \
  --output-dir "$RUN_ROOT/summary"

echo "MGVT-D STAGE-D1R scan finished $(date -Iseconds)" | tee "$RUN_ROOT/scan_end.txt"
