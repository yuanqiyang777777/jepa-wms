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
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-4}"
QUICK_DEBUG="${QUICK_DEBUG:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
REQUIRE_LANCE_STORES="${REQUIRE_LANCE_STORES:-1}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"
TRAIN_PERSISTENT_WORKERS="${TRAIN_PERSISTENT_WORKERS:-}"

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
  echo "skill_batch_size=$SKILL_BATCH_SIZE"
  echo "train_num_workers=${TRAIN_NUM_WORKERS:-config-default}"
  echo "train_persistent_workers=${TRAIN_PERSISTENT_WORKERS:-config-default}"
} | tee "$RUN_ROOT/scan_start.txt"

make_config() {
  local base_config="$1"
  local config_path="$2"
  local run_dir="$3"
  local ckpt_dir="$4"
  local seed="$5"
  local pretrained="$6"

  python - "$base_config" "$config_path" "$run_dir" "$ckpt_dir" "$seed" "$QUICK_DEBUG" "$WORLD_SIZE" "$pretrained" "$TRAIN_NUM_WORKERS" "$TRAIN_PERSISTENT_WORKERS" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, run_dir, ckpt_dir, seed, quick_debug, world_size, pretrained, train_num_workers, train_persistent_workers = sys.argv[1:11]
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
loader = cfg.setdefault("data", {}).setdefault("loader", {})
if train_num_workers:
    loader["num_workers"] = int(train_num_workers)
if train_persistent_workers:
    value = train_persistent_workers.strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"TRAIN_PERSISTENT_WORKERS must be boolean-like, got {train_persistent_workers!r}")
    loader["persistent_workers"] = value in {"1", "true", "yes"}
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
import json
import math
import re
import sys
from pathlib import Path

from src.utils.yaml_utils import load_yaml

run_dir = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])
cmd_status = int(sys.argv[3])
errors = []
warnings = []
latest_ckpt = ckpt_dir / "jepa-latest.pth.tar"
train_csv = run_dir / "log_r0.csv"
launch_log = run_dir / "launch.log"
config_path = run_dir / "config.yaml"
health_path = run_dir / "health_check.json"
payload = {
    "cmd_status": cmd_status,
    "latest_checkpoint": str(latest_ckpt),
    "train_csv": str(train_csv),
    "launch_log": str(launch_log),
    "config": str(config_path),
    "known_loader_shutdown_warning": False,
    "errors": errors,
    "warnings": warnings,
}

if cmd_status != 0:
    errors.append(f"app.main command exited non-zero: {cmd_status}")
if not latest_ckpt.exists():
    errors.append(f"missing checkpoint: {latest_ckpt}")
if not config_path.exists():
    errors.append(f"missing config: {config_path}")
else:
    cfg = load_yaml(str(config_path))
    opt = cfg.get("optimization", {}).get("transition_model", {})
    payload["expected_num_epochs"] = int(opt.get("num_epochs", 0) or 0)
    payload["expected_iterations_per_epoch"] = int(opt.get("iterations_per_epoch", 0) or 0)
if not train_csv.exists():
    errors.append(f"missing train csv: {train_csv}")
else:
    with train_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload["train_csv_rows"] = len(rows)
    losses = []
    last_epoch = None
    last_itr = None
    for row in rows:
        try:
            last_epoch = int(row.get("epoch", ""))
            last_itr = int(row.get("itr", ""))
        except ValueError:
            errors.append(f"non-integer epoch/itr in train csv row: {row!r}")
            break
        raw_loss = row.get("loss")
        if raw_loss not in ("", None):
            try:
                losses.append(float(raw_loss))
            except ValueError:
                errors.append(f"non-numeric loss value: {raw_loss!r}")
                break
    payload["last_epoch"] = last_epoch
    payload["last_itr"] = last_itr
    payload["loss_count"] = len(losses)
    if not rows:
        errors.append(f"train csv has no data rows: {train_csv}")
    elif not losses or not all(math.isfinite(v) for v in losses):
        errors.append("train csv has no finite loss values")
    else:
        expected_epochs = int(payload.get("expected_num_epochs", 0) or 0)
        expected_ipe = int(payload.get("expected_iterations_per_epoch", 0) or 0)
        if expected_epochs > 0 and last_epoch is not None and last_epoch < expected_epochs:
            errors.append(f"train csv incomplete: last_epoch={last_epoch}, expected={expected_epochs}")
        if expected_ipe > 0 and last_itr is not None and last_itr < expected_ipe - 1:
            errors.append(f"train csv incomplete: last_itr={last_itr}, expected_at_least={expected_ipe - 1}")
if not launch_log.exists():
    errors.append(f"missing launch log: {launch_log}")
else:
    text = launch_log.read_text(errors="replace").lower()
    for marker in ("out of memory", "cuda error"):
        if marker in text:
            errors.append(f"launch.log contains {marker!r}")
    if re.search(r"(^|[^a-z])nan([^a-z]|$)", text):
        errors.append("launch.log contains 'nan'")
    has_traceback = "traceback" in text
    has_runtimeerror = "runtimeerror" in text
    traceback_count = text.count("traceback (most recent call last):")
    loader_finalizer_count = text.count("exception ignored in: <function _multiprocessingdataloaderiter.__del__")
    extra_tracebacks = max(0, traceback_count - loader_finalizer_count)
    payload["traceback_count"] = traceback_count
    payload["loader_finalizer_count"] = loader_finalizer_count
    payload["extra_tracebacks"] = extra_tracebacks
    unexpected_runtime_lines = [
        line.strip()
        for line in text.splitlines()
        if "runtimeerror" in line
        and not ("dataloader worker" in line and "is killed by signal: aborted" in line)
    ]
    known_loader_shutdown = (
        loader_finalizer_count > 0
        and "dataloader worker" in text
        and "is killed by signal: aborted" in text
        and extra_tracebacks == 0
        and not unexpected_runtime_lines
    )
    if known_loader_shutdown and cmd_status == 0:
        payload["known_loader_shutdown_warning"] = True
        warnings.append("known DataLoader teardown/finalizer warning detected after training")
    if (has_traceback or has_runtimeerror) and not payload["known_loader_shutdown_warning"]:
        errors.append("launch.log contains unexpected traceback/runtimeerror")
    if extra_tracebacks:
        errors.append(f"launch.log contains {extra_tracebacks} traceback(s) not explained by DataLoader finalizer")
    if unexpected_runtime_lines:
        errors.append(f"launch.log contains unexpected RuntimeError lines: {unexpected_runtime_lines[:3]}")
if errors:
    print("Training health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    health_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(1)
health_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["known_loader_shutdown_warning"]:
    print("Training health check passed; known_loader_shutdown_warning=true")
else:
    print("Training health check passed; known_loader_shutdown_warning=false")
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
    --batch-size "$SKILL_BATCH_SIZE" \
    --max-horizon "$SKILL_MAX_HORIZON" \
    --train-log-csv "$run_dir/log_r0.csv" \
    --output "$skill_dir"
  python -m app.vjepa_wm.diagnostics.mgvt_d1_probes \
    --skill-json "$skill_dir/skill_score.json" \
    --output "$probe_dir"
}

score_dh_controls() {
  local variant="$1"
  local stage="$2"
  local run_id="20260604_mgvt_d1r_pusht_${variant}_${stage}_seed234"
  local run_dir="$RUN_ROOT/$run_id"
  local ckpt_dir="$CKPT_ROOT/$run_id"
  local skill_dir="$RUN_ROOT/skill_scores/$run_id"
  local probe_dir="$RUN_ROOT/probes/$run_id"
  local controls_dir="$probe_dir/dh_control_scores"
  local config_dir="$probe_dir/dh_control_configs"
  mkdir -p "$controls_dir" "$config_dir"

  for mode in remove shuffle random; do
    local control_config="$config_dir/config_dh_${mode}.yaml"
    local control_skill_dir="$controls_dir/$mode"
    python - "$run_dir/config.yaml" "$control_config" "$mode" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

source_config, control_config, mode = sys.argv[1:4]
cfg = load_yaml(source_config)
cfg.setdefault("model", {}).setdefault("predictor", {})["dh_ablation"] = mode
Path(control_config).parent.mkdir(parents=True, exist_ok=True)
dump_yaml(cfg, control_config)
PY
    python -m app.vjepa_wm.diagnostics.skill_score \
      --config "$control_config" \
      --checkpoint "$ckpt_dir/jepa-latest.pth.tar" \
      --device "$SKILL_DEVICE" \
      --num-workers 0 \
      --batch-size "$SKILL_BATCH_SIZE" \
      --max-horizon "$SKILL_MAX_HORIZON" \
      --train-log-csv "$run_dir/log_r0.csv" \
      --output "$control_skill_dir"
  done

  python - "$skill_dir/skill_score.json" "$controls_dir" "$probe_dir/dh_controls.json" <<'PY'
import json
import math
import sys
from pathlib import Path

intact_path = Path(sys.argv[1])
controls_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
horizon = "4"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def h4(data):
    value = data.get("moved_region_change_skill_by_horizon", {}).get(horizon)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

intact = load(intact_path)
intact_value = h4(intact)
payload = {
    "probe_status": "rescored-checkpoint-dh-controls",
    "source": str(intact_path.resolve()),
    "model": intact.get("model"),
    "task": intact.get("task"),
    "seed": intact.get("seed"),
    "horizon": int(horizon),
    "moved_region_h4": intact_value,
    "delta_semantics": "intact_moved_region_h4 - ablated_moved_region_h4; positive means the d_h control degraded performance",
    "controls": {},
}
for mode in ("remove", "shuffle", "random"):
    score_path = controls_dir / mode / "skill_score.json"
    data = load(score_path)
    value = h4(data)
    delta = None if intact_value is None or value is None else intact_value - value
    payload[f"{mode}_moved_region_h4"] = value
    payload[f"{mode}_delta"] = delta
    payload["controls"][mode] = {
        "skill_json": str(score_path.resolve()),
        "moved_region_h4": value,
        "delta": delta,
    }
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
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
  score_dh_controls "$variant" "g_refine"
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
