#!/bin/bash
# Train the PushT Terver baseline and write checkpoints only.
#
# Required:
#   RUN_ID=20260518_baseline_pusht_terver_seed234
#   SEED=234
#   BATCH_SIZE=16
#
# Optional:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5"
#   QUICK_DEBUG=1

source "$(dirname "$0")/_common.sh"

unset CUDA_VISIBLE_DEVICES

BASE_CONFIG="${BASE_CONFIG:-configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
QUICK_DEBUG="${QUICK_DEBUG:-0}"

if [ -z "${RUN_ID:-}" ]; then
  echo "ERROR: RUN_ID is required" >&2
  exit 2
fi
if [ -z "${SEED:-}" ]; then
  echo "ERROR: SEED is required" >&2
  exit 2
fi
if [ -z "${BATCH_SIZE:-}" ]; then
  echo "ERROR: BATCH_SIZE is required" >&2
  exit 2
fi
if [ -z "${JEPAWM_LOGS:-}" ]; then
  echo "ERROR: JEPAWM_LOGS is not set; source ~/.jepawm_env first" >&2
  exit 2
fi
if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 2
fi

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

RUN_DIR="$JEPAWM_LOGS/$RUN_ID"
mkdir -p "$RUN_DIR"
CONFIG_PATH="$RUN_DIR/config.yaml"

python - "$BASE_CONFIG" "$CONFIG_PATH" "$SEED" "$BATCH_SIZE" "$RUN_DIR" "$QUICK_DEBUG" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, seed, batch_size, run_dir, quick_debug = sys.argv[1:7]
cfg = load_yaml(base_config)

cfg["folder"] = run_dir
cfg["meta"]["seed"] = int(seed)
cfg["data"]["seed"] = int(seed)
cfg["data"]["loader"]["batch_size"] = int(batch_size)
cfg["evals"]["eval_cfg_paths"] = None

if quick_debug == "1":
    # Config path: optimization.transition_model.num_epochs
    cfg["optimization"]["transition_model"]["num_epochs"] = 2
    cfg["meta"]["quick_debug"] = True
    cfg["logging"]["wandb"]["use_wandb"] = False

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
dump_yaml(cfg, config_path)
PY

cp "$0" "$RUN_DIR/launcher.sh"
{
  git rev-parse HEAD
  git describe --dirty --always
} > "$RUN_DIR/git_commit.txt"

GLOBAL_BATCH=$((WORLD_SIZE * BATCH_SIZE))
{
  echo "RUN_ID=$RUN_ID"
  echo "SEED=$SEED"
  echo "DEVICES=$DEVICES"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "world_size x batch_size = ${WORLD_SIZE} x ${BATCH_SIZE} = ${GLOBAL_BATCH} (paper global batch: 256)"
  echo "config=$CONFIG_PATH"
  echo "run_dir=$RUN_DIR"
} >&2

set +e
python -m app.main --fname "$CONFIG_PATH" --devices "${DEVICE_ARRAY[@]}" 2>&1 | tee "$RUN_DIR/launch.log"
cmd_status=${PIPESTATUS[0]}
set -e

python - "$RUN_DIR" "$cmd_status" <<'PY'
import csv
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
cmd_status = int(sys.argv[2])
errors = []

launch_log = run_dir / "launch.log"
latest_ckpt = run_dir / "jepa-latest.pth.tar"
train_csv = run_dir / "log_r0.csv"

if cmd_status != 0:
    errors.append(f"app.main command exited non-zero: {cmd_status}")
if not latest_ckpt.exists():
    errors.append(f"missing checkpoint: {latest_ckpt}")
if not train_csv.exists():
    errors.append(f"missing train csv: {train_csv}")
else:
    with train_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        errors.append(f"train csv has no data rows: {train_csv}")
    else:
        losses = []
        for row in rows:
            if "loss" in row and row["loss"] not in ("", None):
                try:
                    losses.append(float(row["loss"]))
                except ValueError:
                    errors.append(f"non-numeric loss value: {row['loss']!r}")
        if not losses:
            errors.append("no loss values found in log_r0.csv")
        elif not all(math.isfinite(v) for v in losses):
            errors.append("loss contains non-finite values")

if not launch_log.exists():
    errors.append(f"missing launch log: {launch_log}")
else:
    text = launch_log.read_text(errors="replace")
    lowered = text.lower()
    for marker in ("traceback", "out of memory", "cuda error"):
        if marker in lowered:
            errors.append(f"launch.log contains {marker!r}")

if errors:
    print("Training health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("Training health check passed")
PY
