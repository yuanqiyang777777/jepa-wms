#!/bin/bash
# Train a JEPA world model with Motion-Focal Loss (MFL); writes checkpoints only.
# MFL strictly degenerates to the Terver baseline at MFL_ETA=0 (bit-exact), so this
# one launcher covers both the eta=0 baseline arm and the eta>0 MFL arm.
#
# Required:
#   RUN_ID=20260519_pilot_wall_mfl_e0.5_g1_seed0
#   SEED=0
#   BATCH_SIZE=32
#   BASE_CONFIG=configs/vjepa_wm/wall_sweep/wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml
#
# Optional:
#   MFL_ETA=0.5        (default 0.0 -> strict baseline degeneration)
#   MFL_GAMMA=1.0      (default 1.0)
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5"
#   QUICK_DEBUG=1      (2-epoch smoke test, wandb off)

source "$(dirname "$0")/_common.sh"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
QUICK_DEBUG="${QUICK_DEBUG:-0}"
MFL_ETA="${MFL_ETA:-0.0}"
MFL_GAMMA="${MFL_GAMMA:-1.0}"

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
if [ -z "${BASE_CONFIG:-}" ]; then
  echo "ERROR: BASE_CONFIG is required" >&2
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

python - "$BASE_CONFIG" "$CONFIG_PATH" "$SEED" "$BATCH_SIZE" "$RUN_DIR" "$QUICK_DEBUG" "$MFL_ETA" "$MFL_GAMMA" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, seed, batch_size, run_dir, quick_debug, mfl_eta, mfl_gamma = sys.argv[1:9]
cfg = load_yaml(base_config)

cfg["folder"] = run_dir
cfg["meta"]["seed"] = int(seed)
cfg["data"]["seed"] = int(seed)
cfg["data"]["loader"]["batch_size"] = int(batch_size)
cfg["evals"]["eval_cfg_paths"] = None

# Single-node local run: the *_2n base configs declare nodes: 2 for submitit only.
cfg["nodes"] = 1

# Motion-Focal Loss knobs. mfl_eta=0 -> compute_loss skips the MFL path entirely
# (strict bit-exact baseline degeneration).
loss_cfg = cfg.setdefault("loss", {})
loss_cfg["mfl_eta"] = float(mfl_eta)
loss_cfg["mfl_gamma"] = float(mfl_gamma)

if quick_debug == "1":
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
  echo "MFL_ETA=$MFL_ETA  MFL_GAMMA=$MFL_GAMMA"
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
