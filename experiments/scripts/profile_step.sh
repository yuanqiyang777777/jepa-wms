#!/bin/bash
# Run the segmented per-step profiler (app/vjepa_wm/profiling.py) on a short,
# self-terminating training run. Prints a per-step cost breakdown to launch.log.
#
# Decides latent-cache vs dataloader-slicing optimization priority (plan step 1).
#
# Required:
#   RUN_ID=20260518_profile_pusht
#
# Optional:
#   BASE_CONFIG=...           default: PushT vjtranoaug repro config
#   DEVICES="cuda:0 cuda:1 ..." default: cuda:0..5  (1-6 devices)
#   BATCH_SIZE=32             per-GPU batch (matches the 1.25 s/step anchor)
#   NUM_WORKERS=16            dataloader workers, kept fixed across raw/Lance timing
#   PROFILE_WARMUP=20         steps skipped before measuring
#   PROFILE_STEPS=200         steps averaged
#   PROFILE_IPE=240           iterations_per_epoch cap (>= WARMUP+STEPS+margin)
#   BACKEND_KIND=raw|swm_lance  optional data backend override; swm_lance is PointMaze-only
#   LANCE_URI=...             required when BACKEND_KIND=swm_lance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

BASE_CONFIG="${BASE_CONFIG:-configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-200}"
PROFILE_IPE="${PROFILE_IPE:-240}"
BACKEND_KIND="${BACKEND_KIND:-}"
LANCE_URI="${LANCE_URI:-}"

if [ -z "${RUN_ID:-}" ]; then
  echo "ERROR: RUN_ID is required" >&2
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
if [ "$BACKEND_KIND" != "" ] && [ "$BACKEND_KIND" != "raw" ] && [ "$BACKEND_KIND" != "swm_lance" ]; then
  echo "ERROR: BACKEND_KIND must be empty, raw, or swm_lance (got $BACKEND_KIND)" >&2
  exit 2
fi
if [ "$BACKEND_KIND" = "swm_lance" ] && [ -z "$LANCE_URI" ]; then
  echo "ERROR: LANCE_URI is required when BACKEND_KIND=swm_lance" >&2
  exit 2
fi

DEVICES="${DEVICES//,/ }"
read -r -a DEVICE_ARRAY <<< "$DEVICES"
WORLD_SIZE="${#DEVICE_ARRAY[@]}"
if [ "$WORLD_SIZE" -lt 1 ] || [ "$WORLD_SIZE" -gt 6 ]; then
  echo "ERROR: DEVICES must resolve to 1-6 devices (got $WORLD_SIZE)" >&2
  exit 2
fi

check_gpus_free "${DEVICE_ARRAY[@]}"

RUN_DIR="$JEPAWM_LOGS/$RUN_ID"
mkdir -p "$RUN_DIR"
CONFIG_PATH="$RUN_DIR/config.yaml"

python - "$BASE_CONFIG" "$CONFIG_PATH" "$BATCH_SIZE" "$NUM_WORKERS" "$RUN_DIR" "$PROFILE_IPE" "$BACKEND_KIND" "$LANCE_URI" <<'PY'
import sys
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

base_config, config_path, batch_size, num_workers, run_dir, profile_ipe, backend_kind, lance_uri = sys.argv[1:9]
cfg = load_yaml(base_config)

cfg["folder"] = run_dir
cfg["data"]["loader"]["batch_size"] = int(batch_size)
cfg["data"]["loader"]["num_workers"] = int(num_workers)
cfg["evals"]["eval_cfg_paths"] = None
cfg["optimization"]["transition_model"]["num_epochs"] = 1
cfg["optimization"]["transition_model"]["iterations_per_epoch"] = int(profile_ipe)
cfg["logging"]["wandb"]["use_wandb"] = False
cfg["meta"]["data_traj_rollout_eval"]["do_data_traj_rollout_eval"] = False
cfg["meta"]["light_eval_freq"] = int(profile_ipe) + 1

if backend_kind:
    if backend_kind not in {"raw", "swm_lance"}:
        raise ValueError(f"BACKEND_KIND must be raw or swm_lance, got {backend_kind!r}")
    datasets = cfg.get("data", {}).get("datasets")
    if backend_kind == "swm_lance" and datasets != ["PointMaze"]:
        raise ValueError(
            f"BACKEND_KIND=swm_lance is supported only for PointMaze timing; datasets={datasets!r}"
        )
    if backend_kind == "swm_lance" and not lance_uri:
        raise ValueError("LANCE_URI is required when BACKEND_KIND=swm_lance")
    backend = cfg["data"].setdefault("backend", {})
    backend["kind"] = backend_kind
    backend["fall_back_to_raw_if_unsupported"] = False
    if lance_uri:
        backend["lance_uri"] = lance_uri

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
dump_yaml(cfg, config_path)
PY

cp "$0" "$RUN_DIR/launcher.sh"
{
  git rev-parse HEAD
  git describe --dirty --always
} > "$RUN_DIR/git_commit.txt"

export JEPAWM_PROFILE=1
export JEPAWM_PROFILE_WARMUP="$PROFILE_WARMUP"
export JEPAWM_PROFILE_STEPS="$PROFILE_STEPS"

{
  echo "RUN_ID=$RUN_ID"
  echo "BASE_CONFIG=$BASE_CONFIG"
  echo "DEVICES=$DEVICES (world_size=$WORLD_SIZE)"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "BACKEND_KIND=${BACKEND_KIND:-unset}"
  echo "LANCE_URI=${LANCE_URI:-unset}"
  echo "JEPAWM_PROFILE=1 warmup=$PROFILE_WARMUP steps=$PROFILE_STEPS ipe=$PROFILE_IPE"
  echo "JEPAWM_DDP_STATIC_GRAPH=${JEPAWM_DDP_STATIC_GRAPH:-unset}"
  echo "JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW=${JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW:-unset}"
  echo "JEPAWM_DDP_BROADCAST_BUFFERS=${JEPAWM_DDP_BROADCAST_BUFFERS:-unset}"
  echo "JEPAWM_DDP_BUCKET_CAP_MB=${JEPAWM_DDP_BUCKET_CAP_MB:-unset}"
  echo "JEPAWM_DDP_COMM_HOOK=${JEPAWM_DDP_COMM_HOOK:-unset}"
  echo "NCCL_ALGO=${NCCL_ALGO:-unset}"
  echo "config=$CONFIG_PATH"
  echo "run_dir=$RUN_DIR"
} >&2

set +e
python -m app.main --fname "$CONFIG_PATH" --devices "${DEVICE_ARRAY[@]}" 2>&1 | tee "$RUN_DIR/launch.log"
cmd_status=${PIPESTATUS[0]}
set -e

echo "" >&2
echo "=== profiler breakdown (from $RUN_DIR/launch.log) ===" >&2
grep -A 18 "StepProfiler] mean per-step" "$RUN_DIR/launch.log" >&2 \
  || echo "WARNING: profiler table not found in launch.log" >&2

exit "$cmd_status"
