#!/bin/bash
# Run CEM-L2 PushT planning evals for selected checkpoints from a training run.
#
# Required:
#   TRAIN_RUN_ID=20260518_baseline_pusht_terver_seed234
#
# Optional:
#   EPOCHS="40 41 42 43 44 45 46 47 48 49"
#   EPOCHS=1
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5"

source "$(dirname "$0")/_common.sh"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
EPOCHS="${EPOCHS:-40 41 42 43 44 45 46 47 48 49}"
EVAL_CFG="${EVAL_CFG:-configs/online_plan_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2.yaml}"

if [ -z "${TRAIN_RUN_ID:-}" ]; then
  echo "ERROR: TRAIN_RUN_ID is required" >&2
  exit 2
fi
if [ -z "${JEPAWM_LOGS:-}" ]; then
  echo "ERROR: JEPAWM_LOGS is not set; source ~/.jepawm_env first" >&2
  exit 2
fi
if [ ! -f "$EVAL_CFG" ]; then
  echo "ERROR: eval config template not found: $EVAL_CFG" >&2
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

TRAIN_DIR="$JEPAWM_LOGS/$TRAIN_RUN_ID"
TRAIN_CONFIG="$TRAIN_DIR/config.yaml"
EVAL_ROOT="$TRAIN_DIR/eval_last10"
mkdir -p "$EVAL_ROOT/configs" "$EVAL_ROOT/logs"

if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found: $TRAIN_CONFIG" >&2
  exit 2
fi

EPOCHS="${EPOCHS//,/ }"
read -r -a EPOCH_ARRAY <<< "$EPOCHS"
if [ "${#EPOCH_ARRAY[@]}" -lt 1 ]; then
  echo "ERROR: EPOCHS resolved to an empty list" >&2
  exit 2
fi

for epoch in "${EPOCH_ARRAY[@]}"; do
  ckpt="$TRAIN_DIR/jepa-e${epoch}.pth.tar"
  if [ ! -f "$ckpt" ]; then
    echo "ERROR: checkpoint not found: $ckpt" >&2
    exit 2
  fi
done

python - "$TRAIN_CONFIG" "$EVAL_CFG" "$EVAL_ROOT" "${EPOCH_ARRAY[@]}" <<'PY'
import sys
import copy
from pathlib import Path

from src.utils.yaml_utils import dump_yaml, load_yaml

train_config, eval_template, eval_root, *epochs = sys.argv[1:]
train_cfg = load_yaml(train_config)
template_cfg = load_yaml(eval_template)
eval_root = Path(eval_root)
train_dir = str(Path(train_config).parent)

for epoch in epochs:
    checkpoint = f"jepa-e{epoch}.pth.tar"
    out_dir = str(eval_root / f"e{epoch}")
    cfg = copy.deepcopy(template_cfg)
    evals_cfg = train_cfg.get("evals", {})

    cfg["nodes"] = 1
    cfg["tasks_per_node"] = 1
    cfg["folder"] = out_dir
    cfg["checkpoint_folder"] = train_dir
    cfg["meta"]["eval_episodes"] = evals_cfg.get("eval_episodes", cfg["meta"].get("eval_episodes", 96))
    cfg["task_specification"]["num_frames"] = train_cfg["model"]["tubelet_size_enc"]
    cfg["task_specification"]["num_proprios"] = train_cfg["model"]["tubelet_size_enc"]
    cfg["task_specification"]["img_size"] = train_cfg["data"].get("img_size", cfg["task_specification"].get("img_size", 224))
    if evals_cfg.get("obs") is not None:
        cfg["task_specification"]["obs"] = evals_cfg["obs"]
    if evals_cfg.get("max_episode_steps") is not None:
        cfg["task_specification"]["max_episode_steps"] = evals_cfg["max_episode_steps"]
    if evals_cfg.get("goal_H") is not None:
        cfg["task_specification"]["goal_H"] = evals_cfg["goal_H"]
    if evals_cfg.get("horizon") is not None:
        cfg["planner"]["horizon"] = evals_cfg["horizon"]
    if evals_cfg.get("num_act_stepped") is not None:
        cfg["planner"]["num_act_stepped"] = evals_cfg["num_act_stepped"]
    if evals_cfg.get("num_elites") is not None:
        cfg["planner"]["num_elites"] = evals_cfg["num_elites"]
    if evals_cfg.get("alpha") is not None:
        cfg["planner"]["planning_objective"]["alpha"] = evals_cfg["alpha"]
    if evals_cfg.get("decode") is not None:
        cfg["planner"]["decode_each_iteration"] = evals_cfg["decode"]

    cfg["model_kwargs"]["module_name"] = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"
    cfg["model_kwargs"]["checkpoint"] = checkpoint
    cfg["model_kwargs"]["pretrain_kwargs"] = train_cfg["model"]
    cfg["model_kwargs"]["data"] = train_cfg["data"]
    cfg["model_kwargs"]["data_aug"] = train_cfg["data_aug"]
    for key, value in evals_cfg.get("wrapper_kwargs", {}).items():
        cfg["model_kwargs"]["wrapper_kwargs"][key] = value
    img_size = cfg["task_specification"]["img_size"]
    alpha = cfg["planner"]["planning_objective"]["alpha"]
    eval_episodes = cfg["meta"]["eval_episodes"]
    cfg["tag"] = f"online_gc_zeroshot/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r{img_size}_alpha{alpha}_ep{eval_episodes}/epoch-{epoch}"
    cfg_path = eval_root / "configs" / f"eval_e{epoch}.yaml"
    dump_yaml(cfg, cfg_path)
    print(f"{epoch}\t{cfg_path}")
PY

declare -a PIDS=()
declare -a PID_EPOCHS=()
for i in "${!EPOCH_ARRAY[@]}"; do
  epoch="${EPOCH_ARRAY[$i]}"
  device="${DEVICE_ARRAY[$((i % WORLD_SIZE))]}"
  cfg="$EVAL_ROOT/configs/eval_e${epoch}.yaml"
  log="$EVAL_ROOT/logs/eval_e${epoch}.log"
  echo "Launching eval epoch=$epoch device=$device cfg=$cfg" >&2
  python -m evals.main --fname "$cfg" --devices "$device" > "$log" 2>&1 &
  PIDS+=("$!")
  PID_EPOCHS+=("$epoch")

  if [ "${#PIDS[@]}" -ge "$WORLD_SIZE" ]; then
    for j in "${!PIDS[@]}"; do
      if ! wait "${PIDS[$j]}"; then
        echo "ERROR: eval for epoch ${PID_EPOCHS[$j]} failed; see $EVAL_ROOT/logs/eval_e${PID_EPOCHS[$j]}.log" >&2
        exit 1
      fi
    done
    PIDS=()
    PID_EPOCHS=()
  fi
done

for j in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$j]}"; then
    echo "ERROR: eval for epoch ${PID_EPOCHS[$j]} failed; see $EVAL_ROOT/logs/eval_e${PID_EPOCHS[$j]}.log" >&2
    exit 1
  fi
done

python - "$EVAL_ROOT" "${EPOCH_ARRAY[@]}" <<'PY'
import csv
import math
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
epochs = sys.argv[2:]
summary_path = eval_root / "summary.csv"
rows = []
errors = []

for epoch in epochs:
    epoch_dir = eval_root / f"e{epoch}"
    matches = list(epoch_dir.rglob("eval.csv"))
    if len(matches) != 1:
        errors.append(f"epoch {epoch}: expected one eval.csv under {epoch_dir}, found {len(matches)}")
        continue
    eval_csv = matches[0]
    with eval_csv.open(newline="") as f:
        eval_rows = list(csv.DictReader(f))
    if not eval_rows:
        errors.append(f"epoch {epoch}: {eval_csv} has no data rows")
        continue
    value = eval_rows[-1].get("episode_success")
    try:
        success = float(value)
    except (TypeError, ValueError):
        errors.append(f"epoch {epoch}: invalid episode_success {value!r}")
        continue
    if not math.isfinite(success):
        errors.append(f"epoch {epoch}: non-finite episode_success {success!r}")
        continue
    rows.append({"epoch": epoch, "episode_success": success, "eval_csv": str(eval_csv)})

if errors:
    print("Eval health check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

mean_success = sum(float(row["episode_success"]) for row in rows) / len(rows)
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "episode_success", "eval_csv"])
    writer.writeheader()
    writer.writerows(rows)
    writer.writerow({"epoch": "mean", "episode_success": mean_success, "eval_csv": ""})

print(f"Eval summary written to {summary_path}")
print(f"mean_episode_success={mean_success:.6f}")
PY
