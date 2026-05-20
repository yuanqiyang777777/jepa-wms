#!/bin/bash
# Run Wall/Maze CEM-L2 planning evals for the last 10 checkpoints from a training run.
#
# Required:
#   ENV=wall|maze
#   TRAIN_RUN_ID=20260519_pilot_wall_mfl_e0.5_g1_seed0_retry2
#
# Optional:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:7"
#   EPOCHS="40 41 42 43 44 45 46 47 48 49"  # auto-detected from checkpoints when unset
#   EVAL_CFG=configs/online_plan_evals/wall/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2.yaml
#   EVAL_ROOT_NAME=eval_last10_wall_l2_cem

source "$(dirname "$0")/_common.sh"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:7}"
ENV="${ENV:-}"
EPOCHS="${EPOCHS:-}"

if [ -z "${TRAIN_RUN_ID:-}" ]; then
  echo "ERROR: TRAIN_RUN_ID is required" >&2
  exit 2
fi
if [ -z "$ENV" ]; then
  echo "ERROR: ENV is required (wall or maze)" >&2
  exit 2
fi
if [ -z "${JEPAWM_LOGS:-}" ]; then
  echo "ERROR: JEPAWM_LOGS is not set; source ~/.jepawm_env first" >&2
  exit 2
fi

case "$ENV" in
  wall)
    DEFAULT_EVAL_CFG="configs/online_plan_evals/wall/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2.yaml"
    ;;
  maze|mz)
    ENV="maze"
    DEFAULT_EVAL_CFG="configs/online_plan_evals/mz/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2.yaml"
    ;;
  *)
    echo "ERROR: ENV must be wall or maze; got $ENV" >&2
    exit 2
    ;;
esac

EVAL_CFG="${EVAL_CFG:-$DEFAULT_EVAL_CFG}"
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
EVAL_ROOT_NAME="${EVAL_ROOT_NAME:-eval_last10_${ENV}_l2_cem}"
EVAL_ROOT="$TRAIN_DIR/$EVAL_ROOT_NAME"

if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found: $TRAIN_CONFIG" >&2
  exit 2
fi
if [ -e "$EVAL_ROOT" ] && [ "$(find "$EVAL_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: eval root already exists and is non-empty: $EVAL_ROOT" >&2
  echo "Set EVAL_ROOT_NAME to a new name to avoid overwriting raw eval logs." >&2
  exit 2
fi
mkdir -p "$EVAL_ROOT/configs" "$EVAL_ROOT/logs"

if [ -z "$EPOCHS" ]; then
  EPOCHS="$(
    find "$TRAIN_DIR" -maxdepth 1 -name 'jepa-e*.pth.tar' -printf '%f\n' \
      | sed -E 's/^jepa-e([0-9]+)\.pth\.tar$/\1/' \
      | sort -n \
      | tail -10 \
      | tr '\n' ' '
  )"
fi

EPOCHS="${EPOCHS//,/ }"
read -r -a EPOCH_ARRAY <<< "$EPOCHS"
if [ "${#EPOCH_ARRAY[@]}" -ne 10 ]; then
  echo "ERROR: expected exactly 10 epochs for last-10 eval, got ${#EPOCH_ARRAY[@]}: $EPOCHS" >&2
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
import copy
import sys
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
    cfg["task_specification"]["img_size"] = train_cfg["data"].get(
        "img_size", cfg["task_specification"].get("img_size", 224)
    )

    for key in ("obs", "max_episode_steps", "goal_H"):
        if evals_cfg.get(key) is not None:
            cfg["task_specification"][key] = evals_cfg[key]
    for key in ("horizon", "num_act_stepped", "num_elites"):
        if evals_cfg.get(key) is not None:
            cfg["planner"][key] = evals_cfg[key]
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
    base_tag = str(template_cfg.get("tag", "gc_zeroshot")).rstrip("/")
    cfg["tag"] = f"{base_tag}_r{img_size}_alpha{alpha}_ep{eval_episodes}/epoch-{epoch}"
    cfg_path = eval_root / "configs" / f"eval_e{epoch}.yaml"
    dump_yaml(cfg, cfg_path)
    print(f"{epoch}\t{cfg_path}")
PY

{
  echo "TRAIN_RUN_ID=$TRAIN_RUN_ID"
  echo "ENV=$ENV"
  echo "EVAL_CFG=$EVAL_CFG"
  echo "EPOCHS=$EPOCHS"
  echo "DEVICES=$DEVICES"
  echo "EVAL_ROOT=$EVAL_ROOT"
} | tee "$EVAL_ROOT/manifest.txt" >&2

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
    batch_status=0
    for j in "${!PIDS[@]}"; do
      if ! wait "${PIDS[$j]}"; then
        echo "ERROR: eval for epoch ${PID_EPOCHS[$j]} failed; see $EVAL_ROOT/logs/eval_e${PID_EPOCHS[$j]}.log" >&2
        batch_status=1
      fi
    done
    PIDS=()
    PID_EPOCHS=()
    if [ "$batch_status" -ne 0 ]; then
      exit 1
    fi
  fi
done

final_status=0
for j in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$j]}"; then
    echo "ERROR: eval for epoch ${PID_EPOCHS[$j]} failed; see $EVAL_ROOT/logs/eval_e${PID_EPOCHS[$j]}.log" >&2
    final_status=1
  fi
done
if [ "$final_status" -ne 0 ]; then
  exit 1
fi

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
