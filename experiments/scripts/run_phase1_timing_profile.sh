#!/bin/bash
# Run the Phase-1 Profile x3 timing matrix serially on lab01.
#
# Produces one run directory per env/backend/repeat under $JEPAWM_LOGS and writes
# summary CSV/Markdown via summarize_training_time_profile.py.
#
# Optional:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5"
#   REPEATS="1 2 3"
#   RUN_RAW=1
#   RUN_LANCE=1
#   LANCE_ENVS="maze wall pusht mw"
#   JEPAWM_DSET_LANCE=$JEPAWM_DSET/_lance_20260528
#
# Planned run_id examples:
#   20260528_dino_wm_timing_pusht_raw_r${rep}
#   20260528_dino_wm_timing_maze_lance_r${rep}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
REPEATS="${REPEATS:-1 2 3}"
RUN_LANCE="${RUN_LANCE:-1}"
RUN_RAW="${RUN_RAW:-1}"
LANCE_ENVS="${LANCE_ENVS:-maze wall pusht mw}"
BATCH_SIZE=32
NUM_WORKERS=16
PROFILE_WARMUP=30
PROFILE_STEPS=300
PROFILE_IPE=360
DATE_TAG="${DATE_TAG:-20260528}"
RUN_PREFIX="${RUN_PREFIX:-${DATE_TAG}_dino_wm_timing}"
export JEPAWM_DSET_LANCE="${JEPAWM_DSET_LANCE:-$JEPAWM_DSET/_lance_20260528}"

PROFILE_SCRIPT="$SCRIPT_DIR/profile_step.sh"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_training_time_profile.py"

run_profile() {
  local env_name="$1"
  local backend="$2"
  local rep="$3"
  local base_config="$4"
  local run_id="${RUN_PREFIX}_${env_name}_${backend}_r${rep}"

  echo "" >&2
  echo "=== timing run: $run_id ===" >&2
  RUN_ID="$run_id" \
  BASE_CONFIG="$base_config" \
  DEVICES="$DEVICES" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  PROFILE_WARMUP="$PROFILE_WARMUP" \
  PROFILE_STEPS="$PROFILE_STEPS" \
  PROFILE_IPE="$PROFILE_IPE" \
  bash "$PROFILE_SCRIPT"
}

run_lance_profile() {
  local env_name="$1"
  local rep="$2"
  local base_config
  local lance_uri
  case "$env_name" in
    maze)
      base_config="configs/vjepa_wm/mz_sweep/mz_lance_smoke.yaml"
      lance_uri="$JEPAWM_DSET_LANCE/PointMaze.lance"
      ;;
    wall)
      base_config="configs/vjepa_wm/wall_sweep/wall_lance_smoke.yaml"
      lance_uri="$JEPAWM_DSET_LANCE/Wall.lance"
      ;;
    pusht)
      base_config="configs/vjepa_wm/pt_sweep/pt_lance_smoke.yaml"
      lance_uri="$JEPAWM_DSET_LANCE/PushT.lance"
      ;;
    mw)
      base_config="configs/vjepa_wm/mw_final_sweep/mw_lance_smoke.yaml"
      lance_uri="$JEPAWM_DSET_LANCE/Metaworld.lance"
      ;;
    *)
      echo "ERROR: unknown LANCE_ENVS entry: $env_name" >&2
      exit 2
      ;;
  esac
  local run_id="${RUN_PREFIX}_${env_name}_lance_r${rep}"

  if [ ! -d "$lance_uri" ]; then
    echo "ERROR: Lance dataset not found: $lance_uri" >&2
    echo "Convert the dataset first under JEPAWM_DSET_LANCE=$JEPAWM_DSET_LANCE." >&2
    exit 2
  fi

  echo "" >&2
  echo "=== timing run: $run_id ===" >&2
  RUN_ID="$run_id" \
  BASE_CONFIG="$base_config" \
  DEVICES="$DEVICES" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  PROFILE_WARMUP="$PROFILE_WARMUP" \
  PROFILE_STEPS="$PROFILE_STEPS" \
  PROFILE_IPE="$PROFILE_IPE" \
  BACKEND_KIND=swm_lance \
  LANCE_URI="$lance_uri" \
  bash "$PROFILE_SCRIPT"
}

for rep in $REPEATS; do
  if [ "$RUN_RAW" = "1" ]; then
    run_profile "pusht" "raw" "$rep" \
      "configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save.yaml"
    run_profile "wall" "raw" "$rep" \
      "configs/vjepa_wm/wall_sweep/wall_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save_2n.yaml"
    run_profile "maze" "raw" "$rep" \
      "configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save_2n.yaml"
    run_profile "mw" "raw" "$rep" \
      "configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_repro_1roll_save.yaml"
  else
    echo "Skipping raw r${rep} because RUN_RAW=$RUN_RAW" >&2
  fi
  if [ "$RUN_LANCE" = "1" ]; then
    for lance_env in $LANCE_ENVS; do
      run_lance_profile "$lance_env" "$rep"
    done
  else
    echo "Skipping Lance r${rep} because RUN_LANCE=$RUN_LANCE" >&2
  fi
done

python "$SUMMARY_SCRIPT" \
  --logs-root "$JEPAWM_LOGS" \
  --pattern "${RUN_PREFIX}_*" \
  --output-csv "experiments/results/training_time_profile_20260528.csv" \
  --output-md "experiments/results/training_time_profile_20260528.md" \
  --warmup "$PROFILE_WARMUP" \
  --measured-steps "$PROFILE_STEPS"
