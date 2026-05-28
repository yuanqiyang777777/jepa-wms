#!/bin/bash
# Diagnostic sweep: PushT raw timing across NUM_WORKERS values.
#
# Motivation: the 2026-05-28 timing profile flagged PushT raw as needs_review
# (CV 7.22% across 5 repeats, bimodal data_fetch 0.3-600 ms). This sweep asks
# whether worker count alone stabilises the raw path before we invest in PushT
# Lance.
#
# Method: for each NUM_WORKERS in {8, 12, 16, 24, 32}, run profile_step.sh 3
# times on the same PushT raw config used by run_phase1_timing_profile.sh, then
# summarise the 15 runs into a separate CSV/MD.
#
# Run IDs: 20260528_pusht_numworkers_w${N}_r${R}
#   - separate prefix from 20260528_dino_wm_timing_* so the existing summary
#     does not pick these up
#
# Optional env vars:
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5"
#   WORKER_VALUES="8 12 16 24 32"
#   REPEATS="1 2 3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"
cd "$REPO_ROOT"

unset CUDA_VISIBLE_DEVICES

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
WORKER_VALUES="${WORKER_VALUES:-8 12 16 24 32}"
REPEATS="${REPEATS:-1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROFILE_WARMUP="${PROFILE_WARMUP:-30}"
PROFILE_STEPS="${PROFILE_STEPS:-300}"
PROFILE_IPE="${PROFILE_IPE:-360}"
DATE_TAG="${DATE_TAG:-20260528}"
RUN_PREFIX="${RUN_PREFIX:-${DATE_TAG}_pusht_numworkers}"
BASE_CONFIG="${BASE_CONFIG:-configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save.yaml}"

PROFILE_SCRIPT="$SCRIPT_DIR/profile_step.sh"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_training_time_profile.py"

run_one() {
  local num_workers="$1"
  local rep="$2"
  local run_id="${RUN_PREFIX}_w${num_workers}_r${rep}"

  echo "" >&2
  echo "=== num_workers sweep run: $run_id ===" >&2
  RUN_ID="$run_id" \
  BASE_CONFIG="$BASE_CONFIG" \
  DEVICES="$DEVICES" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$num_workers" \
  PROFILE_WARMUP="$PROFILE_WARMUP" \
  PROFILE_STEPS="$PROFILE_STEPS" \
  PROFILE_IPE="$PROFILE_IPE" \
  bash "$PROFILE_SCRIPT"
}

echo "PushT num_workers sweep starting $(date -Iseconds)" >&2
echo "  WORKER_VALUES = $WORKER_VALUES" >&2
echo "  REPEATS       = $REPEATS" >&2
echo "  BASE_CONFIG   = $BASE_CONFIG" >&2

for nw in $WORKER_VALUES; do
  for rep in $REPEATS; do
    run_one "$nw" "$rep"
  done
done

echo "" >&2
echo "=== summarising 20260528_pusht_numworkers_* ===" >&2
python "$SUMMARY_SCRIPT" \
  --logs-root "$JEPAWM_LOGS" \
  --pattern "${RUN_PREFIX}_*" \
  --output-csv "experiments/results/pusht_numworkers_sweep_${DATE_TAG}.csv" \
  --output-md "experiments/results/pusht_numworkers_sweep_${DATE_TAG}.md" \
  --warmup "$PROFILE_WARMUP" \
  --measured-steps "$PROFILE_STEPS"

echo "PushT num_workers sweep finished $(date -Iseconds)" >&2
