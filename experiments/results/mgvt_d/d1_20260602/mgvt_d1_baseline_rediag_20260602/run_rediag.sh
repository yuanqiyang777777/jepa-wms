#!/usr/bin/env bash
set -eo pipefail
source ~/.jepawm_env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms
cd /home/ps/Code/yqy/worktrees/mgvt-d1-20260601
export JEPAWM_DSET_LANCE=/home/ps/Code/yqy/DINO-WM/jepawm_data/_lance_20260528
RUN_ROOT=/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_d1_baseline_rediag_20260602
SRC_LOG=/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_stage2c_3seed_20260531
SRC_CKPT=/home/ps/Code/yqy/DINO-WM/jepawm_checkpoints/mgvt_stage2c_3seed_20260531
mkdir -p "$RUN_ROOT/skill_scores" "$RUN_ROOT/summary"
{
  echo "MGVT-D Stage-D1 baseline re-diagnostics started $(date -Iseconds)"
  echo "run_root=$RUN_ROOT"
  echo "src_log=$SRC_LOG"
  echo "src_ckpt=$SRC_CKPT"
  echo "devices=cuda:0 cuda:1 cuda:2 cuda:3 cuda:4"
  echo "max_horizon=6"
  echo "moved_top_frac=0.20"
  echo "moved_min_floor=0.0"
  echo "moved_denominator_floor=0.0"
  echo "moved_dilation=1"
} | tee "$RUN_ROOT/scan_start.txt"

git rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
git describe --dirty --always > "$RUN_ROOT/git_describe.txt"
git status --short --untracked-files=all > "$RUN_ROOT/git_status_start.txt"
if [ -s "$RUN_ROOT/git_status_start.txt" ]; then
  echo "ERROR: remote worktree dirty" | tee "$RUN_ROOT/scan_failed.txt"
  cat "$RUN_ROOT/git_status_start.txt" | tee -a "$RUN_ROOT/scan_failed.txt"
  exit 2
fi

find "$SRC_LOG" -mindepth 2 -maxdepth 2 -name config.yaml | sort > "$RUN_ROOT/configs.txt"
CONFIG_COUNT=$(wc -l < "$RUN_ROOT/configs.txt")
if [ "$CONFIG_COUNT" -ne 15 ]; then
  echo "ERROR: expected 15 Stage-2c configs, got $CONFIG_COUNT" | tee "$RUN_ROOT/scan_failed.txt"
  exit 2
fi

run_one() {
  local run_id="$1"
  local cfg="$2"
  local device="$3"
  local ckpt="$SRC_CKPT/$run_id/jepa-latest.pth.tar"
  local out_dir="$RUN_ROOT/skill_scores/$run_id"
  local train_log="$SRC_LOG/$run_id/log_r0.csv"
  mkdir -p "$out_dir"
  if [ ! -f "$ckpt" ]; then
    echo "ERROR: missing checkpoint $ckpt" > "$out_dir/skill_score.log"
    return 2
  fi
  local train_args=()
  if [ -f "$train_log" ]; then
    train_args=(--train-log-csv "$train_log")
  fi
  cp "$cfg" "$out_dir/config.yaml"
  {
    echo "=== REDIAG $run_id $(date -Iseconds) device=cuda:$device ==="
    echo "config=$cfg"
    echo "checkpoint=$ckpt"
    python -m app.vjepa_wm.diagnostics.skill_score \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --output "$out_dir" \
      --device "cuda:$device" \
      --num-workers 0 \
      --max-horizon 6 \
      "${train_args[@]}"
    echo "=== DONE $run_id $(date -Iseconds) ==="
  } > "$out_dir/skill_score.log" 2>&1
}

status=0
devices=(0 1 2 3 4)
batch_pids=()
idx=0
while read -r cfg; do
  run_dir=$(dirname "$cfg")
  run_id=$(basename "$run_dir")
  device=${devices[$((idx % ${#devices[@]}))]}
  echo "QUEUE $run_id device=cuda:$device" | tee -a "$RUN_ROOT/scan.log"
  run_one "$run_id" "$cfg" "$device" &
  batch_pids+=("$!")
  idx=$((idx + 1))
  if [ "${#batch_pids[@]}" -ge "${#devices[@]}" ]; then
    for pid in "${batch_pids[@]}"; do
      wait "$pid" || status=1
    done
    batch_pids=()
  fi
done < "$RUN_ROOT/configs.txt"
for pid in "${batch_pids[@]}"; do
  wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
  echo "MGVT-D baseline re-diagnostics failed $(date -Iseconds)" | tee "$RUN_ROOT/scan_failed.txt"
  exit 1
fi

SCORE_COUNT=$(find "$RUN_ROOT/skill_scores" -mindepth 2 -maxdepth 2 -name skill_score.json | wc -l)
if [ "$SCORE_COUNT" -ne 15 ]; then
  echo "ERROR: expected 15 skill_score.json, got $SCORE_COUNT" | tee "$RUN_ROOT/scan_failed.txt"
  exit 1
fi

python experiments/scripts/summarize_mgvt_d1.py \
  --skill-root "$RUN_ROOT/skill_scores" \
  --output-dir "$RUN_ROOT/summary" | tee -a "$RUN_ROOT/scan.log"

echo "MGVT-D Stage-D1 baseline re-diagnostics finished $(date -Iseconds)" | tee "$RUN_ROOT/scan_end.txt"
