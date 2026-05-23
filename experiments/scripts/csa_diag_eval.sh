#!/bin/bash
# CSA-MPC diagnostic eval launcher.
#
# Runs ONE env's instrumented vanilla CEM/MPC eval against the released JEPA-WM
# "Ours" checkpoint, with csa_diagnostics.enabled=true so the PlanEvaluator
# hook writes per-episode pooled-latent dumps. Support scoring + the full
# diagnostic suite (诊断 0-4) run OFFLINE against those dumps via
# experiments/scripts/csa/analyze_csa_pilot.py -- this script does not need
# the support memory.
#
# Usage on lab01 from repo root:
#   bash experiments/scripts/csa_diag_eval.sh <env> [out_yaml]
# Examples:
#   GPU_ID=0 bash experiments/scripts/csa_diag_eval.sh wall
#   GPU_ID=2 EVAL_EPISODES=96 bash experiments/scripts/csa_diag_eval.sh mw-reach
#   QUICK_DEBUG=1 GPU_ID=0 bash experiments/scripts/csa_diag_eval.sh wall
#
# env must be one of: pusht | wall | maze | mw-reach | mw-reach-wall
#
# This script lives in the fork at jepa-wms/experiments/scripts/ and is
# committed to exp/20260522-csa-pilot so it is available on lab01 after a
# `git fetch && git checkout exp/20260522-csa-pilot`.

source "$(dirname "$0")/_common.sh"

ENV="${1:-}"
if [ -z "$ENV" ]; then
  echo "ERROR: env tag is required (pusht | wall | maze | mw-reach | mw-reach-wall)" >&2
  exit 1
fi

OUT_YAML="${2:-/tmp/csa_diag_${ENV}.yaml}"
EVAL_EPISODES="${EVAL_EPISODES:-96}"
QUICK_DEBUG_FLAG=""
if [ "${QUICK_DEBUG:-0}" = "1" ]; then
  QUICK_DEBUG_FLAG="--quick-debug"
  EVAL_EPISODES=1  # quick-debug forces 1-episode anyway
fi

DUMP_DIR_FLAG=""
if [ -n "${DUMP_DIR:-}" ]; then
  DUMP_DIR_FLAG="--dump-dir ${DUMP_DIR}"
fi

SEED_FLAG=""
if [ -n "${SEED:-}" ]; then
  SEED_FLAG="--seed ${SEED}"
fi

FOLDER_FLAG=""
if [ -n "${FOLDER:-}" ]; then
  FOLDER_FLAG="--folder ${FOLDER}"
fi

TOPK_FLAG=""
if [ -n "${TOPK:-}" ]; then
  TOPK_FLAG="--topk-candidates ${TOPK}"
fi

echo ">>> Generating CSA diagnostic config for env=${ENV} -> ${OUT_YAML}"
# shellcheck disable=SC2086
python experiments/scripts/csa/make_official_csa_eval_config.py \
  --env "$ENV" \
  --eval-episodes "$EVAL_EPISODES" \
  --output "$OUT_YAML" \
  $QUICK_DEBUG_FLAG \
  $DUMP_DIR_FLAG \
  $SEED_FLAG \
  $FOLDER_FLAG \
  $TOPK_FLAG

echo ">>> Launching eval on GPU(s) ${CUDA_VISIBLE_DEVICES}"
# IMPORTANT: do NOT pass --debug here. ``evals/main.py`` with ``--debug``
# hardcodes ``devices=["cuda:0"]`` and then ``process_main`` *overwrites*
# CUDA_VISIBLE_DEVICES="0" -- so every parallel csa_diag_eval.sh run lands
# on physical GPU 0 regardless of GPU_ID. Without --debug, the eval respects
# ``--devices cuda:N``.
python -m evals.main \
  --fname "$OUT_YAML" \
  --devices "cuda:${GPU_ID:-0}"
