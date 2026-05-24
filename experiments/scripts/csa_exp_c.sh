#!/bin/bash
# Exp C launcher: build M_resid for one env, then run the residual-correlation
# analysis against the dumps the diagnostic phase already produced.
#
# Two phases:
#   1) ENCODE M_resid -- walks the env's VAL slicer with num_hist=1, num_pred=1,
#      runs the world model's one-step latent prediction, computes
#      e_i = mean((x_hat_t+1 - x_t+1)^2), saves three aligned tensors:
#        <out_dir>/<env>_residual_states.pt
#        <out_dir>/<env>_residual_actions.pt
#        <out_dir>/<env>_residual_errors.pt
#   2) RUN RESIDUAL CORRELATION -- loads dumps + M_real (existing
#      <env>_states.pt / <env>_actions.pt) + M_resid, computes (r, U_state,
#      U_action|x, U_SA) under raw + PCA-64 variants, writes
#      <out_dir>/<env>_residual_correlation_report.json.
#
# Usage on lab01 from repo root:
#   GPU_ID=0 bash experiments/scripts/csa_exp_c.sh wall \
#       --support-dir $JEPAWM_LOGS/csa_pilot/support_data \
#       --dump-dir   $JEPAWM_LOGS/csa_pilot/wall_official_jepa_wm \
#       --out-dir    $JEPAWM_LOGS/csa_pilot/residual
#
# Flags:
#   --support-dir DIR    where <env>_states.pt + <env>_actions.pt live (M_real)
#   --dump-dir    DIR    where csa_diag_dump.pt files live for this env
#   --out-dir     DIR    where M_resid tensors + the report get written
#   --skip-encode        skip phase 1 (re-use existing M_resid tensors)
#   --skip-analyze       skip phase 2 (encoding only)
#   --max-samples N      cap on val transitions to encode (default 100000)
#   --batch-size N       encode batch size (default 32)
#   --base-config PATH   override the env's default base eval YAML
#
# env must be one of: pusht | wall | maze | mw-reach | mw-reach-wall
#
# Does not touch existing dumps or diag/geometry reports; only adds
# <env>_residual_*.pt + <env>_residual_correlation_report.json under --out-dir.

source "$(dirname "$0")/_common.sh"

ENV="${1:-}"
if [ -z "$ENV" ]; then
  echo "ERROR: env tag required (pusht | wall | maze | mw-reach | mw-reach-wall)" >&2
  exit 1
fi
shift

# --- defaults
SUPPORT_DIR="${SUPPORT_DIR:-${JEPAWM_LOGS:-jepawm_logs}/csa_pilot/support_data}"
DUMP_DIR=""
OUT_DIR="${OUT_DIR:-${JEPAWM_LOGS:-jepawm_logs}/csa_pilot/residual}"
SKIP_ENCODE=0
SKIP_ANALYZE=0
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_CONFIG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --support-dir) SUPPORT_DIR="$2"; shift 2 ;;
    --dump-dir)    DUMP_DIR="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --skip-encode) SKIP_ENCODE=1; shift ;;
    --skip-analyze) SKIP_ANALYZE=1; shift ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    *) echo "ERROR: unknown flag '$1'" >&2; exit 1 ;;
  esac
done

if [ -z "$DUMP_DIR" ] && [ "$SKIP_ANALYZE" = "0" ]; then
  echo "ERROR: --dump-dir required for analyze phase (or pass --skip-analyze)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Resolve the per-env base eval config the same way make_official_csa_eval_config
# does. Stays in sync if that table changes.
declare -A ENV_BASE_CONFIG=(
  [pusht]="configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
  [wall]="configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
  [maze]="configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
  [mw-reach]="configs/evals/simu_env_planning/mw/jepa-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
  [mw-reach-wall]="configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
)
if [ -z "$BASE_CONFIG" ]; then
  BASE_CONFIG="${ENV_BASE_CONFIG[$ENV]:-}"
fi
if [ -z "$BASE_CONFIG" ]; then
  echo "ERROR: no base config known for env=${ENV}; pass --base-config <path>" >&2
  exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

RESID_STATES="${OUT_DIR}/${ENV}_residual_states.pt"
RESID_ACTIONS="${OUT_DIR}/${ENV}_residual_actions.pt"
RESID_ERRORS="${OUT_DIR}/${ENV}_residual_errors.pt"
SUPPORT_STATES="${SUPPORT_DIR}/${ENV}_states.pt"
SUPPORT_ACTIONS="${SUPPORT_DIR}/${ENV}_actions.pt"
REPORT="${OUT_DIR}/${ENV}_residual_correlation_report.json"

# --- Phase 1: encode M_resid -----------------------------------------------
if [ "$SKIP_ENCODE" = "0" ]; then
  echo ">>> [Exp C / phase 1] Encoding M_resid for env=${ENV}"
  echo "     eval config:   ${BASE_CONFIG}"
  echo "     output:        ${OUT_DIR}/${ENV}_residual_{states,actions,errors}.pt"
  python experiments/scripts/csa/encode_residual_data.py \
    --eval-config "$BASE_CONFIG" \
    --env "$ENV" \
    --output-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --device "cuda:${GPU_ID:-0}"
else
  echo ">>> [Exp C / phase 1] SKIPPED (--skip-encode)"
fi

# --- Phase 2: residual correlation analysis --------------------------------
if [ "$SKIP_ANALYZE" = "0" ]; then
  if [ ! -f "$SUPPORT_STATES" ] || [ ! -f "$SUPPORT_ACTIONS" ]; then
    echo "ERROR: M_real tensors missing under ${SUPPORT_DIR}:" >&2
    echo "        ${SUPPORT_STATES}" >&2
    echo "        ${SUPPORT_ACTIONS}" >&2
    echo "       Run encode_support_data.py + build_support_memory.py first" >&2
    exit 1
  fi
  if [ ! -f "$RESID_STATES" ] || [ ! -f "$RESID_ACTIONS" ] || [ ! -f "$RESID_ERRORS" ]; then
    echo "ERROR: M_resid tensors missing under ${OUT_DIR}:" >&2
    echo "        ${RESID_STATES}" >&2
    echo "        ${RESID_ACTIONS}" >&2
    echo "        ${RESID_ERRORS}" >&2
    exit 1
  fi
  echo ">>> [Exp C / phase 2] Running residual correlation for env=${ENV}"
  echo "     dump dir:      ${DUMP_DIR}"
  echo "     report:        ${REPORT}"
  python experiments/scripts/csa/run_residual_correlation.py \
    --env "$ENV" \
    --dump-dir "$DUMP_DIR" \
    --raw-support-states "$SUPPORT_STATES" \
    --raw-support-actions "$SUPPORT_ACTIONS" \
    --raw-residual-states "$RESID_STATES" \
    --raw-residual-actions "$RESID_ACTIONS" \
    --raw-residual-errors "$RESID_ERRORS" \
    --output "$REPORT"
else
  echo ">>> [Exp C / phase 2] SKIPPED (--skip-analyze)"
fi

echo ">>> Done for env=${ENV}"
