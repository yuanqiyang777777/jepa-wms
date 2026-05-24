#!/bin/bash
# Run residual_correlation for all 8 (env, seed) combinations from the
# diagnostic phase. Intentionally not part of csa_exp_c.sh -- this is the
# specific one-shot driver for the existing dump layout on lab01.
#
# Expected directory layout (set via env vars):
#   SUPPORT_DIR_PUSHT  = /tmp/csa_pusht
#   SUPPORT_DIR_WALL   = /tmp/csa_wall
#   SUPPORT_DIR_MAZE   = /tmp/csa_maze
#   SUPPORT_DIR_MW     = /tmp/csa_metaworld   # shared by mw-reach + mw-reach-wall
#   RESID_DIR          = /tmp/csa_residual    # output of encode_residual_data.py
#   DUMP_ROOT          = $JEPAWM_LOGS/csa_pilot
#   OUT_DIR            = $JEPAWM_LOGS/csa_pilot/residual  (where reports land)
set -euo pipefail

SUPPORT_DIR_PUSHT="${SUPPORT_DIR_PUSHT:-/tmp/csa_pusht}"
SUPPORT_DIR_WALL="${SUPPORT_DIR_WALL:-/tmp/csa_wall}"
SUPPORT_DIR_MAZE="${SUPPORT_DIR_MAZE:-/tmp/csa_maze}"
SUPPORT_DIR_MW="${SUPPORT_DIR_MW:-/tmp/csa_metaworld}"
RESID_DIR="${RESID_DIR:-/tmp/csa_residual}"
DUMP_ROOT="${DUMP_ROOT:-${JEPAWM_LOGS}/csa_pilot}"
OUT_DIR="${OUT_DIR:-${JEPAWM_LOGS}/csa_pilot/residual}"

mkdir -p "$OUT_DIR"

# env_seed | env_tag | dump_dir | M_real prefix | M_resid prefix
RUNS=(
  "pusht_seed1|pusht|${DUMP_ROOT}/pusht_official_jepa_wm/simu_env_planning/csa_diag/full/pusht-base|${SUPPORT_DIR_PUSHT}/pusht|${RESID_DIR}/pusht_residual"
  "wall_seed1|wall|${DUMP_ROOT}/wall_official_jepa_wm/simu_env_planning/csa_diag/full/wall-base|${SUPPORT_DIR_WALL}/wall|${RESID_DIR}/wall_residual"
  "wall_seed2|wall|${DUMP_ROOT}/wall_seed2_official_jepa_wm/simu_env_planning/csa_diag/full/wall-base|${SUPPORT_DIR_WALL}/wall|${RESID_DIR}/wall_residual"
  "maze_seed1|maze|${DUMP_ROOT}/maze_official_jepa_wm/simu_env_planning/csa_diag/full/maze-base|${SUPPORT_DIR_MAZE}/maze|${RESID_DIR}/maze_residual"
  "mw-reach_seed1|mw-reach|${DUMP_ROOT}/mw-reach_official_jepa_wm/simu_env_planning/csa_diag/full/mw-reach|${SUPPORT_DIR_MW}/metaworld|${RESID_DIR}/metaworld_residual"
  "mw-reach_seed2|mw-reach|${DUMP_ROOT}/mw-reach_seed2_official_jepa_wm/simu_env_planning/csa_diag/full/mw-reach|${SUPPORT_DIR_MW}/metaworld|${RESID_DIR}/metaworld_residual"
  "mw-reach-wall_seed1|mw-reach-wall|${DUMP_ROOT}/mw-reach-wall_official_jepa_wm/simu_env_planning/csa_diag/full/mw-reach-wall|${SUPPORT_DIR_MW}/metaworld|${RESID_DIR}/metaworld_residual"
  "mw-reach-wall_seed2|mw-reach-wall|${DUMP_ROOT}/mw-reach-wall_seed2_official_jepa_wm/simu_env_planning/csa_diag/full/mw-reach-wall|${SUPPORT_DIR_MW}/metaworld|${RESID_DIR}/metaworld_residual"
)

for line in "${RUNS[@]}"; do
  IFS='|' read -r ES ENV DUMP_DIR SUPP_PREFIX RESID_PREFIX <<< "$line"
  REPORT="${OUT_DIR}/${ES}_residual_correlation_report.json"
  echo ""
  echo ">>> Running ${ES} (env=${ENV})"
  echo "    dump dir : ${DUMP_DIR}"
  echo "    M_real   : ${SUPP_PREFIX}_{states,actions}.pt"
  echo "    M_resid  : ${RESID_PREFIX}_{states,actions,errors}.pt"
  echo "    out      : ${REPORT}"
  python experiments/scripts/csa/run_residual_correlation.py \
    --env "$ENV" \
    --dump-dir "$DUMP_DIR" \
    --raw-support-states  "${SUPP_PREFIX}_states.pt" \
    --raw-support-actions "${SUPP_PREFIX}_actions.pt" \
    --raw-residual-states  "${RESID_PREFIX}_states.pt" \
    --raw-residual-actions "${RESID_PREFIX}_actions.pt" \
    --raw-residual-errors  "${RESID_PREFIX}_errors.pt" \
    --output "$REPORT"
done

echo ""
echo ">>> All 8 reports written to ${OUT_DIR}/"
ls -la "$OUT_DIR"/*_residual_correlation_report.json
