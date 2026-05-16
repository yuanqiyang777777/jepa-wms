#!/bin/bash
# Re-run the PushT JEPA-WM smoke test that originally validated the lab01 deployment.
# Expected: R ~= .27, S=0, T ~= 3s.
#
# Usage on lab01 from repo root:
#   bash experiments/scripts/smoke_pusht.sh
# Or with explicit GPU:
#   GPU_ID=2 bash experiments/scripts/smoke_pusht.sh
#
# This script lives in the fork at jepa-wms/experiments/scripts/ and is
# committed to the mfl branch so it is available on lab01 after git pull.

source "$(dirname "$0")/_common.sh"

SMOKE_YAML="${SMOKE_YAML:-/tmp/jepawm_smoke_pt/smoke.yaml}"

if [ ! -f "$SMOKE_YAML" ]; then
  echo "ERROR: smoke yaml not found at $SMOKE_YAML" >&2
  echo "Hint: this is the config from the original deployment smoke test." >&2
  echo "If lost, regenerate from configs/evals/simu_env_planning/pusht/jepa-wm/<base>.yaml + quick_debug=true overrides." >&2
  exit 1
fi

python -m evals.main \
  --fname "$SMOKE_YAML" \
  --debug \
  --devices "cuda:${GPU_ID:-0}"
