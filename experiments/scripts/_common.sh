#!/bin/bash
# Common header for every lab01 launcher script.
# Source this at the top of any reproduce_*.sh / pilot_*.sh / eval_*.sh.
#
# Usage in a launcher:
#   #!/bin/bash
#   set -euo pipefail
#   source "$(dirname "$0")/_common.sh"
#   # ... your command(s) ...

set -euo pipefail

# Footgun 2: load env vars before activating conda
source ~/.jepawm_env

# Activate conda env
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms

# Footgun 1: cd to repo root before any python (avoid ~/datasets.py shadow)
cd "$JEPAWM_HOME/jepa-wms"

# Sanity: confirm we're in the right place and the right env
if [ ! -f "evals/main.py" ] && [ ! -f "app/main.py" ]; then
  echo "ERROR: not in jepa-wms repo root. CWD=$(pwd)" >&2
  exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "jepa-wms" ]; then
  echo "ERROR: conda env is not jepa-wms. CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}" >&2
  exit 1
fi

# Footgun 4: pick a GPU (default 0; override with GPU_ID env var)
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# lab01 L40 local P2P currently makes NCCL collectives hang; keep DDP on SHM/socket.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

device_ids() {
  local device
  for device in "$@"; do
    device="${device//,/ }"
    for token in $device; do
      token="${token#cuda:}"
      token="${token#gpu:}"
      if [ -n "$token" ]; then
        echo "$token"
      fi
    done
  done
}

check_gpus_free() {
  local threshold_mb="${GPU_MEM_THRESHOLD_MB:-500}"
  local util_threshold="${GPU_UTIL_THRESHOLD_PCT:-20}"
  local ids
  ids="$(device_ids "$@")"

  if [ -z "$ids" ]; then
    echo "ERROR: check_gpus_free called without devices" >&2
    return 2
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot verify GPU availability" >&2
    return 2
  fi

  local gpu_csv
  gpu_csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  local bad=0
  local id
  for id in $ids; do
    local row mem util
    row="$(printf '%s\n' "$gpu_csv" | awk -F',' -v target="$id" '$1 ~ "^[[:space:]]*" target "[[:space:]]*$" {print; exit}')"
    if [ -z "$row" ]; then
      echo "ERROR: GPU $id not found in nvidia-smi output" >&2
      bad=1
      continue
    fi
    mem="$(printf '%s' "$row" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
    util="$(printf '%s' "$row" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
    if [ "$mem" -gt "$threshold_mb" ] || [ "$util" -gt "$util_threshold" ]; then
      echo "ERROR: GPU $id is occupied: memory.used=${mem}MB utilization=${util}%" >&2
      bad=1
    else
      echo "GPU $id available: memory.used=${mem}MB utilization=${util}%" >&2
    fi
  done

  if [ "$bad" -ne 0 ]; then
    echo "Choose free GPUs from 0-5 or 7, then rerun with DEVICES=\"cuda:<id> ...\"." >&2
    return 1
  fi
}

# Useful diagnostics (echo to stderr so they don't pollute downstream pipelines)
{
  echo "=== launcher diagnostics ==="
  echo "host:        $(hostname)"
  echo "user:        $(whoami)"
  echo "cwd:         $(pwd)"
  echo "branch:      $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "commit:      $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "conda env:   ${CONDA_DEFAULT_ENV}"
  echo "python:      $(which python)"
  echo "GPU(s):      ${CUDA_VISIBLE_DEVICES}"
  echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}"
  echo "================================"
} >&2
