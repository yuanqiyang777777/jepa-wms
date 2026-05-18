#!/bin/bash
# Run short PushT profilers for DDP/NCCL scaling and knob sweeps.
#
# Defaults are intentionally short and protocol-neutral: all runs delegate to
# profile_step.sh, which caps training to one 240-iteration profiling epoch.
#
# Usage:
#   RUN_PREFIX=20260518_ddp_scaling MODE=scaling bash experiments/scripts/profile_ddp_matrix.sh
#   RUN_PREFIX=20260518_ddp_knobs   MODE=ddp     bash experiments/scripts/profile_ddp_matrix.sh
#   RUN_PREFIX=20260518_ddp_all     MODE=all     bash experiments/scripts/profile_ddp_matrix.sh

source "$(dirname "$0")/_common.sh"

RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M%S)_ddp_profile}"
MODE="${MODE:-scaling}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-200}"
PROFILE_IPE="${PROFILE_IPE:-240}"
PROFILE_CASE_CLEANUP_TIMEOUT_SEC="${PROFILE_CASE_CLEANUP_TIMEOUT_SEC:-180}"

PROFILE_STEP_SCRIPT="$(dirname "$0")/profile_step.sh"

wait_for_case_gpus() {
  local devices="$1"
  local deadline=$((SECONDS + PROFILE_CASE_CLEANUP_TIMEOUT_SEC))
  while true; do
    if check_gpus_free "$devices" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "WARNING: GPUs for case did not become fully free before timeout: $devices" >&2
      return 0
    fi
    sleep 5
  done
}

run_case() {
  local name="$1"
  local devices="$2"
  local static_graph="$3"
  local grad_bucket_view="$4"
  local broadcast_buffers="$5"
  local bucket_cap="$6"
  local comm_hook="$7"
  local nccl_algo="$8"

  local run_id="${RUN_PREFIX}_${name}"
  local run_dir="$JEPAWM_LOGS/$run_id"

  echo "" >&2
  echo "=== DDP profile case: $name ===" >&2
  echo "run_id=$run_id devices=$devices" >&2

  rm -f "$JEPAWM_LOGS/$run_id.nohup.log"
  mkdir -p "$run_dir"

  set +e
  (
    export RUN_ID="$run_id"
    export DEVICES="$devices"
    export BATCH_SIZE
    export PROFILE_WARMUP
    export PROFILE_STEPS
    export PROFILE_IPE
    export JEPAWM_DDP_STATIC_GRAPH="$static_graph"
    export JEPAWM_DDP_GRADIENT_AS_BUCKET_VIEW="$grad_bucket_view"
    export JEPAWM_DDP_BROADCAST_BUFFERS="$broadcast_buffers"
    export JEPAWM_DDP_COMM_HOOK="$comm_hook"
    if [ -n "$bucket_cap" ]; then
      export JEPAWM_DDP_BUCKET_CAP_MB="$bucket_cap"
    else
      unset JEPAWM_DDP_BUCKET_CAP_MB
    fi
    if [ -n "$nccl_algo" ]; then
      export NCCL_ALGO="$nccl_algo"
    else
      unset NCCL_ALGO
    fi
    bash "$PROFILE_STEP_SCRIPT"
  ) 2>&1 | tee "$JEPAWM_LOGS/$run_id.nohup.log"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  if ! grep -q "StepProfiler] mean per-step" "$run_dir/launch.log" 2>/dev/null; then
    echo "WARNING: $run_id exited with status $cmd_status and no profiler table was found" >&2
  elif [ "$cmd_status" -ne 0 ]; then
    echo "WARNING: $run_id exited with status $cmd_status after writing a profiler table" >&2
  fi
  wait_for_case_gpus "$devices"
}

run_scaling_cases() {
  run_case "scale_1gpu_baseline" "cuda:0" "0" "0" "1" "" "none" ""
  run_case "scale_2gpu_baseline" "cuda:0 cuda:1" "0" "0" "1" "" "none" ""
  run_case "scale_4gpu_baseline" "cuda:0 cuda:1 cuda:2 cuda:3" "0" "0" "1" "" "none" ""
  run_case "scale_6gpu_baseline" "cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5" "0" "0" "1" "" "none" ""
}

run_ddp_knob_cases() {
  local devices="${DDP_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5}"
  run_case "ddp_knobs_bucket_default" "$devices" "1" "1" "0" "" "none" ""
  run_case "ddp_knobs_bucket50" "$devices" "1" "1" "0" "50" "none" ""
  run_case "ddp_knobs_bucket100" "$devices" "1" "1" "0" "100" "none" ""
  run_case "ddp_knobs_bucket200" "$devices" "1" "1" "0" "200" "none" ""
  run_case "ddp_knobs_bucket100_tree" "$devices" "1" "1" "0" "100" "none" "Tree"
  run_case "ddp_knobs_bucket100_ring" "$devices" "1" "1" "0" "100" "none" "Ring"
  if [ "${INCLUDE_BF16_HOOK:-0}" = "1" ]; then
    run_case "ddp_knobs_bucket100_bf16_hook" "$devices" "1" "1" "0" "100" "bf16" ""
  fi
}

case "$MODE" in
  scaling)
    run_scaling_cases
    ;;
  ddp)
    run_ddp_knob_cases
    ;;
  all)
    run_scaling_cases
    run_ddp_knob_cases
    ;;
  *)
    echo "ERROR: MODE must be scaling, ddp, or all (got $MODE)" >&2
    exit 2
    ;;
esac

python - "$JEPAWM_LOGS" "$RUN_PREFIX" "$BATCH_SIZE" <<'PY'
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
run_prefix = sys.argv[2]
batch_size = int(sys.argv[3])

section_re = re.compile(r"^\s*(?P<name>[a-z_]+)\s+(?P<ms>[0-9.]+)\s+")
step_re = re.compile(r"^\s*measured step wall\s+(?P<ms>[0-9.]+)")


def parse_run(path: Path):
    log_path = path / "launch.log"
    if not log_path.exists():
        return None
    sections = {}
    step_wall = None
    in_table = False
    for line in log_path.read_text(errors="replace").splitlines():
        if "StepProfiler] mean per-step" in line:
            in_table = True
            continue
        if not in_table:
            continue
        match = section_re.match(line)
        if match:
            sections[match.group("name")] = float(match.group("ms"))
            continue
        match = step_re.match(line)
        if match:
            step_wall = float(match.group("ms"))
            break
    if step_wall is None or "backward" not in sections:
        return None
    config_path = path / "config.yaml"
    world_size = 0
    if config_path.exists():
        for line in config_path.read_text(errors="replace").splitlines():
            if line.startswith("folder:"):
                continue
    launcher = path / "launcher.sh"
    devices = ""
    if launcher.exists():
        # The copied launcher contains defaults, not the selected devices. Prefer nohup diagnostics below.
        pass
    nohup_path = path.with_suffix(".nohup.log")
    if nohup_path.exists():
        for line in nohup_path.read_text(errors="replace").splitlines():
            if line.startswith("DEVICES="):
                devices = line.split("=", 1)[1].split(" (world_size=", 1)[0].strip()
                world_size_match = re.search(r"\(world_size=(\d+)\)", line)
                if world_size_match:
                    world_size = int(world_size_match.group(1))
                break
    if world_size <= 0:
        world_size = max(1, len(devices.split()))
    throughput = world_size * batch_size / (step_wall / 1000.0)
    return {
        "name": path.name.replace(run_prefix + "_", ""),
        "world_size": world_size,
        "step_wall": step_wall,
        "backward": sections["backward"],
        "throughput": throughput,
    }


runs = []
for path in sorted(log_root.glob(f"{run_prefix}_*")):
    if path.is_dir():
        parsed = parse_run(path)
        if parsed is not None:
            runs.append(parsed)

baseline_1gpu = next((r for r in runs if r["world_size"] == 1 and "baseline" in r["name"]), None)
base_backward = baseline_1gpu["backward"] if baseline_1gpu is not None else None

print("")
print("=== DDP/NCCL profiler summary ===")
print(f"{'case':<32s} {'gpus':>4s} {'step_ms':>10s} {'back_ms':>10s} {'comm_est':>10s} {'samples/s':>10s}")
for run in runs:
    comm = run["backward"] - base_backward if base_backward is not None else float("nan")
    print(
        f"{run['name']:<32s} {run['world_size']:>4d} "
        f"{run['step_wall']:>10.3f} {run['backward']:>10.3f} "
        f"{comm:>10.3f} {run['throughput']:>10.2f}"
    )
if not runs:
    print("No complete StepProfiler tables found.")
PY
