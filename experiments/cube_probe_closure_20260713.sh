#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

BASE=/home/ps/Code/yqy/DINO-WM/stablewm_home
CODE=/home/ps/Code/yqy/DINO-WM
RUN_ID=adadim_cube_probe_closure_20260713
ROOT="$BASE/$RUN_ID"
SOURCE_JEPA="$CODE/jepa-wms"
SOURCE_STABLE="$CODE/stable-worldmodel"
JEPA_WT="$CODE/wt_cube_probe_closure_jepawms_20260713"
SWM_WT_REUSE="$CODE/wt_probe_f2compat_stablewm_20260713"
SWM_WT_ALT="$CODE/wt_cube_probe_closure_stablewm_20260713"
JEPA_BASE_COMMIT=7dbbf8277a3f4c9071ba9d1eb0e11f23c8657f4d
STABLE_COMMIT=314201d0347baf61304c7558a98463c6a01080f0
EXPECTED_BRANCH=exp/20260713-cube-probe-closure

PY="$SOURCE_STABLE/.venv/bin/python"
F2="$BASE/latent_dimension_interference_20260702/probe_planning_link_20260703"
SOURCE_PROBE="$F2/scripts_v2_fenced/probe_physical_cost_cem_adadim_block_20260703.py"
SOURCE_COMMON="$F2/scripts_v2_fenced/candidate_ranking_alignment_adadim_block_20260703.py"
SOURCE_MANIFEST="$F2/mainarm_manifest_20260704.json"
SOURCE_BUDGET="$SOURCE_STABLE/experiments/scripts/adadim_budget_sweep_runner.py"
SOURCE_CACHE="$JEPA_WT/experiments/scripts/cube_probe_cache_20260713.py"
SOURCE_LAUNCHER="$JEPA_WT/experiments/cube_probe_closure_20260713.sh"

EXPECTED_PROBE=564d7469840a38bf6015149d9d61191694beef9e0f5ed6610c5db9934b7dbdc8
EXPECTED_COMMON=9c57a82343277e3378f24df8ca38e5f9a4dcc53a4fead909e2a6126001d1a89c
EXPECTED_MANIFEST=74e723c932c0320c5bde2117d496ad4f0c2e6a14c0163e69486d9e5d2ccd8c2f
EXPECTED_BUDGET=cefd2b267601377aa1201a5302828807f4189ca11f123c1128746938f51dfd8f
EXPECTED_ANALYZER_SHA256=074c867afe56f8a68ebf76465ccb46661baf033fb9f9328915ca041600d250f8

PROV="$ROOT/provenance"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
COMMANDS="$ROOT/commands"
PROBE="$PROV/probe_physical_cost_cem_adadim_block_20260703.py"
COMMON="$PROV/candidate_ranking_alignment_adadim_block_20260703.py"
MANIFEST="$PROV/mainarm_manifest_20260704.json"
BUDGET="$PROV/adadim_budget_sweep_runner_f2compat.py"
CACHE="$PROV/cube_probe_cache_20260713.py"
LAUNCHER="$PROV/cube_probe_closure_20260713.sh"
ANALYZER="$PROV/analyze_cube_probe_closure_20260713.py"
DATASET="$BASE/datasets/ogbench/cube_single_expert.h5"


die() {
  echo "ERROR: $*" >&2
  exit 1
}


sha_of() {
  sha256sum "$1" | awk '{print $1}'
}


require_hash() {
  local file=$1
  local expected=$2
  local actual
  [[ -f "$file" ]] || die "missing file: $file"
  actual=$(sha_of "$file")
  [[ "$actual" == "$expected" ]] || die "hash mismatch: $file expected=$expected actual=$actual"
}


try_atomic_text() {
  local path=$1
  local value=$2
  local tmp="${path}.tmp.$$"
  [[ ! -e "$path" && ! -e "$tmp" ]] || return 1
  if ! printf '%s\n' "$value" > "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  if ! ln "$tmp" "$path"; then
    rm -f "$tmp"
    return 1
  fi
  rm -f "$tmp"
  return 0
}


atomic_text() {
  local path=$1
  local value=$2
  try_atomic_text "$path" "$value" \
    || die "atomic no-replace publish failed: $path"
}


publish_file() {
  local tmp=$1
  local path=$2
  [[ -f "$tmp" ]] || die "missing file to publish: $tmp"
  [[ ! -e "$path" ]] || die "refusing to overwrite immutable file: $path"
  if ! ln "$tmp" "$path"; then
    rm -f "$tmp"
    die "atomic no-replace publish failed: $path"
  fi
  rm -f "$tmp"
}


reserve_file() {
  local path=$1
  [[ ! -e "$path" ]] || die "refusing to overwrite reserved file: $path"
  ( set -o noclobber; : > "$path" ) 2>/dev/null \
    || die "exclusive file reservation failed: $path"
}


write_command() {
  local path=$1
  shift
  local tmp="${path}.tmp.$$"
  [[ ! -e "$path" && ! -e "$tmp" ]] || die "command snapshot collision: $path"
  printf '%q ' "$@" > "$tmp"
  printf '\n' >> "$tmp"
  publish_file "$tmp" "$path"
}


run_locked() {
  local fn=$1
  shift
  [[ -d "$STATE" ]] || die "run state directory is absent; setup must complete first"
  exec 9>>"$STATE/.mutation.lock"
  flock -n 9 || die "another launcher mutation is active"
  "$fn" "$@"
  local rc=$?
  flock -u 9
  exec 9>&-
  return "$rc"
}


iso_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}


require_source_hashes() {
  require_hash "$SOURCE_PROBE" "$EXPECTED_PROBE"
  require_hash "$SOURCE_COMMON" "$EXPECTED_COMMON"
  require_hash "$SOURCE_MANIFEST" "$EXPECTED_MANIFEST"
  require_hash "$SOURCE_BUDGET" "$EXPECTED_BUDGET"
}


require_checkpoints() {
  local seed expected path
  for seed in 42 43 44; do
    path="$BASE/checkpoints/adadim_cube_matched3_lbproj_seed${seed}_20260615/weights_epoch_10.pt"
    case "$seed" in
      42) expected=81f2b3023f57299cb2d0504ec168bdf92d827d0d54b65a05718b4c9b492ba3e7 ;;
      43) expected=0ccce2664439ac5a286bfcfd485f30d78c41e0d54de181de717e09657b510b0d ;;
      44) expected=c16a351e13b294c809ca6ebe4ac00278287ac5e0e97be68d0b5980851fe1600d ;;
    esac
    require_hash "$path" "$expected"
  done
  [[ -f "$DATASET" ]] || die "missing Cube dataset"
}


require_dataset_identity() {
  [[ -f "$PROV/cube_dataset_sha256.txt" ]] || die "dataset hash was not captured"
  local expected actual
  expected=$(<"$PROV/cube_dataset_sha256.txt")
  actual=$(sha_of "$DATASET")
  [[ "$actual" == "$expected" ]] || die "Cube dataset hash drift: expected=$expected actual=$actual"
}


require_dataset_stat() {
  [[ -f "$PROV/cube_dataset_stat.txt" ]] || die "dataset stat was not captured"
  local current
  current=$(stat -c '%y %s %n' "$DATASET")
  [[ "$current" == "$(<"$PROV/cube_dataset_stat.txt")" ]] || die "Cube dataset stat drift"
}


require_jepa_branch() {
  local top branch status_count diff_count diff_status diff_path
  [[ -e "$JEPA_WT/.git" ]] || die "missing execution worktree: $JEPA_WT"
  top=$(git -C "$JEPA_WT" rev-parse --show-toplevel)
  [[ "$(realpath "$top")" == "$(realpath "$JEPA_WT")" ]] || die "unexpected jepa worktree root: $top"
  branch=$(git -C "$JEPA_WT" branch --show-current)
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || die "wrong experiment branch: $branch"
  git -C "$JEPA_WT" merge-base --is-ancestor "$JEPA_BASE_COMMIT" HEAD || die "branch is not based on $JEPA_BASE_COMMIT"
  status_count=$(git -C "$JEPA_WT" status --porcelain | wc -l)
  [[ "$status_count" -eq 0 ]] || die "jepa execution worktree is dirty"
  diff_count=0
  while IFS=$'\t' read -r diff_status diff_path; do
    [[ -n "$diff_status" ]] || continue
    [[ "$diff_status" == A ]] || die "branch modifies a tracked file: $diff_status $diff_path"
    case "$diff_path" in
      experiments/cube_probe_closure_20260713.sh|experiments/scripts/cube_probe_cache_20260713.py)
        ;;
      *) die "branch contains unauthorized file: $diff_path" ;;
    esac
    diff_count=$((diff_count + 1))
  done < <(git -C "$JEPA_WT" diff --name-status "$JEPA_BASE_COMMIT"..HEAD)
  [[ "$diff_count" -eq 2 ]] || die "expected exactly two added experiment files, found $diff_count"
}


select_stable_worktree() {
  local selected
  if [[ -e "$SWM_WT_REUSE/.git" ]] \
    && [[ "$(git -C "$SWM_WT_REUSE" rev-parse HEAD)" == "$STABLE_COMMIT" ]] \
    && [[ -z "$(git -C "$SWM_WT_REUSE" status --porcelain)" ]]; then
    selected=$SWM_WT_REUSE
  else
    [[ ! -e "$SWM_WT_ALT" ]] || die "alternate stable worktree collision: $SWM_WT_ALT"
    git -C "$SOURCE_STABLE" worktree add --detach "$SWM_WT_ALT" "$STABLE_COMMIT" >&2
    selected=$SWM_WT_ALT
  fi
  printf '%s\n' "$selected"
}


require_stable_worktree() {
  local selected
  [[ -f "$PROV/stable_worktree_path.txt" ]] || die "stable worktree path was not captured"
  selected=$(<"$PROV/stable_worktree_path.txt")
  [[ "$selected" == "$SWM_WT_REUSE" || "$selected" == "$SWM_WT_ALT" ]] || die "unexpected stable worktree path: $selected"
  [[ -e "$selected/.git" ]] || die "missing stable worktree: $selected"
  [[ "$(git -C "$selected" rev-parse HEAD)" == "$STABLE_COMMIT" ]] || die "stable worktree commit drift"
  [[ -z "$(git -C "$selected" status --porcelain)" ]] || die "stable worktree is dirty"
}


stable_worktree() {
  cat "$PROV/stable_worktree_path.txt"
}


require_snapshot_hashes() {
  require_hash "$PROBE" "$EXPECTED_PROBE"
  require_hash "$COMMON" "$EXPECTED_COMMON"
  require_hash "$MANIFEST" "$EXPECTED_MANIFEST"
  require_hash "$BUDGET" "$EXPECTED_BUDGET"
  [[ "$(sha_of "$CACHE")" == "$(<"$PROV/cache_script_sha256.txt")" ]] || die "cache-script snapshot drift"
  [[ "$(sha_of "$LAUNCHER")" == "$(<"$PROV/launcher_sha256.txt")" ]] || die "launcher snapshot drift"
}


require_analyzer_snapshot() {
  [[ -f "$PROV/analyzer_sha256.txt" ]] || die "analyzer has not been registered"
  [[ "$(<"$PROV/analyzer_sha256.txt")" == "$EXPECTED_ANALYZER_SHA256" ]] \
    || die "registered analyzer hash differs from the frozen review hash"
  require_hash "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
}


require_smoke_attestation() {
  local attestation="$ROOT/smoke_validation_attestation.json"
  [[ -f "$attestation" && -f "$ROOT/smoke_validated.json" ]] \
    || die "smoke validation attestation is absent"
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    "$PY" - "$attestation" "$ROOT/smoke_validated.json" \
      "$ROOT/cache_validated.json" "$PROV/analyzer_sha256.txt" \
      "$PROV/cube_dataset_sha256.txt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

attestation, smoke, cache, analyzer_hash, dataset_hash = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
actual = json.loads(attestation.read_text(encoding="utf-8"))
expected = {
    "run_id": "adadim_cube_probe_closure_20260713",
    "state": "SMOKE_VALIDATION_ATTESTED",
    "smoke_validated_sha256": sha(smoke),
    "cache_validated_sha256": sha(cache),
    "analyzer_sha256": analyzer_hash.read_text(encoding="utf-8").strip(),
    "dataset_sha256": dataset_hash.read_text(encoding="utf-8").strip(),
}
if actual != expected:
    raise SystemExit(f"smoke validation attestation drift: {actual} != {expected}")
print("smoke_attestation_ok")
PY
  )
}


require_setup() {
  [[ -f "$STATE/setup.done.json" ]] || die "setup has not completed"
  require_jepa_branch
  [[ "$(git -C "$JEPA_WT" rev-parse HEAD)" == "$(<"$PROV/jepawms_git_commit.txt")" ]] || die "reviewed experiment commit drift"
  require_stable_worktree
  require_snapshot_hashes
  require_source_hashes
  require_checkpoints
  require_dataset_stat
}


setup_run() {
  source ~/.jepawm_env
  require_source_hashes
  require_checkpoints
  require_jepa_branch
  [[ -x "$PY" ]] || die "missing Python runtime: $PY"
  [[ -f "$SOURCE_CACHE" && -f "$SOURCE_LAUNCHER" ]] || die "missing reviewed experiment sources"
  mkdir "$ROOT" || die "run-id collision or atomic setup claim failed: $ROOT"

  local setup_ok=0
  setup_failure_trap() {
    local rc=$?
    trap - EXIT
    if [[ "$setup_ok" -ne 1 && -d "$STATE" && ! -e "$STATE/setup.failed.json" ]]; then
      set +e
      try_atomic_text "$STATE/setup.failed.json" \
        "{\"run_id\":\"$RUN_ID\",\"state\":\"SETUP_FAILED\",\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap setup_failure_trap EXIT

  mkdir -p "$PROV" "$STATE" "$LOGS" "$COMMANDS" "$ROOT/smoke" "$ROOT/full"
  local selected_swm
  selected_swm=$(select_stable_worktree)
  printf '%s\n' "$selected_swm" > "$PROV/stable_worktree_path.txt"

  cp --preserve=mode,timestamps "$SOURCE_PROBE" "$PROBE"
  cp --preserve=mode,timestamps "$SOURCE_COMMON" "$COMMON"
  cp --preserve=mode,timestamps "$SOURCE_MANIFEST" "$MANIFEST"
  cp --preserve=mode,timestamps "$SOURCE_BUDGET" "$BUDGET"
  cp --preserve=mode,timestamps "$SOURCE_CACHE" "$CACHE"
  cp --preserve=mode,timestamps "$SOURCE_LAUNCHER" "$LAUNCHER"

  git -C "$JEPA_WT" rev-parse HEAD > "$PROV/jepawms_git_commit.txt"
  git -C "$JEPA_WT" describe --always --dirty > "$PROV/jepawms_git_describe.txt"
  git -C "$JEPA_WT" status --porcelain > "$PROV/jepawms_git_status_porcelain.txt"
  git -C "$selected_swm" rev-parse HEAD > "$PROV/stableworldmodel_git_commit.txt"
  git -C "$selected_swm" describe --always --dirty > "$PROV/stableworldmodel_git_describe.txt"
  git -C "$selected_swm" status --porcelain > "$PROV/stableworldmodel_git_status_porcelain.txt"
  sha_of "$CACHE" > "$PROV/cache_script_sha256.txt"
  sha_of "$LAUNCHER" > "$PROV/launcher_sha256.txt"
  sha256sum "$PROBE" "$COMMON" "$MANIFEST" "$BUDGET" "$CACHE" "$LAUNCHER" > "$PROV/source_sha256.txt"
  stat -c '%y %s %n' "$DATASET" > "$PROV/cube_dataset_stat.txt"
  sha_of "$DATASET" > "$PROV/cube_dataset_sha256.txt"

  cd "$JEPA_WT"
  "$PY" -m py_compile "$PROBE" "$COMMON" "$BUDGET" "$CACHE"
  PYTHONPATH="$selected_swm${PYTHONPATH:+:$PYTHONPATH}" \
    ADADIM_STABLEWM_REPO="$selected_swm" \
    "$PY" "$CACHE" --help > "$PROV/cache_help.txt"
  atomic_text "$STATE/setup.done.json" \
    "{\"run_id\":\"$RUN_ID\",\"state\":\"SETUP_DONE\",\"at\":\"$(iso_now)\",\"jepa_commit\":\"$(<"$PROV/jepawms_git_commit.txt")\",\"stable_commit\":\"$STABLE_COMMIT\"}"
  setup_ok=1
  trap - EXIT
  echo "setup_ok root=$ROOT stable_worktree=$selected_swm"
}


register_analyzer() {
  source ~/.jepawm_env
  require_setup
  local source_path=$1
  [[ -f "$source_path" ]] || die "missing analyzer upload: $source_path"
  require_hash "$source_path" "$EXPECTED_ANALYZER_SHA256"
  [[ ! -e "$ANALYZER" && ! -e "$PROV/analyzer_sha256.txt" ]] \
    || die "analyzer was already registered"
  local tmp="$ANALYZER.tmp.$$"
  cp --preserve=mode,timestamps "$source_path" "$tmp"
  require_hash "$tmp" "$EXPECTED_ANALYZER_SHA256"
  publish_file "$tmp" "$ANALYZER"
  atomic_text "$PROV/analyzer_sha256.txt" "$EXPECTED_ANALYZER_SHA256"
  require_analyzer_snapshot
  echo "analyzer_registered sha256=$EXPECTED_ANALYZER_SHA256"
}


gpu_free() {
  local gpu=$1
  local values mem util
  [[ "$gpu" != 6 ]] || die "GPU 6 is forbidden"
  values=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  IFS=, read -r mem util <<< "$values"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || die "cannot parse GPU $gpu state: $values"
  [[ "$mem" -le 500 && "$util" -le 20 ]] || die "GPU $gpu is occupied: memory=${mem}MiB util=${util}%"
}


boot_id() {
  cat /proc/sys/kernel/random/boot_id
}


proc_start_ticks() {
  local pid=$1 stat rest
  [[ -r "/proc/$pid/stat" ]] || return 1
  stat=$(<"/proc/$pid/stat")
  rest=${stat##*) }
  set -- $rest
  [[ $# -ge 20 ]] || return 1
  printf '%s\n' "${20}"
}


worker_exit_trap() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ -n "${WORKER_CHILD_PID:-}" ]]; then
    stop_worker_child TERM
  fi
  if [[ "$rc" -ne 0 && -n "${WORKER_FAILED_MARKER:-}" \
        && ! -e "${WORKER_DONE_MARKER:-/nonexistent}" \
        && ! -e "$WORKER_FAILED_MARKER" ]]; then
    set +e
    try_atomic_text "$WORKER_FAILED_MARKER" \
      "{${WORKER_FAILURE_FIELDS:-\"state\":\"WORKER_FAILED\"},\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" \
      >/dev/null 2>&1
  fi
  exit "$rc"
}


worker_child_matches() {
  local pid=${WORKER_CHILD_PID:-}
  [[ -n "$pid" && -r "/proc/$pid/stat" \
      && "$(proc_start_ticks "$pid" 2>/dev/null || true)" == "${WORKER_CHILD_TICKS:-missing}" ]]
}


worker_group_alive() {
  local pid=${WORKER_CHILD_PID:-}
  [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null
}


stop_worker_child() {
  local signal_name=$1
  local pid=${WORKER_CHILD_PID:-}
  [[ -n "$pid" ]] || return 0
  if worker_child_matches; then
    kill -s "$signal_name" -- "-$pid" 2>/dev/null || true
    local attempt
    for attempt in {1..50}; do
      worker_group_alive || break
      sleep 0.2
    done
    if worker_group_alive; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
  WORKER_CHILD_PID=
  WORKER_CHILD_TICKS=
}


terminate_worker_child() {
  local signal_name=$1
  local exit_code=$2
  trap - HUP INT TERM
  stop_worker_child "$signal_name"
  exit "$exit_code"
}


arm_worker_traps() {
  trap worker_exit_trap EXIT
  trap 'terminate_worker_child HUP 129' HUP
  trap 'terminate_worker_child INT 130' INT
  trap 'terminate_worker_child TERM 143' TERM
}


run_cache_internal() {
  source ~/.jepawm_env
  [[ -f "$STATE/cache.attempted.json" ]] || die "cache worker lacks an immutable launch claim"
  [[ -n "${LAUNCH_TOKEN:-}" && -f "$STATE/cache.launch.token" \
      && "$LAUNCH_TOKEN" == "$(<"$STATE/cache.launch.token")" ]] \
    || die "cache worker launch token mismatch"
  WORKER_DONE_MARKER="$STATE/cache.done.json"
  WORKER_FAILED_MARKER="$STATE/cache.failed.json"
  WORKER_FAILURE_FIELDS='"state":"CACHE_FAILED","gpu":0,"pid":'"$$"
  arm_worker_traps
  require_setup
  require_dataset_stat
  [[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || die "cache fit must use physical GPU 0"
  [[ ! -e "$STATE/cache.running.json" && ! -e "$STATE/cache.done.json" && ! -e "$STATE/cache.failed.json" ]] || die "cache state already exists"
  local self_boot self_ticks
  self_boot=$(boot_id)
  self_ticks=$(proc_start_ticks $$)
  atomic_text "$STATE/cache.running.json" \
    "{\"state\":\"CACHE_RUNNING\",\"pid\":$$,\"gpu\":0,\"boot_id\":\"$self_boot\",\"start_ticks\":\"$self_ticks\",\"launch_token\":\"$LAUNCH_TOKEN\",\"at\":\"$(iso_now)\"}"
  local swm rc
  swm=$(stable_worktree)
  cd "$JEPA_WT"
  write_command "$COMMANDS/cache_fit.command.txt" "$PY" "$CACHE" fit-cache \
    --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
    --task-manifest "$MANIFEST" --run-root "$ROOT" --device cuda
  set +e
  PYTHONPATH="$swm${PYTHONPATH:+:$PYTHONPATH}" ADADIM_STABLEWM_REPO="$swm" WANDB_MODE=disabled \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    setsid "$PY" "$CACHE" fit-cache \
      --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
      --task-manifest "$MANIFEST" --run-root "$ROOT" --device cuda &
  WORKER_CHILD_PID=$!
  WORKER_CHILD_TICKS=$(proc_start_ticks "$WORKER_CHILD_PID")
  wait "$WORKER_CHILD_PID"
  rc=$?
  WORKER_CHILD_PID=
  WORKER_CHILD_TICKS=
  set -e
  if [[ "$rc" -eq 0 && -f "$ROOT/probes/probe_fit_summary.json" ]]; then
    atomic_text "$STATE/cache.done.json" \
      "{\"state\":\"CACHE_DONE\",\"pid\":$$,\"gpu\":0,\"at\":\"$(iso_now)\"}"
    exit 0
  fi
  [[ "$rc" -ne 0 ]] || rc=92
  atomic_text "$STATE/cache.failed.json" \
    "{\"state\":\"CACHE_FAILED\",\"pid\":$$,\"gpu\":0,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}"
  exit "$rc"
}


start_cache() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  [[ ! -e "$ROOT/probe_caches" && ! -e "$ROOT/probes" ]] || die "cache output collision"
  [[ ! -e "$STATE/cache.attempted.json" && ! -e "$STATE/cache.running.json" \
      && ! -e "$STATE/cache.done.json" && ! -e "$STATE/cache.failed.json" ]] \
    || die "cache batch already attempted"
  [[ ! -e "$ROOT/cache_pid.txt" ]] || die "cache pid file already exists"
  [[ ! -e "$STATE/cache.launch.token" ]] || die "cache launch token already exists"
  [[ ! -e "$STATE/cache.launch.record.json" ]] || die "cache launch record already exists"
  [[ ! -e "$STATE/cache.launched.json" && ! -e "$STATE/cache.launch.failed.json" ]] \
    || die "cache parent launch state already exists"
  gpu_free 0
  local launch_token launch_complete=0 pid=0 worker_boot=unknown worker_ticks=unknown
  cache_launch_trap() {
    local rc=$?
    trap - EXIT
    if [[ "$launch_complete" -ne 1 && ! -e "$STATE/cache.launch.failed.json" ]]; then
      set +e
      try_atomic_text "$STATE/cache.launch.failed.json" \
        "{\"state\":\"CACHE_LAUNCH_FAILED\",\"pid\":$pid,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap cache_launch_trap EXIT
  launch_token=$(< /proc/sys/kernel/random/uuid)
  atomic_text "$STATE/cache.launch.token" "$launch_token"
  atomic_text "$STATE/cache.attempted.json" \
    "{\"state\":\"CACHE_ATTEMPTED\",\"gpu\":0,\"launch_token\":\"$launch_token\",\"dataset_sha256\":\"$(<"$PROV/cube_dataset_sha256.txt")\",\"at\":\"$(iso_now)\"}"
  reserve_file "$LOGS/cache_fit.log"
  CUDA_VISIBLE_DEVICES=0 LAUNCH_TOKEN="$launch_token" setsid nohup bash "$LAUNCHER" run-cache 9>&- >> "$LOGS/cache_fit.log" 2>&1 &
  pid=$!
  worker_boot=$(boot_id)
  worker_ticks=$(proc_start_ticks "$pid") || worker_ticks=unavailable
  atomic_text "$ROOT/cache_pid.txt" "$pid $worker_boot $worker_ticks"
  atomic_text "$STATE/cache.launch.record.json" \
    "{\"state\":\"CACHE_LAUNCH_RECORD\",\"gpu\":0,\"pid\":$pid,\"boot_id\":\"$worker_boot\",\"start_ticks\":\"$worker_ticks\",\"launch_token\":\"$launch_token\",\"at\":\"$(iso_now)\"}"
  atomic_text "$STATE/cache.launched.json" \
    "{\"state\":\"CACHE_LAUNCHED\",\"gpu\":0,\"pid\":$pid,\"at\":\"$(iso_now)\"}"
  launch_complete=1
  trap - EXIT
  echo "cache_started gpu=0 pid=$pid"
}


validate_cache() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  [[ -f "$STATE/cache.attempted.json" \
      && -f "$STATE/cache.launch.record.json" \
      && -f "$STATE/cache.launched.json" \
      && -f "$ROOT/cache_pid.txt" ]] \
    || die "cache launch ledger is incomplete"
  [[ ! -e "$STATE/cache.launch.failed.json" ]] || die "cache parent launch failed"
  [[ -f "$STATE/cache.done.json" ]] || die "cache process is not done"
  [[ ! -e "$STATE/cache.failed.json" ]] || die "cache process failed"
  [[ ! -e "$ROOT/cache_validated.json" ]] || die "cache was already validated"
  cd "$JEPA_WT"
  "$PY" - "$CACHE" "$ROOT" create <<'PY'
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
root = Path(sys.argv[2])
mode = sys.argv[3]
state = root / "state"
attempted = json.loads((state / "cache.attempted.json").read_text(encoding="utf-8"))
record = json.loads((state / "cache.launch.record.json").read_text(encoding="utf-8"))
launched = json.loads((state / "cache.launched.json").read_text(encoding="utf-8"))
running = json.loads((state / "cache.running.json").read_text(encoding="utf-8"))
done = json.loads((state / "cache.done.json").read_text(encoding="utf-8"))
pid_fields = (root / "cache_pid.txt").read_text(encoding="utf-8").split()
if len(pid_fields) != 3:
    raise SystemExit(f"cache pid ledger malformed: {pid_fields}")
pid, boot, ticks = pid_fields
if (
    attempted.get("state") != "CACHE_ATTEMPTED"
    or record.get("state") != "CACHE_LAUNCH_RECORD"
    or launched.get("state") != "CACHE_LAUNCHED"
    or attempted.get("launch_token") != record.get("launch_token")
    or int(record.get("gpu", -1)) != 0
    or str(record.get("pid")) != pid
    or str(record.get("boot_id")) != boot
    or str(record.get("start_ticks")) != ticks
    or int(launched.get("pid", -1)) != int(pid)
    or int(running.get("pid", -1)) != int(pid)
    or int(done.get("pid", -1)) != int(pid)
    or running.get("launch_token") != attempted.get("launch_token")
):
    raise SystemExit("cache attempted/record/launched/pid ledgers disagree")
spec = importlib.util.spec_from_file_location("cube_probe_cache_reviewed", cache_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summary_path = root / "probes" / "probe_fit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("schema_version") != module.SCHEMA_VERSION or summary.get("run_id") != module.RUN_ID:
    raise SystemExit("fit summary schema/run id drift")
expected_sources = {
    "probe_runner": module.PROBE_RUNNER_SHA256,
    "probe_common": module.PROBE_COMMON_SHA256,
    "budget_runner": module.BUDGET_RUNNER_SHA256,
    "task_manifest": module.TASK_MANIFEST_SHA256,
}
if summary.get("source_sha256") != expected_sources:
    raise SystemExit("fit summary source hashes drift")
if summary.get("checkpoint_sha256") != {str(k): v for k, v in module.CHECKPOINT_SHA256.items()}:
    raise SystemExit("fit summary checkpoint hashes drift")
records = summary.get("records", [])
if len(records) != 45 or summary.get("n_cache_artifacts") != 45:
    raise SystemExit("expected exactly 45 immutable cache records")
keys = {(r["family"], int(r["train_seed"]), int(r["eval_seed"])) for r in records}
expected = {(f, ts, es) for f in ("p0", "p1", "p2") for ts in (42,43,44) for es in (42,43,44,45,46)}
if keys != expected:
    raise SystemExit("cache pair matrix mismatch")
artifact_map = {}
for record in records:
    rel = Path(record["path"])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "probe_caches":
        raise SystemExit(f"unsafe cache path: {rel}")
    meta, _ = module.load_npz_cache(root / rel, expected_sha256=record["sha256"])
    module._validate_loaded_metadata(meta, family=record["family"], train_seed=int(record["train_seed"]), eval_seed=int(record["eval_seed"]))
    artifact_map[str(rel)] = record["sha256"]
if len(artifact_map) != 45 or summary.get("cache_artifact_sha256") != artifact_map:
    raise SystemExit("cache artifact hash map differs from the 45 records")
gate = module.recompute_strict_p1_gate(root, summary)
barred = root / "probes" / "B3_GATE_BARRED.json"
if bool(gate["pass"]) == barred.exists():
    raise SystemExit("B3 gate marker disagrees with strict gate verdict")
if barred.exists():
    barred_payload = json.loads(barred.read_text(encoding="utf-8"))
    expected_barred = {
        "cell": "mlp_l05",
        "reason": "strict preregistered P1 gate failed",
        "h1_fixed_p": 1.0,
        "h1_family_size": 3,
        "gate": gate,
    }
    if barred_payload != expected_barred:
        raise SystemExit("B3 gate-barred marker payload drift")
artifact_set_sha = hashlib.sha256(
    json.dumps(artifact_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
payload = {
    "run_id": module.RUN_ID,
    "state": "CACHE_VALIDATED",
    "n_cache_artifacts": 45,
    "fit_summary_sha256": module.sha256_file(summary_path),
    "cache_artifact_set_sha256": artifact_set_sha,
    "mlp_hybrid_gate": gate,
}
dest = root / "cache_validated.json"
if mode == "create":
    module.atomic_write_json(dest, payload)
elif mode == "check":
    frozen = json.loads(dest.read_text(encoding="utf-8"))
    if frozen != payload:
        raise SystemExit("cache validation marker differs from independently verified caches")
else:
    raise SystemExit(f"invalid integrity mode: {mode}")
print(json.dumps(payload, indent=2, allow_nan=False))
PY
}


require_validated_cache_integrity() {
  [[ -f "$ROOT/cache_validated.json" ]] || die "cache validation is required"
  (
    cd "$JEPA_WT"
    "$PY" - "$CACHE" "$ROOT" check <<'PY'
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
root = Path(sys.argv[2])
mode = sys.argv[3]
spec = importlib.util.spec_from_file_location("cube_probe_cache_reviewed", cache_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summary_path = root / "probes" / "probe_fit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("schema_version") != module.SCHEMA_VERSION or summary.get("run_id") != module.RUN_ID:
    raise SystemExit("fit summary schema/run id drift")
expected_sources = {
    "probe_runner": module.PROBE_RUNNER_SHA256,
    "probe_common": module.PROBE_COMMON_SHA256,
    "budget_runner": module.BUDGET_RUNNER_SHA256,
    "task_manifest": module.TASK_MANIFEST_SHA256,
}
if summary.get("source_sha256") != expected_sources:
    raise SystemExit("fit summary source hashes drift")
if summary.get("checkpoint_sha256") != {str(k): v for k, v in module.CHECKPOINT_SHA256.items()}:
    raise SystemExit("fit summary checkpoint hashes drift")
records = summary.get("records", [])
if len(records) != 45 or summary.get("n_cache_artifacts") != 45:
    raise SystemExit("expected exactly 45 immutable cache records")
keys = {(r["family"], int(r["train_seed"]), int(r["eval_seed"])) for r in records}
expected = {(f, ts, es) for f in ("p0", "p1", "p2") for ts in (42,43,44) for es in (42,43,44,45,46)}
if keys != expected:
    raise SystemExit("cache pair matrix mismatch")
artifact_map = {}
for record in records:
    rel = Path(record["path"])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "probe_caches":
        raise SystemExit(f"unsafe cache path: {rel}")
    meta, _ = module.load_npz_cache(root / rel, expected_sha256=record["sha256"])
    module._validate_loaded_metadata(meta, family=record["family"], train_seed=int(record["train_seed"]), eval_seed=int(record["eval_seed"]))
    artifact_map[str(rel)] = record["sha256"]
if len(artifact_map) != 45 or summary.get("cache_artifact_sha256") != artifact_map:
    raise SystemExit("cache artifact hash map differs from the 45 records")
gate = module.recompute_strict_p1_gate(root, summary)
barred = root / "probes" / "B3_GATE_BARRED.json"
if bool(gate["pass"]) == barred.exists():
    raise SystemExit("B3 gate marker disagrees with strict gate verdict")
if barred.exists():
    barred_payload = json.loads(barred.read_text(encoding="utf-8"))
    expected_barred = {
        "cell": "mlp_l05",
        "reason": "strict preregistered P1 gate failed",
        "h1_fixed_p": 1.0,
        "h1_family_size": 3,
        "gate": gate,
    }
    if barred_payload != expected_barred:
        raise SystemExit("B3 gate-barred marker payload drift")
payload = {
    "run_id": module.RUN_ID,
    "state": "CACHE_VALIDATED",
    "n_cache_artifacts": 45,
    "fit_summary_sha256": module.sha256_file(summary_path),
    "cache_artifact_set_sha256": hashlib.sha256(
        json.dumps(artifact_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
    "mlp_hybrid_gate": gate,
}
frozen = json.loads((root / "cache_validated.json").read_text(encoding="utf-8"))
if mode != "check" or frozen != payload:
    raise SystemExit("cache validation marker differs from independently verified caches")
print("cache_integrity_ok")
PY
  )
}


gate_passes() {
  (
  source ~/.jepawm_env
  cd "$JEPA_WT"
  "$PY" - "$ROOT/cache_validated.json" <<'PY'
import json, sys
print("yes" if json.load(open(sys.argv[1], encoding="utf-8"))["mlp_hybrid_gate"]["pass"] else "no")
PY
  )
}


eligible_cells() {
  printf '%s\n' linear_l03 linear_l04
  if [[ "$(gate_passes)" == yes ]]; then
    printf '%s\n' mlp_l05
  fi
  printf '%s\n' mlp_pure decoy_pure decoy_l05
}


cell_gpu() {
  case "$1" in
    linear_l03) echo 0 ;;
    linear_l04) echo 1 ;;
    mlp_l05) echo 2 ;;
    mlp_pure) echo 3 ;;
    decoy_pure) echo 4 ;;
    decoy_l05) echo 5 ;;
    *) die "unknown cell: $1" ;;
  esac
}


cell_is_eligible() {
  local wanted=$1 cell
  while read -r cell; do
    [[ "$cell" == "$wanted" ]] && return 0
  done < <(eligible_cells)
  return 1
}


phase_prerequisites() {
  local phase=$1
  require_validated_cache_integrity
  require_dataset_stat
  require_analyzer_snapshot
  if [[ "$phase" == full ]]; then
    [[ -f "$ROOT/smoke_validated.json" ]] || die "smoke validation is required before full launch"
  fi
}


preflight_batch() {
  local phase=$1
  shift
  local cell gpu
  [[ ! -e "$STATE/${phase}.attempted.json" \
      && ! -e "$STATE/${phase}.launched.json" \
      && ! -e "$STATE/${phase}.launch.failed.json" ]] \
    || die "$phase batch was already attempted"
  [[ ! -e "$ROOT/${phase}_pids.txt" ]] || die "$phase pid manifest already exists"
  [[ ! -e "$STATE/${phase}.launch.records" ]] || die "$phase launch-record directory already exists"
  for cell in "$@"; do
    gpu=$(cell_gpu "$cell")
    gpu_free "$gpu"
    [[ ! -e "$ROOT/$phase/cube_${cell}" ]] || die "$phase final output collision for $cell"
    [[ ! -e "$ROOT/$phase/cube_${cell}.partial" ]] || die "$phase partial output collision for $cell"
    [[ ! -e "$STATE/${phase}_${cell}.running.json" ]] || die "$phase running marker collision for $cell"
    [[ ! -e "$STATE/${phase}_${cell}.attempted.json" ]] || die "$phase attempt marker collision for $cell"
    [[ ! -e "$STATE/${phase}_${cell}.done.json" ]] || die "$phase done marker collision for $cell"
    [[ ! -e "$STATE/${phase}_${cell}.failed.json" ]] || die "$phase failed marker collision for $cell"
    [[ ! -e "$COMMANDS/${phase}_${cell}.command.txt" ]] || die "$phase command collision for $cell"
    [[ ! -e "$LOGS/${phase}_${cell}.log" ]] || die "$phase log collision for $cell"
  done
}


run_cell_internal() {
  local phase=$1
  local cell=$2
  local gpu out_partial out_final swm jepa_commit rc
  gpu=$(cell_gpu "$cell")
  source ~/.jepawm_env
  [[ -f "$STATE/${phase}.attempted.json" ]] || die "$phase worker lacks a batch launch claim"
  [[ -f "$STATE/${phase}_${cell}.attempted.json" ]] || die "$phase/$cell worker lacks a cell launch claim"
  [[ -n "${LAUNCH_TOKEN:-}" && -f "$STATE/${phase}.launch.token" \
      && "$LAUNCH_TOKEN" == "$(<"$STATE/${phase}.launch.token")" ]] \
    || die "$phase/$cell worker launch token mismatch"
  WORKER_DONE_MARKER="$STATE/${phase}_${cell}.done.json"
  WORKER_FAILED_MARKER="$STATE/${phase}_${cell}.failed.json"
  WORKER_FAILURE_FIELDS='"state":"'"${phase^^}"'_FAILED","cell":"'"$cell"'","gpu":'"$gpu"',"pid":'"$$"
  arm_worker_traps
  require_setup
  phase_prerequisites "$phase"
  cell_is_eligible "$cell" || die "cell is gate-barred or unknown: $cell"
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "$gpu" ]] || die "$cell must use physical GPU $gpu"
  out_partial="$ROOT/$phase/cube_${cell}.partial"
  out_final="$ROOT/$phase/cube_${cell}"
  [[ ! -e "$out_partial" && ! -e "$out_final" ]] || die "cell output collision: $phase $cell"
  local self_boot self_ticks
  self_boot=$(boot_id)
  self_ticks=$(proc_start_ticks $$)
  atomic_text "$STATE/${phase}_${cell}.running.json" \
    "{\"state\":\"${phase^^}_RUNNING\",\"cell\":\"$cell\",\"pid\":$$,\"gpu\":$gpu,\"boot_id\":\"$self_boot\",\"start_ticks\":\"$self_ticks\",\"launch_token\":\"$LAUNCH_TOKEN\",\"at\":\"$(iso_now)\"}"
  swm=$(stable_worktree)
  jepa_commit=$(<"$PROV/jepawms_git_commit.txt")
  cd "$JEPA_WT"
  write_command "$COMMANDS/${phase}_${cell}.command.txt" "$PY" "$CACHE" run-cached \
    --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
    --task-manifest "$MANIFEST" --run-root "$ROOT" --out-root "$out_partial" \
    --cell "$cell" --mode "$phase" --jepa-commit "$jepa_commit" \
    --stable-commit "$STABLE_COMMIT" --device cuda
  set +e
  PYTHONPATH="$swm${PYTHONPATH:+:$PYTHONPATH}" ADADIM_STABLEWM_REPO="$swm" WANDB_MODE=disabled \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    setsid "$PY" "$CACHE" run-cached \
      --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
      --task-manifest "$MANIFEST" --run-root "$ROOT" --out-root "$out_partial" \
      --cell "$cell" --mode "$phase" --jepa-commit "$jepa_commit" \
      --stable-commit "$STABLE_COMMIT" --device cuda &
  WORKER_CHILD_PID=$!
  WORKER_CHILD_TICKS=$(proc_start_ticks "$WORKER_CHILD_PID")
  wait "$WORKER_CHILD_PID"
  rc=$?
  WORKER_CHILD_PID=
  WORKER_CHILD_TICKS=
  set -e
  if [[ "$rc" -eq 0 ]]; then
    local required
    for required in run_metadata.json cell_summaries.json probe_physical_cem_episode_records.csv \
      probe_physical_cem_replan_records.csv probe_physical_cem_summary.csv probe_physical_cem_summary.md; do
      [[ -f "$out_partial/$required" ]] || rc=92
    done
  fi
  if [[ "$rc" -eq 0 ]]; then
    mv "$out_partial" "$out_final"
    atomic_text "$STATE/${phase}_${cell}.done.json" \
      "{\"state\":\"${phase^^}_DONE\",\"cell\":\"$cell\",\"pid\":$$,\"gpu\":$gpu,\"at\":\"$(iso_now)\"}"
    exit 0
  fi
  atomic_text "$STATE/${phase}_${cell}.failed.json" \
    "{\"state\":\"${phase^^}_FAILED\",\"cell\":\"$cell\",\"pid\":$$,\"gpu\":$gpu,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}"
  exit "$rc"
}


start_batch() {
  local phase=$1
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  require_validated_cache_integrity
  require_analyzer_snapshot
  if [[ "$phase" == full ]]; then
    require_smoke_attestation
  fi
  local -a cells
  mapfile -t cells < <(eligible_cells)
  [[ "${#cells[@]}" -eq 5 || "${#cells[@]}" -eq 6 ]] \
    || die "unexpected eligible-cell count: ${#cells[@]}"
  preflight_batch "$phase" "${cells[@]}"
  local cells_csv gate dataset_sha cache_validated_sha analyzer_sha smoke_attestation_sha launch_token
  cells_csv=$(IFS=,; echo "${cells[*]}")
  gate=$(gate_passes)
  dataset_sha=$(<"$PROV/cube_dataset_sha256.txt")
  cache_validated_sha=$(sha_of "$ROOT/cache_validated.json")
  analyzer_sha=$(<"$PROV/analyzer_sha256.txt")
  smoke_attestation_sha=
  if [[ "$phase" == full ]]; then
    smoke_attestation_sha=$(sha_of "$ROOT/smoke_validation_attestation.json")
  fi
  [[ ! -e "$STATE/${phase}.launch.token" ]] || die "$phase launch token already exists"
  local cell gpu pid worker_boot worker_ticks launched_count=0 launch_complete=0
  batch_launch_trap() {
    local rc=$?
    trap - EXIT
    if [[ "$launch_complete" -ne 1 && ! -e "$STATE/${phase}.launch.failed.json" ]]; then
      set +e
      try_atomic_text "$STATE/${phase}.launch.failed.json" \
        "{\"state\":\"${phase^^}_LAUNCH_FAILED\",\"launched_count\":$launched_count,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap batch_launch_trap EXIT
  launch_token=$(< /proc/sys/kernel/random/uuid)
  atomic_text "$STATE/${phase}.launch.token" "$launch_token"
  atomic_text "$STATE/${phase}.attempted.json" \
    "{\"state\":\"${phase^^}_ATTEMPTED\",\"cells\":\"$cells_csv\",\"strict_p1_gate_pass\":\"$gate\",\"launch_token\":\"$launch_token\",\"dataset_sha256\":\"$dataset_sha\",\"cache_validated_sha256\":\"$cache_validated_sha\",\"analyzer_sha256\":\"$analyzer_sha\",\"smoke_attestation_sha256\":\"$smoke_attestation_sha\",\"at\":\"$(iso_now)\"}"
  reserve_file "$ROOT/${phase}_pids.txt"
  mkdir "$STATE/${phase}.launch.records"
  for cell in "${cells[@]}"; do
    gpu=$(cell_gpu "$cell")
    reserve_file "$LOGS/${phase}_${cell}.log"
    atomic_text "$STATE/${phase}_${cell}.attempted.json" \
      "{\"state\":\"${phase^^}_CELL_ATTEMPTED\",\"cell\":\"$cell\",\"gpu\":$gpu,\"at\":\"$(iso_now)\"}"
    CUDA_VISIBLE_DEVICES="$gpu" LAUNCH_TOKEN="$launch_token" setsid nohup bash "$LAUNCHER" run-cell "$phase" "$cell" 9>&- >> "$LOGS/${phase}_${cell}.log" 2>&1 &
    pid=$!
    worker_boot=$(boot_id)
    worker_ticks=$(proc_start_ticks "$pid") || worker_ticks=unavailable
    atomic_text "$STATE/${phase}.launch.records/${cell}.json" \
      "{\"state\":\"${phase^^}_LAUNCH_RECORD\",\"cell\":\"$cell\",\"gpu\":$gpu,\"pid\":$pid,\"boot_id\":\"$worker_boot\",\"start_ticks\":\"$worker_ticks\",\"launch_token\":\"$launch_token\",\"at\":\"$(iso_now)\"}"
    printf '%s %s %s %s %s\n' "$cell" "$gpu" "$pid" "$worker_boot" "$worker_ticks" >> "$ROOT/${phase}_pids.txt"
    launched_count=$((launched_count + 1))
  done
  atomic_text "$STATE/${phase}.launched.json" \
    "{\"state\":\"${phase^^}_LAUNCHED\",\"cells\":\"$cells_csv\",\"launched_count\":$launched_count,\"at\":\"$(iso_now)\"}"
  launch_complete=1
  trap - EXIT
  echo "started_$phase"
  cat "$ROOT/${phase}_pids.txt"
}


require_phase_done() {
  local phase=$1 cell required expected_count=0 pid_count record_count
  [[ -f "$STATE/${phase}.launched.json" ]] || die "$phase launch did not complete"
  [[ ! -e "$STATE/${phase}.launch.failed.json" ]] || die "$phase launch failed"
  while read -r cell; do
    expected_count=$((expected_count + 1))
    [[ -f "$STATE/${phase}.launch.records/${cell}.json" ]] || die "$phase launch record missing: $cell"
    [[ -f "$STATE/${phase}_${cell}.done.json" ]] || die "$phase cell not done: $cell"
    [[ ! -e "$STATE/${phase}_${cell}.failed.json" ]] || die "$phase cell failed: $cell"
    [[ -d "$ROOT/$phase/cube_${cell}" ]] || die "$phase final output directory missing: $cell"
    [[ ! -e "$ROOT/$phase/cube_${cell}.partial" ]] || die "$phase partial output remains: $cell"
    for required in run_metadata.json cell_summaries.json probe_physical_cem_episode_records.csv \
      probe_physical_cem_replan_records.csv probe_physical_cem_summary.csv probe_physical_cem_summary.md; do
      [[ -f "$ROOT/$phase/cube_${cell}/$required" ]] \
        || die "$phase final output missing for $cell: $required"
    done
  done < <(eligible_cells)
  [[ -f "$ROOT/${phase}_pids.txt" ]] || die "$phase pid manifest missing"
  pid_count=$(grep -cve '^[[:space:]]*$' "$ROOT/${phase}_pids.txt")
  record_count=$(find "$STATE/${phase}.launch.records" -maxdepth 1 -type f -name '*.json' | wc -l)
  [[ "$pid_count" -eq "$expected_count" && "$record_count" -eq "$expected_count" ]] \
    || die "$phase launch ledger count drift: expected=$expected_count pids=$pid_count records=$record_count"
}


validate_smoke() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  require_validated_cache_integrity
  require_phase_done smoke
  [[ ! -e "$ROOT/smoke_validated.json" ]] || die "smoke was already validated"
  require_analyzer_snapshot
  local swm
  swm=$(stable_worktree)
  cd "$JEPA_WT"
  PYTHONPATH="$swm${PYTHONPATH:+:$PYTHONPATH}" ADADIM_STABLEWM_REPO="$swm" \
    "$PY" "$ANALYZER" validate-smoke --run-root "$ROOT"
  [[ -f "$ROOT/smoke_validated.json" ]] || die "analyzer did not write smoke validation marker"
  atomic_text "$ROOT/smoke_validation_attestation.json" \
    "{\"analyzer_sha256\":\"$(<"$PROV/analyzer_sha256.txt")\",\"cache_validated_sha256\":\"$(sha_of "$ROOT/cache_validated.json")\",\"dataset_sha256\":\"$(<"$PROV/cube_dataset_sha256.txt")\",\"run_id\":\"$RUN_ID\",\"smoke_validated_sha256\":\"$(sha_of "$ROOT/smoke_validated.json")\",\"state\":\"SMOKE_VALIDATION_ATTESTED\"}"
  require_smoke_attestation
}


pid_process_state() {
  local pid=$1 expected_boot=$2 expected_ticks=$3 expected_a=$4 expected_b=$5 expected_c=${6:-}
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    echo exited
    return 0
  fi
  if [[ "$expected_boot" != "$(boot_id)" \
        || "$expected_ticks" == unavailable \
        || "$expected_ticks" != "$(proc_start_ticks "$pid" 2>/dev/null || echo missing)" ]]; then
    echo reused_or_unrelated
    return 0
  fi
  local command_line
  command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  if [[ "$command_line" == *"$LAUNCHER"* \
        && "$command_line" == *"$expected_a"* \
        && "$command_line" == *"$expected_b"* \
        && ( -z "$expected_c" || "$command_line" == *"$expected_c"* ) ]]; then
    echo running
  else
    echo reused_or_unrelated
  fi
}


status_phase() {
  local phase=$1 pidfile="$ROOT/${phase}_pids.txt"
  [[ -f "$STATE/${phase}.launch.failed.json" ]] && echo "$phase: launch_failed=yes"
  if [[ ! -f "$pidfile" ]]; then
    echo "$phase: not started"
    return 0
  fi
  local cell gpu pid expected_boot expected_ticks process_state marker
  while read -r cell gpu pid expected_boot expected_ticks; do
    [[ -n "${cell:-}" ]] || continue
    process_state=$(pid_process_state "$pid" "$expected_boot" "$expected_ticks" run-cell "$phase" "$cell")
    marker=none
    [[ -f "$STATE/${phase}_${cell}.running.json" ]] && marker=running
    [[ -f "$STATE/${phase}_${cell}.done.json" ]] && marker=done
    [[ -f "$STATE/${phase}_${cell}.failed.json" ]] && marker=failed
    echo "$phase cell=$cell gpu=$gpu pid=$pid process=$process_state marker=$marker"
  done < "$pidfile"
}


status_all() {
  local cache_state=not_started
  [[ -f "$STATE/cache.running.json" ]] && cache_state=running
  [[ -f "$STATE/cache.done.json" ]] && cache_state=done
  [[ -f "$STATE/cache.failed.json" ]] && cache_state=failed
  local cache_process=not_started
  if [[ -f "$ROOT/cache_pid.txt" ]]; then
    local cache_pid cache_boot cache_ticks
    read -r cache_pid cache_boot cache_ticks < "$ROOT/cache_pid.txt"
    cache_process=$(pid_process_state "$cache_pid" "$cache_boot" "$cache_ticks" run-cache run-cache)
  fi
  echo "cache state=$cache_state process=$cache_process validated=$([[ -f "$ROOT/cache_validated.json" ]] && echo yes || echo no)"
  [[ -f "$STATE/cache.launch.failed.json" ]] && echo "cache launch_failed=yes"
  status_phase smoke
  status_phase full
  echo "sealed=$([[ -f "$ROOT/READY_TO_PULL.json" ]] && echo yes || echo no)"
}


seal_full() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  require_validated_cache_integrity
  require_analyzer_snapshot
  require_smoke_attestation
  require_phase_done full
  [[ ! -e "$ROOT/artifact_sha256.txt" && ! -e "$ROOT/READY_TO_PULL.json" ]] || die "run root is already sealed"
  local tmp="$ROOT/artifact_sha256.txt.tmp.$$"
  (
    cd "$ROOT"
    find . -type f ! -name artifact_sha256.txt ! -name READY_TO_PULL.json ! -name 'artifact_sha256.txt.tmp.*' -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "$tmp"
  publish_file "$tmp" "$ROOT/artifact_sha256.txt"
  atomic_text "$ROOT/READY_TO_PULL.json" \
    "{\"run_id\":\"$RUN_ID\",\"state\":\"FULL_SEALED\",\"at\":\"$(iso_now)\",\"artifact_manifest_sha256\":\"$(sha_of "$ROOT/artifact_sha256.txt")\",\"jepa_commit\":\"$(<"$PROV/jepawms_git_commit.txt")\"}"
  echo "full_sealed manifest=$ROOT/artifact_sha256.txt"
}


usage() {
  cat <<'EOF'
usage: cube_probe_closure_20260713.sh COMMAND

Public commands:
  setup
  register-analyzer PATH
  start-cache | status | validate-cache
  start-smoke | status-smoke | validate-smoke
  start-full | status-full | seal-full

Private worker commands:
  run-cache
  run-cell {smoke|full} CELL
EOF
}


case "${1:-}" in
  setup) setup_run ;;
  register-analyzer)
    [[ $# -eq 2 ]] || die "register-analyzer requires PATH"
    run_locked register_analyzer "$2"
    ;;
  start-cache) run_locked start_cache ;;
  validate-cache) run_locked validate_cache ;;
  start-smoke) run_locked start_batch smoke ;;
  validate-smoke) run_locked validate_smoke ;;
  start-full) run_locked start_batch full ;;
  seal-full) run_locked seal_full ;;
  status-smoke) status_phase smoke ;;
  status-full) status_phase full ;;
  status) status_all ;;
  run-cache) run_cache_internal ;;
  run-cell)
    [[ $# -eq 3 ]] || die "run-cell requires PHASE CELL"
    [[ "$2" == smoke || "$2" == full ]] || die "invalid phase: $2"
    run_cell_internal "$2" "$3"
    ;;
  -h|--help|help|'') usage ;;
  *) usage >&2; exit 2 ;;
esac
