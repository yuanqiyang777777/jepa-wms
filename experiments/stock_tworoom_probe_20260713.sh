#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

# Immutable experiment identity.
BASE=/home/ps/Code/yqy/DINO-WM/stablewm_home
CODE=/home/ps/Code/yqy/DINO-WM
RUN_ID=stock_tworoom_probe_20260713_retry1
ROOT="$BASE/$RUN_ID"
SOURCE_JEPA="$CODE/jepa-wms"
SOURCE_STABLE="$CODE/stable-worldmodel"
JEPA_WT="$CODE/wt_stock_tworoom_probe_jepawms_20260713"
SWM_WT_REUSE="$CODE/wt_probe_f2compat_stablewm_20260713"
SWM_WT_ALT="$CODE/wt_stock_tworoom_probe_stablewm_20260713"
JEPA_BASE_COMMIT=3214dc52ff3290256ef77b985c9e1f1e339fde44
STABLE_COMMIT=314201d0347baf61304c7558a98463c6a01080f0
EXPECTED_BRANCH=exp/20260713-stock-tworoom-probe

PY="$SOURCE_STABLE/.venv/bin/python"
F2="$BASE/latent_dimension_interference_20260702/probe_planning_link_20260703"
SOURCE_PROBE="$F2/scripts_v2_fenced/probe_physical_cost_cem_adadim_block_20260703.py"
SOURCE_COMMON="$F2/scripts_v2_fenced/candidate_ranking_alignment_adadim_block_20260703.py"
# Reuse the already frozen F2-compatible source snapshot from the sealed cube run.
SOURCE_BUDGET="$BASE/adadim_cube_probe_closure_20260713/provenance/adadim_budget_sweep_runner_f2compat.py"
SOURCE_MANIFEST="$JEPA_WT/experiments/stock_tworoom_probe_manifest_20260713.json"
SOURCE_CACHE="$JEPA_WT/experiments/scripts/stock_tworoom_probe_cache_20260713.py"
SOURCE_LAUNCHER="$JEPA_WT/experiments/stock_tworoom_probe_20260713.sh"

EXPECTED_PROBE=564d7469840a38bf6015149d9d61191694beef9e0f5ed6610c5db9934b7dbdc8
EXPECTED_COMMON=9c57a82343277e3378f24df8ca38e5f9a4dcc53a4fead909e2a6126001d1a89c
EXPECTED_BUDGET=cefd2b267601377aa1201a5302828807f4189ca11f123c1128746938f51dfd8f
EXPECTED_MANIFEST=ffffb1ae170ec45db59a7b155890ef868e145a30e112a580e99bfc14bf4987ad
EXPECTED_ENV_SOURCE=5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0
EXPECTED_BASELINE_SHA256=474065cefd579152dc670cbcc5896d7d97790b887e1ee93946c3d2bfd48c0640
# Review identity only.  The analyzer is local-only and is never copied to lab01.
EXPECTED_ANALYZER_SHA256=0fb5063db36d7439fc92004fbc9f60c29e98b00da00691ab72c66ba8a5d03320

DATASET="$BASE/datasets/tworoom.h5"
CELLS=(
  t1_linear_pure
  t2_linear_hybrid_l05
  t3_mlp_hybrid_l05
  t4_mlp_pure
)
CELLS_JSON='["t1_linear_pure","t2_linear_hybrid_l05","t3_mlp_hybrid_l05","t4_mlp_pure"]'

PROV="$ROOT/provenance"
STATE="$ROOT/state"
LOGS="$ROOT/logs"
COMMANDS="$ROOT/commands"
PROBE="$PROV/probe_physical_cost_cem_adadim_block_20260703.py"
COMMON="$PROV/candidate_ranking_alignment_adadim_block_20260703.py"
BUDGET="$PROV/adadim_budget_sweep_runner_f2compat.py"
MANIFEST="$PROV/stock_tworoom_probe_manifest_20260713.json"
CACHE="$PROV/stock_tworoom_probe_cache_20260713.py"
LAUNCHER="$PROV/stock_tworoom_probe_20260713.sh"
ENV_SOURCE="$PROV/two_room_env_at_314201d0.py"


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
  [[ "$actual" == "$expected" ]] \
    || die "hash mismatch: $file expected=$expected actual=$actual"
}


try_atomic_text() {
  local path=$1
  local value=$2
  local tmp="${path}.tmp.$$"
  [[ ! -e "$path" && ! -e "$tmp" ]] || return 1
  if ! printf '%s\n' "$value" > "$tmp"; then
    rm -f -- "$tmp"
    return 1
  fi
  if ! ln -- "$tmp" "$path"; then
    rm -f -- "$tmp"
    return 1
  fi
  rm -f -- "$tmp"
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
  if ! ln -- "$tmp" "$path"; then
    rm -f -- "$tmp"
    die "atomic no-replace publish failed: $path"
  fi
  rm -f -- "$tmp"
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


checkpoint_path() {
  local seed=$1
  printf '%s/checkpoints/stock_lewm_tworoom_matched3_seed%s_20260615/weights_epoch_10.pt\n' \
    "$BASE" "$seed"
}


checkpoint_sha() {
  case "$1" in
    42) echo 96da8768f4e51707417b548f00efb19bceab39adff7da49b1a6e50b4d3491d56 ;;
    43) echo 9b6cdf2fa1d8a6f89512201aa422142c7158220d83eebd7b532d4ae5ed560973 ;;
    44) echo 43d783e3950231c17441e81b7bbe76c2d81ab1d1fb260be9af894519a484231e ;;
    *) die "unsupported stock train seed: $1" ;;
  esac
}


config_sha() {
  case "$1" in
    42) echo 9e0362012040493d793d6297d658fd729d9bc1e15443dd33534f39e1d5505938 ;;
    43) echo a2165f9c8541b5d97fd6253d3ef6ee05828e32d944134115605cd97a1567a0f6 ;;
    44) echo 76aea934e7e3e96288c18829b256da4589b76aafa36d4084fdee58abea04aaf4 ;;
    *) die "unsupported stock train seed: $1" ;;
  esac
}


require_source_hashes() {
  require_hash "$SOURCE_PROBE" "$EXPECTED_PROBE"
  require_hash "$SOURCE_COMMON" "$EXPECTED_COMMON"
  require_hash "$SOURCE_BUDGET" "$EXPECTED_BUDGET"
  require_hash "$SOURCE_MANIFEST" "$EXPECTED_MANIFEST"
}


require_checkpoints() {
  local seed checkpoint config
  for seed in 42 43 44; do
    checkpoint=$(checkpoint_path "$seed")
    config="$(dirname "$checkpoint")/config.json"
    require_hash "$checkpoint" "$(checkpoint_sha "$seed")"
    require_hash "$config" "$(config_sha "$seed")"
  done
  [[ -f "$DATASET" ]] || die "missing TwoRoom dataset: $DATASET"
}


require_plain_stock_configs() {
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    "$PY" - \
      "$(dirname "$(checkpoint_path 42)")/config.json" \
      "$(dirname "$(checkpoint_path 43)")/config.json" \
      "$(dirname "$(checkpoint_path 44)")/config.json" <<'PY'
import json
import sys
from pathlib import Path

for path_text in sys.argv[1:]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = str(payload.get("model", {}).get("_target_", ""))
    if not target.endswith(".LeWM") or "NestedLeWM" in target:
        raise SystemExit(f"checkpoint config is not plain stock LeWM: {path}: {target!r}")
print("plain_stock_config_ok")
PY
  )
}


require_dataset_identity() {
  [[ -f "$PROV/tworoom_dataset_sha256.txt" ]] || die "dataset hash was not captured"
  local expected actual
  expected=$(<"$PROV/tworoom_dataset_sha256.txt")
  actual=$(sha_of "$DATASET")
  [[ "$actual" == "$expected" ]] \
    || die "TwoRoom dataset hash drift: expected=$expected actual=$actual"
}


require_dataset_stat() {
  [[ -f "$PROV/tworoom_dataset_stat.txt" ]] || die "dataset stat was not captured"
  local current
  current=$(stat -c '%y %s %n' "$DATASET")
  [[ "$current" == "$(<"$PROV/tworoom_dataset_stat.txt")" ]] \
    || die "TwoRoom dataset stat drift"
}


require_jepa_branch() {
  local top branch status_count diff_count diff_status diff_path
  [[ -e "$JEPA_WT/.git" ]] || die "missing execution worktree: $JEPA_WT"
  top=$(git -C "$JEPA_WT" rev-parse --show-toplevel)
  [[ "$(realpath "$top")" == "$(realpath "$JEPA_WT")" ]] \
    || die "unexpected jepa worktree root: $top"
  branch=$(git -C "$JEPA_WT" branch --show-current)
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || die "wrong experiment branch: $branch"
  git -C "$JEPA_WT" merge-base --is-ancestor "$JEPA_BASE_COMMIT" HEAD \
    || die "branch is not based on $JEPA_BASE_COMMIT"
  status_count=$(git -C "$JEPA_WT" status --porcelain | wc -l)
  [[ "$status_count" -eq 0 ]] || die "jepa execution worktree is dirty"
  diff_count=0
  while IFS=$'\t' read -r diff_status diff_path; do
    [[ -n "$diff_status" ]] || continue
    [[ "$diff_status" == A ]] || die "branch modifies a tracked file: $diff_status $diff_path"
    case "$diff_path" in
      experiments/stock_tworoom_probe_20260713.sh|experiments/scripts/stock_tworoom_probe_cache_20260713.py|experiments/stock_tworoom_probe_manifest_20260713.json)
        ;;
      *) die "branch contains unauthorized file: $diff_path" ;;
    esac
    diff_count=$((diff_count + 1))
  done < <(git -C "$JEPA_WT" diff --name-status "$JEPA_BASE_COMMIT"..HEAD)
  [[ "$diff_count" -eq 3 ]] \
    || die "expected exactly three added experiment files, found $diff_count"
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
  local selected env_file
  [[ -f "$PROV/stable_worktree_path.txt" ]] || die "stable worktree path was not captured"
  selected=$(<"$PROV/stable_worktree_path.txt")
  [[ "$selected" == "$SWM_WT_REUSE" || "$selected" == "$SWM_WT_ALT" ]] \
    || die "unexpected stable worktree path: $selected"
  [[ -e "$selected/.git" ]] || die "missing stable worktree: $selected"
  [[ "$(git -C "$selected" rev-parse HEAD)" == "$STABLE_COMMIT" ]] \
    || die "stable worktree commit drift"
  [[ -z "$(git -C "$selected" status --porcelain)" ]] || die "stable worktree is dirty"
  env_file="$selected/stable_worldmodel/envs/two_room/env.py"
  require_hash "$env_file" "$EXPECTED_ENV_SOURCE"
}


stable_worktree() {
  cat "$PROV/stable_worktree_path.txt"
}


require_snapshot_hashes() {
  require_hash "$PROBE" "$EXPECTED_PROBE"
  require_hash "$COMMON" "$EXPECTED_COMMON"
  require_hash "$BUDGET" "$EXPECTED_BUDGET"
  require_hash "$MANIFEST" "$EXPECTED_MANIFEST"
  require_hash "$ENV_SOURCE" "$EXPECTED_ENV_SOURCE"
  local seed
  for seed in 42 43 44; do
    require_hash "$PROV/stock_seed${seed}_config.json" "$(config_sha "$seed")"
  done
  [[ "$(sha_of "$CACHE")" == "$(<"$PROV/cache_script_sha256.txt")" ]] \
    || die "cache-script snapshot drift"
  [[ "$(sha_of "$LAUNCHER")" == "$(<"$PROV/launcher_sha256.txt")" ]] \
    || die "launcher snapshot drift"
  [[ "$(<"$PROV/baseline_manifest_sha256.txt")" == "$EXPECTED_BASELINE_SHA256" ]] \
    || die "frozen local baseline hash provenance drift"
  [[ "$(<"$PROV/local_analyzer_review_sha256.txt")" == "$EXPECTED_ANALYZER_SHA256" ]] \
    || die "local-only analyzer review hash provenance drift"
}


require_setup() {
  [[ -f "$STATE/setup.done.json" ]] || die "setup has not completed"
  require_jepa_branch
  [[ "$(git -C "$JEPA_WT" rev-parse HEAD)" == "$(<"$PROV/jepawms_git_commit.txt")" ]] \
    || die "reviewed experiment commit drift"
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
  [[ -f "$SOURCE_CACHE" && -f "$SOURCE_LAUNCHER" ]] \
    || die "missing reviewed experiment sources"
  mkdir "$ROOT" || die "run-id collision or atomic setup claim failed: $ROOT"

  local setup_ok=0
  setup_failure_trap() {
    local rc=$?
    trap - EXIT
    if [[ "$setup_ok" -ne 1 && -d "$STATE" && ! -e "$STATE/setup.failed.json" ]]; then
      set +e
      try_atomic_text "$STATE/setup.failed.json" \
        "{\"run_id\":\"$RUN_ID\",\"state\":\"SETUP_FAILED\",\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" \
        >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap setup_failure_trap EXIT

  mkdir -p "$PROV" "$STATE" "$LOGS" "$COMMANDS" "$ROOT/smoke" "$ROOT/full"
  local selected_swm env_file seed checkpoint config
  selected_swm=$(select_stable_worktree)
  printf '%s\n' "$selected_swm" > "$PROV/stable_worktree_path.txt"
  env_file="$selected_swm/stable_worldmodel/envs/two_room/env.py"
  require_hash "$env_file" "$EXPECTED_ENV_SOURCE"

  cp --preserve=mode,timestamps "$SOURCE_PROBE" "$PROBE"
  cp --preserve=mode,timestamps "$SOURCE_COMMON" "$COMMON"
  cp --preserve=mode,timestamps "$SOURCE_BUDGET" "$BUDGET"
  cp --preserve=mode,timestamps "$SOURCE_MANIFEST" "$MANIFEST"
  cp --preserve=mode,timestamps "$SOURCE_CACHE" "$CACHE"
  cp --preserve=mode,timestamps "$SOURCE_LAUNCHER" "$LAUNCHER"
  cp --preserve=mode,timestamps "$env_file" "$ENV_SOURCE"
  for seed in 42 43 44; do
    checkpoint=$(checkpoint_path "$seed")
    config="$(dirname "$checkpoint")/config.json"
    cp --preserve=mode,timestamps "$config" "$PROV/stock_seed${seed}_config.json"
  done

  git -C "$JEPA_WT" rev-parse HEAD > "$PROV/jepawms_git_commit.txt"
  git -C "$JEPA_WT" describe --always --dirty > "$PROV/jepawms_git_describe.txt"
  git -C "$JEPA_WT" status --porcelain > "$PROV/jepawms_git_status_porcelain.txt"
  git -C "$selected_swm" rev-parse HEAD > "$PROV/stableworldmodel_git_commit.txt"
  git -C "$selected_swm" describe --always --dirty > "$PROV/stableworldmodel_git_describe.txt"
  git -C "$selected_swm" status --porcelain > "$PROV/stableworldmodel_git_status_porcelain.txt"
  sha_of "$CACHE" > "$PROV/cache_script_sha256.txt"
  sha_of "$LAUNCHER" > "$PROV/launcher_sha256.txt"
  printf '%s\n' "$EXPECTED_BASELINE_SHA256" > "$PROV/baseline_manifest_sha256.txt"
  printf '%s\n' "$EXPECTED_ANALYZER_SHA256" > "$PROV/local_analyzer_review_sha256.txt"
  sha256sum "$PROBE" "$COMMON" "$BUDGET" "$MANIFEST" "$CACHE" "$LAUNCHER" \
    "$ENV_SOURCE" "$PROV"/stock_seed*_config.json > "$PROV/source_sha256.txt"
  for seed in 42 43 44; do
    checkpoint=$(checkpoint_path "$seed")
    printf '%s  %s\n' "$(checkpoint_sha "$seed")" "$checkpoint"
  done > "$PROV/checkpoint_sha256.txt"
  stat -c '%y %s %n' "$DATASET" > "$PROV/tworoom_dataset_stat.txt"
  sha_of "$DATASET" > "$PROV/tworoom_dataset_sha256.txt"

  require_plain_stock_configs
  require_snapshot_hashes
  cd "$JEPA_WT"
  "$PY" -m py_compile "$PROBE" "$COMMON" "$BUDGET" "$CACHE"
  PYTHONPATH="$selected_swm${PYTHONPATH:+:$PYTHONPATH}" \
    ADADIM_STABLEWM_REPO="$selected_swm" \
    "$PY" "$CACHE" --help > "$PROV/cache_help.txt"
  atomic_text "$STATE/setup.done.json" \
    "{\"run_id\":\"$RUN_ID\",\"state\":\"SETUP_DONE\",\"at\":\"$(iso_now)\",\"jepa_commit\":\"$(<"$PROV/jepawms_git_commit.txt")\",\"jepa_base_commit\":\"$JEPA_BASE_COMMIT\",\"stable_commit\":\"$STABLE_COMMIT\",\"baseline_manifest_sha256\":\"$EXPECTED_BASELINE_SHA256\",\"local_analyzer_review_sha256\":\"$EXPECTED_ANALYZER_SHA256\"}"
  setup_ok=1
  trap - EXIT
  echo "setup_ok root=$ROOT stable_worktree=$selected_swm"
  echo "local_baseline_manifest_sha256=$EXPECTED_BASELINE_SHA256"
  echo "local_analyzer_review_sha256=$EXPECTED_ANALYZER_SHA256 (record-only; not uploaded)"
}


gpu_free() {
  local gpu=$1
  local values mem util
  [[ "$gpu" != 6 ]] || die "GPU 6 is forbidden"
  values=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')
  IFS=, read -r mem util <<< "$values"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
    || die "cannot parse GPU $gpu state: $values"
  [[ "$mem" -le 500 && "$util" -le 20 ]] \
    || die "GPU $gpu is occupied: memory=${mem}MiB util=${util}%"
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
  [[ -f "$STATE/cache.attempted.json" ]] \
    || die "cache worker lacks an immutable launch claim"
  [[ -n "${LAUNCH_TOKEN:-}" && -f "$STATE/cache.launch.token" \
      && "$LAUNCH_TOKEN" == "$(<"$STATE/cache.launch.token")" ]] \
    || die "cache worker launch token mismatch"
  WORKER_DONE_MARKER="$STATE/cache.done.json"
  WORKER_FAILED_MARKER="$STATE/cache.failed.json"
  WORKER_FAILURE_FIELDS='"state":"CACHE_FAILED","gpu":0,"pid":'"$$"
  arm_worker_traps
  require_setup
  require_dataset_identity
  [[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || die "cache fit must use physical GPU 0"
  [[ ! -e "$STATE/cache.running.json" && ! -e "$STATE/cache.done.json" \
      && ! -e "$STATE/cache.failed.json" ]] || die "cache state already exists"
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
  [[ ! -e "$ROOT/probe_caches" && ! -e "$ROOT/probes" ]] \
    || die "cache output collision"
  [[ ! -e "$STATE/cache.attempted.json" && ! -e "$STATE/cache.running.json" \
      && ! -e "$STATE/cache.done.json" && ! -e "$STATE/cache.failed.json" ]] \
    || die "cache batch already attempted"
  [[ ! -e "$ROOT/cache_pid.txt" && ! -e "$STATE/cache.launch.token" \
      && ! -e "$STATE/cache.launch.record.json" ]] || die "cache launch ledger collision"
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
        "{\"state\":\"CACHE_LAUNCH_FAILED\",\"pid\":$pid,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" \
        >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap cache_launch_trap EXIT
  launch_token=$(< /proc/sys/kernel/random/uuid)
  atomic_text "$STATE/cache.launch.token" "$launch_token"
  atomic_text "$STATE/cache.attempted.json" \
    "{\"state\":\"CACHE_ATTEMPTED\",\"gpu\":0,\"launch_token\":\"$launch_token\",\"dataset_sha256\":\"$(<"$PROV/tworoom_dataset_sha256.txt")\",\"baseline_manifest_sha256\":\"$EXPECTED_BASELINE_SHA256\",\"at\":\"$(iso_now)\"}"
  reserve_file "$LOGS/cache_fit.log"
  CUDA_VISIBLE_DEVICES=0 LAUNCH_TOKEN="$launch_token" \
    setsid nohup bash "$LAUNCHER" run-cache 9>&- >> "$LOGS/cache_fit.log" 2>&1 &
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


cache_integrity() {
  local mode=$1
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    "$PY" - "$CACHE" "$ROOT" "$mode" "$EXPECTED_BASELINE_SHA256" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
root = Path(sys.argv[2])
mode = sys.argv[3]
baseline_sha = sys.argv[4]
if mode not in {"fit", "create", "check"}:
    raise SystemExit(f"invalid cache-integrity mode: {mode}")

spec = importlib.util.spec_from_file_location("stock_tworoom_cache_reviewed", cache_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
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
    or attempted.get("baseline_manifest_sha256") != baseline_sha
    or record.get("state") != "CACHE_LAUNCH_RECORD"
    or launched.get("state") != "CACHE_LAUNCHED"
    or running.get("state") != "CACHE_RUNNING"
    or done.get("state") != "CACHE_DONE"
    or attempted.get("launch_token") != record.get("launch_token")
    or attempted.get("launch_token") != running.get("launch_token")
    or int(record.get("gpu", -1)) != 0
    or int(attempted.get("gpu", -1)) != 0
    or str(record.get("pid")) != pid
    or str(record.get("boot_id")) != boot
    or str(record.get("start_ticks")) != ticks
    or int(launched.get("pid", -1)) != int(pid)
    or int(running.get("pid", -1)) != int(pid)
    or int(done.get("pid", -1)) != int(pid)
):
    raise SystemExit("cache attempted/record/launched/running/done/pid ledgers disagree")
summary_path = root / "probes" / "probe_fit_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_sources = {
    "probe_runner": module.PROBE_RUNNER_SHA256,
    "probe_common": module.PROBE_COMMON_SHA256,
    "budget_runner": module.BUDGET_RUNNER_SHA256,
    "task_manifest": module.TASK_MANIFEST_SHA256,
    "baseline_manifest": baseline_sha,
    "baseline_eval_arrays": module.BASELINE_EVAL_ARRAYS_SHA256,
}
if (
    summary.get("schema_version") != module.SCHEMA_VERSION
    or summary.get("run_id") != module.RUN_ID
    or summary.get("n_cache_artifacts") != 30
    or summary.get("source_sha256") != expected_sources
):
    raise SystemExit("fit-summary identity/source/count drift")
if summary.get("checkpoint_sha256") != {str(k): v for k, v in module.CHECKPOINT_SHA256.items()}:
    raise SystemExit("fit-summary stock checkpoint hashes drift")
if summary.get("checkpoint_config_sha256") != {str(k): v for k, v in module.CONFIG_SHA256.items()}:
    raise SystemExit("fit-summary stock config hashes drift")
module._validate_dataset_unit_attestation(summary.get("dataset_unit_attestation"))

records = summary.get("records", [])
expected_keys = {
    (family, train_seed, eval_seed)
    for family in ("s0", "s1")
    for train_seed in (42, 43, 44)
    for eval_seed in (42, 43, 44, 45, 46)
}
keys = {(str(row["family"]), int(row["train_seed"]), int(row["eval_seed"])) for row in records}
if len(records) != 30 or keys != expected_keys:
    raise SystemExit("cache record matrix differs from 2x3x5")
artifact_map = {}
for row in records:
    family = str(row["family"])
    train_seed = int(row["train_seed"])
    eval_seed = int(row["eval_seed"])
    relative = Path(str(row["path"]))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("probe_caches",):
        raise SystemExit(f"unsafe cache path: {relative}")
    metadata, _ = module.load_npz_cache(root / relative, expected_sha256=str(row["sha256"]))
    module._validate_loaded_metadata(
        metadata, family=family, train_seed=train_seed, eval_seed=eval_seed
    )
    module._validate_dataset_unit_attestation(metadata.get("dataset_unit_attestation"))
    if row.get("gate_rows") != metadata.get("gate_rows"):
        raise SystemExit(f"fit-summary gate rows differ from cache metadata: {relative}")
    if row.get("checkpoint_sha256") != module.CHECKPOINT_SHA256[train_seed]:
        raise SystemExit(f"fit-summary checkpoint binding drift: {relative}")
    artifact_map[str(relative)] = str(row["sha256"])
artifact_map = {key: artifact_map[key] for key in sorted(artifact_map)}
if summary.get("cache_artifact_sha256") != artifact_map:
    raise SystemExit("fit-summary artifact map differs from verified immutable caches")

prediction_path = root / "probes" / "prospective_gate_predictions.json"
validated_path = root / "cache_validated.json"
if mode == "fit":
    if prediction_path.exists() or validated_path.exists():
        raise SystemExit("prospective/cache validation artifact existed before publication")
    print("cache_fit_integrity_ok n=30")
    raise SystemExit(0)

if not prediction_path.is_file():
    raise SystemExit("prospective prediction artifact is absent")
prediction_sha = module.sha256_file(prediction_path)
recorded_prediction = (root / "provenance" / "prospective_gate_predictions_sha256.txt").read_text(
    encoding="utf-8"
).strip()
recorded_baseline = (root / "provenance" / "baseline_manifest_sha256.txt").read_text(
    encoding="utf-8"
).strip()
if prediction_sha != recorded_prediction or recorded_baseline != baseline_sha:
    raise SystemExit("prediction/baseline provenance hash drift")
published = json.loads(prediction_path.read_text(encoding="utf-8"))
recomputed = module.recompute_prospective_predictions(root, summary)
if published != recomputed:
    raise SystemExit("published prospective predictions differ from the sealed caches")
if list(published.get("cells", {})) != list(module.CELL_SPECS):
    raise SystemExit("prospective prediction cell order/set drift")

payload = {
    "run_id": module.RUN_ID,
    "state": "CACHE_AND_PREDICTIONS_VALIDATED",
    "n_cache_artifacts": 30,
    "fit_summary_sha256": module.sha256_file(summary_path),
    "cache_artifact_set_sha256": module.sha256_json(artifact_map),
    "prospective_gate_predictions_sha256": prediction_sha,
    "baseline_manifest_sha256": baseline_sha,
    "prediction_cells": list(module.CELL_SPECS),
}
if mode == "create":
    module.atomic_write_json(validated_path, payload)
else:
    frozen = json.loads(validated_path.read_text(encoding="utf-8"))
    if frozen != payload:
        raise SystemExit("cache_validated.json differs from independently reconstructed payload")
print(json.dumps(payload, indent=2, allow_nan=False))
PY
  )
}


validate_cache() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  [[ -f "$STATE/cache.attempted.json" && -f "$STATE/cache.launch.record.json" \
      && -f "$STATE/cache.launched.json" && -f "$ROOT/cache_pid.txt" ]] \
    || die "cache launch ledger is incomplete"
  [[ ! -e "$STATE/cache.launch.failed.json" ]] || die "cache parent launch failed"
  [[ -f "$STATE/cache.done.json" ]] || die "cache process is not done"
  [[ ! -e "$STATE/cache.failed.json" ]] || die "cache process failed"
  [[ ! -e "$ROOT/cache_validated.json" ]] || die "cache was already validated"
  [[ ! -e "$STATE/smoke.attempted.json" && ! -e "$STATE/full.attempted.json" ]] \
    || die "planning was attempted before prospective prediction publication"
  [[ -z "$(find "$ROOT/smoke" "$ROOT/full" -mindepth 1 -print -quit)" ]] \
    || die "planning output exists before prospective prediction publication"

  cache_integrity fit
  [[ ! -e "$ROOT/probes/prospective_gate_predictions.json" \
      && ! -e "$PROV/prospective_gate_predictions_sha256.txt" ]] \
    || die "prospective prediction publication collision"
  reserve_file "$LOGS/publish_predictions.log"
  cd "$JEPA_WT"
  write_command "$COMMANDS/publish_predictions.command.txt" \
    "$PY" "$CACHE" publish-predictions --run-root "$ROOT"
  "$PY" "$CACHE" publish-predictions --run-root "$ROOT" \
    >> "$LOGS/publish_predictions.log" 2>&1
  [[ -f "$ROOT/probes/prospective_gate_predictions.json" ]] \
    || die "cache publisher did not emit prospective predictions"
  atomic_text "$PROV/prospective_gate_predictions_sha256.txt" \
    "$(sha_of "$ROOT/probes/prospective_gate_predictions.json")"
  cache_integrity create
  echo "cache_and_predictions_validated n=30"
  echo "baseline_manifest_sha256=$EXPECTED_BASELINE_SHA256"
  echo "prospective_gate_predictions_sha256=$(<"$PROV/prospective_gate_predictions_sha256.txt")"
}


require_validated_cache_integrity() {
  [[ -f "$ROOT/cache_validated.json" ]] || die "cache/prediction validation is required"
  cache_integrity check >/dev/null
}


cell_gpu() {
  case "$1" in
    t1_linear_pure) echo 0 ;;
    t2_linear_hybrid_l05) echo 1 ;;
    t3_mlp_hybrid_l05) echo 2 ;;
    t4_mlp_pure) echo 3 ;;
    *) die "unknown cell: $1" ;;
  esac
}


cell_is_fixed() {
  local wanted=$1 cell
  for cell in "${CELLS[@]}"; do
    [[ "$cell" == "$wanted" ]] && return 0
  done
  return 1
}


verify_runtime_attestation() {
  local phase=$1
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    "$PY" - "$ROOT" "$phase" "$EXPECTED_BASELINE_SHA256" "$EXPECTED_ANALYZER_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
phase = sys.argv[2]
baseline_sha = sys.argv[3]
analyzer_sha = sys.argv[4]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
runtime = root / f"{phase}_runtime_validated.json"
attestation = root / f"{phase}_validation_attestation.json"
if not runtime.is_file() or not attestation.is_file():
    raise SystemExit(f"{phase} runtime validation/attestation is absent")
expected = {
    "run_id": root.name,
    "state": f"{phase.upper()}_VALIDATION_ATTESTED",
    "runtime_validation_sha256": sha(runtime),
    "cache_validated_sha256": sha(root / "cache_validated.json"),
    "prospective_gate_predictions_sha256": (
        root / "provenance" / "prospective_gate_predictions_sha256.txt"
    ).read_text(encoding="utf-8").strip(),
    "baseline_manifest_sha256": baseline_sha,
    "dataset_sha256": (root / "provenance" / "tworoom_dataset_sha256.txt").read_text(
        encoding="utf-8"
    ).strip(),
    "local_analyzer_review_sha256": analyzer_sha,
    "smoke_validation_attestation_sha256": (
        "" if phase == "smoke" else sha(root / "smoke_validation_attestation.json")
    ),
}
actual = json.loads(attestation.read_text(encoding="utf-8"))
if actual != expected:
    raise SystemExit(f"{phase} validation attestation drift")
print(f"{phase}_attestation_ok")
PY
  )
}


phase_prerequisites() {
  local phase=$1
  require_validated_cache_integrity
  # The parent launch/validation controller re-hashes the 12-GB dataset once.
  # Workers re-check the immutable stat to avoid four simultaneous full reads.
  require_dataset_stat
  if [[ "$phase" == full ]]; then
    verify_runtime_attestation smoke >/dev/null
  fi
}


preflight_batch() {
  local phase=$1
  local cell gpu
  [[ ! -e "$STATE/${phase}.attempted.json" && ! -e "$STATE/${phase}.launched.json" \
      && ! -e "$STATE/${phase}.launch.failed.json" ]] \
    || die "$phase batch was already attempted"
  [[ ! -e "$ROOT/${phase}_pids.txt" && ! -e "$STATE/${phase}.launch.records" \
      && ! -e "$STATE/${phase}.launch.token" ]] || die "$phase launch ledger collision"
  for cell in "${CELLS[@]}"; do
    gpu=$(cell_gpu "$cell")
    gpu_free "$gpu"
    [[ ! -e "$ROOT/$phase/tworoom_${cell}" \
        && ! -e "$ROOT/$phase/tworoom_${cell}.partial" ]] \
      || die "$phase output collision for $cell"
    [[ ! -e "$STATE/${phase}_${cell}.running.json" \
        && ! -e "$STATE/${phase}_${cell}.attempted.json" \
        && ! -e "$STATE/${phase}_${cell}.done.json" \
        && ! -e "$STATE/${phase}_${cell}.failed.json" ]] \
      || die "$phase state collision for $cell"
    [[ ! -e "$COMMANDS/${phase}_${cell}.command.txt" \
        && ! -e "$LOGS/${phase}_${cell}.log" ]] \
      || die "$phase command/log collision for $cell"
  done
}


run_cell_internal() {
  local phase=$1
  local cell=$2
  local gpu out_partial out_final swm jepa_commit baseline_sha prediction_sha rc
  gpu=$(cell_gpu "$cell")
  source ~/.jepawm_env
  [[ -f "$STATE/${phase}.attempted.json" \
      && -f "$STATE/${phase}_${cell}.attempted.json" ]] \
    || die "$phase/$cell worker lacks immutable launch claims"
  [[ -n "${LAUNCH_TOKEN:-}" && -f "$STATE/${phase}.launch.token" \
      && "$LAUNCH_TOKEN" == "$(<"$STATE/${phase}.launch.token")" ]] \
    || die "$phase/$cell worker launch token mismatch"
  WORKER_DONE_MARKER="$STATE/${phase}_${cell}.done.json"
  WORKER_FAILED_MARKER="$STATE/${phase}_${cell}.failed.json"
  WORKER_FAILURE_FIELDS='"state":"'"${phase^^}"'_FAILED","cell":"'"$cell"'","gpu":'"$gpu"',"pid":'"$$"
  arm_worker_traps
  require_setup
  phase_prerequisites "$phase"
  cell_is_fixed "$cell" || die "cell is not one of frozen unconditional T1-T4: $cell"
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "$gpu" ]] \
    || die "$cell must use physical GPU $gpu"
  out_partial="$ROOT/$phase/tworoom_${cell}.partial"
  out_final="$ROOT/$phase/tworoom_${cell}"
  [[ ! -e "$out_partial" && ! -e "$out_final" ]] \
    || die "cell output collision: $phase $cell"
  local self_boot self_ticks
  self_boot=$(boot_id)
  self_ticks=$(proc_start_ticks $$)
  atomic_text "$STATE/${phase}_${cell}.running.json" \
    "{\"state\":\"${phase^^}_RUNNING\",\"cell\":\"$cell\",\"pid\":$$,\"gpu\":$gpu,\"boot_id\":\"$self_boot\",\"start_ticks\":\"$self_ticks\",\"launch_token\":\"$LAUNCH_TOKEN\",\"at\":\"$(iso_now)\"}"
  swm=$(stable_worktree)
  jepa_commit=$(<"$PROV/jepawms_git_commit.txt")
  baseline_sha=$(<"$PROV/baseline_manifest_sha256.txt")
  prediction_sha=$(<"$PROV/prospective_gate_predictions_sha256.txt")
  cd "$JEPA_WT"
  write_command "$COMMANDS/${phase}_${cell}.command.txt" "$PY" "$CACHE" run-cached \
    --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
    --task-manifest "$MANIFEST" --run-root "$ROOT" --out-root "$out_partial" \
    --cell "$cell" --mode "$phase" --jepa-commit "$jepa_commit" \
    --stable-commit "$STABLE_COMMIT" --baseline-manifest-sha256 "$baseline_sha" \
    --prediction-sha256 "$prediction_sha" --device cuda
  set +e
  PYTHONPATH="$swm${PYTHONPATH:+:$PYTHONPATH}" ADADIM_STABLEWM_REPO="$swm" WANDB_MODE=disabled \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    setsid "$PY" "$CACHE" run-cached \
      --probe-runner "$PROBE" --probe-common "$COMMON" --budget-runner "$BUDGET" \
      --task-manifest "$MANIFEST" --run-root "$ROOT" --out-root "$out_partial" \
      --cell "$cell" --mode "$phase" --jepa-commit "$jepa_commit" \
      --stable-commit "$STABLE_COMMIT" --baseline-manifest-sha256 "$baseline_sha" \
      --prediction-sha256 "$prediction_sha" --device cuda &
  WORKER_CHILD_PID=$!
  WORKER_CHILD_TICKS=$(proc_start_ticks "$WORKER_CHILD_PID")
  wait "$WORKER_CHILD_PID"
  rc=$?
  WORKER_CHILD_PID=
  WORKER_CHILD_TICKS=
  set -e
  if [[ "$rc" -eq 0 ]]; then
    local required
    for required in run_metadata.json cell_summaries.json \
      probe_physical_cem_episode_records.csv probe_physical_cem_replan_records.csv \
      probe_physical_cem_summary.csv probe_physical_cem_summary.md; do
      [[ -f "$out_partial/$required" ]] || rc=92
    done
  fi
  if [[ "$rc" -eq 0 ]]; then
    mv -- "$out_partial" "$out_final"
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
  phase_prerequisites "$phase"
  preflight_batch "$phase"
  local dataset_sha cache_sha prediction_sha baseline_sha smoke_attestation_sha launch_token
  dataset_sha=$(<"$PROV/tworoom_dataset_sha256.txt")
  cache_sha=$(sha_of "$ROOT/cache_validated.json")
  prediction_sha=$(<"$PROV/prospective_gate_predictions_sha256.txt")
  baseline_sha=$(<"$PROV/baseline_manifest_sha256.txt")
  smoke_attestation_sha=
  if [[ "$phase" == full ]]; then
    smoke_attestation_sha=$(sha_of "$ROOT/smoke_validation_attestation.json")
  fi
  local cell gpu pid worker_boot worker_ticks launched_count=0 launch_complete=0
  batch_launch_trap() {
    local rc=$?
    trap - EXIT
    if [[ "$launch_complete" -ne 1 && ! -e "$STATE/${phase}.launch.failed.json" ]]; then
      set +e
      try_atomic_text "$STATE/${phase}.launch.failed.json" \
        "{\"state\":\"${phase^^}_LAUNCH_FAILED\",\"launched_count\":$launched_count,\"exit_code\":$rc,\"at\":\"$(iso_now)\"}" \
        >/dev/null 2>&1
    fi
    exit "$rc"
  }
  trap batch_launch_trap EXIT
  launch_token=$(< /proc/sys/kernel/random/uuid)
  atomic_text "$STATE/${phase}.launch.token" "$launch_token"
  atomic_text "$STATE/${phase}.attempted.json" \
    "{\"state\":\"${phase^^}_ATTEMPTED\",\"cells\":$CELLS_JSON,\"cache_validated_sha256\":\"$cache_sha\",\"prospective_gate_predictions_sha256\":\"$prediction_sha\",\"baseline_manifest_sha256\":\"$baseline_sha\",\"launch_token\":\"$launch_token\",\"dataset_sha256\":\"$dataset_sha\",\"at\":\"$(iso_now)\",\"smoke_validation_attestation_sha256\":\"$smoke_attestation_sha\"}"
  reserve_file "$ROOT/${phase}_pids.txt"
  mkdir "$STATE/${phase}.launch.records"
  echo "launch_binding phase=$phase baseline_manifest_sha256=$baseline_sha prospective_gate_predictions_sha256=$prediction_sha"
  for cell in "${CELLS[@]}"; do
    gpu=$(cell_gpu "$cell")
    reserve_file "$LOGS/${phase}_${cell}.log"
    atomic_text "$STATE/${phase}_${cell}.attempted.json" \
      "{\"state\":\"${phase^^}_CELL_ATTEMPTED\",\"cell\":\"$cell\",\"gpu\":$gpu,\"at\":\"$(iso_now)\"}"
    CUDA_VISIBLE_DEVICES="$gpu" LAUNCH_TOKEN="$launch_token" \
      setsid nohup bash "$LAUNCHER" run-cell "$phase" "$cell" 9>&- \
      >> "$LOGS/${phase}_${cell}.log" 2>&1 &
    pid=$!
    worker_boot=$(boot_id)
    worker_ticks=$(proc_start_ticks "$pid") || worker_ticks=unavailable
    atomic_text "$STATE/${phase}.launch.records/${cell}.json" \
      "{\"state\":\"${phase^^}_LAUNCH_RECORD\",\"cell\":\"$cell\",\"gpu\":$gpu,\"pid\":$pid,\"boot_id\":\"$worker_boot\",\"start_ticks\":\"$worker_ticks\",\"launch_token\":\"$launch_token\",\"at\":\"$(iso_now)\"}"
    printf '%s %s %s %s %s\n' "$cell" "$gpu" "$pid" "$worker_boot" "$worker_ticks" \
      >> "$ROOT/${phase}_pids.txt"
    launched_count=$((launched_count + 1))
  done
  atomic_text "$STATE/${phase}.launched.json" \
    "{\"state\":\"${phase^^}_LAUNCHED\",\"cells\":$CELLS_JSON,\"launched_count\":$launched_count,\"at\":\"$(iso_now)\"}"
  launch_complete=1
  trap - EXIT
  echo "started_$phase"
  cat "$ROOT/${phase}_pids.txt"
}


require_phase_done() {
  local phase=$1
  local cell required expected_count=0 pid_count record_count
  [[ -f "$STATE/${phase}.launched.json" ]] || die "$phase launch did not complete"
  [[ ! -e "$STATE/${phase}.launch.failed.json" ]] || die "$phase launch failed"
  for cell in "${CELLS[@]}"; do
    expected_count=$((expected_count + 1))
    [[ -f "$STATE/${phase}.launch.records/${cell}.json" ]] \
      || die "$phase launch record missing: $cell"
    [[ -f "$STATE/${phase}_${cell}.done.json" ]] \
      || die "$phase cell not done: $cell"
    [[ ! -e "$STATE/${phase}_${cell}.failed.json" ]] \
      || die "$phase cell failed: $cell"
    [[ -d "$ROOT/$phase/tworoom_${cell}" ]] \
      || die "$phase final output directory missing: $cell"
    [[ ! -e "$ROOT/$phase/tworoom_${cell}.partial" ]] \
      || die "$phase partial output remains: $cell"
    for required in run_metadata.json cell_summaries.json \
      probe_physical_cem_episode_records.csv probe_physical_cem_replan_records.csv \
      probe_physical_cem_summary.csv probe_physical_cem_summary.md; do
      [[ -f "$ROOT/$phase/tworoom_${cell}/$required" ]] \
        || die "$phase output missing for $cell: $required"
    done
  done
  [[ -f "$ROOT/${phase}_pids.txt" ]] || die "$phase pid manifest missing"
  pid_count=$(grep -cve '^[[:space:]]*$' "$ROOT/${phase}_pids.txt")
  record_count=$(find "$STATE/${phase}.launch.records" -maxdepth 1 -type f -name '*.json' | wc -l)
  [[ "$pid_count" -eq "$expected_count" && "$record_count" -eq "$expected_count" ]] \
    || die "$phase launch ledger count drift: expected=$expected_count pids=$pid_count records=$record_count"
}


runtime_validate() {
  local phase=$1
  local swm
  swm=$(stable_worktree)
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    PYTHONPATH="$swm${PYTHONPATH:+:$PYTHONPATH}" ADADIM_STABLEWM_REPO="$swm" \
      "$PY" - "$CACHE" "$ROOT" "$phase" "$EXPECTED_BASELINE_SHA256" <<'PY'
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

cache_path = Path(sys.argv[1])
root = Path(sys.argv[2])
mode = sys.argv[3]
baseline_sha = sys.argv[4]
if mode not in {"smoke", "full"}:
    raise SystemExit(f"invalid runtime validation mode: {mode}")

spec = importlib.util.spec_from_file_location("stock_tworoom_cache_runtime", cache_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cells = list(module.CELL_SPECS)
expected_dirs = {f"tworoom_{cell}" for cell in cells}
phase_root = root / mode
actual_dirs = {path.name for path in phase_root.iterdir() if path.is_dir()}
if actual_dirs != expected_dirs:
    raise SystemExit(f"{mode} cell directories drift: {actual_dirs} != {expected_dirs}")
barred = [
    str(path) for path in root.rglob("*")
    if path.is_file() and ("gate_barred" in path.name.lower() or "gate-barred" in path.name.lower())
]
if barred:
    raise SystemExit(f"all-cells launch policy violated: {barred}")

fit_path = root / "probes" / "probe_fit_summary.json"
fit = json.loads(fit_path.read_text(encoding="utf-8"))
lookup = module._record_lookup(fit)
prediction_path = root / "probes" / "prospective_gate_predictions.json"
prediction_sha = module.sha256_file(prediction_path)
prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
if prediction != module.recompute_prospective_predictions(root, fit):
    raise SystemExit("prospective prediction artifact drift")
experiment_commit = (root / "provenance" / "jepawms_git_commit.txt").read_text(
    encoding="utf-8"
).strip()
dataset_sha = (root / "provenance" / "tworoom_dataset_sha256.txt").read_text(
    encoding="utf-8"
).strip()
cache_validated_sha = module.sha256_file(root / "cache_validated.json")
attempted = json.loads((root / "state" / f"{mode}.attempted.json").read_text(encoding="utf-8"))
launched = json.loads((root / "state" / f"{mode}.launched.json").read_text(encoding="utf-8"))
expected_attempted = {
    "state": f"{mode.upper()}_ATTEMPTED",
    "cells": cells,
    "cache_validated_sha256": cache_validated_sha,
    "prospective_gate_predictions_sha256": prediction_sha,
    "baseline_manifest_sha256": baseline_sha,
}
for field, expected in expected_attempted.items():
    if attempted.get(field) != expected:
        raise SystemExit(f"{mode} attempted-state binding drift: {field}")
if not attempted.get("launch_token") or not attempted.get("at"):
    raise SystemExit(f"{mode} attempted-state token/timestamp is absent")
if attempted.get("dataset_sha256") != dataset_sha:
    raise SystemExit(f"{mode} attempted-state dataset hash drift")
smoke_attestation = attempted.get("smoke_validation_attestation_sha256")
if mode == "smoke":
    if smoke_attestation != "":
        raise SystemExit("smoke attempted state must precede its attestation")
else:
    expected_smoke_attestation = module.sha256_file(root / "smoke_validation_attestation.json")
    if smoke_attestation != expected_smoke_attestation:
        raise SystemExit("full attempted state is not bound to the smoke attestation")
if (
    launched.get("state") != f"{mode.upper()}_LAUNCHED"
    or launched.get("cells") != cells
    or int(launched.get("launched_count", -1)) != 4
):
    raise SystemExit(f"{mode} launched-state drift")
pid_lines = [line.split() for line in (root / f"{mode}_pids.txt").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]
if len(pid_lines) != 4 or [line[0] for line in pid_lines] != cells:
    raise SystemExit(f"{mode} pid ledger does not contain ordered unconditional T1-T4")
for expected_gpu, fields in enumerate(pid_lines):
    if len(fields) != 5:
        raise SystemExit(f"{mode} malformed pid ledger row: {fields}")
    cell, gpu, pid, boot, ticks = fields
    record = json.loads(
        (root / "state" / f"{mode}.launch.records" / f"{cell}.json").read_text(
            encoding="utf-8"
        )
    )
    cell_attempted = json.loads(
        (root / "state" / f"{mode}_{cell}.attempted.json").read_text(encoding="utf-8")
    )
    cell_done = json.loads(
        (root / "state" / f"{mode}_{cell}.done.json").read_text(encoding="utf-8")
    )
    if (
        int(gpu) != expected_gpu
        or record.get("state") != f"{mode.upper()}_LAUNCH_RECORD"
        or record.get("cell") != cell
        or int(record.get("gpu", -1)) != expected_gpu
        or str(record.get("pid")) != pid
        or str(record.get("boot_id")) != boot
        or str(record.get("start_ticks")) != ticks
        or record.get("launch_token") != attempted.get("launch_token")
        or cell_attempted.get("state") != f"{mode.upper()}_CELL_ATTEMPTED"
        or cell_attempted.get("cell") != cell
        or int(cell_attempted.get("gpu", -1)) != expected_gpu
        or cell_done.get("state") != f"{mode.upper()}_DONE"
        or cell_done.get("cell") != cell
        or int(cell_done.get("gpu", -1)) != expected_gpu
        or str(cell_done.get("pid")) != pid
    ):
        raise SystemExit(f"{mode}/{cell} attempted/record/done/pid ledger drift")

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"non-finite {label}: {value!r}")
    return number

def success_value(value, label):
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true"}:
        return 1
    if normalized in {"0", "0.0", "false"}:
        return 0
    raise SystemExit(f"invalid binary {label}: {value!r}")

def finite_json(value, label="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"non-finite JSON number at {label}")

def budget_maps(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_by_budget") and isinstance(child, dict):
                yield name, child
            yield from budget_maps(child, name)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from budget_maps(child, f"{prefix}[{index}]")

required = (
    "run_metadata.json",
    "cell_summaries.json",
    "probe_physical_cem_episode_records.csv",
    "probe_physical_cem_replan_records.csv",
    "probe_physical_cem_summary.csv",
    "probe_physical_cem_summary.md",
)
output_hashes = {}
expected_total = 2 if mode == "smoke" else 750
train_seeds = [42] if mode == "smoke" else [42, 43, 44]
eval_seeds = [42] if mode == "smoke" else [42, 43, 44, 45, 46]
num_eval = 2 if mode == "smoke" else 50
expected_pairs = {(ts, es) for ts in train_seeds for es in eval_seeds}

for cell in cells:
    cell_root = phase_root / f"tworoom_{cell}"
    for name in required:
        if not (cell_root / name).is_file():
            raise SystemExit(f"missing {mode}/{cell}/{name}")
    output_hashes[cell] = {name: module.sha256_file(cell_root / name) for name in required}
    metadata = json.loads((cell_root / "run_metadata.json").read_text(encoding="utf-8"))
    finite_json(metadata, f"{cell}.metadata")
    cell_spec = module.CELL_SPECS[cell]
    exact = {
        "run_id": module.RUN_ID,
        "cell": cell,
        "mode": mode,
        "tasks": ["tworoom"],
        "arms": ["stock"],
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "num_eval": num_eval,
        "methods": ["fixed_full"],
        "budgets": [24, 48, 96, 192],
        "probe_train_samples": 3000,
        "probe_batch_size": 128,
        "probe_seed": 1703,
        "probe_budget": 192,
        "hybrid_lambda": cell_spec["hybrid_lambda"],
        "beta_tr": 0.0,
        "probe_mode": "linear",
        "probe_family": module.FAMILY_SPECS[cell_spec["family"]]["probe_family"],
        "target_variant": "success",
        "probe_cache_family": cell_spec["family"],
        "precomputed_probes": True,
        "planning_probe_fit_calls": 0,
        "planning_probe_cache_load_calls": 1 if mode == "smoke" else 15,
        "cache_exclusion_basis": "full_n50",
        "runtime_smoke_exclusion_subset": mode == "smoke",
        "checkpoint_epoch": 10,
        "probe_runner_sha256": module.PROBE_RUNNER_SHA256,
        "probe_common_sha256": module.PROBE_COMMON_SHA256,
        "budget_runner_sha256": module.BUDGET_RUNNER_SHA256,
        "task_manifest_sha256": module.TASK_MANIFEST_SHA256,
        "baseline_manifest_sha256": baseline_sha,
        "prospective_gate_predictions_sha256": prediction_sha,
        "prospective_gate_cell_prediction": prediction["cells"][cell],
        "stock_full_a192_planning_shim": True,
        "stock_full_a192_lower_budget_hard_reject": True,
        "jepa_wms_commit": experiment_commit,
        "jepa_wms_base_commit": module.JEPA_WMS_BASE_COMMIT,
        "stable_worldmodel_commit": module.STABLE_WORLDMODEL_COMMIT,
        "worktree_clean": True,
    }
    for field, expected in exact.items():
        if metadata.get(field) != expected:
            raise SystemExit(f"{cell} metadata drift {field}: {metadata.get(field)!r} != {expected!r}")
    if finite(metadata.get("elapsed_sec"), f"{cell}.elapsed_sec") <= 0:
        raise SystemExit(f"{cell} elapsed_sec is not positive")
    if metadata.get("checkpoint_sha256") != {
        str(seed): module.CHECKPOINT_SHA256[seed] for seed in train_seeds
    }:
        raise SystemExit(f"{cell} checkpoint hash binding drift")
    if metadata.get("checkpoint_config_sha256") != {
        str(seed): module.CONFIG_SHA256[seed] for seed in train_seeds
    }:
        raise SystemExit(f"{cell} checkpoint config binding drift")
    expected_cache_keys = {
        f"train_seed{ts}_eval_seed{es}" for ts in train_seeds for es in eval_seeds
    }
    hashes = metadata.get("probe_artifact_sha256", {})
    paths = metadata.get("probe_cache_paths", {})
    if set(hashes) != expected_cache_keys or set(paths) != expected_cache_keys:
        raise SystemExit(f"{cell} cache-binding key matrix drift")
    for ts, es in expected_pairs:
        key = f"train_seed{ts}_eval_seed{es}"
        record = lookup[(cell_spec["family"], ts, es)]
        if hashes[key] != record["sha256"] or paths[key] != record["path"]:
            raise SystemExit(f"{cell} cache hash/path drift for {key}")

    episodes = read_csv(cell_root / "probe_physical_cem_episode_records.csv")
    if len(episodes) != expected_total:
        raise SystemExit(f"{cell} expected {expected_total} episode rows, got {len(episodes)}")
    required_episode = {
        "task", "arm", "train_seed", "eval_seed", "method", "planning_budget",
        "env_index", "episode_idx", "eval_episode_idx", "eval_start_idx", "episode_success",
        "selected_probe_physical_cost_mean", "selected_latent_cost_active_budget_mean",
        "cem_final_topk_probe_cost_mean",
    }
    if not episodes or not required_episode.issubset(episodes[0]):
        raise SystemExit(f"{cell} episode CSV schema drift")
    pair_rows = {}
    runtime_arrays = []
    runtime_keys = []
    occurrences = Counter()
    for row in episodes:
        ts, es = int(row["train_seed"]), int(row["eval_seed"])
        env_index = int(row["env_index"])
        episode = int(row["eval_episode_idx"])
        start = int(row["eval_start_idx"])
        if (
            row["task"] != "tworoom" or row["arm"] != "stock"
            or row["method"] != "fixed_full" or int(float(row["planning_budget"])) != 192
            or int(row["episode_idx"]) != episode or (ts, es) not in expected_pairs
            or success_value(row["episode_success"], f"{cell}.episode_success") not in (0, 1)
        ):
            raise SystemExit(f"{cell} episode protocol drift at {(ts, es, env_index)}")
        for field in (
            "selected_probe_physical_cost_mean", "selected_latent_cost_active_budget_mean",
            "cem_final_topk_probe_cost_mean",
        ):
            finite(row[field], f"{cell}.{field}")
        pair_rows.setdefault((ts, es), []).append(row)
        runtime_arrays.append([ts, es, env_index, episode, start])
        base = (ts, es, episode)
        occurrence = occurrences[base]
        occurrences[base] += 1
        runtime_keys.append([ts, es, env_index, episode, occurrence])
    if set(pair_rows) != expected_pairs or {len(rows) for rows in pair_rows.values()} != {num_eval}:
        raise SystemExit(f"{cell} pair/count matrix drift")
    if mode == "full":
        by_seed = Counter()
        for (ts, _), rows in pair_rows.items():
            by_seed[ts] += len(rows)
        if dict(by_seed) != {42: 250, 43: 250, 44: 250}:
            raise SystemExit(f"{cell} per-train-seed counts drift: {dict(by_seed)}")
    for ts in train_seeds:
        for es in eval_seeds:
            rows = pair_rows[(ts, es)]
            record = lookup[(cell_spec["family"], ts, es)]
            cache_meta, arrays = module.load_npz_cache(
                root / record["path"], expected_sha256=record["sha256"]
            )
            module._validate_loaded_metadata(
                cache_meta, family=cell_spec["family"], train_seed=ts, eval_seed=es
            )
            if [int(row["env_index"]) for row in rows] != list(range(num_eval)):
                raise SystemExit(f"{cell} env_index order drift for {(ts, es)}")
            runtime_episode = [int(row["eval_episode_idx"]) for row in rows]
            runtime_start = [int(row["eval_start_idx"]) for row in rows]
            frozen_episode = arrays["planning_eval_episodes"].astype(int).tolist()
            frozen_start = arrays["planning_eval_start_idx"].astype(int).tolist()
            if mode == "full":
                if runtime_episode != frozen_episode:
                    raise SystemExit(f"{cell} eval episode array drift for {(ts, es)}")
                if runtime_start != frozen_start:
                    raise SystemExit(f"{cell} eval start array drift for {(ts, es)}")
            elif any(episode not in set(frozen_episode) for episode in runtime_episode):
                raise SystemExit(f"{cell} smoke episode is outside the full exclusion set")
    if mode == "full":
        if module.sha256_json(runtime_arrays) != "b93d3ce7d92ba185f3be58eb13eefe668b8e4add4e058fec40f2b5e3dff49357":
            raise SystemExit(f"{cell} ordered full runtime arrays differ from frozen baseline")
        if module.sha256_json(runtime_keys) != "315cec59ee560414dd94ec94157dc203e69d04d51d3ed9bdb812f0ee3482420c":
            raise SystemExit(f"{cell} occurrence-aware runtime keys differ from frozen baseline")

    summaries = json.loads((cell_root / "cell_summaries.json").read_text(encoding="utf-8"))
    if not isinstance(summaries, list) or len(summaries) != len(expected_pairs):
        raise SystemExit(f"{cell} cell summary count drift")
    finite_json(summaries, f"{cell}.summaries")
    seen = set()
    for summary in summaries:
        ts, es = int(summary["train_seed"]), int(summary["eval_seed"])
        if (ts, es) in seen or (ts, es) not in expected_pairs:
            raise SystemExit(f"{cell} duplicate/unexpected summary pair {(ts, es)}")
        seen.add((ts, es))
        expected_checkpoint = (
            "/home/ps/Code/yqy/DINO-WM/stablewm_home/checkpoints/"
            f"stock_lewm_tworoom_matched3_seed{ts}_20260615/weights_epoch_10.pt"
        )
        if (
            summary.get("task") != "tworoom" or summary.get("arm") != "stock"
            or int(summary.get("epoch", -1)) != 10 or summary.get("method") != "fixed_full"
            or int(summary.get("planning_budget", -1)) != 192
            or int(summary.get("num_eval", -1)) != num_eval
            or summary.get("checkpoint") != expected_checkpoint
        ):
            raise SystemExit(f"{cell} cell summary protocol drift for {(ts, es)}")
        pair_successes = sum(
            success_value(row["episode_success"], f"{cell}.episode_success")
            for row in pair_rows[(ts, es)]
        )
        expected_rate = 100.0 * pair_successes / num_eval
        if not math.isclose(finite(summary.get("success_rate"), f"{cell}.success_rate"), expected_rate):
            raise SystemExit(f"{cell} success_rate disagrees with episode rows for {(ts, es)}")
        solver = summary.get("solver_summary", {})
        adapter = solver.get("adapter_stats", {})
        if (
            int(adapter.get("active_budget", -1)) != 192
            or int(adapter.get("probe_budget", -1)) != 192
            or adapter.get("target_names") != ["agent_position"]
            or int(adapter.get("target_dim", -1)) != 2
            or not math.isclose(float(adapter.get("hybrid_lambda", -1)), cell_spec["hybrid_lambda"])
        ):
            raise SystemExit(f"{cell} solver adapter is not frozen A192")
        for label, budget_map in budget_maps(solver):
            for lower in ("24", "48", "96"):
                if lower in budget_map and finite(budget_map[lower], label) != 0.0:
                    raise SystemExit(f"{cell} lower-budget activity in {label}")
        latent = solver.get("latent_adapter_stats", {})
        for name in ("rollout_calls_by_budget", "cost_calls_by_budget", "candidates_by_budget"):
            budget_map = latent.get(name)
            if not isinstance(budget_map, dict) or set(budget_map) != {"24", "48", "96", "192"}:
                raise SystemExit(f"{cell} missing exact latent budget map {name}")
            if any(finite(budget_map[str(b)], name) != 0.0 for b in (24, 48, 96)):
                raise SystemExit(f"{cell} lower-budget activity in {name}")
            if finite(budget_map["192"], name) <= 0.0:
                raise SystemExit(f"{cell} has no A192 activity in {name}")

    replans = read_csv(cell_root / "probe_physical_cem_replan_records.csv")
    required_replan = {
        "task", "arm", "train_seed", "eval_seed", "method", "planning_budget",
        "probe_budget", "hybrid_lambda", "beta_tr", "env_index", "eval_episode_idx",
        "episode_success", "selected_probe_physical_cost",
        "selected_latent_cost_active_budget", "cem_final_topk_probe_cost", "solver_wall_sec",
    }
    if not replans or not required_replan.issubset(replans[0]):
        raise SystemExit(f"{cell} replan CSV schema drift")
    episode_by_key = {
        (r["train_seed"], r["eval_seed"], r["env_index"]): r for r in episodes
    }
    episode_keys = set(episode_by_key)
    replan_keys = Counter()
    for row in replans:
        key = (row["train_seed"], row["eval_seed"], row["env_index"])
        if key not in episode_keys:
            raise SystemExit(f"{cell} replan without episode: {key}")
        if (
            row["task"] != "tworoom" or row["arm"] != "stock" or row["method"] != "fixed_full"
            or int(float(row["planning_budget"])) != 192 or int(float(row["probe_budget"])) != 192
            or not math.isclose(float(row["hybrid_lambda"]), cell_spec["hybrid_lambda"])
            or float(row["beta_tr"]) != 0.0
            or row["eval_episode_idx"] != episode_by_key[key]["eval_episode_idx"]
            or success_value(row["episode_success"], f"{cell}.replan_success")
            != success_value(episode_by_key[key]["episode_success"], f"{cell}.episode_success")
        ):
            raise SystemExit(f"{cell} replan protocol drift: {key}")
        for field in (
            "selected_probe_physical_cost", "selected_latent_cost_active_budget",
            "cem_final_topk_probe_cost", "solver_wall_sec",
        ):
            finite(row[field], f"{cell}.replan.{field}")
        replan_keys[key] += 1
    if set(replan_keys) != episode_keys or min(replan_keys.values()) <= 0:
        raise SystemExit(f"{cell} episode without replan")
    for row in read_csv(cell_root / "probe_physical_cem_summary.csv"):
        for field, value in row.items():
            if value and any(token in field.lower() for token in ("cost", "rate", "sec", "mean", "std")):
                finite(value, f"{cell}.summary.{field}")

state = f"{mode.upper()}_RUNTIME_VALIDATED"
payload = {
    "schema_version": "stock_tworoom_remote_runtime_validation_v1",
    "run_id": module.RUN_ID,
    "state": state,
    "mode": mode,
    "cells": cells,
    "all_cells_ran_regardless_of_gate": True,
    "n_episode_rows_per_cell": expected_total,
    "cache_validated_sha256": cache_validated_sha,
    "prospective_gate_predictions_sha256": prediction_sha,
    "baseline_manifest_sha256": baseline_sha,
    "dataset_sha256": dataset_sha,
    "experiment_commit": experiment_commit,
    "cell_output_sha256": output_hashes,
}
module.atomic_write_json(root / f"{mode}_runtime_validated.json", payload)
print(json.dumps(payload, indent=2, allow_nan=False))
PY
  )
}


create_runtime_attestation() {
  local phase=$1
  (
    source ~/.jepawm_env
    cd "$JEPA_WT"
    "$PY" - "$CACHE" "$ROOT" "$phase" "$EXPECTED_BASELINE_SHA256" \
      "$EXPECTED_ANALYZER_SHA256" <<'PY'
import importlib.util
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
root = Path(sys.argv[2])
phase = sys.argv[3]
baseline_sha = sys.argv[4]
analyzer_sha = sys.argv[5]
spec = importlib.util.spec_from_file_location("stock_tworoom_cache_attestation", cache_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runtime = root / f"{phase}_runtime_validated.json"
payload = {
    "run_id": module.RUN_ID,
    "state": f"{phase.upper()}_VALIDATION_ATTESTED",
    "runtime_validation_sha256": module.sha256_file(runtime),
    "cache_validated_sha256": module.sha256_file(root / "cache_validated.json"),
    "prospective_gate_predictions_sha256": (
        root / "provenance" / "prospective_gate_predictions_sha256.txt"
    ).read_text(encoding="utf-8").strip(),
    "baseline_manifest_sha256": baseline_sha,
    "dataset_sha256": (root / "provenance" / "tworoom_dataset_sha256.txt").read_text(
        encoding="utf-8"
    ).strip(),
    "local_analyzer_review_sha256": analyzer_sha,
    "smoke_validation_attestation_sha256": (
        "" if phase == "smoke" else module.sha256_file(root / "smoke_validation_attestation.json")
    ),
}
module.atomic_write_json(root / f"{phase}_validation_attestation.json", payload)
print(f"{phase}_validation_attested")
PY
  )
}


validate_phase() {
  local phase=$1
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  require_validated_cache_integrity
  require_phase_done "$phase"
  [[ ! -e "$ROOT/${phase}_runtime_validated.json" \
      && ! -e "$ROOT/${phase}_validation_attestation.json" ]] \
    || die "$phase runtime validation was already published"
  if [[ "$phase" == full ]]; then
    verify_runtime_attestation smoke >/dev/null
  fi
  runtime_validate "$phase"
  create_runtime_attestation "$phase"
  verify_runtime_attestation "$phase"
}


pid_process_state() {
  local pid=$1
  local expected_boot=$2
  local expected_ticks=$3
  local expected_a=$4
  local expected_b=$5
  local expected_c=${6:-}
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    echo exited
    return 0
  fi
  if [[ "$expected_boot" != "$(boot_id)" || "$expected_ticks" == unavailable \
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
  local phase=$1
  local pidfile="$ROOT/${phase}_pids.txt"
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
  [[ -d "$ROOT" ]] || { echo "run: not setup"; return 0; }
  local cache_state=not_started cache_process=not_started
  [[ -f "$STATE/cache.running.json" ]] && cache_state=running
  [[ -f "$STATE/cache.done.json" ]] && cache_state=done
  [[ -f "$STATE/cache.failed.json" ]] && cache_state=failed
  if [[ -f "$ROOT/cache_pid.txt" ]]; then
    local cache_pid cache_boot cache_ticks
    read -r cache_pid cache_boot cache_ticks < "$ROOT/cache_pid.txt"
    cache_process=$(pid_process_state "$cache_pid" "$cache_boot" "$cache_ticks" run-cache run-cache)
  fi
  echo "cache state=$cache_state process=$cache_process validated=$([[ -f "$ROOT/cache_validated.json" ]] && echo yes || echo no)"
  [[ -f "$STATE/cache.launch.failed.json" ]] && echo "cache launch_failed=yes"
  status_phase smoke
  status_phase full
  echo "smoke_attested=$([[ -f "$ROOT/smoke_validation_attestation.json" ]] && echo yes || echo no)"
  echo "full_attested=$([[ -f "$ROOT/full_validation_attestation.json" ]] && echo yes || echo no)"
  echo "sealed=$([[ -f "$ROOT/READY_TO_PULL.json" ]] && echo yes || echo no)"
}


assert_no_live_workers() {
  local pid boot ticks cell gpu state
  if [[ -f "$ROOT/cache_pid.txt" ]]; then
    read -r pid boot ticks < "$ROOT/cache_pid.txt"
    state=$(pid_process_state "$pid" "$boot" "$ticks" run-cache run-cache)
    [[ "$state" != running ]] || die "cache worker/controller is still live: pid=$pid"
  fi
  local phase
  for phase in smoke full; do
    [[ -f "$ROOT/${phase}_pids.txt" ]] || continue
    while read -r cell gpu pid boot ticks; do
      [[ -n "${cell:-}" ]] || continue
      state=$(pid_process_state "$pid" "$boot" "$ticks" run-cell "$phase" "$cell")
      [[ "$state" != running ]] \
        || die "$phase worker/controller is still live: cell=$cell pid=$pid"
    done < "$ROOT/${phase}_pids.txt"
  done
  local proc pid command_line
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [[ "$pid" != "$$" && -r "$proc/cmdline" ]] || continue
    command_line=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    if [[ "$command_line" == *"$LAUNCHER"* \
          && ( "$command_line" == *"run-cache"* || "$command_line" == *"run-cell"* ) ]]; then
      die "unledgered launcher worker/controller is still live: pid=$pid"
    fi
    if [[ "$command_line" == *"$CACHE"* && "$command_line" == *"$ROOT"* \
          && ( "$command_line" == *"fit-cache"* || "$command_line" == *"run-cached"* ) ]]; then
      die "unledgered cache/planning Python worker is still live: pid=$pid"
    fi
  done
}


require_external_seal_fds() {
  local fd target resolved root_resolved samefile
  root_resolved=$(readlink -f "$ROOT") || die "cannot canonicalize run root"
  for fd in 1 2; do
    target=$(readlink "/proc/$$/fd/$fd") || die "cannot resolve seal fd $fd"
    resolved=$target
    if [[ "$target" == /* ]]; then
      resolved=$(readlink -f "/proc/$$/fd/$fd") || die "cannot canonicalize seal fd $fd"
    fi
    case "$resolved" in
      "$root_resolved"|"$root_resolved"/*) \
        die "seal fd $fd resolves inside run root: $resolved" ;;
    esac
    samefile=$(find "$ROOT" -type f -samefile "/proc/$$/fd/$fd" -print -quit 2>/dev/null || true)
    [[ -z "$samefile" ]] \
      || die "seal fd $fd aliases a file inside the run root: $samefile"
  done
}


seal_file_list() {
  find . -type f \
    ! -path './logs/seal_full_control.log' \
    ! -path './artifact_sha256.txt' \
    ! -path './READY_TO_PULL.json' \
    ! -name 'artifact_sha256.txt.tmp.*' \
    -print0 | sort -z | tr '\0' '\n'
}


seal_full() {
  source ~/.jepawm_env
  require_setup
  require_dataset_identity
  require_validated_cache_integrity
  verify_runtime_attestation smoke >/dev/null
  verify_runtime_attestation full >/dev/null
  require_phase_done full
  assert_no_live_workers
  require_external_seal_fds
  [[ ! -e "$ROOT/artifact_sha256.txt" && ! -e "$ROOT/READY_TO_PULL.json" ]] \
    || die "run root is already or partially sealed"

  local stage="$BASE/.${RUN_ID}.seal.$$"
  [[ ! -e "$stage" ]] || die "external seal staging collision: $stage"
  mkdir -m 700 "$stage"
  [[ "$(stat -c %d "$ROOT")" == "$(stat -c %d "$stage")" ]] \
    || die "seal staging is not on the run-root filesystem"
  local before="$stage/file_set.before.txt"
  local after="$stage/file_set.after.txt"
  local manifest="$stage/artifact_sha256.txt"
  local check_log="$stage/sha256sum_check.log"
  (
    cd "$ROOT"
    seal_file_list > "$before"
    while IFS= read -r file; do
      [[ -n "$file" ]] || continue
      sha256sum -- "$file"
    done < "$before" > "$manifest"
    sha256sum -c "$manifest" > "$check_log"
    seal_file_list > "$after"
  )
  cmp -s "$before" "$after" || die "run-root file set changed while constructing seal"
  sed -E 's/^[0-9a-f]{64}  //' "$manifest" > "$stage/manifest_file_set.txt"
  cmp -s "$before" "$stage/manifest_file_set.txt" \
    || die "seal manifest paths differ from the exact file set"
  publish_file "$manifest" "$ROOT/artifact_sha256.txt"
  (
    cd "$ROOT"
    sha256sum -c artifact_sha256.txt > "$check_log"
    seal_file_list > "$after"
  )
  cmp -s "$before" "$after" || die "run-root file set changed before READY publication"
  sed -E 's/^[0-9a-f]{64}  //' "$ROOT/artifact_sha256.txt" > "$stage/published_file_set.txt"
  cmp -s "$before" "$stage/published_file_set.txt" \
    || die "published seal manifest does not cover the exact intended file set"
  atomic_text "$ROOT/READY_TO_PULL.json" \
    "{\"run_id\":\"$RUN_ID\",\"state\":\"FULL_SEALED\",\"at\":\"$(iso_now)\",\"artifact_manifest_sha256\":\"$(sha_of "$ROOT/artifact_sha256.txt")\",\"full_validation_attestation_sha256\":\"$(sha_of "$ROOT/full_validation_attestation.json")\",\"jepa_commit\":\"$(<"$PROV/jepawms_git_commit.txt")\"}"
  case "$(readlink -f "$stage")" in
    "$BASE"/."$RUN_ID".seal.*) ;;
    *) die "refusing to clean unexpected seal staging path: $stage" ;;
  esac
  rm -f -- "$stage"/*
  rmdir -- "$stage"
  echo "full_sealed manifest=$ROOT/artifact_sha256.txt"
}


usage() {
  cat <<'EOF'
usage: stock_tworoom_probe_20260713.sh COMMAND

Public commands:
  setup
  start-cache | validate-cache
  start-smoke | validate-smoke
  start-full | validate-full
  seal-full | status

Private worker commands:
  run-cache
  run-cell {smoke|full} CELL
EOF
}


case "${1:-}" in
  setup) setup_run ;;
  start-cache) run_locked start_cache ;;
  validate-cache) run_locked validate_cache ;;
  start-smoke) run_locked start_batch smoke ;;
  validate-smoke) run_locked validate_phase smoke ;;
  start-full) run_locked start_batch full ;;
  validate-full) run_locked validate_phase full ;;
  seal-full) run_locked seal_full ;;
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
