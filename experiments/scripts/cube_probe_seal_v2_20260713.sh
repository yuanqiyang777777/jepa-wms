#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

BASE=/home/ps/Code/yqy/DINO-WM/stablewm_home
CODE=/home/ps/Code/yqy/DINO-WM
RUN_ID=adadim_cube_probe_closure_20260713
ROOT="$BASE/$RUN_ID"
JEPA_WT="$CODE/wt_cube_probe_closure_jepawms_20260713"
STATE="$ROOT/state"
PROV="$ROOT/provenance"
INCIDENT_DIR="$PROV/seal_incident_20260713"
ORIGINAL_MANIFEST="$ROOT/artifact_sha256.txt"
ORIGINAL_READY="$ROOT/READY_TO_PULL.json"
CONTROL_LOG="$ROOT/logs/seal_full_control.log"
V2_MANIFEST="$ROOT/artifact_sha256_v2.txt"
V2_READY="$ROOT/READY_TO_PULL_v2.json"
STAGE_PREFIX="$BASE/.${RUN_ID}.seal_v2_stage.$$"
STAGE_SNAPSHOT="${STAGE_PREFIX}.script"
STAGE_SNAPSHOT_SHA="${STAGE_PREFIX}.script.sha256"
STAGE_INCIDENT="${STAGE_PREFIX}.incident.json"
STAGE_MANIFEST="${STAGE_PREFIX}.manifest"
STAGE_READY="${STAGE_PREFIX}.ready"

EXPECTED_SCIENTIFIC_COMMIT=3bf0aa31741e6ae4bf05274c1def878dd65c30ae
EXPECTED_BRANCH=exp/20260713-cube-probe-closure
EXPECTED_ORIGINAL_MANIFEST_SHA256=c1e749b15677716b706858de50263bfdffcd3df3d0ea2a57754a84ca03ac6199
EXPECTED_ORIGINAL_READY_SHA256=649ace1d8bd7e26026621a0af81cf80b1de1b505728dad23a66103549f9e3bb7
EXPECTED_MANIFEST_LOG_SHA256=daab0882edf77c1db5883299645d2877ecfad6212fcb20755fddf5e5276b3b46
EXPECTED_FINAL_LOG_SHA256=25b591c035a8fd179c1246305d66d113fa398e26e5b237b8a508748e2f2576a2
EXPECTED_MANIFEST_ENTRIES=288
MISMATCH_PATH=logs/seal_full_control.log


die() {
  echo "ERROR: $*" >&2
  exit 1
}


sha_of() {
  sha256sum "$1" | awk '{print $1}'
}


iso_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}


require_external_output_sinks() {
  local fd target samefile
  for fd in 1 2; do
    [[ -f "/proc/$$/fd/$fd" ]] || exit 97
    target=$(readlink -f "/proc/$$/fd/$fd" 2>/dev/null) || exit 97
    [[ -n "$target" && "$target" != "$ROOT" && "$target" != "$ROOT/"* ]] || exit 97
    samefile=$(find "$ROOT" -type f -samefile "/proc/$$/fd/$fd" -print -quit 2>/dev/null) \
      || exit 97
    [[ -z "$samefile" ]] || exit 97
  done
}


cleanup_external_stages() {
  rm -f -- \
    "$STAGE_SNAPSHOT" \
    "$STAGE_SNAPSHOT_SHA" \
    "$STAGE_INCIDENT" \
    "$STAGE_MANIFEST" \
    "$STAGE_READY" \
    2>/dev/null || true
}


publish_file() {
  local tmp=$1
  local path=$2
  [[ -f "$tmp" ]] || die "missing staged file: $tmp"
  [[ ! -e "$path" ]] || die "refusing to overwrite immutable file: $path"
  if ! ln "$tmp" "$path"; then
    rm -f "$tmp"
    die "atomic no-replace publish failed: $path"
  fi
  rm -f "$tmp" 2>/dev/null || true
}


proc_start_ticks() {
  local pid=$1
  awk '{print $22}' "/proc/$pid/stat"
}


recorded_process_is_live() {
  local pid=$1
  local expected_boot=$2
  local expected_ticks=$3
  [[ -r "/proc/$pid/stat" ]] || return 1
  [[ "$(< /proc/sys/kernel/random/boot_id)" == "$expected_boot" ]] || return 1
  [[ "$(proc_start_ticks "$pid")" == "$expected_ticks" ]]
}


require_no_recorded_workers() {
  local pid boot ticks cell gpu
  if [[ -f "$ROOT/cache_pid.txt" ]]; then
    read -r pid boot ticks < "$ROOT/cache_pid.txt"
    ! recorded_process_is_live "$pid" "$boot" "$ticks" \
      || die "recorded cache worker is still live: pid=$pid"
  fi
  local pidfile
  for pidfile in "$ROOT/smoke_pids.txt" "$ROOT/full_pids.txt"; do
    [[ -f "$pidfile" ]] || die "missing pid manifest: $pidfile"
    while read -r cell gpu pid boot ticks; do
      [[ -n "${cell:-}" ]] || continue
      ! recorded_process_is_live "$pid" "$boot" "$ticks" \
        || die "recorded worker is still live: cell=$cell pid=$pid"
    done < "$pidfile"
  done
  if ps -eo args= | grep -E '[c]ube_probe_closure_20260713\.sh (start-cache|validate-cache|start-smoke|validate-smoke|start-full|seal-full|run-cache|run-cell)' >/dev/null; then
    die "an original launcher/controller process is still present"
  fi
}


require_stable_control_log() {
  local before_stat before_sha after_stat after_sha
  before_stat=$(stat -c '%y|%s' "$CONTROL_LOG")
  before_sha=$(sha_of "$CONTROL_LOG")
  sleep 2
  after_stat=$(stat -c '%y|%s' "$CONTROL_LOG")
  after_sha=$(sha_of "$CONTROL_LOG")
  [[ "$before_stat" == "$after_stat" && "$before_sha" == "$after_sha" ]] \
    || die "seal control log is still changing"
  [[ "$after_sha" == "$EXPECTED_FINAL_LOG_SHA256" ]] \
    || die "unexpected final seal control-log hash: $after_sha"
}


require_original_seal_evidence() {
  [[ "$(sha_of "$ORIGINAL_MANIFEST")" == "$EXPECTED_ORIGINAL_MANIFEST_SHA256" ]] \
    || die "original manifest hash drift"
  [[ "$(sha_of "$ORIGINAL_READY")" == "$EXPECTED_ORIGINAL_READY_SHA256" ]] \
    || die "original READY hash drift"
  [[ "$(wc -l < "$ORIGINAL_MANIFEST")" -eq "$EXPECTED_MANIFEST_ENTRIES" ]] \
    || die "unexpected original manifest entry count"
  grep -Fq "\"run_id\":\"$RUN_ID\"" "$ORIGINAL_READY" \
    || die "original READY run id mismatch"
  grep -Fq '"state":"FULL_SEALED"' "$ORIGINAL_READY" \
    || die "original READY state mismatch"
  grep -Fq "\"artifact_manifest_sha256\":\"$EXPECTED_ORIGINAL_MANIFEST_SHA256\"" "$ORIGINAL_READY" \
    || die "original READY does not bind the original manifest"
  grep -Fq "\"jepa_commit\":\"$EXPECTED_SCIENTIFIC_COMMIT\"" "$ORIGINAL_READY" \
    || die "original READY scientific commit mismatch"
  [[ "$(< "$PROV/jepawms_git_commit.txt")" == "$EXPECTED_SCIENTIFIC_COMMIT" ]] \
    || die "frozen scientific commit mismatch"
  local recorded_log_sha
  recorded_log_sha=$(awk -v p="./$MISMATCH_PATH" '$2 == p {print $1}' "$ORIGINAL_MANIFEST")
  [[ "$recorded_log_sha" == "$EXPECTED_MANIFEST_LOG_SHA256" ]] \
    || die "unexpected control-log hash recorded in original manifest"
  local head_without_last_sha
  head_without_last_sha=$(head -n -1 "$CONTROL_LOG" | sha256sum | awk '{print $1}')
  [[ "$head_without_last_sha" == "$EXPECTED_MANIFEST_LOG_SHA256" ]] \
    || die "head-minus-last equivalence does not reproduce the original manifest"
  [[ "$(tail -n 1 "$CONTROL_LOG")" == "full_sealed manifest=$ORIGINAL_MANIFEST" ]] \
    || die "unexpected final line in seal control log"
}


require_original_manifest_shape() {
  local entries=0 passes=0 mismatches=0 expected path actual
  while read -r expected path; do
    [[ -n "${expected:-}" && -n "${path:-}" ]] || die "malformed original manifest line"
    path=${path#./}
    [[ -f "$ROOT/$path" ]] || die "missing original manifest artifact: $path"
    actual=$(sha_of "$ROOT/$path")
    entries=$((entries + 1))
    if [[ "$actual" == "$expected" ]]; then
      passes=$((passes + 1))
    else
      mismatches=$((mismatches + 1))
      [[ "$path" == "$MISMATCH_PATH" ]] \
        || die "unexpected original manifest mismatch: $path"
    fi
  done < "$ORIGINAL_MANIFEST"
  [[ "$entries" -eq 288 && "$passes" -eq 287 && "$mismatches" -eq 1 ]] \
    || die "original manifest mismatch profile drift: entries=$entries passes=$passes mismatches=$mismatches"

  local manifest_set current_set
  manifest_set=$(mktemp)
  current_set=$(mktemp)
  awk '{sub(/^\.\//, "", $2); print $2}' "$ORIGINAL_MANIFEST" | sort > "$manifest_set"
  (
    cd "$ROOT"
    find . -type f \
      ! -path './artifact_sha256.txt' \
      ! -path './READY_TO_PULL.json' \
      -printf '%P\n' | sort
  ) > "$current_set"
  if ! cmp -s "$manifest_set" "$current_set"; then
    diff -u "$manifest_set" "$current_set" >&2 || true
    rm -f "$manifest_set" "$current_set"
    die "original manifest file set does not match the pre-incident run root"
  fi
  rm -f "$manifest_set" "$current_set"
}


require_clean_repair_source() {
  [[ "$(git -C "$JEPA_WT" branch --show-current)" == "$EXPECTED_BRANCH" ]] \
    || die "unexpected repair branch"
  [[ -z "$(git -C "$JEPA_WT" status --porcelain)" ]] \
    || die "repair worktree is dirty"
  local expected_script actual_script
  expected_script="$JEPA_WT/experiments/scripts/cube_probe_seal_v2_20260713.sh"
  actual_script=$(readlink -f "$0")
  [[ "$actual_script" == "$expected_script" ]] \
    || die "repair script must run from the reviewed worktree path"
  git -C "$JEPA_WT" merge-base --is-ancestor "$EXPECTED_SCIENTIFIC_COMMIT" HEAD \
    || die "repair commit does not descend from the frozen scientific commit"
  [[ "$(git -C "$JEPA_WT" rev-list --count "$EXPECTED_SCIENTIFIC_COMMIT"..HEAD)" -eq 1 ]] \
    || die "repair branch must contain exactly one post-science commit"
  local changed_paths
  changed_paths=$(git -C "$JEPA_WT" diff --name-only "$EXPECTED_SCIENTIFIC_COMMIT"..HEAD)
  [[ "$changed_paths" == "experiments/scripts/cube_probe_seal_v2_20260713.sh" ]] \
    || die "repair commit must add only the reviewed v2 seal script"
}


write_incident_evidence() {
  local script_path=$1
  local repair_commit=$2
  local detected_at=$3
  local log_mtime=$4
  local snapshot="$INCIDENT_DIR/cube_probe_seal_v2_20260713.sh"
  local snapshot_sha_file="$INCIDENT_DIR/cube_probe_seal_v2_20260713.sh.sha256"
  local incident_json="$INCIDENT_DIR/incident.json"

  mkdir "$INCIDENT_DIR"

  cp "$script_path" "$STAGE_SNAPSHOT"
  chmod 600 "$STAGE_SNAPSHOT"
  publish_file "$STAGE_SNAPSHOT" "$snapshot"
  local snapshot_sha
  snapshot_sha=$(sha_of "$snapshot")
  printf '%s  %s\n' "$snapshot_sha" cube_probe_seal_v2_20260713.sh > "$STAGE_SNAPSHOT_SHA"
  publish_file "$STAGE_SNAPSHOT_SHA" "$snapshot_sha_file"

  cat > "$STAGE_INCIDENT" <<EOF
{
  "schema_version": "cube_probe_seal_incident_v1",
  "run_id": "$RUN_ID",
  "classification": "provenance_control_log_sealing_defect",
  "scientific_run_valid": true,
  "requires_scientific_retry": false,
  "detected_at": "$detected_at",
  "repair_commit": "$repair_commit",
  "scientific_commit": "$EXPECTED_SCIENTIFIC_COMMIT",
  "original_manifest": {
    "path": "artifact_sha256.txt",
    "sha256": "$EXPECTED_ORIGINAL_MANIFEST_SHA256",
    "entries": 288,
    "passes_after_controller_exit": 287,
    "mismatches_after_controller_exit": 1
  },
  "original_ready": {
    "path": "READY_TO_PULL.json",
    "sha256": "$EXPECTED_ORIGINAL_READY_SHA256"
  },
  "mismatch": {
    "path": "$MISMATCH_PATH",
    "manifest_recorded_sha256": "$EXPECTED_MANIFEST_LOG_SHA256",
    "final_sha256": "$EXPECTED_FINAL_LOG_SHA256",
    "head_minus_last_sha256": "$EXPECTED_MANIFEST_LOG_SHA256",
    "final_mtime": "$log_mtime",
    "cause": "seal stdout was redirected inside the run root; the launcher appended full_sealed after artifact_sha256.txt had hashed the first two control-log lines"
  },
  "repair": {
    "strategy": "post_controller_atomic_v2_reseal",
    "original_files_preserved": true,
    "v2_manifest": "artifact_sha256_v2.txt",
    "v2_ready": "READY_TO_PULL_v2.json",
    "script_snapshot": "provenance/seal_incident_20260713/cube_probe_seal_v2_20260713.sh",
    "script_snapshot_sha256": "$snapshot_sha"
  },
  "known_nonblocking_issue": "status_phase initializes phase and pidfile in one local statement under set -u; status-only display is affected, but no setup, launch, worker, validate, or seal gate calls it"
}
EOF
  publish_file "$STAGE_INCIDENT" "$incident_json"
}


create_v2_manifest() {
  [[ ! -e "$V2_MANIFEST" && ! -e "$V2_READY" ]] \
    || die "v2 seal already exists"

  (
    cd "$ROOT"
    find . -type f \
      ! -path './artifact_sha256_v2.txt' \
      ! -path './READY_TO_PULL_v2.json' \
      -print0 | sort -z | xargs -0 sha256sum
  ) > "$STAGE_MANIFEST"
  publish_file "$STAGE_MANIFEST" "$V2_MANIFEST"
}


verify_v2_manifest() {
  local manifest_sha manifest_set current_set
  [[ ! -e "$V2_READY" ]] || die "v2 READY appeared before manifest verification"
  (
    cd "$ROOT"
    sha256sum -c --quiet artifact_sha256_v2.txt
  ) || die "v2 artifact manifest verification failed"

  manifest_sha=$(sha_of "$V2_MANIFEST")
  manifest_set=$(mktemp)
  current_set=$(mktemp)
  awk '{sub(/^\.\//, "", $2); print $2}' "$V2_MANIFEST" | sort > "$manifest_set"
  (
    cd "$ROOT"
    find . -type f \
      ! -path './artifact_sha256_v2.txt' \
      ! -path './READY_TO_PULL_v2.json' \
      -printf '%P\n' | sort
  ) > "$current_set"
  if ! cmp -s "$manifest_set" "$current_set"; then
    diff -u "$manifest_set" "$current_set" >&2 || true
    rm -f "$manifest_set" "$current_set"
    die "v2 manifest file set mismatch"
  fi
  local entries
  entries=$(wc -l < "$V2_MANIFEST")
  [[ "$entries" -eq 293 ]] || die "unexpected v2 manifest entry count: $entries"
  rm -f "$manifest_set" "$current_set"
  VERIFIED_V2_MANIFEST_SHA=$manifest_sha
  VERIFIED_V2_ENTRIES=$entries
}


publish_v2_ready() {
  local manifest_sha=$1
  local entries=$2
  local created_at expected_ready ready_sha
  [[ ! -e "$V2_READY" && ! -e "$STAGE_READY" ]] || die "v2 READY already exists"
  [[ "$manifest_sha" =~ ^[0-9a-f]{64}$ && "$entries" -eq 293 ]] \
    || die "invalid verified manifest binding for staged v2 READY"
  [[ "$(sha_of "$V2_MANIFEST")" == "$manifest_sha" ]] \
    || die "v2 manifest changed before READY staging"
  created_at=$(iso_now)
  expected_ready="{\"run_id\":\"$RUN_ID\",\"state\":\"FULL_SEALED_V2\",\"at\":\"$created_at\",\"artifact_manifest\":\"artifact_sha256_v2.txt\",\"artifact_manifest_sha256\":\"$manifest_sha\",\"jepa_commit\":\"$EXPECTED_SCIENTIFIC_COMMIT\",\"incident\":\"provenance/seal_incident_20260713/incident.json\",\"supersedes_ready\":\"READY_TO_PULL.json\"}"
  printf '%s\n' "$expected_ready" > "$STAGE_READY"
  [[ "$(< "$STAGE_READY")" == "$expected_ready" ]] || die "staged v2 READY content mismatch"
  [[ "$(wc -l < "$STAGE_READY")" -eq 1 ]] || die "staged v2 READY must be one line"
  ready_sha=$(sha_of "$STAGE_READY")
  publish_file "$STAGE_READY" "$V2_READY"
  echo "v2_manifest_ok entries=$entries sha256=$manifest_sha"
  echo "v2_ready_ok sha256=$ready_sha"
}


main() {
  require_external_output_sinks
  [[ $# -eq 0 ]] || die "this one-shot incident repair accepts no arguments"
  source ~/.jepawm_env
  cd "$JEPA_WT"
  [[ -d "$ROOT" && -d "$STATE" && -d "$PROV" ]] || die "frozen run root is incomplete"
  [[ ! -e "$INCIDENT_DIR" && ! -e "$V2_MANIFEST" && ! -e "$V2_READY" ]] \
    || die "incident repair artifacts already exist"
  [[ ! -e "$STAGE_SNAPSHOT" && ! -e "$STAGE_SNAPSHOT_SHA" && ! -e "$STAGE_INCIDENT" \
    && ! -e "$STAGE_MANIFEST" && ! -e "$STAGE_READY" ]] \
    || die "external staging-file collision"
  trap cleanup_external_stages EXIT

  [[ -f "$STATE/.mutation.lock" && ! -L "$STATE/.mutation.lock" ]] \
    || die "missing immutable regular mutation lock file"
  exec 9<"$STATE/.mutation.lock"
  flock -n 9 || die "another run-root mutation is active"

  require_clean_repair_source
  require_no_recorded_workers
  require_stable_control_log
  require_original_seal_evidence
  require_original_manifest_shape

  local script_path repair_commit detected_at log_mtime
  script_path=$(readlink -f "$0")
  repair_commit=$(git -C "$JEPA_WT" rev-parse HEAD)
  detected_at=$(iso_now)
  log_mtime=$(stat -c '%y' "$CONTROL_LOG")
  write_incident_evidence "$script_path" "$repair_commit" "$detected_at" "$log_mtime"
  create_v2_manifest
  VERIFIED_V2_MANIFEST_SHA=
  VERIFIED_V2_ENTRIES=
  verify_v2_manifest
  publish_v2_ready "$VERIFIED_V2_MANIFEST_SHA" "$VERIFIED_V2_ENTRIES"
}


main "$@"
