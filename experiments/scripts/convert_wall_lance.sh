#!/bin/bash
# Convert the raw Wall dataset into the Phase-1 Lance layout.
#
# Wall observations are float32 tensors, so the default codec is raw_float32 for
# bit-exact parity with the raw loader. Do not use this script as a raw fallback:
# Stage-2b requires the Lance store to exist before training starts.
#
# Optional:
#   JEPAWM_DSET_LANCE=$JEPAWM_DSET/_lance_20260528
#   INPUT_ROOT=$JEPAWM_DSET/wall
#   OUTPUT_URI=$JEPAWM_DSET_LANCE/Wall.lance
#   CODEC=raw_float32
#   MODE=error
#   JPEG_QUALITY=95
#   LIMIT=                 # set to an integer for smoke conversion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/_common.sh"
cd "$REPO_ROOT"

export JEPAWM_DSET_LANCE="${JEPAWM_DSET_LANCE:-$JEPAWM_DSET/_lance_20260528}"
INPUT_ROOT="${INPUT_ROOT:-$JEPAWM_DSET/wall}"
OUTPUT_URI="${OUTPUT_URI:-$JEPAWM_DSET_LANCE/Wall.lance}"
CODEC="${CODEC:-raw_float32}"
MODE="${MODE:-error}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
LIMIT="${LIMIT:-}"

if [ ! -d "$INPUT_ROOT" ]; then
  echo "ERROR: Wall input root not found: $INPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$JEPAWM_DSET_LANCE"

{
  echo "INPUT_ROOT=$INPUT_ROOT"
  echo "OUTPUT_URI=$OUTPUT_URI"
  echo "CODEC=$CODEC"
  echo "MODE=$MODE"
  echo "JPEG_QUALITY=$JPEG_QUALITY"
  echo "LIMIT=${LIMIT:-unset}"
} >&2

python - "$INPUT_ROOT" "$OUTPUT_URI" "$CODEC" "$MODE" "$JPEG_QUALITY" "$LIMIT" <<'PY'
import sys

from app.plan_common.datasets.stores.writer_wall import convert_wall_to_lance

input_root, output_uri, codec, mode, jpeg_quality, limit = sys.argv[1:7]
limit_value = None if limit == "" else int(limit)

meta = convert_wall_to_lance(
    input_root,
    output_uri,
    codec=codec,
    mode=mode,
    jpeg_quality=int(jpeg_quality),
    limit=limit_value,
)
print(
    "converted Wall episodes={episodes} codec={codec} output={output}".format(
        episodes=meta["num_episodes"],
        codec=meta["image_codec"],
        output=output_uri,
    )
)
PY
