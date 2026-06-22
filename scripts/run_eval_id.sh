#!/usr/bin/env bash
#
# run_eval_id.sh
#
# This script runs the standalone body ReID and computes cluster metrics.
#

set -euo pipefail

PROJECT_ROOT="$(pwd)"
VIDEO_DIR="${PROJECT_ROOT}/videos"

if [[ "$@" = "" ]]; then
    echo
    echo "Usage: ${0} --tracker <name> [identity params]"
    python run_identity.py
    exit 1
fi

if [[ "$1" != "--tracker" ]]; then
    echo "Missing parameter: --tracker <name> ; using 'temporary'" >&2
    TRACKER_NAME="temporary"
else
    TRACKER_NAME="$2"
    shift 2
fi

TRACKERS_DIR="${PROJECT_ROOT}/eval/metrics/identity"
OUTPUT_DIR="${TRACKERS_DIR}/${TRACKER_NAME}/"

for VIDEO in KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte; do
    echo "========== Processing "${VIDEO}" =========="
    echo "sequence_dir:" "${VIDEO_DIR}/${VIDEO}"
    python run_identity_metrics.py \
        --sequence_dir="${VIDEO_DIR}/${VIDEO}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"
done
echo "-------------------------------------------"
