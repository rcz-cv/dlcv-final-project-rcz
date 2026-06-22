#!/usr/bin/env bash
#
# run_eval.sh
#
# This script runs the tracker and evaluates with official MOT challenge.
#

set -euo pipefail

PROJECT_ROOT="$(pwd)"

TRACKEVAL_DIR="${PROJECT_ROOT}/external/trackeval"
TRACKEVAL_VENV="${TRACKEVAL_DIR}/.venv"
TRACKEVAL_PY="${TRACKEVAL_VENV}/bin/python"

VIDEO_DIR="${PROJECT_ROOT}/videos"

mkdir -p "${PROJECT_ROOT}/external"

if [[ ! -d "${TRACKEVAL_DIR}" ]]; then
    echo "Cloning TrackEval..."
    git clone https://github.com/JonathonLuiten/TrackEval.git "${TRACKEVAL_DIR}"
    (
        cd "${TRACKEVAL_DIR}"
        git checkout
    )
fi

if [[ ! -x "${TRACKEVAL_PY}" ]]; then
    echo "Creating TrackEval virtual environment..."
    python3 -m venv "${TRACKEVAL_VENV}"

    "${TRACKEVAL_PY}" -m pip install --upgrade pip
    "${TRACKEVAL_PY}" -m pip install numpy scipy pycocotools matplotlib opencv_python scikit_image pytest Pillow tqdm tabulate
fi

if [[ "$@" = "" ]]; then
    echo
    echo "Usage: ${0} --tracker <name> [DeepSORT params]"
    python run_tracker.py
    exit 1
fi

if [[ "$1" != "--tracker" ]]; then
    echo "Missing parameter: --tracker <name> ; using 'temporary'" >&2
    TRACKER_NAME="temporary"
else
    TRACKER_NAME="$2"
    shift 2
fi
TRACKERS_DIR="${PROJECT_ROOT}/eval/trackers/DLCV"
OUTPUT_DIR="${TRACKERS_DIR}/DLCV-train/${TRACKER_NAME}/data/"

for VIDEO in KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte; do
    echo "========== Processing "${VIDEO}" =========="
    python run_tracker.py \
        --sequence_dir=./videos/"${VIDEO}" \
        --output_dir "${OUTPUT_DIR}" \
        --no-display \
        "$@"
done
echo "-------------------------------------------"

(
    cd ../TrackEval

    .venv/bin/python scripts/run_mot_challenge.py \
        --GT_FOLDER "${PROJECT_ROOT}/eval/gt/DLCV" \
        --TRACKERS_FOLDER "${TRACKERS_DIR}" \
        --BENCHMARK DLCV \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL ${TRACKER_NAME} \
        --DO_PREPROC False \
        --METRICS HOTA
) 2>&1 | tee eval/logs/trackeval_${TRACKER_NAME}_hota.log

python update_hota.py  --metadata_dir "${OUTPUT_DIR}"
