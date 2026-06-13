#!/usr/bin/env bash
#
# setup_eval.sh
#
# This script populates the eval folder with symlinks to the video datasets,
# allowing them to be be downloaded (and git-ignored) here in videos.
#
# Usage: bash ./videos/setup_eval.sh
#

set -euo pipefail

CHALLENGE="DLCV"
SPLIT="${CHALLENGE}-train"

VIDEOS=(
    "KITTI-17"
    "MOT16-09"
    "MOT16-11"
    "PETS09-S2L1"
    "TUD-Campus"
    "TUD-Stadtmitte"
)

TRACKERS=(
    "deep_sort_baseline"
)

echo "Populating eval directory structure..."

GT_ROOT="eval/gt/DLCV"
TRACKER_ROOT="eval/trackers/DLCV/${SPLIT}"

mkdir -p "${GT_ROOT}/${SPLIT}"
mkdir -p "${GT_ROOT}/seqmaps"
mkdir -p "${TRACKER_ROOT}"

#
# Create seqmap
#
SEQMAP="${GT_ROOT}/seqmaps/${SPLIT}.txt"

echo "name" > "${SEQMAP}"

for VIDEO in "${VIDEOS[@]}"; do
    echo "Setting up ${VIDEO}..."

    VIDEO_DIR="videos/${VIDEO}"

    if [[ ! -d "${VIDEO_DIR}" ]]; then
        echo "ERROR: ${VIDEO_DIR} not found"
        exit 1
    fi

    if [[ ! -f "${VIDEO_DIR}/seqinfo.ini" ]]; then
        echo "ERROR: ${VIDEO_DIR}/seqinfo.ini not found"
        exit 1
    fi

    if [[ ! -f "${VIDEO_DIR}/gt/gt.txt" ]]; then
        echo "ERROR: ${VIDEO_DIR}/gt/gt.txt not found"
        exit 1
    fi

    EVAL_VIDEO_DIR="${GT_ROOT}/${SPLIT}/${VIDEO}"

    mkdir -p "${EVAL_VIDEO_DIR}"

    #
    # Create symlink to gt folder
    #
    ln -snf \
        "$(realpath "${VIDEO_DIR}/gt")" \
        "${EVAL_VIDEO_DIR}/gt"

    #
    # Create symlink to seqinfo.ini
    #
    ln -snf \
        "$(realpath "${VIDEO_DIR}/seqinfo.ini")" \
        "${EVAL_VIDEO_DIR}/seqinfo.ini"

    #
    # Add sequence to seqmap
    #
    echo "${VIDEO}" >> "${SEQMAP}"
done

#
# Create tracker folders
#
for TRACKER in "${TRACKERS[@]}"; do
    mkdir -p "${TRACKER_ROOT}/${TRACKER}/data"
done
