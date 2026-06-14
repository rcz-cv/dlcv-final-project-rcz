#!/usr/bin/env bash
#
# track_eval.sh
#
# This script evaluates the tracker with the specified configuration.
#
# Usage: bash ./videos/setup_videos.sh

set -euo pipefail

PROJECT_ROOT="$(pwd)"

#TRACKER_NAME="deep_sort_baseline"
TRACKER_NAME="yolov5m"

(
    cd ../TrackEval

    .venv/bin/python scripts/run_mot_challenge.py \
        --GT_FOLDER "${PROJECT_ROOT}/eval/gt/DLCV" \
        --TRACKERS_FOLDER "${PROJECT_ROOT}/eval/trackers/DLCV" \
        --BENCHMARK DLCV \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL ${TRACKER_NAME} \
        --DO_PREPROC False \
        --METRICS HOTA
) 2>&1 | tee eval/logs/trackeval_${TRACKER_NAME}_hota.log
