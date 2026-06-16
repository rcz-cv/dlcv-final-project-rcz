
#!/usr/bin/env bash
#
# track_eval.sh
#

if [ -z "$1" ]; then
    echo "Usage: $0 <tracker-name>"
    exit 1
fi

PROJECT_ROOT="$(pwd)"
TRACKER_NAME="$1"

(
    cd ../TrackEval

    .venv/bin/python scripts/run_mot_challenge.py \
        --GT_FOLDER "${PROJECT_ROOT}/eval/gt/DLCV" \
        --TRACKERS_FOLDER "${PROJECT_ROOT}/eval/trackers/DLCV" \
        --BENCHMARK DLCV \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL "${TRACKER_NAME}" \
        --DO_PREPROC False \
        --METRICS HOTA
) 2>&1 | tee "eval/logs/trackeval_${TRACKER_NAME}_hota.log"
