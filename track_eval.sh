
#!/usr/bin/env bash
#
# track_eval.sh
#

PROJECT_ROOT="$(pwd)"

(
    cd ../TrackEval

    .venv/bin/python scripts/run_mot_challenge.py \
        --GT_FOLDER "${PROJECT_ROOT}/eval/gt/DLCV" \
        --TRACKERS_FOLDER "${PROJECT_ROOT}/eval/trackers/DLCV" \
        --BENCHMARK DLCV \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL deep_sort_baseline \
        --DO_PREPROC False \
        --METRICS HOTA
) 2>&1 | tee eval/logs/trackeval_deep_sort_baseline_hota.log
