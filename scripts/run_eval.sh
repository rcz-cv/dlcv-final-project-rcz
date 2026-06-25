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

        # Apply patches
        #
        # The following patch to TrackEval fixes old numpy conventions
        # that that are no longer supported in the current library.
        # I can't BELIEVE that this is necessary. What a PITA.
        #
        python - <<'PY'
from pathlib import Path

old1 = "np.float"
new1 = "float"
old2 = "np.int"
new2 = "int"

p = Path("trackeval/datasets/mot_challenge_2d_box.py")
text = p.read_text()
if old1 in text or old2 in text:
    text = text.replace(old1, new1)
    text = text.replace(old2, new2)
    p.write_text(text)
    print("Patched TrackEval:mot_challenge_2d_box.py for numpy")

p = Path("trackeval/metrics/hota.py")
text = p.read_text()
if old1 in text or old2 in text:
    text = text.replace(old1, new1)
    text = text.replace(old2, new2)
    p.write_text(text)
    print("Patched TrackEval:hota.py for numpy")
PY
    )
fi

if [[ ! -x "${TRACKEVAL_PY}" && -z "${COLAB_RELEASE_TAG:-}" ]]; then
    echo "Creating TrackEval virtual environment..."
    (
        python3 -m venv "${TRACKEVAL_VENV}"
        source .venv/bin/activate

        "${TRACKEVAL_PY}" -m pip install --upgrade pip
        "${TRACKEVAL_PY}" -m pip install numpy scipy pycocotools matplotlib opencv_python scikit_image pytest Pillow tqdm tabulate
    )
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
    cd external/TrackEval
    source .venv/bin/activate

    MPLBACKEND=Agg .venv/bin/python scripts/run_mot_challenge.py \
        --GT_FOLDER "${PROJECT_ROOT}/eval/gt/DLCV" \
        --TRACKERS_FOLDER "${TRACKERS_DIR}" \
        --BENCHMARK DLCV \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL ${TRACKER_NAME} \
        --DO_PREPROC False \
        --METRICS HOTA
) 2>&1 | tee "${PROJECT_ROOT}/eval/logs/trackeval_${TRACKER_NAME}_hota.log"

python scripts/update_hota.py  --metadata_dir "${OUTPUT_DIR}"
