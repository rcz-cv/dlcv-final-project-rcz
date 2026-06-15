#!/usr/bin/env bash
#
# gen_mot_yolov5.sh
#
# This script creates yolov5 detections using the ultralytics/yolov5 GitHub
# repo. The repo is cloned if needed, pinned to v7.0, and then a venv created
# if needed, and then our generate_mot_detections.py script runs in that venv
# but with access to local dependencies via the process environment.
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

YOLOV5_DIR="${PROJECT_ROOT}/external/yolov5"
YOLOV5_VENV="${YOLOV5_DIR}/.venv"
YOLOV5_PY="${YOLOV5_VENV}/bin/python"

VIDEO_DIR="${PROJECT_ROOT}/videos"
OUTPUT_DIR="${PROJECT_ROOT}/detections"
MODEL="yolov5m"
YOLOV5_TAG="v7.0"

mkdir -p "${PROJECT_ROOT}/external"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${YOLOV5_DIR}" ]]; then
    echo "Cloning YOLOv5..."
    git clone https://github.com/ultralytics/yolov5.git "${YOLOV5_DIR}"
    (
        cd "${YOLOV5_DIR}"
        git checkout "${YOLOV5_TAG}"

        # Apply patches
        #
        # The following patch to Yolov5 v7.0 fixes the PyTorch 2.6+ torch.load change
        # that was implemented to prevent possible excution of arbitrary code.
        #
        python - <<'PY'
from pathlib import Path

p = Path("models/experimental.py")
text = p.read_text()
old = "torch.load(attempt_download(w), map_location='cpu')"
new = "torch.load(attempt_download(w), map_location='cpu', weights_only=False)"
if old in text:
    p.write_text(text.replace(old, new))
    print("Patched YOLOv5 for PyTorch 2.6+ torch.load(weights_only=False)")
PY
    )
fi

if [[ ! -x "${YOLOV5_PY}" ]]; then
    echo "Creating YOLOv5 virtual environment..."
    python3 -m venv "${YOLOV5_VENV}"

    "${YOLOV5_PY}" -m pip install --upgrade pip
    "${YOLOV5_PY}" -m pip install -r "${YOLOV5_DIR}/requirements.txt"
fi

echo "Generating MOT detections with ${MODEL}..."

PYTHONPATH="${PROJECT_ROOT}" \
"${YOLOV5_PY}" "${PROJECT_ROOT}/tools/generate_mot_detections.py" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --repo_path "${YOLOV5_DIR}"

echo "Done."
echo "Detections written to:"
echo "  ${OUTPUT_DIR}"
