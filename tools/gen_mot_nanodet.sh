#!/usr/bin/env bash
#
# gen_mot_nanodet.sh
#
# This script creates nanodet detections using the RangiLyu/nanodet GitHub
# repo. The repo is cloned if needed, pinned to v1.0.0, and then a venv created
# if needed, and then our generate_mot_detections.py script runs in that venv
# but with access to local dependencies via the process environment.
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NANODET_DIR="${PROJECT_ROOT}/external/nanodet"
NANODET_VENV="${NANODET_DIR}/.venv"
NANODET_PY="${NANODET_VENV}/bin/python"

VIDEO_DIR="${PROJECT_ROOT}/videos"
OUTPUT_DIR="${PROJECT_ROOT}/detections"
REPO_NAME="nanodet"
MODEL="nanodet-plus-m-416"
MODEL_PATH="${PROJECT_ROOT}/resources/networks/nanodet/nanodet-plus-m_416.pth"
MODEL_TAG="v1.0.0"

mkdir -p "${PROJECT_ROOT}/external"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${NANODET_DIR}" ]]; then
    echo "Cloning ${REPO_NAME}..."
    git clone https://github.com/RangiLyu/nanodet.git "${NANODET_DIR}"
    (
        cd "${NANODET_DIR}"
        git checkout "${MODEL_TAG}"

        # Apply patches
        #
        # The following patch to Yolov5 v7.0 fixes the PyTorch 2.6+ torch.load change
        # that was implemented to prevent possible excution of arbitrary code.
        #
        python - <<'PY'
from pathlib import Path

p = Path("nanodet/model/arch/one_stage_detector.py")
text = p.read_text()

old = "torch.cuda.synchronize()"
new = "if torch.cuda.is_available():\n                torch.cuda.synchronize()"

if old in text and new not in text:
    p.write_text(text.replace(old, new))
    print("Patched NanoDet CUDA synchronize guard")
PY

        python - <<'PY'
from pathlib import Path

p = Path("nanodet/model/arch/one_stage_detector.py")
text = p.read_text()

text = text.replace(
    'print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")',
    '# print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")',
)

text = text.replace(
    'print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")',
    '# print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")',
)

p.write_text(text)
print("Patched NanoDet inference timing prints")
PY
    )

fi

if [[ ! -x "${NANODET_PY}" ]]; then
    echo "Creating ${REPO_NAME} virtual environment..."
    python3 -m venv "${NANODET_VENV}"

    "${NANODET_PY}" -m pip install --upgrade pip
    "${NANODET_PY}" -m pip install torch torchvision opencv-python

    NANODET_REQ="${NANODET_DIR}/requirements.txt"
    NANODET_REQ_FILTERED="${NANODET_DIR}/requirements.local.txt"

    grep -vE '^(torch|torchvision|torchaudio|pytorch)([<=> ].*)?$' "${NANODET_REQ}" \
    > "${NANODET_REQ_FILTERED}"

    "${NANODET_PY}" -m pip install torch torchvision
    "${NANODET_PY}" -m pip install -r "${NANODET_REQ_FILTERED}"
fi

echo "Generating MOT detections with ${MODEL}..."

PYTHONPATH="${PROJECT_ROOT}" \
"${NANODET_PY}" "${PROJECT_ROOT}/tools/generate_mot_detections.py" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --repo_path "${NANODET_DIR}" \
    --model_path "${MODEL_PATH}"

echo "Done."
echo "Detections written to:"
echo "  ${OUTPUT_DIR}"
