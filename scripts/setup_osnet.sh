#!/usr/bin/env bash
#
# setup_osnet.sh
#
# This script installs the deep-person-reid repo so we can use OSNet ReID.
#

set -euo pipefail

PROJECT_ROOT="$(cd ..; pwd)"

OSNET_DIR="${PROJECT_ROOT}/external/deep-person-reid"

mkdir -p "${PROJECT_ROOT}/external"

if [[ ! -d "${OSNET_DIR}" ]]; then
    echo "Cloning OSNet..."
    git clone https://github.com/KaiyangZhou/deep-person-reid.git "${OSNET_DIR}"
    (
        cd "${OSNET_DIR}"
        git checkout
    )
fi

# install OSNet dependencies in our current environment
python -m pip install torch torchvision yacs gdown scipy opencv-python tensorboard
