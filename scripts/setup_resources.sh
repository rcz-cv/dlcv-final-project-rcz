#!/usr/bin/env bash
#
# setup_resources.sh
#
# This script populates the resources folder the correct folder structure,
# including DeepSORT's baseline MARS model.
#
# Usage: bash scripts/setup_resources.sh
#

set -euo pipefail

PROJECT_ROOT="$(pwd)"

mkdir -p "${PROJECT_ROOT}/resources/networks"
tar xzf "${PROJECT_ROOT}/mars-small128.tar.gz" -C "${PROJECT_ROOT}/resources/networks/"
