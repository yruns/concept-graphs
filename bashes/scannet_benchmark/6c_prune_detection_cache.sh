#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/6c_prune_detection_cache.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCENE_PATH="${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
DETECTIONS_DIR="${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python -m conceptgraph.scripts.prune_gsa_detection_cache \
    --detections_dir "${DETECTIONS_DIR}"

echo "[DONE] Pruned detection cache: ${DETECTIONS_DIR}"
