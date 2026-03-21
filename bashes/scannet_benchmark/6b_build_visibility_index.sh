#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/6b_build_visibility_index.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"
SCENE_PATH="${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
PCD_FILE="$(find "${SCENE_PATH}/pcd_saves" -maxdepth 1 -type f -name '*ram*_post.pkl.gz' | sort | head -n 1)"

if [[ -z "${PCD_FILE}" ]]; then
    echo "[ERROR] Post-processed object map not found under ${SCENE_PATH}/pcd_saves"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python -m conceptgraph.scripts.build_visibility_index \
    --scene_path "${SCENE_PATH}" \
    --pcd_file "${PCD_FILE}" \
    --stride "${SCANNET_PROCESS_STRIDE}"

echo "[DONE] Visibility index built: ${SCENE_PATH}/indices/visibility_index.pkl"
