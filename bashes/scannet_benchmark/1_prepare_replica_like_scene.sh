#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/1_prepare_replica_like_scene.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-5}"
SCANNET_MAX_FRAMES="${SCANNET_MAX_FRAMES:-}"
SCANNET_OVERWRITE="${SCANNET_OVERWRITE:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

CMD=(
    python -m conceptgraph.scripts.prepare_scannet_replica_scene
    --scene_id "${SCENE_NAME}"
    --scannet_root "${SCANNET_RAW_ROOT}"
    --output_root "${SCANNET_SCENE_ROOT}"
    --frame_skip "${SCANNET_FRAME_SKIP}"
)

if [[ -n "${SCANNET_MAX_FRAMES}" ]]; then
    CMD+=(--max_frames "${SCANNET_MAX_FRAMES}")
fi

if [[ "${SCANNET_OVERWRITE}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    CMD+=(--overwrite)
fi

"${CMD[@]}"

echo "[DONE] Prepared scene at ${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
echo "[PREVIEW] ${SCANNET_SCENE_ROOT}/${SCENE_NAME}/checks/00_scene_wrapper_preview.jpg"
