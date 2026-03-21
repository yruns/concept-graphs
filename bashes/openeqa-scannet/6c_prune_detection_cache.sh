#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa-scannet/6c_prune_detection_cache.sh <clip_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${CLIP_ID}/conceptgraph"
DETECTIONS_DIR="${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python -m conceptgraph.scripts.prune_gsa_detection_cache \
    --detections_dir "${DETECTIONS_DIR}"

echo "[DONE] Pruned detection cache: ${DETECTIONS_DIR}"

