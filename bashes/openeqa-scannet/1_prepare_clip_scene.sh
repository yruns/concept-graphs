#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa-scannet/1_prepare_clip_scene.sh <clip_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT:-${HOME}/Datasets/open-eqa/data/frames/scannet-v0}"
OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
OPENEQA_PREFER_SYMLINK="${OPENEQA_PREFER_SYMLINK:-1}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

CMD=(
    python -m conceptgraph.scripts.prepare_openeqa_scannet_scene
    --clip_id "${CLIP_ID}"
    --frames_root "${OPENEQA_SCANNET_FRAMES_ROOT}"
    --scannet_root "${SCANNET_RAW_ROOT}"
    --output_root "${OPENEQA_SCANNET_ROOT}"
)

if [[ "${OPENEQA_PREFER_SYMLINK}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    CMD+=(--prefer_symlink)
fi

"${CMD[@]}"

echo "[DONE] Prepared ${OPENEQA_SCANNET_ROOT}/${CLIP_ID}/conceptgraph"

