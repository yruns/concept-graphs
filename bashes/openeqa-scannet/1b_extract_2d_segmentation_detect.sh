#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa-scannet/1b_extract_2d_segmentation_detect.sh <clip_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
OPENEQA_SCANNET_CONFIG_PATH="${OPENEQA_SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
SCENE_SPEC="${CLIP_ID}/conceptgraph"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

cd "${ROOT_DIR}/conceptgraph"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH:-}"
fi

python scripts/generate_gsa_results.py \
    --dataset_root "${OPENEQA_SCANNET_ROOT}" \
    --dataset_config "${OPENEQA_SCANNET_CONFIG_PATH}" \
    --scene_id "${SCENE_SPEC}" \
    --class_set ram \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride "${OPENEQA_PROCESS_STRIDE}" \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${SCENE_SPEC}"
PREVIEW_DIR="${SCENE_PATH}/checks"
mkdir -p "${PREVIEW_DIR}"
FIRST_VIS="$(find "${SCENE_PATH}/gsa_vis_ram_withbg_allclasses" -maxdepth 1 -type f | sort | sed -n '1p')"
if [[ -n "${FIRST_VIS}" ]]; then
    cp -f "${FIRST_VIS}" "${PREVIEW_DIR}/01_detection_preview.png"
fi

echo "[DONE] 2D detections saved under ${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"
echo "[PREVIEW] ${PREVIEW_DIR}/01_detection_preview.png"
