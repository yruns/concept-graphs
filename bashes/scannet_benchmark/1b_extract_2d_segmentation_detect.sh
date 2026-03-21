#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/1b_extract_2d_segmentation_detect.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_CONFIG_PATH="${SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/base.yaml}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

cd "${ROOT_DIR}/conceptgraph"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH:-}"
fi

python scripts/generate_gsa_results.py \
    --dataset_root "${SCANNET_SCENE_ROOT}" \
    --dataset_config "${SCANNET_CONFIG_PATH}" \
    --scene_id "${SCENE_NAME}" \
    --class_set ram \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride "${SCANNET_PROCESS_STRIDE}" \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

SCENE_PATH="${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
PREVIEW_DIR="${SCENE_PATH}/checks"
mkdir -p "${PREVIEW_DIR}"
FIRST_VIS="$(find "${SCENE_PATH}/gsa_vis_ram_withbg_allclasses" -maxdepth 1 -type f | sort | sed -n '1p')"
if [[ -n "${FIRST_VIS}" ]]; then
    cp -f "${FIRST_VIS}" "${PREVIEW_DIR}/01_detection_preview.jpg"
fi

echo "[DONE] 2D detections saved under ${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"
echo "[PREVIEW] ${PREVIEW_DIR}/01_detection_preview.jpg"
