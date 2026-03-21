#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/run_full_detect_pipeline_to_6b.sh <scene_id>"
    exit 1
fi

RUN_CAPTIONS="${RUN_CAPTIONS:-0}"
SCANNET_PRUNE_DETECTIONS="${SCANNET_PRUNE_DETECTIONS:-1}"

bash "${SCRIPT_DIR}/1_prepare_replica_like_scene.sh" "${SCENE_NAME}"
bash "${SCRIPT_DIR}/1b_extract_2d_segmentation_detect.sh" "${SCENE_NAME}"
bash "${SCRIPT_DIR}/2b_build_3d_object_map_detect.sh" "${SCENE_NAME}"

if [[ "${RUN_CAPTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    bash "${SCRIPT_DIR}/4b_extract_object_captions_detect.sh" "${SCENE_NAME}"
    bash "${SCRIPT_DIR}/5b_refine_with_affordance.sh" "${SCENE_NAME}"
fi

bash "${SCRIPT_DIR}/6b_build_visibility_index.sh" "${SCENE_NAME}"

if [[ "${SCANNET_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${SCENE_NAME}"
fi

echo "[DONE] ScanNet labeled pipeline completed for ${SCENE_NAME}"
