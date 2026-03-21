#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa-scannet/run_full_detect_pipeline_to_6c.sh <clip_id>"
    exit 1
fi

OPENEQA_PRUNE_DETECTIONS="${OPENEQA_PRUNE_DETECTIONS:-1}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
THRESHOLD="${THRESHOLD:-1.2}"
SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${CLIP_ID}/conceptgraph"
PCD_POST_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
INDEX_FILE="${SCENE_PATH}/indices/visibility_index.pkl"
BUILD_INFO_FILE="${SCENE_PATH}/indices/build_info.json"

bash "${SCRIPT_DIR}/1_prepare_clip_scene.sh" "${CLIP_ID}"
bash "${SCRIPT_DIR}/1b_extract_2d_segmentation_detect.sh" "${CLIP_ID}"
bash "${SCRIPT_DIR}/2b_build_3d_object_map_detect.sh" "${CLIP_ID}"
bash "${SCRIPT_DIR}/6b_build_visibility_index.sh" "${CLIP_ID}"

if [[ "${OPENEQA_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${CLIP_ID}"
fi

if [[ ! -f "${PCD_POST_FILE}" ]]; then
    echo "[ERROR] Missing post-processed object map: ${PCD_POST_FILE}"
    exit 1
fi
if [[ ! -f "${INDEX_FILE}" ]]; then
    echo "[ERROR] Missing visibility index: ${INDEX_FILE}"
    exit 1
fi

python - "${BUILD_INFO_FILE}" "${CLIP_ID}" "${OPENEQA_PROCESS_STRIDE}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
clip_id = sys.argv[2]
process_stride = int(sys.argv[3])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(
        {
            "clip_id": clip_id,
            "pipeline": "openeqa-scannet",
            "process_stride": process_stride,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
PY

echo "[DONE] OpenEQA-ScanNet full pipeline completed for ${CLIP_ID}"
echo "[SCENE] ${SCENE_PATH}"
