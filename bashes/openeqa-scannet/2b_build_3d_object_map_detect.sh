#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa-scannet/2b_build_3d_object_map_detect.sh <clip_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
OPENEQA_SCANNET_CONFIG_PATH="${OPENEQA_SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
THRESHOLD="${THRESHOLD:-1.2}"
SCENE_SPEC="${CLIP_ID}/conceptgraph"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

cd "${ROOT_DIR}/conceptgraph"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH:-}"
fi

python slam/cfslam_pipeline_batch.py \
    dataset_root="${OPENEQA_SCANNET_ROOT}" \
    dataset_config="${OPENEQA_SCANNET_CONFIG_PATH}" \
    stride="${OPENEQA_PROCESS_STRIDE}" \
    scene_id="${SCENE_SPEC}" \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold="${THRESHOLD}" \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    class_agnostic=False \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix="overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub" \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${SCENE_SPEC}"
PKL_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
PREVIEW_DIR="${SCENE_PATH}/checks/02_object_map_preview"
MESH_PATH="${SCENE_PATH}/mesh.ply"

VIS_CMD=(
    python scripts/visualize_cfslam_results_offscreen.py
    --result_path "${PKL_FILE}" \
    --output_dir "${PREVIEW_DIR}" \
    --output_format images \
    --num_views 1 \
    --image_width 1600 \
    --image_height 900
)

if [[ -f "${MESH_PATH}" ]]; then
    VIS_CMD+=(--original_mesh "${MESH_PATH}")
fi

"${VIS_CMD[@]}"

echo "[DONE] 3D object map saved to ${PKL_FILE}"
echo "[PREVIEW] ${PREVIEW_DIR}/images/scene_graph_view_00.png"
