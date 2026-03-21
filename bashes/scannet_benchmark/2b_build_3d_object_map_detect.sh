#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/2b_build_3d_object_map_detect.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_CONFIG_PATH="${SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/base.yaml}"
THRESHOLD="${THRESHOLD:-1.2}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

cd "${ROOT_DIR}/conceptgraph"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH:-}"
fi

python slam/cfslam_pipeline_batch.py \
    dataset_root="${SCANNET_SCENE_ROOT}" \
    dataset_config="${SCANNET_CONFIG_PATH}" \
    stride="${SCANNET_PROCESS_STRIDE}" \
    scene_id="${SCENE_NAME}" \
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

SCENE_PATH="${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
PKL_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
PREVIEW_DIR="${SCENE_PATH}/checks/02_object_map_preview"

python scripts/visualize_cfslam_results_offscreen.py \
    --result_path "${PKL_FILE}" \
    --output_dir "${PREVIEW_DIR}" \
    --output_format images \
    --num_views 1 \
    --image_width 1600 \
    --image_height 900 \
    --original_mesh "${SCANNET_SCENE_ROOT}/${SCENE_NAME}_mesh.ply"

echo "[DONE] 3D object map saved to ${PKL_FILE}"
echo "[PREVIEW] ${PREVIEW_DIR}/images/scene_graph_view_00.png"
