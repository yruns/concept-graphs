#!/bin/bash
# =============================================================================
# Step 7: CLIP Feature Clustering for Scene Segmentation
# =============================================================================
# Based on CLIP features and spatial coordinates for joint clustering
#
# Prerequisites:
#   1b -> 2b (must be run first)
#
# Input:
#   - 3D object map (pcd_saves/*.pkl.gz)
#
# Output:
#   - clip_zones.json
#   - zones_colored.ply
#   - zones_legend.png
#   - clustering_stats.json
#
# Usage:
#   ./7_clip_feature_clustering.sh [SCENE_NAME] [ALPHA] [BETA]
# =============================================================================

set -e

# Activate environment
if [ -f "/home/shyue/anaconda3/bin/activate" ]; then
    source /home/shyue/anaconda3/bin/activate conceptgraph
fi

# Load environment variables
if [ -f "./env_vars.bash" ]; then
    source ./env_vars.bash
fi

# Default parameters
SCENE_NAME="${1:-room0}"
ALPHA="${2:-1.0}"             # CLIP feature weight
BETA="${3:-0.3}"              # Coordinate weight
MIN_CLUSTER_SIZE="${4:-50}"   # Min cluster size (reduced for faster)
MIN_SAMPLES="${5:-10}"        # Min samples (reduced)
DOWNSAMPLE_VOXEL="${6:-0.08}" # Voxel downsample size (larger = fewer points)
PCA_DIM="${7:-32}"            # PCA dimensions (smaller = less memory)

# Dataset path
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# PCD file path
THRESHOLD=1.2
PCD_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

# Output directory
OUTPUT_DIR="${SCENE_PATH}/clip_feature_zones"

echo "============================================================"
echo "Step 7: CLIP Feature Clustering"
echo "============================================================"
echo "Scene: ${SCENE_NAME}"
echo "PCD file: ${PCD_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  Alpha (CLIP weight): ${ALPHA}"
echo "  Beta (Coord weight): ${BETA}"
echo "  Min cluster size: ${MIN_CLUSTER_SIZE}"
echo "  Min samples: ${MIN_SAMPLES}"
echo "  Downsample voxel: ${DOWNSAMPLE_VOXEL}m"
echo "  PCA dimension: ${PCA_DIM}"
echo "============================================================"

# Check input file
if [ ! -f "${PCD_FILE}" ]; then
    echo "Error: PCD file not found: ${PCD_FILE}"
    echo "Please run step 2b first: ./2b_build_3d_object_map_detect.sh ${SCENE_NAME}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run clustering
cd "$(dirname "$0")/.."

python -m conceptgraph.segmentation.clip_feature_clustering \
    --pcd_file "${PCD_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --min_cluster_size "${MIN_CLUSTER_SIZE}" \
    --min_samples "${MIN_SAMPLES}" \
    --downsample_voxel "${DOWNSAMPLE_VOXEL}" \
    --pca_dim "${PCA_DIM}" \
    --use_dbscan \
    --dbscan_eps 0.8

echo ""
echo "============================================================"
echo "Done! Output files:"
echo "  - ${OUTPUT_DIR}/clip_zones.json"
echo "  - ${OUTPUT_DIR}/zones_colored.ply"
echo "  - ${OUTPUT_DIR}/zones_legend.png"
echo "  - ${OUTPUT_DIR}/clustering_stats.json"
echo "============================================================"
