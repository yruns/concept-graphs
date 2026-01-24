#!/bin/bash

# ======================================================================
# 6b_build_visibility_index.sh
# Build offline visibility index for query-driven keyframe selection
#
# This script precomputes the bidirectional object-view visibility index
# for faster keyframe selection during inference.
#
# Prerequisites:
#   - Run 2b_build_3d_object_map_detect.sh first to generate 3D objects
#
# Output:
#   - scene_path/indices/visibility_index.pkl
# ======================================================================

set -e

# Default parameters
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_NAME="${1:-room0}"
STRIDE="${STRIDE:-5}"
MAX_DISTANCE="${MAX_DISTANCE:-5.0}"
USE_DEPTH="${USE_DEPTH:-false}"

# Scene path
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

echo "========================================"
echo "Building Visibility Index"
echo "========================================"
echo "Scene: ${SCENE_NAME}"
echo "Path: ${SCENE_PATH}"
echo "Stride: ${STRIDE}"
echo "Max Distance: ${MAX_DISTANCE}"
echo "Use Depth: ${USE_DEPTH}"
echo "========================================"

# Check if scene exists
if [ ! -d "$SCENE_PATH" ]; then
    echo "Error: Scene path does not exist: $SCENE_PATH"
    exit 1
fi

# Check if PCD file exists
PCD_DIR="${SCENE_PATH}/pcd_saves"
if [ ! -d "$PCD_DIR" ]; then
    echo "Error: PCD directory not found: $PCD_DIR"
    echo "Please run 2b_build_3d_object_map_detect.sh first"
    exit 1
fi

# Build command
CMD="python -m conceptgraph.scripts.build_visibility_index \
    --scene_path ${SCENE_PATH} \
    --stride ${STRIDE} \
    --max_distance ${MAX_DISTANCE}"

if [ "$USE_DEPTH" = "true" ]; then
    CMD="$CMD --use_depth"
fi

echo "Running: $CMD"
echo ""

# Run
eval $CMD

echo ""
echo "========================================"
echo "Visibility Index Built Successfully!"
echo "Output: ${SCENE_PATH}/indices/visibility_index.pkl"
echo "========================================"
