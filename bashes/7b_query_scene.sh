#!/bin/bash

# ======================================================================
# 7b_query_scene.sh
# Query-driven keyframe selection demo
#
# This script runs keyframe selection for natural language queries,
# selecting optimal camera views for the specified objects.
#
# Prerequisites:
#   - Run 2b_build_3d_object_map_detect.sh first to generate 3D objects
#   - Run 5b_refine_with_affordance.sh for better object descriptions
#   - Run 6b_build_visibility_index.sh for faster inference
#
# Usage:
#   ./7b_query_scene.sh room0 "pillow on the sofa"
#   ./7b_query_scene.sh room0 "lamp near the table" 5
# ======================================================================

set -e

# Default parameters
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_NAME="${1:-room0}"
QUERY="${2:-pillow on the sofa}"
K="${3:-3}"

# LLM configuration
export LLM_BASE_URL="${LLM_BASE_URL:-http://10.21.231.7:8006}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-2024-08-06}"

# Scene path
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# Output directory
OUTPUT_DIR="${SCENE_PATH}/query_results"

echo "========================================"
echo "Query-Driven Keyframe Selection"
echo "========================================"
echo "Scene: ${SCENE_NAME}"
echo "Path: ${SCENE_PATH}"
echo "Query: ${QUERY}"
echo "K: ${K}"
echo "LLM: ${LLM_MODEL} @ ${LLM_BASE_URL}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Check if scene exists
if [ ! -d "$SCENE_PATH" ]; then
    echo "Error: Scene path does not exist: $SCENE_PATH"
    exit 1
fi

# Check if visibility index exists
INDEX_FILE="${SCENE_PATH}/indices/visibility_index.pkl"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Warning: Visibility index not found: $INDEX_FILE"
    echo "Run 6b_build_visibility_index.sh first for faster inference"
    echo ""
fi

# Build command
CMD="python -m conceptgraph.query_scene.examples.query_keyframes \
    --scene_path ${SCENE_PATH} \
    --query \"${QUERY}\" \
    --k ${K} \
    --output_dir ${OUTPUT_DIR}"

echo ""
echo "Executing:"
echo "----------------------------------------"
echo "$CMD"
echo "----------------------------------------"
echo ""

# Run
eval $CMD

echo ""
echo "========================================"
echo "Query Complete!"
echo "========================================"
