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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Show the Python command that will be executed
echo ""
echo "Executing Python command:"
echo "----------------------------------------"
cat << 'SHOWCMD'
python << 'PYEOF'
from pathlib import Path
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector
from loguru import logger
import cv2
import numpy as np
import os

# Load scene
scene_path = Path(os.environ['SCENE_PATH'])
output_dir = Path(os.environ['OUTPUT_DIR'])
query = os.environ['QUERY']
k = int(os.environ['K'])

selector = KeyframeSelector.from_scene_path(scene_path)

# Run query
logger.info(f'Running query: {query}')
result = selector.select_keyframes(query, k=k)

# Print results
print()
print('=' * 50)
print(f'Query: {query}')
print(f'Target: {result.target_term} -> {len(result.target_objects)} objects')
if result.anchor_term:
    print(f'Anchor: {result.anchor_term} -> {len(result.anchor_objects)} objects')
print(f'Selected keyframes: {result.keyframe_indices}')
print('=' * 50)

# Save visualization
if result.keyframe_paths:
    images = []
    for i, (idx, path) in enumerate(zip(result.keyframe_indices, result.keyframe_paths)):
        if path.exists():
            img = cv2.imread(str(path))
            cv2.putText(img, f'View {idx} (rank {i+1})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            images.append(img)
    
    if images:
        combined = np.hstack(images[:5])
        safe_name = query.replace(' ', '_')[:30]
        out_path = output_dir / f'{safe_name}.jpg'
        cv2.imwrite(str(out_path), combined)
        print(f'Saved: {out_path}')
PYEOF
SHOWCMD
echo "----------------------------------------"
echo ""

# Export variables for Python
export SCENE_PATH
export OUTPUT_DIR
export QUERY
export K

# Run Python
python << 'PYEOF'
from pathlib import Path
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector
from loguru import logger
import cv2
import numpy as np
import os

# Load scene
scene_path = Path(os.environ['SCENE_PATH'])
output_dir = Path(os.environ['OUTPUT_DIR'])
query = os.environ['QUERY']
k = int(os.environ['K'])

selector = KeyframeSelector.from_scene_path(scene_path)

# Run query
logger.info(f'Running query: {query}')
result = selector.select_keyframes(query, k=k)

# Print results
print()
print('=' * 50)
print(f'Query: {query}')
print(f'Target: {result.target_term} -> {len(result.target_objects)} objects')
if result.anchor_term:
    print(f'Anchor: {result.anchor_term} -> {len(result.anchor_objects)} objects')
print(f'Selected keyframes: {result.keyframe_indices}')
print('=' * 50)

# Save visualization
if result.keyframe_paths:
    images = []
    for i, (idx, path) in enumerate(zip(result.keyframe_indices, result.keyframe_paths)):
        if path.exists():
            img = cv2.imread(str(path))
            cv2.putText(img, f'View {idx} (rank {i+1})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            images.append(img)
    
    if images:
        combined = np.hstack(images[:5])
        safe_name = query.replace(' ', '_')[:30]
        out_path = output_dir / f'{safe_name}.jpg'
        cv2.imwrite(str(out_path), combined)
        print(f'Saved: {out_path}')
PYEOF

echo ""
echo "========================================"
echo "Query Complete!"
echo "========================================"
