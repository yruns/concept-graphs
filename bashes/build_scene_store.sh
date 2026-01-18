#!/bin/bash
# =============================================================================
# Build Scene Vector Store
# =============================================================================
# 从ConceptGraphs的输出构建隐式场景表示
#
# Input:
#   - pcd_saves/*.pkl.gz (3D物体地图)
#   - gsa_detections_*/*.pkl.gz (帧级特征, 可选)
#
# Output:
#   - scene_store.pkl.gz (场景向量存储)
#
# Usage:
#   ./build_scene_store.sh [SCENE_NAME]
# =============================================================================

set -e

# Activate environment
if [ -f "/home/shyue/anaconda3/bin/activate" ]; then
    source /home/shyue/anaconda3/bin/activate conceptgraph
fi

# Parameters
SCENE_NAME="${1:-room0}"
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# File paths
THRESHOLD=1.2
PCD_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
GSA_DIR="${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"
OUTPUT_DIR="${SCENE_PATH}/implicit_scene"

echo "============================================================"
echo "Building Scene Vector Store"
echo "============================================================"
echo "Scene: ${SCENE_NAME}"
echo "PCD file: ${PCD_FILE}"
echo "GSA dir: ${GSA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

# Check input
if [ ! -f "${PCD_FILE}" ]; then
    echo "Error: PCD file not found"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run builder
cd /home/shyue/codebase/concept-graphs

python << EOF
import sys
sys.path.insert(0, '/home/shyue/codebase/concept-graphs')

from implicit_scene.store.vector_store import SceneVectorStore
import json

# Create store
store = SceneVectorStore()

# Load data
pcd_file = "${PCD_FILE}"
gsa_dir = "${GSA_DIR}" if "${GSA_DIR}" else None

store.load_from_pcd(pcd_file, gsa_dir)

# Save
output_path = "${OUTPUT_DIR}/scene_store.pkl.gz"
store.save(output_path)

# Save summary
summary = store.summary()
with open("${OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nScene summary:")
print(f"  Objects: {summary['n_objects']}")
print(f"  Frames: {summary['n_frames']}")
print(f"  Top tags: {list(summary['object_tags'].keys())[:10]}")
EOF

echo ""
echo "============================================================"
echo "Done! Output files:"
echo "  - ${OUTPUT_DIR}/scene_store.pkl.gz"
echo "  - ${OUTPUT_DIR}/summary.json"
echo "============================================================"
