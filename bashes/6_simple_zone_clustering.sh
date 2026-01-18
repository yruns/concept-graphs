#!/bin/bash
# =============================================================================
# Step 6: Simple Zone Clustering (DBSCAN + Heuristic/CLIP naming)
# =============================================================================
# Fast prototype for functional zone segmentation
#
# Input:
#   - 3D object map (pcd_saves/*.pkl.gz)
#
# Output:
#   - simple_zones.json: Zone clustering result
#
# Usage:
#   ./6_simple_zone_clustering.sh [SCENE_NAME] [EPS] [MIN_SAMPLES]
#
# Example:
#   ./6_simple_zone_clustering.sh room0           # Default params
#   ./6_simple_zone_clustering.sh room0 1.5 2     # Custom DBSCAN params
#   ./6_simple_zone_clustering.sh room0 2.0 3     # Larger clusters
# =============================================================================

set -e

# Default parameters
SCENE_NAME="${1:-room0}"
EPS="${2:-1.5}"           # DBSCAN epsilon (max distance in cluster)
MIN_SAMPLES="${3:-2}"     # DBSCAN min samples per cluster

# Dataset path
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# Use category-aware pcd file if available, otherwise category-agnostic
THRESHOLD=1.2
PCD_FILE_DETECT="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
PCD_FILE_NONE="${SCENE_PATH}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

if [ -f "${PCD_FILE_DETECT}" ]; then
    PCD_FILE="${PCD_FILE_DETECT}"
    OUTPUT_DIR="${SCENE_PATH}/simple_zones_detect"
elif [ -f "${PCD_FILE_NONE}" ]; then
    PCD_FILE="${PCD_FILE_NONE}"
    OUTPUT_DIR="${SCENE_PATH}/simple_zones"
else
    echo "Error: No pcd file found for ${SCENE_NAME}"
    echo "Please run step 2 first (2_build_3d_object_map.sh or 2b_build_3d_object_map_detect.sh)"
    exit 1
fi

echo "============================================================"
echo "Step 6: Simple Zone Clustering"
echo "============================================================"
echo "Scene: ${SCENE_NAME}"
echo "PCD file: ${PCD_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "DBSCAN params: eps=${EPS}, min_samples=${MIN_SAMPLES}"
echo "============================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run clustering
cd "$(dirname "$0")/.."
python -m conceptgraph.segmentation.simple_zone_clustering \
    --pcd_file "${PCD_FILE}" \
    --output "${OUTPUT_DIR}/simple_zones.json" \
    --eps "${EPS}" \
    --min_samples "${MIN_SAMPLES}"

# Generate visualization
echo ""
echo "Generating visualization..."
python << EOF
import json
import gzip
import pickle
import numpy as np
from pathlib import Path

# Load zones
with open("${OUTPUT_DIR}/simple_zones.json") as f:
    data = json.load(f)
zones = data["zones"]

# Load pcd for point cloud
with gzip.open("${PCD_FILE}", 'rb') as f:
    pcd_data = pickle.load(f)
objects = pcd_data.get('objects', [])

# Generate colors for zones
import distinctipy
n_zones = len(zones)
colors = distinctipy.get_colors(max(n_zones, 1), pastel_factor=0.5)

# Build object-to-zone mapping
obj_to_zone = {}
for i, zone in enumerate(zones):
    for obj in zone["objects"]:
        obj_to_zone[obj["id"]] = i

# Write colored PLY
output_ply = "${OUTPUT_DIR}/zones_colored.ply"
all_points = []
all_colors = []
default_color = [128, 128, 128]

for i, obj in enumerate(objects):
    pcd_np = obj.get('pcd_np')
    if pcd_np is None or len(pcd_np) == 0:
        continue
    
    zone_idx = obj_to_zone.get(i)
    if zone_idx is not None and zone_idx < len(colors):
        color = [int(c * 255) for c in colors[zone_idx]]
    else:
        color = default_color
    
    all_points.append(pcd_np.astype(np.float32))
    all_colors.append(np.tile(color, (len(pcd_np), 1)))

if all_points:
    points = np.vstack(all_points)
    colors_arr = np.vstack(all_colors)
    
    import struct
    import sys
    ply_format = 'binary_little_endian' if sys.byteorder == 'little' else 'binary_big_endian'
    
    with open(output_ply, 'wb') as f:
        header = f"""ply
format {ply_format} 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        for i in range(len(points)):
            f.write(struct.pack('fffBBB',
                float(points[i, 0]), float(points[i, 1]), float(points[i, 2]),
                int(colors_arr[i, 0]), int(colors_arr[i, 1]), int(colors_arr[i, 2])))
    
    print(f"PLY saved to: {output_ply} ({len(points)} points)")

# Generate legend
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, max(2, n_zones * 0.6)))
ax.set_xlim(0, 1)
ax.set_ylim(0, n_zones + 1)
ax.axis('off')
ax.set_title("Zone Color Legend", fontsize=12, fontweight='bold')

for i, zone in enumerate(zones):
    y = n_zones - i
    color = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
    ax.add_patch(plt.Rectangle((0.02, y - 0.4), 0.08, 0.8, facecolor=color, edgecolor='black'))
    ax.text(0.12, y, f"{zone['zone_name']} ({zone['n_objects']} objects)", fontsize=10, va='center')

plt.tight_layout()
plt.savefig("${OUTPUT_DIR}/zones_legend.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Legend saved to: ${OUTPUT_DIR}/zones_legend.png")
EOF

echo ""
echo "============================================================"
echo "Done! Output files:"
echo "  - ${OUTPUT_DIR}/simple_zones.json"
echo "  - ${OUTPUT_DIR}/zones_colored.ply"
echo "  - ${OUTPUT_DIR}/zones_legend.png"
echo "============================================================"
