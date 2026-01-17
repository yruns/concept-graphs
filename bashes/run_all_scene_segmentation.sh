#!/bin/bash
# =============================================================================
# Run Region-Aware Scene Segmentation for All Scenes
# =============================================================================

set -e

DATASET_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/shyue/codebase/concept-graphs}"
OUTPUT_DIR="${WORKSPACE_ROOT}/scene_seg"
STRIDE=5
MAX_REGIONS=10

SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

echo "========================================"
echo "  Region-Aware Scene Segmentation"
echo "========================================"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir:   ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

for scene in "${SCENES[@]}"; do
    echo "Processing: ${scene}"
    scene_path="${DATASET_ROOT}/${scene}"
    seg_output="${scene_path}/sg_cache/segmentation_regions"
    
    # Check prerequisites
    if [ ! -f "${scene_path}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz" ]; then
        echo "  [SKIP] Missing 3D object map"
        continue
    fi
    
    if [ ! -d "${scene_path}/sg_cache/cfslam_gpt-4_responses" ]; then
        echo "  [SKIP] Missing GPT responses"
        continue
    fi
    
    # Run segmentation (includes GIF generation)
    echo "  [1/2] Running segmentation + GIF generation..."
    python "${WORKSPACE_ROOT}/conceptgraph/segmentation/region_aware_segmenter.py" \
        --dataset_root "${DATASET_ROOT}" \
        --scene "${scene}" \
        --stride "${STRIDE}" \
        --max_regions "${MAX_REGIONS}" || { echo "  [ERROR] Segmentation failed"; continue; }
    
    # Generate additional interpretability visualizations
    echo "  [2/2] Generating interpretability charts..."
    python -c "
import json, numpy as np, gzip, pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
scene_path = Path('${scene_path}')
output_dir = scene_path / 'sg_cache' / 'segmentation_regions'
stride = ${STRIDE}
with open(output_dir / 'regions.json') as f: regions = json.load(f)
with gzip.open(scene_path / 'pcd_saves' / 'full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz', 'rb') as f: objects = pickle.load(f)['objects']
object_tags = {}
for gf in (scene_path / 'sg_cache' / 'cfslam_gpt-4_responses').glob('*.json'):
    try: object_tags[int(gf.stem)] = json.loads(json.load(open(gf)).get('response','{}')).get('object_tag', f'obj_{gf.stem}')
    except: object_tags[int(gf.stem)] = f'obj_{gf.stem}'
rgb_files = sorted((scene_path / 'results').glob('frame*.jpg'))
n_frames = len(rgb_files) // stride if rgb_files else 400
frame_objects = [set() for _ in range(n_frames)]
for oid, obj in enumerate(objects):
    for fi in obj.get('image_idx', []):
        if 0 <= fi < n_frames: frame_objects[fi].add(oid)
boundaries = []
for r in regions:
    for s in r['segments']:
        sf = s['start_frame_original'] // stride
        if sf > 0 and sf not in [b['frame'] for b in boundaries]: boundaries.append({'frame': sf, 'frame_orig': s['start_frame_original']})
boundaries.sort(key=lambda x: x['frame'])
# Fig1: Reasons
if boundaries and rgb_files:
    fig, axes = plt.subplots(len(boundaries), 3, figsize=(16, len(boundaries) * 5))
    if len(boundaries) == 1: axes = axes.reshape(1, -1)
    for idx, b in enumerate(boundaries):
        bf, af = max(0, b['frame'] - 5), min(n_frames - 1, b['frame'] + 5)
        bo, ao = (frame_objects[bf] if bf < len(frame_objects) else set()), (frame_objects[af] if af < len(frame_objects) else set())
        if bf * stride < len(rgb_files): axes[idx, 0].imshow(Image.open(rgb_files[bf * stride]))
        axes[idx, 0].set_title(f'BEFORE ({bf * stride})', fontweight='bold'); axes[idx, 0].axis('off')
        if af * stride < len(rgb_files): axes[idx, 1].imshow(Image.open(rgb_files[af * stride]))
        axes[idx, 1].set_title(f'AFTER ({af * stride})', fontweight='bold'); axes[idx, 1].axis('off')
        axes[idx, 2].axis('off'); axes[idx, 2].text(0.5, 0.95, f'SPLIT: Frame {b[\"frame_orig\"]}', transform=axes[idx, 2].transAxes, fontsize=14, fontweight='bold', ha='center', va='top')
        y = 0.85; axes[idx, 2].text(0.05, y, 'LEAVING:', fontsize=11, fontweight='bold', color='red', transform=axes[idx, 2].transAxes)
        for o in list(bo - ao)[:6]: y -= 0.06; axes[idx, 2].text(0.08, y, f'- {object_tags.get(o, str(o))}', fontsize=10, color='darkred', transform=axes[idx, 2].transAxes)
        y -= 0.08; axes[idx, 2].text(0.05, y, 'ENTERING:', fontsize=11, fontweight='bold', color='green', transform=axes[idx, 2].transAxes)
        for o in list(ao - bo)[:6]: y -= 0.06; axes[idx, 2].text(0.08, y, f'+ {object_tags.get(o, str(o))}', fontsize=10, color='darkgreen', transform=axes[idx, 2].transAxes)
    plt.suptitle('SEGMENTATION REASONS', fontsize=16, fontweight='bold', y=1.02); plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_reasons.png', dpi=150, bbox_inches='tight'); plt.close()
# Fig2: Semantics
if regions and rgb_files:
    fig = plt.figure(figsize=(18, 6 * len(regions))); cols = plt.cm.Set2(np.linspace(0, 1, len(regions)))
    for i, r in enumerate(regions):
        s = r['segments'][0]; mid = (s['start_frame_original'] + s['end_frame_original']) // 2
        ax1 = fig.add_subplot(len(regions), 3, i*3 + 1)
        if mid < len(rgb_files): ax1.imshow(Image.open(rgb_files[mid]))
        ax1.set_title(f'Region {r[\"region_id\"]} (Frame {mid})', fontweight='bold'); ax1.axis('off')
        ax2 = fig.add_subplot(len(regions), 3, i*3 + 2)
        tc = Counter([object_tags.get(o, '') for o in r.get('object_ids', []) if o in object_tags])
        if tc: tt = tc.most_common(8); ax2.pie([t[1] for t in tt], labels=[t[0][:18] for t in tt], autopct='%1.0f%%', colors=plt.cm.Set3(np.linspace(0, 1, len(tt))))
        ax2.set_title(f'Region {r[\"region_id\"]} Objects', fontweight='bold')
        ax3 = fig.add_subplot(len(regions), 3, i*3 + 3); ax3.axhline(y=0.5, color='gray', linewidth=2, alpha=0.5); ax3.set_xlim(0, 2000); ax3.set_ylim(0, 1)
        for s in r['segments']: ax3.axvspan(s['start_frame_original'], s['end_frame_original'], alpha=0.6, color=cols[i])
        ax3.set_title(f'Region {r[\"region_id\"]} Timeline', fontweight='bold'); ax3.set_xlabel('Frame'); ax3.set_yticks([])
    plt.suptitle('REGION SEMANTICS', fontsize=16, fontweight='bold', y=1.01); plt.tight_layout()
    plt.savefig(output_dir / 'region_semantics.png', dpi=150, bbox_inches='tight'); plt.close()
# Fig3: Timeline
valid = [o for o in range(len(objects)) if any(o in f for f in frame_objects)]
if valid:
    fig, ax = plt.subplots(figsize=(20, max(12, len(valid) * 0.4))); ylbl = []; pc = plt.cm.tab20(np.linspace(0, 1, 20))
    for y, oid in enumerate(valid):
        ylbl.append(f'ID={oid}: {object_tags.get(oid, str(oid))[:35]}')
        vf = [f * stride for f in range(n_frames) if oid in frame_objects[f]]
        if vf:
            segs, st, pv = [], vf[0], vf[0]
            for f in vf[1:]:
                if f - pv > stride * 2: segs.append((st, pv)); st = f
                pv = f
            segs.append((st, pv))
            for s, e in segs: ax.barh(y, e - s + stride, left=s, height=0.8, color=pc[oid % 20], alpha=0.8, edgecolor='black', linewidth=0.3)
    for i, b in enumerate(boundaries): ax.axvline(x=b['frame_orig'], color='red', linestyle='--', linewidth=2, alpha=0.9)
    ax.set_yticks(range(len(ylbl))); ax.set_yticklabels(ylbl, fontsize=8)
    ax.set_xlabel('Frame', fontsize=12, fontweight='bold'); ax.set_xlim(0, 2000); ax.set_ylim(-0.5, len(valid) + 1.5)
    ax.set_title(f'OBJECT VISIBILITY TIMELINE ({len(objects)} objects)', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout(); plt.savefig(output_dir / 'object_visibility_timeline_full.png', dpi=120, bbox_inches='tight'); plt.close()
print('  Visualizations generated')
" 2>/dev/null || echo "  [WARN] Visualization generation failed"
    
    # Create symlink
    ln -sfn "${seg_output}" "${OUTPUT_DIR}/${scene}"
    echo "  [DONE] Symlink: ${OUTPUT_DIR}/${scene}"
    echo ""
done

echo "========================================"
echo "  Complete!"
echo "========================================"
echo "Output: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}/"
