#!/usr/bin/env python3
"""
Generate interpretability visualizations for scene segmentation.
"""
import argparse
import json
import numpy as np
import gzip
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    
    scene_path = Path(args.scene_path)
    output_dir = scene_path / 'sg_cache' / 'segmentation_regions'
    stride = args.stride
    
    # Load data
    with open(output_dir / 'regions.json') as f:
        regions = json.load(f)
    
    pcd_file = scene_path / 'pcd_saves' / 'full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz'
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    objects = data['objects']
    
    # Load object tags
    gpt_dir = scene_path / 'sg_cache' / 'cfslam_gpt-4_responses'
    object_tags = {}
    for gpt_file in gpt_dir.glob('*.json'):
        obj_id = int(gpt_file.stem)
        with open(gpt_file) as f:
            resp_data = json.load(f)
        try:
            resp = json.loads(resp_data.get('response', '{}'))
            object_tags[obj_id] = resp.get('object_tag', f'object_{obj_id}')
        except:
            object_tags[obj_id] = f'object_{obj_id}'
    
    # RGB files
    rgb_dir = scene_path / 'results'
    rgb_files = sorted(rgb_dir.glob('frame*.jpg'))
    n_frames = len(rgb_files) // stride if rgb_files else 400
    
    # Build frame_objects
    frame_objects = [set() for _ in range(n_frames)]
    seen = set()
    for region in regions:
        for seg in region['segments']:
            start = seg['start_frame_original'] // stride
            if start > 0 and start not in seen:
                boundaries.append({'frame': start, 'frame_orig': seg['start_frame_original']})
                seen.add(start)
    boundaries = sorted(boundaries, key=lambda x: x['frame'])
    
    # Seg reasons
    if boundaries and rgb_files:
        n_b = len(boundaries)
        fig, axes = plt.subplots(n_b, 3, figsize=(16, n_b * 5))
        if n_b == 1: axes = axes.reshape(1, -1)
        for idx, b in enumerate(boundaries):
            fr, fo = b['frame'], b['frame_orig']
            bf, af = max(0, fr - 5), min(n_frames - 1, fr + 5)
            bo = frame_objects[bf] if bf < len(frame_objects) else set()
            ao = frame_objects[af] if af < len(frame_objects) else set()
            if bf * stride < len(rgb_files): axes[idx, 0].imshow(Image.open(rgb_files[bf * stride]))
            axes[idx, 0].set_title(f'BEFORE (Frame {bf * stride})', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            if af * stride < len(rgb_files): axes[idx, 1].imshow(Image.open(rgb_files[af * stride]))
            axes[idx, 1].set_title(f'AFTER (Frame {af * stride})', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            axes[idx, 2].axis('off')
            axes[idx, 2].text(0.5, 0.95, f'SPLIT: Frame {fo}', transform=axes[idx, 2].transAxes, fontsize=14, fontweight='bold', ha='center', va='top')
            y = 0.85
            axes[idx, 2].text(0.05, y, 'LEAVING:', fontsize=11, fontweight='bold', color='red', transform=axes[idx, 2].transAxes)
            for oid in list(bo - ao)[:6]: y -= 0.06; axes[idx, 2].text(0.08, y, f'- {object_tags.get(oid, str(oid))}', fontsize=10, color='darkred', transform=axes[idx, 2].transAxes)
            y -= 0.08
            axes[idx, 2].text(0.05, y, 'ENTERING:', fontsize=11, fontweight='bold', color='green', transform=axes[idx, 2].transAxes)
            for oid in list(ao - bo)[:6]: y -= 0.06; axes[idx, 2].text(0.08, y, f'+ {object_tags.get(oid, str(oid))}', fontsize=10, color='darkgreen', transform=axes[idx, 2].transAxes)
        plt.suptitle('SEGMENTATION REASONS', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(); plt.savefig(output_dir / 'segmentation_reasons.png', dpi=150, bbox_inches='tight'); plt.close()
        print("    Saved: segmentation_reasons.png")
    
    # Region semantics
    if regions and rgb_files:
        fig = plt.figure(figsize=(18, 6 * len(regions)))
        cols = plt.cm.Set2(np.linspace(0, 1, len(regions)))
        for i, r in enumerate(regions):
            seg = r['segments'][0]
            mid = (seg['start_frame_original'] + seg['end_frame_original']) // 2
            ax1 = fig.add_subplot(len(regions), 3, i*3 + 1)
            if mid < len(rgb_files): ax1.imshow(Image.open(rgb_files[mid]))
            ax1.set_title(f'Region {r["region_id"]} (Frame {mid})', fontsize=12, fontweight='bold'); ax1.axis('off')
            ax2 = fig.add_subplot(len(regions), 3, i*3 + 2)
            tags = Counter([object_tags.get(o, '') for o in r.get('object_ids', []) if o in object_tags])
            if tags:
                tt = tags.most_common(8)
                ax2.pie([t[1] for t in tt], labels=[t[0][:18] for t in tt], autopct='%1.0f%%', colors=plt.cm.Set3(np.linspace(0, 1, len(tt))), textprops={'fontsize': 9})
            ax2.set_title(f'Region {r["region_id"]} - Objects', fontsize=12, fontweight='bold')
            ax3 = fig.add_subplot(len(regions), 3, i*3 + 3)
            ax3.axhline(y=0.5, color='gray', linewidth=2, alpha=0.5); ax3.set_xlim(0, 2000); ax3.set_ylim(0, 1)
            for s in r['segments']: ax3.axvspan(s['start_frame_original'], s['end_frame_original'], alpha=0.6, color=cols[i])
            ax3.set_title(f'Region {r["region_id"]} - Timeline', fontsize=12, fontweight='bold'); ax3.set_xlabel('Frame'); ax3.set_yticks([])
        plt.suptitle('REGION SEMANTICS', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout(); plt.savefig(output_dir / 'region_semantics.png', dpi=150, bbox_inches='tight'); plt.close()
        print("    Saved: region_semantics.png")
    
    # Full timeline
    valid = [o for o in range(len(objects)) if any(o in f for f in frame_objects)]
    if valid:
        fig, ax = plt.subplots(figsize=(20, max(12, len(valid) * 0.4)))
        ylbl = []; pc = plt.cm.tab20(np.linspace(0, 1, 20))
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
        for i, b in enumerate(boundaries):
            ax.axvline(x=b['frame_orig'], color='red', linestyle='--', linewidth=2, alpha=0.9)
            ax.annotate(f"Split {i+1}\n({b['frame_orig']})", xy=(b['frame_orig'], len(valid) + 0.5), fontsize=9, ha='center', va='bottom', color='red', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        ax.set_yticks(range(len(ylbl))); ax.set_yticklabels(ylbl, fontsize=8)
        ax.set_xlabel('Frame', fontsize=12, fontweight='bold'); ax.set_ylabel('Object ID', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 2000); ax.set_ylim(-0.5, len(valid) + 1.5); ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.set_title(f'OBJECT VISIBILITY TIMELINE (Total: {len(objects)} objects)', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout(); plt.savefig(output_dir / 'object_visibility_timeline_full.png', dpi=120, bbox_inches='tight'); plt.close()
        print("    Saved: object_visibility_timeline_full.png")
    print(f"  Done: {output_dir}")
        plt.savefig(output_dir / 'region_semantics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: region_semantics.png")
    
    # === Figure 3: Full Object Visibility Timeline ===
    valid_obj_ids = [oid for oid in range(len(objects)) if any(oid in fo for fo in frame_objects)]
    if valid_obj_ids:
        fig, ax = plt.subplots(figsize=(20, max(12, len(valid_obj_ids) * 0.4)))
        y_labels = []
        plot_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for y_pos, obj_id in enumerate(valid_obj_ids):
            tag = object_tags.get(obj_id, f'obj_{obj_id}')
            y_labels.append(f'ID={obj_id}: {tag[:35]}')
            
            visible_frames = [f * stride for f in range(n_frames) if obj_id in frame_objects[f]]
            if visible_frames:
                segs = []
                start = visible_frames[0]
                prev = start
                for f in visible_frames[1:]:
                    if f - prev > stride * 2:
                        segs.append((start, prev))
                        start = f
                    prev = f
                segs.append((start, prev))
                
                bar_color = plot_colors[obj_id % 20]
                for seg_start, seg_end in segs:
                    ax.barh(y_pos, seg_end - seg_start + stride, left=seg_start, height=0.8,
                            color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.3)
        
        # Draw boundaries
        for i, b in enumerate(boundaries):
            ax.axvline(x=b['frame_orig'], color='red', linestyle='--', linewidth=2, alpha=0.9)
            ax.annotate(f'Split {i+1}\n(Frame {b["frame_orig"]})',
                        xy=(b['frame_orig'], len(valid_obj_ids) + 0.5),
                        fontsize=9, ha='center', va='bottom', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Object (ID = unique identifier)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 2000)
        ax.set_ylim(-0.5, len(valid_obj_ids) + 1.5)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.set_title(f'COMPLETE OBJECT VISIBILITY TIMELINE\n'
                     f'Total objects: {len(objects)} | Red lines = split points',
                     fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'object_visibility_timeline_full.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    Saved: object_visibility_timeline_full.png")
    
    print(f"  All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
