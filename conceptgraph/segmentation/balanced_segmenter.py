#!/usr/bin/env python3
"""平衡版时序场景分段器"""

import gzip, pickle, json, argparse
import numpy as np
from pathlib import Path
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/home/shyue/Datasets/Replica/Replica')
    parser.add_argument('--scene', default='room0')
    parser.add_argument('--gsa_mode', default='ram_withbg_allclasses')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--min_frames', type=int, default=40)
    parser.add_argument('--target_regions', type=int, default=6)
    args = parser.parse_args()

    scene_path = Path(args.dataset_root) / args.scene
    
    # Load trajectory
    poses = []
    with open(scene_path / 'traj.txt') as f:
        for i, line in enumerate(f):
            if i % args.stride == 0:
                poses.append(np.array(list(map(float, line.split()))).reshape(4, 4))
    poses = np.array(poses)

    # Load GSA data
    gsa_dir = scene_path / f'gsa_detections_{args.gsa_mode}'
    if not gsa_dir.exists():
        gsa_dir = scene_path / 'gsa_detections_none'
    gsa_files = natsorted(gsa_dir.glob('frame*.pkl.gz'))
    n_frames = min(len(poses), len(gsa_files))
    poses, gsa_files = poses[:n_frames], gsa_files[:n_frames]

    frame_feats, all_classes = [], []
    for gf in gsa_files:
        with gzip.open(gf, 'rb') as f:
            d = pickle.load(f)
        feats = d.get('image_feats', np.zeros((1, 1024)))
        if len(feats) > 0:
            conf = d.get('confidence', np.ones(len(feats)))
            frame_feats.append(np.average(feats, axis=0, weights=conf/(conf.sum()+1e-8)))
        else:
            frame_feats.append(np.zeros(1024))
        classes = d.get('classes', [])
        all_classes.append(set(classes) if isinstance(classes, list) else set())

    frame_feats = np.array(frame_feats)
    norms = np.linalg.norm(frame_feats, axis=1, keepdims=True)
    feats_norm = frame_feats / (norms + 1e-8)

    # Motion signal
    positions = poses[:, :3, 3]
    pos_diff = np.concatenate([[0], np.linalg.norm(np.diff(positions, axis=0), axis=1)])
    motion = gaussian_filter1d(pos_diff, 2.0)
    motion = motion / (motion.max() + 1e-8)

    # Visual signal (multi-scale)
    visual_sum = np.zeros(n_frames)
    for w in [10, 20, 30]:
        for i in range(n_frames):
            sim = np.dot(feats_norm[max(0, i-w)], feats_norm[i])
            visual_sum[i] += 1 - sim
    visual = gaussian_filter1d(visual_sum / 3, 2.0)
    visual = visual / (visual.max() + 1e-8)

    # Semantic signal
    class_change = [0.0]
    for i in range(1, n_frames):
        prev, curr = all_classes[i-1], all_classes[i]
        union = len(prev | curr)
        class_change.append(1 - len(prev & curr) / union if union > 0 else 0)
    semantic = gaussian_filter1d(np.array(class_change), 2.0)
    semantic = semantic / (semantic.max() + 1e-8)

    # Fuse signals
    fused = 0.25 * motion + 0.50 * visual + 0.25 * semantic
    gradient = gaussian_filter1d(np.abs(np.gradient(fused)), 1.5)

    # Find peaks
    for prom in np.arange(0.01, 0.10, 0.005):
        peaks, _ = find_peaks(gradient, prominence=prom, distance=20)
        if len(peaks) <= args.target_regions:
            break

    # Create segments
    boundaries = [0] + sorted(peaks.tolist()) + [n_frames]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        region_classes = set()
        for j in range(start, end):
            if j < len(all_classes):
                region_classes.update(all_classes[j])
        segments.append({
            'region_id': i, 'start_frame': start, 'end_frame': end,
            'n_frames': end - start, 'dominant_classes': list(region_classes)[:8]
        })

    # Merge short segments
    merged = []
    for seg in segments:
        if seg['n_frames'] < args.min_frames and len(merged) > 0:
            merged[-1]['end_frame'] = seg['end_frame']
            merged[-1]['n_frames'] = merged[-1]['end_frame'] - merged[-1]['start_frame']
        else:
            merged.append(seg)

    # Merge similar segments
    def get_feat(s, e):
        return feats_norm[s:e].mean(axis=0)

    final = [merged[0]]
    for seg in merged[1:]:
        sim = np.dot(get_feat(final[-1]['start_frame'], final[-1]['end_frame']),
                     get_feat(seg['start_frame'], seg['end_frame']))
        if sim > 0.92:
            final[-1]['end_frame'] = seg['end_frame']
            final[-1]['n_frames'] = final[-1]['end_frame'] - final[-1]['start_frame']
        else:
            final.append(seg)

    for i, seg in enumerate(final):
        seg['region_id'] = i

    # Save
    output_dir = scene_path / 'sg_cache' / 'segmentation_balanced'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'trajectory_segments.json', 'w') as f:
        json.dump(final, f, indent=2)

    print(f"分段结果: {len(final)} 个区域")
    for seg in final:
        print(f"  Region {seg['region_id']}: frames {seg['start_frame']}-{seg['end_frame']} ({seg['n_frames']} frames)")
    print(f"\n保存到: {output_dir}")

if __name__ == '__main__':
    main()
