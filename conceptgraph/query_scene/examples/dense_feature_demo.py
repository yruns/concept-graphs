#!/usr/bin/env python3
"""
Demo: Dense per-pixel feature extraction (OpenScene-style).

Uses CLIP with bilinear interpolation to get pixel-level features,
then back-projects to 3D point cloud.

Usage:
    python dense_feature_demo.py --scene /path/to/scene --query "lamp"
"""

import argparse
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Dense feature extraction demo")
    parser.add_argument("--scene", type=str, 
                       default="/home/shyue/Datasets/Replica/Replica/room0",
                       help="Path to scene directory")
    parser.add_argument("--query", type=str, default="lamp",
                       help="Text query for search")
    parser.add_argument("--stride", type=int, default=5,
                       help="Process every N-th frame")
    parser.add_argument("--downsample", type=int, default=4,
                       help="Image downsample factor (4 = 1/4 resolution)")
    parser.add_argument("--voxel", type=float, default=0.02,
                       help="Voxel size for fusion (meters)")
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene)
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Settings: stride={args.stride}, downsample={args.downsample}, voxel={args.voxel}")
    
    # Import here to avoid import errors during arg parsing
    from conceptgraph.query_scene import (
        DensePointFeatureExtractor, LSegConfig, PointFeatureIndex
    )
    import cv2
    
    # =========================================
    # Step 1: Setup extractor
    # =========================================
    config = LSegConfig(feature_dim=512)
    extractor = DensePointFeatureExtractor(config=config, device="cuda")
    extractor.voxel_size = args.voxel
    
    # =========================================
    # Step 2: Find images and poses
    # =========================================
    results_dir = scene_path / 'results'
    rgb_paths = sorted(results_dir.glob('frame*.jpg'))
    if not rgb_paths:
        rgb_paths = sorted(results_dir.glob('*.jpg'))
    
    depth_paths = sorted(results_dir.glob('depth*.png'))
    
    logger.info(f"Found {len(rgb_paths)} RGB, {len(depth_paths)} depth images")
    
    # Load poses
    traj_file = scene_path / 'traj.txt'
    poses = []
    if traj_file.exists():
        with open(traj_file) as f:
            lines = f.readlines()
        
        # Check format: each line could be a full 4x4 matrix (16 numbers)
        # or traditional format (4 lines per matrix)
        first_line_nums = len(lines[0].split())
        
        if first_line_nums == 16:
            # Each line is a full 4x4 matrix
            for line in lines:
                nums = [float(x) for x in line.split()]
                if len(nums) == 16:
                    pose = np.array(nums).reshape(4, 4)
                    poses.append(pose)
        else:
            # Traditional format: 4 lines per matrix
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    pose = np.array([
                        [float(x) for x in lines[i].split()],
                        [float(x) for x in lines[i+1].split()],
                        [float(x) for x in lines[i+2].split()],
                        [float(x) for x in lines[i+3].split()],
                    ])
                    poses.append(pose)
    
    logger.info(f"Loaded {len(poses)} camera poses")
    
    # Default intrinsics (Replica)
    sample_img = cv2.imread(str(rgb_paths[0]))
    H, W = sample_img.shape[:2]
    intrinsics = np.array([
        [600.0, 0, W / 2],
        [0, 600.0, H / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    
    # =========================================
    # Step 3: Extract dense features
    # =========================================
    logger.info("Extracting dense per-pixel features...")
    
    points, features = extractor.extract_scene_features(
        rgb_paths, depth_paths, intrinsics, poses,
        stride=args.stride, downsample=args.downsample
    )
    
    logger.info(f"Extracted {len(points)} points with {features.shape[1]}D features")
    
    if len(points) == 0:
        logger.error("No points extracted!")
        return
    
    # =========================================
    # Step 4: Build index and search
    # =========================================
    logger.info("Building feature index...")
    
    feature_index = PointFeatureIndex(feature_dim=features.shape[1])
    feature_index.build(points, features)
    
    logger.info(f"Searching for: '{args.query}'")
    
    # Text encoding
    text_feat = extractor.lseg.encode_text(args.query)
    logger.debug(f"Text feature dim: {len(text_feat)}")
    
    def encode_text_fn(text):
        return extractor.lseg.encode_text(text)
    
    results = feature_index.search_by_text(args.query, encode_text_fn, top_k=20)
    
    logger.info(f"Found {len(results)} matching points:")
    for i, (idx, score) in enumerate(results[:10]):
        pt = points[idx]
        logger.info(f"  {i+1}. Point {idx}: score={score:.3f}, pos=({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    
    # =========================================
    # Step 5: Save features
    # =========================================
    output_path = scene_path / "dense_features.npz"
    np.savez(output_path, points=points, features=features)
    logger.success(f"Saved features to {output_path}")
    
    # Print statistics
    logger.info("=" * 50)
    logger.info("Statistics:")
    logger.info(f"  Total points: {len(points)}")
    logger.info(f"  Feature dim: {features.shape[1]}")
    logger.info(f"  Spatial range:")
    logger.info(f"    X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    logger.info(f"    Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    logger.info(f"    Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
