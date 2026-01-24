#!/usr/bin/env python3
"""
Demo: OpenScene-style per-point CLIP feature extraction.

This script demonstrates how to:
1. Extract per-point CLIP features from multi-view RGB-D
2. Build a searchable point feature index
3. Query points by text

Usage:
    python openscene_demo.py --scene /path/to/scene --query "wooden chair"
"""

import argparse
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from conceptgraph.query_scene import (
    PointFeatureExtractor,
    PointFeatureIndex,
    PointFeatureConfig,
    PointLevelIndex,
    QuerySceneRepresentation,
)
from loguru import logger


def extract_and_search(
    scene_path: str,
    query: str = "chair",
    stride: int = 20,  # Process every 20th frame
    voxel_size: float = 0.03,  # 3cm voxels
    top_k: int = 20,
):
    """Extract OpenScene features and perform text search."""
    
    scene_path = Path(scene_path)
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Query: {query}")
    
    # =========================================
    # Step 1: Load scene for reference
    # =========================================
    logger.info("Loading scene representation...")
    
    pcd_files = list((scene_path / "pcd_saves").glob("*ram*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list((scene_path / "pcd_saves").glob("*.pkl.gz"))
    
    if pcd_files:
        scene = QuerySceneRepresentation.from_pcd_file(
            str(pcd_files[0]), scene_path, stride=5
        )
        logger.info(f"Loaded {len(scene.objects)} objects, {len(scene.camera_poses)} poses")
    else:
        scene = None
        logger.warning("No pcd file found, proceeding without object reference")
    
    # =========================================
    # Step 2: Extract per-point CLIP features
    # =========================================
    logger.info("Extracting OpenScene-style per-point features...")
    
    config = PointFeatureConfig(
        feature_dim=512,
        min_observations=1,  # Keep all points, even with single observation
        weight_by_angle=True,
        weight_by_distance=True,
    )
    
    extractor = PointFeatureExtractor(
        config=config,
        clip_model="ViT-B/32",  # Smaller model for demo
        device="cuda",
    )
    
    # Find images and depths
    results_dir = scene_path / "results"
    rgb_paths = sorted(results_dir.glob("frame*.jpg"))
    if not rgb_paths:
        rgb_paths = sorted(results_dir.glob("*.jpg"))
    
    depth_paths = sorted(results_dir.glob("depth*.png"))
    
    logger.info(f"Found {len(rgb_paths)} RGB, {len(depth_paths)} depth images")
    
    if not rgb_paths or not depth_paths:
        logger.error("No images found")
        return
    
    # Load camera poses from scene or trajectory file
    if scene and scene.camera_poses:
        extrinsics = []
        for pose in scene.camera_poses:
            if hasattr(pose, 'pose_matrix') and pose.pose_matrix is not None:
                extrinsics.append(pose.pose_matrix)
            else:
                # Construct from position/rotation if available
                T = np.eye(4)
                if pose.position is not None:
                    T[:3, 3] = pose.position
                extrinsics.append(T)
    else:
        # Load from traj.txt
        traj_file = scene_path / "traj.txt"
        extrinsics = []
        if traj_file.exists():
            with open(traj_file) as f:
                lines = f.readlines()
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    pose = np.array([
                        [float(x) for x in lines[i].split()],
                        [float(x) for x in lines[i+1].split()],
                        [float(x) for x in lines[i+2].split()],
                        [float(x) for x in lines[i+3].split()],
                    ])
                    extrinsics.append(pose)
    
    if not extrinsics:
        logger.error("No camera poses found")
        return
    
    # Default intrinsics
    import cv2
    sample_img = cv2.imread(str(rgb_paths[0]))
    H, W = sample_img.shape[:2]
    intrinsics = np.array([
        [600.0, 0, W / 2],
        [0, 600.0, H / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    
    # Extract features
    points, features = extractor.extract_scene_features(
        rgb_paths, depth_paths, intrinsics, extrinsics,
        stride=stride, voxel_size=voxel_size
    )
    
    feature_dim = features.shape[1] if len(features) > 0 else 512
    logger.info(f"Extracted {len(points)} points with {feature_dim}D features")
    
    if len(points) == 0:
        logger.error("No points extracted")
        return
    
    # =========================================
    # Step 3: Build feature index
    # =========================================
    logger.info("Building point feature index...")
    
    actual_feature_dim = features.shape[1]
    feature_index = PointFeatureIndex(feature_dim=actual_feature_dim)
    feature_index.build(points, features)
    logger.info(f"Index built with {actual_feature_dim}D features")
    
    # =========================================
    # Step 4: Search by text
    # =========================================
    logger.info(f"Searching for: '{query}'")
    
    # Create text encoder - use same model as image encoder
    _text_encoder_cache = {}
    
    def encode_text(text):
        import torch
        
        # Try OpenAI's official CLIP first (same as image encoder)
        try:
            import clip
            
            if 'model' not in _text_encoder_cache:
                _text_encoder_cache['model'], _ = clip.load("ViT-B/32", device="cpu")
                _text_encoder_cache['model'].eval()
            
            model = _text_encoder_cache['model']
            tokens = clip.tokenize([text])
            with torch.no_grad():
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                result = feat.cpu().numpy().flatten().astype(np.float32)
                logger.debug(f"Text feature dim: {len(result)}")
                return result
        except Exception as e:
            logger.warning(f"openai/clip text encoding failed: {e}")
        
        # Fallback to transformers
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            
            with torch.no_grad():
                feat = model.get_text_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None
    
    results = feature_index.search_by_text(query, encode_text, top_k=top_k)
    
    logger.info(f"Found {len(results)} matching points:")
    for i, (idx, score) in enumerate(results[:10]):
        point = points[idx]
        logger.info(f"  {i+1}. Point {idx}: score={score:.3f}, pos=({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    
    # =========================================
    # Step 5: Optionally save features
    # =========================================
    output_path = scene_path / "openscene_features.npz"
    np.savez(
        output_path,
        points=points,
        features=features,
    )
    logger.info(f"Saved features to {output_path}")
    
    return points, features, results


def main():
    parser = argparse.ArgumentParser(description="OpenScene feature extraction demo")
    parser.add_argument("--scene", type=str, 
                       default="/home/shyue/Datasets/Replica/Replica/room0",
                       help="Path to scene directory")
    parser.add_argument("--query", type=str, default="lamp",
                       help="Text query for search")
    parser.add_argument("--stride", type=int, default=20,
                       help="Process every N-th frame")
    parser.add_argument("--voxel", type=float, default=0.03,
                       help="Voxel size for fusion (meters)")
    
    args = parser.parse_args()
    
    extract_and_search(
        args.scene,
        query=args.query,
        stride=args.stride,
        voxel_size=args.voxel,
    )


if __name__ == "__main__":
    main()
