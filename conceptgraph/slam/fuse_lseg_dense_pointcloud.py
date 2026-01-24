#!/usr/bin/env python3
"""
Generate Dense Scene Point Cloud with LSeg Features (Memory Efficient)

This script generates a dense point cloud from depth images and fuses 
LSeg features from multi-view images for the ENTIRE scene.

Uses incremental voxel fusion to avoid memory issues.

Must be run in the 'lseg' conda environment.

Usage:
    python fuse_lseg_dense_pointcloud.py \
        --scene_path /path/to/Replica/room0 \
        --output_file room0_dense_lseg.npz \
        --stride 5 \
        --voxel_size 0.02
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import imageio
import gc

# Add lseg_feature_extraction to path for imports
LSEG_PROJECT_DIR = "/home/shyue/codebase/lseg_feature_extraction"
sys.path.insert(0, LSEG_PROJECT_DIR)

from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from fusion_util import extract_lseg_img_feature


def get_args():
    parser = argparse.ArgumentParser(description="Generate Dense Scene Point Cloud with LSeg Features")
    parser.add_argument('--scene_path', type=str, required=True,
                        help='Path to scene directory (e.g., /path/to/Replica/room0)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: <scene_path>/dense_pcd_lseg.npz)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Frame stride for processing')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for downsampling (meters)')
    parser.add_argument('--lseg_model', type=str, 
                        default=os.path.join(LSEG_PROJECT_DIR, 'checkpoints/demo_e200.ckpt'),
                        help='Path to LSeg checkpoint')
    
    # Replica camera parameters
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--fy', type=float, default=600.0)
    parser.add_argument('--cx', type=float, default=599.5)
    parser.add_argument('--cy', type=float, default=339.5)
    parser.add_argument('--image_width', type=int, default=1200)
    parser.add_argument('--image_height', type=int, default=680)
    parser.add_argument('--depth_scale', type=float, default=6553.5)
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help='Maximum depth value (meters)')
    parser.add_argument('--lseg_img_size', type=int, default=480,
                        help='LSeg input image long side (smaller=faster, default 480)')
    
    args = parser.parse_args()
    return args


def load_lseg_model(model_path, img_long_side=640):
    """Load LSeg model and create evaluator."""
    print(f"Loading LSeg model from: {model_path}")
    
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=model_path,
        data_path='../datasets/',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )
    
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module
    
    model = model.eval()
    model = model.cpu()
    
    scales = [1]
    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    model.crop_size = 2 * img_long_side
    model.base_size = 2 * img_long_side
    
    evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
    evaluator.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    return evaluator, transform


def load_replica_poses(scene_path, stride):
    """Load camera poses from Replica traj.txt file."""
    traj_path = Path(scene_path) / "traj.txt"
    print(f"Loading poses from: {traj_path}")
    
    all_poses = np.loadtxt(traj_path)
    all_poses = all_poses.reshape(-1, 4, 4)
    selected_poses = all_poses[::stride]
    
    print(f"Total poses: {len(all_poses)}, selected with stride {stride}: {len(selected_poses)}")
    
    return selected_poses


def get_replica_frames(scene_path, stride):
    """Get list of RGB and depth frames from Replica scene."""
    results_dir = Path(scene_path) / "results"
    
    frame_pattern = str(results_dir / "frame*.jpg")
    all_rgb_frames = sorted(glob(frame_pattern))
    
    if not all_rgb_frames:
        raise ValueError(f"No RGB frames found in {results_dir}")
    
    selected_rgb = all_rgb_frames[::stride]
    
    selected_depth = []
    for rgb_path in selected_rgb:
        frame_name = Path(rgb_path).stem
        depth_path = results_dir / f"depth{frame_name[5:]}.png"
        selected_depth.append(str(depth_path))
    
    print(f"Selected {len(selected_rgb)} frames with stride {stride}")
    
    return selected_rgb, selected_depth


def depth_to_pointcloud(depth, pose, fx, fy, cx, cy, max_depth=10.0):
    """Convert depth image to 3D point cloud in world coordinates."""
    h, w = depth.shape
    
    # Create pixel grid
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Valid depth mask
    valid = (depth > 0) & (depth < max_depth)
    
    # Unproject to camera coordinates
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    # Stack to (N, 3)
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates
    points_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=-1)
    points_world = (pose @ points_homo.T).T[:, :3]
    
    # Get pixel coordinates for feature lookup
    pixel_coords = np.stack([v[valid], u[valid]], axis=-1)  # (N, 2) - row, col
    
    return points_world.astype(np.float32), pixel_coords.astype(np.int32)


class VoxelGrid:
    """Memory-efficient voxel grid for incremental fusion.
    
    Fusion Strategy (same as ConceptGraphs):
    =========================================
    1. For each voxel, accumulate:
       - sum of points (for position averaging)
       - sum of features (for feature averaging)
       - sum of colors (for color averaging)
       - observation count
    
    2. Final feature computation:
       feat_avg = sum_feat / count      # Weighted average by observation count
       feat_final = feat_avg / ||feat_avg||  # L2 normalization
       
    This matches ConceptGraphs' approach:
       clip_ft = (clip_ft_1 * n1 + clip_ft_2 * n2) / (n1 + n2)
       clip_ft = F.normalize(clip_ft, dim=0)
    """
    
    def __init__(self, voxel_size, feature_dim=512):
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        
        # Use dictionary for sparse storage
        self.voxels = {}  # key -> (sum_point, sum_feature, sum_color, count)
    
    def _point_to_key(self, point):
        """Convert point to voxel key."""
        voxel_idx = tuple(np.floor(point / self.voxel_size).astype(np.int32))
        return voxel_idx
    
    def add_points(self, points, features, colors):
        """Add points to the voxel grid with incremental averaging."""
        for i in range(len(points)):
            key = self._point_to_key(points[i])
            
            if key in self.voxels:
                sum_pt, sum_ft, sum_cl, count = self.voxels[key]
                sum_pt += points[i]
                sum_ft += features[i]
                sum_cl += colors[i].astype(np.float32)
                count += 1
                self.voxels[key] = (sum_pt, sum_ft, sum_cl, count)
            else:
                self.voxels[key] = (
                    points[i].copy(),
                    features[i].copy(),
                    colors[i].astype(np.float32),
                    1
                )
    
    def add_points_batch(self, points, features, colors):
        """Batch add points - optimized with numpy groupby."""
        # Compute voxel keys for all points
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)
        
        # Convert to hashable keys using a large prime multiplier
        # This is much faster than tuple conversion in a loop
        keys = (voxel_indices[:, 0] * 73856093 ^ 
                voxel_indices[:, 1] * 19349663 ^ 
                voxel_indices[:, 2] * 83492791)
        
        # Get unique keys and their inverse indices
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        
        # Pre-convert to float32
        features_f32 = features.astype(np.float32)
        colors_f32 = colors.astype(np.float32)
        
        # For each unique voxel, aggregate points
        for i, ukey in enumerate(unique_keys):
            mask = (inverse_indices == i)
            pts_in_voxel = points[mask]
            feats_in_voxel = features_f32[mask]
            colors_in_voxel = colors_f32[mask]
            count_new = mask.sum()
            
            # Sum within this batch
            sum_pt_new = pts_in_voxel.sum(axis=0)
            sum_ft_new = feats_in_voxel.sum(axis=0)
            sum_cl_new = colors_in_voxel.sum(axis=0)
            
            if ukey in self.voxels:
                sum_pt, sum_ft, sum_cl, count = self.voxels[ukey]
                sum_pt += sum_pt_new
                sum_ft += sum_ft_new
                sum_cl += sum_cl_new
                count += count_new
                self.voxels[ukey] = (sum_pt, sum_ft, sum_cl, count)
            else:
                self.voxels[ukey] = (sum_pt_new, sum_ft_new, sum_cl_new, count_new)
    
    def get_points_and_features(self):
        """Extract averaged points and features.
        
        Uses ConceptGraphs-style fusion:
        1. Weighted average by observation count
        2. L2 normalization of features
        """
        n_voxels = len(self.voxels)
        print(f"Total voxels: {n_voxels:,}")
        
        points = np.zeros((n_voxels, 3), dtype=np.float32)
        features = np.zeros((n_voxels, self.feature_dim), dtype=np.float32)
        colors = np.zeros((n_voxels, 3), dtype=np.uint8)
        counts = np.zeros(n_voxels, dtype=np.int32)
        
        for i, (key, (sum_pt, sum_ft, sum_cl, count)) in enumerate(self.voxels.items()):
            points[i] = sum_pt / count
            features[i] = sum_ft / count  # Weighted average
            colors[i] = np.clip(sum_cl / count, 0, 255).astype(np.uint8)
            counts[i] = count
        
        # L2 normalize features (ConceptGraphs style)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        features = features / norms
        
        print(f"Feature norm after L2 normalization: mean={np.linalg.norm(features, axis=1).mean():.4f}")
        print(f"Observation counts: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
        return points, features.astype(np.float16), colors


def process_scene_incremental(rgb_frames, depth_frames, poses, evaluator, transform, args):
    """Process all frames with incremental voxel fusion."""
    
    voxel_grid = VoxelGrid(args.voxel_size, feature_dim=512)
    
    print(f"\nProcessing {len(rgb_frames)} frames with incremental fusion...")
    print(f"Voxel size: {args.voxel_size}m")
    
    for idx, (rgb_path, depth_path, pose) in enumerate(
            tqdm(zip(rgb_frames, depth_frames, poses), total=len(rgb_frames), desc="Processing frames")):
        
        # Load depth
        depth = imageio.imread(depth_path).astype(np.float32) / args.depth_scale
        
        # Load RGB for colors
        rgb = imageio.imread(rgb_path)
        
        # Convert depth to point cloud
        points, pixel_coords = depth_to_pointcloud(
            depth, pose, args.fx, args.fy, args.cx, args.cy, args.max_depth
        )
        
        if len(points) == 0:
            continue
        
        # Extract LSeg features
        feat_2d = extract_lseg_img_feature(rgb_path, transform, evaluator)
        # feat_2d: (512, H', W')
        
        feat_h, feat_w = feat_2d.shape[1], feat_2d.shape[2]
        img_h, img_w = depth.shape
        
        # Scale pixel coordinates to feature map size
        row_coords = pixel_coords[:, 0].astype(np.float32) * (feat_h / img_h)
        col_coords = pixel_coords[:, 1].astype(np.float32) * (feat_w / img_w)
        
        row_coords = np.clip(row_coords.astype(int), 0, feat_h - 1)
        col_coords = np.clip(col_coords.astype(int), 0, feat_w - 1)
        
        # Get features for each point
        feat_2d_cpu = feat_2d.cpu().numpy()
        point_features = feat_2d_cpu[:, row_coords, col_coords].T  # (N, 512)
        
        # Get colors
        point_colors = rgb[pixel_coords[:, 0], pixel_coords[:, 1]]  # (N, 3)
        
        # Add to voxel grid (incremental fusion)
        voxel_grid.add_points_batch(points, point_features, point_colors)
        
        # Clear GPU cache periodically
        if idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print progress every 100 frames
        if (idx + 1) % 100 == 0:
            print(f"  Frame {idx+1}: {len(voxel_grid.voxels):,} voxels")
    
    # Extract final points and features
    print("\nExtracting final point cloud...")
    points, features, colors = voxel_grid.get_points_and_features()
    
    return points, features, colors


def main(args):
    torch.manual_seed(1457)
    
    scene_path = Path(args.scene_path)
    
    # Set output path
    if args.output_file is None:
        output_path = scene_path / "dense_pcd_lseg.npz"
    else:
        if os.path.isabs(args.output_file):
            output_path = Path(args.output_file)
        else:
            output_path = scene_path / args.output_file
    
    # Load camera poses
    poses = load_replica_poses(scene_path, args.stride)
    
    # Get frame paths
    rgb_frames, depth_frames = get_replica_frames(scene_path, args.stride)
    
    # Verify frame count
    if len(rgb_frames) != len(poses):
        min_count = min(len(rgb_frames), len(poses))
        rgb_frames = rgb_frames[:min_count]
        depth_frames = depth_frames[:min_count]
        poses = poses[:min_count]
    
    # Load LSeg model
    evaluator, transform = load_lseg_model(args.lseg_model, img_long_side=args.lseg_img_size)
    
    # Process scene with incremental fusion
    points, features, colors = process_scene_incremental(
        rgb_frames, depth_frames, poses, evaluator, transform, args
    )
    
    # Save
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(
        output_path,
        points=points,           # (N, 3) float32
        lseg_features=features,  # (N, 512) float16
        colors=colors,           # (N, 3) uint8
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output file size: {file_size:.1f} MB")
    print(f"Total points: {len(points):,}")
    print(f"Feature shape: {features.shape}")
    
    print("\nDone!")


if __name__ == "__main__":
    args = get_args()
    print("=" * 70)
    print("Generate Dense Scene Point Cloud with LSeg Features")
    print("(Memory Efficient - Incremental Voxel Fusion)")
    print("=" * 70)
    print(f"Scene path: {args.scene_path}")
    print(f"Stride: {args.stride}")
    print(f"Voxel size: {args.voxel_size}m")
    print(f"Max depth: {args.max_depth}m")
    print(f"LSeg model: {args.lseg_model}")
    print("=" * 70)
    
    main(args)
