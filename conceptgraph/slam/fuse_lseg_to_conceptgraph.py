#!/usr/bin/env python3
"""
Fuse LSeg Features to ConceptGraphs Point Cloud

This script loads a ConceptGraphs pkl.gz file, extracts 3D point coordinates,
fuses LSeg features from multi-view images, and saves the result.

Must be run in the 'lseg' conda environment.

Usage:
    python fuse_lseg_to_conceptgraph.py \
        --scene_path /path/to/Replica/room0 \
        --pcd_file pcd_saves/full_pcd_xxx.pkl.gz \
        --output_file pcd_saves/full_pcd_xxx_with_lseg.pkl.gz \
        --stride 5
"""

import os
import sys
import argparse
import gzip
import pickle
import json
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import imageio

# Add lseg_feature_extraction to path for imports
LSEG_PROJECT_DIR = "/home/shyue/codebase/lseg_feature_extraction"
sys.path.insert(0, LSEG_PROJECT_DIR)

from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from fusion_util import extract_lseg_img_feature, PointCloudToImageMapper, make_intrinsic


def get_args():
    parser = argparse.ArgumentParser(description="Fuse LSeg Features to ConceptGraphs Point Cloud")
    parser.add_argument('--scene_path', type=str, required=True,
                        help='Path to scene directory (e.g., /path/to/Replica/room0)')
    parser.add_argument('--pcd_file', type=str, required=True,
                        help='Path to ConceptGraphs pkl.gz file (relative to scene_path or absolute)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: add _with_lseg suffix)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Frame stride, should match ConceptGraphs stride')
    parser.add_argument('--lseg_model', type=str, 
                        default=os.path.join(LSEG_PROJECT_DIR, 'checkpoints/demo_e200.ckpt'),
                        help='Path to LSeg checkpoint')
    
    # Replica camera parameters (can be overridden)
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--fy', type=float, default=600.0)
    parser.add_argument('--cx', type=float, default=599.5)
    parser.add_argument('--cy', type=float, default=339.5)
    parser.add_argument('--image_width', type=int, default=1200)
    parser.add_argument('--image_height', type=int, default=680)
    parser.add_argument('--depth_scale', type=float, default=6553.5)
    parser.add_argument('--visibility_threshold', type=float, default=0.25)
    
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


def load_conceptgraph_pcd(pcd_path):
    """Load ConceptGraphs pkl.gz file and extract point coordinates."""
    print(f"Loading ConceptGraphs pcd from: {pcd_path}")
    
    with gzip.open(pcd_path, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    print(f"Number of objects: {len(objects)}")
    
    # Extract all point coordinates
    all_points = []
    object_point_ranges = []  # (start_idx, end_idx) for each object
    
    current_idx = 0
    for obj_idx, obj in enumerate(objects):
        if 'pcd_np' in obj:
            points = obj['pcd_np']
        elif 'pcd' in obj:
            points = np.asarray(obj['pcd'].points)
        else:
            points = np.array([]).reshape(0, 3)
        
        n_points = len(points)
        if n_points > 0:
            all_points.append(points)
            object_point_ranges.append((current_idx, current_idx + n_points))
            current_idx += n_points
        else:
            object_point_ranges.append((current_idx, current_idx))
    
    if all_points:
        all_points = np.vstack(all_points).astype(np.float32)
    else:
        all_points = np.array([]).reshape(0, 3).astype(np.float32)
    
    print(f"Total points: {len(all_points):,}")
    
    return data, objects, all_points, object_point_ranges


def load_replica_poses(scene_path, stride):
    """Load camera poses from Replica traj.txt file."""
    traj_path = Path(scene_path) / "traj.txt"
    print(f"Loading poses from: {traj_path}")
    
    all_poses = np.loadtxt(traj_path)
    # Each row is a flattened 4x4 camera_to_world matrix
    all_poses = all_poses.reshape(-1, 4, 4)
    
    # Apply stride
    selected_poses = all_poses[::stride]
    
    print(f"Total poses: {len(all_poses)}, selected with stride {stride}: {len(selected_poses)}")
    
    return selected_poses


def get_replica_frames(scene_path, stride):
    """Get list of RGB and depth frames from Replica scene."""
    results_dir = Path(scene_path) / "results"
    
    # Find all frame images
    frame_pattern = str(results_dir / "frame*.jpg")
    all_rgb_frames = sorted(glob(frame_pattern))
    
    if not all_rgb_frames:
        raise ValueError(f"No RGB frames found in {results_dir}")
    
    # Apply stride
    selected_rgb = all_rgb_frames[::stride]
    
    # Generate corresponding depth paths
    selected_depth = []
    for rgb_path in selected_rgb:
        frame_name = Path(rgb_path).stem  # e.g., "frame000000"
        depth_path = results_dir / f"depth{frame_name[5:]}.png"
        selected_depth.append(str(depth_path))
    
    print(f"Selected {len(selected_rgb)} frames with stride {stride}")
    
    return selected_rgb, selected_depth


def fuse_lseg_features(all_points, rgb_frames, depth_frames, poses, 
                       evaluator, transform, point2img_mapper, args):
    """Fuse LSeg features from multi-view images to 3D points."""
    
    n_points = len(all_points)
    feat_dim = 512
    device = torch.device('cpu')
    
    # Initialize accumulators
    counter = torch.zeros((n_points, 1), device=device)
    sum_features = torch.zeros((n_points, feat_dim), device=device)
    
    print(f"\nFusing LSeg features from {len(rgb_frames)} frames...")
    
    for img_idx, (rgb_path, depth_path, pose) in enumerate(
            tqdm(zip(rgb_frames, depth_frames, poses), total=len(rgb_frames), desc="Processing frames")):
        
        # Load depth
        depth = imageio.imread(depth_path).astype(np.float32) / args.depth_scale
        
        # Compute 3D to 2D mapping
        mapping = point2img_mapper.compute_mapping(pose, all_points, depth)
        # mapping: (N, 3) - (row, col, valid_mask)
        
        valid_mask = mapping[:, 2] == 1
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            continue
        
        # Extract LSeg features
        feat_2d = extract_lseg_img_feature(rgb_path, transform, evaluator)
        # feat_2d: (512, H', W') - may be different resolution
        
        feat_h, feat_w = feat_2d.shape[1], feat_2d.shape[2]
        img_h, img_w = depth.shape
        
        # Scale mapping coordinates if feature map size differs from image size
        row_coords = mapping[:, 0].astype(np.float32)
        col_coords = mapping[:, 1].astype(np.float32)
        
        if feat_h != img_h or feat_w != img_w:
            row_coords = row_coords * (feat_h / img_h)
            col_coords = col_coords * (feat_w / img_w)
        
        row_coords = np.clip(row_coords.astype(int), 0, feat_h - 1)
        col_coords = np.clip(col_coords.astype(int), 0, feat_w - 1)
        
        # Get features for valid points
        feat_2d_cpu = feat_2d.cpu()
        point_features = feat_2d_cpu[:, row_coords[valid_mask], col_coords[valid_mask]]
        point_features = point_features.permute(1, 0)  # (n_valid, 512)
        
        # Accumulate
        valid_indices = torch.from_numpy(np.where(valid_mask)[0])
        counter[valid_indices] += 1
        sum_features[valid_indices] += point_features
    
    # Average
    counter[counter == 0] = 1e-5  # Avoid division by zero
    fused_features = sum_features / counter
    
    # Statistics
    n_observed = (counter > 0.5).sum().item()
    print(f"\nPoints observed from at least one view: {n_observed:,} / {n_points:,} ({100*n_observed/n_points:.1f}%)")
    
    return fused_features.half().numpy()


def save_result(data, objects, fused_features, object_point_ranges, output_path):
    """Save the result with LSeg features added to each object."""
    
    print(f"\nAdding LSeg features to objects...")
    
    for obj_idx, obj in enumerate(objects):
        start_idx, end_idx = object_point_ranges[obj_idx]
        if end_idx > start_idx:
            obj['point_lseg_feats'] = fused_features[start_idx:end_idx]
        else:
            obj['point_lseg_feats'] = np.array([]).reshape(0, 512).astype(np.float16)
    
    # Update the data dict
    data['objects'] = objects
    data['lseg_fusion_info'] = {
        'feature_dim': 512,
        'dtype': 'float16',
        'fusion_method': 'multi_view_average',
    }
    
    print(f"Saving to: {output_path}")
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # File size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output file size: {file_size:.1f} MB")


def main(args):
    torch.manual_seed(1457)
    
    scene_path = Path(args.scene_path)
    
    # Resolve pcd file path
    if os.path.isabs(args.pcd_file):
        pcd_path = Path(args.pcd_file)
    else:
        pcd_path = scene_path / args.pcd_file
    
    if not pcd_path.exists():
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    # Set output path
    if args.output_file is None:
        output_path = pcd_path.parent / (pcd_path.stem.replace('.pkl', '') + '_with_lseg.pkl.gz')
    else:
        if os.path.isabs(args.output_file):
            output_path = Path(args.output_file)
        else:
            output_path = scene_path / args.output_file
    
    # Load ConceptGraphs data
    data, objects, all_points, object_point_ranges = load_conceptgraph_pcd(pcd_path)
    
    if len(all_points) == 0:
        print("No points found in the pcd file!")
        return
    
    # Load camera poses
    poses = load_replica_poses(scene_path, args.stride)
    
    # Get frame paths
    rgb_frames, depth_frames = get_replica_frames(scene_path, args.stride)
    
    # Verify frame count matches pose count
    if len(rgb_frames) != len(poses):
        print(f"Warning: Frame count ({len(rgb_frames)}) != Pose count ({len(poses)})")
        min_count = min(len(rgb_frames), len(poses))
        rgb_frames = rgb_frames[:min_count]
        depth_frames = depth_frames[:min_count]
        poses = poses[:min_count]
    
    # Load LSeg model
    evaluator, transform = load_lseg_model(args.lseg_model)
    
    # Setup point to image mapper
    intrinsic = make_intrinsic(fx=args.fx, fy=args.fy, mx=args.cx, my=args.cy)
    
    # Image dimension for mapping (original image size)
    img_dim = (args.image_width, args.image_height)
    
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim,
        intrinsics=intrinsic,
        visibility_threshold=args.visibility_threshold,
        cut_bound=10
    )
    
    # Fuse features
    fused_features = fuse_lseg_features(
        all_points, rgb_frames, depth_frames, poses,
        evaluator, transform, point2img_mapper, args
    )
    
    # Save result
    save_result(data, objects, fused_features, object_point_ranges, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    args = get_args()
    print("=" * 70)
    print("Fuse LSeg Features to ConceptGraphs Point Cloud")
    print("=" * 70)
    print(f"Scene path: {args.scene_path}")
    print(f"PCD file: {args.pcd_file}")
    print(f"Stride: {args.stride}")
    print(f"LSeg model: {args.lseg_model}")
    print(f"Image size: {args.image_width} x {args.image_height}")
    print(f"Camera: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print("=" * 70)
    
    main(args)
