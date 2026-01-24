import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from encoding.models.sseg import BaseNet


def get_args():
    parser = argparse.ArgumentParser(
        description="Multi-view LSeg feature fusion on Replica scenes."
    )
    parser.add_argument("--scene_path", type=str, required=True,
                        help="Path to Replica scene (e.g., /path/to/Replica/room0)")
    parser.add_argument("--output_path", type=str, default="",
                        help="Output .npz path (default: <scene_path>/dense_pcd_lseg.npz)")
    parser.add_argument("--lseg_model", type=str, default="checkpoints/demo_e200.ckpt",
                        help="Path to LSeg checkpoint")
    parser.add_argument("--stride", type=int, default=8,
                        help="Frame stride for processing")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Image downsample factor (integer stride)")
    parser.add_argument("--voxel_size", type=float, default=0.025,
                        help="Voxel size for multi-view fusion (meters)")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Max depth in meters")
    parser.add_argument("--depth_scale", type=float, default=6553.5,
                        help="Depth scale for Replica depth PNGs")
    parser.add_argument("--lseg_img_long_side", type=int, default=640,
                        help="LSeg input long side for multi-scale eval")
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=599.5)
    parser.add_argument("--cy", type=float, default=339.5)
    args = parser.parse_args()
    return args


def load_poses(traj_path: Path) -> np.ndarray:
    data = np.loadtxt(traj_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 16:
        poses = data.reshape(-1, 4, 4)
    elif data.shape[1] == 4:
        poses = data.reshape(-1, 4, 4)
    else:
        raise ValueError(f"Unexpected traj.txt shape: {data.shape}")
    return poses.astype(np.float32)


def get_frame_paths(scene_path: Path) -> Tuple[list, list]:
    results_dir = scene_path / "results"
    rgb_paths = sorted(results_dir.glob("frame*.jpg"))
    if not rgb_paths:
        rgb_paths = sorted(results_dir.glob("*.jpg"))
    depth_paths = sorted(results_dir.glob("depth*.png"))
    if not rgb_paths or not depth_paths:
        raise FileNotFoundError(f"No RGB/Depth found in {results_dir}")
    return rgb_paths, depth_paths


def load_lseg_model(model_path: str, img_long_side: int):
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=model_path,
        data_path="../datasets/",
        dataset="ade20k",
        backbone="clip_vitl16_384",
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
        activation="lrelu",
    )

    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval().cpu()
    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    scales = ([1])
    model.crop_size = img_long_side
    model.base_size = img_long_side

    evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
    evaluator.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return evaluator, transform


def extract_lseg_img_feature_from_array(img: np.ndarray, transform, evaluator) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    image = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = evaluator.parallel_forward(image, "")
        feat_2d = outputs[0][0].float()
    return feat_2d.cpu().numpy()


def backproject_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float,
    feat_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    valid = (depth > 0) & (depth < max_depth)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, feat_map.shape[0]), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    rows, cols = np.where(valid)
    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    ones = np.ones_like(z)
    points_cam = np.stack([x, y, z, ones], axis=1)
    points_world = (pose @ points_cam.T).T[:, :3]

    feat_h, feat_w = feat_map.shape[1], feat_map.shape[2]
    if feat_h != h or feat_w != w:
        row_coords = np.clip((rows * (feat_h / h)).astype(int), 0, feat_h - 1)
        col_coords = np.clip((cols * (feat_w / w)).astype(int), 0, feat_w - 1)
    else:
        row_coords = rows
        col_coords = cols

    feat_map_chw = feat_map
    features = feat_map_chw[:, row_coords, col_coords].T
    colors = rgb[rows, cols].astype(np.float32)
    weights = 1.0 / (1.0 + z / 5.0)

    return points_world.astype(np.float32), features.astype(np.float32), colors, weights.astype(np.float32)


def update_voxel_accumulators(
    voxel_size: float,
    points: np.ndarray,
    features: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_index: Dict[int, int],
    feat_sum: list,
    color_sum: list,
    coord_sum: list,
    weight_sum: list,
):
    if points.shape[0] == 0:
        return

    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_keys = (
        voxel_indices[:, 0].astype(np.int64) * 1000000000
        + voxel_indices[:, 1].astype(np.int64) * 1000000
        + voxel_indices[:, 2].astype(np.int64)
    )

    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    n_voxels = len(unique_keys)
    d = features.shape[1]

    feat_buf = np.zeros((n_voxels, d), dtype=np.float64)
    color_buf = np.zeros((n_voxels, 3), dtype=np.float64)
    coord_buf = np.zeros((n_voxels, 3), dtype=np.float64)
    weight_buf = np.zeros(n_voxels, dtype=np.float64)

    np.add.at(feat_buf, inverse, features * weights[:, None])
    np.add.at(color_buf, inverse, colors * weights[:, None])
    np.add.at(coord_buf, inverse, points * weights[:, None])
    np.add.at(weight_buf, inverse, weights)

    for i, key in enumerate(unique_keys):
        if key in voxel_index:
            idx = voxel_index[key]
            feat_sum[idx] += feat_buf[i]
            color_sum[idx] += color_buf[i]
            coord_sum[idx] += coord_buf[i]
            weight_sum[idx] += weight_buf[i]
        else:
            voxel_index[key] = len(feat_sum)
            feat_sum.append(feat_buf[i])
            color_sum.append(color_buf[i])
            coord_sum.append(coord_buf[i])
            weight_sum.append(weight_buf[i])


def main(args):
    scene_path = Path(args.scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene path not found: {scene_path}")

    output_path = args.output_path
    if not output_path:
        output_path = str(scene_path / "dense_pcd_lseg.npz")

    rgb_paths, depth_paths = get_frame_paths(scene_path)
    poses = load_poses(scene_path / "traj.txt")

    n_frames = min(len(rgb_paths), len(depth_paths), len(poses))
    rgb_paths = rgb_paths[:n_frames]
    depth_paths = depth_paths[:n_frames]
    poses = poses[:n_frames]

    evaluator, transform = load_lseg_model(args.lseg_model, args.lseg_img_long_side)

    voxel_index: Dict[int, int] = {}
    feat_sum = []
    color_sum = []
    coord_sum = []
    weight_sum = []

    for idx in tqdm(range(0, n_frames, args.stride), desc="Fusing frames"):
        rgb = imageio.imread(rgb_paths[idx])
        depth = imageio.imread(depth_paths[idx]).astype(np.float32) / args.depth_scale

        if args.downsample > 1:
            rgb = rgb[:: args.downsample, :: args.downsample]
            depth = depth[:: args.downsample, :: args.downsample]
            fx = args.fx / args.downsample
            fy = args.fy / args.downsample
            cx = args.cx / args.downsample
            cy = args.cy / args.downsample
        else:
            fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

        feat_map = extract_lseg_img_feature_from_array(rgb, transform, evaluator)
        points, features, colors, weights = backproject_points(
            depth=depth,
            rgb=rgb,
            pose=poses[idx],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_depth=args.max_depth,
            feat_map=feat_map,
        )

        update_voxel_accumulators(
            args.voxel_size,
            points,
            features,
            colors,
            weights,
            voxel_index,
            feat_sum,
            color_sum,
            coord_sum,
            weight_sum,
        )

    if not feat_sum:
        raise RuntimeError("No valid points fused. Check depth scale, max depth, and inputs.")

    feat_sum_np = np.stack(feat_sum, axis=0)
    color_sum_np = np.stack(color_sum, axis=0)
    coord_sum_np = np.stack(coord_sum, axis=0)
    weight_sum_np = np.stack(weight_sum, axis=0)

    valid = weight_sum_np > 0
    points = coord_sum_np[valid] / weight_sum_np[valid, None]
    features = feat_sum_np[valid] / weight_sum_np[valid, None]
    colors = color_sum_np[valid] / weight_sum_np[valid, None]
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    np.savez(
        output_path,
        points=points.astype(np.float32),
        lseg_features=features.astype(np.float16),
        colors=colors,
        voxel_size=np.float32(args.voxel_size),
        stride=np.int32(args.stride),
        downsample=np.int32(args.downsample),
        depth_scale=np.float32(args.depth_scale),
    )

    print(f"Saved {points.shape[0]:,} points to {output_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
