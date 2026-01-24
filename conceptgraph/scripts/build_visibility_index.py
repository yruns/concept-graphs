"""
Offline Visibility Index Builder

Builds a bidirectional object-view visibility index for a scene.
This is a preprocessing step that should be run once per scene.

Usage:
    python -m conceptgraph.scripts.build_visibility_index \
        --scene_path /path/to/scene \
        --stride 5 \
        --use_depth  # Optional: use depth maps for occlusion detection

Output:
    scene_path/indices/visibility_index.pkl
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def load_poses(traj_file: Path) -> List[np.ndarray]:
    """Load camera poses from trajectory file.
    
    Supports two formats:
    1. One 4x4 matrix per line (16 values, space-separated)
    2. One 4x4 matrix per 5 lines (frame_id, then 4 rows)
    """
    poses = []
    with open(traj_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    if not lines:
        return poses
    
    # Detect format by counting values in first line
    first_values = lines[0].split()
    
    if len(first_values) == 16:
        # Format 1: Each line is a flattened 4x4 matrix
        for line in lines:
            values = [float(x) for x in line.split()]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    else:
        # Format 2: 5 lines per pose (frame_id + 4 rows)
        for i in range(0, len(lines), 5):
            if i + 4 >= len(lines):
                break
            pose = np.zeros((4, 4))
            for j in range(4):
                pose[j] = [float(x) for x in lines[i + 1 + j].split()]
            poses.append(pose)
    
    return poses


def load_objects(pcd_file: Path) -> List[Dict[str, Any]]:
    """Load objects from pkl.gz file."""
    logger.info(f"Loading objects from {pcd_file}")
    
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    logger.info(f"Loaded {len(objects)} objects")
    
    return objects


def get_object_centroid(obj: Dict[str, Any]) -> np.ndarray:
    """Extract object centroid from raw object data."""
    # Try pcd_np first (numpy array of points)
    if 'pcd_np' in obj:
        points = obj['pcd_np']
        if isinstance(points, np.ndarray) and len(points) > 0:
            return points.mean(axis=0)
    
    # Try bbox_np (bounding box corners)
    if 'bbox_np' in obj:
        bbox = obj['bbox_np']
        if isinstance(bbox, np.ndarray) and len(bbox) > 0:
            return bbox.mean(axis=0)
    
    # Try Open3D pcd object
    if 'pcd' in obj:
        pcd = obj['pcd']
        if hasattr(pcd, 'get_center'):
            center = pcd.get_center()
            return np.array(center)
        elif hasattr(pcd, 'points'):
            points = np.asarray(pcd.points)
            if len(points) > 0:
                return points.mean(axis=0)
    
    # Try Open3D bbox object
    if 'bbox' in obj:
        bbox = obj['bbox']
        if hasattr(bbox, 'get_center'):
            return np.array(bbox.get_center())
    
    return np.zeros(3)


def compute_geometric_visibility(
    centroid: np.ndarray,
    pose: np.ndarray,
    max_distance: float = 5.0,
) -> Tuple[float, float]:
    """Compute geometric visibility score.
    
    Args:
        centroid: Object centroid [x, y, z]
        pose: Camera pose matrix 4x4
        max_distance: Maximum viewing distance
    
    Returns:
        (distance_score, angle_score)
    """
    # Camera position
    cam_pos = pose[:3, 3]
    
    # Distance
    distance = np.linalg.norm(centroid - cam_pos)
    if distance > max_distance:
        return 0.0, 0.0
    
    dist_score = max(0, 1 - distance / max_distance)
    
    # Viewing angle
    view_dir = centroid - cam_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    
    # Camera forward (-Z axis)
    cam_forward = -pose[:3, 2]
    angle_score = max(0, np.dot(view_dir, cam_forward))
    
    return dist_score, angle_score


def check_depth_visibility(
    centroid: np.ndarray,
    pose: np.ndarray,
    depth_path: Path,
    intrinsics: np.ndarray,
    tolerance: float = 0.3,
) -> float:
    """Check if object is visible using depth map (not occluded).
    
    Args:
        centroid: Object centroid [x, y, z]
        pose: Camera pose matrix 4x4
        depth_path: Path to depth image
        intrinsics: Camera intrinsic matrix 3x3
        tolerance: Depth tolerance in meters
    
    Returns:
        Occlusion ratio (0 = fully visible, 1 = fully occluded)
    """
    if not depth_path.exists():
        return 0.0  # Assume visible if no depth
    
    try:
        # Load depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return 0.0
        
        # Convert to meters (assuming mm)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        
        # Transform centroid to camera frame
        pose_inv = np.linalg.inv(pose)
        centroid_h = np.append(centroid, 1.0)
        centroid_cam = pose_inv @ centroid_h
        
        if centroid_cam[2] <= 0:
            return 1.0  # Behind camera
        
        # Project to image
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        u = int(fx * centroid_cam[0] / centroid_cam[2] + cx)
        v = int(fy * centroid_cam[1] / centroid_cam[2] + cy)
        
        # Check bounds
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return 1.0  # Outside image
        
        # Check depth
        measured_depth = depth[v, u]
        expected_depth = centroid_cam[2]
        
        if measured_depth < 0.1:  # Invalid depth
            return 0.0
        
        # Object is occluded if measured depth is significantly less than expected
        if measured_depth < expected_depth - tolerance:
            return 1.0
        
        return 0.0
        
    except Exception as e:
        logger.warning(f"Depth check failed: {e}")
        return 0.0


def build_visibility_index(
    objects: List[Dict[str, Any]],
    poses: List[np.ndarray],
    depth_paths: Optional[List[Path]] = None,
    intrinsics: Optional[np.ndarray] = None,
    max_distance: float = 5.0,
    use_depth: bool = False,
    stride: int = 1,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    """Build bidirectional visibility index using detection ground truth.
    
    Uses the object's image_idx field (frames where object was actually detected)
    as ground truth, combined with geometric scoring for ranking.
    
    Args:
        objects: List of objects with image_idx and centroids
        poses: Camera poses
        depth_paths: Depth image paths (optional, unused)
        intrinsics: Camera intrinsics (optional, unused)
        max_distance: Maximum viewing distance for scoring
        use_depth: Unused (kept for API compatibility)
        stride: Frame stride (unused, image_idx is already in view coordinates)
    
    Returns:
        Tuple of:
        - object_to_views: object_id -> [(view_id, score), ...] sorted by score desc
        - view_to_objects: view_id -> [(object_id, score), ...] sorted by score desc
    """
    logger.info(f"Building visibility index for {len(objects)} objects")
    logger.info("Using detection ground truth (image_idx) for visibility")
    
    object_to_views: Dict[int, List[Tuple[int, float]]] = {}
    view_to_objects: Dict[int, List[Tuple[int, float]]] = {}
    
    # Image dimensions for completeness calculation (Replica default)
    img_width, img_height = 1200, 680
    img_area = img_width * img_height
    
    for obj_idx, obj in enumerate(tqdm(objects, desc="Building visibility index")):
        # Get frames where this object was actually detected
        image_idxs = obj.get('image_idx', [])
        if not image_idxs:
            continue
        
        # Get bbox and pixel area lists for completeness scoring
        xyxy_list = obj.get('xyxy', [])
        pixel_area_list = obj.get('pixel_area', [])
        
        # Get object centroid for geometric scoring
        centroid = get_object_centroid(obj)
        
        scores = []
        
        # Build a lookup: view_id -> list of indices in image_idxs
        view_to_indices = {}
        for i, vid in enumerate(image_idxs):
            if vid not in view_to_indices:
                view_to_indices[vid] = []
            view_to_indices[vid].append(i)
        
        for view_id, indices in view_to_indices.items():
            if view_id >= len(poses):
                continue
            
            # === 1. Completeness Score (most important, 50% weight) ===
            # Based on bbox size and whether it's clipped by image boundary
            best_completeness = 0.0
            for idx in indices:
                if idx < len(xyxy_list) and xyxy_list[idx] is not None:
                    xyxy = xyxy_list[idx]
                    if len(xyxy) == 4:
                        x1, y1, x2, y2 = xyxy
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        bbox_area = bbox_w * bbox_h
                        
                        # Size score: larger bbox is better (normalized by image area)
                        # Cap at 0.3 of image area to avoid oversized objects
                        size_score = min(1.0, bbox_area / (img_area * 0.3))
                        
                        # Completeness penalty: penalize if bbox touches image boundary
                        margin = 10  # pixels
                        is_clipped = (x1 < margin or y1 < margin or 
                                     x2 > img_width - margin or y2 > img_height - margin)
                        clip_penalty = 0.3 if is_clipped else 0.0
                        
                        completeness = max(0, size_score - clip_penalty)
                        best_completeness = max(best_completeness, completeness)
            
            # === 2. Geometric Score (30% weight) ===
            geo_score = 0.0
            if not np.allclose(centroid, 0):
                pose = poses[view_id]
                dist_score, angle_score = compute_geometric_visibility(
                    centroid, pose, max_distance
                )
                geo_score = 0.6 * dist_score + 0.4 * angle_score
            
            # === 3. Detection Quality Score (20% weight) ===
            # Based on pixel area if available, or detection count
            quality_score = 0.0
            for idx in indices:
                if idx < len(pixel_area_list) and pixel_area_list[idx]:
                    # Normalize pixel area (larger is better)
                    pa = pixel_area_list[idx]
                    quality_score = max(quality_score, min(1.0, pa / (img_area * 0.2)))
            if quality_score == 0:
                # Fallback: more detections = higher quality
                quality_score = min(1.0, len(indices) / 3.0)
            
            # === Combined Score ===
            # Completeness is most important (50%), then geometry (30%), then quality (20%)
            combined = 0.5 * best_completeness + 0.3 * geo_score + 0.2 * quality_score
            
            scores.append((view_id, float(combined)))
            
            # Add to reverse index
            if view_id not in view_to_objects:
                view_to_objects[view_id] = []
            view_to_objects[view_id].append((obj_idx, float(combined)))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        object_to_views[obj_idx] = scores
    
    # Sort view_to_objects by score descending
    for view_id in view_to_objects:
        view_to_objects[view_id].sort(key=lambda x: x[1], reverse=True)
    
    total_obj_mappings = sum(len(v) for v in object_to_views.values())
    total_view_mappings = sum(len(v) for v in view_to_objects.values())
    
    logger.success(
        f"Built bidirectional index: {len(object_to_views)} objects, "
        f"{len(view_to_objects)} views, {total_obj_mappings} mappings"
    )
    
    return object_to_views, view_to_objects


def save_visibility_index(
    object_to_views: Dict[int, List[Tuple[int, float]]],
    view_to_objects: Dict[int, List[Tuple[int, float]]],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save bidirectional visibility index to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'object_to_views': object_to_views,  # object_id -> [(view_id, score), ...]
        'view_to_objects': view_to_objects,  # view_id -> [(object_id, score), ...]
        'metadata': metadata or {},
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.success(f"Saved bidirectional visibility index to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build offline visibility index")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene")
    parser.add_argument("--pcd_file", type=str, default=None, help="Path to pkl.gz file")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride")
    parser.add_argument("--max_distance", type=float, default=5.0, help="Max viewing distance")
    parser.add_argument("--use_depth", action="store_true", help="Use depth for occlusion")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene_path)
    
    # Find PCD file
    if args.pcd_file:
        pcd_file = Path(args.pcd_file)
    else:
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            raise FileNotFoundError(f"No pkl.gz files found in {pcd_dir}")
        pcd_file = pcd_files[0]
    
    logger.info(f"Using PCD file: {pcd_file}")
    
    # Load objects
    objects = load_objects(pcd_file)
    
    # Load poses
    traj_file = scene_path / "traj.txt"
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    all_poses = load_poses(traj_file)
    poses = all_poses[::args.stride]
    logger.info(f"Loaded {len(poses)} poses (stride={args.stride})")
    
    # Depth paths
    depth_paths = None
    intrinsics = None
    
    if args.use_depth:
        depth_dir = scene_path / "results"
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob("depth*.png"))
            depth_paths = [depth_files[i] for i in range(0, len(depth_files), args.stride)
                          if i < len(depth_files)]
            logger.info(f"Found {len(depth_paths)} depth images")
            
            # Default intrinsics for Replica
            intrinsics = np.array([
                [600.0, 0, 599.5],
                [0, 600.0, 339.5],
                [0, 0, 1],
            ], dtype=np.float32)
    
    # Build bidirectional index
    start_time = time.time()
    
    object_to_views, view_to_objects = build_visibility_index(
        objects=objects,
        poses=poses,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        max_distance=args.max_distance,
        use_depth=args.use_depth,
        stride=args.stride,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Index built in {elapsed:.2f} seconds")
    
    # Save
    output_path = Path(args.output) if args.output else scene_path / "indices" / "visibility_index.pkl"
    
    metadata = {
        'scene_path': str(scene_path),
        'pcd_file': str(pcd_file),
        'stride': args.stride,
        'max_distance': args.max_distance,
        'use_depth': args.use_depth,
        'num_objects': len(objects),
        'num_views': len(poses),
        'num_object_mappings': sum(len(v) for v in object_to_views.values()),
        'num_view_mappings': sum(len(v) for v in view_to_objects.values()),
        'build_time': elapsed,
    }
    
    save_visibility_index(object_to_views, view_to_objects, output_path, metadata)


if __name__ == "__main__":
    main()
