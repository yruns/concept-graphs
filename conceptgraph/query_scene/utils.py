"""Utility functions for QueryScene."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# =============================================================================
# 3D -> 2D Projection Functions
# =============================================================================

def project_point_to_image(
    point_3d: np.ndarray,
    camera_pose: Any,
    K: np.ndarray,
    image_size: Tuple[int, int] = (640, 480),
) -> Optional[Tuple[int, int]]:
    """Project a 3D point to 2D image coordinates.
    
    Args:
        point_3d: 3D point [x, y, z]
        camera_pose: Camera pose with position and rotation
        K: Camera intrinsic matrix (3x3)
        image_size: (width, height)
    
    Returns:
        (u, v) pixel coordinates or None if behind camera
    """
    point_3d = np.asarray(point_3d).flatten()
    
    # Transform to camera frame
    if hasattr(camera_pose, 'position'):
        cam_pos = camera_pose.position
        R = camera_pose.rotation
    else:
        cam_pos = np.array(camera_pose[:3])
        R = np.eye(3)
    
    point_cam = R.T @ (point_3d - cam_pos)
    
    # Check if point is in front of camera
    if point_cam[2] <= 0.01:
        return None
    
    # Project to image
    point_proj = K @ point_cam
    u = int(point_proj[0] / point_proj[2])
    v = int(point_proj[1] / point_proj[2])
    
    # Check bounds
    if 0 <= u < image_size[0] and 0 <= v < image_size[1]:
        return (u, v)
    return None


def project_3d_bbox_to_2d(
    bbox_3d: Any,
    camera_pose: Any,
    K: np.ndarray,
    image_size: Tuple[int, int] = (640, 480),
) -> Optional[Tuple[int, int, int, int]]:
    """Project 3D bounding box to 2D image bounding box.
    
    Args:
        bbox_3d: 3D bounding box with min_point and max_point
        camera_pose: Camera pose
        K: Camera intrinsic matrix
        image_size: (width, height)
    
    Returns:
        (x1, y1, x2, y2) 2D bounding box or None if not visible
    """
    if bbox_3d is None:
        return None
    
    # Get 8 corners of 3D box
    if hasattr(bbox_3d, 'min_point'):
        min_pt = bbox_3d.min_point
        max_pt = bbox_3d.max_point
    else:
        min_pt = np.array(bbox_3d['min'])
        max_pt = np.array(bbox_3d['max'])
    
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    
    # Project all corners
    projected = []
    for corner in corners:
        pt_2d = project_point_to_image(corner, camera_pose, K, image_size)
        if pt_2d is not None:
            projected.append(pt_2d)
    
    if len(projected) < 2:
        return None
    
    projected = np.array(projected)
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_size[0] - 1, x2)
    y2 = min(image_size[1] - 1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return (int(x1), int(y1), int(x2), int(y2))


def crop_object_from_image(
    image: np.ndarray,
    bbox_2d: Tuple[int, int, int, int],
    padding: float = 0.1,
    min_size: int = 32,
) -> Optional[np.ndarray]:
    """Crop object region from image with padding.
    
    Args:
        image: RGB image (H, W, 3)
        bbox_2d: (x1, y1, x2, y2) bounding box
        padding: Padding ratio around bbox
        min_size: Minimum crop size
    
    Returns:
        Cropped image region or None
    """
    if bbox_2d is None:
        return None
    
    x1, y1, x2, y2 = bbox_2d
    h, w = image.shape[:2]
    
    # Add padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    # Check minimum size
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    
    return image[y1:y2, x1:x2].copy()


# =============================================================================
# Image Annotation Functions (Visual Prompting)
# =============================================================================

def annotate_image_with_objects(
    image: np.ndarray,
    objects: List[Any],
    camera_pose: Any,
    K: np.ndarray,
    highlight_ids: Optional[List[int]] = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Annotate RGB image with object bounding boxes and IDs.
    
    This is crucial for VLM visual prompting - the VLM can reference
    objects by their labeled IDs.
    
    Args:
        image: RGB image (H, W, 3)
        objects: List of ObjectNode instances
        camera_pose: Camera pose for projection
        K: Camera intrinsic matrix
        highlight_ids: Object IDs to highlight (red), others are green
        show_labels: Whether to show category labels
    
    Returns:
        Annotated image with bounding boxes and ID labels
    """
    annotated = image.copy()
    h, w = image.shape[:2]
    
    highlight_ids = highlight_ids or []
    
    for obj in objects:
        if obj.bbox_3d is None:
            continue
        
        bbox_2d = project_3d_bbox_to_2d(obj.bbox_3d, camera_pose, K, (w, h))
        if bbox_2d is None:
            continue
        
        x1, y1, x2, y2 = bbox_2d
        
        # Determine color
        if obj.obj_id in highlight_ids:
            color = (255, 0, 0)  # Red for targets
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for others
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw ID label with background
        label = f"#{obj.obj_id}"
        if show_labels:
            label += f": {obj.category}"
        
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 4, y1),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    
    return annotated


def annotate_bev_with_distances(
    bev: np.ndarray,
    objects: List[Any],
    anchor_id: Optional[int],
    candidate_ids: List[int],
    scene_bounds: Tuple[np.ndarray, np.ndarray],
    distance_rings: List[float] = [1.0, 2.0],
) -> np.ndarray:
    """Annotate BEV with distance reference rings from anchor.
    
    Args:
        bev: Base BEV image
        objects: List of ObjectNode instances
        anchor_id: Anchor object ID (red)
        candidate_ids: Candidate object IDs (blue)
        scene_bounds: Scene bounding box
        distance_rings: Distance rings in meters
    
    Returns:
        Annotated BEV image
    """
    annotated = bev.copy()
    h, w = bev.shape[:2]
    
    min_pt, max_pt = scene_bounds
    scale = min(w, h) / max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1] + 1e-6) * 0.9
    offset_x = (w - (max_pt[0] - min_pt[0]) * scale) / 2
    offset_y = (h - (max_pt[1] - min_pt[1]) * scale) / 2
    
    def world_to_bev(pos):
        x = int((pos[0] - min_pt[0]) * scale + offset_x)
        y = int((pos[1] - min_pt[1]) * scale + offset_y)
        return (x, y)
    
    # Find anchor position
    anchor_pos = None
    anchor_obj = None
    for obj in objects:
        if obj.obj_id == anchor_id and obj.centroid is not None:
            anchor_pos = world_to_bev(obj.centroid)
            anchor_obj = obj
            break
    
    # Draw distance rings from anchor
    if anchor_pos is not None:
        for dist in distance_rings:
            radius = int(dist * scale)
            cv2.circle(annotated, anchor_pos, radius, (180, 180, 180), 1, cv2.LINE_AA)
            # Label the distance
            cv2.putText(
                annotated,
                f"{dist}m",
                (anchor_pos[0] + radius + 5, anchor_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 100, 100),
                1,
            )
    
    # Draw all objects
    for obj in objects:
        if obj.centroid is None:
            continue
        
        pos = world_to_bev(obj.centroid)
        
        if obj.obj_id == anchor_id:
            # Anchor: red filled circle
            cv2.circle(annotated, pos, 15, (255, 0, 0), -1)
            cv2.putText(
                annotated,
                f"REF: {obj.category}",
                (pos[0] + 18, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        elif obj.obj_id in candidate_ids:
            # Candidate: blue circle with ID
            cv2.circle(annotated, pos, 12, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"#{obj.obj_id}: {obj.category}",
                (pos[0] + 15, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        else:
            # Other: gray small circle
            cv2.circle(annotated, pos, 5, (150, 150, 150), -1)
    
    return annotated


# =============================================================================
# Original Functions
# =============================================================================

def load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as RGB array."""
    img = cv2.imread(str(path))
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def load_depth(path: Path, scale: float = 1000.0) -> Optional[np.ndarray]:
    """Load depth image and convert to meters."""
    if path.suffix == '.npy':
        return np.load(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is not None:
        return depth.astype(np.float32) / scale
    return None


def generate_bev(
    objects: List,
    scene_bounds: Tuple[np.ndarray, np.ndarray],
    resolution: float = 0.05,
    size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Generate bird's eye view image of the scene."""
    bev = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    min_pt, max_pt = scene_bounds
    scale_x = size[0] / (max_pt[0] - min_pt[0] + 1e-6)
    scale_y = size[1] / (max_pt[1] - min_pt[1] + 1e-6)
    scale = min(scale_x, scale_y) * 0.9
    
    offset_x = (size[0] - (max_pt[0] - min_pt[0]) * scale) / 2
    offset_y = (size[1] - (max_pt[1] - min_pt[1]) * scale) / 2
    
    for obj in objects:
        if obj.centroid is None:
            continue
        x = int((obj.centroid[0] - min_pt[0]) * scale + offset_x)
        y = int((obj.centroid[1] - min_pt[1]) * scale + offset_y)
        
        if 0 <= x < size[0] and 0 <= y < size[1]:
            color = (100, 100, 200)  # Default color
            cv2.circle(bev, (x, y), 8, color, -1)
            cv2.putText(bev, str(obj.obj_id), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return bev


def annotate_bev(
    bev: np.ndarray,
    objects: List,
    highlight_ids: List[int],
    scene_bounds: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Annotate BEV image with highlighted objects."""
    annotated = bev.copy()
    size = bev.shape[:2][::-1]
    
    min_pt, max_pt = scene_bounds
    scale = min(size[0], size[1]) / max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1] + 1e-6) * 0.9
    offset_x = (size[0] - (max_pt[0] - min_pt[0]) * scale) / 2
    offset_y = (size[1] - (max_pt[1] - min_pt[1]) * scale) / 2
    
    for obj in objects:
        if obj.obj_id not in highlight_ids or obj.centroid is None:
            continue
        x = int((obj.centroid[0] - min_pt[0]) * scale + offset_x)
        y = int((obj.centroid[1] - min_pt[1]) * scale + offset_y)
        
        if 0 <= x < size[0] and 0 <= y < size[1]:
            cv2.circle(annotated, (x, y), 12, (255, 0, 0), 3)
            cv2.putText(annotated, f"[{obj.obj_id}]{obj.category}", (x+15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return annotated


def compute_iou_3d(box1: Dict, box2: Dict) -> float:
    """Compute 3D IoU between two axis-aligned bounding boxes."""
    min1, max1 = np.array(box1["min"]), np.array(box1["max"])
    min2, max2 = np.array(box2["min"]), np.array(box2["max"])
    
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_size)
    
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / (union_vol + 1e-8)


def format_result(result) -> str:
    """Format a GroundingResult for display."""
    if not result.success:
        return f"Failed: {result.reason}"
    
    pos = result.centroid
    pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos is not None else "N/A"
    
    return f"""
Result:
  Object ID: {result.object_id}
  Category: {result.object_node.category if result.object_node else 'N/A'}
  Position: {pos_str}
  Confidence: {result.confidence:.3f}
  Reasoning: {result.reasoning}
"""
