"""Utility functions for QueryScene."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


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
