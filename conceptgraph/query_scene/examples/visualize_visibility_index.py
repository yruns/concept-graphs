"""
Visualize Bidirectional Visibility Index

This script visualizes the visibility index to verify correctness:
1. Object -> Top5 Views: Show an object's point cloud and its best viewing angles
2. View -> Objects: Show which objects are visible in a view with masks/bboxes

Usage:
    python -m conceptgraph.query_scene.examples.visualize_visibility_index \
        --scene_path /path/to/scene \
        --obj_id 10 \
        --view_id 100
"""

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from loguru import logger

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    logger.warning("open3d not available, point cloud visualization disabled")


def load_visibility_index(index_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load bidirectional visibility index."""
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
    
    obj_to_views = data.get('object_to_views', {})
    view_to_objs = data.get('view_to_objects', {})
    metadata = data.get('metadata', {})
    
    return obj_to_views, view_to_objs, metadata


def load_objects(pcd_file: Path) -> List[Dict[str, Any]]:
    """Load objects from pkl.gz file."""
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    return data.get('objects', [])


def get_image_paths(scene_path: Path, stride: int = 5) -> List[Path]:
    """Get RGB image paths."""
    results_dir = scene_path / "results"
    rgb_files = sorted(results_dir.glob("frame*.jpg"))
    if not rgb_files:
        rgb_files = sorted(results_dir.glob("rgb*.png"))
    
    # Apply stride
    return [rgb_files[i] for i in range(0, len(rgb_files), stride) if i < len(rgb_files)]


def save_object_pointcloud(obj: Dict[str, Any], output_path: Path, color: Tuple[int, int, int] = None):
    """Save object point cloud to PLY file."""
    if not HAS_O3D:
        logger.warning("open3d not available")
        return
    
    points = obj.get('pcd_np')
    if points is None or len(points) == 0:
        logger.warning("No points in object")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Use object color or provided color
    if color is not None:
        colors = np.tile(np.array(color) / 255.0, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif 'pcd_color_np' in obj:
        colors = obj['pcd_color_np']
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(str(output_path), pcd)
    logger.info(f"Saved point cloud: {output_path}")


def save_scene_with_highlighted_objects(
    objects: List[Dict[str, Any]],
    highlight_obj_ids: List[int],
    output_path: Path,
    highlight_colors: List[Tuple[int, int, int]] = None,
    background_color: Tuple[int, int, int] = (128, 128, 128),
):
    """Save all objects' point clouds with highlighted objects in different colors.
    
    Args:
        objects: All scene objects
        highlight_obj_ids: Object IDs to highlight
        output_path: Output PLY path
        highlight_colors: Colors for highlighted objects (RGB 0-255)
        background_color: Color for non-highlighted objects (default gray)
    """
    if not HAS_O3D:
        logger.warning("open3d not available")
        return
    
    # Default highlight colors
    if highlight_colors is None:
        highlight_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
    
    all_points = []
    all_colors = []
    
    highlight_set = set(highlight_obj_ids)
    
    for obj_id, obj in enumerate(objects):
        points = obj.get('pcd_np')
        if points is None or len(points) == 0:
            continue
        
        all_points.append(points)
        
        if obj_id in highlight_set:
            # Use highlight color
            idx = highlight_obj_ids.index(obj_id)
            color = highlight_colors[idx % len(highlight_colors)]
            colors = np.tile(np.array(color) / 255.0, (len(points), 1))
        else:
            # Use background color (gray)
            colors = np.tile(np.array(background_color) / 255.0, (len(points), 1))
        
        all_colors.append(colors)
    
    if not all_points:
        logger.warning("No points to save")
        return
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    logger.success(f"Saved scene point cloud with {len(highlight_obj_ids)} highlighted objects: {output_path}")


def visualize_object_top_views(
    obj_id: int,
    objects: List[Dict],
    obj_to_views: Dict[int, List[Tuple[int, float]]],
    image_paths: List[Path],
    output_dir: Path,
    top_k: int = 5,
):
    """Visualize an object and its top-k views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if obj_id >= len(objects):
        logger.error(f"Object {obj_id} not found (total: {len(objects)})")
        return
    
    obj = objects[obj_id]
    class_name = obj.get('class_name', ['unknown'])[0] if isinstance(obj.get('class_name'), list) else obj.get('class_name', 'unknown')
    
    logger.info(f"=== Object {obj_id}: {class_name} ===")
    
    # Save scene point cloud with highlighted object
    ply_path = output_dir / f"scene_highlight_object_{obj_id}.ply"
    save_scene_with_highlighted_objects(
        objects, 
        highlight_obj_ids=[obj_id],
        output_path=ply_path,
        highlight_colors=[(255, 0, 0)],  # Red for target object
    )
    
    # Get top views
    views = obj_to_views.get(obj_id, [])[:top_k]
    logger.info(f"Top {top_k} views: {views}")
    
    if not views:
        logger.warning("No views found for this object")
        return
    
    # Get object's image_idx and xyxy for drawing bbox
    image_indices = obj.get('image_idx', [])
    xyxy_list = obj.get('xyxy', [])
    
    # Save top view images
    view_images = []
    for rank, (view_id, score) in enumerate(views):
        if view_id >= len(image_paths):
            continue
        
        img_path = image_paths[view_id]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Draw bbox for this object in this view
        bbox_color = (0, 0, 255)  # Red for target object
        for i, img_idx in enumerate(image_indices):
            if img_idx == view_id and i < len(xyxy_list):
                xyxy = xyxy_list[i]
                if xyxy is not None and len(xyxy) == 4:
                    x1, y1, x2, y2 = [int(c) for c in xyxy]
                    cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 3)
                    # Add label
                    label = f"{class_name}"
                    cv2.putText(img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                break
        
        # Add text overlay with score breakdown
        text = f"View {view_id} (score: {score:.3f})"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save individual view
        view_path = output_dir / f"object_{obj_id}_view_{rank+1}_id{view_id}.jpg"
        cv2.imwrite(str(view_path), img)
        
        # Resize for combined image
        img_resized = cv2.resize(img, (400, 300))
        view_images.append(img_resized)
    
    # Create combined image
    if view_images:
        # Arrange in a row or 2 rows
        if len(view_images) <= 3:
            combined = np.hstack(view_images)
        else:
            row1 = np.hstack(view_images[:3])
            row2_imgs = view_images[3:] + [np.zeros_like(view_images[0])] * (3 - len(view_images[3:]))
            row2 = np.hstack(row2_imgs[:3])
            combined = np.vstack([row1, row2])
        
        combined_path = output_dir / f"object_{obj_id}_top{top_k}_views.jpg"
        cv2.imwrite(str(combined_path), combined)
        logger.success(f"Saved combined views: {combined_path}")


def visualize_view_objects(
    view_id: int,
    objects: List[Dict],
    view_to_objs: Dict[int, List[Tuple[int, float]]],
    image_paths: List[Path],
    output_dir: Path,
    stride: int = 5,
):
    """Visualize which objects are visible in a view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if view_id >= len(image_paths):
        logger.error(f"View {view_id} not found (total: {len(image_paths)})")
        return
    
    # Get visible objects
    visible_objs = view_to_objs.get(view_id, [])
    logger.info(f"=== View {view_id}: {len(visible_objs)} visible objects ===")
    
    if not visible_objs:
        logger.warning("No objects visible in this view")
        return
    
    # Load image
    img_path = image_paths[view_id]
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Failed to load image: {img_path}")
        return
    
    img_annotated = img.copy()
    
    # Note: image_idx stores view_id (sampled index), not frame_idx
    # So we search for view_id directly in image_idx
    
    # Colors for different objects
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
    ]
    
    # Annotate each visible object
    obj_info = []
    for rank, (obj_id, score) in enumerate(visible_objs[:8]):  # Limit to top 8
        if obj_id >= len(objects):
            continue
        
        obj = objects[obj_id]
        class_names = obj.get('class_name', ['unknown'])
        class_name = class_names[0] if isinstance(class_names, list) else class_names
        
        color = colors[rank % len(colors)]
        
        # Try to find mask/bbox for this view
        image_indices = obj.get('image_idx', [])
        mask_indices = obj.get('mask_idx', [])
        xyxy_list = obj.get('xyxy', [])
        mask_list = obj.get('mask', [])
        
        # Find if this object was detected in this view
        # image_idx stores view_id (0-399), not frame number
        bbox_found = False
        for i, img_idx in enumerate(image_indices):
            if img_idx == view_id and i < len(xyxy_list):
                # Draw bounding box
                xyxy = xyxy_list[i]
                if xyxy is not None and len(xyxy) == 4:
                    x1, y1, x2, y2 = [int(c) for c in xyxy]
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{obj_id}:{class_name[:15]} ({score:.2f})"
                    cv2.putText(img_annotated, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    bbox_found = True
                
                # Draw mask if available
                if i < len(mask_list) and mask_list[i] is not None:
                    mask = mask_list[i]
                    if isinstance(mask, np.ndarray) and mask.shape[:2] == img.shape[:2]:
                        overlay = img_annotated.copy()
                        overlay[mask > 0] = color
                        cv2.addWeighted(overlay, 0.3, img_annotated, 0.7, 0, img_annotated)
                break
        
        if not bbox_found:
            # Object visible by geometry but not detected in this specific frame
            # Draw centroid projection instead
            centroid = obj.get('pcd_np', np.zeros((1, 3))).mean(axis=0)
            obj_info.append((obj_id, class_name, score, centroid))
    
    # Save scene point cloud with highlighted visible objects
    highlight_obj_ids = [obj_id for obj_id, _ in visible_objs[:8]]
    ply_path = output_dir / f"scene_view_{view_id}_visible_objects.ply"
    save_scene_with_highlighted_objects(
        objects,
        highlight_obj_ids=highlight_obj_ids,
        output_path=ply_path,
        highlight_colors=colors,
    )
    
    # Add legend
    legend_y = 30
    for rank, (obj_id, score) in enumerate(visible_objs[:8]):
        if obj_id >= len(objects):
            continue
        obj = objects[obj_id]
        class_names = obj.get('class_name', ['unknown'])
        class_name = class_names[0] if isinstance(class_names, list) else class_names
        color = colors[rank % len(colors)]
        
        text = f"{obj_id}: {class_name} (score={score:.2f})"
        cv2.putText(img_annotated, text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        legend_y += 20
    
    # Save annotated image
    annotated_path = output_dir / f"view_{view_id}_objects_annotated.jpg"
    cv2.imwrite(str(annotated_path), img_annotated)
    logger.success(f"Saved annotated view: {annotated_path}")
    
    # Save original for comparison
    original_path = output_dir / f"view_{view_id}_original.jpg"
    cv2.imwrite(str(original_path), img)
    
    # Print summary
    logger.info("Visible objects:")
    for obj_id, score in visible_objs[:10]:
        if obj_id < len(objects):
            obj = objects[obj_id]
            class_names = obj.get('class_name', ['unknown'])
            class_name = class_names[0] if isinstance(class_names, list) else class_names
            logger.info(f"  Object {obj_id}: {class_name} (score={score:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Visualize visibility index")
    parser.add_argument("--scene_path", type=str, 
                       default="/home/shyue/Datasets/Replica/Replica/room0")
    parser.add_argument("--obj_id", type=int, default=10, help="Object ID to visualize")
    parser.add_argument("--view_id", type=int, default=100, help="View ID to visualize")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene_path)
    output_dir = Path(args.output_dir) if args.output_dir else scene_path / "vis_index_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load visibility index
    index_path = scene_path / "indices" / "visibility_index.pkl"
    if not index_path.exists():
        logger.error(f"Visibility index not found: {index_path}")
        return
    
    obj_to_views, view_to_objs, metadata = load_visibility_index(index_path)
    logger.info(f"Loaded index: {len(obj_to_views)} objects, {len(view_to_objs)} views")
    
    # Load objects
    pcd_dir = scene_path / "pcd_saves"
    pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
    
    if not pcd_files:
        logger.error("No PCD files found")
        return
    
    objects = load_objects(pcd_files[0])
    logger.info(f"Loaded {len(objects)} objects")
    
    # Get image paths
    image_paths = get_image_paths(scene_path, args.stride)
    logger.info(f"Found {len(image_paths)} images")
    
    # 1. Visualize object -> top views
    logger.info("=" * 60)
    logger.info("PART 1: Object -> Top Views")
    logger.info("=" * 60)
    obj_output = output_dir / f"object_{args.obj_id}"
    visualize_object_top_views(
        args.obj_id, objects, obj_to_views, image_paths, obj_output
    )
    
    # 2. Visualize view -> objects
    logger.info("=" * 60)
    logger.info("PART 2: View -> Objects")
    logger.info("=" * 60)
    view_output = output_dir / f"view_{args.view_id}"
    visualize_view_objects(
        args.view_id, objects, view_to_objs, image_paths, view_output, args.stride
    )
    
    logger.info("=" * 60)
    logger.success(f"Visualization complete! Check: {output_dir}")


if __name__ == "__main__":
    main()
