"""
QuerySceneRepresentation: Container for the complete scene representation.

This module provides the main container class that holds all scene data
including objects, regions, views, and various indices for efficient querying.
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_structures import (
    BoundingBox3D,
    CameraPose,
    ObjectDescriptions,
    ObjectNode,
    RegionNode,
    ViewScore,
)


@dataclass
class QuerySceneRepresentation:
    """Container for the complete scene representation.
    
    This class holds all the data needed for query-driven scene understanding:
    - Object nodes with geometry, features, and descriptions
    - Region nodes for spatial organization
    - Camera poses and RGB-D images (paths)
    - Various indices for efficient retrieval
    
    Attributes:
        scene_id: Unique identifier for the scene.
        objects: List of ObjectNode instances.
        regions: List of RegionNode instances.
        camera_poses: List of CameraPose instances.
        image_paths: Paths to RGB images indexed by view_id.
        depth_paths: Paths to depth images indexed by view_id.
        bev_image: Bird's eye view image of the scene.
        scene_bounds: Axis-aligned bounding box of the entire scene.
    """
    
    scene_id: str
    
    # Core data
    objects: List[ObjectNode] = field(default_factory=list)
    regions: List[RegionNode] = field(default_factory=list)
    camera_poses: List[CameraPose] = field(default_factory=list)
    
    # Image paths (lazy loading)
    image_paths: List[Path] = field(default_factory=list)
    depth_paths: List[Path] = field(default_factory=list)
    
    # Scene-level data
    bev_image: Optional[np.ndarray] = None
    scene_bounds: Optional[BoundingBox3D] = None
    
    # Feature dimension
    feature_dim: int = 1024
    
    # Indices (populated by build_indices)
    _object_features: Optional[np.ndarray] = None
    _region_features: Optional[np.ndarray] = None
    _object_index: Optional[Any] = None  # FAISS index
    _region_index: Optional[Any] = None
    _visibility_index: Optional[Dict] = None  # object_id -> [(view_id, ViewScore)]
    _spatial_tree: Optional[Any] = None  # KDTree
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        pass
    
    @classmethod
    def from_pcd_file(
        cls,
        pcd_file: Path,
        scene_path: Path,
        stride: int = 5,
    ) -> QuerySceneRepresentation:
        """Load scene representation from ConceptGraphs pcd file.
        
        Args:
            pcd_file: Path to the .pkl.gz file containing objects.
            scene_path: Root path of the scene (contains results/, traj.txt, etc.)
            stride: Frame stride used during mapping.
        
        Returns:
            QuerySceneRepresentation instance.
        """
        pcd_file = Path(pcd_file)
        scene_path = Path(scene_path)
        
        print(f"Loading scene from: {pcd_file}")
        
        # Load pcd data
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        raw_objects = data.get('objects', [])
        print(f"  Found {len(raw_objects)} raw objects")
        
        # Create scene representation
        scene = cls(scene_id=scene_path.name)
        
        # Parse objects
        scene._parse_objects(raw_objects)
        
        # Load camera poses
        scene._load_camera_poses(scene_path, stride)
        
        # Set image paths
        scene._set_image_paths(scene_path, stride)
        
        # Compute scene bounds
        scene._compute_scene_bounds()
        
        print(f"  Loaded {len(scene.objects)} objects, {len(scene.camera_poses)} poses")
        
        return scene
    
    def _parse_objects(self, raw_objects: List[Dict]) -> None:
        """Parse raw object data into ObjectNode instances."""
        for i, obj in enumerate(raw_objects):
            # Get point cloud
            pcd_np = obj.get('pcd_np')
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            pcd_np = np.asarray(pcd_np, dtype=np.float32)
            
            # Get CLIP feature
            clip_ft = obj.get('clip_ft')
            if clip_ft is not None:
                if hasattr(clip_ft, 'cpu'):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.asarray(clip_ft, dtype=np.float32).flatten()
            
            # Get category
            category = self._extract_category(obj, i)
            
            # Create bounding box
            bbox = BoundingBox3D.from_points(pcd_np)
            
            # Create object node
            node = ObjectNode(
                obj_id=len(self.objects),
                category=category,
                point_cloud=pcd_np,
                bbox_3d=bbox,
                centroid=pcd_np.mean(axis=0),
                n_points=len(pcd_np),
                clip_feature=clip_ft,
                detection_confidence=obj.get('conf', 1.0),
            )
            
            self.objects.append(node)
    
    def _extract_category(self, obj: Dict, default_id: int) -> str:
        """Extract category name from object data."""
        class_names = obj.get('class_name', [])
        if class_names:
            valid = [n for n in class_names if n and n.lower() not in ['item', 'object']]
            if valid:
                return Counter(valid).most_common(1)[0][0]
            if class_names[0]:
                return class_names[0]
        return f"object_{default_id}"
    
    def _load_camera_poses(self, scene_path: Path, stride: int) -> None:
        """Load camera poses from trajectory file."""
        # Try different pose file names
        pose_files = [
            scene_path / 'traj.txt',
            scene_path / 'traj_w_c.txt',
            scene_path / 'results' / 'traj.txt',
        ]
        
        pose_file = None
        for pf in pose_files:
            if pf.exists():
                pose_file = pf
                break
        
        if pose_file is None:
            print(f"  Warning: No pose file found in {scene_path}")
            return
        
        # Load intrinsics if available
        intrinsics = self._load_intrinsics(scene_path)
        
        # Parse poses
        poses = []
        with open(pose_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    # Format: tx ty tz qw qx qy qz
                    pose_data = [float(p) for p in parts[:7]]
                    poses.append(pose_data)
        
        # Apply stride
        for i, pose_data in enumerate(poses[::stride]):
            position = np.array(pose_data[:3], dtype=np.float32)
            quaternion = np.array(pose_data[3:7], dtype=np.float32)
            
            # Convert quaternion to rotation matrix (simplified)
            rotation = self._quaternion_to_rotation(quaternion)
            
            self.camera_poses.append(CameraPose(
                position=position,
                rotation=rotation,
                intrinsics=intrinsics,
            ))
    
    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [qw, qx, qy, qz] to rotation matrix."""
        qw, qx, qy, qz = q
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
        ], dtype=np.float32)
        
        return R
    
    def _load_intrinsics(self, scene_path: Path) -> Optional[np.ndarray]:
        """Load camera intrinsics from config files."""
        # Try to load from common locations
        intrinsic_files = [
            scene_path / 'cam_params.json',
            scene_path / 'intrinsics.txt',
        ]
        
        for f in intrinsic_files:
            if f.exists():
                if f.suffix == '.json':
                    with open(f) as fp:
                        data = json.load(fp)
                    if 'camera_matrix' in data:
                        return np.array(data['camera_matrix'], dtype=np.float32)
                elif f.suffix == '.txt':
                    return np.loadtxt(f, dtype=np.float32).reshape(3, 3)
        
        # Default Replica intrinsics
        return np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    
    def _set_image_paths(self, scene_path: Path, stride: int) -> None:
        """Set paths to RGB and depth images.
        
        Supports multiple directory structures:
        - results/color/*.jpg + results/depth/*.png (standard)
        - results/frame*.jpg + results/depth*.png (Replica flat)
        """
        results_dir = scene_path / 'results'
        color_dir = results_dir / 'color'
        depth_dir = results_dir / 'depth'
        
        # Try standard structure first
        if color_dir.exists():
            all_images = sorted(color_dir.glob('*.jpg')) + sorted(color_dir.glob('*.png'))
        elif results_dir.exists():
            # Replica flat structure: results/frame*.jpg
            all_images = sorted(results_dir.glob('frame*.jpg'))
            if not all_images:
                all_images = sorted(results_dir.glob('*.jpg'))
        else:
            all_images = []
        
        if all_images:
            self.image_paths = [all_images[i] for i in range(0, len(all_images), stride) 
                               if i < len(all_images)]
        
        # Try standard depth structure first
        if depth_dir.exists():
            all_depths = sorted(depth_dir.glob('*.png')) + sorted(depth_dir.glob('*.npy'))
        elif results_dir.exists():
            # Replica flat structure: results/depth*.png
            all_depths = sorted(results_dir.glob('depth*.png'))
        else:
            all_depths = []
        
        if all_depths:
            self.depth_paths = [all_depths[i] for i in range(0, len(all_depths), stride)
                               if i < len(all_depths)]
    
    def _compute_scene_bounds(self) -> None:
        """Compute the bounding box of the entire scene."""
        if not self.objects:
            return
        
        all_centroids = np.array([obj.centroid for obj in self.objects if obj.centroid is not None])
        if len(all_centroids) == 0:
            return
        
        # Add margin
        margin = 0.5
        self.scene_bounds = BoundingBox3D(
            min_point=all_centroids.min(axis=0) - margin,
            max_point=all_centroids.max(axis=0) + margin,
        )
    
    def get_object_by_id(self, obj_id: int) -> Optional[ObjectNode]:
        """Get object by ID."""
        if 0 <= obj_id < len(self.objects):
            return self.objects[obj_id]
        return None
    
    def get_objects_by_category(self, category: str) -> List[ObjectNode]:
        """Get all objects matching a category."""
        category_lower = category.lower()
        return [obj for obj in self.objects if category_lower in obj.category.lower()]
    
    def get_image(self, view_id: int) -> Optional[np.ndarray]:
        """Load and return RGB image for a view."""
        if 0 <= view_id < len(self.image_paths):
            import cv2
            img = cv2.imread(str(self.image_paths[view_id]))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    
    def get_depth(self, view_id: int) -> Optional[np.ndarray]:
        """Load and return depth image for a view."""
        if 0 <= view_id < len(self.depth_paths):
            path = self.depth_paths[view_id]
            if path.suffix == '.npy':
                return np.load(path)
            else:
                import cv2
                return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the scene."""
        category_counts = Counter(obj.category for obj in self.objects)
        
        return {
            "scene_id": self.scene_id,
            "n_objects": len(self.objects),
            "n_regions": len(self.regions),
            "n_views": len(self.camera_poses),
            "scene_bounds": self.scene_bounds.to_dict() if self.scene_bounds else None,
            "categories": dict(category_counts.most_common(20)),
        }
    
    def save(self, output_path: Path) -> None:
        """Save the scene representation to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        data = {
            "scene_id": self.scene_id,
            "objects": [obj.to_dict(include_features=True) for obj in self.objects],
            "regions": [reg.to_dict() for reg in self.regions],
            "image_paths": [str(p) for p in self.image_paths],
            "depth_paths": [str(p) for p in self.depth_paths],
            "scene_bounds": self.scene_bounds.to_dict() if self.scene_bounds else None,
            "feature_dim": self.feature_dim,
        }
        
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved scene to: {output_path}")
    
    @classmethod
    def load(cls, path: Path) -> QuerySceneRepresentation:
        """Load scene representation from disk."""
        path = Path(path)
        
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        
        scene = cls(scene_id=data["scene_id"])
        scene.objects = [ObjectNode.from_dict(d) for d in data.get("objects", [])]
        scene.regions = [RegionNode.from_dict(d) for d in data.get("regions", [])]
        scene.image_paths = [Path(p) for p in data.get("image_paths", [])]
        scene.depth_paths = [Path(p) for p in data.get("depth_paths", [])]
        scene.feature_dim = data.get("feature_dim", 1024)
        
        if data.get("scene_bounds"):
            scene.scene_bounds = BoundingBox3D.from_dict(data["scene_bounds"])
        
        return scene
    
    def __repr__(self) -> str:
        return f"QuerySceneRepresentation(scene_id='{self.scene_id}', objects={len(self.objects)}, views={len(self.camera_poses)})"
