"""
Keyframe Selector: Query-driven keyframe selection for VLM grounding.

This module implements the core functionality of selecting relevant keyframes
(RGB images) given a natural language query about a 3D scene.

Key Features:
1. Scene-aware query parsing: LLM maps query terms to scene object labels
2. CLIP-based semantic retrieval: Handles synonyms and semantic similarity
3. Visibility-based view selection: Finds views that best observe target objects
4. Joint coverage optimization: Selects views covering both target and anchor objects
5. Nested spatial query support: Complex queries like "pillow on sofa nearest door"
6. Multi-hypothesis fallback: DIRECT → PROXY → CONTEXT for robust grounding

Usage:
    selector = KeyframeSelector.from_scene_path("/path/to/scene", llm_model="gemini-2.5-pro")
    result = selector.select_keyframes_v2("the pillow on the sofa nearest the door", k=3)

    # result.keyframe_indices: [42, 67, 89]
    # result.keyframe_paths: [Path("frame000042.jpg"), ...]
    # result.target_objects: [ObjectNode(...), ...]
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Iterable
import numpy as np
from loguru import logger

# Optional imports
try:
    import torch
    import open_clip
    HAS_CLIP = True
except Exception as e:
    HAS_CLIP = False
    logger.warning(f"open_clip not available, CLIP features will not work: {e}")

# Import nested query modules
from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    HypothesisOutputV1,
    QueryHypothesis,
    HypothesisKind,
    ParseMode,
)
from .query_parser import QueryParser
from .query_executor import QueryExecutor, ExecutionResult
from .spatial_relations import SpatialRelationChecker, check_relation


@dataclass
class SceneObject:
    """Scene object representation with all attributes from pkl.gz file.
    
    Attributes are based on the output of:
    - 1b_extract_2d_segmentation_detect.sh (2D segmentation)
    - 2b_build_3d_object_map_detect.sh (3D object map)
    """
    
    # Core identification
    obj_id: int
    category: str  # Primary category (most common from class_name list)
    object_tag: str = ""  # Display tag (same as category if not specified)
    centroid: Optional[np.ndarray] = None  # 3D position [x, y, z], computed from pcd_np
    
    # 3D point cloud data
    pcd_np: Optional[np.ndarray] = None  # Point cloud (N, 3)
    pcd_color_np: Optional[np.ndarray] = None  # Point colors (N, 3)
    bbox_np: Optional[np.ndarray] = None  # 3D bounding box corners (8, 3)
    
    # CLIP features
    clip_ft: Optional[np.ndarray] = None  # CLIP visual feature (1024,)
    text_ft: Optional[np.ndarray] = None  # CLIP text feature (1024,)
    
    # Detection data (per-frame lists)
    image_idx: List[int] = field(default_factory=list)  # Frame indices where detected
    mask_idx: List[int] = field(default_factory=list)  # Mask index per frame
    class_name: List[str] = field(default_factory=list)  # Class name per detection
    class_id: List[int] = field(default_factory=list)  # Class ID per detection
    conf: List[float] = field(default_factory=list)  # Detection confidence
    xyxy: List[Any] = field(default_factory=list)  # 2D bounding boxes
    n_points: List[int] = field(default_factory=list)  # Points per detection
    pixel_area: List[int] = field(default_factory=list)  # Pixel area per detection
    
    # Metadata
    num_detections: int = 0  # Total detection count
    inst_color: Optional[np.ndarray] = None  # Instance color (3,)
    is_background: bool = False
    
    # Optional rich information (from affordance extraction)
    summary: str = ""  # Description
    affordance_category: str = ""  # e.g., "lighting", "seating"
    co_objects: List[str] = field(default_factory=list)  # Related objects
    affordances: Dict[str, Any] = field(default_factory=dict)  # Full affordance payload
    
    # Alias for backward compatibility
    @property
    def clip_feature(self) -> Optional[np.ndarray]:
        return self.clip_ft
    
    @property
    def point_cloud(self) -> Optional[np.ndarray]:
        return self.pcd_np
    
    @classmethod
    def from_dict(cls, obj_id: int, data: Dict[str, Any]) -> "SceneObject":
        """Create SceneObject from raw pkl.gz dict data."""
        from collections import Counter
        
        # Extract category (most common class_name)
        class_names = data.get('class_name', [])
        if class_names:
            valid_names = [n for n in class_names if n and str(n).lower() not in ('item', 'none', '')]
            if valid_names:
                category = Counter(valid_names).most_common(1)[0][0]
            else:
                category = class_names[0] if class_names else f"object_{obj_id}"
        else:
            category = f"object_{obj_id}"
        
        # Get point cloud and compute centroid
        pcd_np = data.get('pcd_np')
        centroid = None
        if pcd_np is not None and len(pcd_np) > 0:
            pcd_np = np.asarray(pcd_np)
            centroid = pcd_np.mean(axis=0)
        
        return cls(
            obj_id=obj_id,
            category=category,
            object_tag=category,
            centroid=centroid,
            # 3D data
            pcd_np=pcd_np,
            pcd_color_np=np.asarray(data['pcd_color_np']) if data.get('pcd_color_np') is not None else None,
            bbox_np=np.asarray(data['bbox_np']) if data.get('bbox_np') is not None else None,
            # Features
            clip_ft=np.asarray(data['clip_ft']) if data.get('clip_ft') is not None else None,
            text_ft=np.asarray(data['text_ft']) if data.get('text_ft') is not None else None,
            # Detection data
            image_idx=list(data.get('image_idx', [])),
            mask_idx=list(data.get('mask_idx', [])),
            class_name=list(data.get('class_name', [])),
            class_id=list(data.get('class_id', [])),
            conf=list(data.get('conf', [])),
            xyxy=list(data.get('xyxy', [])),
            n_points=list(data.get('n_points', [])),
            pixel_area=list(data.get('pixel_area', [])),
            # Metadata
            num_detections=data.get('num_detections', 0),
            inst_color=np.asarray(data['inst_color']) if data.get('inst_color') is not None else None,
            is_background=bool(data.get('is_background', False)),
        )


@dataclass
class KeyframeResult:
    """Result of keyframe selection."""
    
    query: str
    target_term: str
    anchor_term: Optional[str]
    
    # Selected keyframes
    keyframe_indices: List[int]  # Frame indices (view_id)
    keyframe_paths: List[Path]  # Paths to RGB images
    
    # Matched objects
    target_objects: List[SceneObject]
    anchor_objects: List[SceneObject]
    
    # Scores and metadata
    selection_scores: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Query: {self.query}",
            f"Target: '{self.target_term}' -> {len(self.target_objects)} objects",
        ]
        if self.anchor_term:
            lines.append(f"Anchor: '{self.anchor_term}' -> {len(self.anchor_objects)} objects")
        lines.append(f"Selected {len(self.keyframe_indices)} keyframes: {self.keyframe_indices}")
        return "\n".join(lines)


class KeyframeSelector:
    """Query-driven keyframe selector for VLM grounding.
    
    This class encapsulates the complete pipeline from query to keyframe selection:
    1. Load scene data (objects, poses, images)
    2. Build object-view visibility index
    3. Parse query with scene context
    4. Select optimal keyframes
    """
    
    def __init__(
        self,
        scene_path: Path,
        pcd_file: Optional[Path] = None,
        affordance_file: Optional[Path] = None,
        stride: int = 5,
        llm_model: str = None,
        use_pool: bool = False,
    ):
        """Initialize keyframe selector.

        Args:
            scene_path: Root path of the scene
            pcd_file: Path to .pkl.gz file with 3D objects
            affordance_file: Path to object_affordances.json (optional)
            stride: Frame stride used during mapping
            llm_model: LLM model name (e.g., "gemini-2.5-pro")
            use_pool: If True, use Gemini pool for concurrent requests
        """
        self.scene_path = Path(scene_path)
        self.stride = stride
        self.llm_model = llm_model
        self.use_pool = use_pool

        # Data containers
        self.objects: List[SceneObject] = []
        self.object_features: Optional[np.ndarray] = None  # (N, D)
        self.camera_poses: List[np.ndarray] = []  # List of 4x4 matrices
        self.image_paths: List[Path] = []
        self.depth_paths: List[Path] = []
        self.intrinsics: np.ndarray = np.array([
            [600.0, 0, 599.5],
            [0, 600.0, 339.5],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # Category index for query parsing
        self.scene_categories: List[str] = []
        
        # Bidirectional visibility index
        # object_to_views: obj_id -> [(view_id, score), ...] sorted by score desc
        self.object_to_views: Dict[int, List[Tuple[int, float]]] = {}
        # view_to_objects: view_id -> [(obj_id, score), ...] sorted by score desc
        self.view_to_objects: Dict[int, List[Tuple[int, float]]] = {}
        
        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_tokenizer = None
        
        # Query parsing components (lazy loaded)
        self._query_parser: Optional[QueryParser] = None
        self._query_executor: Optional[QueryExecutor] = None
        self._relation_checker: Optional[SpatialRelationChecker] = None
        
        # Load data
        self._load_scene(pcd_file, affordance_file)
        self._load_or_build_visibility_index()
    
    @classmethod
    def from_scene_path(
        cls,
        scene_path: str,
        stride: int = 5,
        **kwargs,
    ) -> KeyframeSelector:
        """Create selector from scene path, auto-detecting files.
        
        Args:
            scene_path: Path to scene directory
            stride: Frame stride
            **kwargs: Additional arguments passed to __init__
        
        Returns:
            Initialized KeyframeSelector
        """
        scene_path = Path(scene_path)
        
        # Auto-detect PCD file
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))
        
        pcd_file = pcd_files[0] if pcd_files else None
        logger.info(f"Auto-detected PCD file: {pcd_file}")
        
        # Auto-detect affordance file
        affordance_file = scene_path / "sg_cache_detect" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = scene_path / "sg_cache" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = None
            logger.warning("No affordance file found, using basic object info")
        else:
            logger.info(f"Using affordance file: {affordance_file}")
        
        return cls(scene_path, pcd_file, affordance_file, stride, **kwargs)
    
    def _load_scene(self, pcd_file: Optional[Path], affordance_file: Optional[Path]) -> None:
        """Load scene data from files."""
        logger.info(f"Loading scene from: {self.scene_path}")
        
        # Load 3D objects from PCD file
        if pcd_file and pcd_file.exists():
            self._load_objects_from_pcd(pcd_file)
        
        # Enrich with affordance data if available
        if affordance_file and affordance_file.exists():
            self._load_affordances(affordance_file)
        
        # Load camera poses
        self._load_camera_poses()
        
        # Set image paths
        self._set_image_paths()
        
        # Build category index
        self.scene_categories = list(set(
            obj.object_tag if obj.object_tag else obj.category 
            for obj in self.objects
        ))
        
        logger.success(f"Loaded {len(self.objects)} objects, {len(self.camera_poses)} poses")
        logger.info(f"Scene categories: {self.scene_categories[:20]}...")
    
    def _load_objects_from_pcd(self, pcd_file: Path) -> None:
        """Load objects from ConceptGraphs PCD file."""
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        raw_objects = data.get('objects', [])
        features = []
        
        for i, obj in enumerate(raw_objects):
            # Get point cloud for centroid
            pcd_np = obj.get('pcd_np')
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            pcd_np = np.asarray(pcd_np, dtype=np.float32)
            centroid = pcd_np.mean(axis=0)
            
            # Get CLIP feature
            clip_ft = obj.get('clip_ft')
            if clip_ft is not None:
                if hasattr(clip_ft, 'cpu'):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.asarray(clip_ft, dtype=np.float32).flatten()
                features.append(clip_ft)
            else:
                features.append(None)
            
            # Get category from class_name list
            class_names = obj.get('class_name', [])
            if class_names:
                valid = [n for n in class_names if n and n.lower() not in ['item', 'object']]
                category = Counter(valid).most_common(1)[0][0] if valid else class_names[0]
            else:
                category = f"object_{i}"
            
            # Get detection data for visibility scoring
            image_idx = obj.get('image_idx', [])
            xyxy = obj.get('xyxy', [])
            
            scene_obj = SceneObject(
                obj_id=len(self.objects),
                category=category,
                centroid=centroid,
                clip_ft=clip_ft,
                image_idx=image_idx,
                xyxy=xyxy,
            )
            self.objects.append(scene_obj)
        
        # Build CLIP feature matrix aligned with self.objects indices.
        # Missing features keep zero vectors to preserve index consistency.
        valid_features = [f for f in features if f is not None]
        if valid_features:
            feat_dim = int(valid_features[0].shape[0])
            aligned = np.zeros((len(features), feat_dim), dtype=np.float32)
            for idx, feat in enumerate(features):
                if feat is None:
                    continue
                if feat.shape[0] != feat_dim:
                    logger.warning(
                        f"Object {idx} clip_ft dimension mismatch: "
                        f"{feat.shape[0]} vs expected {feat_dim}, skipping"
                    )
                    continue
                aligned[idx] = feat

            # L2 normalize non-zero rows only
            norms = np.linalg.norm(aligned, axis=1, keepdims=True)
            non_zero = norms.squeeze(-1) > 0
            aligned[non_zero] = aligned[non_zero] / (norms[non_zero] + 1e-8)
            self.object_features = aligned
    
    def _load_affordances(self, affordance_file: Path) -> None:
        """Load and merge affordance data."""
        with open(affordance_file) as f:
            affordances = json.load(f)
        
        # Create mapping by ID
        aff_by_id = {a['id']: a for a in affordances}
        
        for obj in self.objects:
            if obj.obj_id in aff_by_id:
                aff = aff_by_id[obj.obj_id]
                obj.object_tag = aff.get('object_tag', obj.object_tag)
                if obj.object_tag:
                    obj.category = obj.object_tag
                obj.summary = aff.get('summary', obj.summary)
                obj.affordance_category = aff.get('category', obj.affordance_category)

                affs = aff.get('affordances', {})
                if isinstance(affs, dict):
                    obj.affordances = affs
                    obj.co_objects = affs.get('co_objects', obj.co_objects)
    
    def _load_camera_poses(self) -> None:
        """Load camera poses from trajectory file."""
        traj_file = self.scene_path / 'traj.txt'
        if not traj_file.exists():
            logger.warning(f"Trajectory file not found: {traj_file}")
            return
        
        with open(traj_file) as f:
            lines = f.readlines()
        
        # Check format: each line could be 16 numbers (4x4 matrix) or other formats
        first_line_nums = len(lines[0].split()) if lines else 0
        
        all_poses = []
        if first_line_nums == 16:
            # Each line is a full 4x4 matrix
            for line in lines:
                nums = [float(x) for x in line.split()]
                if len(nums) == 16:
                    pose = np.array(nums).reshape(4, 4)
                    all_poses.append(pose)
        else:
            # Traditional format: 4 lines per matrix
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    try:
                        pose = np.array([
                            [float(x) for x in lines[i].split()],
                            [float(x) for x in lines[i+1].split()],
                            [float(x) for x in lines[i+2].split()],
                            [float(x) for x in lines[i+3].split()],
                        ])
                        all_poses.append(pose)
                    except:
                        continue
        
        # Apply stride
        self.camera_poses = [all_poses[i] for i in range(0, len(all_poses), self.stride)
                           if i < len(all_poses)]
    
    def _set_image_paths(self) -> None:
        """Set paths to RGB and depth images."""
        results_dir = self.scene_path / 'results'
        
        # Find all RGB images
        all_images = sorted(results_dir.glob('frame*.jpg'))
        if not all_images:
            all_images = sorted(results_dir.glob('*.jpg'))
        
        # Apply stride
        self.image_paths = [all_images[i] for i in range(0, len(all_images), self.stride)
                          if i < len(all_images)]
        
        # Find depth images
        all_depths = sorted(results_dir.glob('depth*.png'))
        self.depth_paths = [all_depths[i] for i in range(0, len(all_depths), self.stride)
                          if i < len(all_depths)]
    
    def _load_or_build_visibility_index(self) -> None:
        """Load precomputed visibility index or build online.
        
        Prefers offline index from scene_path/indices/visibility_index.pkl
        Falls back to online computation if not available.
        """
        index_path = self.scene_path / "indices" / "visibility_index.pkl"
        
        if index_path.exists():
            try:
                self._load_visibility_index(index_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load visibility index: {e}")
        
        logger.warning(
            f"Visibility index not found at {index_path}. "
            "Building online (slower). Run 6b_build_visibility_index.sh for faster inference."
        )
        self._build_visibility_index_online()
    
    def _load_visibility_index(self, index_path: Path) -> None:
        """Load precomputed bidirectional visibility index from file."""
        logger.info(f"Loading visibility index from {index_path}")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        
        # Support both old format (visibility_index) and new format (object_to_views, view_to_objects)
        if 'object_to_views' in data:
            # New bidirectional format
            raw_obj_to_views = data.get('object_to_views', {})
            raw_view_to_objs = data.get('view_to_objects', {})
            
            self.object_to_views = {int(k): v for k, v in raw_obj_to_views.items()}
            self.view_to_objects = {int(k): v for k, v in raw_view_to_objs.items()}
        else:
            # Old format - only object_to_views, build reverse index
            raw_index = data.get('visibility_index', {})
            self.object_to_views = {int(k): v for k, v in raw_index.items()}
            
            # Build reverse index
            self.view_to_objects = {}
            for obj_id, views in self.object_to_views.items():
                for view_id, score in views:
                    if view_id not in self.view_to_objects:
                        self.view_to_objects[view_id] = []
                    self.view_to_objects[view_id].append((obj_id, score))
            
            # Sort by score
            for view_id in self.view_to_objects:
                self.view_to_objects[view_id].sort(key=lambda x: x[1], reverse=True)
        
        # Check stride consistency
        saved_stride = metadata.get('stride', self.stride)
        if saved_stride != self.stride:
            logger.warning(
                f"Stride mismatch: saved={saved_stride}, current={self.stride}. "
                "View indices may be incorrect."
            )
        
        logger.success(
            f"Loaded bidirectional visibility index: "
            f"{len(self.object_to_views)} objects, {len(self.view_to_objects)} views"
        )
    
    def _build_visibility_index_online(self) -> None:
        """Build bidirectional visibility index online using detection ground truth."""
        logger.info("Building visibility index online (using detection data)...")
        
        img_width, img_height = 1200, 680
        img_area = img_width * img_height
        max_distance = 5.0
        
        self.view_to_objects = {}
        
        for obj in self.objects:
            if obj.image_idx:
                scores = self._compute_visibility_scores(obj, img_width, img_height, img_area, max_distance)
            else:
                scores = self._compute_geometric_scores(obj, max_distance)
            
            scores.sort(key=lambda x: x[1], reverse=True)
            self.object_to_views[obj.obj_id] = scores
            
            for view_id, score in scores:
                if view_id not in self.view_to_objects:
                    self.view_to_objects[view_id] = []
                self.view_to_objects[view_id].append((obj.obj_id, score))
        
        for vid in self.view_to_objects:
            self.view_to_objects[vid].sort(key=lambda x: x[1], reverse=True)
        
        logger.success(f"Built index: {len(self.object_to_views)} objects, {len(self.view_to_objects)} views")
    
    def _compute_visibility_scores(self, obj: SceneObject, img_w: int, img_h: int, img_area: int, max_dist: float) -> List[Tuple[int, float]]:
        """Compute visibility scores using detection data."""
        scores = []
        view_indices = {}
        for i, vid in enumerate(obj.image_idx):
            view_indices.setdefault(vid, []).append(i)
        
        for view_id, indices in view_indices.items():
            if view_id >= len(self.camera_poses):
                continue
            
            completeness = 0.0
            for idx in indices:
                if idx < len(obj.xyxy) and obj.xyxy[idx] is not None:
                    xyxy = obj.xyxy[idx]
                    if hasattr(xyxy, '__len__') and len(xyxy) == 4:
                        x1, y1, x2, y2 = xyxy
                        bbox_area = (x2 - x1) * (y2 - y1)
                        size_score = min(1.0, bbox_area / (img_area * 0.3))
                        is_clipped = x1 < 10 or y1 < 10 or x2 > img_w - 10 or y2 > img_h - 10
                        completeness = max(completeness, max(0, size_score - (0.3 if is_clipped else 0)))
            
            geo = 0.0
            if not np.allclose(obj.centroid, 0):
                pose = self.camera_poses[view_id]
                cam_pos = pose[:3, 3]
                dist = np.linalg.norm(obj.centroid - cam_pos)
                if dist <= max_dist:
                    dist_s = max(0, 1 - dist / max_dist)
                    view_dir = (obj.centroid - cam_pos) / (np.linalg.norm(obj.centroid - cam_pos) + 1e-8)
                    angle_s = max(0, np.dot(view_dir, -pose[:3, 2]))
                    geo = 0.6 * dist_s + 0.4 * angle_s
            
            quality = min(1.0, len(indices) / 3.0)
            scores.append((view_id, 0.5 * completeness + 0.3 * geo + 0.2 * quality))
        return scores
    
    def _compute_geometric_scores(self, obj: SceneObject, max_dist: float) -> List[Tuple[int, float]]:
        """Fallback: geometric-only scoring."""
        scores = []
        for view_id, pose in enumerate(self.camera_poses):
            cam_pos = pose[:3, 3]
            dist = np.linalg.norm(obj.centroid - cam_pos)
            if dist > max_dist:
                continue
            dist_s = max(0, 1 - dist / max_dist)
            view_dir = (obj.centroid - cam_pos) / (np.linalg.norm(obj.centroid - cam_pos) + 1e-8)
            angle_s = max(0, np.dot(view_dir, -pose[:3, 2]))
            combined = 0.6 * dist_s + 0.4 * angle_s
            if combined > 0.1:
                scores.append((view_id, combined))
        return scores
    
    def _compute_visibility_scores_from_detections(
        self, obj: SceneObject, img_width: int, img_height: int, img_area: int, max_distance: float
    ) -> List[Tuple[int, float]]:
        """Compute visibility scores using detection ground truth."""
        scores = []
        
        # Build view_id -> list of detection indices
        view_to_indices: Dict[int, List[int]] = {}
        for i, vid in enumerate(obj.image_idx):
            if vid not in view_to_indices:
                view_to_indices[vid] = []
            view_to_indices[vid].append(i)
        
        for view_id, indices in view_to_indices.items():
            if view_id >= len(self.camera_poses):
                continue
            
            # 1. Completeness score (50% weight)
            best_completeness = 0.0
            for idx in indices:
                if idx < len(obj.xyxy) and obj.xyxy[idx] is not None:
                    xyxy = obj.xyxy[idx]
                    if hasattr(xyxy, '__len__') and len(xyxy) == 4:
                        x1, y1, x2, y2 = xyxy
                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        bbox_area = bbox_w * bbox_h
                        
                        # Size score
                        size_score = min(1.0, bbox_area / (img_area * 0.3))
                        
                        # Clip penalty
                        margin = 10
                        is_clipped = (x1 < margin or y1 < margin or 
                                     x2 > img_width - margin or y2 > img_height - margin)
                        clip_penalty = 0.3 if is_clipped else 0.0
                        
                        completeness = max(0, size_score - clip_penalty)
                        best_completeness = max(best_completeness, completeness)
            
            # 2. Geometric score (30% weight)
            geo_score = 0.0
            if not np.allclose(obj.centroid, 0):
                pose = self.camera_poses[view_id]
                cam_pos = pose[:3, 3]
                distance = np.linalg.norm(obj.centroid - cam_pos)
                if distance <= max_distance:
                    dist_score = max(0, 1 - distance / max_distance)
                    view_dir = (obj.centroid - cam_pos)
                    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                    cam_forward = -pose[:3, 2]
                    angle_score = max(0, np.dot(view_dir, cam_forward))
                    geo_score = 0.6 * dist_score + 0.4 * angle_score
            
            # 3. Detection quality (20% weight)
            quality_score = min(1.0, len(indices) / 3.0)
            
            # Combined score
            combined = 0.5 * best_completeness + 0.3 * geo_score + 0.2 * quality_score
            scores.append((view_id, float(combined)))
        
        return scores
    
    def _compute_visibility_scores_geometric(
        self, obj: SceneObject, max_distance: float
    ) -> List[Tuple[int, float]]:
        """Fallback: compute visibility using only geometric scoring."""
        scores = []
        for view_id, pose in enumerate(self.camera_poses):
            cam_pos = pose[:3, 3]
            distance = np.linalg.norm(obj.centroid - cam_pos)
            if distance > max_distance:
                continue
            dist_score = max(0, 1 - distance / max_distance)
            view_dir = (obj.centroid - cam_pos)
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
            cam_forward = -pose[:3, 2]
            angle_score = max(0, np.dot(view_dir, cam_forward))
            combined = 0.6 * dist_score + 0.4 * angle_score
            if combined > 0.1:
                scores.append((view_id, combined))
        return scores
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for text encoding."""
        if self._clip_model is not None:
            return
        
        if not HAS_CLIP:
            logger.error("CLIP not available")
            return
        
        logger.info("Loading CLIP model...")
        
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            self._clip_model = model.eval()
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.cuda()
            
            logger.success("CLIP model loaded")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
    
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to CLIP feature."""
        self._load_clip_model()
        
        if self._clip_model is None:
            return None
        
        try:
            tokens = self._clip_tokenizer([text])
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            
            with torch.no_grad():
                feat = self._clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Text encoding failed: {e}")
            return None

    def find_objects(self, query_term: str, top_k: int = 10) -> List[SceneObject]:
        """Find objects matching a query term.
        
        Two-stage matching:
        1. Exact/fuzzy string match on object_tag or category
        2. CLIP semantic similarity fallback
        
        Args:
            query_term: Search term
            top_k: Maximum objects to return
            
        Returns:
            List of matching SceneObject
        """
        query_lower = query_term.lower()
        
        # Stage 1: String matching
        matches = []
        for obj in self.objects:
            tag = (obj.object_tag or obj.category).lower()
            # Exact substring match
            if query_lower in tag or tag in query_lower:
                matches.append((obj, 1.0))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in matches[:top_k]]
        
        # Stage 2: CLIP semantic matching
        if self.object_features is not None:
            query_feat = self._encode_text(query_term)
            if query_feat is not None:
                # Compute similarities
                similarities = self.object_features @ query_feat
                top_indices = np.argsort(-similarities)[:top_k]
                
                # Filter by minimum similarity
                min_sim = 0.2
                matches = [
                    (self.objects[i], similarities[i])
                    for i in top_indices if similarities[i] > min_sim
                ]
                
                if matches:
                    logger.info(f"CLIP matched '{query_term}' -> {[(m[0].object_tag or m[0].category, f'{m[1]:.2f}') for m in matches[:5]]}")
                    return [m[0] for m in matches]
        
        logger.warning(f"No objects found for '{query_term}'")
        return []
    
    def get_best_views_for_object(self, obj_id: int, top_k: int = 5) -> List[int]:
        """Get best view indices for a single object."""
        views = self.object_to_views.get(obj_id, [])
        return [v[0] for v in views[:top_k]]
    
    def get_joint_coverage_views(
        self,
        object_ids: List[int],
        max_views: int = 3,
    ) -> List[int]:
        """Greedy selection of views that maximize joint coverage of objects.
        
        Args:
            object_ids: Object IDs to cover
            max_views: Maximum views to select
            
        Returns:
            List of selected view indices
        """
        if not object_ids:
            return []
        
        # Collect all candidate views
        candidate_views: Set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)
        
        if not candidate_views:
            return []
        
        # Build view -> {obj_id: score} mapping
        view_scores: Dict[int, Dict[int, float]] = {}
        for obj_id in object_ids:
            for view_id, score in self.object_to_views.get(obj_id, []):
                if view_id not in view_scores:
                    view_scores[view_id] = {}
                view_scores[view_id][obj_id] = score
        
        # Greedy selection
        selected = []
        covered_quality = {obj_id: 0.0 for obj_id in object_ids}
        
        for _ in range(max_views):
            best_view, best_gain = None, 0.0
            
            for view_id in candidate_views - set(selected):
                # Compute marginal gain
                gain = 0.0
                for obj_id in object_ids:
                    obj_score = view_scores.get(view_id, {}).get(obj_id, 0)
                    if obj_score > covered_quality[obj_id]:
                        gain += obj_score - covered_quality[obj_id]
                
                if gain > best_gain:
                    best_gain, best_view = gain, view_id
            
            if best_view is None:
                break
            
            selected.append(best_view)
            
            # Update covered quality
            for obj_id in object_ids:
                obj_score = view_scores.get(best_view, {}).get(obj_id, 0)
                covered_quality[obj_id] = max(covered_quality[obj_id], obj_score)
        
        return selected

    def _spatial_filter(
        self,
        candidates: List[SceneObject],
        anchor: SceneObject,
        relation: str,
    ) -> List[SceneObject]:
        """Filter candidates by spatial relation to anchor.
        
        Uses the SpatialRelationChecker for accurate relation checking.
        """
        if self._relation_checker is None:
            self._relation_checker = SpatialRelationChecker()
        
        filtered = []
        for obj in candidates:
            result = self._relation_checker.check(obj, anchor, relation)
            if result.satisfies:
                filtered.append((obj, result.score))
        
        # Sort by score and return objects
        if filtered:
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in filtered]
        
        # Fallback: return original candidates if none matched
        logger.warning(f"No candidates matched relation '{relation}', returning all")
        return candidates
    
    def _spatial_filter_multi_anchor(
        self,
        candidates: List[SceneObject],
        anchors: List[SceneObject],
        relation: str,
    ) -> List[SceneObject]:
        """Filter candidates by spatial relation to any of the anchors.
        
        Returns candidates that satisfy the relation with at least one anchor.
        """
        if self._relation_checker is None:
            self._relation_checker = SpatialRelationChecker()
        
        filtered = []
        for obj in candidates:
            best_score = 0.0
            satisfies_any = False
            
            for anchor in anchors:
                result = self._relation_checker.check(obj, anchor, relation)
                if result.satisfies:
                    satisfies_any = True
                    best_score = max(best_score, result.score)
            
            if satisfies_any:
                filtered.append((obj, best_score))
        
        if filtered:
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in filtered]
        
        return candidates
    
    def get_image(self, view_id: int) -> Optional[np.ndarray]:
        """Load RGB image for a view."""
        if view_id >= len(self.image_paths):
            return None
        
        import cv2
        img = cv2.imread(str(self.image_paths[view_id]))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    
    # ========== Hypothesis-Based Query Support (V3) ==========

    def _get_query_parser(self) -> QueryParser:
        """Get or create the query parser."""
        if self._query_parser is None:
            if self.llm_model is None:
                raise ValueError("llm_model is required for nested query parsing")
            self._query_parser = QueryParser(
                llm_model=self.llm_model,
                scene_categories=self.scene_categories,
                use_pool=self.use_pool,
            )
        return self._query_parser
    
    def _get_query_executor(self) -> QueryExecutor:
        """Get or create the query executor."""
        if self._query_executor is None:
            if self._relation_checker is None:
                self._relation_checker = SpatialRelationChecker()

            self._query_executor = QueryExecutor(
                objects=self.objects,
                relation_checker=self._relation_checker,
                clip_features=self.object_features,
                clip_encoder=self._encode_text if HAS_CLIP else None,
            )
        return self._query_executor

    def _generate_scene_images(self) -> List[str]:
        """
        Generate scene visualization images for multimodal query parsing.

        Currently generates a single Bird's Eye View (BEV) image with
        mesh background and annotated object labels. Images are saved to scene_path/bev/.

        Returns:
            List of image paths (currently k=1)
        """
        from .bev_builder import ReplicaBEVBuilder

        # Create bev directory under scene path
        bev_dir = self.scene_path / "bev"
        bev_dir.mkdir(parents=True, exist_ok=True)
        output_path = bev_dir / "scene_bev.png"

        # Find mesh file (e.g., room0_mesh.ply in parent directory)
        scene_name = self.scene_path.name
        mesh_path = self.scene_path.parent / f"{scene_name}_mesh.ply"
        if not mesh_path.exists():
            # Try current directory
            mesh_path = self.scene_path / f"{scene_name}_mesh.ply"
        if not mesh_path.exists():
            mesh_path = None

        builder = ReplicaBEVBuilder()

        # Pass SceneObjects and mesh path
        _, bev_path, obj_id_map = builder.build(
            objects=self.objects,
            output_path=output_path,
            mesh_path=mesh_path,
        )
        logger.info(f"[KeyframeSelector] Generated BEV image with {len(obj_id_map)} objects: {bev_path}")

        return [str(bev_path)]

        return [str(bev_path)]

    def normalize_hypothesis_output(self, payload: Any) -> HypothesisOutputV1:
        """
        Normalize parser payload to HypothesisOutputV1.

        Supported input forms:
        1) HypothesisOutputV1
        2) GroundingQuery
        3) dict with `format_version=hypothesis_output_v1`
        4) legacy dict with `grounding_query`
        5) legacy GroundingQuery dict with `root`
        """
        if isinstance(payload, HypothesisOutputV1):
            output = payload
        elif isinstance(payload, GroundingQuery):
            output = HypothesisOutputV1.from_direct_query(payload)
        elif isinstance(payload, dict):
            if payload.get("format_version") == "hypothesis_output_v1":
                output = HypothesisOutputV1.model_validate(payload)
            elif "grounding_query" in payload:
                grounding_query = self.to_grounding_query(payload)
                output = HypothesisOutputV1(
                    parse_mode=ParseMode.SINGLE,
                    hypotheses=[
                        QueryHypothesis(
                            kind=payload.get("kind", HypothesisKind.DIRECT),
                            rank=1,
                            grounding_query=grounding_query,
                            lexical_hints=list(payload.get("lexical_hints", [])),
                        )
                    ],
                )
            elif "root" in payload:
                grounding_query = GroundingQuery.model_validate(payload)
                output = HypothesisOutputV1.from_direct_query(grounding_query)
            else:
                raise ValueError("Unsupported payload dict format for hypothesis output")
        else:
            raise TypeError(
                f"Unsupported payload type for hypothesis normalization: {type(payload)}"
            )

        ordered = output.ordered_hypotheses()
        if [h.rank for h in ordered] != [h.rank for h in output.hypotheses]:
            output = HypothesisOutputV1(parse_mode=output.parse_mode, hypotheses=ordered)
        return output

    def to_grounding_query(self, hypothesis_payload: Any) -> GroundingQuery:
        """
        Convert one hypothesis payload to GroundingQuery with strict validation.

        This method never silently skips malformed payload.
        """
        if isinstance(hypothesis_payload, GroundingQuery):
            return hypothesis_payload
        if isinstance(hypothesis_payload, QueryHypothesis):
            return hypothesis_payload.grounding_query

        if isinstance(hypothesis_payload, dict):
            raw = hypothesis_payload.get("grounding_query", hypothesis_payload)
            if not isinstance(raw, dict):
                raise ValueError("grounding_query must be a dict payload")
            return GroundingQuery.model_validate(raw)

        raise TypeError(
            f"Unsupported hypothesis payload type for GroundingQuery conversion: {type(hypothesis_payload)}"
        )

    def _iter_query_nodes(self, root: QueryNode):
        """Depth-first traversal over query nodes."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            for constraint in node.spatial_constraints:
                stack.extend(constraint.anchors)
            if node.select_constraint and node.select_constraint.reference:
                stack.append(node.select_constraint.reference)

    def _sanitize_grounding_query_categories(self, grounding_query: GroundingQuery) -> GroundingQuery:
        """
        Ensure all executable categories are in scene categories or UNKNOW.
        """
        scene_set = set(self.scene_categories)
        sanitized = grounding_query.model_copy(deep=True)

        for node in self._iter_query_nodes(sanitized.root):
            cleaned = []
            seen = set()
            for cat in node.categories:
                if cat in scene_set or cat == "UNKNOW":
                    if cat not in seen:
                        cleaned.append(cat)
                        seen.add(cat)
            node.categories = cleaned if cleaned else ["UNKNOW"]

        return sanitized

    def validate_categories_in_scene(self, grounding_query: GroundingQuery) -> None:
        """Validate categories in one GroundingQuery against scene categories."""
        scene_set = set(self.scene_categories)
        for cat in grounding_query.get_all_categories():
            if cat != "UNKNOW" and cat not in scene_set:
                raise ValueError(
                    f"Category '{cat}' is not in scene categories and is not UNKNOW"
                )

    def validate_no_mask_leak(
        self,
        grounding_query: GroundingQuery,
        hidden_categories: Optional[Iterable[str]],
    ) -> None:
        """Validate one GroundingQuery does not leak hidden categories."""
        hidden_set = set(hidden_categories or [])
        if not hidden_set:
            return

        for cat in grounding_query.get_all_categories():
            if cat in hidden_set:
                raise ValueError(f"Masked category leak detected: '{cat}'")

    def parse_query_hypotheses(
        self,
        query: str,
        max_hypotheses: int = 3,
        use_visual_context: bool = True,
    ) -> HypothesisOutputV1:
        """
        Parse query into the unified HypothesisOutputV1 structure.

        The LLM directly outputs HypothesisOutputV1 with all hypotheses.
        This method sanitizes categories and validates the output.

        Args:
            query: Natural language query string
            max_hypotheses: Maximum hypotheses (ignored - LLM decides)
            use_visual_context: If True (default), generate BEV image for multimodal parsing

        Returns:
            HypothesisOutputV1 with hypotheses ready for execution
        """
        parser = self._get_query_parser()

        # Generate scene images if visual context requested
        scene_images = None
        if use_visual_context:
            scene_images = self._generate_scene_images()

        # LLM directly outputs HypothesisOutputV1
        output = parser.parse(query, scene_images=scene_images)

        # Sanitize categories in each hypothesis
        sanitized_hypotheses = []
        for hypo in output.hypotheses:
            sanitized_gq = self._sanitize_grounding_query_categories(hypo.grounding_query)
            sanitized_hypo = QueryHypothesis(
                kind=hypo.kind,
                rank=hypo.rank,
                grounding_query=sanitized_gq,
                lexical_hints=hypo.lexical_hints,
            )
            sanitized_hypotheses.append(sanitized_hypo)

        sanitized_output = HypothesisOutputV1(
            parse_mode=output.parse_mode,
            hypotheses=sanitized_hypotheses,
        )
        sanitized_output.validate_categories(self.scene_categories)
        return sanitized_output

    def _has_unknown_anchors(self, grounding_query: GroundingQuery) -> bool:
        """Check if any anchor or reference in the query has UNKNOW category."""
        for node in self._iter_query_nodes(grounding_query.root):
            # Check spatial constraint anchors
            for sc in node.spatial_constraints:
                for anchor in sc.anchors:
                    if "UNKNOW" in anchor.categories:
                        return True
            # Check select constraint reference
            if node.select_constraint and node.select_constraint.reference:
                if "UNKNOW" in node.select_constraint.reference.categories:
                    return True
        return False

    def execute_query(self, grounding_query: GroundingQuery) -> ExecutionResult:
        """Execute a parsed grounding query.
        
        Args:
            grounding_query: Parsed GroundingQuery object
            
        Returns:
            ExecutionResult with matched objects
        """
        executor = self._get_query_executor()
        return executor.execute(grounding_query)

    def execute_hypotheses(
        self,
        hypothesis_output: Any,
        hidden_categories: Optional[Iterable[str]] = None,
    ) -> Tuple[str, Optional[QueryHypothesis], ExecutionResult]:
        """
        Execute hypotheses by rank and return first non-empty result.

        Special handling:
        - If a hypothesis has UNKNOW anchors/references, skip it even if executor
          returns results (because those results don't satisfy the spatial constraint).
        """
        normalized = self.normalize_hypothesis_output(hypothesis_output)
        normalized.validate_categories(self.scene_categories)

        for hypothesis in normalized.ordered_hypotheses():
            grounding_query = self.to_grounding_query(hypothesis)
            self.validate_categories_in_scene(grounding_query)
            self.validate_no_mask_leak(grounding_query, hidden_categories)

            # Skip hypothesis if it has UNKNOW anchors - spatial constraint can't be satisfied
            if self._has_unknown_anchors(grounding_query):
                logger.debug(
                    f"[execute_hypotheses] Skipping hypothesis {hypothesis.kind.value} "
                    f"with UNKNOW anchors"
                )
                continue

            result = self.execute_query(grounding_query)
            if result.is_empty:
                continue

            if hypothesis.kind == HypothesisKind.DIRECT:
                return "direct_grounded", hypothesis, result
            if hypothesis.kind == HypothesisKind.PROXY:
                return "proxy_grounded", hypothesis, result
            return "context_only", hypothesis, result

        return "no_evidence", None, ExecutionResult(node_id="none", matched_objects=[])

    def map_view_to_frame(self, view_id: int) -> int:
        """Map sampled view index to original frame index."""
        return view_id * self.stride

    def _resolve_keyframe_path(self, view_id: int) -> Tuple[Optional[Path], int]:
        """
        Resolve image path from view index with stride-aware fallback.

        Returns:
            Tuple[path, resolved_view_id]
        """
        search_order = [view_id, view_id - 1, view_id + 1, view_id - 2, view_id + 2]
        for cand_view in search_order:
            if cand_view < 0:
                continue

            cand_frame = self.map_view_to_frame(cand_view)
            expected = self.scene_path / "results" / f"frame{cand_frame:06d}.jpg"
            if expected.exists():
                return expected, cand_view

            if 0 <= cand_view < len(self.image_paths):
                fallback = self.image_paths[cand_view]
                if fallback.exists():
                    return fallback, cand_view

        return None, view_id
    
    def select_keyframes_v2(
        self,
        query: str,
        k: int = 3,
        strategy: str = "joint_coverage",
        hidden_categories: Optional[Iterable[str]] = None,
    ) -> KeyframeResult:
        """
        Select keyframes from the new structured output `HypothesisOutputV1`.

        Execution order:
        1) Parse query into hypotheses (single or multi)
        2) Execute hypotheses by rank
        3) Select keyframes from first successful hypothesis
        """
        logger.info(f"[V3] Selecting {k} keyframes for: '{query}'")

        # Step 1: Parse to new unified structure
        hypothesis_output = self.parse_query_hypotheses(query, max_hypotheses=3)
        logger.info(
            f"[V3] Parsed format={hypothesis_output.format_version}, "
            f"mode={hypothesis_output.parse_mode.value}, "
            f"hypotheses={len(hypothesis_output.hypotheses)}"
        )

        # Step 2: Execute hypotheses by rank
        status, selected_hypothesis, result = self.execute_hypotheses(
            hypothesis_output=hypothesis_output,
            hidden_categories=hidden_categories,
        )

        if status == "no_evidence" or selected_hypothesis is None or result.is_empty:
            logger.warning("[V3] No executable hypothesis produced evidence")
            first = hypothesis_output.ordered_hypotheses()[0]
            return KeyframeResult(
                query=query,
                target_term=first.grounding_query.root.category,
                anchor_term=None,
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={
                    "status": status,
                    "error": "No matching objects",
                    "hypothesis_output": hypothesis_output.model_dump(),
                    "version": "v3",
                }
            )

        target_objects = result.matched_objects
        selected_query = selected_hypothesis.grounding_query
        logger.info(
            f"[V3] Selected hypothesis kind={selected_hypothesis.kind.value}, "
            f"matched={len(target_objects)}"
        )

        # Step 3: Collect anchor objects for joint coverage
        anchor_objects: List[SceneObject] = []
        for constraint in selected_query.root.spatial_constraints:
            for anchor_node in constraint.anchors:
                anchor_result = self._get_query_executor()._execute_node(anchor_node)
                anchor_objects.extend(anchor_result.matched_objects)

        # Step 4: Select keyframes
        all_object_ids = [obj.obj_id for obj in target_objects[:5]]
        if anchor_objects:
            all_object_ids.extend([obj.obj_id for obj in anchor_objects[:3]])

        if strategy == "joint_coverage":
            keyframe_indices = self.get_joint_coverage_views(all_object_ids, max_views=k)
        else:
            keyframe_indices = self.get_joint_coverage_views(
                [obj.obj_id for obj in target_objects[:5]], max_views=k
            )

        keyframe_paths = []
        frame_mappings = []
        for view_id in keyframe_indices:
            requested_frame_id = self.map_view_to_frame(view_id)
            path, resolved_view_id = self._resolve_keyframe_path(view_id)
            resolved_frame_id = self.map_view_to_frame(resolved_view_id)
            if path is not None:
                keyframe_paths.append(path)
            frame_mappings.append(
                {
                    "requested_view_id": view_id,
                    "requested_frame_id": requested_frame_id,
                    "resolved_view_id": resolved_view_id,
                    "resolved_frame_id": resolved_frame_id,
                    "path": str(path) if path is not None else None,
                }
            )

        # Extract anchor term
        anchor_term = None
        if selected_query.root.spatial_constraints:
            anchors = selected_query.root.spatial_constraints[0].anchors
            if anchors:
                anchor_term = anchors[0].category

        return KeyframeResult(
            query=query,
            target_term=selected_query.root.category,
            anchor_term=anchor_term,
            keyframe_indices=keyframe_indices,
            keyframe_paths=keyframe_paths,
            target_objects=target_objects,
            anchor_objects=anchor_objects,
            metadata={
                "status": status,
                "selected_hypothesis_kind": selected_hypothesis.kind.value,
                "selected_hypothesis_rank": selected_hypothesis.rank,
                "strategy": strategy,
                "all_object_ids": all_object_ids,
                "frame_mappings": frame_mappings,
                "hypothesis_output": hypothesis_output.model_dump(),
                "version": "v3",
            }
        )


# Convenience function
def select_keyframes(
    scene_path: str,
    query: str,
    k: int = 3,
    **kwargs,
) -> KeyframeResult:
    """Convenience function to select keyframes for a query.

    Args:
        scene_path: Path to scene directory
        query: Natural language query
        k: Number of keyframes
        **kwargs: Additional arguments for KeyframeSelector

    Returns:
        KeyframeResult
    """
    selector = KeyframeSelector.from_scene_path(scene_path, **kwargs)
    return selector.select_keyframes_v2(query, k=k)
