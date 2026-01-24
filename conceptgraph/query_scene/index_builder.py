"""
Index building for efficient retrieval.

This module provides classes for building:
1. Hierarchical CLIP index (region -> object -> point)
2. View-object visibility index with semantic scoring
3. Spatial KD-tree for proximity queries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from .data_structures import ObjectNode, RegionNode, ViewScore


# Optional FAISS import
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not available, using numpy fallback for vector search")


class CLIPIndex:
    """Hierarchical CLIP feature index for semantic retrieval.
    
    Provides efficient nearest-neighbor search over CLIP features
    at multiple granularities: region, object, and optionally point-level.
    
    Attributes:
        feature_dim: Dimension of CLIP features (default 1024 for ViT-H-14).
        index: FAISS or numpy index for search.
        metadata: List of metadata associated with each indexed item.
    """
    
    def __init__(self, feature_dim: int = 1024):
        self.feature_dim = feature_dim
        self.features: Optional[np.ndarray] = None
        self.index: Optional[Any] = None
        self.metadata: List[Dict] = []
    
    def build_from_objects(self, objects: List[ObjectNode]) -> None:
        """Build index from object CLIP features.
        
        Args:
            objects: List of ObjectNode instances with clip_feature set.
        """
        features = []
        self.metadata = []
        
        for obj in objects:
            if obj.clip_feature is not None:
                features.append(obj.clip_feature)
                self.metadata.append({
                    "obj_id": obj.obj_id,
                    "category": obj.category,
                    "centroid": obj.centroid.tolist() if obj.centroid is not None else None,
                })
        
        if not features:
            print("Warning: No features to index")
            return
        
        self.features = np.array(features, dtype=np.float32)
        self._build_index()
        
        print(f"Built CLIP index with {len(self.features)} objects")
    
    def _build_index(self) -> None:
        """Build the search index from features."""
        if self.features is None or len(self.features) == 0:
            return
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.features / norms
        
        if HAS_FAISS:
            # Use FAISS with inner product (equivalent to cosine after normalization)
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self.index.add(normalized.astype(np.float32))
        else:
            # Numpy fallback
            self.index = normalized
    
    def search(
        self,
        query_feature: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float, Dict]]:
        """Search for nearest neighbors.
        
        Args:
            query_feature: Query CLIP feature vector.
            top_k: Number of results to return.
        
        Returns:
            List of (index, score, metadata) tuples sorted by score descending.
        """
        if self.index is None:
            return []
        
        query = np.asarray(query_feature, dtype=np.float32).reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)
        
        if HAS_FAISS:
            scores, indices = self.index.search(query, min(top_k, len(self.metadata)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            scores = (self.index @ query.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.metadata):
                results.append((int(idx), float(score), self.metadata[idx]))
        
        return results
    
    def search_by_text(
        self,
        text: str,
        clip_model: Any,
        top_k: int = 10,
    ) -> List[Tuple[int, float, Dict]]:
        """Search using text query via CLIP encoding.
        
        Args:
            text: Text query string.
            clip_model: CLIP model with encode_text method.
            top_k: Number of results to return.
        
        Returns:
            List of (index, score, metadata) tuples.
        """
        import torch
        import open_clip
        
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        with torch.no_grad():
            tokens = tokenizer([text])
            if hasattr(clip_model, 'cuda'):
                tokens = tokens.cuda()
            text_feat = clip_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu().numpy().flatten()
        
        return self.search(text_feat, top_k)


class VisibilityIndex:
    """Bidirectional index between objects and views.
    
    Stores ViewScore for each (object, view) pair where the object
    is visible, enabling efficient lookup in both directions.
    """
    
    def __init__(self):
        # object_id -> [(view_id, ViewScore)]
        self.object_to_views: Dict[int, List[Tuple[int, ViewScore]]] = {}
        # view_id -> [(object_id, ViewScore)]
        self.view_to_objects: Dict[int, List[Tuple[int, ViewScore]]] = {}
    
    def build(
        self,
        objects: List[ObjectNode],
        camera_poses: List[Any],
        depth_paths: List[Any],
        rgb_images: Optional[List[np.ndarray]] = None,
        clip_model: Optional[Any] = None,
        visibility_radius: float = 5.0,
        min_visible_ratio: float = 0.1,
    ) -> None:
        """Build visibility index with geometric and semantic scoring.
        
        Args:
            objects: List of ObjectNode instances.
            camera_poses: List of CameraPose instances.
            depth_paths: List of paths to depth images.
            rgb_images: Optional list of RGB images for semantic scoring.
            clip_model: Optional CLIP model for semantic scoring.
            visibility_radius: Maximum distance for visibility consideration.
            min_visible_ratio: Minimum visibility ratio to include in index.
        """
        print(f"Building visibility index for {len(objects)} objects, {len(camera_poses)} views...")
        
        # Pre-compute category text features if CLIP model is provided
        category_text_features = {}
        if clip_model is not None:
            category_text_features = self._compute_category_features(objects, clip_model)
        
        for view_id, pose in enumerate(camera_poses):
            if view_id % 50 == 0:
                print(f"  Processing view {view_id}/{len(camera_poses)}")
            
            view_objects = []
            
            for obj in objects:
                if obj.centroid is None:
                    continue
                
                # Compute geometric visibility
                distance = np.linalg.norm(obj.centroid - pose.position)
                if distance > visibility_radius:
                    continue
                
                # Compute view score
                score = self._compute_view_score(obj, pose, view_id, distance)
                
                if score.visible_ratio < min_visible_ratio:
                    continue
                
                # Compute semantic score if CLIP model and images available
                if clip_model is not None and rgb_images is not None and obj.category in category_text_features:
                    score.semantic_score = self._compute_semantic_score(
                        obj, view_id, rgb_images, clip_model, 
                        category_text_features[obj.category],
                        camera_pose=pose,
                        K=pose.intrinsics if hasattr(pose, 'intrinsics') else None,
                    )
                
                # Add to indices
                if obj.obj_id not in self.object_to_views:
                    self.object_to_views[obj.obj_id] = []
                self.object_to_views[obj.obj_id].append((view_id, score))
                
                view_objects.append((obj.obj_id, score))
            
            if view_objects:
                self.view_to_objects[view_id] = view_objects
        
        # Sort by composite score
        for obj_id in self.object_to_views:
            self.object_to_views[obj_id].sort(
                key=lambda x: x[1].get_composite_score(), reverse=True
            )
        
        for view_id in self.view_to_objects:
            self.view_to_objects[view_id].sort(
                key=lambda x: x[1].get_composite_score(), reverse=True
            )
        
        print(f"Built visibility index: {len(self.object_to_views)} objects, {len(self.view_to_objects)} views")
    
    def _compute_category_features(
        self, 
        objects: List[ObjectNode], 
        clip_model: Any
    ) -> Dict[str, np.ndarray]:
        """Pre-compute CLIP features for category text prompts."""
        import torch
        import open_clip
        
        categories = set(obj.category for obj in objects)
        features = {}
        
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        for cat in categories:
            prompt = f"a photo of a {cat}"
            with torch.no_grad():
                tokens = tokenizer([prompt])
                if hasattr(clip_model, 'cuda') and next(clip_model.parameters()).is_cuda:
                    tokens = tokens.cuda()
                feat = clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features[cat] = feat.cpu().numpy().flatten()
        
        return features
    
    def _compute_view_score(
        self,
        obj: ObjectNode,
        pose: Any,
        view_id: int,
        distance: float,
    ) -> ViewScore:
        """Compute geometric visibility score for an object in a view."""
        # Simple distance-based visibility (can be enhanced with depth-based visibility)
        max_dist = 5.0
        visible_ratio = max(0, 1 - distance / max_dist)
        
        # View quality based on viewing angle (simplified)
        if obj.centroid is not None:
            view_dir = obj.centroid - pose.position
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
            
            # Assuming forward is -Z in camera frame
            forward = pose.rotation[:, 2]  # Third column
            cos_angle = np.dot(view_dir, forward)
            view_quality = max(0, cos_angle)
        else:
            view_quality = 0.5
        
        # Resolution estimate (pixels per meter)
        resolution = pose.fx / max(distance, 0.1)
        
        return ViewScore(
            view_id=view_id,
            visible_ratio=visible_ratio,
            view_quality=view_quality,
            resolution=resolution,
            occlusion_ratio=0.0,  # TODO: implement depth-based occlusion
            semantic_score=0.0,
        )
    
    def _compute_semantic_score(
        self,
        obj: ObjectNode,
        view_id: int,
        rgb_images: Optional[List[np.ndarray]],
        clip_model: Any,
        category_feature: np.ndarray,
        camera_pose: Any = None,
        K: Optional[np.ndarray] = None,
    ) -> float:
        """Compute semantic score using CLIP similarity.
        
        This implements the key insight from the plan: a view that is
        "geometrically clear" may not be "semantically representative".
        We compute CLIP similarity between the object crop and category text.
        
        Args:
            obj: Object node with bbox_3d
            view_id: View index
            rgb_images: List of RGB images
            clip_model: CLIP model for encoding
            category_feature: Pre-computed CLIP feature for category text
            camera_pose: Camera pose for projection
            K: Camera intrinsic matrix
        
        Returns:
            Semantic score in [0, 1]
        """
        if rgb_images is None or view_id >= len(rgb_images):
            return 0.0
        
        if obj.bbox_3d is None:
            return 0.0
        
        rgb = rgb_images[view_id]
        h, w = rgb.shape[:2]
        
        # Use default intrinsics if not provided
        if K is None:
            K = np.array([
                [600.0, 0.0, w / 2],
                [0.0, 600.0, h / 2],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        
        try:
            from .utils import project_3d_bbox_to_2d, crop_object_from_image
            
            # Project 3D bbox to 2D
            bbox_2d = project_3d_bbox_to_2d(obj.bbox_3d, camera_pose, K, (w, h))
            if bbox_2d is None:
                return 0.0
            
            # Crop object region with padding
            crop = crop_object_from_image(rgb, bbox_2d, padding=0.15, min_size=32)
            if crop is None:
                return 0.0
            
            # Encode crop with CLIP
            import torch
            from PIL import Image
            import open_clip
            
            # Get CLIP preprocessing transform
            _, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            
            # Convert to PIL and preprocess
            crop_pil = Image.fromarray(crop)
            crop_tensor = preprocess(crop_pil).unsqueeze(0)
            
            if next(clip_model.parameters()).is_cuda:
                crop_tensor = crop_tensor.cuda()
            
            with torch.no_grad():
                crop_feature = clip_model.encode_image(crop_tensor)
                crop_feature = crop_feature / crop_feature.norm(dim=-1, keepdim=True)
                crop_feature = crop_feature.cpu().numpy().flatten()
            
            # Compute cosine similarity
            similarity = np.dot(crop_feature, category_feature)
            return float(np.clip(similarity, 0, 1))
            
        except Exception as e:
            # Fallback to placeholder if anything fails
            return 0.5
    
    def get_best_views(
        self,
        obj_id: int,
        top_k: int = 5,
    ) -> List[Tuple[int, ViewScore]]:
        """Get best views for an object."""
        views = self.object_to_views.get(obj_id, [])
        return views[:top_k]
    
    def get_visible_objects(
        self,
        view_id: int,
        min_score: float = 0.0,
    ) -> List[Tuple[int, ViewScore]]:
        """Get objects visible in a view."""
        objects = self.view_to_objects.get(view_id, [])
        if min_score > 0:
            objects = [(oid, score) for oid, score in objects 
                      if score.get_composite_score() >= min_score]
        return objects


class SpatialIndex:
    """Spatial index for proximity queries using KD-tree."""
    
    def __init__(self):
        self.tree: Optional[KDTree] = None
        self.object_ids: List[int] = []
    
    def build(self, objects: List[ObjectNode]) -> None:
        """Build spatial index from object centroids."""
        positions = []
        self.object_ids = []
        
        for obj in objects:
            if obj.centroid is not None:
                positions.append(obj.centroid)
                self.object_ids.append(obj.obj_id)
        
        if positions:
            self.tree = KDTree(np.array(positions))
            print(f"Built spatial index with {len(positions)} objects")
    
    def query_radius(
        self,
        center: np.ndarray,
        radius: float,
    ) -> List[int]:
        """Find all objects within radius of center."""
        if self.tree is None:
            return []
        
        indices = self.tree.query_ball_point(center, radius)
        return [self.object_ids[i] for i in indices]
    
    def query_nearest(
        self,
        center: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find k nearest objects to center."""
        if self.tree is None:
            return []
        
        distances, indices = self.tree.query(center, k=min(k, len(self.object_ids)))
        
        if isinstance(indices, int):
            indices = [indices]
            distances = [distances]
        
        return [(self.object_ids[i], float(d)) for i, d in zip(indices, distances)]
    
    def find_nearby(self, obj_id: int, radius: float = 1.5) -> List[Tuple[int, float]]:
        """Find objects within radius of given object.
        
        Args:
            obj_id: Source object ID
            radius: Search radius in meters
            
        Returns:
            List of (object_id, distance) tuples, sorted by distance
        """
        if self.tree is None or obj_id not in self.object_ids:
            return []
        
        # Find source object index
        idx = self.object_ids.index(obj_id)
        center = self.tree.data[idx]
        
        # Query radius
        nearby_indices = self.tree.query_ball_point(center, radius)
        
        # Calculate distances and return
        results = []
        for i in nearby_indices:
            if i != idx:  # Exclude self
                dist = float(np.linalg.norm(self.tree.data[i] - center))
                results.append((self.object_ids[i], dist))
        
        return sorted(results, key=lambda x: x[1])


class RegionIndex:
    """Region-level index for hierarchical retrieval.
    
    Supports automatic region clustering when explicit regions
    are not available, using spatial and semantic clustering.
    """
    
    def __init__(self, feature_dim: int = 1024):
        self.feature_dim = feature_dim
        self.regions: List[Dict] = []  # {region_id, centroid, clip_feature, object_ids}
        self.region_features: Optional[np.ndarray] = None
        self.object_to_region: Dict[int, int] = {}
    
    def build_from_regions(self, regions: List[RegionNode]) -> None:
        """Build index from explicit region nodes."""
        if not regions:
            print("No explicit regions provided")
            return
        
        self.regions = []
        features = []
        
        for region in regions:
            self.regions.append({
                "region_id": region.region_id,
                "centroid": region.centroid if hasattr(region, 'centroid') else None,
                "object_ids": region.object_ids,
            })
            if region.clip_feature is not None:
                features.append(region.clip_feature)
            
            for obj_id in region.object_ids:
                self.object_to_region[obj_id] = region.region_id
        
        if features:
            self.region_features = np.array(features, dtype=np.float32)
            # Normalize
            norms = np.linalg.norm(self.region_features, axis=1, keepdims=True)
            self.region_features = self.region_features / (norms + 1e-8)
        
        print(f"Built region index with {len(self.regions)} regions")
    
    def build_from_objects(
        self,
        objects: List[ObjectNode],
        n_clusters: Optional[int] = None,
    ) -> None:
        """Build regions automatically using spatial-semantic clustering.
        
        When explicit regions are not available, clusters objects
        based on their spatial positions and semantic features.
        """
        if len(objects) < 3:
            # Too few objects, single region
            self._build_single_region(objects)
            return
        
        # Determine number of clusters
        if n_clusters is None:
            # Heuristic: sqrt(n_objects), clamped to [2, 10]
            n_clusters = max(2, min(10, int(np.sqrt(len(objects)))))
        
        # Collect features: [centroid (3d), clip_feature]
        positions = []
        features = []
        valid_objects = []
        
        for obj in objects:
            if obj.centroid is None:
                continue
            valid_objects.append(obj)
            positions.append(obj.centroid)
            if obj.clip_feature is not None:
                features.append(obj.clip_feature)
            else:
                features.append(np.zeros(self.feature_dim))
        
        if len(valid_objects) < n_clusters:
            self._build_single_region(objects)
            return
        
        positions = np.array(positions)
        features = np.array(features)
        
        # Normalize positions to [0, 1] range
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        pos_normalized = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        
        # Normalize CLIP features
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feat_normalized = features / (feat_norms + 1e-8)
        
        # Combine: weight spatial more for region clustering
        # [x, y, z] + 0.3 * [clip_features...]
        combined = np.hstack([pos_normalized * 1.0, feat_normalized * 0.3])
        
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(combined)
        except Exception as e:
            # Fallback: simple spatial clustering (covers ImportError, numpy dtype issues, etc.)
            print(f"Warning: sklearn clustering failed ({e}), using simple spatial clustering")
            labels = self._simple_spatial_cluster(positions, n_clusters)
        
        # Build regions from clusters
        self.regions = []
        region_feats = []
        
        for cluster_id in range(n_clusters):
            cluster_objs = [obj for obj, label in zip(valid_objects, labels) if label == cluster_id]
            if not cluster_objs:
                continue
            
            # Compute region centroid
            centroids = np.array([obj.centroid for obj in cluster_objs])
            region_centroid = centroids.mean(axis=0)
            
            # Compute aggregated CLIP feature
            clip_feats = [obj.clip_feature for obj in cluster_objs if obj.clip_feature is not None]
            if clip_feats:
                region_clip = np.mean(clip_feats, axis=0)
            else:
                region_clip = np.zeros(self.feature_dim)
            
            region_id = len(self.regions)
            self.regions.append({
                "region_id": region_id,
                "centroid": region_centroid,
                "object_ids": [obj.obj_id for obj in cluster_objs],
            })
            region_feats.append(region_clip)
            
            for obj in cluster_objs:
                self.object_to_region[obj.obj_id] = region_id
        
        if region_feats:
            self.region_features = np.array(region_feats, dtype=np.float32)
            norms = np.linalg.norm(self.region_features, axis=1, keepdims=True)
            self.region_features = self.region_features / (norms + 1e-8)
        
        print(f"Built region index with {len(self.regions)} auto-clustered regions")
    
    def _build_single_region(self, objects: List[ObjectNode]) -> None:
        """Build single region containing all objects."""
        obj_ids = [obj.obj_id for obj in objects]
        self.regions = [{
            "region_id": 0,
            "centroid": None,
            "object_ids": obj_ids,
        }]
        for obj_id in obj_ids:
            self.object_to_region[obj_id] = 0
        
        # Aggregate features
        feats = [obj.clip_feature for obj in objects if obj.clip_feature is not None]
        if feats:
            self.region_features = np.mean(feats, axis=0, keepdims=True).astype(np.float32)
        
        print("Built single-region index (few objects)")
    
    def _simple_spatial_cluster(self, positions: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple grid-based spatial clustering as fallback."""
        # Divide space into grid cells
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        ranges = maxs - mins + 1e-8
        
        # Assign to grid cells
        grid_size = int(np.ceil(np.cbrt(n_clusters)))
        cell_ids = ((positions - mins) / ranges * (grid_size - 0.01)).astype(int)
        labels = cell_ids[:, 0] * grid_size**2 + cell_ids[:, 1] * grid_size + cell_ids[:, 2]
        
        # Remap to consecutive labels
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])
    
    def search(
        self,
        query_feature: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[int, float, List[int]]]:
        """Search for most relevant regions.
        
        Returns:
            List of (region_id, score, object_ids) tuples
        """
        if self.region_features is None or len(self.regions) == 0:
            return [(0, 1.0, [r["object_ids"] for r in self.regions if r["object_ids"]])]
        
        query = np.asarray(query_feature, dtype=np.float32).flatten()
        query = query / (np.linalg.norm(query) + 1e-8)
        
        scores = self.region_features @ query
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.regions):
                region = self.regions[idx]
                results.append((
                    region["region_id"],
                    float(scores[idx]),
                    region["object_ids"],
                ))
        
        return results
    
    def get_objects_in_region(self, region_id: int) -> List[int]:
        """Get all object IDs in a region."""
        for region in self.regions:
            if region["region_id"] == region_id:
                return region["object_ids"]
        return []
    
    def get_region_for_object(self, obj_id: int) -> Optional[int]:
        """Get region ID for an object."""
        return self.object_to_region.get(obj_id)


@dataclass
class SceneIndices:
    """Container for all scene indices including hierarchical region index.
    
    Provides unified access to:
    - clip_index: Object-level CLIP feature search
    - visibility_index: View-object bidirectional mapping
    - spatial_index: Object centroid proximity search
    - region_index: Region-level hierarchical search
    - point_index: Point-level fine-grained search (Level 3)
    """
    
    clip_index: CLIPIndex
    visibility_index: VisibilityIndex
    spatial_index: SpatialIndex
    region_index: Optional[RegionIndex] = None
    point_index: Optional[PointLevelIndex] = None
    
    @classmethod
    def build_all(
        cls,
        scene: Any,  # QuerySceneRepresentation
        clip_model: Optional[Any] = None,
        build_regions: bool = True,
    ) -> SceneIndices:
        """Build all indices for a scene.
        
        Args:
            scene: QuerySceneRepresentation instance
            clip_model: Optional CLIP model for semantic scoring
            build_regions: Whether to build region index
        
        Returns:
            SceneIndices with all indices built
        """
        print("Building scene indices...")
        
        # 1. Object-level CLIP index
        clip_index = CLIPIndex(feature_dim=scene.feature_dim)
        clip_index.build_from_objects(scene.objects)
        
        # 2. Visibility index
        visibility_index = VisibilityIndex()
        visibility_index.build(
            scene.objects,
            scene.camera_poses,
            scene.depth_paths,
            clip_model=clip_model,
        )
        
        # 3. Spatial index
        spatial_index = SpatialIndex()
        spatial_index.build(scene.objects)
        
        # 4. Region index (hierarchical level 1)
        region_index = None
        if build_regions:
            region_index = RegionIndex(feature_dim=scene.feature_dim)
            if scene.regions:
                region_index.build_from_regions(scene.regions)
            else:
                # Auto-cluster when no explicit regions
                region_index.build_from_objects(scene.objects)
        
        # 5. Point-level index (hierarchical level 3)
        point_index = PointLevelIndex(feature_dim=scene.feature_dim)
        point_index.build(scene.objects)
        
        print("All indices built successfully")
        
        return cls(
            clip_index=clip_index,
            visibility_index=visibility_index,
            spatial_index=spatial_index,
            region_index=region_index,
            point_index=point_index,
        )
    
    def hierarchical_search(
        self,
        query_feature: np.ndarray,
        top_k_regions: int = 3,
        top_k_objects: int = 10,
    ) -> List[Tuple[int, float, Dict]]:
        """Perform hierarchical search: Region -> Object.
        
        First narrows down to top regions, then searches within
        those regions for relevant objects.
        
        Args:
            query_feature: Query CLIP feature
            top_k_regions: Number of regions to consider
            top_k_objects: Number of objects to return
        
        Returns:
            List of (index, score, metadata) from CLIP index
        """
        # If no region index, fall back to direct object search
        if self.region_index is None:
            return self.clip_index.search(query_feature, top_k_objects)
        
        # Step 1: Find top regions
        region_results = self.region_index.search(query_feature, top_k_regions)
        
        # Collect object IDs from top regions
        candidate_obj_ids = set()
        for region_id, score, obj_ids in region_results:
            candidate_obj_ids.update(obj_ids)
        
        # Step 2: Search objects, filtering to candidate regions
        all_results = self.clip_index.search(query_feature, top_k_objects * 2)
        
        # Filter to objects in candidate regions
        filtered_results = [
            (idx, score, meta) for idx, score, meta in all_results
            if meta.get("obj_id") in candidate_obj_ids
        ]
        
        # If not enough results, include all
        if len(filtered_results) < top_k_objects:
            return all_results[:top_k_objects]
        
        return filtered_results[:top_k_objects]
    
    def point_search(
        self,
        query_point: np.ndarray,
        query_text: str = None,
        radius: float = 0.5,
        top_k: int = 100,
    ) -> List[Tuple[int, int, float, float]]:
        """Point-level search combining spatial proximity and semantics.
        
        Args:
            query_point: 3D query point [x, y, z]
            query_text: Optional text query for semantic filtering
            radius: Search radius in meters
            top_k: Maximum points to return
        
        Returns:
            List of (point_idx, obj_id, distance, semantic_score)
        """
        if self.point_index is None:
            return []
        
        return self.point_index.search(
            query_point, query_text, radius=radius, top_k=top_k
        )


class PointLevelIndex:
    """Point-level index for fine-grained 3D localization.
    
    Supports two modes:
    1. Object-propagated features: Use object's CLIP feature for all its points
    2. OpenScene-style features: True per-point CLIP features from multi-view fusion
    
    Implements a hybrid approach inspired by OpenScene:
    - Spatial KDTree for efficient point lookup
    - Point-to-object mapping for semantic features
    - Multi-scale search (point, local neighborhood, object context)
    
    This enables queries like "the armrest of the chair" by:
    1. Finding chair object via CLIP
    2. Searching within chair's point cloud for "armrest" features
    """
    
    def __init__(self, feature_dim: int = 1024, use_per_point_features: bool = False):
        self.feature_dim = feature_dim
        self.use_per_point_features = use_per_point_features
        
        # All points in scene
        self.all_points: Optional[np.ndarray] = None  # (N, 3)
        self.all_colors: Optional[np.ndarray] = None  # (N, 3) 
        self.point_tree: Optional[KDTree] = None
        
        # Point to object mapping
        self.point_to_obj: Optional[np.ndarray] = None  # (N,) object IDs
        
        # Object features for semantic scoring (fallback mode)
        self.obj_features: Dict[int, np.ndarray] = {}
        
        # Per-point CLIP features (OpenScene mode)
        self.point_features: Optional[np.ndarray] = None  # (N, D)
        self._feature_index = None  # FAISS index for feature search
        
        # Per-object point indices for efficient object-scoped search
        self.obj_point_ranges: Dict[int, Tuple[int, int]] = {}  # obj_id -> (start, end)
        
        # Statistics
        self.n_points: int = 0
        self.n_objects: int = 0
    
    def build(self, objects: List[ObjectNode]) -> None:
        """Build point-level index from object point clouds.
        
        Concatenates all object point clouds and builds a unified
        spatial index with object ID mapping.
        
        Args:
            objects: List of ObjectNode with point_cloud attribute
        """
        if not objects:
            print("No objects to build point index from")
            return
        
        all_points = []
        all_colors = []
        point_to_obj = []
        current_idx = 0
        
        for obj in objects:
            if obj.point_cloud is None or len(obj.point_cloud) == 0:
                continue
            
            points = np.asarray(obj.point_cloud)
            if points.ndim == 1:
                continue
            
            n_pts = len(points)
            
            # Store point range for this object
            self.obj_point_ranges[obj.obj_id] = (current_idx, current_idx + n_pts)
            
            # Append points
            if points.shape[1] >= 3:
                all_points.append(points[:, :3])
            
            # Point colors if available (assuming RGB in columns 3-5 or separate)
            if hasattr(obj, 'point_colors') and obj.point_colors is not None:
                all_colors.append(obj.point_colors[:, :3])
            else:
                # Default color from object
                all_colors.append(np.tile([128, 128, 128], (n_pts, 1)))
            
            # Object ID for each point
            point_to_obj.extend([obj.obj_id] * n_pts)
            
            # Store object feature
            if obj.clip_feature is not None:
                self.obj_features[obj.obj_id] = obj.clip_feature.flatten()
            
            current_idx += n_pts
        
        if not all_points:
            print("No valid point clouds found")
            return
        
        self.all_points = np.vstack(all_points).astype(np.float32)
        self.all_colors = np.vstack(all_colors).astype(np.float32)
        self.point_to_obj = np.array(point_to_obj, dtype=np.int32)
        
        # Build KDTree
        self.point_tree = KDTree(self.all_points)
        
        self.n_points = len(self.all_points)
        self.n_objects = len(self.obj_point_ranges)
        
        print(f"Built point index: {self.n_points} points from {self.n_objects} objects")
    
    def load_openscene_features(
        self,
        points: np.ndarray,
        features: np.ndarray,
        objects: Optional[List[ObjectNode]] = None,
    ) -> None:
        """Load OpenScene-style per-point CLIP features.
        
        This replaces the object-propagated features with true per-point features
        computed via multi-view fusion (OpenScene method).
        
        Args:
            points: (N, 3) 3D point coordinates
            features: (N, D) per-point CLIP features
            objects: Optional object list for point-to-object mapping
        """
        self.all_points = np.asarray(points, dtype=np.float32)
        self.point_features = np.asarray(features, dtype=np.float32)
        self.use_per_point_features = True
        
        # Normalize features
        norms = np.linalg.norm(self.point_features, axis=1, keepdims=True)
        self.point_features = self.point_features / (norms + 1e-8)
        
        self.n_points = len(self.all_points)
        self.feature_dim = self.point_features.shape[1]
        
        # Build spatial tree
        self.point_tree = KDTree(self.all_points)
        
        # Build FAISS index for feature search
        try:
            import faiss
            self._feature_index = faiss.IndexFlatIP(self.feature_dim)
            self._feature_index.add(self.point_features)
            print(f"Built FAISS index for {self.n_points} point features")
        except ImportError:
            self._feature_index = None
            print("FAISS not available, using numpy for feature search")
        
        # Map points to objects if provided
        if objects:
            self._map_points_to_objects(objects)
        
        print(f"Loaded OpenScene features: {self.n_points} points, {self.feature_dim}D")
    
    def _map_points_to_objects(self, objects: List[ObjectNode]) -> None:
        """Map loaded points to nearest objects."""
        if self.all_points is None:
            return
        
        # Build object centroids
        obj_centroids = []
        obj_ids = []
        for obj in objects:
            if obj.centroid is not None:
                obj_centroids.append(obj.centroid)
                obj_ids.append(obj.obj_id)
        
        if not obj_centroids:
            return
        
        obj_centroids = np.array(obj_centroids)
        obj_tree = KDTree(obj_centroids)
        
        # Find nearest object for each point
        _, nearest_obj_indices = obj_tree.query(self.all_points)
        self.point_to_obj = np.array([obj_ids[i] for i in nearest_obj_indices], dtype=np.int32)
        
        # Store object features
        for obj in objects:
            if obj.clip_feature is not None:
                self.obj_features[obj.obj_id] = obj.clip_feature.flatten()
    
    def search_by_text(
        self,
        query_text: str,
        text_encoder: Any,
        top_k: int = 100,
    ) -> List[Tuple[int, float]]:
        """Search for points by text query using per-point features.
        
        Only works when OpenScene features are loaded.
        
        Args:
            query_text: Text query (e.g., "chair", "wooden surface")
            text_encoder: Function to encode text to CLIP feature
            top_k: Number of results
        
        Returns:
            List of (point_index, similarity_score)
        """
        if self.point_features is None:
            print("No per-point features available. Use load_openscene_features() first.")
            return []
        
        # Encode query
        query_feat = text_encoder(query_text)
        if query_feat is None:
            return []
        
        query_feat = np.asarray(query_feat, dtype=np.float32).flatten()
        
        # Ensure same dimension
        if len(query_feat) != self.feature_dim:
            print(f"Feature dimension mismatch: query={len(query_feat)}, index={self.feature_dim}")
            return []
        
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        
        # Search
        if self._feature_index is not None:
            # FAISS search
            query_feat = query_feat.reshape(1, -1)
            scores, indices = self._feature_index.search(query_feat, top_k)
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # Numpy fallback
            similarities = self.point_features @ query_feat
            top_indices = np.argsort(-similarities)[:top_k]
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def search(
        self,
        query_point: np.ndarray,
        query_text: str = None,
        radius: float = 0.5,
        top_k: int = 100,
        clip_encoder: Any = None,
    ) -> List[Tuple[int, int, float, float]]:
        """Search for points near query location with optional semantic filter.
        
        Args:
            query_point: 3D query point [x, y, z]
            query_text: Optional text for semantic scoring
            radius: Search radius in meters
            top_k: Maximum points to return
        
        Returns:
            List of (point_idx, obj_id, distance, semantic_score) sorted by score
        """
        if self.point_tree is None:
            return []
        
        query_point = np.asarray(query_point).flatten()[:3]
        
        # Spatial search
        indices = self.point_tree.query_ball_point(query_point, radius)
        
        if not indices:
            # Fall back to k-nearest
            distances, indices = self.point_tree.query(query_point, k=min(top_k, self.n_points))
            if isinstance(indices, int):
                indices = [indices]
                distances = [distances]
            indices = [i for i, d in zip(indices, distances) if d <= radius * 2]
        
        if not indices:
            return []
        
        # Compute scores
        results = []
        query_feat = None
        
        # Encode text query if provided
        if query_text and clip_encoder:
            try:
                query_feat = clip_encoder(query_text)
                if query_feat is not None:
                    query_feat = query_feat.flatten()
                    query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
            except Exception:
                query_feat = None
        
        for idx in indices:
            obj_id = int(self.point_to_obj[idx]) if self.point_to_obj is not None else -1
            point = self.all_points[idx]
            dist = float(np.linalg.norm(point - query_point))
            
            # Semantic score - use per-point features if available
            semantic_score = 0.5  # default
            if query_feat is not None:
                if self.use_per_point_features and self.point_features is not None:
                    # Use true per-point CLIP features (OpenScene mode)
                    point_feat = self.point_features[idx]
                    point_feat = point_feat / (np.linalg.norm(point_feat) + 1e-8)
                    semantic_score = float(np.dot(query_feat, point_feat))
                elif obj_id in self.obj_features:
                    # Fallback: use object-level features
                    obj_feat = self.obj_features[obj_id]
                    obj_feat = obj_feat / (np.linalg.norm(obj_feat) + 1e-8)
                    semantic_score = float(np.dot(query_feat, obj_feat))
            
            # Combined score: proximity + semantics
            proximity_score = 1.0 - min(dist / radius, 1.0)
            combined_score = 0.4 * proximity_score + 0.6 * max(semantic_score, 0)
            
            results.append((idx, obj_id, dist, combined_score))
        
        # Sort by combined score descending
        results.sort(key=lambda x: -x[3])
        
        return results[:top_k]
    
    def search_within_object(
        self,
        obj_id: int,
        query_point: np.ndarray,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Search for points within a specific object, sorted by distance.
        
        Useful for part-level localization within an object.
        
        Args:
            obj_id: Object ID to search within
            query_point: 3D query point
            top_k: Maximum points to return
        
        Returns:
            List of (point_idx, distance) for points in that object
        """
        if obj_id not in self.obj_point_ranges:
            return []
        
        start, end = self.obj_point_ranges[obj_id]
        obj_points = self.all_points[start:end]
        
        query_point = np.asarray(query_point).flatten()[:3]
        distances = np.linalg.norm(obj_points - query_point, axis=1)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)[:top_k]
        
        return [(start + int(i), float(distances[i])) for i in sorted_indices]
    
    def get_object_points(self, obj_id: int) -> Optional[np.ndarray]:
        """Get all points belonging to an object.
        
        Args:
            obj_id: Object ID
        
        Returns:
            Points array (N, 3) or None
        """
        if obj_id not in self.obj_point_ranges:
            return None
        
        start, end = self.obj_point_ranges[obj_id]
        return self.all_points[start:end]
    
    def get_local_neighborhood(
        self,
        point_idx: int,
        radius: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get local point neighborhood around a point.
        
        Useful for computing local geometric features.
        
        Args:
            point_idx: Center point index
            radius: Search radius
        
        Returns:
            Tuple of (points, object_ids) in neighborhood
        """
        if self.point_tree is None or point_idx >= self.n_points:
            return np.array([]), np.array([])
        
        center = self.all_points[point_idx]
        indices = self.point_tree.query_ball_point(center, radius)
        
        points = self.all_points[indices]
        obj_ids = self.point_to_obj[indices]
        
        return points, obj_ids
    
    def compute_point_features(
        self,
        point_indices: List[int],
        method: str = "object_propagate",
    ) -> np.ndarray:
        """Compute or retrieve features for specified points.
        
        Methods:
        - "object_propagate": Use parent object's CLIP feature
        - "neighborhood_blend": Blend features from nearby objects
        
        Args:
            point_indices: List of point indices
            method: Feature computation method
        
        Returns:
            Features array (N, feature_dim)
        """
        n_points = len(point_indices)
        features = np.zeros((n_points, self.feature_dim), dtype=np.float32)
        
        if method == "object_propagate":
            # Simple: use object feature for each point
            for i, pt_idx in enumerate(point_indices):
                obj_id = int(self.point_to_obj[pt_idx])
                if obj_id in self.obj_features:
                    features[i] = self.obj_features[obj_id]
        
        elif method == "neighborhood_blend":
            # Blend features from nearby objects
            for i, pt_idx in enumerate(point_indices):
                _, neighbor_obj_ids = self.get_local_neighborhood(pt_idx, radius=0.2)
                
                if len(neighbor_obj_ids) == 0:
                    continue
                
                # Weighted average of neighbor object features
                unique_objs = np.unique(neighbor_obj_ids)
                total_weight = 0
                blended = np.zeros(self.feature_dim)
                
                for obj_id in unique_objs:
                    obj_id = int(obj_id)
                    if obj_id in self.obj_features:
                        weight = np.sum(neighbor_obj_ids == obj_id)
                        blended += weight * self.obj_features[obj_id]
                        total_weight += weight
                
                if total_weight > 0:
                    features[i] = blended / total_weight
        
        return features
    
    def summary(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "n_points": self.n_points,
            "n_objects": self.n_objects,
            "n_objects_with_features": len(self.obj_features),
            "feature_dim": self.feature_dim,
            "has_tree": self.point_tree is not None,
        }
