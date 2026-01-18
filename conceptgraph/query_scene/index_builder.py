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
                
                # Compute semantic score if CLIP model available
                if clip_model is not None and obj.category in category_text_features:
                    score.semantic_score = self._compute_semantic_score(
                        obj, view_id, rgb_images, clip_model, 
                        category_text_features[obj.category]
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
    ) -> float:
        """Compute semantic score using CLIP similarity."""
        if rgb_images is None or view_id >= len(rgb_images):
            return 0.0
        
        # This would require projecting the object to 2D and cropping
        # For now, return a placeholder
        # TODO: Implement proper object crop and CLIP encoding
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


@dataclass
class SceneIndices:
    """Container for all scene indices."""
    
    clip_index: CLIPIndex
    visibility_index: VisibilityIndex
    spatial_index: SpatialIndex
    
    @classmethod
    def build_all(
        cls,
        scene: Any,  # QuerySceneRepresentation
        clip_model: Optional[Any] = None,
    ) -> SceneIndices:
        """Build all indices for a scene."""
        clip_index = CLIPIndex(feature_dim=scene.feature_dim)
        clip_index.build_from_objects(scene.objects)
        
        visibility_index = VisibilityIndex()
        visibility_index.build(
            scene.objects,
            scene.camera_poses,
            scene.depth_paths,
            clip_model=clip_model,
        )
        
        spatial_index = SpatialIndex()
        spatial_index.build(scene.objects)
        
        return cls(
            clip_index=clip_index,
            visibility_index=visibility_index,
            spatial_index=spatial_index,
        )
