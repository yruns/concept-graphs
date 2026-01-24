"""
OpenScene-style per-point CLIP feature extraction.

This module implements multi-view feature fusion to compute dense
CLIP features for every 3D point in a scene, following the approach
from OpenScene (CVPR 2023).

Core idea:
1. Extract per-pixel CLIP features from each RGB frame
2. Back-project 2D features to 3D using depth and camera pose
3. Fuse multi-view features with weighted averaging
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class PointFeatureConfig:
    """Configuration for point feature extraction."""
    
    feature_dim: int = 512  # CLIP ViT-B/32 or ViT-L/14
    patch_size: int = 14    # CLIP patch size
    depth_threshold: float = 0.1  # Min depth for valid projection
    max_depth: float = 10.0  # Max depth for valid projection
    fusion_method: str = "weighted_average"  # or "max_pooling"
    min_observations: int = 2  # Min views to compute feature
    weight_by_angle: bool = True  # Weight by viewing angle quality
    weight_by_distance: bool = True  # Weight by distance to camera


class PointFeatureExtractor:
    """Extract per-point CLIP features using multi-view fusion.
    
    Implements the OpenScene approach:
    1. For each view, extract dense CLIP features (patch-level)
    2. Back-project each patch to 3D using depth
    3. Accumulate features for each 3D point across views
    4. Normalize final features
    """
    
    def __init__(
        self,
        config: Optional[PointFeatureConfig] = None,
        clip_model: str = "ViT-B/32",
        device: str = "cuda",
    ):
        self.config = config or PointFeatureConfig()
        self.clip_model_name = clip_model
        self.device = device
        
        # Lazy load models
        self._clip_model = None
        self._clip_preprocess = None
    
    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is not None:
            return
        
        import torch
        
        # Try OpenAI's official CLIP first
        try:
            import clip
            
            logger.info(f"Loading CLIP via openai/clip: {self.clip_model_name}")
            
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self._clip_model.eval()
            self._use_openai_clip = True
            self._use_transformers = False
            
            if "ViT-B" in self.clip_model_name:
                self.config.feature_dim = 512
            elif "ViT-L" in self.clip_model_name:
                self.config.feature_dim = 768
            
            logger.success(f"CLIP loaded via openai/clip, feature_dim={self.config.feature_dim}")
            return
            
        except Exception as e:
            logger.warning(f"openai/clip failed: {e}, trying transformers...")
        
        # Try transformers
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            logger.info(f"Loading CLIP via transformers: {self.clip_model_name}")
            
            hf_model_map = {
                "ViT-B/32": "openai/clip-vit-base-patch32",
                "ViT-B/16": "openai/clip-vit-base-patch16",
                "ViT-L/14": "openai/clip-vit-large-patch14",
            }
            
            model_id = hf_model_map.get(self.clip_model_name, "openai/clip-vit-base-patch32")
            
            self._clip_model = CLIPModel.from_pretrained(model_id)
            self._clip_preprocess = CLIPProcessor.from_pretrained(model_id)
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            self._use_transformers = True
            self._use_openai_clip = False
            
            if "base" in model_id:
                self.config.feature_dim = 512
            elif "large" in model_id:
                self.config.feature_dim = 768
            
            logger.success(f"CLIP loaded via transformers, feature_dim={self.config.feature_dim}")
            return
            
        except Exception as e:
            logger.warning(f"transformers CLIP failed: {e}, trying open_clip...")
        
        # Fallback to open_clip
        try:
            import open_clip
            
            logger.info(f"Loading CLIP via open_clip: {self.clip_model_name}")
            
            model_map = {
                "ViT-B/32": ("ViT-B-32", "openai"),
                "ViT-B/16": ("ViT-B-16", "openai"),
                "ViT-L/14": ("ViT-L-14", "openai"),
            }
            
            if self.clip_model_name in model_map:
                model_name, pretrained = model_map[self.clip_model_name]
            else:
                model_name = self.clip_model_name
                pretrained = "openai"
            
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            self._use_transformers = False
            self._use_openai_clip = False
            
            if "ViT-B" in model_name:
                self.config.feature_dim = 512
            elif "ViT-L" in model_name:
                self.config.feature_dim = 768
            
            logger.success(f"CLIP loaded via open_clip, feature_dim={self.config.feature_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    def extract_image_features(
        self,
        image: np.ndarray,
        return_spatial: bool = True,
    ) -> np.ndarray:
        """Extract CLIP features from an image.
        
        Args:
            image: RGB image (H, W, 3) uint8
            return_spatial: If True, return spatial feature map (H', W', D)
                           If False, return global feature (D,)
        
        Returns:
            Feature array
        """
        self._load_clip()
        
        import torch
        from PIL import Image
        
        # Convert to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        if getattr(self, '_use_openai_clip', False):
            # Using OpenAI's official CLIP
            input_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if return_spatial:
                    features = self._extract_spatial_features_openai_clip(input_tensor)
                else:
                    features = self._clip_model.encode_image(input_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy().flatten().astype(np.float32)
        elif getattr(self, '_use_transformers', False):
            # Using transformers CLIP
            inputs = self._clip_preprocess(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if return_spatial:
                    features = self._extract_spatial_features_transformers(inputs)
                else:
                    outputs = self._clip_model.get_image_features(**inputs)
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                    features = outputs.cpu().numpy().flatten()
        else:
            # Using open_clip
            input_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if return_spatial:
                    features = self._extract_spatial_features(input_tensor)
                else:
                    features = self._clip_model.encode_image(input_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy().flatten()
        
        return features
    
    def _extract_spatial_features_openai_clip(self, input_tensor) -> np.ndarray:
        """Extract spatial features using OpenAI's official CLIP."""
        import torch
        
        visual = self._clip_model.visual
        dtype = visual.conv1.weight.dtype
        device = next(visual.parameters()).device
        
        # Ensure input matches model dtype and device
        input_tensor = input_tensor.to(device=device, dtype=dtype)
        
        # Patch embedding
        x = visual.conv1(input_tensor)
        
        B, D, H, W = x.shape
        x = x.reshape(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        
        # Add class embedding (ensure dtype matches)
        cls_token = visual.class_embedding.to(dtype).expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)
        
        # Add positional embedding
        x = x + visual.positional_embedding.to(dtype)
        
        # Pre-layer norm
        x = visual.ln_pre(x)
        
        # Transformer
        x = x.permute(1, 0, 2)  # (N+1, B, D)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, N+1, D)
        
        # Remove CLS, keep patches
        patch_features = x[:, 1:, :]  # (B, N, D=768)
        
        # Post layer norm
        patch_features = visual.ln_post(patch_features)
        
        # Apply projection to match global feature dimension (768 -> 512)
        if hasattr(visual, 'proj') and visual.proj is not None:
            proj = visual.proj.to(dtype)
            patch_features = patch_features @ proj  # (B, N, 512)
        
        # Reshape to spatial
        patch_features = patch_features.reshape(B, H, W, -1)
        
        # Normalize
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return patch_features[0].float().cpu().numpy()  # (H', W', 512)
    
    def _extract_spatial_features_transformers(self, inputs) -> np.ndarray:
        """Extract spatial features using transformers CLIP."""
        import torch
        
        vision_model = self._clip_model.vision_model
        
        # Get hidden states from vision model
        outputs = vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use last hidden state (includes CLS + patch tokens)
        hidden_states = outputs.last_hidden_state  # (B, N+1, D)
        
        # Remove CLS token, keep only patch tokens
        patch_tokens = hidden_states[:, 1:, :]  # (B, N, D)
        
        # Determine spatial dimensions (assuming square patches)
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        
        if H * W != N:
            # Non-square, use approximate
            H = int(N ** 0.5)
            W = N // H
        
        # Reshape to spatial
        spatial_features = patch_tokens.reshape(B, H, W, D)
        
        # Normalize
        spatial_features = spatial_features / (spatial_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return spatial_features[0].cpu().numpy()  # (H', W', D)
    
    def _extract_spatial_features(self, input_tensor) -> np.ndarray:
        """Extract spatial feature map from open_clip CLIP ViT.
        
        Returns features for each patch, not just the CLS token.
        """
        import torch
        
        # Get the visual encoder
        visual = self._clip_model.visual
        
        # Forward through patch embedding and transformer
        x = visual.conv1(input_tensor)  # (B, D, H', W')
        
        # Get spatial dimensions
        B, D, H, W = x.shape
        
        x = x.reshape(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        
        # Add positional embedding (skip CLS token position)
        # Note: Different CLIP versions have different architectures
        if hasattr(visual, 'positional_embedding'):
            pos_embed = visual.positional_embedding
            if pos_embed.shape[0] == x.shape[1] + 1:
                # Has CLS token, skip first position
                x = x + pos_embed[1:, :].unsqueeze(0)
            else:
                x = x + pos_embed[:x.shape[1], :].unsqueeze(0)
        
        # Apply layer norm if exists
        if hasattr(visual, 'ln_pre'):
            x = visual.ln_pre(x)
        
        # Pass through transformer
        if hasattr(visual, 'transformer'):
            x = visual.transformer(x)
        
        # Final layer norm
        if hasattr(visual, 'ln_post'):
            x = visual.ln_post(x)
        
        # Reshape to spatial
        x = x.reshape(B, H, W, -1)  # (B, H', W', D)
        
        # Normalize features
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        
        return x[0].cpu().numpy()  # (H', W', D)
    
    def backproject_features(
        self,
        spatial_features: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back-project 2D spatial features to 3D points.
        
        Args:
            spatial_features: (H', W', D) feature map from CLIP
            depth: (H, W) depth image in meters
            intrinsics: (3, 3) camera intrinsic matrix
            extrinsics: (4, 4) camera-to-world transform
            image_size: (H, W) original image size
        
        Returns:
            points_3d: (N, 3) 3D points in world coordinates
            features: (N, D) corresponding features
            weights: (N,) confidence weights for fusion
        """
        H, W = image_size
        feat_H, feat_W, D = spatial_features.shape
        
        # Scale factors from feature map to image
        scale_h = H / feat_H
        scale_w = W / feat_W
        
        points_3d = []
        features = []
        weights = []
        
        # Camera position for angle computation
        cam_pos = extrinsics[:3, 3]
        
        for fh in range(feat_H):
            for fw in range(feat_W):
                # Map to image coordinates (center of patch)
                img_u = int((fw + 0.5) * scale_w)
                img_v = int((fh + 0.5) * scale_h)
                
                if img_u >= W or img_v >= H:
                    continue
                
                # Get depth
                z = depth[img_v, img_u]
                
                if z < self.config.depth_threshold or z > self.config.max_depth:
                    continue
                
                # Back-project to camera coordinates
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                
                x_cam = (img_u - cx) * z / fx
                y_cam = (img_v - cy) * z / fy
                z_cam = z
                
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                
                # Transform to world coordinates
                point_world = extrinsics @ point_cam
                point_3d = point_world[:3]
                
                # Compute weight
                weight = 1.0
                
                if self.config.weight_by_distance:
                    # Closer points get higher weight
                    weight *= 1.0 / (1.0 + z / 5.0)
                
                if self.config.weight_by_angle:
                    # Points viewed from front get higher weight
                    view_dir = point_3d - cam_pos
                    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                    # Assume surface normal roughly faces camera
                    angle_weight = max(0.2, abs(np.dot(view_dir, np.array([0, 0, -1]))))
                    weight *= angle_weight
                
                points_3d.append(point_3d)
                features.append(spatial_features[fh, fw])
                weights.append(weight)
        
        if not points_3d:
            return np.zeros((0, 3)), np.zeros((0, D)), np.zeros(0)
        
        return (
            np.array(points_3d, dtype=np.float32),
            np.array(features, dtype=np.float32),
            np.array(weights, dtype=np.float32),
        )
    
    def fuse_multiview_features(
        self,
        all_points: List[np.ndarray],
        all_features: List[np.ndarray],
        all_weights: List[np.ndarray],
        voxel_size: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse features from multiple views using voxel-based aggregation.
        
        Args:
            all_points: List of (N_i, 3) point arrays from each view
            all_features: List of (N_i, D) feature arrays
            all_weights: List of (N_i,) weight arrays
            voxel_size: Voxel size for aggregation (meters)
        
        Returns:
            fused_points: (M, 3) aggregated point locations
            fused_features: (M, D) fused features
        """
        if not all_points:
            return np.zeros((0, 3)), np.zeros((0, self.config.feature_dim))
        
        # Concatenate all points
        concat_points = np.vstack(all_points)
        concat_features = np.vstack(all_features)
        concat_weights = np.concatenate(all_weights)
        
        logger.debug(f"Fusing {len(concat_points)} points from {len(all_points)} views")
        
        # Voxelize for aggregation
        voxel_indices = np.floor(concat_points / voxel_size).astype(np.int32)
        
        # Create unique voxel keys
        voxel_keys = (
            voxel_indices[:, 0] * 1000000 +
            voxel_indices[:, 1] * 1000 +
            voxel_indices[:, 2]
        )
        
        unique_voxels, inverse_indices = np.unique(voxel_keys, return_inverse=True)
        
        n_voxels = len(unique_voxels)
        D = concat_features.shape[1]
        
        # Aggregate features per voxel
        fused_features = np.zeros((n_voxels, D), dtype=np.float32)
        fused_positions = np.zeros((n_voxels, 3), dtype=np.float32)
        voxel_weights = np.zeros(n_voxels, dtype=np.float32)
        voxel_counts = np.zeros(n_voxels, dtype=np.int32)
        
        for i, (pt, feat, w, vox_idx) in enumerate(zip(
            concat_points, concat_features, concat_weights, inverse_indices
        )):
            fused_features[vox_idx] += w * feat
            fused_positions[vox_idx] += w * pt
            voxel_weights[vox_idx] += w
            voxel_counts[vox_idx] += 1
        
        # Normalize
        valid_mask = voxel_weights > 0
        fused_features[valid_mask] /= voxel_weights[valid_mask, np.newaxis]
        fused_positions[valid_mask] /= voxel_weights[valid_mask, np.newaxis]
        
        # Filter by min observations
        if self.config.min_observations > 1:
            valid_mask &= voxel_counts >= self.config.min_observations
        
        # L2 normalize features
        norms = np.linalg.norm(fused_features, axis=1, keepdims=True)
        fused_features = fused_features / (norms + 1e-8)
        
        logger.debug(f"Fused to {valid_mask.sum()} voxels")
        
        return fused_positions[valid_mask], fused_features[valid_mask]
    
    def extract_scene_features(
        self,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        intrinsics: np.ndarray,
        extrinsics_list: List[np.ndarray],
        stride: int = 5,
        voxel_size: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-point CLIP features for entire scene.
        
        This is the main entry point for OpenScene-style feature extraction.
        
        Args:
            rgb_paths: Paths to RGB images
            depth_paths: Paths to depth images
            intrinsics: (3, 3) camera intrinsic matrix
            extrinsics_list: List of (4, 4) camera-to-world transforms
            stride: Process every N-th frame
            voxel_size: Voxel size for fusion
        
        Returns:
            points: (N, 3) fused 3D points
            features: (N, D) per-point CLIP features
        """
        import cv2
        
        n_frames = min(len(rgb_paths), len(depth_paths), len(extrinsics_list))
        logger.info(f"Extracting features from {n_frames} frames (stride={stride})")
        
        all_points = []
        all_features = []
        all_weights = []
        
        for i in range(0, n_frames, stride):
            if i % 50 == 0:
                logger.info(f"  Processing frame {i}/{n_frames}")
            
            try:
                # Load RGB
                rgb = cv2.imread(str(rgb_paths[i]))
                if rgb is None:
                    continue
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]
                
                # Load depth
                depth_path = depth_paths[i]
                if str(depth_path).endswith('.npy'):
                    depth = np.load(depth_path)
                else:
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth is not None:
                        # Convert to meters (assuming 16-bit depth in mm)
                        depth = depth.astype(np.float32) / 1000.0
                
                if depth is None:
                    continue
                
                # Extract spatial CLIP features
                spatial_features = self.extract_image_features(rgb, return_spatial=True)
                
                # Get extrinsics
                extrinsics = extrinsics_list[i]
                
                # Back-project to 3D
                points, features, weights = self.backproject_features(
                    spatial_features, depth, intrinsics, extrinsics, (H, W)
                )
                
                if len(points) > 0:
                    all_points.append(points)
                    all_features.append(features)
                    all_weights.append(weights)
                    
            except Exception as e:
                logger.warning(f"Frame {i} failed: {e}")
                continue
        
        if not all_points:
            logger.error("No valid points extracted")
            return np.zeros((0, 3)), np.zeros((0, self.config.feature_dim))
        
        # Fuse multi-view features
        fused_points, fused_features = self.fuse_multiview_features(
            all_points, all_features, all_weights, voxel_size
        )
        
        logger.success(f"Extracted {len(fused_points)} points with {self.config.feature_dim}D features")
        
        return fused_points, fused_features


class PointFeatureIndex:
    """Index for per-point CLIP features with efficient search.
    
    Combines spatial KDTree with FAISS for hybrid queries.
    """
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        
        self.points: Optional[np.ndarray] = None  # (N, 3)
        self.features: Optional[np.ndarray] = None  # (N, D)
        
        self._spatial_tree = None
        self._feature_index = None
    
    def build(self, points: np.ndarray, features: np.ndarray):
        """Build index from points and features.
        
        Args:
            points: (N, 3) 3D point coordinates
            features: (N, D) CLIP features
        """
        from scipy.spatial import KDTree
        
        self.points = np.asarray(points, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)
        
        # Normalize features
        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        self.features = self.features / (norms + 1e-8)
        
        # Build spatial tree
        self._spatial_tree = KDTree(self.points)
        
        # Build feature index (FAISS or numpy)
        try:
            import faiss
            self._feature_index = faiss.IndexFlatIP(self.feature_dim)
            self._feature_index.add(self.features)
        except ImportError:
            self._feature_index = None  # Fall back to numpy
        
        logger.info(f"Built point feature index: {len(self.points)} points, {self.feature_dim}D")
    
    def search_by_text(
        self,
        query_text: str,
        text_encoder: Any,
        top_k: int = 100,
    ) -> List[Tuple[int, float]]:
        """Search for points matching text query.
        
        Args:
            query_text: Text query (e.g., "chair", "wooden surface")
            text_encoder: Function to encode text to CLIP feature
            top_k: Number of results
        
        Returns:
            List of (point_index, similarity_score)
        """
        if self.features is None:
            return []
        
        # Encode query
        query_feat = text_encoder(query_text)
        if query_feat is None:
            return []
        
        query_feat = np.asarray(query_feat, dtype=np.float32).flatten()
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        
        # Search
        if self._feature_index is not None:
            # FAISS search
            query_feat = query_feat.reshape(1, -1)
            scores, indices = self._feature_index.search(query_feat, top_k)
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # Numpy fallback
            similarities = self.features @ query_feat
            top_indices = np.argsort(-similarities)[:top_k]
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def search_by_location(
        self,
        query_point: np.ndarray,
        radius: float = 0.5,
        top_k: int = 100,
    ) -> List[Tuple[int, float]]:
        """Search for points near a location.
        
        Args:
            query_point: (3,) 3D query location
            radius: Search radius in meters
            top_k: Maximum results
        
        Returns:
            List of (point_index, distance)
        """
        if self._spatial_tree is None:
            return []
        
        query_point = np.asarray(query_point).flatten()[:3]
        
        indices = self._spatial_tree.query_ball_point(query_point, radius)
        
        if not indices:
            # Fall back to k-nearest
            dists, indices = self._spatial_tree.query(query_point, k=min(top_k, len(self.points)))
            if isinstance(indices, int):
                return [(indices, float(dists))]
            return [(int(idx), float(d)) for idx, d in zip(indices, dists)]
        
        # Sort by distance
        distances = [np.linalg.norm(self.points[i] - query_point) for i in indices]
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1])
        
        return [(int(idx), float(d)) for idx, d in sorted_pairs[:top_k]]
    
    def search_hybrid(
        self,
        query_text: str,
        query_point: np.ndarray,
        text_encoder: Any,
        radius: float = 1.0,
        top_k: int = 50,
        spatial_weight: float = 0.3,
    ) -> List[Tuple[int, float, float, float]]:
        """Hybrid search combining text and spatial queries.
        
        Args:
            query_text: Text query
            query_point: 3D location
            text_encoder: Text encoder function
            radius: Spatial search radius
            top_k: Results to return
            spatial_weight: Weight for spatial score (0-1)
        
        Returns:
            List of (point_index, combined_score, semantic_score, spatial_score)
        """
        if self.features is None:
            return []
        
        # Get semantic matches
        text_results = self.search_by_text(query_text, text_encoder, top_k=top_k * 3)
        text_scores = {idx: score for idx, score in text_results}
        
        # Get spatial matches
        spatial_results = self.search_by_location(query_point, radius, top_k=top_k * 3)
        max_dist = radius
        spatial_scores = {
            idx: 1.0 - min(dist / max_dist, 1.0) 
            for idx, dist in spatial_results
        }
        
        # Combine candidates
        all_candidates = set(text_scores.keys()) | set(spatial_scores.keys())
        
        results = []
        for idx in all_candidates:
            sem_score = text_scores.get(idx, 0.0)
            spat_score = spatial_scores.get(idx, 0.0)
            combined = (1 - spatial_weight) * sem_score + spatial_weight * spat_score
            results.append((idx, combined, sem_score, spat_score))
        
        # Sort by combined score
        results.sort(key=lambda x: -x[1])
        
        return results[:top_k]
    
    def get_points_by_indices(self, indices: List[int]) -> np.ndarray:
        """Get 3D points by indices."""
        return self.points[indices]
    
    def get_features_by_indices(self, indices: List[int]) -> np.ndarray:
        """Get features by indices."""
        return self.features[indices]


def compute_scene_point_features(
    scene_path: str,
    output_path: Optional[str] = None,
    clip_model: str = "ViT-B/32",
    stride: int = 10,
    voxel_size: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to compute per-point features for a scene.
    
    Args:
        scene_path: Path to scene directory (with results/ folder)
        output_path: Optional path to save features
        clip_model: CLIP model to use
        stride: Frame stride
        voxel_size: Voxel size for fusion
    
    Returns:
        points: (N, 3) 3D points
        features: (N, D) CLIP features
    """
    from pathlib import Path
    import cv2
    
    scene_path = Path(scene_path)
    
    # Find RGB images
    results_dir = scene_path / 'results'
    rgb_paths = sorted(results_dir.glob('frame*.jpg'))
    if not rgb_paths:
        rgb_paths = sorted(results_dir.glob('*.jpg'))
    
    # Find depth images
    depth_paths = sorted(results_dir.glob('depth*.png'))
    if not depth_paths:
        depth_paths = sorted(results_dir.glob('*.npy'))
    
    # Load camera poses
    traj_file = scene_path / 'traj.txt'
    if not traj_file.exists():
        traj_file = results_dir / 'traj.txt'
    
    poses = []
    if traj_file.exists():
        with open(traj_file) as f:
            lines = f.readlines()
        for i in range(0, len(lines), 4):
            if i + 4 <= len(lines):
                pose = np.array([
                    [float(x) for x in lines[i].split()],
                    [float(x) for x in lines[i+1].split()],
                    [float(x) for x in lines[i+2].split()],
                    [float(x) for x in lines[i+3].split()],
                ])
                poses.append(pose)
    
    if not poses:
        logger.error("No camera poses found")
        return np.zeros((0, 3)), np.zeros((0, 512))
    
    # Default intrinsics (adjust for your camera)
    if rgb_paths:
        sample_img = cv2.imread(str(rgb_paths[0]))
        H, W = sample_img.shape[:2]
    else:
        H, W = 480, 640
    
    intrinsics = np.array([
        [600.0, 0, W / 2],
        [0, 600.0, H / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    
    # Extract features
    extractor = PointFeatureExtractor(clip_model=clip_model)
    points, features = extractor.extract_scene_features(
        rgb_paths, depth_paths, intrinsics, poses,
        stride=stride, voxel_size=voxel_size
    )
    
    # Optionally save
    if output_path:
        output_path = Path(output_path)
        np.savez(
            output_path,
            points=points,
            features=features,
        )
        logger.info(f"Saved features to {output_path}")
    
    return points, features
