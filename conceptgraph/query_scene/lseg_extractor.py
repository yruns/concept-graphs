"""
LSeg-based per-pixel feature extraction for OpenScene-style dense point features.

LSeg (Language-driven Semantic Segmentation) produces pixel-level CLIP-aligned
features, enabling much denser point cloud features compared to CLIP's patch-level output.

Reference:
- LSeg Paper: https://arxiv.org/abs/2201.03546
- OpenScene: https://github.com/pengsongyou/openscene
- LSeg Feature Extraction: https://github.com/pengsongyou/lseg_feature_extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class LSegConfig:
    """Configuration for LSeg feature extraction."""
    
    feature_dim: int = 512
    image_size: int = 480  # LSeg default input size
    crop_size: int = 480
    base_size: int = 520
    backbone: str = "clip_vitl16_384"  # or "clip_resnet101"
    use_pretrained: bool = True
    checkpoint_path: Optional[str] = None


class LSegFeatureExtractor:
    """Extract per-pixel CLIP-aligned features using LSeg.
    
    LSeg produces dense feature maps where each pixel has a 512-dim
    feature vector aligned with CLIP's text embedding space.
    """
    
    def __init__(
        self,
        config: Optional[LSegConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or LSegConfig()
        self.device = device
        
        self._model = None
        self._transform = None
        self._clip_model = None  # For text encoding
    
    def _load_model(self):
        """Load LSeg model or CLIP fallback."""
        # Check if already loaded
        if self._model is not None or self._clip_model is not None:
            return
        
        import torch
        
        # Try to load LSeg
        try:
            from .lseg_model import LSegNet
            
            logger.info("Loading LSeg model...")
            
            self._model = LSegNet(
                backbone=self.config.backbone,
                features=256,
                crop_size=self.config.crop_size,
                arch_option=0,
                block_depth=0,
                activation='lrelu',
            )
            
            if self.config.checkpoint_path and Path(self.config.checkpoint_path).exists():
                checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
                self._model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.success(f"Loaded LSeg checkpoint: {self.config.checkpoint_path}")
            
            self._model = self._model.to(self.device)
            self._model.eval()
            self._use_clip_fallback = False
            
        except ImportError:
            logger.warning("LSeg model not available, using CLIP with interpolation fallback")
            self._use_clip_fallback = True
            self._load_clip_fallback()
    
    def _load_clip_fallback(self):
        """Load CLIP as fallback with bilinear interpolation for dense features."""
        import torch
        
        try:
            import clip
            
            logger.info("Loading CLIP ViT-L/14 for dense feature extraction (fallback)")
            self._clip_model, self._clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self._clip_model.eval()
            self.config.feature_dim = 768  # ViT-L
            logger.success("CLIP ViT-L/14 loaded as fallback")
            
        except Exception as e:
            logger.warning(f"ViT-L/14 failed: {e}, trying ViT-B/16")
            import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/16", device=self.device)
            self._clip_model.eval()
            self.config.feature_dim = 512
            logger.success("CLIP ViT-B/16 loaded as fallback")
    
    def extract_pixel_features(
        self,
        image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Extract per-pixel features from an image.
        
        Args:
            image: RGB image (H, W, 3) uint8 or float
            output_size: Optional (H', W') for output feature map size.
                        If None, uses original image size.
        
        Returns:
            features: (H', W', D) per-pixel feature map
        """
        self._load_model()
        
        import torch
        import torch.nn.functional as F
        from PIL import Image
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        H, W = image.shape[:2]
        output_size = output_size or (H, W)
        
        if getattr(self, '_use_clip_fallback', False):
            return self._extract_with_clip_interpolation(image, output_size)
        
        # LSeg extraction
        pil_image = Image.fromarray(image)
        
        # Transform
        input_tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self._model(input_tensor)  # (1, 512, H', W')
        
        # Resize to output size
        features = F.interpolate(
            features, size=output_size, mode='bilinear', align_corners=False
        )
        
        # Normalize
        features = F.normalize(features, dim=1)
        
        # (1, D, H, W) -> (H, W, D)
        features = features[0].permute(1, 2, 0).cpu().numpy()
        
        return features.astype(np.float32)
    
    def _extract_with_clip_interpolation(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Extract dense features using CLIP with bilinear interpolation.
        
        This is a fallback when LSeg is not available. It extracts CLIP patch
        features and upsamples them to pixel resolution.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import cv2
        
        # CLIP expects 224x224 input, resize image first
        # This ensures the positional embeddings match
        clip_size = 224
        image_resized = cv2.resize(image, (clip_size, clip_size))
        
        pil_image = Image.fromarray(image_resized)
        input_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get spatial features from CLIP ViT
            features = self._extract_clip_spatial(input_tensor)  # (1, D, H', W')
        
        # Upsample to output size
        features = F.interpolate(
            features, size=output_size, mode='bilinear', align_corners=False
        )
        
        # Normalize
        features = F.normalize(features, dim=1)
        
        # (1, D, H, W) -> (H, W, D)
        features = features[0].permute(1, 2, 0).cpu().numpy()
        
        return features.astype(np.float32)
    
    def _extract_clip_spatial(self, input_tensor) -> 'torch.Tensor':
        """Extract spatial features from CLIP ViT.
        
        Extracts per-patch features from CLIP's vision transformer,
        which can then be upsampled for dense pixel-level features.
        """
        import torch
        
        visual = self._clip_model.visual
        dtype = visual.conv1.weight.dtype
        device = next(visual.parameters()).device
        
        input_tensor = input_tensor.to(dtype=dtype, device=device)
        
        # Patch embedding: conv1 with kernel_size=patch_size, stride=patch_size
        # For ViT-L/14: patch_size=14, so 224/14 = 16 patches per dimension
        x = visual.conv1(input_tensor)  # (B, D_hidden, H', W')
        B, D_hidden, H, W = x.shape
        N = H * W  # Number of patches
        
        x = x.reshape(B, D_hidden, -1).permute(0, 2, 1)  # (B, N, D_hidden)
        
        # Add class token
        cls_token = visual.class_embedding.to(dtype).expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D_hidden)
        
        # Positional embedding - check dimensions match
        pos_embed = visual.positional_embedding.to(dtype)
        if pos_embed.shape[0] != x.shape[1]:
            # Need to interpolate positional embedding
            # Original: (257, D) for 16x16 patches + CLS
            # We need: (N+1, D)
            pos_embed = pos_embed.unsqueeze(0)  # (1, 257, D)
            cls_pos = pos_embed[:, :1, :]  # CLS token position
            patch_pos = pos_embed[:, 1:, :]  # Patch positions (1, 256, D)
            
            # Reshape to spatial and interpolate
            orig_size = int(patch_pos.shape[1] ** 0.5)  # 16
            patch_pos = patch_pos.permute(0, 2, 1).reshape(1, -1, orig_size, orig_size)
            patch_pos = torch.nn.functional.interpolate(
                patch_pos, size=(H, W), mode='bilinear', align_corners=False
            )
            patch_pos = patch_pos.reshape(1, -1, N).permute(0, 2, 1)  # (1, N, D)
            
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)  # (1, N+1, D)
            x = x + pos_embed
        else:
            x = x + pos_embed
        
        # Transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (N+1, B, D)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, N+1, D)
        
        # Remove CLS, keep patches
        patch_features = x[:, 1:, :]  # (B, N, D_hidden)
        patch_features = visual.ln_post(patch_features)
        
        # Apply projection to output dimension
        if hasattr(visual, 'proj') and visual.proj is not None:
            proj = visual.proj.to(dtype)
            patch_features = patch_features @ proj  # (B, N, D_out)
        
        D_out = patch_features.shape[-1]
        
        # Reshape to spatial: (B, N, D) -> (B, D, H, W)
        patch_features = patch_features.permute(0, 2, 1).reshape(B, D_out, H, W)
        
        return patch_features.float()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to CLIP feature space."""
        import torch
        
        self._load_model()
        
        if self._clip_model is None:
            # Load CLIP for text encoding
            try:
                import clip
                self._clip_model, _ = clip.load("ViT-L/14", device=self.device)
            except:
                import clip
                self._clip_model, _ = clip.load("ViT-B/16", device=self.device)
            self._clip_model.eval()
        
        import clip
        tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            feat = self._clip_model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        
        return feat.cpu().numpy().flatten().astype(np.float32)


class DensePointFeatureExtractor:
    """Extract dense per-point features using LSeg pixel features.
    
    This is the main class for OpenScene-style feature extraction:
    1. Extract per-pixel features from each frame using LSeg
    2. Back-project each pixel to 3D using depth
    3. Fuse multi-view features with weighted averaging
    """
    
    def __init__(
        self,
        config: Optional[LSegConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or LSegConfig()
        self.device = device
        self.lseg = LSegFeatureExtractor(config, device)
        
        # Fusion parameters
        self.depth_min = 0.1
        self.depth_max = 10.0
        self.voxel_size = 0.02  # 2cm voxels for aggregation
    
    def extract_scene_features(
        self,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        intrinsics: np.ndarray,
        extrinsics_list: List[np.ndarray],
        stride: int = 5,
        downsample: int = 4,  # Downsample image for faster processing
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-point features for entire scene.
        
        Args:
            rgb_paths: Paths to RGB images
            depth_paths: Paths to depth images  
            intrinsics: (3, 3) camera intrinsic matrix
            extrinsics_list: List of (4, 4) camera-to-world transforms
            stride: Process every N-th frame
            downsample: Downsample factor for feature extraction
        
        Returns:
            points: (N, 3) fused 3D points
            features: (N, D) per-point features
        """
        import cv2
        
        n_frames = min(len(rgb_paths), len(depth_paths), len(extrinsics_list))
        logger.info(f"Extracting dense features from {n_frames} frames (stride={stride})")
        
        all_points = []
        all_features = []
        all_weights = []
        
        # Scale intrinsics for downsampled images
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[0, 0] /= downsample
        scaled_intrinsics[1, 1] /= downsample
        scaled_intrinsics[0, 2] /= downsample
        scaled_intrinsics[1, 2] /= downsample
        
        for i in range(0, n_frames, stride):
            if i % 50 == 0:
                logger.info(f"  Processing frame {i}/{n_frames}")
            
            try:
                # Load RGB
                rgb = cv2.imread(str(rgb_paths[i]))
                if rgb is None:
                    continue
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
                # Load depth
                depth = self._load_depth(depth_paths[i])
                if depth is None:
                    continue
                
                # Downsample for efficiency
                H, W = rgb.shape[:2]
                new_H, new_W = H // downsample, W // downsample
                
                rgb_small = cv2.resize(rgb, (new_W, new_H))
                depth_small = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
                
                # Extract pixel features
                pixel_features = self.lseg.extract_pixel_features(rgb_small, (new_H, new_W))
                
                # Back-project to 3D
                extrinsics = extrinsics_list[i]
                points, features, weights = self._backproject_pixels(
                    pixel_features, depth_small, scaled_intrinsics, extrinsics
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
        fused_points, fused_features = self._fuse_multiview(
            all_points, all_features, all_weights
        )
        
        logger.success(f"Extracted {len(fused_points)} points with {fused_features.shape[1]}D features")
        
        return fused_points, fused_features
    
    def _load_depth(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth image and convert to meters."""
        import cv2
        
        depth_path = Path(depth_path)
        
        if depth_path.suffix == '.npy':
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                # Assume 16-bit depth in mm
                depth = depth.astype(np.float32) / 1000.0
        
        return depth
    
    def _backproject_pixels(
        self,
        pixel_features: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back-project pixel features to 3D.
        
        Args:
            pixel_features: (H, W, D) per-pixel feature map
            depth: (H, W) depth in meters
            intrinsics: (3, 3) camera matrix
            extrinsics: (4, 4) camera-to-world transform
        
        Returns:
            points: (N, 3) valid 3D points
            features: (N, D) corresponding features
            weights: (N,) confidence weights
        """
        H, W, D = pixel_features.shape
        
        # Create pixel grid
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        
        # Get valid depth mask
        valid = (depth > self.depth_min) & (depth < self.depth_max)
        
        # Get valid pixels
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth[valid]
        
        if len(z_valid) == 0:
            return np.zeros((0, 3)), np.zeros((0, D)), np.zeros(0)
        
        # Back-project to camera coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        
        # Stack to (N, 4) homogeneous coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        
        # Transform to world coordinates
        points_world = (extrinsics @ points_cam.T).T[:, :3]
        
        # Get features for valid pixels
        features = pixel_features[valid]
        
        # Compute weights (closer = higher weight)
        weights = 1.0 / (1.0 + z_valid / 5.0)
        
        return points_world.astype(np.float32), features.astype(np.float32), weights.astype(np.float32)
    
    def _fuse_multiview(
        self,
        all_points: List[np.ndarray],
        all_features: List[np.ndarray],
        all_weights: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse features from multiple views using voxel aggregation."""
        
        # Concatenate all data
        concat_points = np.vstack(all_points)
        concat_features = np.vstack(all_features)
        concat_weights = np.concatenate(all_weights)
        
        logger.debug(f"Fusing {len(concat_points)} points from {len(all_points)} views")
        
        # Voxelize
        voxel_indices = np.floor(concat_points / self.voxel_size).astype(np.int32)
        
        # Create unique voxel keys
        voxel_keys = (
            voxel_indices[:, 0].astype(np.int64) * 1000000000 +
            voxel_indices[:, 1].astype(np.int64) * 1000000 +
            voxel_indices[:, 2].astype(np.int64)
        )
        
        unique_voxels, inverse_indices = np.unique(voxel_keys, return_inverse=True)
        
        n_voxels = len(unique_voxels)
        D = concat_features.shape[1]
        
        # Aggregate
        fused_features = np.zeros((n_voxels, D), dtype=np.float64)
        fused_positions = np.zeros((n_voxels, 3), dtype=np.float64)
        voxel_weights = np.zeros(n_voxels, dtype=np.float64)
        
        # Use numpy bincount for efficient aggregation
        for d in range(D):
            np.add.at(fused_features[:, d], inverse_indices, concat_weights * concat_features[:, d])
        
        for dim in range(3):
            np.add.at(fused_positions[:, dim], inverse_indices, concat_weights * concat_points[:, dim])
        
        np.add.at(voxel_weights, inverse_indices, concat_weights)
        
        # Normalize
        valid = voxel_weights > 0
        fused_features[valid] /= voxel_weights[valid, np.newaxis]
        fused_positions[valid] /= voxel_weights[valid, np.newaxis]
        
        # L2 normalize features
        norms = np.linalg.norm(fused_features, axis=1, keepdims=True)
        fused_features = fused_features / (norms + 1e-8)
        
        logger.debug(f"Fused to {valid.sum()} voxels")
        
        return fused_positions[valid].astype(np.float32), fused_features[valid].astype(np.float32)


def extract_dense_scene_features(
    scene_path: str,
    output_path: Optional[str] = None,
    stride: int = 5,
    downsample: int = 4,
    voxel_size: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to extract dense features for a scene.
    
    Args:
        scene_path: Path to scene directory
        output_path: Optional path to save features
        stride: Frame stride
        downsample: Image downsample factor
        voxel_size: Voxel size for fusion
    
    Returns:
        points: (N, 3) 3D points
        features: (N, D) per-point features
    """
    import cv2
    
    scene_path = Path(scene_path)
    
    # Find images
    results_dir = scene_path / 'results'
    rgb_paths = sorted(results_dir.glob('frame*.jpg'))
    if not rgb_paths:
        rgb_paths = sorted(results_dir.glob('*.jpg'))
    
    depth_paths = sorted(results_dir.glob('depth*.png'))
    
    # Load poses
    traj_file = scene_path / 'traj.txt'
    poses = []
    if traj_file.exists():
        with open(traj_file) as f:
            lines = f.readlines()
        
        # Check format: each line could be a full 4x4 matrix (16 numbers)
        first_line_nums = len(lines[0].split()) if lines else 0
        
        if first_line_nums == 16:
            # Each line is a full 4x4 matrix
            for line in lines:
                nums = [float(x) for x in line.split()]
                if len(nums) == 16:
                    pose = np.array(nums).reshape(4, 4)
                    poses.append(pose)
        else:
            # Traditional format: 4 lines per matrix
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
    
    # Default intrinsics
    if rgb_paths:
        sample_img = cv2.imread(str(rgb_paths[0]))
        H, W = sample_img.shape[:2]
    else:
        H, W = 480, 640
    
    # Replica default intrinsics
    intrinsics = np.array([
        [600.0, 0, W / 2],
        [0, 600.0, H / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    
    # Extract features
    extractor = DensePointFeatureExtractor()
    extractor.voxel_size = voxel_size
    
    points, features = extractor.extract_scene_features(
        rgb_paths, depth_paths, intrinsics, poses,
        stride=stride, downsample=downsample
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        np.savez(output_path, points=points, features=features)
        logger.info(f"Saved dense features to {output_path}")
    
    return points, features
