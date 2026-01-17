"""
信号提取器模块

从轨迹和检测数据中提取用于场景分段的多模态信号
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d


class MotionSignalExtractor:
    """运动模态信号提取器"""
    
    def __init__(self, smooth_sigma: float = 2.0):
        self.smooth_sigma = smooth_sigma
    
    def extract(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        n_frames = len(poses)
        positions = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        
        position_diff = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        position_diff = np.concatenate([[0], position_diff])
        angle_diff = self._compute_rotation_diff(rotations)
        
        position_diff_smooth = gaussian_filter1d(position_diff, self.smooth_sigma)
        angle_diff_smooth = gaussian_filter1d(angle_diff, self.smooth_sigma)
        
        pos_norm = position_diff_smooth / (position_diff_smooth.max() + 1e-8)
        ang_norm = angle_diff_smooth / (angle_diff_smooth.max() + 1e-8)
        motion_intensity = 0.7 * pos_norm + 0.3 * ang_norm
        exploration_score = ang_norm / (pos_norm + 0.1)
        
        return {
            'position_diff': position_diff,
            'angle_diff': angle_diff,
            'position_diff_smooth': position_diff_smooth,
            'angle_diff_smooth': angle_diff_smooth,
            'motion_intensity': motion_intensity,
            'exploration_score': exploration_score,
            'positions': positions,
        }
    
    def _compute_rotation_diff(self, rotations: np.ndarray) -> np.ndarray:
        n_frames = len(rotations)
        angle_diffs = [0.0]
        for i in range(1, n_frames):
            R_rel = rotations[i-1].T @ rotations[i]
            try:
                rot = Rotation.from_matrix(R_rel)
                angle = rot.magnitude()
            except:
                angle = 0.0
            angle_diffs.append(angle)
        return np.array(angle_diffs)


class VisualSignalExtractor:
    """视觉模态信号提取器"""
    
    def __init__(self, smooth_sigma: float = 2.0):
        self.smooth_sigma = smooth_sigma
    
    def extract(self, clip_features: np.ndarray) -> Dict[str, np.ndarray]:
        n_frames = len(clip_features)
        norms = np.linalg.norm(clip_features, axis=1, keepdims=True)
        features_norm = clip_features / (norms + 1e-8)
        
        similarities = [1.0]
        for i in range(1, n_frames):
            sim = np.dot(features_norm[i-1], features_norm[i])
            similarities.append(float(sim))
        similarities = np.array(similarities)
        
        visual_change = 1 - similarities
        visual_change_smooth = gaussian_filter1d(visual_change, self.smooth_sigma)
        window_consistency = self._compute_window_consistency(features_norm, 5)
        
        return {
            'frame_similarity': similarities,
            'visual_change': visual_change,
            'visual_change_smooth': visual_change_smooth,
            'window_consistency': window_consistency,
        }
    
    def _compute_window_consistency(self, features: np.ndarray, window_size: int) -> np.ndarray:
        n_frames = len(features)
        consistency = np.zeros(n_frames)
        half_window = window_size // 2
        for i in range(n_frames):
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            window_features = features[start:end]
            if len(window_features) > 1:
                sims = []
                for j in range(len(window_features)):
                    for k in range(j + 1, len(window_features)):
                        sim = np.dot(window_features[j], window_features[k])
                        sims.append(sim)
                consistency[i] = np.mean(sims)
            else:
                consistency[i] = 1.0
        return consistency


class SemanticSignalExtractor:
    """语义模态信号提取器"""
    
    BOUNDARY_OBJECTS = ['door', 'doorframe', 'doorway', 'gate', 'entrance', 
                        'wall', 'partition', 'divider', 'archway']
    
    def __init__(self, smooth_sigma: float = 3.0):
        self.smooth_sigma = smooth_sigma
    
    def extract(self, frame_detections: List[Dict]) -> Dict[str, np.ndarray]:
        n_frames = len(frame_detections)
        
        detection_counts = np.array([
            len(det.get('class_names', [])) or det.get('n_masks', 0)
            for det in frame_detections
        ])
        
        seen_objects = set()
        new_object_counts = []
        cumulative_objects = []
        
        for det in frame_detections:
            class_names = det.get('class_names', [])
            if not class_names:
                new_count = det.get('n_masks', 0)
            else:
                new_count = 0
                for name in class_names:
                    if name not in seen_objects:
                        seen_objects.add(name)
                        new_count += 1
            new_object_counts.append(new_count)
            cumulative_objects.append(len(seen_objects))
        
        new_object_counts = np.array(new_object_counts)
        cumulative_objects = np.array(cumulative_objects)
        
        boundary_signals = []
        for det in frame_detections:
            class_names = det.get('class_names', [])
            has_boundary = any(
                any(b in name.lower() for b in self.BOUNDARY_OBJECTS)
                for name in class_names
            )
            boundary_signals.append(1.0 if has_boundary else 0.0)
        boundary_signals = np.array(boundary_signals)
        
        new_object_smooth = gaussian_filter1d(new_object_counts.astype(float), self.smooth_sigma)
        semantic_change = new_object_smooth / (new_object_smooth.max() + 1e-8) + 0.5 * boundary_signals
        
        return {
            'detection_counts': detection_counts,
            'new_object_counts': new_object_counts,
            'cumulative_objects': cumulative_objects,
            'boundary_signals': boundary_signals,
            'new_object_smooth': new_object_smooth,
            'semantic_change': semantic_change,
        }


class MultiModalSignalFusion:
    """多模态信号融合器"""
    
    def __init__(self, motion_weight: float = 0.4, visual_weight: float = 0.4, semantic_weight: float = 0.2):
        self.motion_weight = motion_weight
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
    
    def fuse(self, motion_signals, visual_signals=None, semantic_signals=None):
        motion_change = motion_signals['motion_intensity']
        motion_change_norm = self._normalize(motion_change)
        
        total_weight = self.motion_weight
        signals = [motion_change_norm * self.motion_weight]
        
        if visual_signals is not None:
            visual_change = visual_signals['visual_change_smooth']
            visual_change_norm = self._normalize(visual_change)
            signals.append(visual_change_norm * self.visual_weight)
            total_weight += self.visual_weight
        
        if semantic_signals is not None:
            semantic_change = semantic_signals['semantic_change']
            semantic_change_norm = self._normalize(semantic_change)
            signals.append(semantic_change_norm * self.semantic_weight)
            total_weight += self.semantic_weight
        
        fused_signal = sum(signals) / total_weight
        change_point_score = self._compute_change_point_score(fused_signal)
        
        return {
            'fused_signal': fused_signal,
            'change_point_score': change_point_score,
            'motion_contribution': motion_change_norm,
            'visual_contribution': visual_signals['visual_change_smooth'] if visual_signals else None,
            'semantic_contribution': semantic_signals['semantic_change'] if semantic_signals else None,
        }
    
    def _normalize(self, signal):
        min_val = signal.min()
        max_val = signal.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(signal)
        return (signal - min_val) / (max_val - min_val)
    
    def _compute_change_point_score(self, signal, window=10):
        grad1 = np.gradient(signal)
        grad2 = np.gradient(grad1)
        change_score = np.abs(grad1) + 0.5 * np.abs(grad2)
        return change_score

class VisualSignalExtractor:
    def __init__(self, smooth_sigma: float = 2.0):
        self.smooth_sigma = smooth_sigma
    
    def extract(self, clip_features: np.ndarray) -> Dict[str, np.ndarray]:
        norms = np.linalg.norm(clip_features, axis=1, keepdims=True)
        features_norm = clip_features / (norms + 1e-8)
        similarities = [1.0]
        for i in range(1, len(clip_features)):
            similarities.append(float(np.dot(features_norm[i-1], features_norm[i])))
        similarities = np.array(similarities)
        visual_change = 1 - similarities
        visual_change_smooth = gaussian_filter1d(visual_change, self.smooth_sigma)
        return {"frame_similarity": similarities, "visual_change": visual_change, "visual_change_smooth": visual_change_smooth}

class SemanticSignalExtractor:
    BOUNDARY_OBJECTS = ["door", "doorframe", "wall", "partition"]
    def __init__(self, smooth_sigma: float = 3.0):
        self.smooth_sigma = smooth_sigma
    
    def extract(self, frame_detections: List[Dict]) -> Dict[str, np.ndarray]:
        detection_counts = np.array([len(det.get("class_names", [])) or det.get("n_masks", 0) for det in frame_detections])
        seen, new_counts, cumulative = set(), [], []
        for det in frame_detections:
            names = det.get("class_names", [])
            new_count = sum(1 for n in names if n not in seen and not seen.add(n)) if names else det.get("n_masks", 0)
            new_counts.append(new_count)
            cumulative.append(len(seen))
        new_counts = np.array(new_counts)
        new_smooth = gaussian_filter1d(new_counts.astype(float), self.smooth_sigma)
        semantic_change = new_smooth / (new_smooth.max() + 1e-8)
        return {"detection_counts": detection_counts, "new_object_counts": new_counts, "semantic_change": semantic_change}

