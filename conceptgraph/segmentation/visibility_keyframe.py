#!/usr/bin/env python3
"""
基于可见性的关键帧选取
====================

关键帧选取策略：
1. 物体可见性变化：检测物体进入/离开视野的帧
2. 稳定片段中心：在可见性稳定的片段中选择中心帧
3. 覆盖最大化：确保关键帧覆盖尽可能多的物体
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import gzip
import pickle


@dataclass
class KeyframeInfo:
    """关键帧信息"""
    frame_idx: int                       # 帧索引
    original_frame_idx: int              # 原始视频帧索引（stride之前）
    selection_reason: str                # 选取原因
    visible_objects: List[int]           # 可见物体ID列表
    visibility_change_score: float       # 可见性变化分数
    stability_score: float               # 稳定性分数
    coverage_score: float                # 覆盖度分数
    
    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "original_frame_idx": self.original_frame_idx,
            "selection_reason": self.selection_reason,
            "visible_objects": self.visible_objects,
            "visibility_change_score": self.visibility_change_score,
            "stability_score": self.stability_score,
            "coverage_score": self.coverage_score
        }


class VisibilityBasedKeyframeSelector:
    """基于可见性变化的关键帧选取器"""
    
    def __init__(
        self, 
        visibility_matrix: np.ndarray,
        n_keyframes: int = 15,
        stride: int = 5,
        min_stability_length: int = 10
    ):
        """
        Args:
            visibility_matrix: 可见性矩阵 [n_frames, n_objects]，值为0/1
            n_keyframes: 目标关键帧数量
            stride: 帧采样步长
            min_stability_length: 最小稳定片段长度
        """
        self.visibility_matrix = visibility_matrix
        self.n_keyframes = n_keyframes
        self.stride = stride
        self.min_stability_length = min_stability_length
        
        self.n_frames = visibility_matrix.shape[0]
        self.n_objects = visibility_matrix.shape[1]
        
        # 计算可见性变化
        self.visibility_changes = self._compute_visibility_changes()
    
    def _compute_visibility_changes(self) -> np.ndarray:
        """计算相邻帧间的可见性变化"""
        if self.n_frames < 2:
            return np.zeros(self.n_frames)
        
        # 使用Jaccard距离计算变化
        changes = np.zeros(self.n_frames)
        for i in range(1, self.n_frames):
            prev_visible = set(np.where(self.visibility_matrix[i-1] > 0)[0])
            curr_visible = set(np.where(self.visibility_matrix[i] > 0)[0])
            
            if len(prev_visible) == 0 and len(curr_visible) == 0:
                changes[i] = 0
            else:
                union = prev_visible | curr_visible
                intersection = prev_visible & curr_visible
                if len(union) > 0:
                    changes[i] = 1 - len(intersection) / len(union)
                else:
                    changes[i] = 0
        
        return changes
    
    def select(self) -> List[KeyframeInfo]:
        """
        选取关键帧
        
        Returns:
            List[KeyframeInfo]: 关键帧列表
        """
        keyframes = []
        
        # 策略1: 在可见性变化点选取关键帧
        change_keyframes = self._select_at_change_points()
        keyframes.extend(change_keyframes)
        
        # 策略2: 在稳定片段中心选取关键帧
        stable_keyframes = self._select_from_stable_segments()
        keyframes.extend(stable_keyframes)
        
        # 策略3: 补充以最大化覆盖
        coverage_keyframes = self._select_for_coverage(keyframes)
        keyframes.extend(coverage_keyframes)
        
        # 去重并排序
        unique_frames = {}
        for kf in keyframes:
            if kf.frame_idx not in unique_frames:
                unique_frames[kf.frame_idx] = kf
            else:
                # 保留综合分数更高的
                existing = unique_frames[kf.frame_idx]
                new_score = kf.visibility_change_score + kf.stability_score + kf.coverage_score
                old_score = existing.visibility_change_score + existing.stability_score + existing.coverage_score
                if new_score > old_score:
                    unique_frames[kf.frame_idx] = kf
        
        keyframes = sorted(unique_frames.values(), key=lambda x: x.frame_idx)
        
        # 如果超过目标数量，选择最重要的
        if len(keyframes) > self.n_keyframes:
            # 按综合分数排序
            keyframes = sorted(
                keyframes, 
                key=lambda x: x.visibility_change_score + x.stability_score + x.coverage_score,
                reverse=True
            )[:self.n_keyframes]
            keyframes = sorted(keyframes, key=lambda x: x.frame_idx)
        
        return keyframes
    
    def _select_at_change_points(self, top_k: int = None) -> List[KeyframeInfo]:
        """在可见性变化点选取关键帧"""
        top_k = top_k or self.n_keyframes // 2
        
        # 找到变化最大的帧
        change_indices = np.argsort(self.visibility_changes)[::-1]
        
        keyframes = []
        for idx in change_indices[:top_k * 2]:  # 多取一些，后面会过滤
            if self.visibility_changes[idx] < 0.1:  # 变化太小
                continue
            
            visible = list(np.where(self.visibility_matrix[idx] > 0)[0])
            
            kf = KeyframeInfo(
                frame_idx=int(idx),
                original_frame_idx=int(idx * self.stride),
                selection_reason="visibility_change",
                visible_objects=visible,
                visibility_change_score=float(self.visibility_changes[idx]),
                stability_score=0.0,
                coverage_score=len(visible) / max(self.n_objects, 1)
            )
            keyframes.append(kf)
            
            if len(keyframes) >= top_k:
                break
        
        return keyframes
    
    def _select_from_stable_segments(self) -> List[KeyframeInfo]:
        """从稳定片段中选取关键帧"""
        # 找到稳定片段（可见性变化小的连续区间）
        stable_segments = self._find_stable_segments()
        
        keyframes = []
        for start, end in stable_segments:
            if end - start < self.min_stability_length:
                continue
            
            # 选择片段中心
            center = (start + end) // 2
            
            visible = list(np.where(self.visibility_matrix[center] > 0)[0])
            
            # 计算稳定性分数（片段内的平均变化）
            segment_changes = self.visibility_changes[start:end+1]
            stability_score = 1.0 - np.mean(segment_changes) if len(segment_changes) > 0 else 0.0
            
            kf = KeyframeInfo(
                frame_idx=int(center),
                original_frame_idx=int(center * self.stride),
                selection_reason="stable_segment_center",
                visible_objects=visible,
                visibility_change_score=0.0,
                stability_score=float(stability_score),
                coverage_score=len(visible) / max(self.n_objects, 1)
            )
            keyframes.append(kf)
        
        return keyframes
    
    def _find_stable_segments(self, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """找到可见性稳定的片段"""
        segments = []
        start = None
        
        for i in range(self.n_frames):
            if self.visibility_changes[i] < threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= self.min_stability_length:
                        segments.append((start, i - 1))
                    start = None
        
        # 处理末尾
        if start is not None and self.n_frames - start >= self.min_stability_length:
            segments.append((start, self.n_frames - 1))
        
        return segments
    
    def _select_for_coverage(self, existing_keyframes: List[KeyframeInfo]) -> List[KeyframeInfo]:
        """补充关键帧以最大化物体覆盖"""
        # 统计已覆盖的物体
        covered_objects = set()
        for kf in existing_keyframes:
            covered_objects.update(kf.visible_objects)
        
        # 找到未覆盖的物体
        all_objects = set(range(self.n_objects))
        uncovered = all_objects - covered_objects
        
        if not uncovered:
            return []
        
        keyframes = []
        
        # 贪心选择覆盖最多未覆盖物体的帧
        remaining = self.n_keyframes - len(existing_keyframes)
        for _ in range(remaining):
            if not uncovered:
                break
            
            best_frame = None
            best_coverage = 0
            
            for i in range(self.n_frames):
                visible = set(np.where(self.visibility_matrix[i] > 0)[0])
                new_coverage = len(visible & uncovered)
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_frame = i
            
            if best_frame is not None and best_coverage > 0:
                visible = list(np.where(self.visibility_matrix[best_frame] > 0)[0])
                
                kf = KeyframeInfo(
                    frame_idx=int(best_frame),
                    original_frame_idx=int(best_frame * self.stride),
                    selection_reason="coverage_maximization",
                    visible_objects=visible,
                    visibility_change_score=0.0,
                    stability_score=0.5,
                    coverage_score=len(visible) / max(self.n_objects, 1)
                )
                keyframes.append(kf)
                
                # 更新已覆盖物体
                covered_objects.update(visible)
                uncovered = all_objects - covered_objects
        
        return keyframes


def build_visibility_matrix(
    objects: List[Dict],
    poses: np.ndarray,
    visibility_radius: float = 3.0
) -> np.ndarray:
    """
    构建物体可见性矩阵
    
    Args:
        objects: 物体列表
        poses: 相机位姿 [n_frames, 7] (tx, ty, tz, qw, qx, qy, qz)
        visibility_radius: 可见性半径
        
    Returns:
        visibility_matrix: [n_frames, n_objects]
    """
    n_frames = len(poses)
    n_objects = len(objects)
    
    visibility_matrix = np.zeros((n_frames, n_objects), dtype=np.float32)
    
    # 获取物体中心
    object_centers = []
    for obj in objects:
        if 'pcd_np' in obj and len(obj['pcd_np']) > 0:
            center = np.mean(obj['pcd_np'], axis=0)
        elif 'bbox_np' in obj and len(obj['bbox_np']) > 0:
            center = np.mean(obj['bbox_np'], axis=0)
        else:
            center = np.zeros(3)
        object_centers.append(center)
    
    object_centers = np.array(object_centers)
    
    # 计算每帧的可见物体
    for i in range(n_frames):
        camera_pos = poses[i, :3]  # 假设前3个是位置
        
        # 计算到各物体的距离
        distances = np.linalg.norm(object_centers - camera_pos, axis=1)
        
        # 在可见性半径内的物体标记为可见
        visible = distances < visibility_radius
        visibility_matrix[i] = visible.astype(np.float32)
    
    return visibility_matrix


def select_keyframes_from_scene(
    scene_path: str,
    n_keyframes: int = 15,
    stride: int = 5,
    visibility_radius: float = 3.0
) -> List[KeyframeInfo]:
    """
    从场景数据中选取关键帧
    
    Args:
        scene_path: 场景路径
        n_keyframes: 关键帧数量
        stride: 帧步长
        visibility_radius: 可见性半径
        
    Returns:
        List[KeyframeInfo]: 关键帧列表
    """
    scene_path = Path(scene_path)
    
    # 加载位姿
    pose_file = scene_path / 'traj.txt'
    if not pose_file.exists():
        pose_file = scene_path / 'traj_w_c.txt'
    
    poses = []
    with open(pose_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                poses.append([float(p) for p in parts[:7]])
    
    poses = np.array(poses[::stride])
    
    # 加载物体
    pcd_files = list((scene_path / 'pcd_saves').glob('*_post.pkl.gz'))
    if not pcd_files:
        pcd_files = list((scene_path / 'pcd_saves').glob('*.pkl.gz'))
    
    with gzip.open(pcd_files[0], 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    
    # 构建可见性矩阵
    visibility_matrix = build_visibility_matrix(objects, poses, visibility_radius)
    
    # 选取关键帧
    selector = VisibilityBasedKeyframeSelector(
        visibility_matrix=visibility_matrix,
        n_keyframes=n_keyframes,
        stride=stride
    )
    
    return selector.select()


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--n_keyframes", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    keyframes = select_keyframes_from_scene(
        args.scene_path,
        n_keyframes=args.n_keyframes,
        stride=args.stride
    )
    
    print(f"选取了 {len(keyframes)} 个关键帧:")
    print("=" * 60)
    
    for kf in keyframes:
        print(f"帧 {kf.frame_idx:4d} (原始: {kf.original_frame_idx:5d})")
        print(f"  原因: {kf.selection_reason}")
        print(f"  可见物体: {len(kf.visible_objects)} 个")
        print(f"  变化分数: {kf.visibility_change_score:.3f}")
        print(f"  稳定分数: {kf.stability_score:.3f}")
        print(f"  覆盖分数: {kf.coverage_score:.3f}")
        print()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump([kf.to_dict() for kf in keyframes], f, indent=2)
        print(f"保存到: {args.output}")
