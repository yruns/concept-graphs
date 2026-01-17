#!/usr/bin/env python3
"""
增强版时序场景分段器

融合三种模态信号:
1. 运动信号 - 位姿变化
2. 视觉信号 - CLIP特征相似度变化
3. 语义信号 - 物体检测数量和分布变化
"""

import numpy as np
import gzip
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from natsort import natsorted
import argparse


@dataclass
class RegionSegment:
    """区域分段数据结构"""
    region_id: int
    start_frame: int
    end_frame: int
    object_indices: List[int] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    n_objects_first_seen: int = 0
    dominant_objects: List[str] = field(default_factory=list)
    
    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame
    
    def to_dict(self) -> Dict:
        return {
            'region_id': self.region_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'n_frames': self.n_frames,
            'object_indices': self.object_indices,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'n_objects_first_seen': self.n_objects_first_seen,
            'dominant_objects': self.dominant_objects,
        }


class EnhancedTrajectorySegmenter:
    """
    增强版轨迹分段器
    
    融合运动、视觉、语义三种模态信号
    """
    
    # 边界物体关键词
    BOUNDARY_OBJECTS = ['door', 'doorframe', 'doorway', 'gate', 'entrance', 
                        'wall', 'partition', 'archway', 'threshold']
    
    def __init__(
        self,
        motion_weight: float = 0.3,
        visual_weight: float = 0.4,
        semantic_weight: float = 0.3,
        min_segment_frames: int = 10,
        peak_prominence: float = 0.1,
        peak_distance: int = 15,
        smooth_sigma: float = 2.0,
    ):
        self.motion_weight = motion_weight
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
        self.min_segment_frames = min_segment_frames
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
        self.smooth_sigma = smooth_sigma
    
    def extract_motion_signals(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        """提取运动信号"""
        positions = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        
        # 位置变化
        position_diff = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        position_diff = np.concatenate([[0], position_diff])
        
        # 角度变化
        angle_diff = [0.0]
        for i in range(1, len(rotations)):
            R_rel = rotations[i-1].T @ rotations[i]
            try:
                angle = Rotation.from_matrix(R_rel).magnitude()
            except:
                angle = 0.0
            angle_diff.append(angle)
        angle_diff = np.array(angle_diff)
        
        # 平滑
        position_smooth = gaussian_filter1d(position_diff, self.smooth_sigma)
        angle_smooth = gaussian_filter1d(angle_diff, self.smooth_sigma)
        
        # 归一化并融合
        pos_norm = position_smooth / (position_smooth.max() + 1e-8)
        ang_norm = angle_smooth / (angle_smooth.max() + 1e-8)
        motion_intensity = 0.7 * pos_norm + 0.3 * ang_norm
        
        return {
            'position_diff': position_diff,
            'angle_diff': angle_diff,
            'position_smooth': position_smooth,
            'angle_smooth': angle_smooth,
            'motion_intensity': motion_intensity,
            'positions': positions,
        }
    
    def extract_visual_signals(self, gsa_files: List[Path]) -> Dict[str, np.ndarray]:
        """
        提取视觉信号 - 基于 CLIP 特征
        
        关键: 相邻帧视觉相似度骤降 = 场景变化
        """
        n_frames = len(gsa_files)
        frame_features = []
        n_detections = []
        
        print("  加载 CLIP 特征...")
        for gsa_file in gsa_files:
            try:
                with gzip.open(gsa_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 获取每帧的 CLIP 特征 (对所有检测取平均)
                if 'image_feats' in data and len(data['image_feats']) > 0:
                    feats = data['image_feats']
                    # 按置信度加权平均
                    if 'confidence' in data:
                        weights = data['confidence']
                        weights = weights / (weights.sum() + 1e-8)
                        frame_feat = np.average(feats, axis=0, weights=weights)
                    else:
                        frame_feat = feats.mean(axis=0)
                    frame_features.append(frame_feat)
                    n_detections.append(len(feats))
                else:
                    frame_features.append(None)
                    n_detections.append(0)
            except Exception as e:
                frame_features.append(None)
                n_detections.append(0)
        
        # 处理缺失特征
        feat_dim = None
        for f in frame_features:
            if f is not None:
                feat_dim = len(f)
                break
        
        if feat_dim is None:
            print("  警告: 未找到有效的 CLIP 特征")
            return None
        
        # 填充缺失值
        for i in range(len(frame_features)):
            if frame_features[i] is None:
                frame_features[i] = np.zeros(feat_dim)
        
        frame_features = np.array(frame_features)
        n_detections = np.array(n_detections)
        
        # 归一化特征
        norms = np.linalg.norm(frame_features, axis=1, keepdims=True)
        features_norm = frame_features / (norms + 1e-8)
        
        # 计算相邻帧相似度
        similarities = [1.0]
        for i in range(1, n_frames):
            sim = np.dot(features_norm[i-1], features_norm[i])
            similarities.append(float(sim))
        similarities = np.array(similarities)
        
        # 视觉变化 = 1 - 相似度
        visual_change = 1 - similarities
        visual_change_smooth = gaussian_filter1d(visual_change, self.smooth_sigma)
        
        # 归一化
        visual_change_norm = visual_change_smooth / (visual_change_smooth.max() + 1e-8)
        
        return {
            'frame_similarity': similarities,
            'visual_change': visual_change,
            'visual_change_smooth': visual_change_smooth,
            'visual_change_norm': visual_change_norm,
            'n_detections': n_detections,
            'frame_features': frame_features,
        }
    
    def extract_semantic_signals(self, gsa_files: List[Path]) -> Dict[str, np.ndarray]:
        """
        提取语义信号 - 基于物体检测
        
        关键:
        1. 新物体涌现峰值 = 进入新区域
        2. 检测到边界物体 (门/墙) = 区域边界
        """
        n_frames = len(gsa_files)
        
        detection_counts = []
        new_object_counts = []
        boundary_signals = []
        cumulative_objects = []
        
        # 用于追踪已见物体 (基于特征相似度)
        seen_features = []
        feature_threshold = 0.85  # 相似度阈值
        
        print("  分析物体分布...")
        for gsa_file in gsa_files:
            try:
                with gzip.open(gsa_file, 'rb') as f:
                    data = pickle.load(f)
                
                n_det = len(data.get('xyxy', []))
                detection_counts.append(n_det)
                
                # 检测新物体 (基于特征相似度)
                new_count = 0
                if 'image_feats' in data and len(data['image_feats']) > 0:
                    for feat in data['image_feats']:
                        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                        is_new = True
                        for seen_feat in seen_features:
                            sim = np.dot(feat_norm, seen_feat)
                            if sim > feature_threshold:
                                is_new = False
                                break
                        if is_new:
                            seen_features.append(feat_norm)
                            new_count += 1
                
                new_object_counts.append(new_count)
                cumulative_objects.append(len(seen_features))
                
                # 检测边界物体 (基于类别名称)
                has_boundary = False
                classes = data.get('classes', [])
                if isinstance(classes, list):
                    for cls in classes:
                        if isinstance(cls, str):
                            if any(b in cls.lower() for b in self.BOUNDARY_OBJECTS):
                                has_boundary = True
                                break
                
                boundary_signals.append(1.0 if has_boundary else 0.0)
                
            except Exception as e:
                detection_counts.append(0)
                new_object_counts.append(0)
                boundary_signals.append(0.0)
                cumulative_objects.append(len(seen_features))
        
        detection_counts = np.array(detection_counts)
        new_object_counts = np.array(new_object_counts)
        boundary_signals = np.array(boundary_signals)
        cumulative_objects = np.array(cumulative_objects)
        
        # 平滑新物体信号
        new_object_smooth = gaussian_filter1d(new_object_counts.astype(float), self.smooth_sigma)
        
        # 检测数量变化
        detection_change = np.abs(np.diff(detection_counts, prepend=detection_counts[0]))
        detection_change_smooth = gaussian_filter1d(detection_change.astype(float), self.smooth_sigma)
        
        # 综合语义变化信号
        # 新物体涌现 + 检测数量突变 + 边界物体
        new_obj_norm = new_object_smooth / (new_object_smooth.max() + 1e-8)
        det_change_norm = detection_change_smooth / (detection_change_smooth.max() + 1e-8)
        
        semantic_change = 0.5 * new_obj_norm + 0.3 * det_change_norm + 0.2 * boundary_signals
        
        return {
            'detection_counts': detection_counts,
            'new_object_counts': new_object_counts,
            'cumulative_objects': cumulative_objects,
            'boundary_signals': boundary_signals,
            'new_object_smooth': new_object_smooth,
            'detection_change': detection_change_smooth,
            'semantic_change': semantic_change,
        }
    
    def fuse_signals(
        self,
        motion_signals: Dict,
        visual_signals: Optional[Dict],
        semantic_signals: Optional[Dict],
    ) -> Dict[str, np.ndarray]:
        """融合多模态信号"""
        
        motion_change = motion_signals['motion_intensity']
        
        # 动态调整权重
        total_weight = self.motion_weight
        weighted_signal = motion_change * self.motion_weight
        
        if visual_signals is not None:
            visual_change = visual_signals['visual_change_norm']
            weighted_signal += visual_change * self.visual_weight
            total_weight += self.visual_weight
        
        if semantic_signals is not None:
            semantic_change = semantic_signals['semantic_change']
            weighted_signal += semantic_change * self.semantic_weight
            total_weight += self.semantic_weight
        
        # 归一化
        fused_signal = weighted_signal / total_weight
        
        # 计算变化点得分
        gradient = np.abs(np.gradient(fused_signal))
        
        return {
            'fused_signal': fused_signal,
            'gradient': gradient,
            'motion_contribution': motion_change,
            'visual_contribution': visual_signals['visual_change_norm'] if visual_signals else None,
            'semantic_contribution': semantic_signals['semantic_change'] if semantic_signals else None,
        }
    
    def detect_change_points(self, fused_signals: Dict) -> List[int]:
        """检测变化点"""
        fused = fused_signals['fused_signal']
        gradient = fused_signals['gradient']
        
        # 在梯度信号上找峰值
        gradient_smooth = gaussian_filter1d(gradient, sigma=1.0)
        
        peaks, properties = find_peaks(
            gradient_smooth,
            prominence=self.peak_prominence,
            distance=self.peak_distance,
        )
        
        # 也考虑融合信号本身的峰值
        signal_peaks, _ = find_peaks(
            fused,
            prominence=self.peak_prominence * 0.5,
            distance=self.peak_distance,
        )
        
        # 合并
        all_peaks = np.unique(np.concatenate([peaks, signal_peaks]))
        all_peaks = np.sort(all_peaks)
        
        return all_peaks.tolist()
    
    def create_segments(
        self,
        change_points: List[int],
        n_frames: int,
        positions: np.ndarray,
        semantic_signals: Optional[Dict] = None,
    ) -> List[RegionSegment]:
        """创建分段"""
        boundaries = [0] + list(change_points) + [n_frames]
        boundaries = sorted(set(boundaries))
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            seg = RegionSegment(
                region_id=i,
                start_frame=start,
                end_frame=end,
            )
            
            # 计算区域中心
            if end <= len(positions):
                seg.centroid = positions[start:end].mean(axis=0)
            
            # 统计该区域内首次出现的物体数
            if semantic_signals is not None:
                new_objs = semantic_signals['new_object_counts'][start:end]
                seg.n_objects_first_seen = int(new_objs.sum())
            
            segments.append(seg)
        
        return segments
    
    def merge_short_segments(self, segments: List[RegionSegment]) -> List[RegionSegment]:
        """合并过短的分段"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if current.n_frames < self.min_segment_frames:
                # 合并
                current = RegionSegment(
                    region_id=current.region_id,
                    start_frame=current.start_frame,
                    end_frame=next_seg.end_frame,
                    n_objects_first_seen=current.n_objects_first_seen + next_seg.n_objects_first_seen,
                )
                if current.centroid is not None and next_seg.centroid is not None:
                    current.centroid = (current.centroid + next_seg.centroid) / 2
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        
        # 重新编号
        for i, seg in enumerate(merged):
            seg.region_id = i
        
        return merged
    
    def segment(
        self,
        trajectory_path: str,
        gsa_dir: str,
        output_dir: str,
        stride: int = 5,
    ) -> Tuple[List[RegionSegment], Dict]:
        """执行完整的分段流程"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("增强版时序场景分段")
        print("=" * 60)
        print(f"信号权重: 运动={self.motion_weight}, 视觉={self.visual_weight}, 语义={self.semantic_weight}")
        print()
        
        # 1. 加载轨迹
        print("[1/5] 加载轨迹...")
        poses = self._load_trajectory(trajectory_path, stride)
        n_frames = len(poses)
        print(f"  加载了 {n_frames} 帧")
        
        # 2. 加载 GSA 文件列表
        gsa_path = Path(gsa_dir)
        gsa_files = natsorted(gsa_path.glob('*.pkl.gz'))
        # 按 stride 采样
        gsa_files_sampled = gsa_files[::stride]
        print(f"  找到 {len(gsa_files_sampled)} 个 GSA 文件")
        
        # 3. 提取信号
        print("\n[2/5] 提取运动信号...")
        motion_signals = self.extract_motion_signals(poses)
        
        print("\n[3/5] 提取视觉信号 (CLIP)...")
        visual_signals = None
        if len(gsa_files_sampled) >= n_frames:
            visual_signals = self.extract_visual_signals(gsa_files_sampled[:n_frames])
        
        print("\n[4/5] 提取语义信号 (物体分布)...")
        semantic_signals = None
        if len(gsa_files_sampled) >= n_frames:
            semantic_signals = self.extract_semantic_signals(gsa_files_sampled[:n_frames])
        
        # 4. 融合信号
        print("\n[5/5] 融合信号并检测变化点...")
        fused_signals = self.fuse_signals(motion_signals, visual_signals, semantic_signals)
        
        # 5. 检测变化点
        change_points = self.detect_change_points(fused_signals)
        print(f"  检测到 {len(change_points)} 个变化点")
        
        # 6. 创建分段
        segments = self.create_segments(
            change_points, n_frames,
            motion_signals['positions'],
            semantic_signals
        )
        print(f"  生成 {len(segments)} 个初始分段")
        
        # 7. 合并过短分段
        segments = self.merge_short_segments(segments)
        print(f"  合并后 {len(segments)} 个分段")
        
        # 收集调试信息
        debug_info = {
            'n_frames': n_frames,
            'change_points': change_points,
            'motion_signals': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in motion_signals.items() if k != 'positions'},
            'visual_signals': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in (visual_signals or {}).items() if k != 'frame_features'},
            'semantic_signals': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in (semantic_signals or {}).items()},
            'fused_signals': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in fused_signals.items()},
        }
        
        # 保存结果
        self._save_results(segments, debug_info, output_path)
        
        return segments, debug_info
    
    def _load_trajectory(self, traj_path: str, stride: int) -> np.ndarray:
        """加载轨迹"""
        poses = []
        with open(traj_path, 'r') as f:
            for i, line in enumerate(f):
                if i % stride != 0:
                    continue
                values = list(map(float, line.strip().split()))
                if len(values) == 16:
                    poses.append(np.array(values).reshape(4, 4))
        return np.array(poses)
    
    def _save_results(self, segments: List[RegionSegment], debug_info: Dict, output_path: Path):
        """保存结果"""
        # 分段结果
        segments_data = [seg.to_dict() for seg in segments]
        with open(output_path / 'trajectory_segments.json', 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        # 信号数据
        with open(output_path / 'segmentation_signals.json', 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        # 报告
        report = []
        report.append("=" * 50)
        report.append("Enhanced Segmentation Report")
        report.append("=" * 50)
        report.append(f"Frames: {debug_info['n_frames']}")
        report.append(f"Segments: {len(segments)}")
        report.append(f"Change points: {len(debug_info['change_points'])}")
        report.append("")
        for seg in segments:
            new_obj = seg.n_objects_first_seen
            report.append(f"Region {seg.region_id}: frames {seg.start_frame}-{seg.end_frame} ({seg.n_frames} frames), {new_obj} new objects")
        
        with open(output_path / 'segmentation_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\n" + '\n'.join(report))
        print(f"\n结果保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='增强版时序场景分段')
    parser.add_argument('--trajectory', type=str, required=True)
    parser.add_argument('--gsa_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--motion_weight', type=float, default=0.3)
    parser.add_argument('--visual_weight', type=float, default=0.4)
    parser.add_argument('--semantic_weight', type=float, default=0.3)
    parser.add_argument('--min_segment_frames', type=int, default=10)
    
    args = parser.parse_args()
    
    segmenter = EnhancedTrajectorySegmenter(
        motion_weight=args.motion_weight,
        visual_weight=args.visual_weight,
        semantic_weight=args.semantic_weight,
        min_segment_frames=args.min_segment_frames,
    )
    
    segmenter.segment(
        trajectory_path=args.trajectory,
        gsa_dir=args.gsa_dir,
        output_dir=args.output,
        stride=args.stride,
    )


if __name__ == '__main__':
    main()
