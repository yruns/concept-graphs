"""
时序场景分段器

核心模块：将连续的探索轨迹分割为语义连贯的区域片段
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path

# 支持相对导入和绝对导入
try:
    from .signal_extractors import (
        MotionSignalExtractor,
        VisualSignalExtractor,
        SemanticSignalExtractor,
        MultiModalSignalFusion
    )
except ImportError:
    from signal_extractors import (
        MotionSignalExtractor,
        VisualSignalExtractor,
        SemanticSignalExtractor,
        MultiModalSignalFusion
    )


@dataclass
class RegionSegment:
    """表示一个场景区域的数据结构"""
    region_id: int
    start_frame: int
    end_frame: int
    object_indices: List[int] = field(default_factory=list)
    
    # 区域特征
    centroid: Optional[np.ndarray] = None  # 区域中心位置
    dominant_objects: List[str] = field(default_factory=list)  # 主要物体类别
    region_type: str = "unknown"  # 区域类型推断
    
    # 元信息
    confidence: float = 1.0
    
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
            'dominant_objects': self.dominant_objects,
            'region_type': self.region_type,
            'confidence': self.confidence,
        }


class TrajectorySegmenter:
    """
    轨迹分段器
    
    基于多模态信号将探索轨迹分割为语义连贯的区域
    """
    
    def __init__(
        self,
        motion_weight: float = 0.4,
        visual_weight: float = 0.4,
        semantic_weight: float = 0.2,
        min_segment_frames: int = 10,
        peak_prominence: float = 0.1,
        peak_distance: int = 15,
        smooth_sigma: float = 2.0,
    ):
        """
        Args:
            motion_weight: 运动信号权重
            visual_weight: 视觉信号权重
            semantic_weight: 语义信号权重
            min_segment_frames: 最小分段帧数
            peak_prominence: 峰值检测的显著性阈值
            peak_distance: 峰值之间的最小距离
            smooth_sigma: 信号平滑的标准差
        """
        self.motion_extractor = MotionSignalExtractor(smooth_sigma)
        self.visual_extractor = VisualSignalExtractor(smooth_sigma)
        self.semantic_extractor = SemanticSignalExtractor(smooth_sigma)
        self.fusion = MultiModalSignalFusion(motion_weight, visual_weight, semantic_weight)
        
        self.min_segment_frames = min_segment_frames
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
    
    def segment(
        self,
        poses: np.ndarray,
        clip_features: Optional[np.ndarray] = None,
        frame_detections: Optional[List[Dict]] = None,
        object_frame_mapping: Optional[Dict[int, List[int]]] = None,
    ) -> Tuple[List[RegionSegment], Dict[str, Any]]:
        """
        执行轨迹分段
        
        Args:
            poses: (N, 4, 4) 相机位姿序列
            clip_features: (N, D) 每帧的 CLIP 特征（可选）
            frame_detections: 每帧的检测结果列表（可选）
            object_frame_mapping: 物体ID到帧索引的映射（可选）
            
        Returns:
            segments: 区域分段列表
            debug_info: 调试信息
        """
        n_frames = len(poses)
        print(f"开始分段，共 {n_frames} 帧")
        
        # 1. 提取各模态信号
        motion_signals = self.motion_extractor.extract(poses)
        print(f"  运动信号提取完成")
        
        visual_signals = None
        if clip_features is not None and len(clip_features) > 0:
            visual_signals = self.visual_extractor.extract(clip_features)
            print(f"  视觉信号提取完成")
        
        semantic_signals = None
        if frame_detections is not None and len(frame_detections) > 0:
            semantic_signals = self.semantic_extractor.extract(frame_detections)
            print(f"  语义信号提取完成")
        
        # 2. 融合信号
        fused = self.fusion.fuse(motion_signals, visual_signals, semantic_signals)
        print(f"  信号融合完成")
        
        # 3. 检测变化点
        change_points = self._detect_change_points(fused['fused_signal'])
        print(f"  检测到 {len(change_points)} 个变化点: {change_points}")
        
        # 4. 生成分段
        segments = self._create_segments(
            change_points, n_frames, 
            motion_signals['positions'],
            object_frame_mapping
        )
        print(f"  生成 {len(segments)} 个区域分段")
        
        # 5. 合并过短的分段
        segments = self._merge_short_segments(segments)
        print(f"  合并后剩余 {len(segments)} 个区域分段")
        
        # 收集调试信息
        debug_info = {
            'motion_signals': motion_signals,
            'visual_signals': visual_signals,
            'semantic_signals': semantic_signals,
            'fused_signals': fused,
            'change_points': change_points,
            'n_frames': n_frames,
        }
        
        return segments, debug_info
    
    def _detect_change_points(self, signal: np.ndarray) -> List[int]:
        """
        检测变化点
        
        使用峰值检测找到信号中的突变位置
        """
        # 使用一阶导数的绝对值
        gradient = np.abs(np.gradient(signal))
        gradient_smooth = gaussian_filter1d(gradient, sigma=2.0)
        
        # 峰值检测
        peaks, properties = find_peaks(
            gradient_smooth,
            prominence=self.peak_prominence,
            distance=self.peak_distance,
        )
        
        # 也考虑信号本身的峰值
        signal_peaks, _ = find_peaks(
            signal,
            prominence=self.peak_prominence * 0.5,
            distance=self.peak_distance,
        )
        
        # 合并两种峰值
        all_peaks = np.unique(np.concatenate([peaks, signal_peaks]))
        all_peaks = np.sort(all_peaks)
        
        return all_peaks.tolist()
    
    def _create_segments(
        self,
        change_points: List[int],
        n_frames: int,
        positions: np.ndarray,
        object_frame_mapping: Optional[Dict[int, List[int]]] = None,
    ) -> List[RegionSegment]:
        """
        根据变化点创建分段
        """
        # 添加起始和结束点
        boundaries = [0] + change_points + [n_frames]
        boundaries = sorted(set(boundaries))
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            segment = RegionSegment(
                region_id=i,
                start_frame=start,
                end_frame=end,
            )
            
            # 计算区域中心
            segment.centroid = positions[start:end].mean(axis=0)
            
            # 找到该区域包含的物体
            if object_frame_mapping is not None:
                segment.object_indices = self._find_objects_in_segment(
                    start, end, object_frame_mapping
                )
            
            segments.append(segment)
        
        return segments
    
    def _find_objects_in_segment(
        self,
        start_frame: int,
        end_frame: int,
        object_frame_mapping: Dict[int, List[int]],
    ) -> List[int]:
        """找出在给定帧范围内首次出现的物体"""
        objects_in_segment = []
        
        for obj_id, frame_indices in object_frame_mapping.items():
            # 检查物体是否主要在这个分段中出现
            frames_in_segment = [f for f in frame_indices if start_frame <= f < end_frame]
            if len(frames_in_segment) > 0:
                # 如果物体首次出现在这个分段，则归属于这个分段
                first_appearance = min(frame_indices)
                if start_frame <= first_appearance < end_frame:
                    objects_in_segment.append(obj_id)
        
        return objects_in_segment
    
    def _merge_short_segments(self, segments: List[RegionSegment]) -> List[RegionSegment]:
        """合并过短的分段"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if current.n_frames < self.min_segment_frames:
                # 当前分段太短，与下一个合并
                current = RegionSegment(
                    region_id=current.region_id,
                    start_frame=current.start_frame,
                    end_frame=next_seg.end_frame,
                    object_indices=current.object_indices + next_seg.object_indices,
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
    
    def segment_from_files(
        self,
        trajectory_path: str,
        map_file_path: Optional[str] = None,
        gsa_results_path: Optional[str] = None,
        stride: int = 1,
    ) -> Tuple[List[RegionSegment], Dict[str, Any]]:
        """
        从文件加载数据并执行分段
        
        Args:
            trajectory_path: traj.txt 文件路径
            map_file_path: 3D 对象地图 pkl.gz 文件路径（可选）
            gsa_results_path: GSA 结果目录路径（可选）
            stride: 帧采样步长
            
        Returns:
            segments: 区域分段列表
            debug_info: 调试信息
        """
        import gzip
        import pickle
        from pathlib import Path
        
        # 1. 加载轨迹
        poses = self._load_trajectory(trajectory_path, stride)
        print(f"加载轨迹: {len(poses)} 帧 (stride={stride})")
        
        # 2. 尝试从地图文件提取物体-帧映射和 CLIP 特征
        object_frame_mapping = None
        clip_features = None
        
        if map_file_path and Path(map_file_path).exists():
            print(f"加载对象地图: {map_file_path}")
            with gzip.open(map_file_path, 'rb') as f:
                map_data = pickle.load(f)
            
            objects = map_data.get('objects', [])
            print(f"  加载了 {len(objects)} 个物体")
            
            # 提取物体-帧映射（如果存在）
            # 注：标准 ConceptGraphs 输出可能不包含这个信息
            # 但我们可以从 num_detections 推断
            object_frame_mapping = {}
            for i, obj in enumerate(objects):
                # 如果有 frame_indices 属性就使用
                if 'frame_indices' in obj:
                    object_frame_mapping[i] = obj['frame_indices']
                else:
                    # 否则创建一个占位符
                    object_frame_mapping[i] = []
            
            # 提取聚合的 CLIP 特征（每帧的平均）
            # 这里我们使用物体的 CLIP 特征
            # 实际上应该从 gsa_results 中读取每帧的特征
        
        # 3. 尝试加载 GSA 结果
        frame_detections = None
        if gsa_results_path and Path(gsa_results_path).exists():
            frame_detections, clip_features = self._load_gsa_results(
                gsa_results_path, stride
            )
            print(f"  加载了 {len(frame_detections)} 帧的检测结果")
        
        # 4. 执行分段
        return self.segment(
            poses, 
            clip_features, 
            frame_detections,
            object_frame_mapping
        )
    
    def _load_trajectory(self, trajectory_path: str, stride: int = 1) -> np.ndarray:
        """加载轨迹文件"""
        poses = []
        with open(trajectory_path, 'r') as f:
            for i, line in enumerate(f):
                if i % stride != 0:
                    continue
                values = list(map(float, line.strip().split()))
                if len(values) == 16:
                    pose = np.array(values).reshape(4, 4)
                    poses.append(pose)
                elif len(values) == 12:
                    # 3x4 矩阵
                    pose = np.eye(4)
                    pose[:3, :] = np.array(values).reshape(3, 4)
                    poses.append(pose)
        return np.array(poses)
    
    def _load_gsa_results(
        self, 
        gsa_results_path: str, 
        stride: int = 1
    ) -> Tuple[List[Dict], np.ndarray]:
        """加载 GSA 检测结果"""
        import gzip
        import pickle
        from pathlib import Path
        
        gsa_path = Path(gsa_results_path)
        frame_detections = []
        all_clip_features = []
        
        # 获取所有 pkl.gz 文件
        pkl_files = sorted(gsa_path.glob("*.pkl.gz"))
        
        for i, pkl_file in enumerate(pkl_files):
            if i % stride != 0:
                continue
            
            try:
                with gzip.open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 提取检测信息
                detection = {
                    'n_masks': len(data.get('masks', [])) if 'masks' in data else 0,
                    'class_names': data.get('classes', []),
                }
                frame_detections.append(detection)
                
                # 提取 CLIP 特征（如果有）
                if 'clip_ft' in data and data['clip_ft'] is not None:
                    # 对该帧所有检测的特征取平均
                    clip_ft = data['clip_ft']
                    if isinstance(clip_ft, np.ndarray) and len(clip_ft) > 0:
                        frame_clip = clip_ft.mean(axis=0)
                        all_clip_features.append(frame_clip)
                    else:
                        all_clip_features.append(None)
                else:
                    all_clip_features.append(None)
            except Exception as e:
                print(f"警告: 无法加载 {pkl_file}: {e}")
                frame_detections.append({'n_masks': 0, 'class_names': []})
                all_clip_features.append(None)
        
        # 处理 CLIP 特征
        clip_features = None
        if all_clip_features and any(f is not None for f in all_clip_features):
            # 找到第一个非空特征的维度
            feat_dim = next(f.shape[0] for f in all_clip_features if f is not None)
            
            # 用零向量填充缺失的特征
            processed_features = []
            for f in all_clip_features:
                if f is not None:
                    processed_features.append(f)
                else:
                    processed_features.append(np.zeros(feat_dim))
            
            clip_features = np.stack(processed_features)
        
        return frame_detections, clip_features
    
    def save_results(
        self,
        segments: List[RegionSegment],
        debug_info: Dict[str, Any],
        output_dir: str,
    ):
        """保存分段结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存分段结果
        segments_data = [seg.to_dict() for seg in segments]
        with open(output_path / "trajectory_segments.json", 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        # 2. 保存区域-物体映射
        region_objects = {
            seg.region_id: seg.object_indices for seg in segments
        }
        with open(output_path / "region_objects.json", 'w') as f:
            json.dump(region_objects, f, indent=2)
        
        # 3. 保存帧-区域映射
        frame_to_region = {}
        for seg in segments:
            for frame_idx in range(seg.start_frame, seg.end_frame):
                frame_to_region[frame_idx] = seg.region_id
        with open(output_path / "frame_to_region.json", 'w') as f:
            json.dump(frame_to_region, f, indent=2)
        
        # 4. 保存信号数据（用于可视化和调试）
        import pickle
        signals_to_save = {}
        for key, value in debug_info.items():
            if isinstance(value, dict):
                signals_to_save[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            elif isinstance(value, np.ndarray):
                signals_to_save[key] = value.tolist()
            else:
                signals_to_save[key] = value
        
        with open(output_path / "segmentation_signals.json", 'w') as f:
            json.dump(signals_to_save, f, indent=2)
        
        print(f"结果保存到: {output_path}")
        print(f"  - trajectory_segments.json: 分段列表")
        print(f"  - region_objects.json: 区域-物体映射")
        print(f"  - frame_to_region.json: 帧-区域映射")
        print(f"  - segmentation_signals.json: 信号数据")
