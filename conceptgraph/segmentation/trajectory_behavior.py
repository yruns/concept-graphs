#!/usr/bin/env python3
"""
轨迹行为分析
============

分析相机轨迹中的行为模式：
1. 停留点(Dwell Points): 相机长时间停留的位置
2. 环顾事件(Look Around): 相机在原地旋转观察
3. 快速穿越(Fast Traverse): 相机快速移动经过的区域
4. 重要性热图: 基于行为推断的区域重要性
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial.transform import Rotation


@dataclass
class DwellPoint:
    """停留点"""
    frame_start: int
    frame_end: int
    position: List[float]        # 平均位置 [x, y, z]
    duration_seconds: float
    n_frames: int
    visible_objects: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "position": self.position,
            "duration_seconds": self.duration_seconds,
            "n_frames": self.n_frames,
            "visible_objects": self.visible_objects
        }


@dataclass
class LookAroundEvent:
    """环顾事件"""
    frame_start: int
    frame_end: int
    position: List[float]        # 中心位置
    total_rotation_deg: float    # 总旋转角度
    direction: str               # left/right/both
    n_frames: int
    
    def to_dict(self) -> Dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "position": self.position,
            "total_rotation_deg": self.total_rotation_deg,
            "direction": self.direction,
            "n_frames": self.n_frames
        }


@dataclass
class TraverseSegment:
    """快速穿越片段"""
    frame_start: int
    frame_end: int
    start_position: List[float]
    end_position: List[float]
    distance: float
    avg_speed: float             # m/s
    n_frames: int
    
    def to_dict(self) -> Dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "distance": self.distance,
            "avg_speed": self.avg_speed,
            "n_frames": self.n_frames
        }


@dataclass
class TrajectoryBehaviorAnalysis:
    """轨迹行为分析结果"""
    dwell_points: List[DwellPoint] = field(default_factory=list)
    look_around_events: List[LookAroundEvent] = field(default_factory=list)
    traverse_segments: List[TraverseSegment] = field(default_factory=list)
    importance_heatmap: Optional[np.ndarray] = None
    grid_size: float = 0.5
    grid_origin: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "dwell_points": [d.to_dict() for d in self.dwell_points],
            "look_around_events": [e.to_dict() for e in self.look_around_events],
            "traverse_segments": [s.to_dict() for s in self.traverse_segments],
            "grid_size": self.grid_size,
            "grid_origin": self.grid_origin,
            "importance_heatmap": self.importance_heatmap.tolist() if self.importance_heatmap is not None else None
        }


class TrajectoryBehaviorAnalyzer:
    """轨迹行为分析器"""
    
    def __init__(
        self,
        poses: np.ndarray,
        fps: float = 5.0,
        stride: int = 1
    ):
        """
        Args:
            poses: 相机位姿数组 [n_frames, 7] 或 [n_frames, 4, 4]
                   7元素格式: [tx, ty, tz, qw, qx, qy, qz]
                   4x4格式: 变换矩阵
            fps: 帧率（采样后）
            stride: 采样步长（用于计算原始帧号）
        """
        self.poses = self._normalize_poses(poses)
        self.fps = fps
        self.stride = stride
        self.n_frames = len(poses)
        
        # 提取位置和朝向
        self.positions = self._extract_positions()
        self.orientations = self._extract_orientations()
        
        # 计算运动特征
        self.velocities = self._compute_velocities()
        self.angular_velocities = self._compute_angular_velocities()
    
    def _normalize_poses(self, poses: np.ndarray) -> np.ndarray:
        """归一化位姿格式"""
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            # 4x4变换矩阵格式
            return poses
        elif poses.ndim == 2 and poses.shape[1] >= 7:
            # [tx, ty, tz, qw, qx, qy, qz] 格式
            n = len(poses)
            matrices = np.zeros((n, 4, 4))
            for i in range(n):
                t = poses[i, :3]
                q = poses[i, 3:7]  # qw, qx, qy, qz
                # 转换四元数格式：scipy使用 [qx, qy, qz, qw]
                r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                matrices[i, :3, :3] = r.as_matrix()
                matrices[i, :3, 3] = t
                matrices[i, 3, 3] = 1
            return matrices
        else:
            raise ValueError(f"不支持的位姿格式: {poses.shape}")
    
    def _extract_positions(self) -> np.ndarray:
        """提取位置"""
        return self.poses[:, :3, 3]
    
    def _extract_orientations(self) -> np.ndarray:
        """提取朝向（旋转矩阵）"""
        return self.poses[:, :3, :3]
    
    def _compute_velocities(self) -> np.ndarray:
        """计算速度"""
        if self.n_frames < 2:
            return np.zeros((self.n_frames, 3))
        
        dt = 1.0 / self.fps
        velocities = np.zeros((self.n_frames, 3))
        velocities[1:] = (self.positions[1:] - self.positions[:-1]) / dt
        velocities[0] = velocities[1] if self.n_frames > 1 else 0
        
        return velocities
    
    def _compute_angular_velocities(self) -> np.ndarray:
        """计算角速度（简化：只计算yaw变化）"""
        if self.n_frames < 2:
            return np.zeros(self.n_frames)
        
        dt = 1.0 / self.fps
        angular_vel = np.zeros(self.n_frames)
        
        for i in range(1, self.n_frames):
            # 计算相对旋转
            R_rel = self.orientations[i] @ self.orientations[i-1].T
            # 提取旋转角度
            r = Rotation.from_matrix(R_rel)
            angle = np.linalg.norm(r.as_rotvec())  # 弧度
            angular_vel[i] = np.degrees(angle) / dt  # 度/秒
        
        angular_vel[0] = angular_vel[1] if self.n_frames > 1 else 0
        
        return angular_vel
    
    def analyze(self) -> TrajectoryBehaviorAnalysis:
        """执行完整的轨迹行为分析"""
        result = TrajectoryBehaviorAnalysis()
        
        # 1. 检测停留点
        result.dwell_points = self._detect_dwell_points()
        
        # 2. 检测环顾事件
        result.look_around_events = self._detect_look_around()
        
        # 3. 检测快速穿越
        result.traverse_segments = self._detect_fast_traverse()
        
        # 4. 计算重要性热图
        heatmap, grid_origin = self._compute_importance_heatmap()
        result.importance_heatmap = heatmap
        result.grid_origin = grid_origin
        
        return result
    
    def _detect_dwell_points(
        self, 
        min_duration: float = 2.0,
        max_movement: float = 0.3
    ) -> List[DwellPoint]:
        """
        检测停留点
        
        Args:
            min_duration: 最小停留时间（秒）
            max_movement: 最大移动距离（米）
        """
        min_frames = int(min_duration * self.fps)
        dwell_points = []
        
        i = 0
        while i < self.n_frames:
            start = i
            center = self.positions[i]
            
            # 找到停留结束的帧
            while i < self.n_frames:
                dist = np.linalg.norm(self.positions[i] - center)
                if dist > max_movement:
                    break
                # 更新中心为滑动平均
                n = i - start + 1
                center = (center * (n - 1) + self.positions[i]) / n
                i += 1
            
            duration_frames = i - start
            if duration_frames >= min_frames:
                dwell_points.append(DwellPoint(
                    frame_start=start,
                    frame_end=i - 1,
                    position=center.tolist(),
                    duration_seconds=duration_frames / self.fps,
                    n_frames=duration_frames
                ))
            
            i = max(i, start + 1)
        
        return dwell_points
    
    def _detect_look_around(
        self,
        min_rotation: float = 90.0,
        max_translation: float = 0.5,
        min_frames: int = 5
    ) -> List[LookAroundEvent]:
        """
        检测环顾事件
        
        Args:
            min_rotation: 最小累计旋转角度（度）
            max_translation: 最大位移（米）
            min_frames: 最小帧数
        """
        events = []
        
        i = 0
        while i < self.n_frames - min_frames:
            start = i
            start_pos = self.positions[i]
            total_rotation = 0.0
            
            # 累计旋转
            while i < self.n_frames - 1:
                translation = np.linalg.norm(self.positions[i] - start_pos)
                if translation > max_translation:
                    break
                
                # 计算帧间旋转
                if i > start:
                    R_rel = self.orientations[i] @ self.orientations[i-1].T
                    r = Rotation.from_matrix(R_rel)
                    # 提取yaw分量
                    euler = r.as_euler('zyx', degrees=True)
                    total_rotation += abs(euler[0])
                
                i += 1
            
            if i - start >= min_frames and total_rotation >= min_rotation:
                center_pos = np.mean(self.positions[start:i], axis=0)
                events.append(LookAroundEvent(
                    frame_start=start,
                    frame_end=i - 1,
                    position=center_pos.tolist(),
                    total_rotation_deg=total_rotation,
                    direction="both",  # 简化处理
                    n_frames=i - start
                ))
            
            i = max(i, start + 1)
        
        return events
    
    def _detect_fast_traverse(
        self,
        min_speed: float = 1.0,
        min_frames: int = 5
    ) -> List[TraverseSegment]:
        """
        检测快速穿越片段
        
        Args:
            min_speed: 最小速度（m/s）
            min_frames: 最小帧数
        """
        segments = []
        speeds = np.linalg.norm(self.velocities, axis=1)
        
        i = 0
        while i < self.n_frames:
            if speeds[i] >= min_speed:
                start = i
                
                while i < self.n_frames and speeds[i] >= min_speed * 0.7:  # 稍微放宽
                    i += 1
                
                if i - start >= min_frames:
                    start_pos = self.positions[start]
                    end_pos = self.positions[i - 1]
                    distance = np.linalg.norm(end_pos - start_pos)
                    duration = (i - start) / self.fps
                    
                    segments.append(TraverseSegment(
                        frame_start=start,
                        frame_end=i - 1,
                        start_position=start_pos.tolist(),
                        end_position=end_pos.tolist(),
                        distance=float(distance),
                        avg_speed=float(distance / duration) if duration > 0 else 0,
                        n_frames=i - start
                    ))
            else:
                i += 1
        
        return segments
    
    def _compute_importance_heatmap(
        self,
        grid_size: float = 0.5
    ) -> Tuple[np.ndarray, List[float]]:
        """
        计算重要性热图
        
        基于以下因素：
        - 停留时间
        - 环顾事件
        - 访问频率
        """
        # 计算网格范围
        min_pos = self.positions.min(axis=0)
        max_pos = self.positions.max(axis=0)
        
        # 只使用x, y (俯视图)
        grid_origin = [float(min_pos[0] - grid_size), float(min_pos[1] - grid_size)]
        
        n_x = int(np.ceil((max_pos[0] - min_pos[0] + 2 * grid_size) / grid_size)) + 1
        n_y = int(np.ceil((max_pos[1] - min_pos[1] + 2 * grid_size) / grid_size)) + 1
        
        heatmap = np.zeros((n_y, n_x))
        
        # 访问频率
        for pos in self.positions:
            gx = int((pos[0] - grid_origin[0]) / grid_size)
            gy = int((pos[1] - grid_origin[1]) / grid_size)
            if 0 <= gx < n_x and 0 <= gy < n_y:
                heatmap[gy, gx] += 1
        
        # 停留点加权
        dwell_points = self._detect_dwell_points()
        for dp in dwell_points:
            gx = int((dp.position[0] - grid_origin[0]) / grid_size)
            gy = int((dp.position[1] - grid_origin[1]) / grid_size)
            if 0 <= gx < n_x and 0 <= gy < n_y:
                # 停留时间越长，权重越高
                heatmap[gy, gx] += dp.duration_seconds * 2
        
        # 环顾事件加权
        look_arounds = self._detect_look_around()
        for la in look_arounds:
            gx = int((la.position[0] - grid_origin[0]) / grid_size)
            gy = int((la.position[1] - grid_origin[1]) / grid_size)
            if 0 <= gx < n_x and 0 <= gy < n_y:
                # 旋转角度越大，权重越高
                heatmap[gy, gx] += la.total_rotation_deg / 90.0
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap, grid_origin


def analyze_trajectory_from_scene(
    scene_path: str,
    stride: int = 5,
    fps: float = 30.0
) -> TrajectoryBehaviorAnalysis:
    """
    从场景数据中分析轨迹行为
    
    Args:
        scene_path: 场景路径
        stride: 帧采样步长
        fps: 原始帧率
        
    Returns:
        TrajectoryBehaviorAnalysis: 分析结果
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
    
    # 分析
    analyzer = TrajectoryBehaviorAnalyzer(
        poses=poses,
        fps=fps / stride,
        stride=stride
    )
    
    return analyzer.analyze()


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    analysis = analyze_trajectory_from_scene(
        args.scene_path,
        stride=args.stride,
        fps=args.fps
    )
    
    print("轨迹行为分析结果:")
    print("=" * 60)
    print(f"停留点: {len(analysis.dwell_points)} 个")
    for dp in analysis.dwell_points:
        print(f"  帧 {dp.frame_start}-{dp.frame_end}: {dp.duration_seconds:.1f}秒")
    
    print(f"\n环顾事件: {len(analysis.look_around_events)} 个")
    for la in analysis.look_around_events:
        print(f"  帧 {la.frame_start}-{la.frame_end}: {la.total_rotation_deg:.1f}度")
    
    print(f"\n快速穿越: {len(analysis.traverse_segments)} 段")
    for ts in analysis.traverse_segments:
        print(f"  帧 {ts.frame_start}-{ts.frame_end}: {ts.distance:.2f}m, {ts.avg_speed:.2f}m/s")
    
    if analysis.importance_heatmap is not None:
        print(f"\n重要性热图: {analysis.importance_heatmap.shape}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"\n保存到: {args.output}")
