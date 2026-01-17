#!/usr/bin/env python3
"""
3D 轨迹分段可视化

生成:
1. 3D 轨迹分段图（带物体点云）
2. 俯视图（2D 投影）
3. 分段统计报告
"""

import numpy as np
import json
import gzip
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


def load_trajectory(traj_path, stride=1):
    """加载轨迹文件"""
    poses = []
    with open(traj_path, 'r') as f:
        for i, line in enumerate(f):
            if i % stride != 0:
                continue
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    poses = np.array(poses)
    positions = poses[:, :3, 3]
    return poses, positions


def load_segments(segments_path):
    """加载分段结果"""
    with open(segments_path, 'r') as f:
        segments = json.load(f)
    return segments


def load_objects(mapfile_path):
    """加载物体数据"""
    with gzip.open(mapfile_path, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    obj_centers = []
    obj_sizes = []
    
    for obj in objects:
        if 'bbox_np' in obj:
            bbox_points = obj['bbox_np']
            center = bbox_points.mean(axis=0)
            size = np.linalg.norm(bbox_points.max(axis=0) - bbox_points.min(axis=0))
        elif 'pcd_np' in obj:
            points = obj['pcd_np']
            center = points.mean(axis=0)
            size = len(points) / 100
        else:
            continue
        obj_centers.append(center)
        obj_sizes.append(size)
    
    return np.array(obj_centers) if obj_centers else None, obj_sizes


def plot_3d_trajectory_with_segments(positions, segments, obj_centers=None, 
                                      output_path=None, figsize=(14, 10)):
    """绘制 3D 轨迹分段图"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用不同颜色绘制各分段
    n_segs = len(segments)
    colors = plt.cm.tab20(np.linspace(0, 1, n_segs))
    
    legend_handles = []
    
    for i, seg in enumerate(segments):
        start = seg['start_frame']
        end = seg['end_frame']
        
        if end > len(positions):
            end = len(positions)
        
        seg_positions = positions[start:end]
        
        if len(seg_positions) > 0:
            ax.plot(seg_positions[:, 0], seg_positions[:, 1], seg_positions[:, 2],
                   color=colors[i], linewidth=2.5, alpha=0.8)
            
            # 标记起点
            ax.scatter(seg_positions[0, 0], seg_positions[0, 1], seg_positions[0, 2],
                      color=colors[i], s=100, marker='o', edgecolors='black', linewidths=1)
            
            # 标记区域中心
            if seg.get('centroid'):
                centroid = np.array(seg['centroid'])
                ax.scatter(centroid[0], centroid[1], centroid[2],
                          color=colors[i], s=200, marker='*', edgecolors='black', linewidths=1)
        
        legend_handles.append(mpatches.Patch(color=colors[i], label='R%d (%d frames)' % (i, seg['n_frames'])))
    
    # 绘制物体位置
    if obj_centers is not None and len(obj_centers) > 0:
        ax.scatter(obj_centers[:, 0], obj_centers[:, 1], obj_centers[:, 2],
                  c='red', s=30, alpha=0.5, marker='^', label='Objects (%d)' % len(obj_centers))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Segmentation (%d regions)' % n_segs)
    
    # 添加图例
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('3D trajectory saved to: %s' % output_path)
    
    return fig


def plot_topdown_view(positions, segments, obj_centers=None,
                       output_path=None, figsize=(12, 10)):
    """绘制俯视图（2D 投影）"""
    fig, ax = plt.subplots(figsize=figsize)
    
    n_segs = len(segments)
    colors = plt.cm.tab20(np.linspace(0, 1, n_segs))
    
    # 绘制各分段轨迹
    for i, seg in enumerate(segments):
        start = seg['start_frame']
        end = min(seg['end_frame'], len(positions))
        seg_positions = positions[start:end]
        
        if len(seg_positions) > 0:
            ax.plot(seg_positions[:, 0], seg_positions[:, 1],
                   color=colors[i], linewidth=2, alpha=0.8, label='R%d' % i)
            
            # 标记区域中心
            if seg.get('centroid'):
                centroid = np.array(seg['centroid'])
                ax.scatter(centroid[0], centroid[1], color=colors[i], 
                          s=200, marker='*', edgecolors='black', linewidths=1, zorder=10)
                ax.annotate('R%d' % i, (centroid[0], centroid[1]), 
                           textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight='bold')
    
    # 绘制物体位置
    if obj_centers is not None and len(obj_centers) > 0:
        ax.scatter(obj_centers[:, 0], obj_centers[:, 1],
                  c='red', s=40, alpha=0.6, marker='^', label='Objects', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-Down View - Trajectory Segmentation')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('Top-down view saved to: %s' % output_path)
    
    return fig


def plot_segment_statistics(segments, positions, output_path=None, figsize=(14, 6)):
    """绘制分段统计图"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 各区域帧数
    ax = axes[0]
    region_ids = [s['region_id'] for s in segments]
    n_frames = [s['n_frames'] for s in segments]
    colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))
    ax.bar(region_ids, n_frames, color=colors)
    ax.set_xlabel('Region ID')
    ax.set_ylabel('Number of Frames')
    ax.set_title('Frames per Region')
    
    # 2. 各区域轨迹长度
    ax = axes[1]
    trajectory_lengths = []
    for seg in segments:
        start = seg['start_frame']
        end = min(seg['end_frame'], len(positions))
        seg_pos = positions[start:end]
        if len(seg_pos) > 1:
            diffs = np.diff(seg_pos, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            length = 0
        trajectory_lengths.append(length)
    ax.bar(region_ids, trajectory_lengths, color=colors)
    ax.set_xlabel('Region ID')
    ax.set_ylabel('Trajectory Length (m)')
    ax.set_title('Trajectory Length per Region')
    
    # 3. 时间线视图
    ax = axes[2]
    for i, seg in enumerate(segments):
        ax.barh(0, seg['n_frames'], left=seg['start_frame'], 
               color=colors[i], edgecolor='black', linewidth=0.5, height=0.5)
        mid = seg['start_frame'] + seg['n_frames'] / 2
        ax.text(mid, 0, 'R%d' % i, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Frame Index')
    ax.set_yticks([])
    ax.set_title('Temporal Segmentation Timeline')
    ax.set_xlim(0, segments[-1]['end_frame'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('Statistics saved to: %s' % output_path)
    
    return fig


def compute_evaluation_metrics(segments, positions):
    """计算评估指标"""
    metrics = {}
    
    # 1. 基本统计
    n_segments = len(segments)
    total_frames = segments[-1]['end_frame']
    avg_frames_per_segment = total_frames / n_segments
    
    frame_counts = [s['n_frames'] for s in segments]
    std_frames = np.std(frame_counts)
    
    metrics['n_segments'] = n_segments
    metrics['total_frames'] = total_frames
    metrics['avg_frames_per_segment'] = avg_frames_per_segment
    metrics['std_frames_per_segment'] = std_frames
    
    # 2. 区域空间分离度
    centroids = []
    for seg in segments:
        if seg.get('centroid'):
            centroids.append(seg['centroid'])
    
    if len(centroids) > 1:
        centroids = np.array(centroids)
        # 计算区域间距离
        from scipy.spatial.distance import pdist
        inter_region_dists = pdist(centroids)
        metrics['avg_inter_region_distance'] = np.mean(inter_region_dists)
        metrics['min_inter_region_distance'] = np.min(inter_region_dists)
    
    # 3. 区域内紧凑度
    compactness = []
    for seg in segments:
        start = seg['start_frame']
        end = min(seg['end_frame'], len(positions))
        seg_pos = positions[start:end]
        if len(seg_pos) > 1:
            center = seg_pos.mean(axis=0)
            dists = np.linalg.norm(seg_pos - center, axis=1)
            compactness.append(np.mean(dists))
    
    if compactness:
        metrics['avg_region_compactness'] = np.mean(compactness)
    
    # 4. 运动一致性（区域内速度方差）
    velocity_vars = []
    for seg in segments:
        start = seg['start_frame']
        end = min(seg['end_frame'], len(positions))
        seg_pos = positions[start:end]
        if len(seg_pos) > 2:
            velocities = np.linalg.norm(np.diff(seg_pos, axis=0), axis=1)
            velocity_vars.append(np.var(velocities))
    
    if velocity_vars:
        metrics['avg_velocity_variance'] = np.mean(velocity_vars)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='3D 轨迹分段可视化')
    parser.add_argument('--trajectory', type=str, required=True, help='轨迹文件')
    parser.add_argument('--segments', type=str, required=True, help='分段结果JSON')
    parser.add_argument('--mapfile', type=str, default=None, help='物体地图文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--stride', type=int, default=5, help='帧步长')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('3D Trajectory Segmentation Visualization')
    print('=' * 60)
    
    # 加载数据
    poses, positions = load_trajectory(args.trajectory, args.stride)
    print('Loaded trajectory: %d frames' % len(positions))
    
    segments = load_segments(args.segments)
    print('Loaded segments: %d regions' % len(segments))
    
    obj_centers = None
    if args.mapfile and Path(args.mapfile).exists():
        obj_centers, _ = load_objects(args.mapfile)
        if obj_centers is not None:
            print('Loaded objects: %d' % len(obj_centers))
    
    # 生成可视化
    print('\nGenerating visualizations...')
    
    # 3D 轨迹图
    plot_3d_trajectory_with_segments(
        positions, segments, obj_centers,
        output_path=str(output_dir / 'trajectory_3d.png')
    )
    
    # 俯视图
    plot_topdown_view(
        positions, segments, obj_centers,
        output_path=str(output_dir / 'trajectory_topdown.png')
    )
    
    # 统计图
    plot_segment_statistics(
        segments, positions,
        output_path=str(output_dir / 'segment_statistics.png')
    )
    
    # 计算评估指标
    print('\nEvaluation Metrics:')
    print('-' * 40)
    metrics = compute_evaluation_metrics(segments, positions)
    for key, value in metrics.items():
        if isinstance(value, float):
            print('  %s: %.4f' % (key, value))
        else:
            print('  %s: %s' % (key, value))
    
    # 保存指标
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('\nMetrics saved to: %s' % (output_dir / 'evaluation_metrics.json'))
    
    print('\n' + '=' * 60)
    print('Visualization complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
