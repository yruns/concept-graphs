#!/usr/bin/env python3
"""
时序场景分段命令行工具

用法:
    python segment_trajectory.py --trajectory <traj.txt> --output <output_dir>
    
示例:
    python segment_trajectory.py \
        --trajectory /path/to/room0/traj.txt \
        --mapfile /path/to/room0/pcd_saves/xxx.pkl.gz \
        --gsa_results /path/to/room0/gsa_detections_none \
        --output /path/to/room0/sg_cache/segmentation \
        --stride 5 \
        --visualize
"""

import argparse
import sys
import os
from pathlib import Path

# 添加父目录到路径
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))


def main():
    parser = argparse.ArgumentParser(description='时序场景分段')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='轨迹文件路径 (traj.txt)')
    parser.add_argument('--mapfile', type=str, default=None,
                        help='3D对象地图文件路径 (*.pkl.gz)')
    parser.add_argument('--gsa_results', type=str, default=None,
                        help='GSA检测结果目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--stride', type=int, default=5,
                        help='帧采样步长 (默认: 5)')
    parser.add_argument('--min_segment_frames', type=int, default=10,
                        help='最小分段帧数 (默认: 10)')
    parser.add_argument('--motion_weight', type=float, default=0.5,
                        help='运动信号权重 (默认: 0.5)')
    parser.add_argument('--visual_weight', type=float, default=0.3,
                        help='视觉信号权重 (默认: 0.3)')
    parser.add_argument('--semantic_weight', type=float, default=0.2,
                        help='语义信号权重 (默认: 0.2)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图片')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.trajectory).exists():
        print('Error: trajectory file not found: %s' % args.trajectory)
        sys.exit(1)
    
    # 导入分段器 (使用绝对导入)
    from segmentation.trajectory_segmenter import TrajectorySegmenter
    from segmentation.visualizer import plot_signals, generate_summary
    
    print('=' * 60)
    print('时序场景分段')
    print('=' * 60)
    print('轨迹文件: %s' % args.trajectory)
    print('地图文件: %s' % (args.mapfile or '未指定'))
    print('GSA结果: %s' % (args.gsa_results or '未指定'))
    print('输出目录: %s' % args.output)
    print('采样步长: %d' % args.stride)
    print('=' * 60)
    
    # 创建分段器
    segmenter = TrajectorySegmenter(
        motion_weight=args.motion_weight,
        visual_weight=args.visual_weight,
        semantic_weight=args.semantic_weight,
        min_segment_frames=args.min_segment_frames,
    )
    
    # 执行分段
    segments, debug_info = segmenter.segment_from_files(
        trajectory_path=args.trajectory,
        map_file_path=args.mapfile,
        gsa_results_path=args.gsa_results,
        stride=args.stride,
    )
    
    # 保存结果
    segmenter.save_results(segments, debug_info, args.output)
    
    # 生成摘要
    summary_path = Path(args.output) / 'segmentation_report.txt'
    generate_summary(segments, debug_info, str(summary_path))
    
    # 可视化
    if args.visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig_path = Path(args.output) / 'segmentation_signals.png'
            plot_signals(debug_info, segments, str(fig_path))
        except Exception as e:
            print('Warning: visualization failed: %s' % e)
    
    print('')
    print('=' * 60)
    print('分段完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()
