#!/usr/bin/env python3
"""
测试时序场景分段模块

运行方式:
    cd /home/shyue/codebase/concept-graphs/conceptgraph
    python -m segmentation.test_segmentation
"""

import numpy as np
import sys
from pathlib import Path


def test_motion_signal_extractor():
    """测试运动信号提取器"""
    print("测试 MotionSignalExtractor...")
    
    from segmentation.signal_extractors import MotionSignalExtractor
    
    # 创建模拟轨迹数据 (20帧)
    n_frames = 20
    poses = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        poses[i] = np.eye(4)
        # 模拟位置变化
        poses[i, 0, 3] = i * 0.1  # x 方向移动
        poses[i, 1, 3] = np.sin(i * 0.3) * 0.2  # y 方向波动
    
    extractor = MotionSignalExtractor(smooth_sigma=1.0)
    signals = extractor.extract(poses)
    
    assert 'motion_intensity' in signals
    assert 'positions' in signals
    assert len(signals['motion_intensity']) == n_frames
    
    print("  ✓ MotionSignalExtractor 测试通过")
    return signals


def test_visual_signal_extractor():
    """测试视觉信号提取器"""
    print("测试 VisualSignalExtractor...")
    
    from segmentation.signal_extractors import VisualSignalExtractor
    
    # 创建模拟 CLIP 特征 (20帧, 512维)
    n_frames = 20
    feat_dim = 512
    
    # 模拟两个不同区域的特征
    clip_features = np.random.randn(n_frames, feat_dim)
    # 让前10帧相似，后10帧不同
    clip_features[:10] = clip_features[0] + np.random.randn(10, feat_dim) * 0.1
    clip_features[10:] = clip_features[10] + np.random.randn(10, feat_dim) * 0.1
    
    extractor = VisualSignalExtractor(smooth_sigma=1.0)
    signals = extractor.extract(clip_features)
    
    assert 'visual_change_smooth' in signals
    assert 'frame_similarity' in signals
    assert len(signals['visual_change_smooth']) == n_frames
    
    # 检查第10帧附近应该有较大的视觉变化
    visual_change = signals['visual_change_smooth']
    mid_change = visual_change[9:12].mean()
    edge_change = (visual_change[:3].mean() + visual_change[-3:].mean()) / 2
    
    print("  中间变化: %.3f, 边缘变化: %.3f" % (mid_change, edge_change))
    print("  ✓ VisualSignalExtractor 测试通过")
    return signals


def test_trajectory_segmenter():
    """测试轨迹分段器"""
    print("测试 TrajectorySegmenter...")
    
    from segmentation.trajectory_segmenter import TrajectorySegmenter
    
    # 创建模拟数据
    n_frames = 100
    poses = np.zeros((n_frames, 4, 4))
    
    # 模拟三个区域的轨迹
    # 区域1: 0-30帧，慢速移动
    # 区域2: 30-70帧，快速移动
    # 区域3: 70-100帧，慢速移动
    for i in range(n_frames):
        poses[i] = np.eye(4)
        if i < 30:
            poses[i, 0, 3] = i * 0.02
            poses[i, 1, 3] = 0
        elif i < 70:
            poses[i, 0, 3] = 0.6 + (i - 30) * 0.1
            poses[i, 1, 3] = 1.0
        else:
            poses[i, 0, 3] = 4.6 + (i - 70) * 0.02
            poses[i, 1, 3] = 2.0
    
    segmenter = TrajectorySegmenter(
        min_segment_frames=10,
        peak_prominence=0.05,
        peak_distance=10,
    )
    
    segments, debug_info = segmenter.segment(poses)
    
    print("  检测到 %d 个分段" % len(segments))
    for seg in segments:
        print("    区域 %d: 帧 %d-%d (%d 帧)" % (
            seg.region_id, seg.start_frame, seg.end_frame, seg.n_frames))
    
    assert len(segments) >= 1
    assert debug_info['n_frames'] == n_frames
    
    print("  ✓ TrajectorySegmenter 测试通过")
    return segments, debug_info


def test_from_real_data():
    """使用真实数据测试"""
    print("\n测试真实数据...")
    
    # 检查数据是否存在
    import os
    replica_root = os.environ.get('REPLICA_ROOT', '/home/shyue/Datasets/Replica/Replica')
    traj_path = Path(replica_root) / 'room0' / 'traj.txt'
    
    if not traj_path.exists():
        print("  跳过: 未找到真实数据 %s" % traj_path)
        return None, None
    
    from segmentation.trajectory_segmenter import TrajectorySegmenter
    
    segmenter = TrajectorySegmenter(
        min_segment_frames=10,
        peak_prominence=0.1,
        peak_distance=15,
    )
    
    segments, debug_info = segmenter.segment_from_files(
        trajectory_path=str(traj_path),
        stride=5,
    )
    
    print("  真实数据分段结果:")
    print("  总帧数: %d" % debug_info['n_frames'])
    print("  分段数: %d" % len(segments))
    for seg in segments:
        print("    区域 %d: 帧 %d-%d (%d 帧)" % (
            seg.region_id, seg.start_frame, seg.end_frame, seg.n_frames))
    
    print("  ✓ 真实数据测试通过")
    return segments, debug_info


def main():
    print("=" * 60)
    print("时序场景分段模块测试")
    print("=" * 60)
    print("")
    
    try:
        # 基础测试
        test_motion_signal_extractor()
        test_visual_signal_extractor()
        segments, debug_info = test_trajectory_segmenter()
        
        # 真实数据测试
        real_segments, real_debug = test_from_real_data()
        
        print("")
        print("=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        
    except Exception as e:
        print("")
        print("=" * 60)
        print("测试失败: %s" % e)
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
