#!/usr/bin/env python3
"""
高级视觉特征分析模块

改进的 CLIP 特征分析策略：
1. 多尺度窗口分析 - 检测不同时间尺度的变化
2. 帧间差异累积 - 追踪渐进式的场景变化
3. 关键帧检测 - 识别视觉转折点
4. 物体级特征分析 - 分析单个物体的特征变化
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional
import gzip
import pickle
from pathlib import Path
from natsort import natsorted


class AdvancedVisualAnalyzer:
    """高级视觉特征分析器"""
    
    def __init__(
        self,
        windows: List[int] = [5, 10, 20],  # 多尺度窗口
        smooth_sigma: float = 2.0,
        keyframe_threshold: float = 0.1,
    ):
        self.windows = windows
        self.smooth_sigma = smooth_sigma
        self.keyframe_threshold = keyframe_threshold
    
    def load_features(self, gsa_files: List[Path]) -> Dict:
        """加载所有帧的特征"""
        n_frames = len(gsa_files)
        
        frame_features = []  # 帧级聚合特征
        object_features = []  # 物体级特征列表
        detection_counts = []
        all_classes = []  # 类别信息
        confidences = []
        
        for gsa_file in gsa_files:
            with gzip.open(gsa_file, 'rb') as f:
                data = pickle.load(f)
            
            # 物体级特征
            feats = data.get('image_feats', np.zeros((1, 1024)))
            conf = data.get('confidence', np.ones(len(feats)) if len(feats) > 0 else np.array([1.0]))
            classes = data.get('classes', [])
            
            if len(feats) > 0:
                # 加权平均得到帧级特征
                weights = conf / (conf.sum() + 1e-8)
                frame_feat = np.average(feats, axis=0, weights=weights)
                frame_features.append(frame_feat)
                object_features.append(feats)
                confidences.append(conf)
            else:
                frame_features.append(np.zeros(1024))
                object_features.append(np.zeros((1, 1024)))
                confidences.append(np.array([1.0]))
            
            detection_counts.append(len(data.get('xyxy', [])))
            all_classes.append(classes if isinstance(classes, list) else [])
        
        # 归一化帧特征
        frame_features = np.array(frame_features)
        norms = np.linalg.norm(frame_features, axis=1, keepdims=True)
        frame_features_norm = frame_features / (norms + 1e-8)
        
        return {
            'frame_features': frame_features,
            'frame_features_norm': frame_features_norm,
            'object_features': object_features,
            'detection_counts': np.array(detection_counts),
            'classes': all_classes,
            'confidences': confidences,
            'n_frames': n_frames,
        }
    
    def compute_multiscale_similarity(self, features_norm: np.ndarray) -> Dict[int, np.ndarray]:
        """
        多尺度相似度分析
        
        使用不同窗口大小检测不同时间尺度的变化:
        - 小窗口 (5帧): 检测快速的视角变化
        - 中窗口 (10帧): 检测区域切换
        - 大窗口 (20帧): 检测场景级变化
        """
        n_frames = len(features_norm)
        multiscale_sims = {}
        
        for window in self.windows:
            similarities = []
            for i in range(n_frames):
                if i < window:
                    # 与窗口起始帧比较
                    sim = np.dot(features_norm[0], features_norm[i])
                else:
                    # 与 window 帧前比较
                    sim = np.dot(features_norm[i - window], features_norm[i])
                similarities.append(float(sim))
            
            multiscale_sims[window] = np.array(similarities)
        
        return multiscale_sims
    
    def compute_cumulative_change(self, features_norm: np.ndarray) -> np.ndarray:
        """
        累积变化追踪
        
        追踪从起始帧到当前帧的累积变化，
        用于检测渐进式的场景转换
        """
        n_frames = len(features_norm)
        
        # 与起始帧的相似度
        start_similarities = np.array([
            np.dot(features_norm[0], features_norm[i]) 
            for i in range(n_frames)
        ])
        
        # 累积变化 = 1 - 与起始帧的相似度
        cumulative_change = 1 - start_similarities
        
        return cumulative_change
    
    def compute_object_diversity(self, object_features: List[np.ndarray]) -> np.ndarray:
        """
        物体多样性分析
        
        计算每帧内物体特征的多样性，
        高多样性可能指示复杂场景或场景边界
        """
        diversities = []
        
        for feats in object_features:
            if len(feats) <= 1:
                diversities.append(0.0)
                continue
            
            # 归一化
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats_norm = feats / (norms + 1e-8)
            
            # 计算物体间的平均相似度
            n_objects = len(feats_norm)
            total_sim = 0
            count = 0
            for i in range(n_objects):
                for j in range(i + 1, n_objects):
                    total_sim += np.dot(feats_norm[i], feats_norm[j])
                    count += 1
            
            if count > 0:
                avg_sim = total_sim / count
                diversity = 1 - avg_sim  # 多样性 = 1 - 平均相似度
            else:
                diversity = 0.0
            
            diversities.append(diversity)
        
        return np.array(diversities)
    
    def detect_keyframes(self, multiscale_sims: Dict[int, np.ndarray]) -> List[int]:
        """
        关键帧检测
        
        综合多尺度信号检测关键帧
        """
        n_frames = len(list(multiscale_sims.values())[0])
        
        # 计算多尺度变化信号
        change_signals = []
        for window, sims in multiscale_sims.items():
            change = 1 - sims
            change_smooth = gaussian_filter1d(change, self.smooth_sigma)
            change_norm = change_smooth / (change_smooth.max() + 1e-8)
            change_signals.append(change_norm)
        
        # 融合多尺度信号 (加权平均，大窗口权重更高)
        weights = np.array([1.0, 1.5, 2.0])[:len(change_signals)]
        weights = weights / weights.sum()
        
        fused_signal = np.zeros(n_frames)
        for i, signal in enumerate(change_signals):
            fused_signal += weights[i] * signal
        
        # 检测峰值
        gradient = np.abs(np.gradient(fused_signal))
        peaks, _ = find_peaks(gradient, prominence=0.03, distance=8)
        
        return peaks.tolist(), fused_signal, change_signals
    
    def analyze_class_transitions(self, all_classes: List[List[str]]) -> Dict:
        """
        类别转换分析
        
        追踪物体类别的出现和消失
        """
        n_frames = len(all_classes)
        
        # 累积类别
        seen_classes = set()
        new_class_counts = []
        disappeared_class_counts = []
        
        prev_classes = set()
        
        for frame_classes in all_classes:
            current_classes = set(frame_classes) if isinstance(frame_classes, list) else set()
            
            # 新出现的类别
            new_classes = current_classes - seen_classes
            new_class_counts.append(len(new_classes))
            seen_classes.update(current_classes)
            
            # 消失的类别 (相对于前一帧)
            disappeared = prev_classes - current_classes
            disappeared_class_counts.append(len(disappeared))
            
            prev_classes = current_classes
        
        # Jaccard 距离
        class_change = [0.0]
        for i in range(1, n_frames):
            prev = set(all_classes[i-1]) if isinstance(all_classes[i-1], list) else set()
            curr = set(all_classes[i]) if isinstance(all_classes[i], list) else set()
            
            union = len(prev | curr)
            intersection = len(prev & curr)
            
            if union > 0:
                jaccard_dist = 1 - intersection / union
            else:
                jaccard_dist = 0
            
            class_change.append(jaccard_dist)
        
        return {
            'new_class_counts': np.array(new_class_counts),
            'disappeared_class_counts': np.array(disappeared_class_counts),
            'class_change': np.array(class_change),
            'total_classes': len(seen_classes),
            'seen_classes': list(seen_classes),
        }
    
    def analyze(self, gsa_files: List[Path]) -> Dict:
        """执行完整的视觉分析"""
        
        print("  [1] 加载特征...")
        data = self.load_features(gsa_files)
        
        print("  [2] 多尺度相似度分析...")
        multiscale_sims = self.compute_multiscale_similarity(data['frame_features_norm'])
        
        print("  [3] 累积变化追踪...")
        cumulative_change = self.compute_cumulative_change(data['frame_features_norm'])
        
        print("  [4] 物体多样性分析...")
        object_diversity = self.compute_object_diversity(data['object_features'])
        
        print("  [5] 关键帧检测...")
        keyframes, fused_signal, change_signals = self.detect_keyframes(multiscale_sims)
        
        print("  [6] 类别转换分析...")
        class_analysis = self.analyze_class_transitions(data['classes'])
        
        return {
            'n_frames': data['n_frames'],
            'detection_counts': data['detection_counts'].tolist(),
            
            # 多尺度相似度
            'multiscale_similarities': {k: v.tolist() for k, v in multiscale_sims.items()},
            'multiscale_changes': {k: (1 - v).tolist() for k, v in multiscale_sims.items()},
            
            # 累积变化
            'cumulative_change': cumulative_change.tolist(),
            
            # 物体多样性
            'object_diversity': object_diversity.tolist(),
            
            # 融合信号
            'fused_visual_signal': fused_signal.tolist(),
            'change_signals': [s.tolist() for s in change_signals],
            
            # 关键帧
            'keyframes': keyframes,
            
            # 类别分析
            'class_analysis': {
                'new_class_counts': class_analysis['new_class_counts'].tolist(),
                'class_change': class_analysis['class_change'].tolist(),
                'total_classes': class_analysis['total_classes'],
                'seen_classes': class_analysis['seen_classes'],
            },
        }


def main():
    """测试高级视觉分析"""
    import json
    import matplotlib.pyplot as plt
    
    REPLICA_ROOT = '/home/shyue/Datasets/Replica/Replica'
    scene = 'room0'
    
    gsa_dir = Path(f'{REPLICA_ROOT}/{scene}/gsa_detections_none')
    gsa_files = natsorted(gsa_dir.glob('frame*.pkl.gz'))
    
    print(f"分析 {len(gsa_files)} 帧...")
    
    analyzer = AdvancedVisualAnalyzer(windows=[5, 10, 20])
    results = analyzer.analyze(gsa_files)
    
    # 保存结果
    output_dir = Path(f'{REPLICA_ROOT}/{scene}/sg_cache/visual_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'advanced_visual_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    n_frames = results['n_frames']
    frames = np.arange(n_frames)
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    
    # 1. 多尺度相似度变化
    ax = axes[0]
    colors = ['blue', 'green', 'red']
    for i, (window, changes) in enumerate(results['multiscale_changes'].items()):
        ax.plot(frames, changes, color=colors[i], lw=1.5, alpha=0.8, label=f'Window={window}')
    ax.set_ylabel('Visual Change')
    ax.set_title('Multi-Scale Visual Change (CLIP Features)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 融合视觉信号
    ax = axes[1]
    ax.plot(frames, results['fused_visual_signal'], 'purple', lw=2)
    ax.fill_between(frames, 0, results['fused_visual_signal'], alpha=0.3, color='purple')
    for kf in results['keyframes']:
        ax.axvline(x=kf, color='red', ls='--', alpha=0.7)
    ax.set_ylabel('Fused Signal')
    ax.set_title(f"Fused Visual Signal ({len(results['keyframes'])} Keyframes)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 累积变化
    ax = axes[2]
    ax.plot(frames, results['cumulative_change'], 'orange', lw=2)
    ax.fill_between(frames, 0, results['cumulative_change'], alpha=0.3, color='orange')
    ax.set_ylabel('Cumulative Change')
    ax.set_title('Cumulative Change from Start Frame', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 物体多样性
    ax = axes[3]
    ax.plot(frames, results['object_diversity'], 'green', lw=2)
    ax.fill_between(frames, 0, results['object_diversity'], alpha=0.3, color='green')
    ax.set_ylabel('Diversity')
    ax.set_title('Object Feature Diversity per Frame', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. 检测数量
    ax = axes[4]
    ax.bar(frames, results['detection_counts'], width=1.0, alpha=0.6, color='gray')
    ax.set_ylabel('Count')
    ax.set_xlabel('Frame Index')
    ax.set_title('Detection Count per Frame', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/shyue/codebase/concept-graphs/room0_seg/advanced_visual_analysis.png', 
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: advanced_visual_analysis.png")
    print(f"Results: {output_dir / 'advanced_visual_analysis.json'}")


if __name__ == '__main__':
    main()
