#!/usr/bin/env python3
"""
Region-Aware Scene Segmenter (Step 4.5 Enhanced)
================================================
两阶段场景划分:
1. 时序分割: 检测连续变化点
2. 片段聚类: 将相似片段合并为逻辑场景区域

支持识别不连续但属于同一区域的片段
"""
import argparse
import gzip
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


@dataclass 
class RegionSegmenterConfig:
    # 信号权重
    trajectory_weight: float = 0.15
    object_density_weight: float = 0.20
    visibility_weight: float = 0.35
    semantic_weight: float = 0.30
    
    # 时序分割参数
    peak_distance: int = 15
    min_segment_frames: int = 20
    
    # 区域聚类参数
    n_regions: int = 6
    merge_similarity: float = 0.5
    
    # 其他
    visibility_radius: float = 3.0
    smooth_sigma: float = 2.0


class RegionAwareSegmenter:
    """区域感知的场景分割器"""
    
    def __init__(self, config=None, stride=5):
        self.config = config or RegionSegmenterConfig()
        self.stride = stride
        self.poses = None
        self.objects = None
        self.captions = None
        self.signals = {}
        self.n_frames = 0
        
        # 可见性矩阵
        self.visibility_matrix = None
        
        # 多种相似度矩阵
        self.visibility_sim_matrix = None   # 基于物体可见性的 Jaccard 相似度
        self.clip_sim_matrix = None         # 基于 CLIP 特征的语义相似度
        self.similarity_matrix = None       # 融合相似度矩阵
        
        # CLIP 特征
        self.object_clip_features = None
        self.frame_clip_features = None  # 帧级 CLIP 特征
        
        # 结果
        self.temporal_segments = []
        self.region_labels = None
        self.regions = []
    
    def load_data(self, scene_path):
        """加载所有数据"""
        scene_path = Path(scene_path)
        
        # 位姿
        pose_file = scene_path / 'traj.txt'
        if not pose_file.exists():
            pose_file = scene_path / 'traj_w_c.txt'
        
        all_poses = []
        with open(pose_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    all_poses.append([float(p) for p in parts[:7]])
        self.poses = np.array(all_poses[::self.stride])
        self.n_frames = len(self.poses)
        print(f"  位姿: {self.n_frames} 帧 (stride={self.stride})")
        
        # 3D物体地图
        pcd_files = list((scene_path / 'pcd_saves').glob('*_post.pkl.gz'))
        if not pcd_files:
            pcd_files = list((scene_path / 'pcd_saves').glob('*.pkl.gz'))
        if pcd_files:
            with gzip.open(pcd_files[0], 'rb') as f:
                data = pickle.load(f)
            self.objects = data.get('objects', [])
            print(f"  3D物体: {len(self.objects)} 个")
            
            # 物体中心
            self.object_centers = []
            for obj in self.objects:
                if 'pcd_np' in obj:
                    center = np.mean(obj['pcd_np'], axis=0)
                else:
                    center = np.zeros(3)
                self.object_centers.append(center)
            self.object_centers = np.array(self.object_centers)
            
            # 加载 CLIP 特征
            self.object_clip_features = []
            for obj in self.objects:
                if 'clip_ft' in obj and obj['clip_ft'] is not None:
                    clip_ft = obj['clip_ft']
                    # 处理 tensor 或 numpy
                    if hasattr(clip_ft, 'cpu'):
                        clip_ft = clip_ft.cpu().numpy()
                    self.object_clip_features.append(clip_ft.flatten())
                else:
                    self.object_clip_features.append(None)
            
            n_with_clip = sum(1 for f in self.object_clip_features if f is not None)
            print(f"  CLIP特征: {n_with_clip}/{len(self.objects)} 个物体")
        
        # 加载帧级 CLIP 特征 (从 gsa_detections_none)
        # 注意：GSA 文件已经是按 stride 采样后的，文件名中的帧号对应原始视频帧
        gsa_dir = scene_path / 'gsa_detections_none'
        if gsa_dir.exists():
            gsa_files = sorted(gsa_dir.glob('*.pkl.gz'))
            self.frame_clip_features = []
            n_with_frame_clip = 0
            
            # GSA 文件数量应该与 n_frames 相近（或更多）
            # 不需要再次 stride 采样，因为 GSA 文件本身就是采样后的
            for i in range(min(self.n_frames, len(gsa_files))):
                gsa_file = gsa_files[i]
                with gzip.open(gsa_file, 'rb') as f:
                    gsa_data = pickle.load(f)
                if 'frame_clip_feat' in gsa_data:
                    self.frame_clip_features.append(gsa_data['frame_clip_feat'])
                    n_with_frame_clip += 1
                else:
                    self.frame_clip_features.append(None)
            
            # 填充到 n_frames
            while len(self.frame_clip_features) < self.n_frames:
                self.frame_clip_features.append(None)
            
            print(f"  帧级CLIP特征: {n_with_frame_clip}/{self.n_frames} 帧")
        
        # 物体描述
        cap_file = scene_path / 'sg_cache' / 'cfslam_llava_captions.json'
        if cap_file.exists():
            with open(cap_file) as f:
                self.captions = json.load(f)
            print(f"  物体描述: {len(self.captions)} 个")
    
    def build_visibility_matrix(self):
        """构建物体可见性矩阵"""
        if not self.objects:
            return
        
        n_objects = len(self.objects)
        self.visibility_matrix = np.zeros((self.n_frames, n_objects))
        
        for obj_id, obj in enumerate(self.objects):
            if 'image_idx' in obj:
                for frame_idx in obj['image_idx']:
                    if 0 <= frame_idx < self.n_frames:
                        self.visibility_matrix[frame_idx, obj_id] = 1
        
        print(f"  可见性矩阵: {self.visibility_matrix.shape}")
    
    def compute_frame_similarity(self):
        """计算帧间相似度矩阵（可见性 + CLIP 语义）"""
        if self.visibility_matrix is None:
            return
        
        n = self.n_frames
        
        # 1. 可见性相似度矩阵 (Jaccard)
        self.visibility_sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                v1, v2 = self.visibility_matrix[i], self.visibility_matrix[j]
                intersection = np.sum(np.logical_and(v1, v2))
                union = np.sum(np.logical_or(v1, v2))
                sim = intersection / union if union > 0 else 0
                self.visibility_sim_matrix[i, j] = sim
                self.visibility_sim_matrix[j, i] = sim
        print(f"  可见性相似度矩阵计算完成")
        
        # 2. CLIP 语义相似度矩阵
        # 优先使用帧级 CLIP 特征（直接对整帧图像编码）
        self.clip_sim_matrix = np.zeros((n, n))
        
        if self.frame_clip_features is not None and any(f is not None for f in self.frame_clip_features):
            # 方法 A：使用帧级 CLIP 特征（推荐）
            print(f"  使用帧级CLIP特征计算语义相似度...")
            for i in range(n):
                for j in range(i, n):
                    f1, f2 = self.frame_clip_features[i], self.frame_clip_features[j]
                    if f1 is not None and f2 is not None:
                        # 余弦相似度（特征已经归一化）
                        sim = np.dot(f1, f2)
                        # 归一化到 [0, 1]（余弦相似度范围是 [-1, 1]）
                        sim = (sim + 1) / 2
                    else:
                        sim = 0.5  # 默认中等相似度
                    self.clip_sim_matrix[i, j] = sim
                    self.clip_sim_matrix[j, i] = sim
        else:
            # 方法 B：基于物体的"变化语义距离"（备用方案）
            print(f"  帧级CLIP特征不可用，使用物体级特征...")
            n_objects = len(self.object_clip_features) if self.object_clip_features else 0
            if n_objects > 0:
                clip_features_array = np.zeros((n_objects, len(self.object_clip_features[0]) if self.object_clip_features[0] is not None else 1024))
                for i, ft in enumerate(self.object_clip_features):
                    if ft is not None:
                        clip_features_array[i] = ft
                
                for i in range(n):
                    for j in range(i, n):
                        visible_i = set(np.where(self.visibility_matrix[i] > 0)[0])
                        visible_j = set(np.where(self.visibility_matrix[j] > 0)[0])
                        
                        leaving = visible_i - visible_j
                        entering = visible_j - visible_i
                        staying = visible_i & visible_j
                        
                        if len(leaving) == 0 and len(entering) == 0:
                            sim = 1.0
                        else:
                            total_objs = len(visible_i | visible_j)
                            change_ratio = (len(leaving) + len(entering)) / max(total_objs, 1)
                            
                            if len(staying) > 0 and (len(leaving) > 0 or len(entering) > 0):
                                change_objs = list(leaving | entering)
                                stay_objs = list(staying)
                                dists = []
                                for co in change_objs:
                                    for so in stay_objs:
                                        cf, sf = clip_features_array[co], clip_features_array[so]
                                        norm_c, norm_s = np.linalg.norm(cf), np.linalg.norm(sf)
                                        if norm_c > 0 and norm_s > 0:
                                            cos_sim = np.dot(cf, sf) / (norm_c * norm_s)
                                            dists.append(1 - cos_sim)
                                avg_dist = np.mean(dists) if dists else 0
                                sim = 1 - change_ratio * (0.5 + 0.5 * avg_dist)
                            else:
                                sim = 1 - change_ratio
                            sim = max(0, min(1, sim))
                        
                        self.clip_sim_matrix[i, j] = sim
                        self.clip_sim_matrix[j, i] = sim
        
        # 统计并打印区分度
        off_diag = self.clip_sim_matrix[np.triu_indices(n, k=1)]
        print(f"  CLIP语义相似度: mean={off_diag.mean():.3f}, std={off_diag.std():.3f}, range=[{off_diag.min():.3f}, {off_diag.max():.3f}]")
        
        # 3. 融合相似度矩阵 (可见性 50% + CLIP 50%)
        alpha = 0.5
        self.similarity_matrix = alpha * self.visibility_sim_matrix + (1 - alpha) * self.clip_sim_matrix
        print(f"  融合相似度矩阵计算完成 (可见性:{alpha:.0%} + CLIP:{1-alpha:.0%})")
    
    def compute_signals(self):
        """计算各类变化信号"""
        n = self.n_frames
        
        # 1. 轨迹信号
        positions = self.poses[:, :3]
        pos_diff = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        pos_diff = np.concatenate([[0], pos_diff])
        
        quaternions = self.poses[:, 3:7]
        angle_diff = [0]
        for i in range(1, n):
            q1, q2 = quaternions[i-1], quaternions[i]
            dot = np.clip(np.abs(np.dot(q1, q2)), -1, 1)
            angle_diff.append(2 * np.arccos(dot))
        angle_diff = np.array(angle_diff)
        
        pos_norm = pos_diff / (np.max(pos_diff) + 1e-8)
        angle_norm = angle_diff / (np.max(angle_diff) + 1e-8)
        self.signals['trajectory'] = 0.6 * pos_norm + 0.4 * angle_norm
        
        # 2. 物体密度信号
        if self.object_centers is not None and len(self.object_centers) > 0:
            radius = self.config.visibility_radius
            density = []
            for i in range(n):
                cam_pos = self.poses[i, :3]
                distances = np.linalg.norm(self.object_centers - cam_pos, axis=1)
                density.append(np.sum(distances < radius))
            density = np.array(density, dtype=float)
            density_smooth = gaussian_filter1d(density, sigma=3)
            density_diff = np.abs(np.diff(density_smooth))
            density_diff = np.concatenate([[0], density_diff])
            if np.max(density_diff) > 0:
                density_diff /= np.max(density_diff)
            self.signals['object_density'] = density_diff
        else:
            self.signals['object_density'] = np.zeros(n)
        
        # 3. 可见性变化信号 (多尺度)
        if self.visibility_matrix is not None:
            frame_objects = [set(np.where(self.visibility_matrix[i] > 0)[0]) 
                           for i in range(n)]
            
            windows = [3, 8, 15]
            multiscale = []
            for window in windows:
                signal = [0.0] * window
                for i in range(window, n):
                    s1, s2 = frame_objects[i-window], frame_objects[i]
                    union = len(s1 | s2)
                    inter = len(s1 & s2)
                    signal.append(1 - inter/union if union > 0 else 0)
                multiscale.append(np.array(signal))
            
            fused = 0.2*multiscale[0] + 0.4*multiscale[1] + 0.4*multiscale[2]
            if np.max(fused) > 0:
                fused /= np.max(fused)
            self.signals['visibility'] = fused
            self.signals['frame_objects'] = frame_objects
        else:
            self.signals['visibility'] = np.zeros(n)
        
        # 4. 语义变化信号 (基于 CLIP 特征)
        if self.clip_sim_matrix is not None:
            # 使用 CLIP 相似度矩阵计算语义变化
            # 语义变化 = 1 - 相似度 (相似度低 = 变化大)
            windows = [3, 8, 15]
            multiscale = []
            
            for window in windows:
                signal = [0.0] * window
                for i in range(window, n):
                    # 语义距离 = 1 - CLIP相似度
                    semantic_distance = 1 - self.clip_sim_matrix[i-window, i]
                    signal.append(semantic_distance)
                multiscale.append(np.array(signal))
            
            fused = 0.2*multiscale[0] + 0.4*multiscale[1] + 0.4*multiscale[2]
            if np.max(fused) > 0:
                fused /= np.max(fused)
            self.signals['semantic'] = fused
        else:
            self.signals['semantic'] = np.zeros(n)
    
    def temporal_segmentation(self):
        """第一阶段: 时序分割"""
        n = self.n_frames
        cfg = self.config
        
        # 融合信号
        fused = (cfg.trajectory_weight * self.signals.get('trajectory', np.zeros(n)) +
                cfg.object_density_weight * self.signals.get('object_density', np.zeros(n)) +
                cfg.visibility_weight * self.signals.get('visibility', np.zeros(n)) +
                cfg.semantic_weight * self.signals.get('semantic', np.zeros(n)))
        fused = gaussian_filter1d(fused, sigma=cfg.smooth_sigma)
        self.signals['fused'] = fused
        
        # 检测变化点
        gradient = gaussian_filter1d(np.abs(np.gradient(fused)), sigma=2)
        self.signals['gradient'] = gradient
        
        peaks, _ = find_peaks(gradient, prominence=0.02, distance=cfg.peak_distance)
        
        # 创建时序片段
        boundaries = [0] + sorted(peaks.tolist()) + [n]
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if end - start >= cfg.min_segment_frames:
                segments.append({'start': start, 'end': end, 'n_frames': end - start})
            elif segments:
                segments[-1]['end'] = end
                segments[-1]['n_frames'] = segments[-1]['end'] - segments[-1]['start']
        
        self.temporal_segments = segments
        self.temporal_segments_boundaries = peaks.tolist()
        print(f"  时序分割: {len(segments)} 个连续片段")
        return segments
    

    def auto_detect_n_regions(self, min_k=3, max_k=10):
        """自动检测最佳区域数量"""
        n_segs = len(self.temporal_segments)
        if self.similarity_matrix is None or n_segs < 2:
            return min(min_k, n_segs)
        
        max_k = min(max_k, n_segs)
        if max_k < min_k:
            return n_segs
        
        # 计算片段间相似度
        seg_sim = np.zeros((n_segs, n_segs))
        for i in range(n_segs):
            for j in range(i, n_segs):
                si, sj = self.temporal_segments[i], self.temporal_segments[j]
                sims = [self.similarity_matrix[fi, fj] 
                       for fi in range(si['start'], si['end'])
                       for fj in range(sj['start'], sj['end'])]
                seg_sim[i, j] = seg_sim[j, i] = np.mean(sims) if sims else 0
        
        seg_dist = 1 - seg_sim
        best_k, best_score = min_k, -1
        
        for k in range(min_k, max_k + 1):
            labels = AgglomerativeClustering(n_clusters=k, metric='precomputed', 
                                            linkage='average').fit_predict(seg_dist)
            # 综合质量分数
            intra = np.mean([np.mean(seg_sim[labels==c][:, labels==c]) 
                            for c in range(k) if np.sum(labels==c) > 1] or [0])
            inter_vals = [seg_dist[labels==c1][:, labels==c2].mean() 
                         for c1 in range(k) for c2 in range(c1+1, k)]
            inter = np.mean(inter_vals) if inter_vals else 0
            sizes = np.array([np.sum(labels==c) for c in range(k)]) / n_segs
            balance = -np.sum(sizes * np.log(sizes + 1e-10)) / np.log(k) if k > 1 else 1
            
            score = 0.4*intra + 0.35*inter + 0.25*balance
            if score > best_score:
                best_score, best_k = score, k
        
        print(f"  自动检测: {best_k} 个区域 (分数={best_score:.3f})")
        return best_k


    def cluster_segments(self):
        """第二阶段: 片段聚类"""
        if self.similarity_matrix is None or len(self.temporal_segments) < 2:
            # 没有相似度矩阵或只有1个片段，直接返回
            self.region_labels = np.arange(self.n_frames)
            for i, seg in enumerate(self.temporal_segments):
                for j in range(seg['start'], seg['end']):
                    self.region_labels[j] = i
            return
        
        n_segs = len(self.temporal_segments)
        
        # 计算片段间相似度
        seg_similarity = np.zeros((n_segs, n_segs))
        for i in range(n_segs):
            for j in range(i, n_segs):
                # 取两个片段所有帧对的平均相似度
                seg_i = self.temporal_segments[i]
                seg_j = self.temporal_segments[j]
                
                sims = []
                for fi in range(seg_i['start'], seg_i['end']):
                    for fj in range(seg_j['start'], seg_j['end']):
                        sims.append(self.similarity_matrix[fi, fj])
                
                avg_sim = np.mean(sims) if sims else 0
                seg_similarity[i, j] = avg_sim
                seg_similarity[j, i] = avg_sim
        
        # 层次聚类
        # 自动检测最佳区域数
        n_clusters = self.auto_detect_n_regions(min_k=3, max_k=min(10, n_segs))
        if n_clusters < 2:
            n_clusters = 2
        
        distance = 1 - seg_similarity
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        seg_labels = clustering.fit_predict(distance)
        
        # 将片段标签映射到帧
        self.region_labels = np.zeros(self.n_frames, dtype=int)
        for seg_id, seg in enumerate(self.temporal_segments):
            for i in range(seg['start'], seg['end']):
                self.region_labels[i] = seg_labels[seg_id]
        
        print(f"  区域聚类: {n_clusters} 个逻辑区域")
        
        # 统计每个区域
        for region_id in range(n_clusters):
            frames = np.where(self.region_labels == region_id)[0]
            # 找不连续片段
            segments = []
            if len(frames) > 0:
                start = frames[0]
                for i in range(1, len(frames)):
                    if frames[i] - frames[i-1] > 1:
                        segments.append((start, frames[i-1]))
                        start = frames[i]
                segments.append((start, frames[-1]))
            
            print(f"    区域 {region_id}: {len(frames)}帧, {len(segments)}个片段")
    
    def create_regions(self):
        """创建最终区域结构"""
        n_regions = len(set(self.region_labels))
        
        for region_id in range(n_regions):
            frames = np.where(self.region_labels == region_id)[0]
            if len(frames) == 0:
                continue
            
            # 识别不连续片段
            segments = []
            start = frames[0]
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 1:
                    segments.append({
                        'start_frame': int(start),
                        'end_frame': int(frames[i-1] + 1),
                        'start_frame_original': int(start * self.stride),
                        'end_frame_original': int((frames[i-1] + 1) * self.stride)
                    })
                    start = frames[i]
            segments.append({
                'start_frame': int(start),
                'end_frame': int(frames[-1] + 1),
                'start_frame_original': int(start * self.stride),
                'end_frame_original': int((frames[-1] + 1) * self.stride)
            })
            
            # 收集区域语义
            frame_sem = self.signals.get('frame_semantics', [])
            frame_obj = self.signals.get('frame_objects', [])
            
            region_sem = set()
            region_obj = set()
            for f in frames:
                if f < len(frame_sem):
                    region_sem |= frame_sem[f]
                if f < len(frame_obj):
                    region_obj |= frame_obj[f]
            
            region = {
                'region_id': region_id,
                'n_frames': int(len(frames)),
                'n_segments': len(segments),
                'segments': segments,
                'dominant_semantics': list(region_sem)[:6],
                'n_objects': len(region_obj),
                'object_ids': list(region_obj)[:15]
            }
            self.regions.append(region)
        
        return self.regions
    
    def segment(self):
        """执行完整分割流程"""
        print("\n[1/6] 构建可见性矩阵...")
        self.build_visibility_matrix()
        
        print("[2/6] 计算帧间相似度...")
        self.compute_frame_similarity()
        
        print("[3/6] 计算变化信号...")
        self.compute_signals()
        
        print("[4/6] 时序分割...")
        self.temporal_segmentation()
        
        print("[5/6] 片段聚类...")
        self.cluster_segments()
        
        print("[6/6] 创建区域结构...")
        self.create_regions()
        
        print(f"\n✓ 完成: {len(self.regions)} 个逻辑场景区域")
        return self.regions
    
    def visualize(self, output_path):
        """可视化结果"""
        n = self.n_frames
        
        # 使用 3x3 布局
        fig = plt.figure(figsize=(18, 18))
        
        # === 第一行：三种相似度矩阵 ===
        
        # 1. 可见性相似度矩阵 (Jaccard)
        ax1 = fig.add_subplot(331)
        if self.visibility_sim_matrix is not None:
            im1 = ax1.imshow(self.visibility_sim_matrix, cmap='hot', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Visibility Similarity\n(Jaccard of visible objects)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Frame')
        
        # 2. CLIP 语义相似度矩阵
        ax2 = fig.add_subplot(332)
        if self.clip_sim_matrix is not None:
            im2 = ax2.imshow(self.clip_sim_matrix, cmap='hot', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        # 根据是否使用帧级 CLIP 特征来设置标题
        if self.frame_clip_features is not None and any(f is not None for f in self.frame_clip_features):
            ax2.set_title('CLIP Semantic Similarity\n(Frame-level CLIP features)', fontsize=11, fontweight='bold')
        else:
            ax2.set_title('CLIP Semantic Similarity\n(Object-level fallback)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Frame')
        
        # 3. 融合相似度矩阵
        ax3 = fig.add_subplot(333)
        if self.similarity_matrix is not None:
            im3 = ax3.imshow(self.similarity_matrix, cmap='hot', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title('Combined Similarity\n(50% Visibility + 50% CLIP)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Frame')
        
        # === 第二行：空间分布 + 区域统计 ===
        
        # 4. 空间分布 (正方形)
        ax4 = fig.add_subplot(334)
        positions = self.poses[:, :3]
        if self.region_labels is not None:
            scatter = ax4.scatter(positions[:, 0], positions[:, 2], 
                       c=self.region_labels, cmap='tab10', s=15, alpha=0.8)
            plt.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04, label='Region')
        ax4.set_aspect('equal', adjustable='box')
        ax4.set_title('Spatial Distribution\n(Camera trajectory colored by region)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        
        # 5. 区域分配时间线
        ax5 = fig.add_subplot(335)
        x = np.arange(n)
        if self.region_labels is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for r in self.regions:
                for seg in r['segments']:
                    ax5.axvspan(seg['start_frame'], seg['end_frame'], 
                               alpha=0.6, color=colors[r['region_id'] % 10],
                               label=f'Region {r["region_id"]}' if seg == r['segments'][0] else '')
            ax5.legend(loc='upper right', fontsize=8, ncol=2)
        ax5.set_xlim(0, n)
        ax5.set_ylim(0, 1)
        ax5.set_yticks([])
        ax5.set_title('Region Assignment Timeline', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Frame Index')
        
        # 6. 区域统计
        ax6 = fig.add_subplot(336)
        ax6.axis('off')
        text = "Logical Scene Regions:\n" + "="*35 + "\n\n"
        for r in self.regions:
            text += f"Region {r['region_id']}: {r['n_frames']} frames, {r['n_segments']} segment(s)\n"
            if r['dominant_semantics']:
                text += f"  → {', '.join(r['dominant_semantics'][:4])}\n"
            text += "\n"
        ax6.text(0.05, 0.95, text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title('Region Summary', fontsize=11, fontweight='bold')
        
        # === 第三行：多模态信号 (横跨底部) ===
        ax7 = fig.add_subplot(313)
        ax7.plot(x, self.signals.get('trajectory', np.zeros(n)), label='Trajectory (15%)', alpha=0.7, linewidth=1.5)
        ax7.plot(x, self.signals.get('object_density', np.zeros(n)), label='Object Density (20%)', alpha=0.7, linewidth=1.5)
        ax7.plot(x, self.signals.get('visibility', np.zeros(n)), label='Visibility (35%)', alpha=0.7, linewidth=1.5)
        ax7.plot(x, self.signals.get('semantic', np.zeros(n)), label='Semantic/CLIP (30%)', alpha=0.7, linewidth=1.5)
        ax7.plot(x, self.signals.get('fused', np.zeros(n)), label='Fused Signal', linewidth=2.5, color='red')
        
        # 添加区域背景色
        if self.region_labels is not None:
            for r in self.regions:
                for seg in r['segments']:
                    ax7.axvspan(seg['start_frame'], seg['end_frame'], 
                               alpha=0.1, color=colors[r['region_id'] % 10])
        
        # 添加边界线
        if hasattr(self, 'temporal_segments_boundaries'):
            for b in self.temporal_segments_boundaries:
                ax7.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax7.legend(loc='upper right', fontsize=10, ncol=3)
        ax7.set_title('Multi-modal Change Signals (with region backgrounds)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Frame Index', fontsize=11)
        ax7.set_ylabel('Signal Value (normalized)', fontsize=11)
        ax7.set_xlim(0, n)
        ax7.grid(axis='y', alpha=0.3, linestyle=':')
        
        plt.suptitle('Region-Aware Scene Segmentation Analysis', fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def generate_region_visualizations(scene_path, regions, output_dir, stride=5):
    """Generate GIF visualizations for each region"""
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)
    
    rgb_dir = scene_path / 'results'
    rgb_files = sorted(rgb_dir.glob('frame*.jpg'))
    
    if not rgb_files:
        print("  Warning: No RGB images found")
        return
    
    (output_dir / 'region_gifs').mkdir(exist_ok=True)
    (output_dir / 'region_keyframes').mkdir(exist_ok=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(regions), 1)))
    
    print(f"  Generating visualizations for {len(regions)} regions...")
    
    for region in regions:
        rid = region['region_id']
        segments = region['segments']
        semantics = region.get('dominant_semantics', [])[:4]
        
        # Collect frames
        frames_data = []
        for seg in segments:
            start = seg['start_frame_original']
            end = seg['end_frame_original']
            for idx in range(start, min(end, len(rgb_files)), stride * 2):
                frames_data.append({'idx': idx, 'path': rgb_files[idx]})
        
        if not frames_data:
            continue
        
        # Generate annotated frames
        annotated = []
        color_rgb = tuple((np.array(colors[rid % len(colors)][:3]) * 255).astype(np.uint8))
        
        for fd in frames_data:
            img = Image.open(fd['path'])
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            # Border
            for i in range(8):
                draw.rectangle([i, i, w-1-i, h-1-i], outline=color_rgb)
            
            # Annotation
            draw.rectangle([10, 10, 320, 75], fill=(0, 0, 0))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
                font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = font_s = ImageFont.load_default()
            
            draw.text((15, 12), f"Region {rid} | Frame {fd['idx']}", fill='white', font=font)
            if semantics:
                draw.text((15, 42), ', '.join(semantics), fill='yellow', font=font_s)
            
            annotated.append(np.array(img))
        
        # Save GIF
        gif_path = output_dir / 'region_gifs' / f'region_{rid:02d}.gif'
        imageio.mimsave(gif_path, annotated, fps=5, loop=0)
        print(f"    Region {rid}: {gif_path.name} ({len(annotated)} frames)")
        
        # Keyframes
        for si, seg in enumerate(segments):
            mid = (seg['start_frame_original'] + seg['end_frame_original']) // 2
            if mid < len(rgb_files):
                img = Image.open(rgb_files[mid])
                img.save(output_dir / 'region_keyframes' / f'region_{rid:02d}_seg_{si:02d}.jpg')
    
    # Overview image
    n = len(regions)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 4))
        if n == 1: axes = [axes]
        
        for i, r in enumerate(regions):
            kf = list((output_dir / 'region_keyframes').glob(f'region_{r["region_id"]:02d}_*.jpg'))
            if kf:
                axes[i].imshow(Image.open(kf[0]))
            axes[i].set_title(f"Region {r['region_id']}\n{r['n_frames']}f, {r['n_segments']} seg", fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle(f'{scene_path.name} - {n} Regions (Auto-detected)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'regions_overview.jpg', dpi=150)
        plt.close()
        print(f"    Overview: regions_overview.jpg")


def generate_region_pointclouds(scene_path, regions, output_dir):
    """Generate 3D point clouds for each region with visualizations"""
    import open3d as o3d
    from mpl_toolkits.mplot3d import Axes3D
    
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)
    
    # Load 3D object map
    pcd_file = scene_path / 'pcd_saves' / 'full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz'
    if not pcd_file.exists():
        print("  Warning: 3D object map not found")
        return
    
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    objects = data['objects']
    
    # Create output directory
    pcd_dir = output_dir / 'region_pointclouds'
    pcd_dir.mkdir(exist_ok=True)
    
    print(f"  Building point clouds for {len(regions)} regions...")
    
    # Region colors
    region_colors = [
        [0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8],
        [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8],
    ]
    
    all_region_data = []
    
    for region in regions:
        rid = region['region_id']
        obj_ids = region.get('object_ids', [])
        
        all_points = []
        all_colors = []
        
        for oid in obj_ids:
            if oid < len(objects):
                obj = objects[oid]
                if 'pcd_np' in obj and obj['pcd_np'] is not None and len(obj['pcd_np']) > 0:
                    points = obj['pcd_np']
                    all_points.append(points)
                    
                    if 'pcd_color_np' in obj and obj['pcd_color_np'] is not None:
                        all_colors.append(obj['pcd_color_np'])
                    else:
                        all_colors.append(np.tile(region_colors[rid % len(region_colors)], (len(points), 1)))
        
        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            
            ply_path = pcd_dir / f'region_{rid:02d}.ply'
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"    Region {rid}: {ply_path.name} ({len(merged_points)} points)")
            
            all_region_data.append({
                'region_id': rid,
                'points': merged_points,
                'color': region_colors[rid % len(region_colors)]
            })
    
    # Create merged point cloud with region colors
    if all_region_data:
        merged_all_points = []
        merged_all_colors = []
        
        for rd in all_region_data:
            merged_all_points.append(rd['points'])
            merged_all_colors.append(np.tile(rd['color'], (len(rd['points']), 1)))
        
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(merged_all_points))
        combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(merged_all_colors))
        
        combined_path = pcd_dir / 'all_regions_colored.ply'
        o3d.io.write_point_cloud(str(combined_path), combined_pcd)
        print(f"    Combined: {combined_path.name} ({len(np.vstack(merged_all_points))} points)")
        
        # Generate multi-view PNG visualization
        print("    Generating viewable visualizations...")
        _generate_pointcloud_views(all_region_data, pcd_dir, scene_path.name)
        _generate_pointcloud_html(all_region_data, pcd_dir, scene_path.name)


def _generate_pointcloud_views(all_region_data, pcd_dir, scene_name):
    """Generate multi-view static PNG images"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 10))
    max_points = 5000
    
    views = [('Top View', 90, 0), ('Front View', 0, 0), ('Side View', 0, 90), ('Isometric', 30, 45)]
    
    for vi, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, vi + 1, projection='3d')
        for rd in all_region_data:
            pts = rd['points']
            if len(pts) > max_points:
                idx = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[rd['color']], s=1, alpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    for i, rd in enumerate(all_region_data[:4]):
        ax = fig.add_subplot(2, 4, 5 + i, projection='3d')
        pts = rd['points']
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[rd['color']], s=2, alpha=0.7)
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"Region {rd['region_id']} ({len(rd['points'])} pts)", fontsize=11, fontweight='bold')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rd['color'], 
               markersize=10, label=f"Region {rd['region_id']}") for rd in all_region_data]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    
    plt.suptitle(f"Point Cloud Visualization - {scene_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pcd_dir / 'pointcloud_views.png', dpi=150, bbox_inches='tight')
    plt.close()


def _generate_pointcloud_html(all_region_data, pcd_dir, scene_name):
    """Generate interactive HTML visualization using Plotly"""
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for rd in all_region_data:
            pts = rd['points']
            if len(pts) > 10000:
                idx = np.random.choice(len(pts), 10000, replace=False)
                pts = pts[idx]
            
            color_str = f'rgb({int(rd["color"][0]*255)},{int(rd["color"][1]*255)},{int(rd["color"][2]*255)})'
            
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color_str, opacity=0.7),
                name=f'Region {rd["region_id"]}'
            ))
        
        fig.update_layout(
            title=dict(text=f'Interactive Point Cloud - {scene_name}', font=dict(size=18)),
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.write_html(str(pcd_dir / 'pointcloud_interactive.html'), include_plotlyjs='cdn')
    except ImportError:
        pass  # Plotly not installed


def generate_interpretability_visualizations(scene_path, regions, output_dir, segmenter, stride=5):
    """生成可解释性可视化：分割原因、区域语义、物体时间线"""
    from PIL import Image
    from collections import Counter
    
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)
    
    # 加载数据
    rgb_files = sorted((scene_path / 'results').glob('frame*.jpg'))
    if not rgb_files:
        print("  Warning: No RGB images found, skipping interpretability visualizations")
        return
    
    n_frames = segmenter.n_frames
    frame_objects = segmenter.signals.get('frame_objects', [set() for _ in range(n_frames)])
    objects = segmenter.objects or []
    
    # 加载物体标签 (从 GPT responses)
    object_tags = {}
    gpt_dir = scene_path / 'sg_cache' / 'cfslam_gpt-4_responses'
    if gpt_dir.exists():
        for gpt_file in gpt_dir.glob('*.json'):
            try:
                obj_id = int(gpt_file.stem)
                with open(gpt_file) as f:
                    resp_data = json.load(f)
                resp = json.loads(resp_data.get('response', '{}'))
                object_tags[obj_id] = resp.get('object_tag', f'obj_{obj_id}')
            except:
                pass
    
    # 如果没有 GPT responses，使用 captions
    if not object_tags and segmenter.captions:
        for cap in segmenter.captions:
            caps = cap.get('captions', [])
            if caps:
                object_tags[cap['id']] = caps[0][:30] if caps[0] else f'obj_{cap["id"]}'
    
    # 提取边界点
    boundaries = []
    seen = set()
    for region in regions:
        for seg in region['segments']:
            start = seg['start_frame']
            if start > 0 and start not in seen:
                boundaries.append({'frame': start, 'frame_orig': seg['start_frame_original']})
                seen.add(start)
    boundaries = sorted(boundaries, key=lambda x: x['frame'])
    
    print(f"  Generating interpretability visualizations...")
    
    # === Figure 1: Segmentation Reasons ===
    if boundaries:
        n_b = len(boundaries)
        fig, axes = plt.subplots(n_b, 3, figsize=(16, n_b * 5))
        if n_b == 1: 
            axes = axes.reshape(1, -1)
        
        for idx, b in enumerate(boundaries):
            fr, fo = b['frame'], b['frame_orig']
            bf = max(0, fr - 5)
            af = min(n_frames - 1, fr + 5)
            bo = frame_objects[bf] if bf < len(frame_objects) else set()
            ao = frame_objects[af] if af < len(frame_objects) else set()
            
            # Before image
            if bf * stride < len(rgb_files):
                axes[idx, 0].imshow(Image.open(rgb_files[bf * stride]))
            axes[idx, 0].set_title(f'BEFORE (Frame {bf * stride})', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # After image
            if af * stride < len(rgb_files):
                axes[idx, 1].imshow(Image.open(rgb_files[af * stride]))
            axes[idx, 1].set_title(f'AFTER (Frame {af * stride})', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Changes text
            axes[idx, 2].axis('off')
            axes[idx, 2].text(0.5, 0.95, f'SPLIT: Frame {fo}', transform=axes[idx, 2].transAxes,
                            fontsize=14, fontweight='bold', ha='center', va='top')
            y = 0.85
            axes[idx, 2].text(0.05, y, 'LEAVING:', fontsize=11, fontweight='bold', 
                            color='red', transform=axes[idx, 2].transAxes)
            for oid in list(bo - ao)[:6]:
                y -= 0.06
                axes[idx, 2].text(0.08, y, f'- {object_tags.get(oid, str(oid))}', 
                                fontsize=10, color='darkred', transform=axes[idx, 2].transAxes)
            y -= 0.08
            axes[idx, 2].text(0.05, y, 'ENTERING:', fontsize=11, fontweight='bold',
                            color='green', transform=axes[idx, 2].transAxes)
            for oid in list(ao - bo)[:6]:
                y -= 0.06
                axes[idx, 2].text(0.08, y, f'+ {object_tags.get(oid, str(oid))}',
                                fontsize=10, color='darkgreen', transform=axes[idx, 2].transAxes)
        
        plt.suptitle('SEGMENTATION REASONS', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'segmentation_reasons.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: segmentation_reasons.png")
    
    # === Figure 2: Region Semantics ===
    if regions:
        fig = plt.figure(figsize=(18, 6 * len(regions)))
        cols = plt.cm.Set2(np.linspace(0, 1, len(regions)))
        
        for i, r in enumerate(regions):
            seg = r['segments'][0]
            mid = (seg['start_frame_original'] + seg['end_frame_original']) // 2
            
            # Keyframe
            ax1 = fig.add_subplot(len(regions), 3, i*3 + 1)
            if mid < len(rgb_files):
                ax1.imshow(Image.open(rgb_files[mid]))
            ax1.set_title(f'Region {r["region_id"]} (Frame {mid})', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Object composition pie
            ax2 = fig.add_subplot(len(regions), 3, i*3 + 2)
            tags = Counter([object_tags.get(o, '') for o in r.get('object_ids', []) if o in object_tags])
            if tags:
                tt = tags.most_common(8)
                ax2.pie([t[1] for t in tt], labels=[t[0][:18] for t in tt], autopct='%1.0f%%',
                       colors=plt.cm.Set3(np.linspace(0, 1, len(tt))), textprops={'fontsize': 9})
            ax2.set_title(f'Region {r["region_id"]} - Objects', fontsize=12, fontweight='bold')
            
            # Timeline
            ax3 = fig.add_subplot(len(regions), 3, i*3 + 3)
            ax3.axhline(y=0.5, color='gray', linewidth=2, alpha=0.5)
            ax3.set_xlim(0, len(rgb_files))
            ax3.set_ylim(0, 1)
            for s in r['segments']:
                ax3.axvspan(s['start_frame_original'], s['end_frame_original'], alpha=0.6, color=cols[i])
            ax3.set_title(f'Region {r["region_id"]} - Timeline', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Frame')
            ax3.set_yticks([])
        
        plt.suptitle('REGION SEMANTICS', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / 'region_semantics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: region_semantics.png")
    
    # === Figure 3: Object Visibility Timeline ===
    valid_obj_ids = [oid for oid in range(len(objects)) if any(oid in fo for fo in frame_objects)]
    if valid_obj_ids:
        fig, ax = plt.subplots(figsize=(20, max(12, len(valid_obj_ids) * 0.4)))
        y_labels = []
        plot_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for y_pos, obj_id in enumerate(valid_obj_ids):
            tag = object_tags.get(obj_id, f'obj_{obj_id}')
            y_labels.append(f'ID={obj_id}: {tag[:35]}')
            
            visible_frames = [f * stride for f in range(n_frames) if obj_id in frame_objects[f]]
            if visible_frames:
                segs = []
                start = visible_frames[0]
                prev = start
                for f in visible_frames[1:]:
                    if f - prev > stride * 2:
                        segs.append((start, prev))
                        start = f
                    prev = f
                segs.append((start, prev))
                
                bar_color = plot_colors[obj_id % 20]
                for seg_start, seg_end in segs:
                    ax.barh(y_pos, seg_end - seg_start + stride, left=seg_start, height=0.8,
                            color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.3)
        
        # Draw boundaries
        for i, b in enumerate(boundaries):
            ax.axvline(x=b['frame_orig'], color='red', linestyle='--', linewidth=2, alpha=0.9)
            ax.annotate(f'Split {i+1}\n(Frame {b["frame_orig"]})',
                        xy=(b['frame_orig'], len(valid_obj_ids) + 0.5),
                        fontsize=9, ha='center', va='bottom', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Object (ID = unique identifier)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, len(rgb_files))
        ax.set_ylim(-0.5, len(valid_obj_ids) + 1.5)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.set_title(f'OBJECT VISIBILITY TIMELINE\n'
                     f'Total: {len(objects)} objects | Red lines = split points',
                     fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'object_visibility_timeline_full.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    Saved: object_visibility_timeline_full.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max_regions', type=int, default=10, help='最大区域数(自动检测)')
    args = parser.parse_args()
    
    scene_path = Path(args.dataset_root) / args.scene
    out_dir = scene_path / 'sg_cache' / 'segmentation_regions'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Region-Aware Scene Segmenter - {args.scene}")
    print(f"{'='*60}")
    
    config = RegionSegmenterConfig()
    segmenter = RegionAwareSegmenter(config, stride=args.stride)
    
    print("\n[加载数据]")
    segmenter.load_data(scene_path)
    
    print("\n[场景划分]")
    regions = segmenter.segment()
    
    # 保存结果
    with open(out_dir / 'regions.json', 'w') as f:
        json.dump(regions, f, indent=2, ensure_ascii=False, default=lambda x: int(x) if hasattr(x, 'item') else x)
    
    segmenter.visualize(out_dir / 'segmentation.png')
    
    # 生成区域GIF和关键帧
    print("\n[生成区域可视化]")
    generate_region_visualizations(scene_path, regions, out_dir, stride=args.stride)
    
    # 生成区域点云
    print("\n[生成区域点云]")
    generate_region_pointclouds(scene_path, regions, out_dir)
    
    # 生成可解释性可视化
    print("\n[生成可解释性可视化]")
    generate_interpretability_visualizations(scene_path, regions, out_dir, segmenter, stride=args.stride)
    
    print(f"\n{'='*60}")
    print("结果:")
    for r in regions:
        segs_str = ', '.join([f"{s['start_frame']}-{s['end_frame']}" for s in r['segments']])
        print(f"  区域 {r['region_id']}: {r['n_frames']}帧 ({r['n_segments']}片段: {segs_str})")
        if r['dominant_semantics']:
            print(f"      语义: {', '.join(r['dominant_semantics'][:4])}")
    print(f"{'='*60}")
    print(f"\n保存: {out_dir}")

if __name__ == '__main__':
    main()
