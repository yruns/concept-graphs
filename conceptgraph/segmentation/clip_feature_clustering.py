#!/usr/bin/env python3
"""
CLIP Feature Clustering for Scene Segmentation
===============================================

基于CLIP特征和空间坐标的联合聚类进行场景分割。

核心思路：
1. 加载3D物体地图，每个物体已有融合后的CLIP特征(clip_ft)
2. 将物体的所有3D点分配该物体的CLIP特征
3. 构建联合特征: [alpha * clip_feat, beta * normalized_coord]
4. 使用HDBSCAN进行聚类
5. 基于簇内物体类别进行区域命名

Usage:
    python -m conceptgraph.segmentation.clip_feature_clustering \
        --pcd_file /path/to/pcd.pkl.gz \
        --output_dir /path/to/output/
"""

import os
import json
import gzip
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import warnings

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    warnings.warn("hdbscan not installed, will use sklearn DBSCAN instead")

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Zone name candidates based on object composition
ZONE_TYPE_KEYWORDS = {
    "seating_area": ["sofa", "couch", "armchair", "chair", "ottoman", "cushion"],
    "dining_area": ["dining", "table", "chair", "plate", "bowl", "cup"],
    "kitchen_area": ["stove", "refrigerator", "sink", "microwave", "oven", "cabinet", "counter"],
    "work_area": ["desk", "computer", "monitor", "keyboard", "office", "book"],
    "bedroom_area": ["bed", "nightstand", "dresser", "pillow", "blanket", "wardrobe"],
    "storage_area": ["shelf", "cabinet", "drawer", "box", "storage"],
    "display_area": ["vase", "plant", "art", "picture", "frame", "decoration"],
    "entertainment_area": ["tv", "television", "speaker", "game", "console"],
}


@dataclass
class ClusterZone:
    """表示一个聚类区域"""
    zone_id: str
    zone_name: str
    cluster_label: int
    objects: List[Dict]  # [{id, tag, position, n_points}]
    points: np.ndarray   # (N, 3) 区域内的所有点
    center: List[float]
    bbox: Dict[str, List[float]]
    n_points: int
    avg_clip_feature: Optional[np.ndarray] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "cluster_label": self.cluster_label,
            "n_objects": len(self.objects),
            "n_points": self.n_points,
            "objects": self.objects,
            "center": self.center,
            "bbox": self.bbox,
            "confidence": self.confidence
        }


class ClipFeatureClustering:
    """
    基于CLIP特征的场景聚类分割
    
    Args:
        alpha: CLIP特征权重 (default: 1.0)
        beta: 坐标权重 (default: 0.3)
        min_cluster_size: HDBSCAN最小簇大小 (default: 100)
        min_samples: HDBSCAN核心点阈值 (default: 20)
        cluster_selection_epsilon: HDBSCAN簇选择epsilon (default: 0.0)
        downsample_voxel: 体素降采样大小 (default: 0.05m)
        pca_dim: PCA降维后的维度 (default: 64, 0表示不降维)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        min_cluster_size: int = 100,
        min_samples: int = 20,
        cluster_selection_epsilon: float = 0.0,
        use_hdbscan: bool = True,
        dbscan_eps: float = 0.5,
        downsample_voxel: float = 0.05,
        pca_dim: int = 64,
    ):
        self.alpha = alpha
        self.beta = beta
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.use_hdbscan = use_hdbscan and HAS_HDBSCAN
        self.dbscan_eps = dbscan_eps
        self.downsample_voxel = downsample_voxel
        self.pca_dim = pca_dim
        
        # Data
        self.objects = []
        self.object_clip_features = []
        self.object_positions = []
        self.object_tags = []
        
        # Point-level data
        self.all_points = None
        self.point_features = None
        self.point_object_ids = None
        
        # For mapping back to full points
        self.full_points = None
        self.full_point_object_ids = None
        
    def load_data(self, pcd_file: str) -> int:
        """
        加载3D物体地图
        
        Args:
            pcd_file: pcd_saves/*.pkl.gz 文件路径
            
        Returns:
            加载的物体数量
        """
        print(f"Loading data from: {pcd_file}")
        
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        self.objects = data.get('objects', [])
        print(f"  Loaded {len(self.objects)} objects")
        
        # 提取每个物体的信息
        self.object_clip_features = []
        self.object_positions = []
        self.object_tags = []
        
        n_with_clip = 0
        n_with_points = 0
        
        for i, obj in enumerate(self.objects):
            # CLIP特征
            clip_ft = obj.get('clip_ft')
            if clip_ft is not None:
                if hasattr(clip_ft, 'cpu'):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.array(clip_ft).flatten()
                n_with_clip += 1
            else:
                # 使用零向量作为占位
                clip_ft = np.zeros(1024)
            self.object_clip_features.append(clip_ft)
            
            # 位置（点云中心）
            pcd_np = obj.get('pcd_np')
            if pcd_np is not None and len(pcd_np) > 0:
                position = pcd_np.mean(axis=0)
                n_with_points += 1
            else:
                position = np.zeros(3)
            self.object_positions.append(position)
            
            # 标签
            tag = self._get_object_tag(obj, i)
            self.object_tags.append(tag)
        
        print(f"  Objects with CLIP features: {n_with_clip}")
        print(f"  Objects with point clouds: {n_with_points}")
        
        return len(self.objects)
    
    def _get_object_tag(self, obj: Dict, obj_id: int) -> str:
        """获取物体标签"""
        class_names = obj.get('class_name', [])
        if class_names:
            # 获取最常见的非'item'类别
            valid_names = [n for n in class_names if n and n.lower() != 'item']
            if valid_names:
                return Counter(valid_names).most_common(1)[0][0]
            elif class_names[0]:
                return class_names[0]
        return f"object_{obj_id}"
    
    def build_point_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建点级特征：每个点获得所属物体的CLIP特征
        
        Returns:
            all_points: (N, 3) 所有点的坐标
            point_features: (N, D) 所有点的CLIP特征
        """
        print("Building point-level features...")
        
        all_points = []
        all_features = []
        point_object_ids = []
        
        for obj_id, obj in enumerate(self.objects):
            pcd_np = obj.get('pcd_np')
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            clip_ft = self.object_clip_features[obj_id]
            
            # 为该物体的所有点分配相同的CLIP特征
            n_points = len(pcd_np)
            all_points.append(pcd_np)
            all_features.append(np.tile(clip_ft, (n_points, 1)))
            point_object_ids.extend([obj_id] * n_points)
        
        if not all_points:
            raise ValueError("No points found in any object")
        
        self.all_points = np.vstack(all_points)
        self.point_features = np.vstack(all_features)
        self.point_object_ids = np.array(point_object_ids)
        
        print(f"  Total points: {len(self.all_points)}")
        print(f"  Feature dimension: {self.point_features.shape[1]}")
        
        return self.all_points, self.point_features
    
    def build_joint_features(self) -> np.ndarray:
        """
        构建联合特征向量: [alpha * normalized_clip, beta * normalized_coord]
        支持PCA降维以减少内存使用
        """
        print("Building joint features...")
        
        # CLIP特征L2归一化
        clip_norms = np.linalg.norm(self.point_features, axis=1, keepdims=True)
        clip_norms = np.maximum(clip_norms, 1e-8)
        clip_normalized = self.point_features / clip_norms
        
        # PCA降维 (可选)
        if self.pca_dim > 0 and self.pca_dim < clip_normalized.shape[1]:
            from sklearn.decomposition import PCA
            print(f"  Applying PCA: {clip_normalized.shape[1]} -> {self.pca_dim}")
            pca = PCA(n_components=self.pca_dim)
            clip_reduced = pca.fit_transform(clip_normalized)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"  PCA explained variance: {explained_var*100:.1f}%")
        else:
            clip_reduced = clip_normalized
        
        # 坐标z-score标准化
        coord_scaler = StandardScaler()
        coord_normalized = coord_scaler.fit_transform(self.all_points)
        
        # 联合特征
        joint_features = np.concatenate([
            self.alpha * clip_reduced,
            self.beta * coord_normalized
        ], axis=1)
        
        print(f"  Joint feature dimension: {joint_features.shape[1]}")
        print(f"  Alpha (CLIP weight): {self.alpha}")
        print(f"  Beta (Coord weight): {self.beta}")
        
        return joint_features
    
    def cluster(self, joint_features: np.ndarray) -> np.ndarray:
        """
        执行聚类
        
        Args:
            joint_features: (N, D) 联合特征
            
        Returns:
            labels: (N,) 聚类标签，-1表示噪声
        """
        print("Clustering...")
        
        if self.use_hdbscan:
            print(f"  Using HDBSCAN (min_cluster_size={self.min_cluster_size}, "
                  f"min_samples={self.min_samples})")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean',
                cluster_selection_method='eom',
                core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(joint_features)
        else:
            print(f"  Using DBSCAN (eps={self.dbscan_eps}, min_samples={self.min_samples})")
            
            clusterer = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.min_samples,
                metric='euclidean',
                n_jobs=-1,
            )
            labels = clusterer.fit_predict(joint_features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        return labels
    
    def build_zones(self, labels: np.ndarray) -> List[ClusterZone]:
        """
        从聚类结果构建区域
        
        Args:
            labels: (N,) 聚类标签
            
        Returns:
            zones: ClusterZone列表
        """
        print("Building zones from clusters...")
        
        zones = []
        unique_labels = set(labels)
        
        # 移除噪声标签
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_points = self.all_points[mask]
            cluster_object_ids = self.point_object_ids[mask]
            cluster_features = self.point_features[mask]
            
            # 获取簇内的物体
            unique_obj_ids = np.unique(cluster_object_ids)
            cluster_objects = []
            for obj_id in unique_obj_ids:
                obj = self.objects[obj_id]
                obj_points_in_cluster = (cluster_object_ids == obj_id).sum()
                cluster_objects.append({
                    "id": int(obj_id),
                    "tag": self.object_tags[obj_id],
                    "position": self.object_positions[obj_id].tolist(),
                    "n_points_in_zone": int(obj_points_in_cluster),
                })
            
            # 计算区域属性
            center = cluster_points.mean(axis=0).tolist()
            bbox = {
                "min": cluster_points.min(axis=0).tolist(),
                "max": cluster_points.max(axis=0).tolist()
            }
            
            # 计算平均CLIP特征
            avg_clip = cluster_features.mean(axis=0)
            avg_clip = avg_clip / (np.linalg.norm(avg_clip) + 1e-8)
            
            # 命名区域
            zone_name = self._name_zone(cluster_objects)
            
            zone = ClusterZone(
                zone_id=f"zone_{cluster_id}",
                zone_name=zone_name,
                cluster_label=int(cluster_id),
                objects=cluster_objects,
                points=cluster_points,
                center=center,
                bbox=bbox,
                n_points=len(cluster_points),
                avg_clip_feature=avg_clip,
            )
            zones.append(zone)
        
        # 处理噪声点
        noise_mask = labels == -1
        if noise_mask.any():
            zones = self._handle_noise_points(zones, labels)
        
        # 按点数排序
        zones.sort(key=lambda z: z.n_points, reverse=True)
        
        # 重新编号zone_id
        for i, zone in enumerate(zones):
            zone.zone_id = f"zone_{i}"
        
        print(f"  Created {len(zones)} zones")
        
        return zones
    
    def _name_zone(self, objects: List[Dict]) -> str:
        """基于物体类别命名区域"""
        if not objects:
            return "unknown_area"
        
        # 统计物体标签
        tags = [obj["tag"].lower() for obj in objects]
        tag_counts = Counter(tags)
        
        # 匹配区域类型
        zone_scores = {}
        for zone_type, keywords in ZONE_TYPE_KEYWORDS.items():
            score = 0
            for tag, count in tag_counts.items():
                for keyword in keywords:
                    if keyword in tag:
                        score += count
                        break
            if score > 0:
                zone_scores[zone_type] = score
        
        if zone_scores:
            best_zone = max(zone_scores, key=zone_scores.get)
            return best_zone
        
        # 使用最常见的物体标签
        most_common = tag_counts.most_common(1)[0][0]
        return f"{most_common}_area"
    
    def _handle_noise_points(self, zones: List[ClusterZone], labels: np.ndarray) -> List[ClusterZone]:
        """将噪声点分配到最近的区域"""
        if not zones:
            return zones
        
        noise_mask = labels == -1
        noise_points = self.all_points[noise_mask]
        noise_object_ids = self.point_object_ids[noise_mask]
        
        # 计算每个噪声点到各区域中心的距离
        zone_centers = np.array([z.center for z in zones])
        
        for i, (point, obj_id) in enumerate(zip(noise_points, noise_object_ids)):
            distances = np.linalg.norm(zone_centers - point, axis=1)
            nearest_zone_idx = distances.argmin()
            
            # 添加到最近的区域
            zones[nearest_zone_idx].n_points += 1
        
        # 重新计算边界框
        for zone in zones:
            zone_mask = labels == zone.cluster_label
            # 包含分配的噪声点（简化处理：只更新点数）
        
        return zones
    
    def run(self, pcd_file: str, object_level: bool = True) -> List[ClusterZone]:
        """
        运行完整的聚类流程
        
        Args:
            pcd_file: 输入的pcd文件路径
            object_level: 是否使用物体级别聚类（推荐，内存友好）
            
        Returns:
            zones: 区域列表
        """
        # 1. 加载数据
        self.load_data(pcd_file)
        
        if object_level:
            # 物体级别聚类（内存友好）
            return self._run_object_level_clustering()
        else:
            # 点级别聚类（需要大内存）
            return self._run_point_level_clustering()
    
    def _run_object_level_clustering(self) -> List[ClusterZone]:
        """物体级别聚类（内存友好）"""
        print("\nUsing object-level clustering (memory efficient)...")
        
        # 准备物体特征
        valid_objects = []
        valid_features = []
        valid_positions = []
        
        for i, obj in enumerate(self.objects):
            clip_ft = self.object_clip_features[i]
            pos = self.object_positions[i]
            
            # 检查有效性
            if np.all(clip_ft == 0) or np.all(pos == 0):
                continue
            if obj.get('pcd_np') is None or len(obj.get('pcd_np', [])) == 0:
                continue
            
            valid_objects.append(i)
            valid_features.append(clip_ft)
            valid_positions.append(pos)
        
        if len(valid_objects) < 2:
            print("  Warning: Not enough valid objects for clustering")
            return []
        
        print(f"  Valid objects: {len(valid_objects)}")
        
        # 构建物体级联合特征
        features = np.array(valid_features)
        positions = np.array(valid_positions)
        
        # L2归一化CLIP特征
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norms = np.maximum(feat_norms, 1e-8)
        features_norm = features / feat_norms
        
        # PCA降维
        if self.pca_dim > 0 and self.pca_dim < features_norm.shape[1]:
            from sklearn.decomposition import PCA
            print(f"  Applying PCA: {features_norm.shape[1]} -> {self.pca_dim}")
            pca = PCA(n_components=self.pca_dim)
            features_reduced = pca.fit_transform(features_norm)
            print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        else:
            features_reduced = features_norm
        
        # 坐标标准化
        coord_scaler = StandardScaler()
        positions_norm = coord_scaler.fit_transform(positions)
        
        # 联合特征
        joint = np.concatenate([
            self.alpha * features_reduced,
            self.beta * positions_norm
        ], axis=1)
        
        print(f"  Joint feature dimension: {joint.shape[1]}")
        
        # DBSCAN聚类
        print(f"  Clustering with DBSCAN (eps={self.dbscan_eps})...")
        clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=max(2, self.min_samples // 10))
        labels = clusterer.fit_predict(joint)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"  Found {n_clusters} clusters, {n_noise} noise objects")
        
        # 构建区域
        zones = self._build_zones_from_object_labels(valid_objects, labels)
        
        return zones
    
    def _build_zones_from_object_labels(self, valid_objects: List[int], labels: np.ndarray) -> List[ClusterZone]:
        """从物体级标签构建区域"""
        zones = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_obj_ids = [valid_objects[i] for i in range(len(valid_objects)) if mask[i]]
            
            # 收集区域内的物体和点
            cluster_objects = []
            all_points = []
            all_features = []
            
            for obj_id in cluster_obj_ids:
                obj = self.objects[obj_id]
                pcd_np = obj.get('pcd_np')
                if pcd_np is not None and len(pcd_np) > 0:
                    all_points.append(pcd_np)
                    all_features.append(self.object_clip_features[obj_id])
                
                cluster_objects.append({
                    "id": int(obj_id),
                    "tag": self.object_tags[obj_id],
                    "position": self.object_positions[obj_id].tolist(),
                    "n_points": len(pcd_np) if pcd_np is not None else 0,
                })
            
            if not all_points:
                continue
            
            points = np.vstack(all_points)
            center = points.mean(axis=0).tolist()
            bbox = {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}
            
            # 计算平均CLIP特征
            avg_clip = np.mean(all_features, axis=0)
            avg_clip = avg_clip / (np.linalg.norm(avg_clip) + 1e-8)
            
            zone_name = self._name_zone(cluster_objects)
            
            zone = ClusterZone(
                zone_id=f"zone_{cluster_id}",
                zone_name=zone_name,
                cluster_label=int(cluster_id),
                objects=cluster_objects,
                points=points,
                center=center,
                bbox=bbox,
                n_points=len(points),
                avg_clip_feature=avg_clip,
            )
            zones.append(zone)
        
        # 处理噪声物体 - 分配到最近的区域
        noise_mask = labels == -1
        if noise_mask.any() and zones:
            noise_obj_ids = [valid_objects[i] for i in range(len(valid_objects)) if noise_mask[i]]
            zone_centers = np.array([z.center for z in zones])
            
            for obj_id in noise_obj_ids:
                pos = np.array(self.object_positions[obj_id])
                distances = np.linalg.norm(zone_centers - pos, axis=1)
                nearest_idx = distances.argmin()
                
                obj = self.objects[obj_id]
                pcd_np = obj.get('pcd_np')
                zones[nearest_idx].objects.append({
                    "id": int(obj_id),
                    "tag": self.object_tags[obj_id],
                    "position": self.object_positions[obj_id].tolist(),
                    "n_points": len(pcd_np) if pcd_np is not None else 0,
                })
                if pcd_np is not None:
                    zones[nearest_idx].n_points += len(pcd_np)
        
        # 排序并重新编号
        zones.sort(key=lambda z: z.n_points, reverse=True)
        for i, zone in enumerate(zones):
            zone.zone_id = f"zone_{i}"
        
        return zones
    
    def _run_point_level_clustering(self) -> List[ClusterZone]:
        """点级别聚类（需要大内存）"""
        print("\nUsing point-level clustering (requires more memory)...")
        
        # 2. 构建点级特征
        self.build_point_features()
        
        # 3. 构建联合特征
        joint_features = self.build_joint_features()
        
        # 4. 聚类
        labels = self.cluster(joint_features)
        
        # 5. 构建区域
        zones = self.build_zones(labels)
        
        return zones
    
    def save_results(self, zones: List[ClusterZone], output_dir: str, labels: np.ndarray = None):
        """
        保存结果
        
        Args:
            zones: 区域列表
            output_dir: 输出目录
            labels: 聚类标签（用于生成点云）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存JSON结果
        result = {
            "method": "clip_feature_clustering",
            "params": {
                "alpha": self.alpha,
                "beta": self.beta,
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "use_hdbscan": self.use_hdbscan,
            },
            "n_zones": len(zones),
            "n_total_points": len(self.all_points) if self.all_points is not None else 0,
            "zones": [z.to_dict() for z in zones]
        }
        
        json_path = output_dir / "clip_zones.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_path}")
        
        # 2. 保存统计信息
        stats = {
            "n_zones": len(zones),
            "n_objects": len(self.objects),
            "n_points": len(self.all_points) if self.all_points is not None else 0,
            "zone_sizes": [z.n_points for z in zones],
            "zone_names": [z.zone_name for z in zones],
        }
        stats_path = output_dir / "clustering_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats: {stats_path}")
    
    def print_summary(self, zones: List[ClusterZone]):
        """打印聚类摘要"""
        print("\n" + "=" * 60)
        print("CLIP FEATURE CLUSTERING SUMMARY")
        print("=" * 60)
        
        total_points = sum(z.n_points for z in zones)
        total_objects = sum(len(z.objects) for z in zones)
        
        print(f"Total zones: {len(zones)}")
        print(f"Total points: {total_points}")
        print(f"Total objects: {total_objects}")
        
        print("\nZones:")
        for zone in zones:
            print(f"\n  [{zone.zone_id}] {zone.zone_name}")
            print(f"    Points: {zone.n_points}")
            print(f"    Objects ({len(zone.objects)}): ", end="")
            tags = [o["tag"] for o in zone.objects[:5]]
            if len(zone.objects) > 5:
                tags.append(f"+{len(zone.objects)-5} more")
            print(", ".join(tags))
            print(f"    Center: ({zone.center[0]:.2f}, {zone.center[1]:.2f}, {zone.center[2]:.2f})")
        
        print("\n" + "=" * 60)


def generate_colored_ply(
    zones: List[ClusterZone],
    all_points: np.ndarray,
    point_object_ids: np.ndarray,
    objects: List[Dict],
    output_path: str
):
    """生成着色PLY点云"""
    import struct
    import sys
    
    try:
        import distinctipy
        colors = distinctipy.get_colors(max(len(zones), 1), pastel_factor=0.5)
    except ImportError:
        # 使用简单的颜色生成
        np.random.seed(42)
        colors = [tuple(np.random.rand(3)) for _ in range(max(len(zones), 1))]
    
    # 构建object_id到zone_id的映射
    obj_to_zone = {}
    for zone_idx, zone in enumerate(zones):
        for obj in zone.objects:
            obj_to_zone[obj["id"]] = zone_idx
    
    # 为每个点分配颜色
    point_colors = np.zeros((len(all_points), 3), dtype=np.uint8)
    default_color = np.array([128, 128, 128], dtype=np.uint8)
    
    for i, obj_id in enumerate(point_object_ids):
        zone_idx = obj_to_zone.get(obj_id)
        if zone_idx is not None and zone_idx < len(colors):
            point_colors[i] = np.array([int(c * 255) for c in colors[zone_idx]], dtype=np.uint8)
        else:
            point_colors[i] = default_color
    
    # 写入PLY
    ply_format = 'binary_little_endian' if sys.byteorder == 'little' else 'binary_big_endian'
    
    with open(output_path, 'wb') as f:
        header = f"""ply
format {ply_format} 1.0
element vertex {len(all_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        for i in range(len(all_points)):
            f.write(struct.pack('fffBBB',
                float(all_points[i, 0]),
                float(all_points[i, 1]),
                float(all_points[i, 2]),
                int(point_colors[i, 0]),
                int(point_colors[i, 1]),
                int(point_colors[i, 2])
            ))
    
    print(f"Saved PLY: {output_path} ({len(all_points)} points)")
    return colors


def generate_legend(zones: List[ClusterZone], colors: List, output_path: str):
    """生成颜色图例"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_zones = len(zones)
    fig, ax = plt.subplots(figsize=(10, max(3, n_zones * 0.8)))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_zones + 1)
    ax.axis('off')
    ax.set_title("Zone Color Legend (CLIP Feature Clustering)", fontsize=14, fontweight='bold')
    
    for i, zone in enumerate(zones):
        y = n_zones - i
        color = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
        
        # 绘制颜色方块
        ax.add_patch(plt.Rectangle((0.02, y - 0.4), 0.06, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=1))
        
        # 标签文本
        label = f"{zone.zone_name} ({zone.n_points:,} pts, {len(zone.objects)} objs)"
        ax.text(0.10, y, label, fontsize=11, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved legend: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Feature Clustering for Scene Segmentation"
    )
    parser.add_argument("--pcd_file", type=str, required=True,
                        help="Path to pcd pickle file (pcd_saves/*.pkl.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CLIP feature weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Coordinate weight (default: 0.3)")
    parser.add_argument("--min_cluster_size", type=int, default=100,
                        help="HDBSCAN min cluster size (default: 100)")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="HDBSCAN min samples (default: 20)")
    parser.add_argument("--use_dbscan", action="store_true",
                        help="Use DBSCAN instead of HDBSCAN")
    parser.add_argument("--dbscan_eps", type=float, default=0.5,
                        help="DBSCAN epsilon (default: 0.5)")
    parser.add_argument("--downsample_voxel", type=float, default=0.05,
                        help="Voxel size for downsampling (default: 0.05m, 0=no downsample)")
    parser.add_argument("--pca_dim", type=int, default=64,
                        help="PCA dimension for CLIP features (default: 64, 0=no PCA)")
    
    args = parser.parse_args()
    
    # 创建聚类器
    clusterer = ClipFeatureClustering(
        alpha=args.alpha,
        beta=args.beta,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        use_hdbscan=not args.use_dbscan,
        dbscan_eps=args.dbscan_eps,
        downsample_voxel=args.downsample_voxel,
        pca_dim=args.pca_dim,
    )
    
    # 运行聚类
    zones = clusterer.run(args.pcd_file)
    
    # 打印摘要
    clusterer.print_summary(zones)
    
    # 保存结果
    clusterer.save_results(zones, args.output_dir)
    
    # 生成可视化
    output_dir = Path(args.output_dir)
    
    # PLY点云 - 从zones收集所有点
    ply_path = output_dir / "zones_colored.ply"
    all_pts, all_obj_ids = [], []
    for obj_id, obj in enumerate(clusterer.objects):
        pcd = obj.get('pcd_np')
        if pcd is not None and len(pcd) > 0:
            all_pts.append(pcd)
            all_obj_ids.extend([obj_id] * len(pcd))
    
    if all_pts:
        all_pts = np.vstack(all_pts)
        all_obj_ids = np.array(all_obj_ids)
        colors = generate_colored_ply(zones, all_pts, all_obj_ids, clusterer.objects, str(ply_path))
    else:
        colors = []
        print("Warning: No points to generate PLY")
    
    # 图例
    legend_path = output_dir / "zones_legend.png"
    generate_legend(zones, colors, str(legend_path))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
