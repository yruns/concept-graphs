"""
SceneVectorStore: 隐式场景表示
==============================

核心设计:
- 多粒度向量索引 (物体级、区域级、帧级)
- 元数据存储 (位置、属性、边界框)
- 语义+空间联合检索
"""

import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from scipy.spatial import KDTree


@dataclass
class ObjectMeta:
    """物体元数据"""
    id: int
    tag: str
    position: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    n_points: int
    clip_feature: np.ndarray  # (D,)
    attributes: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "tag": self.tag,
            "position": self.position.tolist(),
            "bbox": {"min": self.bbox_min.tolist(), "max": self.bbox_max.tolist()},
            "n_points": self.n_points,
            "attributes": self.attributes
        }


@dataclass
class FrameMeta:
    """帧元数据"""
    idx: int
    image_path: str
    pose: np.ndarray  # (7,) or (4,4)
    clip_feature: Optional[np.ndarray] = None
    visible_objects: List[int] = field(default_factory=list)


@dataclass
class SearchResult:
    """检索结果"""
    id: int
    score: float
    meta: Any  # ObjectMeta or FrameMeta
    
    def __repr__(self):
        if isinstance(self.meta, ObjectMeta):
            return f"SearchResult(id={self.id}, tag='{self.meta.tag}', score={self.score:.3f})"
        return f"SearchResult(id={self.id}, score={self.score:.3f})"


class SceneVectorStore:
    """
    隐式场景表示存储
    
    支持:
    - 语义检索: 用文本/特征查找相似物体
    - 空间检索: 按位置范围查找物体
    - 联合检索: 语义+空间组合查询
    """
    
    def __init__(self, feature_dim: int = 1024):
        self.feature_dim = feature_dim
        
        # 物体数据
        self.objects: List[ObjectMeta] = []
        self.object_features: Optional[np.ndarray] = None  # (N, D)
        self.object_index = None  # FAISS index
        
        # 帧数据
        self.frames: List[FrameMeta] = []
        self.frame_features: Optional[np.ndarray] = None  # (M, D)
        self.frame_index = None  # FAISS index
        
        # 空间索引
        self.spatial_tree: Optional[KDTree] = None
        
        # 场景边界
        self.scene_bounds = {"min": None, "max": None}
        
        # CLIP模型 (延迟加载)
        self._clip_model = None
        self._clip_preprocess = None
        
    def load_from_pcd(self, pcd_file: str, gsa_dir: Optional[str] = None):
        """
        从ConceptGraphs的pcd文件加载场景数据
        
        Args:
            pcd_file: pcd_saves/*.pkl.gz 文件
            gsa_dir: gsa_detections目录 (用于加载帧级特征)
        """
        print(f"Loading scene from: {pcd_file}")
        
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        raw_objects = data.get('objects', [])
        print(f"  Found {len(raw_objects)} objects")
        
        # 解析物体
        self.objects = []
        features = []
        positions = []
        
        for i, obj in enumerate(raw_objects):
            # 获取CLIP特征
            clip_ft = obj.get('clip_ft')
            if clip_ft is not None:
                if hasattr(clip_ft, 'cpu'):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.array(clip_ft).flatten()
            else:
                clip_ft = np.zeros(self.feature_dim)
            
            # 获取点云
            pcd_np = obj.get('pcd_np')
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            position = pcd_np.mean(axis=0)
            bbox_min = pcd_np.min(axis=0)
            bbox_max = pcd_np.max(axis=0)
            
            # 获取标签
            tag = self._get_object_tag(obj, i)
            
            meta = ObjectMeta(
                id=len(self.objects),
                tag=tag,
                position=position,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                n_points=len(pcd_np),
                clip_feature=clip_ft,
            )
            
            self.objects.append(meta)
            features.append(clip_ft)
            positions.append(position)
        
        if features:
            self.object_features = np.array(features, dtype=np.float32)
            positions = np.array(positions)
            
            # 构建向量索引
            self._build_object_index()
            
            # 构建空间索引
            self.spatial_tree = KDTree(positions)
            
            # 计算场景边界
            self.scene_bounds = {
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist()
            }
        
        print(f"  Loaded {len(self.objects)} objects with features")
        print(f"  Scene bounds: {self.scene_bounds}")
        
        # 加载帧级特征 (如果有)
        if gsa_dir:
            self._load_frame_features(gsa_dir)
    
    def _get_object_tag(self, obj: Dict, obj_id: int) -> str:
        """获取物体标签"""
        class_names = obj.get('class_name', [])
        if class_names:
            valid_names = [n for n in class_names if n and n.lower() != 'item']
            if valid_names:
                return Counter(valid_names).most_common(1)[0][0]
            elif class_names[0]:
                return class_names[0]
        return f"object_{obj_id}"
    
    def _build_object_index(self):
        """构建物体向量索引"""
        if self.object_features is None or len(self.object_features) == 0:
            return
        
        # L2归一化
        norms = np.linalg.norm(self.object_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.object_features / norms
        
        if HAS_FAISS:
            # 使用FAISS
            self.object_index = faiss.IndexFlatIP(self.feature_dim)  # 内积 (归一化后等价于cosine)
            self.object_index.add(normalized.astype(np.float32))
            print(f"  Built FAISS index with {self.object_index.ntotal} vectors")
        else:
            # 使用numpy (简单版本)
            self.object_index = normalized
            print(f"  Built numpy index with {len(normalized)} vectors")
    
    def _load_frame_features(self, gsa_dir: str):
        """加载帧级CLIP特征"""
        gsa_path = Path(gsa_dir)
        if not gsa_path.exists():
            print(f"  GSA directory not found: {gsa_dir}")
            return
        
        pkl_files = sorted(gsa_path.glob("*.pkl.gz"))
        print(f"  Loading frame features from {len(pkl_files)} files...")
        
        self.frames = []
        frame_features = []
        
        for idx, pkl_file in enumerate(pkl_files):
            try:
                with gzip.open(pkl_file, 'rb') as f:
                    gsa_data = pickle.load(f)
                
                frame_clip = gsa_data.get('frame_clip_feat')
                if frame_clip is not None:
                    frame_clip = np.array(frame_clip).flatten()
                    
                    meta = FrameMeta(
                        idx=idx,
                        image_path=str(pkl_file).replace('.pkl.gz', '.jpg'),
                        pose=np.eye(4),  # TODO: 从traj.txt加载
                        clip_feature=frame_clip,
                    )
                    self.frames.append(meta)
                    frame_features.append(frame_clip)
            except Exception as e:
                continue
        
        if frame_features:
            self.frame_features = np.array(frame_features, dtype=np.float32)
            self._build_frame_index()
            print(f"  Loaded {len(self.frames)} frames with features")
    
    def _build_frame_index(self):
        """构建帧向量索引"""
        if self.frame_features is None or len(self.frame_features) == 0:
            return
        
        norms = np.linalg.norm(self.frame_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.frame_features / norms
        
        if HAS_FAISS:
            self.frame_index = faiss.IndexFlatIP(self.feature_dim)
            self.frame_index.add(normalized.astype(np.float32))
        else:
            self.frame_index = normalized
    
    def _get_clip_model(self):
        """延迟加载CLIP模型"""
        if self._clip_model is None:
            import open_clip
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            self._clip_model.eval()
            print("  Loaded CLIP model")
        return self._clip_model
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为CLIP特征"""
        import torch
        import open_clip
        
        model = self._get_clip_model()
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        with torch.no_grad():
            tokens = tokenizer([text])
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy().flatten()
        
        return feat.astype(np.float32)
    
    def semantic_search(self, query: str, top_k: int = 10, 
                       search_type: str = "objects") -> List[SearchResult]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            search_type: "objects" 或 "frames"
        """
        query_feat = self.encode_text(query)
        
        if search_type == "objects":
            return self._search_objects(query_feat, top_k)
        elif search_type == "frames":
            return self._search_frames(query_feat, top_k)
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
    def _search_objects(self, query_feat: np.ndarray, top_k: int) -> List[SearchResult]:
        """搜索物体"""
        if self.object_index is None:
            return []
        
        query_feat = query_feat.reshape(1, -1).astype(np.float32)
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        
        if HAS_FAISS:
            scores, indices = self.object_index.search(query_feat, min(top_k, len(self.objects)))
            scores = scores[0]
            indices = indices[0]
        else:
            # numpy版本
            scores = (self.object_index @ query_feat.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.objects):
                results.append(SearchResult(
                    id=int(idx),
                    score=float(score),
                    meta=self.objects[idx]
                ))
        
        return results
    
    def _search_frames(self, query_feat: np.ndarray, top_k: int) -> List[SearchResult]:
        """搜索帧"""
        if self.frame_index is None:
            return []
        
        query_feat = query_feat.reshape(1, -1).astype(np.float32)
        query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        
        if HAS_FAISS:
            scores, indices = self.frame_index.search(query_feat, min(top_k, len(self.frames)))
            scores = scores[0]
            indices = indices[0]
        else:
            scores = (self.frame_index @ query_feat.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.frames):
                results.append(SearchResult(
                    id=int(idx),
                    score=float(score),
                    meta=self.frames[idx]
                ))
        
        return results
    
    def spatial_search(self, center: np.ndarray, radius: float) -> List[ObjectMeta]:
        """
        空间范围检索
        
        Args:
            center: 中心点 (3,)
            radius: 搜索半径
        """
        if self.spatial_tree is None:
            return []
        
        indices = self.spatial_tree.query_ball_point(center, radius)
        return [self.objects[i] for i in indices]
    
    def joint_search(self, query: str, spatial_center: Optional[np.ndarray] = None,
                    spatial_radius: float = 2.0, top_k: int = 10) -> List[SearchResult]:
        """
        语义+空间联合检索
        
        Args:
            query: 查询文本
            spatial_center: 空间中心点 (可选)
            spatial_radius: 空间范围
            top_k: 返回数量
        """
        # 先语义检索
        semantic_results = self.semantic_search(query, top_k=top_k * 3)
        
        if spatial_center is None:
            return semantic_results[:top_k]
        
        # 空间过滤
        filtered = []
        for r in semantic_results:
            dist = np.linalg.norm(r.meta.position - spatial_center)
            if dist <= spatial_radius:
                filtered.append(r)
        
        return filtered[:top_k]
    
    def get_object_by_id(self, obj_id: int) -> Optional[ObjectMeta]:
        """根据ID获取物体"""
        if 0 <= obj_id < len(self.objects):
            return self.objects[obj_id]
        return None
    
    def get_neighbors(self, obj_id: int, radius: float = 1.5) -> List[ObjectMeta]:
        """获取物体的邻居"""
        obj = self.get_object_by_id(obj_id)
        if obj is None:
            return []
        
        neighbors = self.spatial_search(obj.position, radius)
        return [n for n in neighbors if n.id != obj_id]
    
    def summary(self) -> Dict:
        """返回场景摘要"""
        tag_counts = Counter(obj.tag for obj in self.objects)
        
        return {
            "n_objects": len(self.objects),
            "n_frames": len(self.frames),
            "scene_bounds": self.scene_bounds,
            "object_tags": dict(tag_counts.most_common(20)),
            "has_faiss": HAS_FAISS,
        }
    
    def save(self, output_path: str):
        """保存场景表示"""
        data = {
            "objects": [obj.to_dict() for obj in self.objects],
            "object_features": self.object_features,
            "scene_bounds": self.scene_bounds,
            "summary": self.summary(),
        }
        
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved scene store to: {output_path}")
    
    def __repr__(self):
        return f"SceneVectorStore(objects={len(self.objects)}, frames={len(self.frames)})"
