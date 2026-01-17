# 区域感知场景划分算法 (Region-Aware Scene Segmentation)

## 概述

本算法在 **Step 4 (物体描述)** 之后执行，充分利用前4步积累的所有信息，将视频轨迹划分为多个**逻辑场景区域**。

### 核心特点

1. **充分利用先验信息**: 综合使用轨迹、3D物体、可见性、CLIP语义特征
2. **支持不连续片段**: 识别相机多次访问同一区域的情况
3. **自动确定区域数量**: 无需手动指定，由算法自动检测最佳K值
4. **丰富的可视化输出**: 信号分析图、区域GIF、关键帧、总览图

---

## Pipeline 位置

```
Step 1:   2D分割 (SAM + CLIP)
          ★ 生成帧级 CLIP 特征 (frame_clip_feat)
              ↓
Step 2:   3D物体地图
              ↓
Step 4:   物体描述 (Vision LLM)
              ↓
Step 4.5: ★ 区域感知场景划分 ← 本算法
              ↓
Step 5:   细化描述
              ↓
Step 6:   场景图
```

---

## 算法架构

### 两阶段方法

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: 时序分割 (Temporal Segmentation)                   │
│  ─────────────────────────────────────────                   │
│  输入: 多模态信号 (轨迹、物体密度、可见性、语义)              │
│  输出: N 个连续时间片段                                       │
│  方法: 加权信号融合 → 高斯平滑 → 梯度检测 → 峰值提取          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: 片段聚类 (Segment Clustering)                      │
│  ─────────────────────────────────────                       │
│  输入: N 个时序片段 + 帧间相似度矩阵                          │
│  输出: K 个逻辑场景区域 (每个区域可含多个不连续片段)          │
│  方法: 层次聚类 + 自动K检测                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 数据加载

### 输入数据来源

| 数据 | 文件路径 | 说明 |
|------|----------|------|
| 相机位姿 | `traj.txt` | 每帧 7 个值: x, y, z, qx, qy, qz, qw |
| 3D物体地图 | `pcd_saves/*.pkl.gz` | 包含物体点云、CLIP特征、可见帧列表 |
| 物体描述 | `sg_cache/cfslam_llava_captions.json` | Vision LLM 生成的物体描述 |
| 帧级CLIP特征 | `gsa_detections_none/*.pkl.gz` | 每帧的整帧 CLIP 特征 (1024维) |

### 帧级 CLIP 特征

在 Step 1 中，对每一帧图像计算整帧的 CLIP 特征：

```python
# generate_gsa_results.py 中的计算
with torch.no_grad():
    frame_clip_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
    frame_clip_feat = clip_model.encode_image(frame_clip_input)
    frame_clip_feat = F.normalize(frame_clip_feat, dim=-1)  # L2 归一化
    frame_clip_feat = frame_clip_feat.cpu().numpy().squeeze()  # shape: (1024,)
```

特征保存在每帧的检测结果文件中：
```python
results = {
    "xyxy": ...,
    "mask": ...,
    "image_feats": ...,
    "frame_clip_feat": frame_clip_feat,  # 帧级 CLIP 特征
}
```

---

## 信号计算

### 1. 轨迹信号 (Trajectory Signal)

**权重: 15%**

计算相邻帧之间的位置和角度变化：

```python
# 位置变化 (欧氏距离)
pos_diff = ||position[i] - position[i-1]||₂

# 角度变化 (四元数夹角)
dot = |q1 · q2|
angle_diff = 2 * arccos(clip(dot, -1, 1))

# 归一化并融合
pos_norm = pos_diff / max(pos_diff)
angle_norm = angle_diff / max(angle_diff)
trajectory = 0.6 * pos_norm + 0.4 * angle_norm
```

### 2. 物体密度信号 (Object Density Signal)

**权重: 20%**

计算相机周围一定半径内的物体数量变化：

```python
radius = 3.0  # 米

for i in range(n_frames):
    cam_pos = poses[i, :3]
    distances = ||object_centers - cam_pos||₂
    density[i] = count(distances < radius)

# 平滑后取差分
density_smooth = gaussian_filter1d(density, sigma=3)
density_signal = |diff(density_smooth)|
density_signal /= max(density_signal)
```

### 3. 可见性变化信号 (Visibility Signal)

**权重: 35%**

基于物体可见性矩阵，计算多尺度的 Jaccard 距离：

```python
# 构建可见性矩阵 [n_frames × n_objects]
for obj_id, obj in enumerate(objects):
    for frame_idx in obj['image_idx']:
        visibility_matrix[frame_idx, obj_id] = 1

# 每帧可见物体集合
frame_objects[i] = {obj_id | visibility_matrix[i, obj_id] == 1}

# 多尺度分析 (窗口: 3, 8, 15 帧)
for window in [3, 8, 15]:
    for i in range(window, n_frames):
        set_before = frame_objects[i - window]
        set_after = frame_objects[i]
        jaccard_dist = 1 - |set_before ∩ set_after| / |set_before ∪ set_after|
        signal[i] = jaccard_dist

# 多尺度融合
visibility = 0.2 * scale_3 + 0.4 * scale_8 + 0.4 * scale_15
visibility /= max(visibility)
```

### 4. 语义信号 (Semantic/CLIP Signal)

**权重: 30%**

基于帧级 CLIP 特征计算语义变化：

```python
# 从 gsa_detections_none/*.pkl.gz 加载帧级 CLIP 特征
frame_clip_features = []
for gsa_file in sorted(gsa_files):
    with gzip.open(gsa_file, 'rb') as f:
        data = pickle.load(f)
    frame_clip_features.append(data['frame_clip_feat'])  # (1024,)

# 计算帧间 CLIP 相似度矩阵
for i in range(n_frames):
    for j in range(i, n_frames):
        f1, f2 = frame_clip_features[i], frame_clip_features[j]
        # 余弦相似度 (特征已归一化，直接点积)
        cos_sim = dot(f1, f2)
        # 归一化到 [0, 1]
        clip_sim_matrix[i, j] = (cos_sim + 1) / 2

# 多尺度语义变化信号
for window in [3, 8, 15]:
    for i in range(window, n_frames):
        semantic_distance = 1 - clip_sim_matrix[i - window, i]
        signal[i] = semantic_distance

semantic = 0.2 * scale_3 + 0.4 * scale_8 + 0.4 * scale_15
semantic /= max(semantic)
```

---

## 信号融合

### 加权融合公式

```python
fused = (0.15 * trajectory +      # 轨迹信号
         0.20 * object_density +  # 物体密度信号
         0.35 * visibility +      # 可见性信号
         0.30 * semantic)         # 语义/CLIP 信号

# 高斯平滑
fused_smooth = gaussian_filter1d(fused, sigma=2.0)
```

### 权重设计理由

| 信号 | 权重 | 理由 |
|------|------|------|
| 轨迹 (Trajectory) | 15% | 运动是基础信号，但 Replica 数据运动较平滑，变化不剧烈 |
| 物体密度 (Object Density) | 20% | 物体分布变化反映空间转换，但受检测质量影响 |
| 可见性 (Visibility) | 35% | **最直接的场景变化指标**，基于物体可见帧列表 |
| 语义/CLIP (Semantic) | 30% | 帧级 CLIP 特征捕获高层语义，区分功能区域 |

---

## 帧间相似度矩阵

用于第二阶段的片段聚类，融合两种相似度：

### 1. 可见性相似度 (Jaccard)

```python
for i in range(n_frames):
    for j in range(i, n_frames):
        v1, v2 = visibility_matrix[i], visibility_matrix[j]
        intersection = sum(v1 & v2)
        union = sum(v1 | v2)
        visibility_sim[i, j] = intersection / union if union > 0 else 0
```

### 2. CLIP 语义相似度

```python
for i in range(n_frames):
    for j in range(i, n_frames):
        f1, f2 = frame_clip_features[i], frame_clip_features[j]
        cos_sim = dot(f1, f2)  # 已归一化
        clip_sim[i, j] = (cos_sim + 1) / 2  # 映射到 [0, 1]
```

### 3. 融合相似度

```python
# 50% 可见性 + 50% CLIP
similarity_matrix = 0.5 * visibility_sim + 0.5 * clip_sim
```

---

## 时序分割 (Stage 1)

### 变化点检测

```python
# 计算融合信号的梯度
gradient = |∇(fused_smooth)|
gradient_smooth = gaussian_filter1d(gradient, sigma=2)

# 峰值检测
peaks = find_peaks(
    gradient_smooth,
    prominence=0.02,    # 最小突出度
    distance=15         # 最小间距 (帧数)
)

# 创建时序片段
boundaries = [0] + peaks + [n_frames]
segments = []
for i in range(len(boundaries) - 1):
    start, end = boundaries[i], boundaries[i+1]
    if end - start >= 20:  # 最小片段长度
        segments.append({'start': start, 'end': end})
```

---

## 片段聚类 (Stage 2)

### 片段间相似度

```python
# 计算任意两个片段之间的平均相似度
for i in range(n_segments):
    for j in range(i, n_segments):
        sims = []
        for fi in range(seg_i['start'], seg_i['end']):
            for fj in range(seg_j['start'], seg_j['end']):
                sims.append(similarity_matrix[fi, fj])
        segment_sim[i, j] = mean(sims)
```

### 层次聚类

```python
distance_matrix = 1 - segment_sim

clustering = AgglomerativeClustering(
    n_clusters=auto_detected_k,
    metric='precomputed',
    linkage='average'
)
labels = clustering.fit_predict(distance_matrix)
```

### 自动 K 检测

```python
def compute_quality_score(labels, similarity_matrix, k):
    # 1. 区域内相似度 (越高越好)
    intra = []
    for c in range(k):
        mask = labels == c
        if sum(mask) > 1:
            intra.append(mean(similarity[mask][:, mask]))
    intra_score = mean(intra)
    
    # 2. 区域间分离度 (越高越好)
    inter = []
    for c1 in range(k):
        for c2 in range(c1+1, k):
            inter.append(mean(distance[labels==c1][:, labels==c2]))
    inter_score = mean(inter)
    
    # 3. 大小均衡度 (熵归一化)
    sizes = [sum(labels==c) for c in range(k)] / n_segments
    balance = -sum(sizes * log(sizes + 1e-10)) / log(k)
    
    # 综合分数
    return 0.40 * intra_score + 0.35 * inter_score + 0.25 * balance

# 遍历可能的 K 值，选择最佳
best_k = argmax([compute_quality_score(labels, sim, k) for k in range(3, 11)])
```

---

## 输出格式

### regions.json

```json
[
  {
    "region_id": 0,
    "n_frames": 68,
    "n_segments": 2,
    "segments": [
      {
        "start_frame": 0,
        "end_frame": 14,
        "start_frame_original": 0,
        "end_frame_original": 70
      },
      {
        "start_frame": 346,
        "end_frame": 400,
        "start_frame_original": 1730,
        "end_frame_original": 2000
      }
    ],
    "dominant_semantics": ["sofa", "lamp", "window"],
    "n_objects": 45,
    "object_ids": [0, 1, 2, ...]
  }
]
```

### 输出目录结构

```
sg_cache/segmentation_regions/
├── regions.json                        # 区域数据 (JSON)
├── segmentation.png                    # 信号分析可视化
├── regions_overview.jpg                # 区域总览图
├── segmentation_reasons.png            # 分割原因可视化
├── region_semantics.png                # 区域语义可视化
├── object_visibility_timeline_full.png # 物体可见性时间线
├── region_gifs/                        # 区域 GIF 动图
│   ├── region_00.gif
│   ├── region_01.gif
│   └── ...
├── region_keyframes/                   # 关键帧图像
│   ├── region_00_seg_00.jpg
│   └── ...
└── region_pointclouds/                 # 区域 3D 点云
    ├── region_00.ply
    ├── all_regions_colored.ply
    ├── pointcloud_views.png
    └── pointcloud_interactive.html
```

---

## 可视化说明

### 1. segmentation.png (信号分析图)

3x3 + 底部信号图布局：

| 位置 | 内容 |
|------|------|
| (1,1) | Visibility Similarity Matrix (Jaccard) |
| (1,2) | CLIP Semantic Similarity Matrix |
| (1,3) | Combined Similarity Matrix (50% + 50%) |
| (2,1) | Spatial Distribution (相机轨迹) |
| (2,2) | Region Assignment Timeline |
| (2,3) | Region Summary (统计信息) |
| (3,*) | Multi-modal Change Signals (全部信号 + 边界线) |

### 2. region_XX.gif (区域动画)

- 左上角: `Region X | Frame XXXX`
- 第二行: 语义标签
- 彩色边框: 区分不同区域
- 帧率: 5 FPS

### 3. segmentation_reasons.png (分割原因)

展示每个分割点的物体变化：
- **BEFORE/AFTER 图像对比**
- **LEAVING** (红色): 离开视野的物体
- **ENTERING** (绿色): 进入视野的物体

### 4. object_visibility_timeline_full.png (物体时间线)

- Y轴: 每个物体一行 (`ID=X: object_tag`)
- X轴: 帧索引
- 彩色条: 物体可见的帧范围
- 红色虚线: 分割边界点

---

## 使用方法

### Bash 脚本

```bash
bash bashes/4.5_semantic_scene_segmentation.sh room0
```

### Python 直接调用

```bash
python -m conceptgraph.segmentation.region_aware_segmenter \
    --dataset_root /path/to/Replica \
    --scene room0 \
    --stride 5 \
    --max_regions 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_root` | (必需) | 数据集根目录 |
| `--scene` | (必需) | 场景名称 |
| `--stride` | 5 | 帧采样步长 |
| `--max_regions` | 10 | 最大区域数 (自动检测) |

---

## 示例结果

### room0 场景

```
输入:
  - 位姿: 400 帧 (stride=5, 原始 2000 帧)
  - 3D物体: 75 个
  - 帧级CLIP特征: 400/400 帧

CLIP语义相似度: mean=0.886, std=0.051, range=[0.722, 0.998]

结果:
  区域 0: 68帧 (2片段: 0-14, 346-400)  ← 不连续片段!
  区域 1: 101帧 (1片段: 14-115)
  区域 2: 231帧 (1片段: 115-346)
```

---

## 配置参数

```python
@dataclass 
class RegionSegmenterConfig:
    # 信号权重
    trajectory_weight: float = 0.15
    object_density_weight: float = 0.20
    visibility_weight: float = 0.35
    semantic_weight: float = 0.30
    
    # 时序分割参数
    peak_distance: int = 15        # 峰值最小间距
    min_segment_frames: int = 20   # 最小片段长度
    
    # 区域聚类参数
    n_regions: int = 6             # (未使用，自动检测)
    merge_similarity: float = 0.5
    
    # 其他
    visibility_radius: float = 3.0 # 物体密度计算半径
    smooth_sigma: float = 2.0      # 高斯平滑参数
```

---

## 代码位置

```
conceptgraph/
├── scripts/
│   └── generate_gsa_results.py     # Step 1: 生成帧级 CLIP 特征
└── segmentation/
    └── region_aware_segmenter.py   # Step 4.5: 区域感知分割
```

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2024-12 | 基础时序分割 |
| v2.0 | 2024-12 | 两阶段架构，支持不连续片段 |
| v2.1 | 2024-12 | 自动 K 检测 (综合质量评分) |
| v2.2 | 2025-01 | 可解释性可视化 |
| v3.0 | 2025-01 | **帧级 CLIP 特征** 用于语义信号 |
