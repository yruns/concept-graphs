# 📊 Replica 数据集文件结构详解

## 概述

**Replica** 是 Facebook Research 提供的高质量室内场景数据集，专门用于：
- 3D 重建
- SLAM (同步定位与地图构建)
- 场景理解
- 机器人导航

数据集包含多个真实场景的**高精度3D重建**和**RGB-D序列**。

---

## 🗂️ Replica根目录结构

```
/home/shyue/Datasets/Replica/Replica/
├── room0/                    ← 场景数据目录
├── room0_mesh.ply            ← 原始场景mesh（Ground Truth）✨
├── room1/
├── room1_mesh.ply
├── room2/
├── room2_mesh.ply
├── office0/
├── office0_mesh.ply
├── office1/
├── office1_mesh.ply
... (其他场景)
```

---

## 📁 单个场景目录详解 (以room0为例)

### 🎯 Replica原始数据（官方提供）

```
room0/
├── results/                   ← RGB-D 图像序列
│   ├── frame000000.jpg        ← RGB彩色图像 (2000张)
│   ├── frame000001.jpg
│   ├── ...
│   ├── depth000000.png        ← 深度图 (2000张)
│   ├── depth000001.png
│   └── ...
│
├── traj.txt                   ← 相机轨迹 (2000行)
│                              每行: 4x4位姿矩阵 (16个浮点数)
│
└── room0_mesh.ply (在根目录)  ← Ground Truth 3D场景
```

### 🔧 ConceptGraphs生成的数据

```
room0/
├── gsa_detections_none/       ← GSA 2D物体检测结果
│   ├── frame000000.pkl.gz
│   └── ...
│
├── gsa_vis_none/              ← GSA 可视化结果
│   ├── frame000000.jpg
│   └── ...
│
├── gsa_classes_none.json      ← 检测到的类别列表
│
├── gsa_classes_none_colors.json  ← 类别颜色映射
│
├── pcd_saves/                 ← 3D对象地图
│   └── full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz
│
├── sg_cache/                  ← 场景图缓存
│   ├── cfslam_captions_llava/      ← LLaVA生成的物体描述
│   ├── cfslam_feat_llava/          ← 物体CLIP特征
│   ├── cfslam_gpt-4_responses/     ← GPT-4精炼的描述
│   ├── cfslam_llava_captions.json  ← 物体描述JSON
│   ├── cfslam_gpt-4_responses.pkl  ← GPT-4响应
│   ├── map/
│   │   └── scene_map_cfslam_pruned.pkl.gz  ← 最终场景图地图
│   ├── cfslam_object_relations.json        ← 物体关系
│   ├── cfslam_object_relation_queries.json ← 关系查询
│   └── cfslam_scenegraph_edges.pkl         ← 场景图边
│
├── scenegraph_output/         ← 旧版场景图输出(已弃用)
│
└── visualization/             ← 可视化输出 ✨
    ├── images/                ← 8张多视角PNG图像
    ├── ply/                   ← 3D模型
    │   ├── scene_pointcloud.ply
    │   └── scene_graph.ply
    ├── html/                  ← 交互式HTML
    │   └── scene_graph_interactive.html
    └── summary.txt            ← 摘要
```

---

## 🎨 关于 room0_mesh.ply

### 文件信息

- **类型**: PLY (Polygon File Format)
- **格式**: Binary Little Endian
- **大小**: ~41MB (room0)
- **内容**: 
  - **顶点数**: 954,492
  - **面数**: 953,647
  - **属性**: 位置(x,y,z) + 法向量(nx,ny,nz) + 颜色(RGB)

### PLY文件头结构

```ply
ply
format binary_little_endian 1.0
element vertex 954492          # 顶点数量
property float x                # 顶点坐标
property float y
property float z
property float nx               # 法向量
property float ny
property float nz
property uchar red              # 颜色 (0-255)
property uchar green
property uchar blue
element face 953647            # 面数量
property list uint8 int vertex_indices  # 面的顶点索引
end_header
[二进制数据...]
```

### ⚠️ 为什么在MeshLab中"颜色不好看"？

**原因分析**：

1. **它是纹理映射的mesh，不是point cloud**
   - Replica的mesh使用**顶点颜色**而不是纹理贴图
   - 可能需要在MeshLab中启用顶点颜色显示

2. **MeshLab显示设置问题**
   - 默认可能显示为灰色/白色
   - 需要手动启用颜色显示

3. **mesh是ground truth场景，不是重建结果**
   - 这是Replica官方提供的**完美3D模型**
   - 用作算法评估的参考标准
   - 比重建结果更加完整和准确

### 🔧 MeshLab中正确查看的方法

1. **打开文件**
   ```bash
   meshlab room0_mesh.ply
   ```

2. **启用顶点颜色**
   - `Render` → `Color` → `Per Vertex`
   - 或者 `Filters` → `Color Creation and Processing` → `Vertex Color from Texture`

3. **调整光照**
   - `Render` → `Lighting` → 调整为合适的光照模式
   - 尝试关闭 `Back Face Culling`

4. **查看wireframe**
   - `Render` → `Render Mode` → `Flat Lines` 
   - 可以同时看到mesh结构和颜色

---

## 📊 文件用途对比表

| 文件/目录 | 来源 | 用途 | 大小 |
|----------|------|------|------|
| `room0_mesh.ply` | Replica官方 | Ground Truth 3D场景 | 41MB |
| `results/*.jpg` | Replica官方 | RGB图像序列 (2000张) | ~1.5GB |
| `results/*.png` | Replica官方 | 深度图序列 (2000张) | ~800MB |
| `traj.txt` | Replica官方 | 相机轨迹（位姿） | 793KB |
| `gsa_detections_none/` | ConceptGraphs | 2D物体检测 | ~500MB |
| `pcd_saves/*.pkl.gz` | ConceptGraphs | 3D物体地图 | ~50MB |
| `sg_cache/` | ConceptGraphs | 场景图数据 | ~200MB |
| `visualization/` | ConceptGraphs | 可视化结果 | ~20MB |

---

## 🎯 mesh.ply vs 重建点云

### room0_mesh.ply (Ground Truth)
✅ **优点**:
- Replica官方提供的**完美3D模型**
- 高质量、完整、准确
- 包含精确的几何和颜色信息
- 可作为评估标准

❌ **缺点**:
- 文件很大（数十MB到数百MB）
- 需要特殊查看器（如MeshLab）
- 不是算法重建的，无法体现重建效果

### scene_pointcloud.ply (重建结果)
✅ **优点**:
- 展示算法的**实际重建能力**
- 文件较小（经过降采样）
- 包含物体分割信息
- 更适合与场景图一起可视化

❌ **缺点**:
- 可能不完整（取决于相机轨迹）
- 质量取决于算法性能
- 没有ground truth准确

---

## 💡 推荐的可视化方案

### 方案1: 叠加显示（已实现）✨

```bash
# 使用修改后的脚本，会自动叠加原始mesh
bash bashes/7_visualize_scene_graph_offscreen.sh
```

**效果**:
- 浅灰色半透明: 原始Replica场景 (ground truth)
- 彩色: 重建的物体点云
- 黄色球体: 场景图节点
- 红色线: 物体关系

### 方案2: 分别查看

```bash
# 查看Ground Truth
meshlab /home/shyue/Datasets/Replica/Replica/room0_mesh.ply

# 查看重建结果
meshlab /home/shyue/Datasets/Replica/Replica/room0/visualization/ply/scene_pointcloud.ply
```

### 方案3: 使用CloudCompare对比

```bash
# 在CloudCompare中同时加载两个文件进行对比
cloudcompare room0_mesh.ply scene_pointcloud.ply
```

---

## 🔍 数据流程图

```
Replica原始数据
    │
    ├─► RGB图像 (frame*.jpg)  ──────┐
    │                                │
    ├─► 深度图 (depth*.png)   ──────┤
    │                                │
    ├─► 相机轨迹 (traj.txt)   ──────┤
    │                                │
    └─► Ground Truth mesh     ──────┤
        (room0_mesh.ply)             │
                                     ▼
                            ConceptGraphs Pipeline
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
          2D分割              3D对象地图            场景图构建
      (GSA检测)              (点云融合)          (关系推理)
                │                    │                    │
                └────────────────────┴────────────────────┘
                                     │
                                     ▼
                              可视化输出
                    (images, ply, html) ✨
```

---

## 📝 常见问题

### Q1: mesh.ply太大，如何减小？
```bash
# 使用meshlab或open3d进行降采样
python -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('room0_mesh.ply')
mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
o3d.io.write_triangle_mesh('room0_mesh_simplified.ply', mesh)
"
```

### Q2: 如何从mesh生成point cloud？
```bash
python -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('room0_mesh.ply')
pcd = mesh.sample_points_uniformly(number_of_points=50000)
o3d.io.write_point_cloud('room0_pointcloud.ply', pcd)
"
```

### Q3: 如何比较ground truth和重建结果？
使用CloudCompare的Cloud-to-Mesh距离计算功能：
```bash
cloudcompare -SILENT \
    -O room0_mesh.ply \
    -O scene_pointcloud.ply \
    -C2M_DIST
```

---

## 🎓 总结

| 文件 | 性质 | 用途 |
|------|------|------|
| **room0_mesh.ply** | Ground Truth | 评估标准、对比参考 |
| **scene_pointcloud.ply** | 算法输出 | 展示重建效果 |
| **scene_graph.ply** | 算法输出 | 场景图可视化 |

**推荐做法**：
- 🎯 评估算法: 使用 `room0_mesh.ply` 作为参考
- 🎨 展示效果: 使用 `scene_pointcloud.ply` + `scene_graph.ply`
- 📊 论文配图: 叠加显示（已在脚本中实现）

---

生成时间: 2025-12-16
作者: ConceptGraphs Pipeline


