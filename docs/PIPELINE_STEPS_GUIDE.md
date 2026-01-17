# ConceptGraphs 流程步骤详解

本文档详细说明了 ConceptGraphs 系统从 2D 图像到 3D 场景图构建的完整流程。

---

## 步骤 1: 提取 2D 分割和 CLIP 特征

### 作用
- 使用 SAM (Segment Anything Model) 对输入图像进行 2D 语义分割
- 为每个分割区域提取 CLIP 视觉特征向量
- 采用类别无关模式，不依赖预定义类别列表

### 输入
- **RGB 图像**: `$REPLICA_ROOT/$SCENE_NAME/results/color/*.jpg`
- **深度图像**: `$REPLICA_ROOT/$SCENE_NAME/results/depth/*.png`

### 输出
- **分割结果**: `$REPLICA_ROOT/$SCENE_NAME/gsa_results_none/*.pkl.gz`
  - 每帧图像的分割掩码
  - CLIP 特征向量
  - 边界框坐标
- **可视化图像**: `$REPLICA_ROOT/$SCENE_NAME/gsa_vis_none/*.jpg`
  - 可视化分割效果

### 关键脚本
```bash
python scripts/generate_gsa_results.py
```

### 关键参数
- `--class_set none`: 类别无关模式
- `--stride 5`: 每隔 5 帧处理一次

---

## 步骤 2: 构建 3D 对象地图

### 作用
- 将 2D 分割结果投影到 3D 空间
- 跨帧关联和匹配同一物体（通过 CLIP 特征相似度）
- 融合多视图点云，构建完整的 3D 物体表示
- 使用 DBSCAN 聚类合并重复检测的物体
- 计算物体的 3D 边界框和空间位置

### 输入
- **2D 分割结果**: `$REPLICA_ROOT/$SCENE_NAME/gsa_results_none/*.pkl.gz`
- **相机位姿**: `$REPLICA_ROOT/$SCENE_NAME/traj.txt`
- **RGB-D 数据**: `$REPLICA_ROOT/$SCENE_NAME/results/`

### 输出
- **3D 对象地图 (后处理版本，推荐使用)**:
  - `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz`
- **3D 对象地图 (原始版本)**:
  - `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz`

### 关键脚本
```bash
python slam/cfslam_pipeline_batch.py
```

### 关键参数
- `spatial_sim_type=overlap`: 使用重叠度计算空间相似度
- `mask_conf_threshold=0.95`: 掩码置信度阈值
- `match_method=sim_sum`: 使用特征相似度和匹配
- `sim_threshold=1.2`: 相似度阈值
- `dbscan_eps=0.1`: DBSCAN 聚类半径
- `merge_interval=20`: 每 20 帧合并一次物体

---

## 步骤 3: 可视化 3D 对象地图 (可选)

### 作用
- 交互式可视化 3D 对象地图
- 验证物体分割和点云融合质量
- 支持多种着色和查看模式

### 输入
- **3D 对象地图**: `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz`

### 输出
- **交互式 Open3D 可视化窗口** (无文件输出)

### 交互快捷键
| 按键 | 功能 |
|------|------|
| `b` | 切换背景点云显示 |
| `c` | 按类别着色 |
| `r` | 按 RGB 着色 |
| `f` | 按 CLIP 文本相似度着色（需输入查询文本）|
| `i` | 按实例 ID 着色 |

### 关键脚本
```bash
python scripts/visualize_cfslam_results.py
```

---

## 步骤 4: 提取物体描述

### 作用
- 使用视觉-语言模型 (Vision-Language Model) 为每个 3D 物体生成自然语言描述
- 处理每个物体的多个观察视角
- 从原始图像中裁剪物体区域作为输入
- 生成初始的物体标注

### 输入
- **3D 对象地图**: `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz`
- **物体裁剪图像**: 从原始 RGB 图像中根据掩码动态提取

### 输出
- **物体描述 JSON**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_llava_captions.json`
  - 每个物体的多视图描述列表
- **CLIP 特征**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_feat_llava/*.pt`
  - 基于描述的 CLIP 文本特征
- **调试可视化**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_captions_llava_debug/*.png`
  - 物体裁剪图像和对应描述

### 关键脚本
```bash
python scenegraph/build_scenegraph_cfslam.py --mode extract-node-captions
```

### 依赖服务
- **LLM 服务器**: 需要运行 Vision-Language 模型服务
- **环境变量**: `LLM_BASE_URL`, `LLM_MODEL`

---

## 步骤 5: 细化物体描述

### 作用
- 使用大语言模型 (LLM) 整合每个物体的多视图描述
- 识别和解决不同视角描述之间的冲突
- 生成统一、准确的物体语义标签
- 过滤无效或低质量的物体检测

### 输入
- **原始描述**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_llava_captions.json`
- **3D 对象地图**: `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz`

### 输出
- **精炼描述 (每个物体单独文件)**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_gpt-4_responses/*.json`
- **汇总文件**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_gpt-4_responses.pkl`

### 关键脚本
```bash
python scenegraph/build_scenegraph_cfslam.py --mode refine-node-captions
```

### 依赖服务
- **LLM 服务器**: 需要运行文本生成模型
- **环境变量**: `LLM_BASE_URL`, `LLM_MODEL`

---

## 步骤 6: 构建场景图

### 作用
- 分析 3D 空间中物体之间的几何和语义关系
- 使用大语言模型推理物体间的空间关系（如"在...上"、"在...里"、"旁边"等）
- 构建场景图（节点 = 物体，边 = 关系）
- 使用最小生成树算法优化和剪枝图结构

### 输入
- **精炼描述**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_gpt-4_responses/`
- **3D 对象地图**: `$REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz`

### 输出
- **场景图地图 (剪枝后)**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/map/scene_map_cfslam_pruned.pkl.gz`
- **物体关系 JSON**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relations.json`
- **关系查询记录**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relation_queries.json`
- **场景图边数据**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_scenegraph_edges.pkl`
- **可读摘要**: 
  - `$REPLICA_ROOT/$SCENE_NAME/sg_cache/scene_graph.json`

### 关键脚本
```bash
python scenegraph/build_scenegraph_cfslam.py --mode build-scenegraph
```

### 依赖服务
- **LLM 服务器**: 需要运行推理模型
- **环境变量**: `LLM_BASE_URL`, `LLM_MODEL`

---

## 步骤 7: 可视化场景图

### 作用
- 交互式可视化完整的 3D 场景图
- 显示物体节点（带标签）和关系边
- 支持多种查看和着色模式
- 用于验证场景图质量和关系正确性

### 输入
- **场景图地图**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/map/scene_map_cfslam_pruned.pkl.gz`
- **物体关系**: `$REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relations.json`

### 输出
- **交互式 Open3D 可视化窗口** (无文件输出)

### 交互快捷键
| 按键 | 功能 |
|------|------|
| `g` | ⭐ 显示/隐藏场景图（关系边和连接）|
| `b` | 切换背景点云显示 |
| `c` | 按类别着色 |
| `r` | 按 RGB 着色 |
| `f` | 按 CLIP 文本相似度着色 |
| `i` | 按实例 ID 着色 |
| `+` | 增大点云点大小 |
| `-` | 减小点云点大小 |

### 关键脚本
```bash
python scripts/visualize_cfslam_results.py --edge_file
```

---

## 流程总结

```
原始数据 (RGB-D 图像)
    ↓
[步骤 1] 2D 分割 + CLIP 特征
    ↓
[步骤 2] 3D 对象地图构建
    ↓
[步骤 3] 可视化验证 (可选)
    ↓
[步骤 4] 物体描述提取 (Vision-Language Model)
    ↓
[步骤 5] 物体描述细化 (Large Language Model)
    ↓
[步骤 6] 场景图构建 (关系推理)
    ↓
[步骤 7] 场景图可视化
```

## 关键技术组件

1. **SAM (Segment Anything Model)**: 2D 图像分割
2. **CLIP**: 视觉和文本特征提取
3. **DBSCAN**: 点云聚类和物体合并
4. **Vision-Language Model**: 物体描述生成
5. **Large Language Model**: 描述细化和关系推理
6. **Open3D**: 3D 可视化

## 主要配置文件

- **环境变量**: `/home/shyue/codebase/concept-graphs/env_vars.bash`
- **数据集配置**: `$REPLICA_CONFIG_PATH`
- **Hydra 配置**: `conceptgraph/hydra_configs/`

## 注意事项

1. **步骤依赖**: 每个步骤依赖前一步骤的输出，必须按顺序执行
2. **LLM 服务**: 步骤 4-6 需要 LLM 服务器运行（需配置 `LLM_BASE_URL`）
3. **存储空间**: 中间结果会占用较大存储空间
4. **处理时间**: 完整流程可能需要较长时间，取决于场景大小和硬件配置
5. **可视化步骤**: 步骤 3 和 7 是可选的，主要用于质量检查

## 快速运行

如果要一次性运行所有步骤，可以使用：
```bash
bash bashes/run_all.sh
```

详细的起步说明请参考：
```bash
bashes/00_START_HERE.md
```
