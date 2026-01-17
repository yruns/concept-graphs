# 时序场景分段算法

## 概述

本模块实现了基于多模态信号融合的时序场景分段算法，将连续的探索轨迹划分为语义连贯的局部区域。这是构建层次化场景图的关键第一步。

## 算法原理

### 理论基础

人类或机器人探索环境时的运动模式与场景功能分区存在内在关联：
- **停留环顾** → 高信息密度区域
- **快速穿越** → 过渡空间
- **视觉突变** → 功能边界跨越

### 多模态信号融合

| 信号类型 | 提取方式 | 语义含义 | 权重 |
|---------|---------|---------|------|
| 运动信号 | 相邻帧位姿的欧氏距离与角度变化 | 运动速度极值点对应行为模式转换 | 25% |
| 视觉信号 | 多尺度窗口的 CLIP 特征余弦相似度 | 相似度骤降指示功能区域转换 | 50% |
| 语义信号 | 物体类别变化 (Jaccard 距离) | 类别组成变化指示进入新区域 | 25% |

## 算法流程

```
输入: 
  - 相机轨迹 (traj.txt)
  - GSA 检测结果 (gsa_detections_ram/)

步骤:
  1. 提取运动信号
     - 计算相邻帧的位置差和角度差
     - 高斯平滑 (σ=2.0)
     - 融合: 0.7 * 位置变化 + 0.3 * 角度变化

  2. 提取多尺度视觉信号
     - 使用 3 个窗口大小: 10, 20, 30 帧
     - 计算窗口内的 CLIP 特征余弦相似度
     - 加权融合: 0.2 * W10 + 0.3 * W20 + 0.5 * W30

  3. 提取语义信号 (需要 RAM 模式检测结果)
     - 计算相邻帧的物体类别 Jaccard 距离
     - 追踪新类别的涌现
     - 检测边界物体 (door, window, wall)

  4. 信号融合
     - fused = 0.25 * motion + 0.50 * visual + 0.25 * semantic

  5. 变化点检测
     - 计算融合信号的梯度
     - 使用 scipy.signal.find_peaks 检测峰值
     - 参数: prominence=0.02, distance=20

  6. 分段生成与合并
     - 根据变化点划分区域
     - 合并过短的分段 (< 40 帧)
     - 合并视觉高度相似的相邻区域 (相似度 > 0.92)

输出:
  - trajectory_segments.json: 分段结果
  - segmentation_signals.json: 信号数据
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MIN_SEGMENT_FRAMES` | 40 | 最小分段帧数 |
| `MERGE_SIMILARITY_THRESHOLD` | 0.92 | 相似区域合并阈值 |
| `PEAK_PROMINENCE` | 0.02 | 峰值显著性阈值 |
| `PEAK_DISTANCE` | 20 | 峰值最小间距 |
| `SMOOTH_SIGMA` | 2.0 | 高斯平滑参数 |
| `VISUAL_WINDOWS` | [10, 20, 30] | 多尺度窗口大小 |

## 使用方法

### 前置条件

需要先运行 RAM 模式的 2D 分割以获取物体类别信息：

```bash
bash bashes/1b_extract_2d_segmentation_detect.sh
```

### 运行分段

```bash
cd /home/shyue/codebase/concept-graphs
source /home/shyue/anaconda3/bin/activate conceptgraph
source env_vars.bash

python conceptgraph/segmentation/balanced_segmenter.py \
    --scene room0 \
    --gsa_mode ram_withbg_allclasses \
    --target_regions 4-6
```

### 输出文件

```
$REPLICA_ROOT/room0/sg_cache/segmentation_balanced/
├── trajectory_segments.json    # 分段结果
└── segmentation_signals.json   # 信号数据

/home/shyue/codebase/concept-graphs/room0_seg/
├── balanced_segmentation.png   # 信号可视化
└── region_vis_balanced/
    ├── regions_overview.jpg    # 区域总览
    ├── region_keyframes/       # 关键帧图像
    └── region_gifs/            # 区域动画
```

## 实验结果

### room0 场景分段结果

使用平衡版参数，将 400 帧（stride=5）的轨迹分为 4 个区域：

| 区域 | 帧范围 | 帧数 | 主要物体 |
|-----|--------|------|---------|
| R0 | 0-23 | 23 | couch, furniture, floor, drawer |
| R1 | 23-107 | 84 | door, drawer, blind, chair |
| R2 | 107-300 | 193 | blind, window, anchor, boat |
| R3 | 300-400 | 100 | sculpture, blind, window |

### 方法对比

| 方法 | 分段数 | 特点 |
|------|--------|------|
| 仅运动信号 | 16 | 过度分割，对小运动敏感 |
| 多模态 (none 模式) | 28 | 无类别信息，语义信号弱 |
| 多模态 (RAM 模式) | 16 | 有类别信息，但分段仍较多 |
| **平衡版 (RAM + 合并)** | **4** | **语义合理，区域明显不同** |

## 改进历程

### V1: 仅运动信号
- 使用相邻帧的位姿变化
- 问题：对小的相机抖动敏感，分段过多

### V2: 加入视觉信号 (相邻帧)
- 计算相邻帧的 CLIP 相似度
- 问题：Replica 数据相邻帧极其相似（相似度 > 0.99），信号太弱

### V3: 多尺度视觉信号
- 使用滑动窗口 (5, 10, 20 帧)
- 改进：能检测到更明显的视觉变化

### V4: 加入 RAM 类别检测
- 使用 RAM 模型获取物体类别
- 改进：可追踪类别变化、检测边界物体

### V5: 平衡版 (最终版)
- 增大窗口 (10, 20, 30 帧)
- 增加最小分段帧数 (40 帧)
- 合并视觉相似的相邻区域
- 结果：4 个语义明确的区域

## 代码结构

```
conceptgraph/segmentation/
├── __init__.py
├── signal_extractors.py      # 信号提取器
├── trajectory_segmenter.py   # 基础分段器
├── enhanced_segmenter.py     # 增强版分段器
├── advanced_visual_analysis.py  # 高级视觉分析
└── visualizer.py             # 可视化工具
```

## 下一步

1. **集成到 Step 6**: 将分段结果用于构建层次化场景图
2. **区域描述生成**: 为每个区域生成自然语言描述
3. **跨区域关系推理**: 分析区域之间的空间和功能关系

## 参考

- ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning
- Segment Anything Model (SAM)
- Recognize Anything Model (RAM)
- CLIP: Contrastive Language-Image Pre-training
