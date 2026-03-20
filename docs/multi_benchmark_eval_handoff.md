# Multi-Benchmark Evaluation Handoff

本文是 2026-03-20 的多 benchmark 评估工作交接文档。目标是让下一个 agent 在 **支持 CUDA 的 Linux 环境**中接力当前任务，完成 ConceptGraph 场景图构建和 Stage 2 评估。

**交接原因**: macOS 不支持 CUDA，无法运行 GroundedSAM 检测和 3D 点云融合 pipeline。

---

## 1. 项目背景（略写）

### 1.1 两阶段研究框架

当前 query scene 主线采用两阶段范式：

1. **Stage 1**: task-conditioned query parsing + keyframe retrieval
2. **Stage 2**: VLM agentic reasoning over retrieved keyframes

核心动机：
- 传统场景图适合低成本召回，不适合表达细粒度视觉细节
- 一旦建图阶段漏检，后续纯场景图推理几乎无法恢复
- Stage 1 负责高召回证据检索，Stage 2 基于原始像素做验证和推理

### 1.2 当前 Stage 2 状态

Stage 2 代码在 `conceptgraph/agents/`，已完成：
- 清晰的研究 framing 和独立 package
- 统一 schema (`Stage2TaskSpec` / `Stage2EvidenceBundle` / `Stage2StructuredResponse`)
- DeepAgents runtime skeleton
- GPT 5.2 默认 backend
- 视觉证据回流闭环
- `MultiBenchmarkAdapter` 支持多种 benchmark

详细状态见 `docs/stage2_agent_handoff.md`。

---

## 2. 当前任务：多 Benchmark 评估

### 2.1 目标

在 OpenEQA、SQA3D、ScanRefer 等 benchmark 上评估 Stage 2 VLM agent 效果，验证两阶段框架的有效性。

### 2.2 支持的 Benchmark

| Benchmark | 任务类型 | 数据来源 | Stage 2 支持 |
|-----------|---------|---------|-------------|
| **OpenEQA** | Embodied QA | HM3D / ScanNet scenes | ✅ |
| **SQA3D** | Situated QA | ScanNet scenes | ✅ |
| **ScanRefer** | Visual Grounding | ScanNet scenes | ✅ |
| **Replica** | Full Pipeline | Replica scenes | ✅ 已验证 |

### 2.3 两种评估模式

1. **Full Pipeline Mode**: Stage 1 场景图构建 → Stage 2 推理
   - 需要 RGB + Depth + Camera Pose
   - 需要 GroundedSAM 检测 (CUDA)
   - 适合 Replica、完整 ScanNet

2. **Frame-Based Mode**: 跳过 Stage 1，直接用 benchmark 帧做 Stage 2
   - 只需要 RGB 帧
   - 已实现的 `MultiBenchmarkAdapter`
   - 适合快速验证 Stage 2 效果

---

## 3. 当前进展（详写）

### 3.1 已完成的数据下载

从 HuggingFace `ellisbrown/OpenEQA` 下载了 benchmark 数据：

```
/Users/bytedance/project/concept-graphs/data/benchmarks/open-eqa/data/
├── frames/
│   ├── hm3d-v0/      # 63 episodes, 15GB
│   └── scannet-v0/   # 89 scenes, 47GB
└── videos/           # 原始视频文件
```

**注意**：这些是从视频提取的 RGB 帧，**没有深度图和相机位姿**。

### 3.2 帧提取脚本

创建了 `videos2frames.py` 从视频提取帧：

```python
# /Users/bytedance/project/concept-graphs/data/benchmarks/open-eqa/data/videos2frames.py
def extract_frames(video_path: Path, output_dir: Path):
    output_pattern = str(output_dir / "%06d-rgb.png")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-q:v", "1", output_pattern]
    subprocess.run(cmd, capture_output=True, text=True)
```

### 3.3 已实现的 Benchmark Adapters

`conceptgraph/agents/benchmark_adapters.py` 提供统一接口：

```python
# Frame Providers
- MockFrameProvider      # 测试用
- OpenEQAFrameProvider   # 从 episode_history 加载帧
- ScanNetFrameProvider   # 从 ScanNet scene 加载帧
- ReplicaFrameProvider   # 从 Replica scene 加载帧

# 核心函数
- extract_sample_info(sample, benchmark_type) -> BenchmarkSampleInfo
- build_evidence_bundle_from_frames(info, frames) -> Stage2EvidenceBundle
- build_task_spec_from_sample(info) -> Stage2TaskSpec
- create_adapter_for_benchmark(type, data_root) -> MultiBenchmarkAdapter
```

### 3.4 已有的测试和示例

```bash
# 单测
.venv/bin/python -m pytest conceptgraph/agents/tests/test_benchmark_adapters.py -q

# Demo 脚本
.venv/bin/python -m conceptgraph.agents.examples.multi_benchmark_demo
```

---

## 4. 关键发现：Full Pipeline 的数据需求

### 4.1 ConceptGraph 场景图构建需要

| 输入 | 描述 | 必需性 |
|-----|------|-------|
| RGB 帧 | 彩色图像序列 | ✅ 必需 |
| Depth 帧 | 深度图像序列 | ✅ 必需 |
| Camera Poses | 每帧 4x4 位姿矩阵 | ✅ 必需 |
| Camera Intrinsics | fx, fy, cx, cy | ✅ 必需 |
| 2D 分割结果 | GroundedSAM 检测 | ✅ 必需 (需 CUDA) |

### 4.2 各数据集的完备性

| 数据集 | RGB | Depth | Pose | Intrinsics | Full Pipeline |
|-------|-----|-------|------|------------|---------------|
| **Replica** | ✅ | ✅ | ✅ | ✅ | ✅ 可行 |
| **ScanNet (完整版)** | ✅ | ✅ | ✅ | ✅ | ✅ 可行 |
| **OpenEQA HM3D (视频)** | ✅ | ❌ | ❌ | ❌ | ❌ 不可行 |
| **OpenEQA ScanNet (视频)** | ✅ | ❌ | ❌ | ❌ | ❌ 不可行 |

### 4.3 已有的 Dataset Loaders

代码库已经有这些 dataset class：

```python
# conceptgraph/dataset/datasets_common.py
- ReplicaDataset        # ✅ 完整支持
- ScannetDataset        # ✅ 完整支持
- Hm3dOpeneqaDataset    # ✅ 完整支持 (需要完整数据)
- Hm3dDataset           # ✅ 完整支持
- Ai2thorDataset        # ✅ 完整支持
```

对应的配置文件：

```
conceptgraph/dataset/dataconfigs/
├── replica/replica.yaml
├── scannet/base.yaml
├── hm3d-openeqa.yaml
├── hm3d.yaml
└── ai2thor/
```

---

## 5. 待解决的问题

### 5.1 获取完整 ScanNet 数据

**问题**: 当前只有从视频提取的 RGB 帧，没有深度和位姿。

**解决方案**: 需要到 http://www.scan-net.org/ 申请完整数据。

ScanNet 完整数据结构：
```
scannet_root/
└── scene0001_00/
    ├── color/*.jpg          # RGB 帧
    ├── depth/*.png          # 深度图
    ├── pose/*.txt           # 相机位姿
    └── intrinsic/
        └── intrinsic_color.txt  # 相机内参
```

### 5.2 获取 HM3D-OpenEQA 完整数据

**问题**: HuggingFace 只提供了视频，没有深度和位姿。

**解决方案**:
1. 从 Habitat 仿真器重新渲染（需要 HM3D 数据集 + Habitat-sim）
2. 或者只使用 Frame-Based 模式评估

### 5.3 运行 GroundedSAM 检测

**问题**: macOS 不支持 CUDA。

**解决方案**: 在 Linux + CUDA 环境中运行：

```bash
# 检测脚本
bash bashes/1b_run_gsa_detections.sh <scene_name>

# 输出目录
<scene_path>/gsa_detections_ram_withbg_allclasses/*.pkl.gz
```

### 5.4 运行 3D 点云融合

**问题**: 需要 CUDA 支持的 Open3D 和 PyTorch。

```bash
# 融合脚本
bash bashes/2b_build_3d_object_map_detect.sh <scene_name>

# 核心命令
python conceptgraph/slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    scene_id=$SCENE_NAME \
    ...
```

---

## 6. 推荐的下一步

### 6.1 短期：Frame-Based 评估（无需 CUDA）

先用 Frame-Based 模式验证 Stage 2 效果：

```python
from conceptgraph.agents import (
    Stage2DeepResearchAgent,
    Stage2DeepAgentConfig,
    create_adapter_for_benchmark,
)

# 创建 adapter
adapter = create_adapter_for_benchmark("openeqa", data_root)

# 准备输入
task, bundle = adapter.prepare_stage2_inputs(sample, "openeqa")

# 运行 Stage 2
agent = Stage2DeepResearchAgent(config=Stage2DeepAgentConfig())
result = agent.run(task, bundle)
```

这可以在任何环境中运行，不需要 CUDA。

### 6.2 中期：ScanNet Full Pipeline（需要 CUDA）

1. 申请 ScanNet 数据
2. 下载到 Linux 环境
3. 运行 GroundedSAM 检测
4. 运行 3D 融合
5. 运行 Stage 1 + Stage 2 完整评估

### 6.3 长期：HM3D Full Pipeline（需要 Habitat）

1. 下载 HM3D 数据集
2. 配置 Habitat-sim
3. 重新渲染 depth + pose
4. 运行完整 pipeline

---

## 7. 环境配置

### 7.1 macOS 环境（当前）

```bash
# Python 环境
.venv/bin/python

# 可以运行
- Stage 2 单测
- Frame-Based 评估
- Benchmark adapter tests

# 不能运行
- GroundedSAM 检测 (需要 CUDA)
- 3D 点云融合 (需要 CUDA)
```

### 7.2 Linux + CUDA 环境（目标）

```bash
# Python 环境
conda activate conceptgraph

# 需要的包
- torch (CUDA)
- open3d
- grounded-sam-2
- gradslam

# 关键环境变量
export REPLICA_ROOT=/path/to/Replica
export SCANNET_ROOT=/path/to/ScanNet
export HM3D_ROOT=/path/to/HM3D
```

---

## 8. 文件清单

### 8.1 关键代码文件

```
conceptgraph/agents/
├── __init__.py
├── models.py                    # Stage 2 数据模型
├── adapters.py                  # Stage 1 -> Stage 2 adapter
├── benchmark_adapters.py        # 多 benchmark 适配器 ⭐
├── stage2_deep_agent.py         # Stage 2 Agent 主体
└── examples/
    └── multi_benchmark_demo.py  # Demo 脚本 ⭐
```

### 8.2 数据文件

```
data/benchmarks/open-eqa/data/
├── frames/
│   ├── hm3d-v0/      # 63 episodes (15GB) ⭐
│   └── scannet-v0/   # 89 scenes (47GB) ⭐
├── videos/           # 原始视频
├── videos2frames.py  # 帧提取脚本 ⭐
└── open-eqa-v0.json  # OpenEQA 元数据
```

### 8.3 配置文件

```
conceptgraph/dataset/dataconfigs/
├── replica/replica.yaml
├── scannet/base.yaml
└── hm3d-openeqa.yaml
```

### 8.4 相关文档

```
docs/
├── stage2_vlm_agent_design.md      # Stage 2 设计文档
├── stage2_agent_handoff.md         # Stage 2 交接文档
└── multi_benchmark_eval_handoff.md # 本文件 ⭐

memory/
├── research_direction.md
├── project_context.md
└── query_scene_knowledge.md
```

---

## 9. 快速开始清单

在 Linux + CUDA 环境中接力任务：

- [ ] Clone 仓库并配置 conda 环境
- [ ] 复制 `data/benchmarks/open-eqa/data/frames/` 到新环境（约 62GB）
- [ ] 申请 ScanNet 完整数据 (http://www.scan-net.org/)
- [ ] 验证 CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 验证 GroundedSAM 安装：`python -c "from grounded_sam_2 import ..."`
- [ ] 对一个 ScanNet scene 跑 full pipeline 验证
- [ ] 批量运行 benchmark 评估

---

## 10. 一句话总结

macOS 环境已完成：benchmark 数据下载、帧提取、adapter 实现。Linux CUDA 环境需要完成：ScanNet 完整数据获取、GroundedSAM 检测、3D 融合、full pipeline 评估。Frame-Based 模式可以在任何环境先行验证 Stage 2 效果。
