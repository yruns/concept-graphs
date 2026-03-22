# 周报：Stage 2 Agent 系统搭建与 3D 场景理解任务调研

**时间**：2026.03.16 - 2026.03.22
**项目**：ConceptGraph 两阶段场景理解框架

---

## 一、Stage 2 Agent 基础架构

本周完成了 Stage 2 Agent 的核心架构搭建，基于 LangChain v1 + DeepAgents 实现了完整的推理运行时。

**DeepAgents Runtime**：实现了 `Stage2DeepResearchAgent`，支持 `qa`、`visual_grounding`、`nav_plan`、`manipulation` 四类下游任务。采用统一的 structured response envelope，并引入 uncertainty-aware stopping 机制，使 agent 能够在证据不足时主动报告置信度。

**可观测性系统**：构建了完整的执行轨迹记录与可视化系统，包括：

- **TraceRecorder**：记录 agent 每一步的工具调用、输入输出和推理过程
- **SQLite TraceDB**：持久化存储执行轨迹，支持历史查询和对比分析
- **TraceServer Web UI**：提供 dark-themed 的浏览器界面，支持实时查看和调试 agent 执行过程

**Stage1-Stage2 交互**：通过 `Stage1BackendCallbacks` 实现了两阶段之间的 `request_more_views` 交互，允许 Stage 2 在推理过程中向 Stage 1 请求更多视觉证据。

---

## 二、3D 场景理解任务调研

系统梳理了 Replica、ScanNet、HM3D 上的 3D 推理相关任务，明确了研究切入点。

### 任务演进脉络

**静态场景 3D QA（2022-2023）**：以 ScanQA 和 SQA3D 为代表，重点考察 3D 感知、对象定位和空间关系理解。这一阶段不强调主动探索，更像"给定完整场景后做问答"。

**Embodied EQA（2024）**：以 OpenEQA 为标志性工作，实现了范式转变——从"看懂场景"转向"探索 → 整合证据 → 开放回答"。OpenEQA 基于 180+ 真实环境构建了 1600+ 人工标注问题，支持 episodic memory 和 active exploration 两种设定，使用开放词表回答。

**强推理细分（2024-2025）**：任务开始分化，Space3D-Bench（Replica）侧重强空间推理，MT-HM3D（HM3D）侧重 memory-centric 的多区域信息整合。

### 研究方向确定

经评估，**OpenEQA 与当前两阶段框架高度契合**：

- Stage 1 的高召回视觉证据检索对应 OpenEQA 的 episodic memory 设定
- Stage 2 的 VLM Agent 推理对应开放词表 + 证据整合的要求

---

## 三、数据准备与基础构建

完成了 OpenEQA 数据集的下载与格式转换：

| 工作项 | 说明 |
|--------|------|
| 数据下载 | HM3D + ScanNet 两个数据源的 episode 视频 |
| 格式转换 | 将 MP4 视频转换为帧序列，适配 Stage 1 输入 |
| 索引构建 | 基于 ConceptGraph 构建场景图和图像-目标双向索引 |

同时下载了 Space3D-Bench 并完成适配器开发，用于空间推理能力的专项评估。

---

## 四、下一步计划

1. **Stage 1 优化**：提升关键帧检索召回率，适配 OpenEQA 视频帧输入格式
2. **Stage 2 端到端评估**：在 OpenEQA 上跑通完整评估流程，接入 LLM-Match 评估器
3. **Baseline 对比**：复现 OpenEQA 论文中的 GPT-4V baseline 作为性能参照
