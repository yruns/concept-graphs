# Research Direction

## 当前研究主线（2026-03-19）

本项目当前的 query scene 研究方向采用两阶段框架：

1. Stage 1: task-conditioned query parsing + keyframe retrieval
2. Stage 2: VLM agentic reasoning over retrieved keyframes

## 为什么需要两阶段

- 传统场景图通常是结构化 `json` / object list / 坐标，细粒度视觉细节不足。
- 场景图构建阶段若出现漏检，后续纯场景图推理几乎无法恢复。
- 因此不能把场景图当最终证据，只能把它当低成本召回与结构先验。

## Stage 1 定位

- 解析用户任务 query（QA / visual grounding / nav plan / manipulation 等）
- 将 query 转成结构化 hypothesis / query program
- 在场景中检索当前任务最相关的关键帧
- 输出内容应视为“视觉证据入口”，而不是最终答案

当前仓库中，`KeyframeSelector.select_keyframes_v2()` 已是 Stage 1 主入口。

## Stage 2 定位

- 输入：Stage 1 选出的关键帧、optional BEV、hypothesis metadata、scene/object context
- 核心：让 VLM 以 Agent 方式做多步推理，而不是一次性 prompt
- 当前 canonical 实现：`LangChain v1 + DeepAgents`
- Stage 2 模型接入采用单 key 的 AzureOpenAI-compatible Gemini client，不走 `GeminiClientPool`
- Stage 2 默认应使用项目内办公网 base url：`https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl`
- 初始化时必须写入稳定的 `extra_body.session_id`，并默认开启 `extra_body.thinking.include_thoughts=True`，以便利用 provider-side prompt caching
- 推荐范式：ReAct + planning + 显式工具调用 + unified structured response

当前代码位置：

- `conceptgraph/agents/models.py`
- `conceptgraph/agents/adapters.py`
- `conceptgraph/agents/stage2_deep_agent.py`
- `conceptgraph/agents/tests/test_stage2_deep_agent.py`

建议最小工具集：

- `inspect_stage1_metadata`
- `retrieve_object_context`
- `request_more_views`
- `request_crops`
- `switch_or_expand_hypothesis`

## Stage 2 设计约束

- Stage 2 不应与 `conceptgraph/query_scene/` 混放；Stage 1 和 Stage 2 代码边界必须清晰。
- Stage 2 必须把 Stage 1 hypothesis 当软先验，而不是真值。
- Stage 2 统一支持：
  - `qa`
  - `visual_grounding`
  - `nav_plan`
  - `manipulation`
- Stage 2 统一输出使用 `Stage2StructuredResponse`：
  - 公共字段包含 `status / summary / confidence / uncertainties / cited_frame_indices / evidence_items / plan / payload`
  - 各任务差异只体现在 `payload`
- planning 通过 `plan_mode=off|brief|full` 控制：
  - `off`：简单任务直接做
  - `brief`：默认模式，先列简短 todo 再推理
  - `full`：多步任务显式规划，并允许使用 DeepAgents subagents

## 学术定位

第二阶段要避免只做工程拼装。更推荐强调：

- adaptive evidence acquisition
- symbolic-to-visual repair
- budget-aware uncertainty-aware reasoning
- unified task-conditioned policy across QA / grounding / nav / manipulation

## 仓库内对应产物

- 设计文档：`docs/stage2_vlm_agent_design.md`
- Agent 主线：`conceptgraph/agents/stage2_deep_agent.py`
- Stage 1 到 Stage 2 adapter：`conceptgraph/agents/adapters.py`

## 后续实现原则

- Stage 1 是 high-recall evidence retriever
- Stage 2 是 evidence-grounded task agent
- Stage 2 应把 Stage 1 hypothesis 当软先验，不当真值
- Stage 2 的 tool layer 要优先服务于：
  - adaptive evidence refinement
  - symbolic-to-visual repair
  - budget-aware uncertainty
  - unified task-conditioned policy
- 未来若修改 Stage 2 主线或接口，需要同步更新本文件和 `AGENTS.md`
