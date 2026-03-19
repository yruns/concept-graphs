# Stage-2 VLM Agent Design

## 1. 问题重述

传统场景图更像是一个压缩过的结构化中间表示：

- 输入通常是 `json` / object list / 坐标。
- 它对细粒度场景细节不敏感，尤其难以表达材质、状态、局部关系、遮挡和文本外观。
- 一旦场景图构建阶段漏检，后续纯场景图推理几乎没有补救能力。

因此，两阶段框架是合理的：

1. Stage 1: 先把用户任务 Query 解析成结构化查询，并从场景中检索当前任务最相关的关键帧。
2. Stage 2: 再把关键帧送入 VLM，由 Agent 在有限视觉预算下做下游任务推理。

这里的关键不是“再加一个大模型”，而是把 Stage 2 明确设计成一个 evidence-seeking agent，而不是一次性 prompt。

## 2. 这项工作的价值

### 2.1 为什么值得做

- 它把“结构化记忆”和“原始视觉证据”结合起来。
- Stage 1 用场景级索引解决长视频 / 长场景的 token budget 问题。
- Stage 2 用原始像素补回场景图的盲区，特别是漏检、小物体、局部状态和复杂空间关系。
- 这条链天然支持多任务统一：
  - QA
  - visual grounding
  - navigation planning
  - manipulation planning

### 2.2 对现有 query scene 的意义

当前仓库里的 `query_scene` 主线已经很适合作为 Stage 1：

- 它能把 query 解析成 `HypothesisOutputV1`
- 它能执行 `direct -> proxy -> context`
- 它能输出 task-relevant keyframes

但它目前基本停在“找到图”这一步。真正面向下游任务的多轮视觉推理 Agent 还没有正式建起来。

## 3. 学术创新性 review

### 3.1 只做工程拼装会有什么问题

如果第二阶段只是：

- 取 top-k 关键帧
- 全部喂给一个 VLM
- 直接让模型吐答案

那更像一个工程 pipeline，而不是一个有研究价值的方法。主要问题：

- 没有新的推理机制
- 没有新的 evidence selection 原理
- 没有对“漏检恢复”给出方法学贡献
- 很难写出清晰的 ablation 和 scientific claim

### 3.2 真正值得强调的研究点

要让第二阶段更像 research，而不是 prompt engineering，建议把创新点放在下面四类：

1. Agentic evidence acquisition
- VLM 不是被动吃图，而是主动决定下一步需要什么证据。
- 它可以请求更多视角、局部 crop、BEV、对象上下文。

2. Symbolic-to-visual repair
- Stage 1 的结构化 hypothesis 只是“软先验”，不是事实。
- Stage 2 要能验证、修正，甚至推翻 Stage 1 的假设。
- 这正是“场景图漏检时如何恢复”的方法学核心。

3. Budget-aware reasoning
- 在固定 token / image budget 下，Agent 如何停止、何时请求更多证据、何时给出不确定回答。
- 这让问题从“多模态问答”变成“受预算约束的视觉证据搜索”。

4. Unified task interface
- 用同一个 agent state / tool API 支撑 QA、grounding、nav、manipulation。
- 学术价值在于统一任务范式，而不是为每个任务手搓 prompt。

## 4. 优化后的两阶段研究思路

### 4.1 Stage 1 的角色

Stage 1 不再只是“query keyframe selection”，而是：

- task-conditioned evidence retriever
- 给出 hypothesis、关键帧、候选上下文和空间先验

Stage 1 的输出应被视为：

- 低成本召回
- 高召回但不一定高精度
- 给 Stage 2 提供视觉证据入口

### 4.2 Stage 2 的角色

Stage 2 不是简单 answerer，而是：

- 一个 ReAct-style VLM agent
- 一个 `LangChain v1 + DeepAgents` runtime，而不是手写循环
- 默认以 `gpt-5.2-2025-12-11` 作为单 key 的 AzureOpenAI-compatible backend，而不是 pool fan-out
- 输入是 keyframes + optional BEV + Stage 1 hypothesis metadata
- 输出是统一 structured response + evidence trace + uncertainty

它的目标不是重复 Stage 1，而是完成以下事情：

- 验证 Stage 1 给的关键帧是否足够
- 判断 query 对应的核心证据是否真的在图里
- 必要时请求补充证据
- 在多任务场景下生成最终任务结果

## 5. Stage-2 Agent 设计

### 5.1 输入

- `task_spec`
  - `qa`
  - `visual_grounding`
  - `nav_plan`
  - `manipulation`
- `stage1_evidence_bundle`
  - 原始 query
  - hypothesis kind / rank / status
  - keyframes
  - optional BEV
  - scene summary
  - object context

### 5.2 Agent 状态

Agent 维护三类内部状态：

1. Belief state
- 当前认为 target / anchor / region / affordance 是什么

2. Evidence ledger
- 哪些 frame 支持了哪些结论
- 哪些 frame 互相冲突

3. Uncertainty state
- 还缺什么证据
- 是否需要更多视角 / crop / metadata

### 5.3 Agent 工具

建议把第二阶段工具化，而不是只做长 prompt。

最小工具集合：

- `inspect_stage1_metadata`
  - 查看 Stage 1 hypothesis、frame mapping、selector metadata

- `retrieve_object_context`
  - 取场景对象 summary / affordance / scene summary

- `request_more_views`
  - 请求更多关键帧
  - 可以围绕 target / anchor / hypothesis kind 扩展

- `request_crops`
  - 请求 object-centric crop 或 frame region crop

- `switch_or_expand_hypothesis`
  - 允许 Stage 2 在 direct / proxy / context 间请求切换或扩展

最终答案不再通过自定义 `finish` action 返回，而是通过 DeepAgents 的 `response_format` 输出统一结构化结果。

### 5.4 ReAct 循环

Agent 每一步只做一件事，并由 planning mode 控制显式规划强度：

1. 读当前 keyframes 和 Stage 1 metadata
2. 判断证据是否足够
3. 如果不够，调用一个工具
4. 如果足够，输出 structured response

推荐三档 planning mode：

- `off`
  - 适合简单 QA / grounding，允许直接求解
- `brief`
  - 默认模式，先列短 todo，再做 evidence acquisition 与 synthesis
- `full`
  - 适合 nav / manipulation，显式维护 todo，并允许使用 DeepAgents subagents

这个设计的重点是：

- 显式证据缺口
- 显式工具调用
- 显式停止条件
- 统一多任务输出
- 稳定 `session_id` 驱动的 provider-side prompt caching

## 6. 不同任务的最终输出

最终统一使用一个 envelope：

- `task_type`
- `status`
- `summary`
- `confidence`
- `uncertainties`
- `cited_frame_indices`
- `evidence_items`
- `plan`
- `payload`

任务差异只放进 `payload`：

- QA
  - `answer`
  - `supporting_claims`

- Visual Grounding
  - `best_frames`
  - `target_description`
  - `grounding_rationale`

- Navigation Planning
  - `subgoals`
  - `landmarks`
  - `risks`

- Manipulation Planning
  - `target_object`
  - `preconditions`
  - `action_sequence`
  - `failure_checks`

这种设计的好处是：

- 多任务共享 evidence policy
- evaluation 层更容易统一统计
- 论文里更容易描述“同一 agent core, 不同 task payload”

## 7. 第二阶段怎么引入学术创新点

### 创新点 A: Adaptive evidence refinement

做法：

- 先给 Agent top-k 关键帧
- 若证据不足，Agent 再请求 `k+delta` 帧或某一对象局部 crop

研究问题：

- 在相同 token budget 下，adaptive refinement 是否优于 one-shot top-k？

### 创新点 B: Hypothesis repair

做法：

- Agent 读取 Stage 1 的 `direct/proxy/context`
- 在视觉上验证 hypothesis 是否成立
- 若 direct 不成立但 proxy 更像真，Agent 允许“视觉修正”

研究问题：

- 视觉修正是否能提高漏检场景下的任务成功率？

### 创新点 C: Evidence-grounded uncertainty

做法：

- Agent 必须输出 uncertainty 和 cited frames
- 若关键证据不可见，就不能强行给 deterministic answer

研究问题：

- 加 uncertainty-aware stopping 后，是否能降低 hallucination？

### 创新点 D: Unified task-conditioned policy

做法：

- 同一个 Agent，用不同 task head 支持 QA / grounding / nav / manipulation
- 共享 evidence selection policy

研究问题：

- 多任务共享的证据搜索策略是否比单任务手工 prompt 更高效？

## 8. 建议的实验与 ablation

### 主实验

- Stage 1 only
- Stage 1 + one-shot VLM
- Stage 1 + Stage-2 Agent

### 关键 ablation

- 无工具调用 vs 有工具调用
- 无更多视角请求 vs 自适应更多视角
- 无 hypothesis metadata vs 有 hypothesis metadata
- 无 uncertainty output vs 有 uncertainty output
- 单任务 agent vs 多任务统一 agent

### 特别建议加入的 stress test

- 人为 drop 掉 target / anchor 的 detection
- 观察 Stage 2 是否能从原始关键帧中恢复任务能力

这会直接击中你的核心 claim：场景图漏检时，纯结构化方法无能为力，而两阶段方法仍可恢复。

## 9. 当前仓库里的落地建议

本次实现建议采用两层交付：

1. 文档层
- 把 research framing 固化，避免后续实现滑回“工程拼装”

2. 代码层
- 把 Stage 2 从 `conceptgraph/query_scene/` 中拆到同级目录 `conceptgraph/agents/`
- 用 `LangChain v1 + DeepAgents` 实现 agent runtime
- 默认用单 key AzureOpenAI-compatible GPT 5.2 初始化模型，并在 `extra_body` 中写入稳定 `session_id`
- 固定 `Stage2TaskSpec + Stage2EvidenceBundle -> Stage2StructuredResponse` 的统一协议
- 具体的 more-view / crop / hypothesis backend 后续逐步接入

## 10. 本次结论

你的研究思路是成立的，而且方向是对的。

真正要避免的不是“用了某个大模型”，而是：

- 第二阶段如果只做一次性多图问答，会太像工程
- 第二阶段如果被设计成 evidence-seeking agent，就有更强的方法学空间

因此，推荐把论文叙事写成：

`Task-conditioned keyframe retrieval + agentic visual evidence reasoning`

而不是：

`Scene graph + VLM pipeline`
