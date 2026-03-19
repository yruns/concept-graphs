# Stage-2 Agent Handoff

本文是 2026-03-20 的 Stage 2 工作交接文档。目标是让下一个进入仓库的 Codex / Claude Code 在完成 `AGENTS.md` 的 memory bootstrap 后，直接接力当前 Stage 2 agent 研发，而不需要重新摸索上下文。

**最近更新 (2026-03-20)**: 已完成 P0 - 视觉证据回流闭环。

## 1. 当前研究 framing

当前 query scene 主线采用两阶段范式：

1. Stage 1: task-conditioned query parsing + keyframe retrieval
2. Stage 2: VLM agentic reasoning over retrieved keyframes

问题动机已经明确：

- 传统场景图更像结构化中间表示，适合低成本召回，不适合表达细粒度视觉细节。
- 一旦建图阶段漏检，后续纯场景图推理几乎无法恢复。
- 因此 Stage 1 应负责高召回视觉证据检索，Stage 2 应负责基于原始像素做验证、修正和下游任务推理。

当前推荐论文叙事：

`Task-conditioned keyframe retrieval + agentic visual evidence reasoning`

而不是：

`Scene graph + VLM pipeline`

## 2. 到目前为止已经完成的工作

### 2.1 Stage 1 分析与文档清理

已经重新梳理了当前 query scene 主线，并同步更新：

- `AGENTS.md`
- `memory/project_context.md`
- `memory/query_scene_knowledge.md`
- `memory/bash_scripts_index.md`
- `memory/room0_artifact_lineage.md`

关键结论：

- 当前 Stage 1 主入口是 `KeyframeSelector.select_keyframes_v2()`
- parser 的 canonical output 是 `HypothesisOutputV1`
- 真正执行链是 `parse_query_hypotheses -> execute_hypotheses -> select_keyframes_v2`
- `direct -> proxy -> context` 是 hypothesis fallback，不是最终选帧 fallback
- parser 使用的是 mesh-only、无 label 的 BEV，而不是带 object labels 的 BEV
- `SimpleQueryParser` 已过时，不应再作为当前回归入口

### 2.2 新建 Stage 2 package

Stage 2 已从 `conceptgraph/query_scene/` 中拆到同级目录：

- `conceptgraph/agents/__init__.py`
- `conceptgraph/agents/models.py`
- `conceptgraph/agents/adapters.py`
- `conceptgraph/agents/stage2_deep_agent.py`
- `conceptgraph/agents/tests/test_stage2_deep_agent.py`

当前代码边界：

- `conceptgraph/query_scene/` 只承载 Stage 1
- `conceptgraph/agents/` 承载 Stage 2 runtime、schema、adapter 和 tests

### 2.3 Stage 2 协议和 runtime

已经固定了 Stage 2 的基本协议：

- 输入：`Stage2TaskSpec + Stage2EvidenceBundle`
- 输出：`Stage2StructuredResponse`
- 结果封装：`Stage2AgentResult`

支持的任务类型：

- `qa`
- `visual_grounding`
- `nav_plan`
- `manipulation`
- `general`

planning 模式：

- `off`
- `brief`
- `full`

当前工具主线：

- `inspect_stage1_metadata`
- `retrieve_object_context`
- `request_more_views`
- `request_crops`
- `switch_or_expand_hypothesis`

当前 runtime 选型已经定死为：

- `LangChain v1 + DeepAgents`

不是自定义手写 agent loop。

### 2.4 Stage 1 -> Stage 2 adapter

已经完成 `KeyframeResult -> Stage2EvidenceBundle` 适配：

- 从 Stage 1 metadata 中抽取 selected hypothesis
- 提取 target / anchor categories
- 把 `keyframe_paths + frame_mappings + selection_scores` 转成 `KeyframeEvidence`
- 构造 `object_context`

入口在：

- `conceptgraph/agents/adapters.py`

### 2.5 模型接入演进

#### 已做过的 Gemini 方案

最初 Stage 2 采用单 key AzureOpenAI-compatible Gemini client，不走 `GeminiClientPool`。这是为了：

- 避免 pool fan-out
- 保持稳定 `session_id`
- 利用 provider-side prompt caching

中间还加了一层兼容 shim，把 `tool_choice in ("any", "required", True)` 降成 `auto`，以规避 Gemini 对 required tool mode 的不兼容。

后来这层被泛化成：

- `ToolChoiceCompatibleAzureChatOpenAI`

#### Gemini 实测结论

Gemini 的 plain call 是通的：

- raw `AzureOpenAI` 文本请求可用
- `AzureChatOpenAI` 文本请求可用

但 `DeepAgents + Gemini` 不稳定，主要问题：

1. `function_calling_config.mode = "required"` 报错
2. 即使改成 `auto`，仍然会触发 `-4333 gemini模型fc报错`

结论：

- Gemini 不是不能聊天
- 但当前 `DeepAgents + tools + response_format + multimodal` 组合下的 FC 路径不稳定
- 不适合作为当前默认调试 backend

#### GPT 5.2 实测结论

已经验证 `gpt-5.2-2025-12-11` 可用：

- 纯文本直连可用，返回 `hello`
- 多模态直连可用，对红色方块图返回 `Bright red square`
- `DeepAgents + GPT 5.2` 不会像 Gemini 一样直接 FC 报错

因此，当前默认 backend 已切到：

- `gpt-5.2-2025-12-11`

对应当前未提交配置：

- base url: `https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl`
- api key: `Eyt11Oeoj77MfGcMweDRODBsbYnPkWUp`
- api version: `2024-03-01-preview`

并且当前默认：

- `include_thoughts=False`

原因是当前默认调试模型是 GPT 5.2，不需要默认带 provider-specific thinking payload；如果需要，可以显式打开。

### 2.6 已完成的文档

已经新增并更新：

- `docs/stage2_vlm_agent_design.md`
- `memory/research_direction.md`

本文件是第二份 handoff 文档，用于沉淀“实现到哪一步了”和“还差什么”。

## 3. 已解决的问题

### 3.1 研究叙事已经收敛

已经从“场景图 + VLM”的模糊说法收敛成：

- Stage 1 是 high-recall evidence retriever
- Stage 2 是 evidence-grounded task agent

这解决了研究表述容易工程化的问题。

### 3.2 Stage 2 代码边界已经拉清

Stage 2 已经不再混在 `query_scene/` 里。后续开发可以直接围绕 `conceptgraph/agents/` 展开，而不用担心 Stage 1 / Stage 2 边界继续混乱。

### 3.3 统一协议已建立

当前 Stage 2 不再依赖散乱 prompt，而是有稳定协议：

- `Stage2TaskSpec`
- `Stage2EvidenceBundle`
- `Stage2StructuredResponse`

这解决了后续：

- task 扩展
- evaluation
- logging / trace
- ablation

容易失控的问题。

### 3.4 Gemini 是否可用已经有实测答案

已经不用再猜：

- plain Gemini 可用
- current DeepAgents agent on Gemini 不稳定

因此当前默认模型切换为 GPT 5.2 是基于 live probe，不是拍脑袋。

### 3.5 GPT 5.2 作为默认 backend 已验证可达

已经证明：

- 文本可用
- 图像可用
- agent 不会直接因为 FC 协议而炸掉

这解决了“Stage 2 现在到底有没有一个能跑的默认模型”这个问题。

## 4. 当前还残留的关键问题

下面这些是当前真正的阻塞点。

### ~~4.1 工具拿到的新证据没有重新注入后续多模态上下文~~ ✅ 已解决

**已在 2026-03-20 解决。**

实现方案：
- `_Stage2RuntimeState` 新增 `evidence_updated` flag 和 `seen_image_paths` 集合
- 三个证据工具 (`request_more_views` / `request_crops` / `switch_or_expand_hypothesis`) 在 callback 返回 `updated_bundle` 时调用 `mark_evidence_updated()`
- `run()` 方法改为 iterative loop，每轮检查 `consume_evidence_update()`
- 若有新证据，调用 `_build_evidence_update_message()` 构造增量图片消息
- 新图片注入后继续下一轮推理，直到完成或达到 `max_reasoning_turns`

关键改动：
- `stage2_deep_agent.py:44-65` - `_Stage2RuntimeState` 扩展
- `stage2_deep_agent.py:494-549` - `_build_evidence_update_message()` 新方法
- `stage2_deep_agent.py:575-635` - iterative `run()` 实现
- 新增 3 个单测覆盖 evidence flag / iterative loop / callback update

### 4.2 三个关键工具还没有接真实 backend

当前：

- `request_more_views`
- `request_crops`
- `switch_or_expand_hypothesis`

在没传 callback 时都只是 stub，会返回：

- `callback is not configured`

所以现在的 Stage 2 还没有真正串回：

- Stage 1 more-view retrieval
- object crop generation
- hypothesis re-query / hypothesis expansion

### 4.3 prompt / policy 过度鼓励工具调用

**部分缓解 (2026-03-20)**：已在 system prompt 中加入 "CRITICAL - Look before requesting" 规则，强调必须先检查现有图片再决定是否调用工具。

但这仍然是软约束。用 GPT 5.2 跑过一个最小 smoke test：

- 输入：一张 64x64 红色方块图
- 任务：`What color is the square in the image?`

plain multimodal GPT 5.2：

- 能正确答 `Bright red square`

当前 DeepAgents agent：

- 返回 `insufficient_evidence`
- 调用了 `inspect_stage1_metadata`
- 然后又调用 `request_more_views`
- 即使图片其实已经在上下文里

这说明：

- 当前主要瓶颈已经不是模型可用性
- 而是 agent policy 没有强约束“先看已有图片，再决定要不要补证据”

### 4.4 `plan_mode` 和 `max_reasoning_turns` 还没有真正变成 runtime 约束

**`max_reasoning_turns` 已实现 (2026-03-20)**: iterative `run()` loop 现在真正消费这个参数。

当前：

- `plan_mode` 主要体现在 prompt 文案上
- `FULL` 模式会挂 subagents
- ~~`max_reasoning_turns` 只在 schema 中定义，但没有被执行层消费~~ ✅ 已实现

因此现在还不能严格声称：

- 有可控的 reasoning budget
- 有真正的 turn limit
- 有计划模式对应的执行差异

### 4.5 response_format 路径依赖较重

当前 `create_deep_agent(...)` 里用了：

- `response_format=Stage2StructuredResponse`

这条路径的好处是输出统一；问题是：

- 它进一步加重了 agent 对 tool-calling / structured-output runtime 的依赖
- 在 Gemini 上已经证明这条路径不稳
- 在 GPT 5.2 上虽然能跑，但也增加了复杂性和 latency

这不是必须立即删除，但需要记住它是当前架构复杂度的重要来源。

## 5. 当前架构 review

### 5.1 优点

- 分层清楚：Stage 1 / Stage 2 边界已经明确
- schema 清楚：输入 / 输出 / evidence bundle 已统一
- framework choice 清楚：`LangChain v1 + DeepAgents`
- tool surface 清楚：已经有最小工具集合
- default backend 清楚：当前默认 GPT 5.2
- 可测试性尚可：已有最小单测和 live smoke tests

### 5.2 当前最真实的评价

当前 Stage 2 已经不是空白设计稿了，**现在是一个真正闭环的 evidence-seeking agent skeleton**。

更准确的状态是：

- 已有明确的 runtime skeleton
- 已有协议、工具接口、adapter、tests
- 已有默认 backend
- **”补证据 -> 回流新视觉证据 -> 再推理”的闭环已打通**

还差：

- 真实的 more-view / crop / hypothesis backend 实现
- 更强的 “look before requesting” 约束
- 完整的 planning mode 行为分化

## 6. 当前仓库状态

### 6.1 已提交的基线

最近一个与 Stage 2 直接相关的提交是：

- `117960a feat(agent): add deepagents-based stage2 research runtime`

这个提交包含：

- `conceptgraph/agents/` 的初始版本
- `docs/stage2_vlm_agent_design.md`
- `memory/research_direction.md`
- 对 `AGENTS.md` / memory 的基础更新

### 6.2 当前工作树中尚未提交的修改

当前未提交修改包括：

- `conceptgraph/agents/models.py`
- `conceptgraph/agents/stage2_deep_agent.py`
- `conceptgraph/agents/tests/test_stage2_deep_agent.py`
- `AGENTS.md`
- `memory/project_context.md`
- `memory/query_scene_knowledge.md`
- `memory/research_direction.md`
- `docs/stage2_vlm_agent_design.md`

这些修改主要做了两件事：

1. 把 Stage 2 默认 backend 从 Gemini 切到 GPT 5.2
2. 把文档叙事从“默认 Gemini”改成“默认 GPT 5.2，Gemini 可选 override”

本文件也是当前未提交改动的一部分。

## 7. 已跑过的验证

### 7.1 Stage 1 回归

跑过：

```bash
.venv/bin/python -m pytest \
  conceptgraph/query_scene/tests/test_keyframe_selector_hypothesis.py \
  conceptgraph/query_scene/tests/test_query_parser_hypothesis.py \
  conceptgraph/query_scene/tests/test_hypothesis_output_schema.py \
  conceptgraph/query_scene/tests/test_open_world_sample_builder.py -q
```

结果：

- `20 passed`

### 7.2 Stage 2 单测

跑过：

```bash
.venv/bin/python -m pytest conceptgraph/agents/tests/test_stage2_deep_agent.py -q
```

历史结果：

- `4 passed`
- `5 passed`
- `6 passed`
- `7 passed`
- 当前最新是 `10 passed`

新增测试覆盖了：

- tool-choice compatibility shim
- Azure chat client 初始化参数
- `session_id` / `extra_body` 注入
- `include_thoughts=False` 时不注入 `thinking`
- adapter 提取 hypothesis/context
- runtime tools 的 callback update
- `build_agent()` 是否正确挂 `response_format` 和 subagents
- **evidence update flag 和 tracking**
- **iterative run 尊重 `max_reasoning_turns`**
- **callback 返回 bundle 时 mark evidence updated**

### 7.3 live probe

已做过下面这些 live probe：

#### Gemini

- raw `AzureOpenAI` 文本请求：通
- `AzureChatOpenAI` 文本请求：通
- `DeepAgents + Gemini`：不稳定，出现 FC 相关报错

#### GPT 5.2

- raw `AzureChatOpenAI` 文本请求：通，返回 `hello`
- `AzureChatOpenAI` 多模态请求：通，红色方块图返回 `Bright red square`
- `DeepAgents + GPT 5.2`：能跑，但当前 policy 仍可能错误请求更多视图

## 8. 推荐的下一步

建议按下面优先级推进。

### ~~P0: 打通 evidence refinement 的闭环~~ ✅ 已完成

目标：

- 工具返回新 bundle 后，新增图片必须重新注入模型上下文

**已实现** (2026-03-20)：
- iterative `run()` loop 在每个 turn 后检查 `evidence_updated` flag
- 若有新图片，构造 `HumanMessage` 将增量图片注入
- 使用 `seen_image_paths` 避免重复注入相同图片
- 尊重 `max_reasoning_turns` 作为 budget

### P1: 先看图，再决定是否调用工具

目标：

- 避免”图明明已经给了，还先去 request_more_views”

可行方向：

- ~~修改 system prompt / user prompt~~ ✅ 已加入 “Look before requesting” 规则
- 增加强规则：已有 keyframes 时必须先做视觉检查，再判断是否 evidence-limited
- 如果需要，可额外加一层 middleware / post-check，抑制无意义的 more-view 请求

### P2: 接真实 backend

把下面三个 callback 接上真正实现：

- `request_more_views`
- `request_crops`
- `switch_or_expand_hypothesis`

建议优先顺序：

1. more views
2. crops
3. hypothesis repair

### P3: 把 budget / plan 变成真实执行约束

当前最好补的包括：

- ~~消费 `max_reasoning_turns`~~ ✅ 已实现
- 让 `off / brief / full` 真正影响 agent loop，而不是只改 prompt
- 给 `FULL` 模式加更可控的 subagent usage policy

### P4: 决定是否继续坚持 DeepAgents 的 `response_format` 路线

这不是最急的，但需要尽快做技术判断：

- 如果未来仍要兼顾 Gemini，可能要降低 runtime 复杂度
- 如果短期只以 GPT 5.2 为默认调试模型，可以先保留现有 `response_format`

## 9. 下一个 agent 建议先读哪些文件

如果下一位 agent 要继续做 Stage 2，建议按这个顺序读：

1. `AGENTS.md`
2. `memory/project_context.md`
3. `memory/room0_artifact_lineage.md`
4. `memory/query_scene_knowledge.md`
5. `memory/bash_scripts_index.md`
6. `memory/research_direction.md`
7. `docs/stage2_vlm_agent_design.md`
8. `docs/stage2_agent_handoff.md`
9. `conceptgraph/agents/models.py`
10. `conceptgraph/agents/adapters.py`
11. `conceptgraph/agents/stage2_deep_agent.py`
12. `conceptgraph/agents/tests/test_stage2_deep_agent.py`

## 10. 一句话状态总结

截至 2026-03-20，Stage 2 已经完成了：

- 清晰的研究 framing
- 独立 package
- 统一 schema
- DeepAgents runtime skeleton
- Stage 1 adapter
- GPT 5.2 默认 backend 切换
- 单测和 live probe
- **视觉证据回流闭环** (iterative evidence refinement)
- **`max_reasoning_turns` budget 约束**
- **”Look before requesting” policy**

但还没有完成：

- ~~真正的视觉证据回流闭环~~ ✅
- 真实 more-view / crop / hypothesis repair backend
- ~~受控的 reasoning budget~~ ✅ (`max_reasoning_turns`)
- “先看现有图再决定是否补证据”的稳定 agent policy (已加 prompt 规则，但仍是软约束)
