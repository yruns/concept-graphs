# Query Scene Knowledge

## 当前主线（2026-03-19）
- `KeyframeSelector.select_keyframes_v2()` 是当前 query scene 主入口；虽然函数名保留 `v2`，但日志与返回 metadata 已统一标记为 `v3`。
- parser 的 canonical structured output 是 `HypothesisOutputV1`，不再把 `GroundingQuery` 作为对外主协议。
- 当前整体研究叙事应区分两阶段：
  - Stage 1：`KeyframeSelector` 负责 query-conditioned keyframe retrieval
  - Stage 2：VLM research agent 负责基于关键帧做下游任务推理
- 目前正式跑通并稳定回归的是 Stage 1。
- Stage 2 已从 `conceptgraph/query_scene/` 中拆出到同级目录 `conceptgraph/agents/`；`query_scene` 只保留 Stage 1 主链路与 handoff metadata。
- 2026-03-19 本地验证通过：
  - `.venv/bin/python -m pytest conceptgraph/query_scene/tests/test_keyframe_selector_hypothesis.py conceptgraph/query_scene/tests/test_query_parser_hypothesis.py conceptgraph/query_scene/tests/test_hypothesis_output_schema.py conceptgraph/query_scene/tests/test_open_world_sample_builder.py -q`
  - 结果：`20 passed in 1.61s`

## 核心模块
- `conceptgraph/query_scene/query_structures.py`
  - 定义嵌套查询结构：`QueryNode`、`SpatialConstraint`、`SelectConstraint`、`GroundingQuery`。
  - 定义统一输出：`HypothesisOutputV1`、`QueryHypothesis`、`HypothesisKind`、`ParseMode`。
  - `HypothesisOutputV1` 自带约束：
    - `single` 只能有一个 `direct`
    - `rank` 必须从 1 开始连续且唯一
    - `validate_categories()` / `validate_no_mask_leak()` 用于执行前校验
- `conceptgraph/query_scene/query_parser.py`
  - `QueryParser.parse()` 直接返回 `HypothesisOutputV1`。
  - 会根据 `scene_categories` 动态构造 schema，允许的类别集合是 `scene_categories + UNKNOW`。
  - 支持 `scene_images=[...]` 多模态输入；Gemini 走 JSON mode，非 Gemini 走 `with_structured_output()`。
  - `use_pool=True` 且模型名包含 `gemini` 时，会通过 `GeminiClientPool` 做多 key 轮转和 rate-limit 重试。
- `conceptgraph/query_scene/bev_builder.py`
  - 提供 `BaseBEVBuilder` / `ReplicaBEVBuilder` / `GenericBEVBuilder` 和工厂 `create_bev_builder()`。
  - `ReplicaBEVBuilder` 默认输出带 object marker + label，标签格式是 `NNN: category`。
  - 但 `KeyframeSelector` 使用的不是默认配置，而是 `ReplicaDefaultBEVConfig`：
    - `image_size=1000`
    - `perspective=True`
    - `show_objects=False`
    - `show_labels=False`
    - 即 parser 输入是 mesh-only 无标签 BEV
- `conceptgraph/query_scene/scene_visualizer.py`
  - 仅做向后兼容 re-export。
  - `SceneBEVGenerator = ReplicaBEVBuilder`。
- `conceptgraph/query_scene/query_executor.py`
  - 递归执行查询树：类别匹配 -> 属性过滤 -> 空间约束 -> 选择约束。
  - 类别匹配顺序：exact -> substring -> CLIP fallback（若 object/text feature 可用）。
  - 空间约束执行两阶段：
    - quick filter：仅用于 view-independent 关系
    - full spatial check：调用 `SpatialRelationChecker`
  - 默认 `strict_mode=False`；若 anchor 没解析出来，直接执行时会 lenient fallback 成“保留所有 candidates”。
- `conceptgraph/query_scene/quick_filters.py`
  - 只给 view-independent 关系做轻量预过滤：
    - vertical: `on/above/below`
    - distance: `near/next_to/beside`
  - `left_of/right_of/in_front_of/behind` 不走 quick filter。
- `conceptgraph/query_scene/spatial_relations.py`
  - 负责 `on/near/between/inside/...` 的几何关系判定和评分。
- `conceptgraph/query_scene/keyframe_selector.py`
  - 自动加载 scene objects、affordance、camera poses、sampled RGB paths、visibility index。
  - 主流程：
    - `parse_query_hypotheses()`
    - `execute_hypotheses()`
    - `select_keyframes_v2()`
  - `parse_query_hypotheses(..., use_visual_context=True)` 默认启用 BEV，并把结果 sanitize 成“仅 scene categories 或 `UNKNOW`”。
  - `normalize_hypothesis_output()` 兼容 legacy payload：`HypothesisOutputV1` / `GroundingQuery` / legacy dict。
  - `execute_hypotheses()` 会按 rank 执行，但如果某个 hypothesis 的 anchor/reference 里仍含 `UNKNOW`，会直接跳过，不让 executor 的 lenient fallback 误命中。
  - 状态值固定为：
    - `direct_grounded`
    - `proxy_grounded`
    - `context_only`
    - `no_evidence`
  - 选帧仍使用 `joint_coverage` 贪心策略，并输出 `frame_mappings`：
    - `requested_view_id/requested_frame_id`
    - `resolved_view_id/resolved_frame_id`
    - `path`
- `conceptgraph/agents/`
  - Stage 2 canonical package，与 `query_scene` 同级。
  - 当前实现框架是 `LangChain v1 + DeepAgents`，不是自定义 agent loop。
  - 默认模型当前切到 `gpt-5.2-2025-12-11`；Gemini 保留为可选 override，但当前 `DeepAgents + Gemini` 的 FC 路径不稳定。
  - 模型初始化当前使用单 key 的 AzureOpenAI-compatible `AzureChatOpenAI` client，不走 `GeminiClientPool`。
  - 默认 base url 应使用项目内办公网地址：`https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl`
  - 初始化 payload 会写入：
    - `extra_body.session_id`
    - `extra_body.thinking.include_thoughts=True`（仅在显式开启时）
    这样可以利用 provider-side prompt caching，并在需要时打开 provider-specific thinking。
  - 关键模块：
    - `models.py`：`Stage2TaskSpec / Stage2EvidenceBundle / Stage2StructuredResponse`
    - `adapters.py`：把 `KeyframeResult` 转成 Stage 2 evidence bundle
    - `stage2_deep_agent.py`：DeepAgents runtime、tool layer、plan mode、subagents
  - 当前工具主线：
    - `inspect_stage1_metadata`
    - `retrieve_object_context`
    - `request_more_views`
    - `request_crops`
    - `switch_or_expand_hypothesis`
  - 当前统一任务输出：
    - `status / summary / confidence / uncertainties / cited_frame_indices / evidence_items / plan / payload`
- `conceptgraph/query_scene/open_world_dataset.py`
  - 自动探测 `pcd_file` / `affordance_file`，构建 `scene_manifest.jsonl`。
  - 生成 deterministic `query_program_pool.jsonl`，包含 `simple` / `spatial` / `superlative` program。
  - 输出 manifest 中的路径是绝对路径字符串。
- `conceptgraph/query_scene/open_world_sample_builder.py`
  - 从 manifest + program pool 生成 `parser_sft.jsonl`。
  - 分桶固定为 `40/30/30`：`direct` / `soft` / `hard`。
  - `hard` 桶会把 target 类别 mask 成 `UNKNOW`，并通过 `validate_no_mask_leak()` 校验。
  - `TeacherQueryGenerator` 支持双教师 query 生成、cache、retry、`generation_report.md` 失败记录。

## 当前执行链路（代码行为）
1. `KeyframeSelector.from_scene_path()` 优先寻找 `pcd_saves/*ram*_post.pkl.gz`，其次 `*_post.pkl.gz`，最后 `*.pkl.gz`。
2. 若存在 `sg_cache_detect/object_affordances.json` 或 `sg_cache/object_affordances.json`，会把 `object_tag/summary/category/affordances` 合并进 `SceneObject`。
3. 若存在 `scene_path/indices/visibility_index.pkl`，直接加载；否则在线重建可见性索引。
4. `parse_query_hypotheses()` 默认生成并缓存 `scene_path/bev/scene_bev_<hash>.png`，将其作为 parser 的 multimodal context。
5. parser 输出 `HypothesisOutputV1` 后，`KeyframeSelector` 会把不在 scene 中的类别剔除；若节点类别全被剔空，则改成 `["UNKNOW"]`。
6. `execute_hypotheses()` 按 rank 顺序尝试 `direct -> proxy -> context`，但会先检查 category validity / hidden leak / `UNKNOW` anchors。
7. 选中 hypothesis 后，target objects 取最终执行结果，anchor objects 通过对 root anchors 单独执行 `_execute_node()` 收集。
8. `select_keyframes_v2()` 用 `joint_coverage` 选择视角，并在 `results/frame%06d.jpg` 缺失时做邻近视角回退。
9. 返回 `KeyframeResult.metadata` 时会附带完整 `hypothesis_output`、状态、选中的 hypothesis kind/rank 和 `version="v3"`。

## 当前有效命令
- 下面命令中的 `python`：Darwin 请替换为 `.venv/bin/python`；Linux 请先激活 `conceptgraph` conda 环境。
- Query scene 单元回归：
  - `python -m pytest conceptgraph/query_scene/tests/test_keyframe_selector_hypothesis.py conceptgraph/query_scene/tests/test_query_parser_hypothesis.py conceptgraph/query_scene/tests/test_hypothesis_output_schema.py conceptgraph/query_scene/tests/test_open_world_sample_builder.py -q`
- Stage 2 Agent 单测：
  - `python -m pytest conceptgraph/agents/tests/test_stage2_deep_agent.py -q`
- 单条查询：
  - `REPLICA_ROOT=/abs/path/to/Replica python -m conceptgraph.query_scene.examples.query_keyframes --scene_path "$REPLICA_ROOT/room0" --query "pillow on the sofa" --k 3 --llm_model gpt-5.2-2025-12-11`
- 端到端 query 可视化：
  - `REPLICA_ROOT=/abs/path/to/Replica SCENE_NAME=room0 python -m conceptgraph.query_scene.examples.e2e_query_test`
- 可见性索引：
  - `REPLICA_ROOT=/abs/path/to/Replica bash bashes/6b_build_visibility_index.sh room0`
- Open-world 资产构建：
  - `python conceptgraph/scripts/build_open_world_dataset_assets.py --scene room0=/abs/path/to/room0 --output_dir plans/generated_open_world`
  - `python conceptgraph/scripts/build_open_world_samples.py --scene_manifest plans/generated_open_world/scene_manifest.jsonl --query_program_pool plans/generated_open_world/query_program_pool.jsonl --output_dir plans/generated_open_world --samples_per_scene 300`
  - `python conceptgraph/scripts/build_open_world_samples.py --scene_manifest plans/generated_open_world/scene_manifest.jsonl --query_program_pool plans/generated_open_world/query_program_pool.jsonl --output_dir plans/generated_open_world_teacher --samples_per_scene 300 --use_teacher_llm --teacher_models gpt-5.2-2025-12-11,gemini-3-pro-preview-new --teacher_max_retries 2`

## 遗留/过时项
- `conceptgraph/query_scene/examples/simple_parse_test.py` 仍在 import 已不存在的 `SimpleQueryParser`，当前不能作为冒烟命令。
- `conceptgraph/query_scene/examples/test_nested_query_parsing.py` 里与 `SimpleQueryParser` 相关的路径同样过时；保留的价值主要是 LLM parse 示例，不适合作为“当前回归基线”。
- `bashes/7b_query_scene.sh` 的默认 `REPLICA_ROOT` 和 `6b` / `run_full_detect_pipeline_to_6b.sh` 不一致；建议显式设置。
- Stage 2 的正式 Agent 化推理入口还没有接成新的 bash 包装脚本；当前主入口仍是 Python 侧 `Stage2DeepResearchAgent.run(...)`。
