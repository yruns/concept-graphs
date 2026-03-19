# Repository Guidelines

## Memory Bootstrap (Required)
- 本仓库的长期背景知识存放在 `memory/`。
- 每次新对话/新任务开始时，必须先按顺序读取：
1. `memory/project_context.md`
2. `memory/room0_artifact_lineage.md`
3. `memory/query_scene_knowledge.md`
4. `memory/bash_scripts_index.md`
5. `memory/research_direction.md`
- 这些文件用于快速拉取项目背景；若与当前源码冲突，以源码为准。
- 当你修改了 query scene 行为或 pipeline 脚本后，必须同步更新对应 `memory/*.md`。
- 当你修改了两阶段研究框架、Stage 2 Agent 协议或研究叙事时，必须同步更新 `memory/research_direction.md` 与本文件。

## Memory References
- 索引入口：`memory/README.md`
- 项目背景：`memory/project_context.md`
- `room0` 产物溯源：`memory/room0_artifact_lineage.md`
- Query Scene 知识：`memory/query_scene_knowledge.md`
- Bash 脚本索引：`memory/bash_scripts_index.md`
- 研究方向：`memory/research_direction.md`

## Fast Commands
- 下列命令中的 `python`：Darwin 请替换为 `.venv/bin/python`；Linux 请先激活 `conceptgraph` conda 环境。
- Query Scene 单元回归：
  - `python -m pytest conceptgraph/query_scene/tests/test_keyframe_selector_hypothesis.py conceptgraph/query_scene/tests/test_query_parser_hypothesis.py conceptgraph/query_scene/tests/test_hypothesis_output_schema.py conceptgraph/query_scene/tests/test_open_world_sample_builder.py -q`
- Stage 2 Agent 单测：
  - `python -m pytest conceptgraph/agents/tests/test_stage2_deep_agent.py -q`
- Query Scene 单条查询：
  - `REPLICA_ROOT=/abs/path/to/Replica python -m conceptgraph.query_scene.examples.query_keyframes --scene_path "$REPLICA_ROOT/room0" --query "pillow on the sofa" --k 3 --llm_model gpt-5.2-2025-12-11`
- Query Scene 端到端：
  - `REPLICA_ROOT=/abs/path/to/Replica SCENE_NAME=room0 python -m conceptgraph.query_scene.examples.e2e_query_test`
- 查询关键帧预处理：
  - `REPLICA_ROOT=/abs/path/to/Replica bash bashes/6b_build_visibility_index.sh room0`
  - `REPLICA_ROOT=/abs/path/to/Replica bash bashes/7b_query_scene.sh room0 "pillow on the sofa" 3`

## Python Environment (Required)
- **检测系统**：先执行 `uname -s` 判断当前 OS
- **macOS (Darwin)**：使用项目根目录的 `.venv`
  ```bash
  .venv/bin/python -m conceptgraph.query_scene.examples.e2e_query_test
  ```
- **Linux**：使用 conda 环境 `conceptgraph`
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
  python -m conceptgraph.query_scene.examples.e2e_query_test
  ```

## Query Scene Notes
- 当前研究叙事采用两阶段范式：
  - Stage 1：解析用户任务 query，并在场景中检索任务最相关关键帧
  - Stage 2：把关键帧送入 VLM Agent 做下游任务推理
- Stage 1 的核心目标是高召回视觉证据检索；Stage 2 的核心目标是利用原始视觉证据弥补传统场景图对细粒度细节和漏检的无能为力。
- Stage 2 不应退化成一次性多图问答；默认研究方向是 ReAct 风格、evidence-seeking、uncertainty-aware 的 Agent。
- Stage 2 的正式代码位于 `conceptgraph/agents/`，与 `conceptgraph/query_scene/` 同级分离：
  - `conceptgraph/query_scene/` 只承载 Stage 1 query parsing / retrieval
  - `conceptgraph/agents/` 承载 Stage 2 LangChain v1 + DeepAgents runtime、schema、adapter、tests
- Stage 2 当前的 canonical framework choice 是 `LangChain v1 + DeepAgents`，而不是自定义 agent loop。
- Stage 2 当前的 Gemini 接入方式是单 key 的 AzureOpenAI-compatible `AzureChatOpenAI` client，不走 `GeminiClientPool`。
- Stage 2 默认使用项目内的办公网 base url：`https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl`；不要默认切到生产网关地址。
- Stage 2 初始化模型时必须带稳定的 `extra_body.session_id`，并默认开启 `extra_body.thinking.include_thoughts=True`，以便启用 provider-side prompt caching。
- Stage 2 支持 `qa / visual_grounding / nav_plan / manipulation` 四类任务，使用统一 structured response envelope，并通过 `plan_mode=off|brief|full` 控制 planning 强度。
- Stage 2 工具主线是 `inspect_stage1_metadata / retrieve_object_context / request_more_views / request_crops / switch_or_expand_hypothesis`。
- 完整设计说明在 `docs/stage2_vlm_agent_design.md`。
- 当前 query parser 的对外协议是 `HypothesisOutputV1`，主入口是 `KeyframeSelector.select_keyframes_v2()`；返回 metadata 中版本号仍写作 `v3`。
- `KeyframeSelector.parse_query_hypotheses()` 默认会生成 `scene_path/bev/scene_bev_<hash>.png` 作为多模态输入；该图使用 `ReplicaDefaultBEVConfig`，是 mesh-only、无 object marker/label 的 BEV。
- `conceptgraph/query_scene/examples/simple_parse_test.py` 以及 `conceptgraph/query_scene/examples/test_nested_query_parsing.py` 中的 `SimpleQueryParser` 分支已过时；当前源码里不存在 `SimpleQueryParser`，不要把它们当作冒烟命令。
- `bashes/6b_build_visibility_index.sh`、`bashes/7b_query_scene.sh`、`bashes/run_e2e_query_test.sh` 目前仍是 Linux/conda 风格脚本，没有统一适配 `.venv`。
- `bashes/7b_query_scene.sh` 的默认 `REPLICA_ROOT` 是 `$HOME/Datasets/Replica/Replica`，与 `6b`/`run_full_detect_pipeline_to_6b.sh` 的 `$HOME/Datasets/Replica` 不一致；运行 query scene bash 脚本前请显式导出 `REPLICA_ROOT`。

## Working Conventions
- Python 4 空格缩进，命名采用 `snake_case` / `PascalCase`。
- 运行配置统一走 `env_vars.bash`，避免硬编码路径。
- 提交优先按子模块拆分：`query_scene`、`slam`、`scenegraph`、`bashes`。
