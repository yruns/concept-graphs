# Project Context

## Repository 结构（主干）
- `conceptgraph/`: 主代码。
- `bashes/`: 端到端流程脚本（分阶段）。
- `docs/`: 说明文档与执行记录。
- `Grounded-Segment-Anything/`, `gradslam/`, `chamferdist/`: 依赖代码树（修改需谨慎）。

## 环境与配置
- Python 入口按 OS 区分：macOS (Darwin) 使用项目根目录 `.venv`，Linux 使用 conda 环境 `conceptgraph`。
- 本地路径/密钥通过 `env_vars.bash` 管理（由 `env_vars.bash.template` 复制）。
- 常见变量：`REPLICA_ROOT`, `GSA_PATH`, `LLM_BASE_URL`, `LLM_MODEL`。
- query scene 相关 bash 包装脚本目前并不完全统一：
  - `bashes/run_full_detect_pipeline_to_6b.sh` 会 source `env_vars.bash`。
  - `bashes/6b_build_visibility_index.sh`、`bashes/7b_query_scene.sh`、`bashes/run_e2e_query_test.sh` 仍直接使用 conda/裸 `python`，在 Darwin 上更稳妥的方式是直接用 `.venv/bin/python -m ...` 运行模块。
- `bashes/7b_query_scene.sh` 的默认 `REPLICA_ROOT` 为 `$HOME/Datasets/Replica/Replica`，与 `6b`/`run_full_detect_pipeline_to_6b.sh` 使用的 `$HOME/Datasets/Replica` 不一致，建议显式设置。
- 路径策略（2026-03-08 起）：`2b/6b` 产物禁止写入绝对路径；`pcd_saves/*.pkl.gz` 与 `indices/visibility_index.pkl` 中路径字段统一使用相对 `REPLICA_ROOT` 的相对路径。

## 贡献约定（简）
- 命名：`snake_case`（函数/文件）、`PascalCase`（类）。
- 代码风格：4 空格缩进，新增公共函数尽量加类型标注。
- 提交建议按模块拆分：`query_scene`、`slam`、`scenegraph`、`bashes`。

## 当前研究方向
- 当前 query scene 的研究主线采用两阶段范式：
  - Stage 1：query parsing + task-conditioned keyframe retrieval
  - Stage 2：基于关键帧的 VLM agent 推理
- Stage 1 的目标是把整场景压缩成任务相关视觉证据，而不是直接输出最终答案。
- Stage 2 的目标是利用原始关键帧补足传统场景图的细粒度缺失与漏检问题，支持 QA / visual grounding / nav plan / manipulation。
- Stage 2 的实现框架已明确为 `LangChain v1 + DeepAgents`，并从 `conceptgraph/query_scene/` 中拆出到同级目录 `conceptgraph/agents/`。
- 当前 Stage 2 运行形态是：
  - 统一任务输入：`Stage2TaskSpec + Stage2EvidenceBundle`
  - 统一任务输出：`Stage2StructuredResponse`
  - planning 强度由 `plan_mode=off|brief|full` 控制
  - 工具层显式支持补证据与 hypothesis repair，而不是一次性多图问答
  - 默认模型切到 `gpt-5.2-2025-12-11`；Gemini 保留为可选 override
  - 模型接入使用单 key 的 AzureOpenAI-compatible client，并通过 `extra_body.session_id` 打开 prompt caching
- 该方向的长期说明记录在 `memory/research_direction.md`，后续做第二阶段相关工作时应先读取。
