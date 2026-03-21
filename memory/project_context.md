# Project Context

## Repository 结构（主干）
- `conceptgraph/`: 主代码。
- `bashes/`: 端到端流程脚本（分阶段）。
- `data/benchmark/`: 多 benchmark 注释、公开下载资产与 scene manifest（2026-03-20 起）。
- `docs/`: 说明文档与执行记录。
- `Grounded-Segment-Anything/`, `gradslam/`, `chamferdist/`: 依赖代码树（修改需谨慎）。

## 环境与配置
- Python 入口按 OS 区分：macOS (Darwin) 使用项目根目录 `.venv`，Linux 使用 conda 环境 `conceptgraph`。
- 本地路径/密钥通过 `env_vars.bash` 管理（由 `env_vars.bash.template` 复制）。
- 常见变量：`REPLICA_ROOT`, `GSA_PATH`, `LLM_BASE_URL`, `LLM_MODEL`。
- 多 benchmark / ScanNet 新增常见变量：
  - `BENCHMARK_ROOT`：默认 `data/benchmark`
  - `SCANNET_RAW_ROOT`：官方 ScanNet `scans/` + `scans_test/` 根目录
  - `SCANNET_SCENE_ROOT`：包装成 Replica-like 目录后的 ScanNet scene 根目录
  - `SCANNET_CONFIG_PATH`：默认 `conceptgraph/dataset/dataconfigs/scannet/base.yaml`
  - `SCANNET_DOWNLOAD_SCRIPT`：默认可指向仓库内置兼容脚本 `tools/scannet/download-scannet.py`；若已有官方授权版也可覆盖
  - `OPENEQA_SCANNET_FRAMES_ROOT`：OpenEQA ScanNet clip 帧目录，默认 `$HOME/Datasets/open-eqa/data/frames/scannet-v0`
  - `OPENEQA_SCANNET_ROOT`：OpenEQA ScanNet clip 场景图根目录，默认 `$HOME/Datasets/OpenEQA/scannet`
  - `OPENEQA_SCANNET_CONFIG_PATH`：默认 `conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml`
- query scene 相关 bash 包装脚本目前并不完全统一：
  - `bashes/run_full_detect_pipeline_to_6b.sh` 会 source `env_vars.bash`。
  - `bashes/6b_build_visibility_index.sh`、`bashes/7b_query_scene.sh`、`bashes/run_e2e_query_test.sh` 仍直接使用 conda/裸 `python`，在 Darwin 上更稳妥的方式是直接用 `.venv/bin/python -m ...` 运行模块。
- 2026-03-20 新增多 benchmark / ScanNet bash 子目录：
  - `bashes/benchmark_data/`: 下载公开 benchmark 注释与构建 scene manifest
  - `bashes/scannet_benchmark/`: ScanNet raw subset 下载、Replica-like wrapper、labeled full pipeline 到 6B
- 2026-03-22 新增 OpenEQA ScanNet bash 子目录：
  - `bashes/openeqa-scannet/`: 直接面向 OpenEQA ScanNet clip 的 scene graph pipeline
  - 输出布局采用 `OpenEQA/scannet/<clip_id>/conceptgraph/`
  - `conceptgraph/` 子目录中保留软链接输入（`*-rgb.png`, `*-depth.png`, pose/intrinsic/extrinsic`）；`mesh.ply` 仅在可用时附带
  - 场景图产物也全部写入这个 `conceptgraph/` 子目录，避免和 clip 原始帧目录混淆
- `bashes/7b_query_scene.sh` 的默认 `REPLICA_ROOT` 为 `$HOME/Datasets/Replica/Replica`，与 `6b`/`run_full_detect_pipeline_to_6b.sh` 使用的 `$HOME/Datasets/Replica` 不一致，建议显式设置。
- 路径策略（2026-03-08 起）：`2b/6b` 产物禁止写入绝对路径；`pcd_saves/*.pkl.gz` 与 `indices/visibility_index.pkl` 中路径字段统一使用相对 `REPLICA_ROOT` 的相对路径。
- ScanNet wrapper 路径策略（2026-03-20 起）：
  - scene 根目录同时保留 `color/ depth/ pose/ intrinsic/` 供 `ScannetDataset` 读取
  - 同时生成 Replica-compatible `results/frame*.jpg`、`results/depth*.png` 和 `traj.txt`
  - mesh 额外放在 `SCANNET_SCENE_ROOT/<scene_id>_mesh.ply`，便于 BEV / offscreen 可视化复用现有约定
  - 2026-03-20 后续修正：wrapper 默认按 `SCANNET_FRAME_SKIP=5` 从 raw `.sens` 抽帧；下游 `1b/2b/6b` 默认在 wrapper scene 上使用 `SCANNET_PROCESS_STRIDE=1`，从而保持“相对原始 ScanNet 每 5 帧处理一次”的有效间隔
  - `gsa_detections_ram_withbg_allclasses/*.pkl.gz` 在 `2b/6b` 之后可安全瘦身：默认仅保留 `xyxy/confidence/class_id/classes/frame_clip_feat/tagging_caption/tagging_text_prompt`，删除 `mask/image_crops/image_feats/text_feats`
- OpenEQA ScanNet clip 路径策略（2026-03-22 起）：
  - 复用新的 `scannet-openeqa` dataset loader，直接读取 root-level `*-rgb.png`, `*-depth.png`, `[0-9]*.txt`
  - `1_prepare_clip_scene.sh` 会在 `conceptgraph/` 目录内软链接这些输入，并额外生成 `traj.txt`
  - OpenEQA ScanNet clip 建图本身只依赖 `rgb/depth/pose/intrinsic`；`mesh.ply` 不是 `1b/2b/6b` 的必需输入，只在离线预览时作为可选 overlay 使用
  - OpenEQA ScanNet 当前默认 `OPENEQA_PROCESS_STRIDE=2`，即 clip 序列按隔 1 帧采样处理，以降低 `1b/2b` 开销
  - `2b/6b` 直接以 `clip_id/conceptgraph` 作为 `scene_id` 运行，不再走 `ScanNetReplicaLike`

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
