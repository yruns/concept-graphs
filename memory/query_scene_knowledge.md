# Query Scene Knowledge

## 核心模块
- `conceptgraph/query_scene/query_structures.py`
  - 定义 `GroundingQuery`、`QueryNode`、`SpatialConstraint`、`SelectConstraint`。
  - 新增统一结构化输出：`HypothesisOutputV1`、`QueryHypothesis`、`HypothesisKind`、`ParseMode`。
  - `HypothesisOutputV1` 约束：`single` 模式只能有 1 个 `direct`；`multi` 支持 `direct/proxy/context` 按 `rank` 执行。
- `conceptgraph/query_scene/query_parser.py`
  - `QueryParser`: LLM 结构化解析。
  - `SimpleQueryParser`: 规则回退解析。
- `conceptgraph/query_scene/query_executor.py`
  - 递归执行查询树。
  - 执行顺序：类别匹配 -> 属性过滤 -> 空间约束 -> 选择约束。
- `conceptgraph/query_scene/spatial_relations.py`
  - 关系判定与评分（如 `on/near/between/inside`）。
- `conceptgraph/query_scene/keyframe_selector.py`
  - 加载场景对象、可见性索引、图像路径。
  - 新流程：`parse_query_hypotheses` -> `execute_hypotheses` -> `select_keyframes_v2`。
  - 关键帧策略仍为 `joint_coverage` 贪心覆盖，但输入改为 `HypothesisOutputV1`。
  - 支持 `normalize_hypothesis_output`（兼容 legacy payload）与 `to_grounding_query`（严格 `model_validate`）。
  - 帧路径解析支持邻近视角回退，输出 `requested_*` 与 `resolved_*` 映射。
- `conceptgraph/query_scene/open_world_dataset.py`
  - 生成 open-world 数据构建基础资产：`scene_manifest`、`query_program_pool`。
  - 提供类别抽取、program hash 去重、JSONL 写出工具。
- `conceptgraph/query_scene/open_world_sample_builder.py`
  - 从 `scene_manifest + query_program_pool` 组装 `parser_sft/retrieval_eval`。
  - 固定 40/30/30（direct/soft/hard）分桶采样，hard 桶执行掩蔽与泄漏校验。
  - 支持双教师 query 生成：`TeacherQueryGenerator`（模型缓存、重试、prompt版本追踪、失败报告）。

## 执行链路（代码行为）
1. 加载对象：优先 `scene_path/pcd_saves/*ram*_post.pkl.gz`，再 `*_post.pkl.gz`，再 `*.pkl.gz`。
2. 可选合并 affordance：`sg_cache_detect/object_affordances.json` 或 `sg_cache/object_affordances.json`。
3. 解析查询：`KeyframeSelector.parse_query_hypotheses` 将用户 query 解析为 `HypothesisOutputV1`。
4. 类别净化：执行前会把不在 `scene_categories` 的类别改为 `UNKNOW`，并可做 hidden category 泄漏检查。
5. 假设执行：`execute_hypotheses` 按 `rank` 依次执行 `direct` -> `proxy` -> `context`，返回第一个非空结果。
6. 执行前校验：`validate_categories_in_scene` + `validate_no_mask_leak`（hard-case 防泄漏）。
7. 状态输出：`direct_grounded` / `proxy_grounded` / `context_only` / `no_evidence`。
8. 关键帧：依据 `object_to_views` / `view_to_objects` 选帧，默认联合覆盖多对象，并输出 `requested/resolved view/frame` 映射。

## 开发与回归命令
- `python -m conceptgraph.query_scene.examples.simple_parse_test`
- `python -m conceptgraph.query_scene.examples.test_nested_query_parsing --llm_model gpt-5.2-2025-12-11`
- `bash bashes/6b_build_visibility_index.sh room0`
- `bash bashes/7b_query_scene.sh room0 "pillow on the sofa" 3`
- `bash bashes/run_e2e_query_test.sh`
- `python conceptgraph/scripts/build_open_world_dataset_assets.py --scene room0=/abs/path/to/room0 --output_dir plans/generated_open_world`
- `python conceptgraph/scripts/build_open_world_samples.py --scene_manifest plans/generated_open_world/scene_manifest.jsonl --query_program_pool plans/generated_open_world/query_program_pool.jsonl --output_dir plans/generated_open_world --samples_per_scene 300`
- `python conceptgraph/scripts/build_open_world_samples.py --scene_manifest plans/generated_open_world/scene_manifest.jsonl --query_program_pool plans/generated_open_world/query_program_pool.jsonl --output_dir plans/generated_open_world_teacher --samples_per_scene 300 --use_teacher_llm --teacher_models gpt-5.2-2025-12-11,gemini-3-pro-preview-new --teacher_max_retries 2`
