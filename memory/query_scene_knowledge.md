# Query Scene Knowledge

## 核心模块
- `conceptgraph/query_scene/query_structures.py`
  - 定义 `GroundingQuery`、`QueryNode`、`SpatialConstraint`、`SelectConstraint`。
  - 支持嵌套空间约束查询树。
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
  - 查询对象并做关键帧选择（`joint_coverage` 贪心覆盖）。

## 执行链路（代码行为）
1. 加载对象：优先 `scene_path/pcd_saves/*ram*_post.pkl.gz`，再 `*_post.pkl.gz`，再 `*.pkl.gz`。
2. 可选合并 affordance：`sg_cache_detect/object_affordances.json` 或 `sg_cache/object_affordances.json`。
3. 解析查询：优先 LLM（`QueryParser.parse`），失败重试后 fallback 到 `SimpleQueryParser`。
4. 执行查询：`QueryExecutor.execute` 递归处理节点。
5. 空间约束：先 quick filter 再 full check；quick filter 仅适用于视角无关关系（`on/above/below/near/next_to/beside`）。
6. 兜底：anchor 未命中时，默认 `strict_mode=False`，会返回原候选；`strict_mode=True` 才返回空。
7. 关键帧：依据 `object_to_views` / `view_to_objects` 选帧，默认联合覆盖多对象。

## 开发与回归命令
- `python -m conceptgraph.query_scene.examples.simple_parse_test`
- `python -m conceptgraph.query_scene.examples.test_nested_query_parsing --llm_model gpt-5.2-2025-12-11`
- `bash bashes/6b_build_visibility_index.sh room0`
- `bash bashes/7b_query_scene.sh room0 "pillow on the sofa" 3`
- `bash bashes/run_e2e_query_test.sh`
