 # E2E Query Test 执行链路详解
 
 本文描述执行命令 `python conceptgraph/query_scene/examples/e2e_query_test.py` 的完整链路，
 包括入口、数据加载、解析与执行、可视化输出，以及所有关键分支与失败路径。
 
 ## 1. 入口与运行环境
 
 - 命令入口：`conceptgraph/query_scene/examples/e2e_query_test.py`
 - 项目路径解析：脚本使用 `project_root = Path(__file__).parent.parent.parent.parent`，
   再 `sys.path.insert(0, project_root)`，确保任意目录执行也能导入 `conceptgraph`。
 - 日志：`loguru` 被重新配置，只输出 INFO 及以上到 stderr。
 
 关键依赖：
 - Python 包：`numpy`, `loguru`, `langchain_openai`, `pydantic` 等。
 - 网络与 LLM：`QueryParser` 依赖 `conceptgraph.utils.llm_client.get_langchain_chat_model`，
   需要可访问的 Azure OpenAI 端点（参数在 `llm_client.py` 内硬编码）。
 
 ## 2. main() 总体执行流程
 
 `main()` 执行顺序：
 
 1. 固定场景路径：`scene_path = project_root / "room0"`。
 2. 输出目录：`output_dir = scene_path / "query_visualizations"`。
 3. 检查 `scene_path` 是否存在，不存在则直接返回。
 4. `load_scene_objects()` 读取 `room0/pcd_saves` 的 `*.pkl.gz` 并构建 `SceneObject` 列表。
 5. 从 `object_tag` 统计 `scene_categories` 供解析器使用。
 6. 遍历 `test_queries` 列表，逐条执行 `run_e2e_test()`。
 7. 汇总并输出结果日志。
 8. 保存 `test_results.json` 至 `room0/query_visualizations/`。
 
 ## 3. 数据加载链路（load_scene_objects）
 
 入口函数：`load_scene_objects(scene_path: str)`。
 
 - 文件选择逻辑（按顺序匹配，取第一个）：
   1. `pcd_saves/*ram_withbg*_post.pkl.gz`
   2. `pcd_saves/*_post.pkl.gz`
   3. `pcd_saves/*.pkl.gz`
 - 读取：`gzip.open` + `pickle.load`。
 - 数据结构适配：
   - 若 `data` 是 `dict` 且有 `objects` 字段，则使用 `data["objects"]`。
   - 否则视为直接对象列表。
 - 对每个对象：
   - 允许对象为 `dict` 或有 `__dict__` 的实例。
   - 构建 `SceneObject.from_dict()`，失败则记录警告并跳过该对象。
 
 `SceneObject.from_dict()` 重要行为：
 - `category` 取 `class_name` 的众数（忽略 `item/none/""`）。
 - `object_tag = category`。
 - `pcd_np` 被转换为 `np.ndarray`，并计算 `centroid`。
 - `image_idx`、`bbox_np`、`clip_ft` 等字段按原始数据填充。
 
 失败与分支：
 - 未找到任何 `*.pkl.gz` 文件，抛出 `FileNotFoundError`（在 `main` 未捕获，直接终止）。
 - 对象加载异常仅跳过该对象，不中断整体流程。
 - 若最终 `objects` 为空，`main()` 直接返回。
 
 ## 4. 单条查询执行（run_e2e_test）
 
 `run_e2e_test()` 的三个主要阶段：
 
 ### 4.1 解析阶段（Step 1）
 
 - `QueryParser(llm_model="gpt-5.2-2025-12-11", scene_categories=...)`
 - 解析流程：
   1. 构造系统提示词 + Few-shot + 场景类别列表。
   2. 调用 LLM `structured_llm.invoke(prompt)` 产出 `GroundingQuery`。
   3. 为所有节点分配 `node_id`（如 `root_sc0_a0`）。
   4. 成功后返回结构化结果。
 - 失败兜底：
   - 最多重试 2 次。
   - 全部失败后 fallback 到 `SimpleQueryParser`。
   - 如果 `SimpleQueryParser` 自身出错，会被 `run_e2e_test` 捕获并返回失败结果。
 
 `SimpleQueryParser` 支持的规则分支：
 - `"between X and Y"`（双 anchor）。
 - 序数词（first/second/...）→ `SelectConstraint(ORDINAL)`。
 - superlative（largest/nearest/...）→ `SelectConstraint(SUPERLATIVE)`。
 - 基本空间词（on/near/behind/next_to/...）。
 - 否则解析为单实体 + 属性列表。
 
### 4.2 执行阶段（Step 2：execute_with_tracking）

该脚本调用 `QueryExecutor.execute()` 完整执行链路，同时单独记录可视化所需的**初始候选**与**最终结果**：

1. **初始候选**：用 `_find_by_category(root.category)` 获取类别匹配对象，仅用于可视化。
2. **完整执行**：调用 `QueryExecutor.execute(parsed_query)`，内部依次执行：
   - 类别匹配 → 属性过滤 → 空间约束过滤 → 选择约束。
3. **最终结果**：记录 `ExecutionResult.matched_objects` 作为最终候选。

关键细节与边界情况（与 `execute()` 逻辑一致）：
- `_find_by_category()` 逻辑：
  - 精确匹配 → 子串匹配 → 同义词映射（pillow→throw_pillow 等）。
  - 如果无匹配且无 CLIP 特征，返回空列表并记录 warning。
- 若 anchor 结果为空：
  - `_apply_spatial_constraint()` 会直接返回原 candidates（即该约束不生效）。
- Quick filter：
  - 仅对 view-independent 关系（on/above/below/near/next_to/beside）可用。
  - 若 quick filter 误杀所有候选，会退回完整候选集合。
 
### 4.3 可视化阶段（Step 3）

- `save_filtering_steps()` 输出：
  - `00_initial_candidates.ply`
  - `01_final_candidates.ply`
  - `final_combined.ply`
  - `color_legend.txt`
- `save_keyframes()` 输出：
  - `keyframes/*.jpg|png`
  - 选择逻辑：按 `image_idx` 频次取 top views，
    并通过 `actual_frame_idx = view_idx * stride` 映射到真实帧文件。
 
 失败与分支：
 - `save_ply_with_colors()` 若所有对象无点云，会直接返回，不生成文件。
 - `results/` 目录不存在则跳过 keyframes。
 - keyframe 图片不存在则忽略该 view。
 
 ## 5. SpatialRelationChecker 与 quick filter 行为
 
 - 关系名规范化：`RELATION_ALIASES` 将常见表达映射到 canonical 名。
 - 未知关系：`SpatialRelationChecker.check()` 回退到 `is_near()`。
 - view-dependent 关系（left/right/front/behind）使用全局坐标系，
   且不进入 quick filter。
 - “between” 需要两个 anchor。
 
 ## 6. 输出与产物结构
 
 在 `room0/query_visualizations/` 下：
 
 - `<query_safe_name>/`
  - `00_initial_candidates.ply`
  - `01_final_candidates.ply`
   - `final_combined.ply`
   - `color_legend.txt`
   - `keyframes/*.jpg|png`
 - `test_results.json`：所有 query 的结构化结果汇总
 
 `safe_name` 规则：
 - 空格替换为 `_`
 - 去掉引号
 - 最长 50 字符
 
 ## 7. 典型失败路径与异常情况（全量覆盖）
 
 - **场景路径不存在**：`room0` 不存在，`main()` 直接返回。
 - **PCD 文件缺失**：`pcd_saves` 下无 `*.pkl.gz` → 抛出异常并终止。
 - **对象加载失败**：单对象失败仅跳过，该对象不会进入后续查询。
 - **LLM 不可用**：
   - `langchain_openai` 缺失或网络不可达 → 解析失败 → fallback 规则解析。
   - 若规则解析本身报错 → `parse_success=False`，该 query 提前返回。
 - **类别无匹配**：`_find_by_category` 可能返回空 → 整条查询无结果。
 - **anchor 解析为空**：空间约束无效，候选保持不变。
 - **选择约束越界**（ordinal 超出范围）：返回空结果。
 - **点云为空**：PLY 不生成或为空。
 - **results/ 帧缺失**：keyframes 目录仍创建，但无图片拷贝。
 - **可视化异常**：只影响 PLY/keyframes，不影响 `test_results.json`。
 
 ## 8. 执行链路总结（调用图）
 
 ```text
 python e2e_query_test.py
   └─ main()
      ├─ load_scene_objects()
      │  └─ SceneObject.from_dict()
      ├─ for query in test_queries:
      │  └─ run_e2e_test()
      │     ├─ QueryParser.parse()  (LLM → fallback)
     │     ├─ execute_with_tracking()
     │     │  ├─ QueryExecutor._find_by_category()   (visualization only)
     │     │  └─ QueryExecutor.execute()
      │     ├─ save_filtering_steps()
      │     └─ save_keyframes()
      └─ write test_results.json
 ```
 
