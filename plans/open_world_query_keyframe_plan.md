# Open-World Query Parsing + Keyframe Retrieval 详细执行计划（v3）

## 1. 核心目标（按执行顺序）
1. **先确定 Qwen 的结构化输出协议**（唯一真源）。
2. **先改 KeyframeSelector 以兼容该协议并可执行**（含必要代码）。
3. **最后按协议构造 Qwen 训练样本**（40/30/30）。

该顺序是强约束，不可颠倒。

## 2. 已知问题与本版修复范围
本版计划明确修复以下问题：
1. 输出结构不清晰（单解/多猜想不统一）。
2. 执行接口类型不匹配（`executor.execute(dict)` 不可行，需要代码适配）。
3. `categories` 与 `scene_categories` 约束冲突。
4. hard-case 存在掩蔽泄漏（隐藏类又出现在假设中）。
5. 样本字段仍有历史冗余（已去除 `gold_keyframes/gold_status`）。
6. 缺少可复现实验切分策略（train/val/test）。
7. 双教师生成缺少可复现控制（prompt版本/缓存/重试）。
8. 视角与检测帧对齐规则未定义（stride映射风险）。

## 3. 第一阶段：先确定 Qwen 输出结构（必须先完成）

### 3.1 输出协议（唯一版本）
定义 `HypothesisOutputV1`：
```json
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single" | "multi",
  "hypotheses": [
    {
      "kind": "direct" | "proxy" | "context",
      "rank": 1,
      "grounding_query": {"...": "标准 GroundingQuery"},
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}
```

### 3.2 协议硬约束
1. `hypotheses` 长度范围：`1..3`。
2. `rank` 必须是整数序（1,2,3），不使用浮点 `confidence` 监督。
3. `grounding_query.root.categories` **必须满足**：
   - 每个 category 属于输入 `scene_categories`，或
   - category 等于 `UNKNOW`。
4. 同义词/近义词信息放在 `lexical_hints`，**不能**放到不在场景内的 `categories`。
5. `parse_mode=single` 时，`hypotheses` 只能有 1 条，且 `kind=direct`。

### 3.3 协议验收
1. 给出 JSON Schema 文件 `schema/hypothesis_output_v1.json`。
2. 给出 20 条样例通过 schema 校验（正例）。
3. 给出 20 条反例被拒绝（负例，包括类别越界、rank重复、kind非法）。

## 4. 第二阶段：改 KeyframeSelector（必须包含代码实现）

### 4.1 代码改造目标
1. 兼容 `single` 与 `multi` 输出。
2. 解决 `dict -> GroundingQuery` 类型不匹配。
3. 增加 hard-case 防泄漏检查。
4. 明确 view/frame 对齐逻辑。

### 4.2 必改代码点（计划内任务）
1. 在 `conceptgraph/query_scene/keyframe_selector.py` 增加解析适配函数：
```python
def normalize_hypothesis_output(payload: dict) -> list[dict]:
    # 1) 兼容老格式（仅 grounding_query）
    # 2) 校验 format_version/parse_mode/rank
    # 3) 按 rank 排序返回 hypotheses
```

2. 在同文件增加类型转换函数：
```python
def to_grounding_query(h: dict) -> GroundingQuery:
    # 使用 GroundingQuery.model_validate(h["grounding_query"])
    # 失败时抛出结构错误，不允许静默跳过
```

3. 在执行前增加类别约束函数：
```python
def validate_categories_in_scene(gq: GroundingQuery, scene_categories: list[str]) -> None:
    # categories 必须 in scene_categories or == "UNKNOW"
```

4. 在 hard-case 数据构建流程增加泄漏检查：
```python
def validate_no_mask_leak(gq: GroundingQuery, hidden_categories: set[str]) -> None:
    # 若命中隐藏类，样本标记为无效
```

5. 在选帧流程增加对齐函数：
```python
def map_view_to_frame(view_id: int, stride: int) -> int:
    return view_id * stride
```
并加一致性检查：`frame{frame_id:06d}.jpg` 必须存在，否则回退相邻视角。

### 4.3 第二阶段验收
1. 适配层单测通过：single/multi/老格式均可执行。
2. `executor.execute` 只接收 `GroundingQuery` 对象。
3. hard-case 样本中隐藏类泄漏率为 0。
4. view/frame 映射日志可追踪（view_id、frame_id、文件路径）。

## 5. 第三阶段：构造 Qwen 训练样本（在前两阶段完成后）

### 5.1 样本分布
固定：`direct 40% + soft 30% + hard 30%`。

### 5.2 样本文件定义

#### 5.2.1 `parser_sft.jsonl`（用于训练 Qwen）
不包含 keyframe 字段。
```json
{
  "sample_id": "room0_direct_000123",
  "bucket": "direct",
  "scene_id": "room0",
  "scene_categories": ["throw_pillow", "pillow", "sofa", "door"],
  "user_query": "the throw pillow on the sofa",
  "target_output": {
    "format_version": "hypothesis_output_v1",
    "parse_mode": "single",
    "hypotheses": [
      {
        "kind": "direct",
        "rank": 1,
        "grounding_query": {"...": "GroundingQuery"},
        "lexical_hints": ["pillow"]
      }
    ]
  }
}
```

### 5.3 hard-case 构造规则（防泄漏版）
1. `M1`：仅从输入 `scene_categories` 删除目标类。
2. `M1+M2`：额外从执行对象池移除目标类实例。
3. 若 `target_output` 中出现隐藏类，样本直接判废。

### 5.4 双教师可复现控制
1. 固定 `prompt_version`（如 `p_qwen_sft_v3_20260307`）。
2. 样本记录 `teacher_model`, `temperature`, `seed`, `prompt_hash`。
3. 生成结果做缓存：key=`scene_id + program_hash + prompt_hash + model`。
4. API失败重试固定 2 次，仍失败写入 `generation_report.md`。

### 5.5 数据切分策略（防泄漏）
1. 先按 `scene_id` 切：train/val/test 场景互斥。
2. 再按 `program_hash` 去重：同结构不能跨 split。
3. paraphrase 仅在同 split 内扩展，禁止跨 split 文本变体。

## 6. 数据质量检查
1. 结构检查：`target_output` 必须通过 `HypothesisOutputV1` 校验。
2. 类别检查：`categories` 必须在 `scene_categories` 或为 `UNKNOW`。
3. hard-case 检查：`hidden_categories` 不得出现在可执行假设中。
4. 字段检查：`parser_sft` 不得包含 `gold_keyframes`、`gold_status` 等历史冗余字段。

## 7. 执行清单（严格顺序）
1. [x] 定义并冻结 `HypothesisOutputV1` schema。  
   - 实现：`schema/hypothesis_output_v1.json`
2. [x] 在 keyframe_selector 中实现 `normalize_hypothesis_output`。  
   - 实现：`conceptgraph/query_scene/keyframe_selector.py`
3. [x] 在 keyframe_selector 中实现 `to_grounding_query`（model_validate）。  
   - 实现：`conceptgraph/query_scene/keyframe_selector.py`
4. [x] 增加 `validate_categories_in_scene` 与 `validate_no_mask_leak`。  
   - 实现：`conceptgraph/query_scene/keyframe_selector.py`
5. [x] 增加 `map_view_to_frame` 与帧存在性校验。  
   - 实现：`map_view_to_frame` + `_resolve_keyframe_path`（邻近视角回退 + 路径存在检查）
6. [x] 跑单测，确认接口与类型稳定。  
   - 单测：`python -m unittest conceptgraph.query_scene.tests.test_hypothesis_output_schema conceptgraph.query_scene.tests.test_keyframe_selector_hypothesis -v`
   - 语法检查：`python -m py_compile ...` + `python -m json.tool schema/hypothesis_output_v1.json`
7. [x] 生成 `scene_manifest.jsonl`。  
   - 产物：`plans/generated_open_world/scene_manifest.jsonl`
8. [x] 生成 `query_program_pool.jsonl`。  
   - 产物：`plans/generated_open_world/query_program_pool.jsonl`（room0 当前 300 条）
9. [x] 按 40/30/30 采样并构造 hard 掩蔽样本。  
   - 产物：`plans/generated_open_world/parser_sft.jsonl`  
   - room0 当前分布：direct=120 / soft=90 / hard=90
10. [x] 双教师生成 query 并缓存。  
   - 实现：`conceptgraph/query_scene/open_world_sample_builder.py::TeacherQueryGenerator`  
   - 脚本：`conceptgraph/scripts/build_open_world_samples.py --use_teacher_llm`  
   - 缓存 key：`scene_id + program_hash + prompt_hash + model`  
   - 已支持：`prompt_version/temperature/seed` 元数据记录、固定重试（默认2次）、失败写入 `generation_report.md`
11. [x] 组装 `parser_sft.jsonl`（无 gold_keyframes）。  
   - 检查：文件无 `gold_keyframes/gold_status` 字段，hard 桶泄漏检查通过。
12. [ ] 实现并固化 train/val/test 切分（scene 互斥 + program_hash 去重）。  
   - 目标产物：`plans/generated_open_world/split_manifest.json`、`plans/generated_open_world/split_stats.md`
13. [ ] 多场景扩样（按同一协议扩展 scene_manifest/program_pool/parser_sft）。  
   - 目标：覆盖更多场景与类别组合，保持 40/30/30 分布。
14. [ ] 双教师链路稳定化。  
   - 目标：降低 API 失败率，提升缓存命中复用，输出生成过程报告。

## 8. 验收标准
1. 协议校验通过率 100%（产线数据）。
2. keyframe 执行输入无 dict 直传执行器（全量检查）。
3. hard-case 掩蔽泄漏率 = 0。
4. train/val/test 结构泄漏率 = 0。
5. 双教师生成链路可复现（prompt_version / prompt_hash / cache_key 稳定）。

## 9. 与案例文件的关系
1. 本计划是“流程与约束”。
2. 具体样本与可执行 case 在：`plans/open_world_query_keyframe_examples.md`。
3. 若案例与本计划冲突，以本计划（v3）为准并同步修订案例文件。
