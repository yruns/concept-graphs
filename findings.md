# KeyframeSelector 机制研究发现

## 0. QueryParser 与 KeyframeSelector 关系

### 架构关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                       KeyframeSelector                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  私有成员:                                                    │   │
│  │  _query_parser: QueryParser (懒加载)                          │   │
│  │  _query_executor: QueryExecutor                               │   │
│  │  _relation_checker: SpatialRelationChecker                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│          ┌───────────────────┼───────────────────┐                  │
│          ▼                   ▼                   ▼                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ QueryParser  │   │QueryExecutor │   │ SpatialRel   │            │
│  │              │   │              │   │   Checker    │            │
│  │ 解析 NL 查询  │──>│ 执行查询结构  │<──│ 验证空间关系  │            │
│  │ → 结构化输出  │   │ → 匹配物体   │   │              │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 职责分离

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **KeyframeSelector** | 协调器，选择最佳视角 | 场景路径 + 查询 | KeyframeResult |
| **QueryParser** | NL → 结构化查询 | str 查询 + 场景类别 | GroundingQuery |
| **QueryExecutor** | 结构化查询 → 物体 | GroundingQuery | ExecutionResult |
| **SpatialRelationChecker** | 验证空间关系 | 物体对 + 关系 | bool + score |

### 调用链路

```
KeyframeSelector.select_keyframes_v2(query)
    │
    ├── 1. parse_query_hypotheses(query)
    │       │
    │       └── _get_query_parser().parse(query)
    │               │
    │               └── QueryParser._do_parse(query)
    │                       │
    │                       ├── LLM 调用 (结构化输出)
    │                       └── → GroundingQuery
    │
    ├── 2. execute_hypotheses(hypothesis_output)
    │       │
    │       └── execute_query(grounding_query)
    │               │
    │               └── QueryExecutor.execute()
    │                       │
    │                       └── → ExecutionResult (matched_objects)
    │
    └── 3. get_joint_coverage_views(object_ids)
            │
            └── → keyframe_indices
```

### 两种解析 API

| API | 位置 | 用途 | 返回 |
|-----|------|------|------|
| `parse_query()` | keyframe_selector.py:770 | V1 简单解析 | (target, anchor, relation) |
| `parse_query_hypotheses()` | keyframe_selector.py:1521 | V3 多假设解析 | HypothesisOutputV1 |

- **V1 API**: 直接 LLM 调用，返回三元组，用于 `select_keyframes()`
- **V3 API**: 通过 QueryParser，返回结构化假设，用于 `select_keyframes_v2()`

### 数据流转换

```
"the pillow on the sofa"
        │
        ▼ QueryParser.parse()
GroundingQuery {
    raw_query: "the pillow on the sofa"
    root: QueryNode {
        categories: ["pillow", "throw_pillow"]
        spatial_constraints: [{
            relation: "on"
            anchors: [QueryNode{categories: ["sofa"]}]
        }]
    }
}
        │
        ▼ KeyframeSelector.normalize_hypothesis_output()
HypothesisOutputV1 {
    parse_mode: SINGLE
    hypotheses: [
        QueryHypothesis {kind: DIRECT, rank: 1, grounding_query: ...}
    ]
}
        │
        ▼ KeyframeSelector.execute_hypotheses()
ExecutionResult {
    matched_objects: [SceneObject, ...]
}
        │
        ▼ get_joint_coverage_views()
KeyframeResult {
    keyframe_indices: [42, 67, 89]
    target_objects: [...]
}
```

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    KeyframeSelector                          │
├─────────────────────────────────────────────────────────────┤
│  输入: 场景路径 + 自然语言查询                                │
│  输出: KeyframeResult (关键帧索引 + 路径 + 匹配物体)          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      ┌──────────────┐               ┌──────────────┐
      │  场景加载     │               │  查询处理     │
      └──────────────┘               └──────────────┘
              │                               │
    ┌─────────┼─────────┐           ┌────────┼────────┐
    ▼         ▼         ▼           ▼        ▼        ▼
  物体      位姿      可见性      解析     匹配     选择
  加载      加载      索引构建    查询     物体     视角
```

## 2. 核心数据结构

### 2.1 SceneObject (行 64-167)
场景物体的完整表示：

| 属性 | 类型 | 说明 |
|------|------|------|
| `obj_id` | int | 物体唯一 ID |
| `category` | str | 主类别 (从 class_name 列表投票) |
| `centroid` | np.ndarray | 3D 质心 [x, y, z] |
| `pcd_np` | np.ndarray | 点云 (N, 3) |
| `clip_ft` | np.ndarray | CLIP 视觉特征 (1024,) |
| `image_idx` | List[int] | 物体出现的帧索引 |
| `xyxy` | List[Any] | 每帧的 2D 边界框 |
| `co_objects` | List[str] | 共现物体 (来自 affordance) |

### 2.2 KeyframeResult (行 169-199)
关键帧选择结果：

```python
@dataclass
class KeyframeResult:
    query: str                          # 原始查询
    target_term: str                    # 解析出的目标词
    anchor_term: Optional[str]          # 锚点词 (如有)
    keyframe_indices: List[int]         # 选中的帧索引
    keyframe_paths: List[Path]          # 帧图片路径
    target_objects: List[SceneObject]   # 匹配的目标物体
    anchor_objects: List[SceneObject]   # 匹配的锚点物体
    metadata: Dict[str, Any]            # 额外元数据
```

## 3. 场景加载流程

### 3.1 初始化入口 (行 273-312)
```python
selector = KeyframeSelector.from_scene_path("/path/to/scene")
```

自动检测文件：
1. `pcd_saves/*ram*_post.pkl.gz` → 3D 物体数据
2. `sg_cache_detect/object_affordances.json` → affordance 数据

### 3.2 物体加载 (_load_objects_from_pcd, 行 341-411)

```
pkl.gz 文件
    │
    ▼
遍历 objects 列表
    │
    ├── 提取点云 pcd_np → 计算质心
    ├── 提取 CLIP 特征 clip_ft
    ├── 投票确定类别 (Counter 最常见)
    └── 构建 SceneObject
    │
    ▼
构建特征矩阵 object_features (N, D)
    │
    ▼
L2 归一化 (用于后续余弦相似度)
```

### 3.3 可见性索引 (行 493-641)

**双向索引结构**：
- `object_to_views`: obj_id → [(view_id, score), ...] 按分数降序
- `view_to_objects`: view_id → [(obj_id, score), ...] 按分数降序

**可见性分数计算** (基于检测数据):
```
score = 0.5 * completeness + 0.3 * geometric + 0.2 * quality

completeness: 边界框面积占比 + 边缘裁剪惩罚
geometric:    距离分数 (0.6) + 视角分数 (0.4)
quality:      检测次数 / 3 (最大 1.0)
```

## 4. 查询解析机制

### 4.1 基础解析 (parse_query, 行 770-858)

使用 LLM 解析查询为三元组：
- `target`: 目标物体 (如 "throw_pillow")
- `anchor`: 锚点物体 (如 "sofa")
- `relation`: 空间关系 (如 "on", "near")

```
Query: "pillow on the sofa"
         │
         ▼ LLM
   ┌─────────────────┐
   │ target: pillow  │
   │ anchor: sofa    │
   │ relation: on    │
   └─────────────────┘
```

### 4.2 多假设系统 (parse_query_hypotheses, 行 1521-1582)

生成 HypothesisOutputV1，包含多个备选假设：

```
           Query
              │
              ▼
    ┌─────────────────┐
    │   DIRECT 假设    │  rank=1, 直接解析
    └─────────────────┘
              │
              ▼ (如果目标是 UNKNOW 或无匹配)
    ┌─────────────────┐
    │   PROXY 假设     │  rank=2, 代理物体
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  CONTEXT 假设    │  rank=3, 上下文兜底
    └─────────────────┘
```

**假设类型**:
| Kind | 说明 | 触发条件 |
|------|------|----------|
| DIRECT | 直接解析结果 | 默认 |
| PROXY | 代理物体 | 目标/锚点为 UNKNOW |
| CONTEXT | 上下文兜底 | 前两者都失败 |

## 5. 物体匹配机制 (find_objects, 行 860-908)

**两阶段匹配**：

```
阶段 1: 字符串匹配 (精确)
    query_lower in tag OR tag in query_lower
           │
           ▼ (如果无匹配)
阶段 2: CLIP 语义匹配 (模糊)
    similarity = object_features @ text_feature
    threshold > 0.2
```

## 6. 关键帧选择策略

### 6.1 联合覆盖策略 (get_joint_coverage_views, 行 915-977)

**贪心算法**：每次选择边际增益最大的视角

```python
for _ in range(max_views):
    best_view = argmax(marginal_gain)
    selected.append(best_view)
    update(covered_quality)
```

**边际增益计算**：
```
gain = Σ max(0, view_score[obj] - covered_quality[obj])
```

### 6.2 空间过滤 (_spatial_filter, 行 1072-1098)

使用 SpatialRelationChecker 验证空间关系：
```python
result = checker.check(target_obj, anchor_obj, relation)
if result.satisfies:
    filtered.append((obj, result.score))
```

## 7. 完整流程 (select_keyframes_v2)

```
         Query: "pillow on the sofa"
                     │
                     ▼
    ┌─────────────────────────────┐
    │ 1. parse_query_hypotheses   │
    │    生成多假设                │
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │ 2. execute_hypotheses       │
    │    按 rank 执行假设          │
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │ 3. get_joint_coverage_views │
    │    贪心选择最佳视角          │
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │ 4. _resolve_keyframe_path   │
    │    映射到实际图片路径        │
    └─────────────────────────────┘
                     │
                     ▼
              KeyframeResult
```

## 8. 关键设计决策

1. **Stride 机制**: 以固定步长采样帧 (默认 5), 减少计算量
2. **双向可见性索引**: 支持快速查询物体→视角和视角→物体
3. **多假设回退**: 直接解析失败时有 PROXY/CONTEXT 兜底
4. **CLIP 语义匹配**: 处理同义词和语义相似性
5. **贪心联合覆盖**: 最大化多物体在选定视角中的可见性
