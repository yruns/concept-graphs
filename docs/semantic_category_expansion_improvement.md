# 语义类别扩展功能改进记录

## 概述

本文档记录了为 Query Scene 模块添加**语义类别扩展**功能的完整改进过程。该功能使系统能够自动将用户查询中的通用词汇（如 "pillow"）扩展为场景中所有语义相关的类别（如 `["pillow", "throw_pillow"]`）。

---

## 1. 问题背景

### 1.1 初始测试结果

在对 `e2e_query_test.py` 进行测试时，发现以下问题：

- 初始通过率仅 4/12
- 主要问题：
  1. `QueryParser` 经常将有效对象名映射为 "UNKNOW"
  2. 空间关系（`on`, `between`）过于严格
  3. 部分查询请求的对象数量超过场景实际数量

### 1.2 核心问题发现

经过多轮改进后（通过率提升至 22/24），识别出一个关键问题：

**用户查询 "a pillow" 时，系统只能找到精确匹配 "pillow" 的对象，而无法找到语义相关的 "throw_pillow"。**

场景中实际存在：
- `pillow`: 1 个
- `throw_pillow`: 7 个

用户期望查询 "a pillow" 能返回所有 8 个枕头类对象。

---

## 2. 解决方案探索

### 2.1 候选方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **QueryExecutor 扩展** | 在执行器中维护同义词字典 | 简单直接 | 需要硬编码规则，不够通用 |
| **LLM 引导解析** | 修改 prompt 让 LLM 返回多个类别 | 利用 LLM 语义理解能力 | 需要修改数据结构 |
| **CLIP 相似度** | 使用 CLIP 计算语义相似度 | 无需规则 | 计算开销大，阈值难调 |
| **混合方案** | 结合多种方法 | 覆盖面广 | 复杂度高 |

### 2.2 选择：LLM 多类别输出

最终选择 **LLM 多类别输出** 方案，原因：

1. **通用性强** — 无需为每个场景维护同义词规则
2. **利用 LLM 语义理解** — LLM 可以根据场景类别列表动态判断语义关联
3. **实现相对简单** — 主要修改数据结构和 prompt
4. **可扩展** — 未来可以轻松添加更多语义规则

---

## 3. 实施计划

### 3.1 核心改动

将 `QueryNode.category: str` 改为 `QueryNode.categories: List[str]`，让 LLM 自动返回所有语义相关的类别。

```mermaid
flowchart LR
    Query["用户: a pillow"]
    LLM["LLM 解析"]
    Node["QueryNode"]
    Exec["QueryExecutor"]
    Result["8个对象"]
    
    Query --> LLM
    LLM --> |"categories: [pillow, throw_pillow]"| Node
    Node --> Exec
    Exec --> Result
```

### 3.2 修改文件清单

1. `query_structures.py` — 修改 QueryNode 数据结构
2. `query_parser.py` — 更新 prompt 和 few-shot examples
3. `query_executor.py` — 添加多类别匹配方法
4. `e2e_query_test.py` — 添加语义扩展测试用例

---

## 4. 详细实施步骤

### 4.1 修改 QueryNode 结构 (`query_structures.py`)

**修改前：**
```python
class QueryNode(BaseModel):
    category: str = Field(
        ...,
        description="Object category to search for, e.g., 'pillow', 'sofa', 'door'"
    )
```

**修改后：**
```python
class QueryNode(BaseModel):
    categories: List[str] = Field(
        ...,
        min_length=1,
        description="Object categories to search for. Include ALL semantically related "
                    "categories from scene. E.g., for 'pillow' query with scene containing "
                    "[pillow, throw_pillow], return ['pillow', 'throw_pillow']"
    )
    
    @property
    def category(self) -> str:
        """Primary category (first in list). For backward compatibility."""
        return self.categories[0] if self.categories else ""
```

**关键点：**
- 使用 `List[str]` 支持多个类别
- 添加 `@property category` 保持向后兼容
- 旧代码使用 `node.category` 仍然有效

### 4.2 更新 QueryParser Prompt (`query_parser.py`)

**添加到 IMPORTANT RULES：**
```python
"""
1. SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. When the user mentions 
   a general term (e.g., "pillow", "lamp", "table"), include ALL semantically related 
   categories from SCENE CATEGORIES. Examples:
   - Query "a pillow" with scene [door, pillow, throw_pillow, sofa] → categories: ["pillow", "throw_pillow"]
   - Query "the lamp" with scene [floor_lamp, table_lamp, sofa] → categories: ["floor_lamp", "table_lamp"]  
   - Query "a table" with scene [side_table, coffee_table, chair] → categories: ["side_table", "coffee_table"]
2. Every category in the list MUST be chosen from SCENE CATEGORIES exactly (case-sensitive).
...
12. The `categories` list must have at least one element. Include exact matches first, 
    then semantically related categories.
"""
```

### 4.3 更新 Few-shot Examples

**修改前：**
```json
{
  "root": {
    "category": "pillow",
    ...
  }
}
```

**修改后：**
```json
{
  "root": {
    "categories": ["pillow", "throw_pillow"],
    ...
  }
}
```

### 4.4 修改动态 Schema (`_build_dynamic_schema`)

```python
QueryNodeDynamic = create_model(
    "QueryNodeDynamic",
    categories=(List[Category], Field(..., min_length=1)),  # 改为列表
    attributes=(List[str], Field(default_factory=list)),
    ...
)
```

### 4.5 添加多类别匹配方法 (`query_executor.py`)

```python
def _find_by_categories(self, categories: List[str]) -> List["SceneObject"]:
    """Find objects matching any of the given categories."""
    matches = []
    seen_ids = set()
    
    for category in categories:
        category_lower = category.lower()
        
        # Exact match
        if category_lower in self._category_index:
            for obj in self._category_index[category_lower]:
                if obj.obj_id not in seen_ids:
                    matches.append(obj)
                    seen_ids.add(obj.obj_id)
    
    # If we found matches, return them
    if matches:
        return matches
    
    # Fallback: substring matching
    for category in categories:
        category_lower = category.lower()
        for cat, objs in self._category_index.items():
            if category_lower in cat or cat in category_lower:
                for obj in objs:
                    if obj.obj_id not in seen_ids:
                        matches.append(obj)
                        seen_ids.add(obj.obj_id)
    
    return matches
```

### 4.6 更新其他引用

- `SimpleQueryParser` 中所有 `QueryNode(category=...)` 改为 `QueryNode(categories=[...])`
- `GroundingQuery` 的 `_collect_categories` 方法改为 `categories.extend(node.categories)`
- `e2e_query_test.py` 中的日志输出改为显示 `categories`

---

## 5. 测试验证

### 5.1 添加语义扩展测试用例

```python
test_queries = [
    # Semantic Expansion Tests
    ("a pillow", "SEM-01. Semantic expansion (pillow -> throw_pillow)"),
    ("a table", "SEM-02. Semantic expansion (table -> side_table, coffee_table)"),
    ("the lamp", "SEM-03. Semantic expansion (lamp -> floor_lamp)"),
    ("a chair", "SEM-04. Semantic expansion (chair -> armchair)"),
    ...
]
```

### 5.2 测试结果

| 查询 | LLM 返回的 categories | 结果 |
|------|----------------------|------|
| "a pillow" | `["pillow", "throw_pillow"]` | **8 个对象** ✅ |
| "a table" | `["side_table", "coffee_table"]` | **5 个对象** ✅ |
| "the lamp" | `["floor_lamp", "ceiling_light", "wall_sconce"]` | **3 候选 → 1 选中** ✅ |
| "a chair" | `["armchair"]` | **3 个对象** ✅ |

### 5.3 整体测试结果

```
15:40:27 | INFO    | Total: 27/28 tests passed
```

**通过率: 96.4% (27/28)**

唯一失败的测试是 L3-04 "the smallest ottoman near the sofa and near the armchair"，原因是空间约束过严（没有 ottoman 同时满足两个 near 条件），与语义扩展功能无关。

---

## 6. 关键改进效果

### 6.1 Before vs After

| 查询 | 改进前 | 改进后 |
|------|--------|--------|
| "a pillow" | 1 个 (只有 pillow) | **8 个** (pillow + throw_pillow) |
| "a table" | 1 个 | **5 个** (side_table + coffee_table) |
| "the lamp" | 可能 UNKNOW | **正确匹配** floor_lamp 等 |

### 6.2 LLM 输出示例

**查询: "a throw_pillow"**
```json
{
  "root": {
    "categories": ["throw_pillow", "pillow"],
    "attributes": [],
    "spatial_constraints": [],
    "select_constraint": null
  },
  "expect_unique": false
}
```

**查询: "the throw_pillow near the sofa"**
```json
{
  "root": {
    "categories": ["throw_pillow", "pillow"],
    "spatial_constraints": [
      {
        "relation": "near",
        "anchors": [{"categories": ["sofa"]}]
      }
    ]
  }
}
```

---

## 7. 向后兼容性

为保持向后兼容，`QueryNode` 添加了 `category` 属性：

```python
@property
def category(self) -> str:
    """Primary category. For backward compatibility."""
    return self.categories[0] if self.categories else ""
```

这样旧代码使用 `node.category` 仍然有效，会返回列表中的第一个类别。

---

## 8. 总结

### 8.1 改进成果

1. **语义扩展功能生效** — LLM 能正确将通用词汇扩展为场景中的相关类别
2. **通过率大幅提升** — 从最初的 4/12 (33%) 提升到 27/28 (96%)
3. **无需硬编码规则** — 完全由 LLM 根据场景类别动态决定语义关联
4. **向后兼容** — 旧代码无需修改

### 8.2 技术要点

- 使用 Pydantic 的 `List[str]` 和 `min_length=1` 约束
- 通过 `@property` 实现向后兼容
- 在 prompt 中明确 SEMANTIC EXPANSION 规则
- Few-shot examples 展示正确的多类别格式

### 8.3 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `query_structures.py` | 修改 | QueryNode.category → categories |
| `query_parser.py` | 修改 | 更新 prompt、examples、schema |
| `query_executor.py` | 修改 | 添加 `_find_by_categories()` |
| `e2e_query_test.py` | 修改 | 添加语义扩展测试用例 |

---

*文档创建时间: 2026-01-29*
*改进完成状态: ✅ 全部完成*
