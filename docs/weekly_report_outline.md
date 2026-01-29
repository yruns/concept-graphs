# 周会汇报大纲：3D场景语义查询系统

## 一、研究背景与动机（2-3分钟）

### 1.1 问题定义
- **任务**：自然语言驱动的 3D 场景物体定位（Language-Grounded 3D Object Retrieval）
- **输入**：自然语言查询 + 3D 场景点云/对象
- **输出**：目标物体的 3D 位置、边界框、点云

### 1.2 应用场景
- 具身智能（Embodied AI）：机器人理解 "拿桌子上的杯子"
- AR/VR 交互：用户用语言操控虚拟物体
- 智能家居：语音控制定位家中物品

### 1.3 核心挑战
| 挑战 | 描述 |
|------|------|
| 复杂空间关系 | "桌子上离门最近的枕头" 需要多层嵌套推理 |
| 语义模糊性 | "pillow" 可能指 pillow、throw_pillow、cushion |
| 多模态融合 | 需要结合语言、视觉、3D 几何信息 |

---

## 二、系统整体架构（5-7分钟）

### 2.1 系统流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     用户自然语言查询                                  │
│              "the pillow on the sofa nearest the door"              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Step 1: Query Parser (LLM)                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ • 自然语言 → 结构化 GroundingQuery                             │  │
│  │ • 语义扩展: "pillow" → ["pillow", "throw_pillow"]              │  │
│  │ • 嵌套解析: 空间约束 + 选择约束                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Step 2: Query Executor                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ • 递归执行查询节点（自底向上）                                  │  │
│  │ • 类别匹配 → 空间过滤 → 选择约束                               │  │
│  │ • 支持任意深度嵌套                                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       返回结果                                       │
│            目标物体 ID、3D 位置、点云、置信度                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块详解

#### 2.2.1 Query Parser（查询解析器）

**功能**：将自然语言查询转换为结构化查询表示

**技术方案**：
- 使用 LLM（GPT/Gemini）+ Structured Output
- 动态生成 Pydantic Schema（基于场景类别）
- Few-shot Prompting 引导正确输出格式

**输入输出示例**：
```
输入: "the pillow on the sofa nearest the door"

输出 (GroundingQuery):
{
  "root": {
    "categories": ["pillow", "throw_pillow"],  // 语义扩展
    "spatial_constraints": [{
      "relation": "on",
      "anchors": [{
        "categories": ["sofa"],
        "select_constraint": {
          "type": "superlative",
          "metric": "distance",
          "order": "min",
          "reference": {"categories": ["door"]}
        }
      }]
    }]
  }
}
```

#### 2.2.2 Query Executor（查询执行器）

**功能**：递归执行结构化查询，返回匹配物体

**执行流程**：
```
1. 类别匹配
   - 根据 categories 列表查找候选物体
   - 支持精确匹配 + 子串匹配 + CLIP 相似度

2. 属性过滤
   - 颜色过滤（red, blue, ...）
   - 大小过滤（large, small）

3. 空间约束过滤（AND 逻辑）
   - 快速过滤（Quick Filter）：坐标级预筛选
   - 完整检查（Full Check）：精确几何关系

4. 选择约束
   - Superlative: nearest, largest, smallest
   - Ordinal: first, second, third from left
```

#### 2.2.3 Spatial Relations（空间关系检查器）

**支持的空间关系**（~15种）：

| 类别 | 关系 |
|------|------|
| 垂直 | on, above, below |
| 水平 | left_of, right_of, in_front_of, behind |
| 距离 | near, next_to, beside |
| 包含 | inside, between |

**两阶段检查**：
1. **Quick Filter**：基于坐标的快速预筛选（O(n)）
2. **Full Check**：基于点云的精确几何检查

### 2.3 数据结构设计

```
GroundingQuery
├── raw_query: str              # 原始查询
├── expect_unique: bool         # 是否期望唯一结果
└── root: QueryNode             # 根查询节点
    ├── categories: List[str]   # 类别列表（支持语义扩展）
    ├── attributes: List[str]   # 属性过滤
    ├── spatial_constraints     # 空间约束（AND 逻辑）
    │   ├── relation: str       # 空间关系
    │   └── anchors: List[QueryNode]  # 参考物体（递归）
    └── select_constraint       # 选择约束
        ├── type: superlative/ordinal
        ├── metric: distance/size/height
        ├── order: min/max/asc/desc
        └── reference: QueryNode  # 参考物体
```

---

## 三、本周改进：语义类别扩展（5-7分钟）

### 3.1 问题发现

**现象**：
```
查询: "a pillow"
场景类别: [pillow(1个), throw_pillow(7个), sofa, door, ...]
期望结果: 8 个枕头类物体
实际结果: 1 个（只匹配 "pillow"）
```

**根本原因**：精确类别匹配无法处理语义相关类别

### 3.2 解决方案对比

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 同义词字典 | 维护 pillow→throw_pillow 映射 | 简单 | 需要硬编码，不通用 |
| CLIP 相似度 | 用 CLIP 计算语义相似度 | 无需规则 | 计算开销大，阈值难调 |
| **LLM 扩展** ✅ | LLM 根据场景类别自动扩展 | 通用、智能 | 依赖 LLM |

### 3.3 实现方案：LLM 多类别输出

**核心思想**：让 LLM 在解析时自动返回所有语义相关的类别

**Prompt 设计**：
```
SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. 
When the user mentions a general term (e.g., "pillow", "lamp", "table"), 
include ALL semantically related categories from SCENE CATEGORIES.

Examples:
- Query "a pillow" with scene [pillow, throw_pillow, sofa] 
  → categories: ["pillow", "throw_pillow"]
- Query "a table" with scene [side_table, coffee_table, chair] 
  → categories: ["side_table", "coffee_table"]
```

**数据结构修改**：
```python
# Before
class QueryNode:
    category: str  # 单个类别

# After
class QueryNode:
    categories: List[str]  # 类别列表
    
    @property
    def category(self) -> str:  # 向后兼容
        return self.categories[0]
```

### 3.4 代码改动量

| 文件 | 改动内容 | 行数 |
|------|----------|------|
| `query_structures.py` | QueryNode 结构重构 | ~30 |
| `query_parser.py` | Prompt + Schema + Examples | ~100 |
| `query_executor.py` | 多类别匹配方法 | ~50 |
| `e2e_query_test.py` | 测试用例 | ~30 |
| **总计** | | **~210行** |

### 3.5 实验结果

**语义扩展效果**：

| 查询 | LLM 输出 categories | 结果 |
|------|---------------------|------|
| "a pillow" | `["pillow", "throw_pillow"]` | 8 个 ✅ |
| "a table" | `["side_table", "coffee_table"]` | 5 个 ✅ |
| "the lamp" | `["floor_lamp", "ceiling_light", "wall_sconce"]` | 3 个 ✅ |
| "a chair" | `["armchair"]` | 3 个 ✅ |

**整体测试结果**：

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 测试通过率 | 33% (4/12) | **96% (27/28)** |
| 语义覆盖率 | 低 | **高** |

---

## 四、可视化展示（2-3分钟）

### 4.1 点云可视化

- 展示 `query_visualizations/` 目录下的 PLY 文件
- 颜色编码：蓝色=初始候选，红色=最终结果

### 4.2 查询解析结果

展示 LLM 输出的结构化 JSON：
```json
{
  "raw_query": "the pillow on the sofa nearest the door",
  "root": {
    "categories": ["pillow", "throw_pillow"],
    "spatial_constraints": [...]
  }
}
```

---

## 五、总结与后续工作（1-2分钟）

### 5.1 本周工作总结

1. ✅ 实现了 LLM 驱动的语义类别扩展
2. ✅ 测试通过率从 33% 提升至 96%
3. ✅ 系统可自动适应不同场景的类别命名

### 5.2 后续工作

| 方向 | 描述 |
|------|------|
| 区域索引 | 添加 "kitchen", "living room" 等区域级检索 |
| VLM 增强 | 集成 VLM 处理多候选歧义情况 |
| 更多场景测试 | 在更多真实场景上验证泛化能力 |

---

## 附录：演示脚本

```bash
# 运行端到端测试
cd /home/ysh/codecase/concept-graphs
bash bashes/run_e2e_query_test.sh

# 查看可视化结果
ls room0/query_visualizations/
```

---

*预计汇报时长：15-20 分钟*
