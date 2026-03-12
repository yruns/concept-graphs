# Query Case Generator 设计方案 v2

> **目标**: 生成带有 Ground Truth 的评估用例，通过在图像上用红框标注目标物体，让 Gemini 针对该物体生成查询。

## Codex Review Round 1 反馈修复

| 问题 | 修复方案 |
|------|----------|
| visibility_index 格式是 List 不是 dict | 添加适配器 `_build_view_score_dict()` |
| frame_idx vs view_id 混淆 | 明确使用 view_id，仅在 I/O 时转换 |
| 标签泄露 obj_id/category | 改用匿名标记 A/B/C |
| anchor_obj_ids 由 LLM 生成不可靠 | 后处理验证 + 从解析结果推断 |
| 多目标采样失败率高 | 添加 fallback 策略 |
| 可变默认值 `=[]` | 使用 `Field(default_factory=list)` |

## Codex Review Round 2 反馈修复

| 问题 | 修复方案 |
|------|----------|
| QueryParser 需要 scene_categories 参数 | 使用工厂函数 `_create_parser()` 集中初始化 |
| hypotheses[0] 可能不是 direct 类型 | 按 kind 优先级选择 (direct > proxy > context) |
| anchor_obj_ids 按 category 匹配太宽泛 | 改用 view-local grounding (在 source_view_id 可见) |
| 验证过于宽松 (set intersection) | 要求 parsed targets 完全匹配 GT |
| source_view_id 占位符容易忘记覆盖 | 作为必传参数传入 post_process_case |
| fallback 采样导致分布偏移 | 添加 quota-aware 采样器，追踪目标数量配额 |
| 串行调用 LLM 效率低 | 添加并行批处理 + worker pool |
| bbox 缺少几何校验 | 添加边界校验 + 最小面积阈值 |
| infer_query_type 仅检查 root | 遍历完整 query tree |
| Image.open 无 context manager | 使用 `with Image.open(...) as im:` |

## 1. 核心思路

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 随机选择场景中的 target 物体 (1-3 个)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. 从可见性索引找到该物体可见的关键帧                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. 在图像上用红色边框圈出该物体 (使用检测框 xyxy)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. 发送标注图像给 Gemini，要求生成针对红框物体的查询                        │
├─────────────────────────────────────────────────────────────────────────┤
│ 5. 保存 EvaluationCase: {query, target_obj_ids, difficulty, ...}         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. 数据结构

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class QueryDifficulty(str, Enum):
    EASY = "easy"       # 单个明显物体
    MEDIUM = "medium"   # 需要空间关系
    HARD = "hard"       # 多个候选需区分

class QueryType(str, Enum):
    DIRECT = "direct"           # "the sofa"
    SPATIAL = "spatial"         # "the pillow on the sofa"
    ATTRIBUTE = "attribute"     # "the red pillow"
    SUPERLATIVE = "superlative" # "the largest sofa"
    MULTI_TARGET = "multi"      # "the pillows on the sofa"

class EvaluationCase(BaseModel):
    """带 Ground Truth 的评估用例"""

    # Query
    query: str                          # "the throw pillow on the sofa"

    # Ground Truth (核心)
    target_obj_ids: List[int]           # [15] - 被红框圈出的目标物体
    target_categories: List[str]        # ["throw_pillow"]

    # Derived fields (后处理推断，非 LLM 直接输出)
    anchor_obj_ids: List[int] = Field(default_factory=list)
    anchor_categories: List[str] = Field(default_factory=list)
    spatial_relation: Optional[str] = None
    query_type: QueryType = QueryType.DIRECT
    difficulty: QueryDifficulty = QueryDifficulty.EASY

    # Generation context
    source_view_id: int                 # 用于生成的视角 (view_id, not frame_idx)
    source_frame_path: str              # 相对路径: "results/frame000127.jpg"

    # LLM generation info
    raw_llm_response: str = ""
    generation_timestamp: str = ""

    # Validation flags
    validated: bool = False             # 是否通过后处理验证
    validation_errors: List[str] = Field(default_factory=list)

class GenerationBatch(BaseModel):
    """一批生成的用例"""
    scene_name: str
    scene_path: str                     # 相对路径
    cases: List[EvaluationCase] = Field(default_factory=list)
    generation_config: dict = Field(default_factory=dict)
    total_generated: int = 0
    failed_count: int = 0
    validation_passed: int = 0
```

## 3. 图像标注策略

### 3.1 获取物体边界框

场景物体有检测数据，包含每帧的 2D bbox：

```python
class SceneObject:
    image_idx: List[int]     # 检测到该物体的 view_id (not frame_idx!)
    xyxy: List[List[float]]  # 每个检测的 2D bbox [x1, y1, x2, y2]
    conf: List[float]        # 置信度
```

**重要**: `image_idx` 是 view_id，需要通过 `map_view_to_frame()` 转换为 frame_idx。

### 3.2 绘制标注 (匿名标记，不泄露 obj_id)

```python
from PIL import Image, ImageDraw, ImageFont

# 匿名标记: A, B, C...
MARKERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def annotate_image_with_targets(
    image_path: Path,
    objects: List[SceneObject],
    view_id: int,  # 使用 view_id，不是 frame_idx
    box_color: str = "red",
    box_width: int = 4,
    label_font_size: int = 28,
    min_bbox_area: int = 500,  # 最小 bbox 面积阈值
) -> Tuple[Image.Image, Dict[str, int]]:
    """在图像上用红框标注目标物体，使用匿名标记

    Args:
        image_path: 原始图像路径
        objects: 要标注的目标物体
        view_id: 当前视角 ID (用于查找对应的 bbox)
        min_bbox_area: 最小边界框面积阈值

    Returns:
        (标注后的 PIL Image, 标记到 obj_id 的映射 {"A": 15, "B": 23})

    Raises:
        ValueError: 如果任何目标物体的 bbox 无效
    """
    # 使用 context manager 确保资源释放
    with Image.open(image_path) as img:
        img = img.copy()  # 复制以便在 context 外使用
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)

        marker_to_obj_id = {}

        for i, obj in enumerate(objects):
            marker = MARKERS[i % len(MARKERS)]

            # 找到该物体在当前 view_id 的最佳 bbox
            bbox = get_best_bbox_in_view(obj, view_id, min_bbox_area)
            if bbox is None:
                raise ValueError(f"Object {obj.obj_id} has no valid bbox in view {view_id}")

            x1, y1, x2, y2 = bbox

            # 绘制红色边框
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            # 添加匿名标签: "A" (不包含 obj_id 和 category)
            # 标签放置: 优先放在 bbox 上方，如果空间不足则放在 bbox 内部顶部
            label_height = 35
            label_width = 35
            if y1 >= label_height:
                # 标签在 bbox 上方
                label_y_start = y1 - label_height
            else:
                # 标签在 bbox 内部顶部
                label_y_start = y1 + 2

            # 确保标签不超出图像边界
            label_x_start = max(0, min(x1, img_width - label_width))
            label_y_start = max(0, label_y_start)

            label_bg = [
                label_x_start,
                label_y_start,
                label_x_start + label_width,
                label_y_start + label_height
            ]
            draw.rectangle(label_bg, fill="white", outline=box_color, width=2)
            draw.text((label_x_start + 8, label_y_start + 3), marker, fill=box_color)

            marker_to_obj_id[marker] = obj.obj_id

    return img, marker_to_obj_id

def get_best_bbox_in_view(
    obj: SceneObject,
    view_id: int,
    min_area: int = 500,
    min_conf: float = 0.3,
) -> Optional[List[float]]:
    """获取物体在指定 view_id 的最佳 2D bbox

    如果有多个检测，选择满足条件且置信度最高的。

    Args:
        obj: 场景物体
        view_id: 视角 ID
        min_area: 最小面积阈值
        min_conf: 最小置信度阈值

    Returns:
        有效的 bbox [x1, y1, x2, y2]，或 None
    """
    if view_id not in obj.image_idx:
        return None

    # 找到所有该 view_id 的检测
    indices = [i for i, v in enumerate(obj.image_idx) if v == view_id]

    if not indices:
        return None

    # 过滤有效的 bbox
    valid_detections = []
    for idx in indices:
        if idx >= len(obj.xyxy) or idx >= len(obj.conf):
            continue

        bbox = obj.xyxy[idx]
        conf = obj.conf[idx]
        x1, y1, x2, y2 = bbox

        # 几何校验
        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        if conf < min_conf:
            continue

        valid_detections.append((idx, conf, bbox))

    if not valid_detections:
        return None

    # 选择置信度最高的
    best = max(valid_detections, key=lambda x: x[1])
    return best[2]
```

### 3.3 可见性索引适配器

```python
def build_view_score_dict(
    obj_id: int,
    object_to_views: Dict[int, List[Tuple[int, float]]],
) -> Dict[int, float]:
    """将 List[(view_id, score)] 格式转为 {view_id: score} 字典

    处理当前 visibility_index 的实际格式。
    """
    views_list = object_to_views.get(obj_id, [])
    return {view_id: score for view_id, score in views_list}

def find_best_view_for_objects(
    objects: List[SceneObject],
    object_to_views: Dict[int, List[Tuple[int, float]]],
    min_visibility_score: float = 0.3,
) -> Optional[int]:
    """找到所有目标物体都可见且得分最高的视角

    策略:
    1. 取所有目标物体可见视角的交集
    2. 按平均可见性得分排序
    3. 返回得分最高的视角
    """
    if not objects:
        return None

    # 转换格式并获取高分视角集合
    view_score_dicts = []
    view_sets = []
    for obj in objects:
        scores = build_view_score_dict(obj.obj_id, object_to_views)
        view_score_dicts.append(scores)
        high_score_views = {v for v, s in scores.items() if s >= min_visibility_score}
        view_sets.append(high_score_views)

    # 取交集
    if not view_sets:
        return None
    common_views = set.intersection(*view_sets)

    if not common_views:
        return None

    # 按平均得分排序
    def avg_score(view_id):
        scores = [d.get(view_id, 0) for d in view_score_dicts]
        return sum(scores) / len(scores)

    return max(common_views, key=avg_score)
```

## 4. Prompt 设计 (匿名标记，不泄露元数据)

### 4.1 系统 Prompt

```
You are a spatial query generator for 3D scene understanding.

Your task:
1. Look at the image with RED BOUNDING BOXES highlighting specific objects (labeled A, B, C...)
2. Generate a natural language query that would uniquely identify the boxed object(s)
3. The query should be something a human might naturally ask to find these objects

Rules:
- The query MUST target the objects inside the RED BOXES
- Use spatial relations (on, near, next to, between) when needed to disambiguate
- Generate realistic, natural queries a human would ask
- Do NOT use the marker letters (A, B, C) in your query - use object descriptions
```

### 4.2 生成 Prompt (简化，只需 query)

```python
GENERATION_PROMPT = '''# Generate Query for Highlighted Objects

## Task
Objects marked with RED BOXES and letters ({markers}) are your targets.
Generate a natural language query to find these specific objects.

## Important
- Do NOT use the letters (A, B, C) in your query
- Describe objects by their appearance, type, or spatial relations
- Query should uniquely identify the marked object(s)

## Output Format (JSON only)
{{
  "query": "<natural language query to find the red-boxed objects>",
  "reasoning": "<why this query uniquely identifies the targets>"
}}

## Examples

Marked: A (a pillow on a sofa)
Output:
{{
  "query": "the throw pillow on the sofa",
  "reasoning": "Uses the sofa as anchor to identify the pillow"
}}

Marked: A (a lamp, only one in scene)
Output:
{{
  "query": "the table lamp",
  "reasoning": "Only one lamp in the scene, direct reference works"
}}

Marked: A, B (two pillows on a sofa)
Output:
{{
  "query": "the pillows on the sofa",
  "reasoning": "Plural form to identify both marked pillows"
}}
'''
```

**注意**:
- 不要求 LLM 输出 `anchor_obj_ids`, `difficulty`, `query_type`
- 这些字段在后处理中通过解析 query 推断

## 5. 后处理验证与推断

```python
from conceptgraph.query_scene.query_parser import QueryParser

def _create_parser(scene_categories: List[str]) -> QueryParser:
    """工厂函数: 集中初始化 QueryParser"""
    return QueryParser(
        llm_model="gemini-2.5-pro",
        scene_categories=scene_categories,
        use_pool=True,
    )

def _select_best_hypothesis(hypotheses: List) -> Optional[Any]:
    """按 kind 优先级选择最佳假设: direct > proxy > context"""
    priority = {"direct": 0, "proxy": 1, "context": 2}
    sorted_hypos = sorted(
        hypotheses,
        key=lambda h: priority.get(getattr(h, 'kind', 'context').lower(), 99)
    )
    return sorted_hypos[0] if sorted_hypos else None

def _find_anchor_obj_ids_in_view(
    anchor_categories: List[str],
    all_objects: List[SceneObject],
    source_view_id: int,
    object_to_views: Dict[int, List[Tuple[int, float]]],
    min_visibility: float = 0.3,
) -> List[int]:
    """View-local grounding: 只返回在 source_view_id 可见的 anchor 物体

    不再粗暴地返回所有同类物体，而是限定到同一视角可见的物体。
    """
    anchor_obj_ids = []
    for obj in all_objects:
        if obj.category not in anchor_categories:
            continue

        # 检查是否在 source_view_id 可见
        view_scores = build_view_score_dict(obj.obj_id, object_to_views)
        if source_view_id in view_scores and view_scores[source_view_id] >= min_visibility:
            anchor_obj_ids.append(obj.obj_id)

    return anchor_obj_ids

def post_process_case(
    raw_query: str,
    target_obj_ids: List[int],
    target_objects: List[SceneObject],
    all_objects: List[SceneObject],
    scene_categories: List[str],
    source_view_id: int,  # 必传参数，不再使用占位符
    source_frame_path: str,  # 必传参数
    object_to_views: Dict[int, List[Tuple[int, float]]],
) -> Tuple[EvaluationCase, List[str]]:
    """后处理：解析 query 推断 anchor/type/difficulty，验证合法性

    不依赖 LLM 输出的 anchor_obj_ids 等字段。
    """
    errors = []
    target_categories = [obj.category for obj in target_objects]

    # 1. 解析 query 获取结构化信息
    parser = _create_parser(scene_categories)

    try:
        hypothesis_output = parser.parse(raw_query)
        hypo = _select_best_hypothesis(hypothesis_output.hypotheses)
        if hypo is None:
            raise ValueError("No valid hypothesis found")
        grounding_query = hypo.grounding_query
    except Exception as e:
        errors.append(f"Query parse failed: {e}")
        # 返回最小化的用例
        return EvaluationCase(
            query=raw_query,
            target_obj_ids=target_obj_ids,
            target_categories=target_categories,
            query_type=QueryType.DIRECT,
            difficulty=QueryDifficulty.EASY,
            source_view_id=source_view_id,
            source_frame_path=source_frame_path,
            validated=False,
            validation_errors=errors,
        ), errors

    # 2. 提取 anchor 信息
    anchor_categories = []
    spatial_relation = None
    if grounding_query.root.spatial_constraints:
        sc = grounding_query.root.spatial_constraints[0]
        spatial_relation = sc.relation
        if sc.anchors:
            anchor_categories = sc.anchors[0].categories

    # 3. 查找 anchor_obj_ids (view-local grounding)
    anchor_obj_ids = _find_anchor_obj_ids_in_view(
        anchor_categories, all_objects, source_view_id, object_to_views
    )

    # 4. 验证 anchor 存在于场景中
    for cat in anchor_categories:
        if cat not in scene_categories and cat != "UNKNOW":
            errors.append(f"Anchor category '{cat}' not in scene")

    # 5. 推断 query_type (遍历完整 tree)
    query_type = infer_query_type_recursive(grounding_query.root)

    # 6. 推断 difficulty
    difficulty = infer_difficulty(
        target_objects, anchor_obj_ids, all_objects, scene_categories
    )

    # 7. 严格验证 target categories (完全匹配而非交集)
    parsed_target_cats = set(grounding_query.root.categories)
    expected_cats = set(target_categories)
    # 允许 UNKNOW 作为 wildcard
    if "UNKNOW" not in parsed_target_cats:
        if not parsed_target_cats.issubset(expected_cats | {"UNKNOW"}):
            errors.append(
                f"Parsed target '{parsed_target_cats}' not subset of GT '{expected_cats}'"
            )

    return EvaluationCase(
        query=raw_query,
        target_obj_ids=target_obj_ids,
        target_categories=target_categories,
        anchor_obj_ids=anchor_obj_ids,
        anchor_categories=anchor_categories,
        spatial_relation=spatial_relation,
        query_type=query_type,
        difficulty=difficulty,
        source_view_id=source_view_id,
        source_frame_path=source_frame_path,
        validated=len(errors) == 0,
        validation_errors=errors,
    ), errors

def infer_query_type_recursive(node) -> QueryType:
    """递归遍历 query tree 推断类型"""
    # 检查当前节点
    if getattr(node, 'select_constraint', None):
        return QueryType.SUPERLATIVE

    if getattr(node, 'spatial_constraints', None):
        # 递归检查 anchors
        for sc in node.spatial_constraints:
            if sc.anchors:
                for anchor in sc.anchors:
                    sub_type = infer_query_type_recursive(anchor)
                    if sub_type == QueryType.SUPERLATIVE:
                        return QueryType.SUPERLATIVE
        return QueryType.SPATIAL

    if getattr(node, 'attributes', None):
        return QueryType.ATTRIBUTE

    return QueryType.DIRECT

def infer_difficulty(
    target_objects: List[SceneObject],
    anchor_obj_ids: List[int],
    all_objects: List[SceneObject],
    scene_categories: List[str],
) -> QueryDifficulty:
    """根据场景复杂度推断难度"""
    target_cats = [obj.category for obj in target_objects]

    # 统计场景中同类物体数量
    same_category_count = sum(
        1 for obj in all_objects if obj.category in target_cats
    )

    # Easy: 只有一个同类物体
    if same_category_count == len(target_objects):
        return QueryDifficulty.EASY

    # Hard: 需要复杂空间关系或多跳
    if len(anchor_obj_ids) > 1 or same_category_count > 5:
        return QueryDifficulty.HARD

    # Medium: 需要简单空间关系
    return QueryDifficulty.MEDIUM
```

## 6. 多目标采样策略 (带 Fallback + Quota Tracking)

```python
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class SamplingQuota:
    """目标数量配额追踪器"""
    target_distribution: Dict[int, float]  # {1: 0.7, 2: 0.25, 3: 0.05}
    total_target: int
    generated: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def remaining(self) -> Dict[int, int]:
        """计算每种目标数量剩余配额"""
        result = {}
        for num_targets, ratio in self.target_distribution.items():
            expected = int(self.total_target * ratio)
            result[num_targets] = max(0, expected - self.generated[num_targets])
        return result

    def should_sample(self, num_targets: int) -> bool:
        """检查是否应该继续生成该目标数量的用例"""
        return self.remaining.get(num_targets, 0) > 0

    def record(self, actual_num_targets: int):
        """记录已生成的用例"""
        self.generated[actual_num_targets] += 1

    def next_target_count(self) -> Optional[int]:
        """智能选择下一个目标数量，优先填充未满配额的 bin"""
        remaining = self.remaining
        # 按剩余数量降序排列
        sorted_bins = sorted(remaining.items(), key=lambda x: -x[1])
        for num_targets, count in sorted_bins:
            if count > 0:
                return num_targets
        return None

def sample_target_objects_with_fallback(
    objects: List[SceneObject],
    object_to_views: Dict[int, List[Tuple[int, float]]],
    num_targets: int,
    max_attempts: int = 100,
    min_bbox_area: int = 500,
    min_conf: float = 0.3,
) -> Tuple[List[SceneObject], int, int]:
    """采样目标物体，带 fallback 策略

    策略:
    1. 尝试找到所有目标共同可见的视角
    2. 如果失败，降级到更少目标
    3. 最终 fallback 到单目标

    Returns:
        (采样的物体列表, 实际采样数量, 最佳视角 view_id)

    Raises:
        ValueError: 如果无法找到任何有效组合
    """
    # 过滤有效候选 (更严格的条件)
    valid_objects = []
    for obj in objects:
        if getattr(obj, 'is_background', False):
            continue
        if getattr(obj, 'num_detections', 0) < 3:
            continue
        if not getattr(obj, 'xyxy', []):
            continue

        # 检查是否有足够大且高置信度的检测
        has_valid_detection = False
        for i, bbox in enumerate(obj.xyxy):
            if i >= len(obj.conf):
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= min_bbox_area and obj.conf[i] >= min_conf:
                has_valid_detection = True
                break

        if has_valid_detection:
            valid_objects.append(obj)

    if not valid_objects:
        raise ValueError("No valid objects for sampling")

    # 尝试目标数量从 num_targets 递减
    for current_num in range(num_targets, 0, -1):
        if len(valid_objects) < current_num:
            continue

        for _ in range(max_attempts):
            targets = random.sample(valid_objects, current_num)
            view_id = find_best_view_for_objects(targets, object_to_views)
            if view_id is not None:
                return targets, current_num, view_id

    raise ValueError("Could not find any valid object combination with visible view")


class QueryCaseGenerator:
    """生成带 Ground Truth 的评估用例"""

    def __init__(
        self,
        scene_path: Path,
        temperature: float = 0.7,
        max_retries: int = 3,
        min_bbox_area: int = 500,
        max_workers: int = 4,  # 并行 worker 数量
    ):
        self.scene_path = Path(scene_path)
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_bbox_area = min_bbox_area
        self.max_workers = max_workers

        self._pool = GeminiClientPool.get_instance()
        self._load_scene()

    def _load_scene(self):
        """加载场景数据"""
        from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

        self.selector = KeyframeSelector.from_scene_path(
            str(self.scene_path),
            llm_model="gemini-2.5-pro",
        )
        self.objects = self.selector.objects
        self.object_to_views = self.selector.object_to_views
        self.scene_categories = list(set(obj.category for obj in self.objects))

    def generate_cases(
        self,
        num_cases: int = 100,
        target_distribution: dict = None,
    ) -> GenerationBatch:
        """生成指定数量的评估用例 (并行执行)"""
        if target_distribution is None:
            target_distribution = {1: 0.7, 2: 0.25, 3: 0.05}

        quota = SamplingQuota(
            target_distribution=target_distribution,
            total_target=num_cases,
        )

        cases = []
        failed = 0
        validation_passed = 0

        # 并行生成用例
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i in range(num_cases):
                # 智能选择目标数量
                num_targets = quota.next_target_count()
                if num_targets is None:
                    num_targets = 1  # fallback

                future = executor.submit(self._generate_single_case_safe, num_targets)
                futures.append((i, future))

            for i, future in futures:
                try:
                    case, actual_num = future.result()
                    cases.append(case)
                    quota.record(actual_num)

                    if case.validated:
                        validation_passed += 1

                    status = "✓" if case.validated else "⚠"
                    logger.info(
                        f"[{i+1}/{num_cases}] {status} '{case.query[:50]}...' "
                        f"-> {case.target_obj_ids}"
                    )

                except Exception as e:
                    logger.warning(f"[{i+1}/{num_cases}] Failed: {e}")
                    failed += 1

        return GenerationBatch(
            scene_name=self.scene_path.name,
            scene_path=str(self.scene_path.relative_to(self.scene_path.parent.parent)),
            cases=cases,
            generation_config={
                "temperature": self.temperature,
                "target_distribution": target_distribution,
                "min_bbox_area": self.min_bbox_area,
                "actual_distribution": dict(quota.generated),
            },
            total_generated=len(cases),
            failed_count=failed,
            validation_passed=validation_passed,
        )

    def _generate_single_case_safe(
        self, num_targets: int
    ) -> Tuple[EvaluationCase, int]:
        """线程安全的单用例生成 (带重试)"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                targets, actual_num, view_id = sample_target_objects_with_fallback(
                    self.objects,
                    self.object_to_views,
                    num_targets,
                    min_bbox_area=self.min_bbox_area,
                )
                case = self._generate_single_case(targets, view_id)
                return case, actual_num
            except Exception as e:
                last_error = e
                continue

        raise last_error or ValueError("Generation failed")

    def _generate_single_case(
        self, targets: List[SceneObject], view_id: int
    ) -> EvaluationCase:
        """为指定目标物体生成单个用例"""

        frame_path = self._get_frame_path_from_view(view_id)

        # 生成标注图像 (会验证 bbox)
        annotated_img, marker_map = annotate_image_with_targets(
            frame_path, targets, view_id, min_bbox_area=self.min_bbox_area
        )

        # 转为 base64
        img_data_url = self._image_to_data_url(annotated_img)

        # 构建 prompt
        markers = ", ".join(sorted(marker_map.keys()))
        prompt = GENERATION_PROMPT.format(markers=markers)

        # 调用 Gemini
        response = self._invoke_with_image(prompt, img_data_url)

        # 解析 LLM 响应
        raw_query = self._parse_llm_response(response)

        # 后处理验证 (传入必要参数)
        case, errors = post_process_case(
            raw_query=raw_query,
            target_obj_ids=[obj.obj_id for obj in targets],
            target_objects=targets,
            all_objects=self.objects,
            scene_categories=self.scene_categories,
            source_view_id=view_id,
            source_frame_path=str(frame_path.relative_to(self.scene_path)),
            object_to_views=self.object_to_views,
        )

        case.raw_llm_response = response
        case.generation_timestamp = datetime.now().isoformat()

        return case
```

## 7. 文件结构

```
conceptgraph/query_scene/
├── query_case_generator.py     # 核心生成器
├── image_annotator.py          # 图像标注工具
├── llm_evaluator.py            # 评估器 (已实现)
└── examples/
    ├── generate_eval_cases.py  # 生成评估用例脚本
    └── run_eval_with_gt.py     # 使用 GT 评估脚本
```

## 8. 使用方式

```bash
# 生成 100 个评估用例
python -m conceptgraph.query_scene.examples.generate_eval_cases \
    --scene room0 \
    --num_cases 100 \
    --output eval_cases_v1.json

# 使用 Ground Truth 评估
python -m conceptgraph.query_scene.examples.run_eval_with_gt \
    --scene room0 \
    --cases eval_cases_v1.json \
    --output eval_results_v1.json
```

## 9. 输出示例

```json
{
  "cases": [
    {
      "query": "the throw pillow on the sofa near the window",
      "target_obj_ids": [15],
      "anchor_obj_ids": [3, 42],
      "query_type": "spatial",
      "difficulty": "medium",
      "target_categories": ["throw_pillow"],
      "anchor_categories": ["sofa", "window"],
      "spatial_relation": "on",
      "source_view_id": 127,
      "source_frame_path": "/path/to/frame000127.jpg"
    }
  ],
  "summary": {
    "total": 100,
    "by_difficulty": {"easy": 30, "medium": 45, "hard": 25},
    "by_type": {"direct": 20, "spatial": 60, "attribute": 15, "multi": 5}
  }
}
```

## 10. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| bbox 不准确导致标注偏移 | 生成的 query 与 GT 不符 | 只选择高置信度检测 |
| 物体太小红框不明显 | Gemini 看不清目标 | 放大 bbox 区域或添加箭头 |
| 多物体共同视角难找 | 采样失败 | 允许单独视角拼接 |
| Gemini 生成的 anchor 与实际不符 | GT 中 anchor_obj_ids 错误 | 后处理验证 anchor 存在性 |

## 11. 实施步骤

### Phase 1: 图像标注模块
- [ ] 实现 `image_annotator.py`
- [ ] `annotate_image_with_targets()` 绘制红框
- [ ] `find_best_view_for_objects()` 找最佳视角

### Phase 2: 生成器核心
- [ ] 实现 `query_case_generator.py`
- [ ] `QueryCaseGenerator` 类
- [ ] Prompt 模板和解析

### Phase 3: 脚本与集成
- [ ] `generate_eval_cases.py` 生成脚本
- [ ] 更新 `llm_evaluator.py` 支持 GT 评估
- [ ] `run_eval_with_gt.py` 评估脚本
