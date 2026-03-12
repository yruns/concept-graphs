# LLMEvaluator v2 设计方案

> **目标**: 使用带 Ground Truth 的 EvaluationCase，评估 QueryParser 和 KeyframeSelector 的端到端质量。

## Codex Review Round 1 反馈修复

| 问题 | 修复方案 |
|------|----------|
| Parse 评估用 Gemini 增加主观性 | Parse 质量用代码精确计算，Gemini 仅做视觉评估 |
| 权重未考虑 anchor/spatial 缺失 | 动态归一化权重 |
| matched_obj_ids 泄露偏置评分 | 从 Gemini prompt 中移除 |
| GT red-box 锚定 Gemini 答案 | 分离两阶段：blind selector eval → GT diagnostic |
| 未处理多假设输出 | 添加假设选择策略 (direct > proxy > context) |
| 无失败情况处理 | 定义 fallback 逻辑和默认分数 |
| 数据模型缺字段 | 扩展 GT/trace 字段 |

## Codex Review Round 2 反馈修复

| 问题 | 修复方案 |
|------|----------|
| `bev_image_path` API 不匹配 | 从 `keyframe_result` 移除，改为独立生成 BEV |
| 假设选择示例不一致 | 示例改用 `select_hypothesis_for_evaluation()` |
| rank vs position 混淆 | 使用原始 `hypothesis.rank`，不重新计算 |
| 失败路径不可执行 | 提供完整可执行 `create_failure_result()` |
| `evaluation_status` 不一致 | 统一到 `EvaluationStatus` 枚举 |
| Gemini 输出 view_id/path 会幻觉 | 只输出 index，代码端映射 |
| 无 UNKNOW/别名统一策略 | 添加 `CategoryResolver` 模块 |
| selector 动态权重公式缺失 | 提供显式计算公式 |
| 无批量/重试/限流策略 | 添加 `BatchEvaluator` 配置 |

## 1. 评估流程 (两阶段分离)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Deterministic Parse Evaluation (代码计算，无 Gemini)            │
│   - 精确比对 parsed vs GT categories/relation                           │
│   - 输出: parse_score + parse_metrics                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ Stage 2: Blind Selector Evaluation (Gemini 视觉评估，无 GT 信息)         │
│   Input: query + parsed_hypothesis + selected_keyframes + BEV           │
│   注意: 不提供 GT、不提供 matched_obj_ids、不提供 GT source frame        │
│   - Gemini 独立判断: keyframes 是否展示了 query 描述的目标                │
│   - 输出: selector_score + per_frame_evals                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Stage 3: GT Comparison & Diagnostic (可选诊断模式)                       │
│   - 比对 matched_obj_ids vs gt_target_obj_ids                           │
│   - 计算 GT coverage                                                    │
│   - 生成诊断报告                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. 数据结构

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path

# ============================================================================
# Stage 1: Parse Evaluation (Deterministic)
# ============================================================================

class CategoryMatchResult(BaseModel):
    """类别匹配结果 (精确计算)"""
    gt_categories: List[str]
    parsed_categories: List[str]
    exact_matches: List[str]           # 精确匹配
    alias_matches: List[str]           # 别名匹配 (pillow <-> throw_pillow)
    missing_in_parsed: List[str]       # GT 有但解析缺失
    extra_in_parsed: List[str]         # 解析有但 GT 没有
    match_score: float                 # 0-1, IoU-style

class ParseMetrics(BaseModel):
    """Parse 评估指标 (代码精确计算，非 Gemini)"""
    target_match: CategoryMatchResult
    anchor_match: Optional[CategoryMatchResult] = None
    spatial_relation_correct: Optional[bool] = None
    hypothesis_kind: str               # direct/proxy/context
    hypothesis_rank: int               # 选择的是第几个假设

    # 综合分数 (动态权重)
    parse_score: float = Field(ge=0, le=10)
    weight_breakdown: Dict[str, float] = Field(default_factory=dict)

# ============================================================================
# Stage 2: Selector Evaluation (Gemini Visual)
# ============================================================================

class SelectorDimension(str, Enum):
    """Selector 评估维度 (Gemini 视觉评估)"""
    TARGET_VISIBILITY = "target_visibility"
    TARGET_COMPLETENESS = "target_completeness"
    SPATIAL_CONTEXT = "spatial_context"       # 空间关系是否可验证
    IMAGE_QUALITY = "image_quality"

class PerKeyframeEval(BaseModel):
    """单个 keyframe 的详细评估"""
    keyframe_idx: int
    view_id: int
    keyframe_path: str
    target_visibility: float = Field(ge=0, le=10)
    target_completeness: float = Field(ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)
    image_quality: float = Field(ge=0, le=10)
    observations: str

class SelectorEvaluation(BaseModel):
    """Selector 评估结果 (Gemini)"""
    # 维度分数 (动态归一化)
    target_visibility: float = Field(ge=0, le=10)
    target_completeness: float = Field(ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)
    image_quality: float = Field(ge=0, le=10)

    # 综合
    selector_score: float = Field(ge=0, le=10)
    best_keyframe_idx: int
    can_answer_query: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)

    # 每帧详情
    per_keyframe_evals: List[PerKeyframeEval] = Field(default_factory=list)

# ============================================================================
# Stage 3: GT Comparison (Diagnostic)
# ============================================================================

class GTComparison(BaseModel):
    """GT 比对结果 (诊断用)"""
    gt_target_obj_ids: List[int]
    matched_obj_ids: List[int]         # Selector 找到的
    gt_found: List[int]                # GT 中被找到的
    gt_missed: List[int]               # GT 中被遗漏的
    extra_matched: List[int]           # 找到但不在 GT 中的
    coverage: float                    # len(gt_found) / len(gt_target_obj_ids)

# ============================================================================
# Complete Input/Output
# ============================================================================

class EvaluationInputV2(BaseModel):
    """评估输入 v2"""

    # === 原始 Query ===
    query: str

    # === Ground Truth (来自 EvaluationCase) ===
    gt_target_obj_ids: List[int]
    gt_target_categories: List[str]
    gt_anchor_categories: List[str] = Field(default_factory=list)
    gt_spatial_relation: Optional[str] = None
    gt_source_view_id: int
    gt_source_frame_path: Path

    # === QueryParser 输出 ===
    parsed_target_categories: List[str]
    parsed_anchor_categories: List[str] = Field(default_factory=list)
    parsed_spatial_relation: Optional[str] = None
    hypothesis_kind: str
    hypothesis_rank: int = 1
    raw_hypothesis_json: str

    # === KeyframeSelector 输出 ===
    selected_keyframe_paths: List[Path]
    selected_view_ids: List[int]
    matched_obj_ids: List[int]

    # === BEV ===
    bev_image_path: Optional[Path] = None

    # === Evaluation Config ===
    enable_diagnostic_mode: bool = False  # 是否启用 GT 诊断模式

class EvaluationResultV2(BaseModel):
    """完整评估结果 v2"""
    query: str

    # === Stage 1: Parse (Deterministic) ===
    parse_metrics: ParseMetrics

    # === Stage 2: Selector (Gemini) ===
    selector_evaluation: SelectorEvaluation

    # === Stage 3: GT Comparison (Diagnostic) ===
    gt_comparison: Optional[GTComparison] = None

    # === Overall ===
    overall_score: float = Field(ge=0, le=10)
    suggestions: List[str] = Field(default_factory=list)

    # === Metadata ===
    raw_llm_response: str = ""
    model_name: str = "gemini-2.5-pro"
    prompt_version: str = "v2"
    timestamp: str = ""
    retry_count: int = 0

    # === Failure Handling ===
    evaluation_status: str = "success"  # success/parse_failed/selector_failed/llm_error
    error_message: Optional[str] = None
```

## 3. Stage 1: Deterministic Parse Evaluation

### 3.1 CategoryResolver (统一类别处理)

```python
class CategoryResolver:
    """统一类别解析，处理别名、UNKNOW、场景特定词汇"""

    # 标准别名映射
    ALIASES = {
        "pillow": ["throw_pillow", "cushion", "decorative_pillow"],
        "couch": ["sofa", "settee", "loveseat"],
        "lamp": ["table_lamp", "floor_lamp", "wall_sconce", "light"],
        "table": ["end_table", "side_table", "coffee_table", "c_table"],
        "chair": ["armchair", "dining_chair", "desk_chair", "office_chair"],
        "cabinet": ["cupboard", "dresser", "wardrobe", "chest"],
    }

    # 需要忽略的模糊类别
    IGNORE_CATEGORIES = {"UNKNOW", "unknown", "object", "thing", "item"}

    def __init__(self, scene_categories: Optional[Set[str]] = None):
        """
        Args:
            scene_categories: 场景中实际存在的类别集合，用于约束匹配
        """
        self.scene_categories = scene_categories or set()
        self._alias_lookup = self._build_alias_lookup()

    def _build_alias_lookup(self) -> Dict[str, str]:
        """构建反向别名查找表"""
        lookup = {}
        for base, aliases in self.ALIASES.items():
            lookup[self._normalize(base)] = base
            for alias in aliases:
                lookup[self._normalize(alias)] = base
        return lookup

    def _normalize(self, cat: str) -> str:
        """归一化类别名称"""
        return cat.lower().replace("_", " ").replace("-", " ").strip()

    def resolve(self, category: str) -> Optional[str]:
        """解析类别到标准形式，返回 None 表示应忽略"""
        norm = self._normalize(category)

        # 忽略模糊类别
        if norm in {self._normalize(c) for c in self.IGNORE_CATEGORIES}:
            return None

        # 查找别名
        if norm in self._alias_lookup:
            return self._alias_lookup[norm]

        # 保留原始类别
        return norm

    def resolve_list(self, categories: List[str]) -> List[str]:
        """解析类别列表，过滤无效类别"""
        resolved = []
        for cat in categories:
            r = self.resolve(cat)
            if r is not None and r not in resolved:
                resolved.append(r)
        return resolved
```

### 3.2 类别匹配计算

```python
def compute_category_match(
    gt_categories: List[str],
    parsed_categories: List[str],
    resolver: Optional[CategoryResolver] = None,
) -> CategoryMatchResult:
    """精确计算类别匹配 (IoU-style)"""

    resolver = resolver or CategoryResolver()

    # 解析并过滤
    gt_resolved = set(resolver.resolve_list(gt_categories))
    parsed_resolved = set(resolver.resolve_list(parsed_categories))

    # 精确匹配
    exact_matches = list(gt_resolved & parsed_resolved)

    # 别名匹配 (已在 resolve 阶段处理)
    alias_matches = []  # resolver 已统一

    # 计算 IoU
    intersection = len(exact_matches)
    union = len(gt_resolved | parsed_resolved)
    iou = intersection / union if union > 0 else 0.0

    return CategoryMatchResult(
        gt_categories=list(gt_categories),
        parsed_categories=list(parsed_categories),
        exact_matches=exact_matches,
        alias_matches=alias_matches,
        missing_in_parsed=list(gt_resolved - parsed_resolved),
        extra_in_parsed=list(parsed_resolved - gt_resolved),
        match_score=iou,
    )
```

### 3.3 Parse 综合分数计算

```python
def compute_parse_score(
    target_match: CategoryMatchResult,
    anchor_match: Optional[CategoryMatchResult],
    spatial_correct: Optional[bool],
) -> Tuple[float, Dict[str, float]]:
    """计算 Parse 综合分数 (动态权重归一化)"""

    # 基础权重
    weights = {
        "target": 0.5,      # 目标类别最重要
        "anchor": 0.3,      # 锚点类别
        "spatial": 0.2,     # 空间关系
    }

    scores = {"target": target_match.match_score * 10}
    active_weights = {"target": weights["target"]}

    if anchor_match is not None:
        scores["anchor"] = anchor_match.match_score * 10
        active_weights["anchor"] = weights["anchor"]

    if spatial_correct is not None:
        scores["spatial"] = 10.0 if spatial_correct else 0.0
        active_weights["spatial"] = weights["spatial"]

    # 归一化权重
    total_weight = sum(active_weights.values())
    normalized = {k: v / total_weight for k, v in active_weights.items()}

    # 加权平均
    parse_score = sum(scores[k] * normalized[k] for k in scores)

    return parse_score, normalized
```

## 4. Stage 2: Blind Selector Evaluation (Gemini)

### 4.1 Gemini 输入 (无 GT 信息泄露)

| 信息类型 | 展示 | 说明 |
|---------|------|------|
| **原始 Query** | ✅ | 评估目标 |
| **Parsed Hypothesis** | ✅ | 结构化解析结果 (categories, relation) |
| **Selected Keyframes** | ✅ | Selector 选择的帧 |
| **BEV 俯视图** | ✅ | 全局空间布局 |
| ~~GT target_obj_ids~~ | ❌ 移除 | 防止答案泄露 |
| ~~matched_obj_ids~~ | ❌ 移除 | 防止偏置评分 |
| ~~GT source frame~~ | ❌ 移除 | 防止锚定效应 |

### 4.2 Prompt 模板 (Blind Mode - 仅输出 index)

```python
BLIND_SELECTOR_PROMPT = '''# Keyframe Selection Evaluation (Blind Mode)

## Task
Evaluate whether the selected keyframes adequately support answering the given query.
You are evaluating the SELECTOR's choices, not the query parsing.

## Original Query
"{query}"

## Parsed Query Structure
The query was parsed as:
- **Target Categories**: {parsed_target_categories}
- **Anchor Categories**: {parsed_anchor_categories}
- **Spatial Relation**: {parsed_spatial_relation}
- **Hypothesis Kind**: {hypothesis_kind}

## Selected Keyframes
{num_keyframes} keyframes were selected (Images 1-{num_keyframes}).
{bev_note}

## Evaluation Dimensions

For EACH keyframe (by index 0 to {max_idx}), score:

1. **target_visibility** (0-10): Can you see objects matching the target categories?
   - Look for: {parsed_target_categories}

2. **target_completeness** (0-10): Are the target objects fully visible, not cropped/occluded?

3. **spatial_context** (0-10 or null): Can you verify the spatial relation "{parsed_spatial_relation}"?
   - Score null if no spatial relation in query

4. **image_quality** (0-10): Overall image quality for this evaluation task

## Important
- Focus on whether the keyframes SHOW the query targets, not whether parsing is correct
- Be strict: if you cannot clearly identify the target object, score low
- Consider if a human could use these frames to answer the query
- Output ONLY index numbers (0, 1, 2...), NOT file paths or view IDs

## Response Format (JSON only)
{{
  "per_keyframe_evals": [
    {{
      "keyframe_idx": 0,
      "target_visibility": <0-10>,
      "target_completeness": <0-10>,
      "spatial_context": <0-10 or null>,
      "image_quality": <0-10>,
      "observations": "<what you see>"
    }}
  ],
  "target_visibility": <avg across frames>,
  "target_completeness": <avg across frames>,
  "spatial_context": <avg or null>,
  "image_quality": <avg>,
  "selector_score": <weighted average>,
  "best_keyframe_idx": <index>,
  "can_answer_query": true/false,
  "reasoning": "<explanation>",
  "issues": ["issue1", ...]
}}
'''

BEV_NOTE_TEMPLATE = "The last image (Image {bev_idx}) is a Bird's Eye View showing the spatial layout."
NO_BEV_NOTE = "No BEV image provided."
```

### 4.3 Selector Score 动态权重计算

```python
def compute_selector_score(
    target_visibility: float,
    target_completeness: float,
    spatial_context: Optional[float],
    image_quality: float,
) -> float:
    """计算 Selector 综合分数 (动态权重归一化)

    权重配置 (spatial_context 存在时):
    - target_visibility: 35%
    - target_completeness: 25%
    - spatial_context: 25%
    - image_quality: 15%

    权重配置 (spatial_context 为 null 时):
    - target_visibility: 45%  (35 / 0.75)
    - target_completeness: 35% (25 / 0.75)
    - image_quality: 20%      (15 / 0.75)
    """
    if spatial_context is not None:
        return (
            0.35 * target_visibility +
            0.25 * target_completeness +
            0.25 * spatial_context +
            0.15 * image_quality
        )
    else:
        # 归一化到剩余维度
        return (
            0.45 * target_visibility +
            0.35 * target_completeness +
            0.20 * image_quality
        )
```

## 5. Stage 3: GT Comparison (Diagnostic)

```python
def compute_gt_comparison(
    gt_target_obj_ids: List[int],
    matched_obj_ids: List[int],
) -> GTComparison:
    """计算 GT 覆盖率 (纯代码计算)"""

    gt_set = set(gt_target_obj_ids)
    matched_set = set(matched_obj_ids)

    gt_found = list(gt_set & matched_set)
    gt_missed = list(gt_set - matched_set)
    extra_matched = list(matched_set - gt_set)

    coverage = len(gt_found) / len(gt_set) if gt_set else 0.0

    return GTComparison(
        gt_target_obj_ids=gt_target_obj_ids,
        matched_obj_ids=matched_obj_ids,
        gt_found=gt_found,
        gt_missed=gt_missed,
        extra_matched=extra_matched,
        coverage=coverage,
    )
```

## 6. 失败情况处理

```python
class EvaluationStatus(str, Enum):
    """统一的评估状态枚举"""
    SUCCESS = "success"
    PARSE_FAILED = "parse_failed"           # QueryParser 失败
    SELECTOR_EMPTY = "selector_empty"       # Selector 返回空
    IMAGE_LOAD_ERROR = "image_load_error"   # 图像加载失败
    LLM_ERROR = "llm_error"                 # Gemini 调用失败
    LLM_PARSE_ERROR = "llm_parse_error"     # Gemini 响应解析失败

def create_failure_result(
    query: str,
    status: EvaluationStatus,
    error_message: str,
    parse_metrics: Optional[ParseMetrics] = None,
) -> EvaluationResultV2:
    """创建失败结果 (完整可执行版本)"""
    from datetime import datetime

    # 失败时的默认空 CategoryMatchResult
    empty_match = CategoryMatchResult(
        gt_categories=[],
        parsed_categories=[],
        exact_matches=[],
        alias_matches=[],
        missing_in_parsed=[],
        extra_in_parsed=[],
        match_score=0.0,
    )

    # 失败时的默认 ParseMetrics
    default_parse = parse_metrics or ParseMetrics(
        target_match=empty_match,
        anchor_match=None,
        spatial_relation_correct=None,
        hypothesis_kind="unknown",
        hypothesis_rank=0,
        parse_score=0.0,
        weight_breakdown={},
    )

    # 失败时的默认 SelectorEvaluation
    default_selector = SelectorEvaluation(
        target_visibility=0.0,
        target_completeness=0.0,
        spatial_context=None,
        image_quality=0.0,
        selector_score=0.0,
        best_keyframe_idx=-1,
        can_answer_query=False,
        reasoning=f"Evaluation failed: {error_message}",
        issues=[error_message],
        per_keyframe_evals=[],
    )

    return EvaluationResultV2(
        query=query,
        parse_metrics=default_parse,
        selector_evaluation=default_selector,
        gt_comparison=None,
        overall_score=0.0,
        suggestions=[f"Fix: {error_message}"],
        raw_llm_response="",
        model_name="",
        prompt_version="v2",
        timestamp=datetime.now().isoformat(),
        retry_count=0,
        evaluation_status=status.value,
        error_message=error_message,
    )
```

## 7. Overall Score 计算

```python
def compute_overall_score(
    parse_score: float,
    selector_score: float,
    gt_coverage: Optional[float] = None,
) -> float:
    """计算综合分数

    权重:
    - parse_score: 30% (解析正确是前提)
    - selector_score: 50% (视觉展示是核心)
    - gt_coverage: 20% (GT 覆盖率作为额外校验)
    """
    if gt_coverage is not None:
        return (
            0.30 * parse_score +
            0.50 * selector_score +
            0.20 * (gt_coverage * 10)
        )
    else:
        # 无 GT 比对时，归一化到 parse + selector
        return 0.375 * parse_score + 0.625 * selector_score
```

## 8. 假设选择策略

```python
def select_hypothesis_for_evaluation(
    hypotheses: List,
) -> Tuple[Any, int, str]:
    """选择用于评估的假设

    策略: direct > proxy > context, 同 kind 取最低 rank
    返回: (hypothesis, original_rank, kind)
    """
    if not hypotheses:
        return None, 0, "unknown"

    priority = {"direct": 0, "proxy": 1, "context": 2}

    def sort_key(h):
        kind = getattr(h, "kind", "context")
        if hasattr(kind, "value"):
            kind = kind.value
        # 使用原始 rank，不重新计算
        rank = getattr(h, "rank", 99) or 99
        return (priority.get(str(kind).lower(), 99), rank)

    sorted_hypos = sorted(hypotheses, key=sort_key)
    best = sorted_hypos[0]

    kind = getattr(best, "kind", "context")
    if hasattr(kind, "value"):
        kind = kind.value

    # 返回原始 rank，不是列表位置
    original_rank = getattr(best, "rank", 1) or 1

    return best, original_rank, str(kind).lower()
```

## 9. 批量评估与限流配置

```python
@dataclass
class BatchEvaluatorConfig:
    """批量评估配置"""
    max_workers: int = 4                    # 并行 worker 数
    per_case_timeout: int = 120             # 单 case 超时 (秒)
    max_retries: int = 3                    # 单 case 最大重试
    retry_backoff_base: float = 2.0         # 退避基数 (秒)
    retry_backoff_max: float = 60.0         # 最大退避 (秒)
    include_bev: bool = True                # 是否包含 BEV 图
    bev_fallback_on_missing: bool = True    # BEV 缺失时继续评估

class BatchEvaluator:
    """批量评估器，集成 GeminiClientPool 限流"""

    def __init__(self, config: BatchEvaluatorConfig):
        self.config = config
        self._pool = GeminiClientPool.get_instance()

    def evaluate_batch(
        self,
        inputs: List[EvaluationInputV2],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EvaluationResultV2]:
        """批量评估，带进度回调"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_with_retry, inp): i
                for i, inp in enumerate(inputs)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.config.per_case_timeout)
                except TimeoutError:
                    result = create_failure_result(
                        inputs[idx].query,
                        EvaluationStatus.LLM_ERROR,
                        "Evaluation timeout",
                    )
                except Exception as e:
                    result = create_failure_result(
                        inputs[idx].query,
                        EvaluationStatus.LLM_ERROR,
                        str(e),
                    )

                results.append((idx, result))

                if progress_callback:
                    progress_callback(len(results), len(inputs))

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _evaluate_with_retry(
        self,
        input_: EvaluationInputV2,
    ) -> EvaluationResultV2:
        """带重试的单 case 评估"""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return self._evaluate_single(input_)

            except RateLimitError as e:
                last_error = e
                backoff = min(
                    self.config.retry_backoff_base ** attempt,
                    self.config.retry_backoff_max,
                )
                time.sleep(backoff)

            except Exception as e:
                last_error = e
                break  # 非限流错误不重试

        return create_failure_result(
            input_.query,
            EvaluationStatus.LLM_ERROR,
            f"Max retries exceeded: {last_error}",
        )
```

## 10. 完整性校验

```python
def validate_evaluation_input(input_: EvaluationInputV2) -> List[str]:
    """校验评估输入的完整性"""
    errors = []

    # keyframe 数量校验
    if len(input_.selected_keyframe_paths) != len(input_.selected_view_ids):
        errors.append(
            f"Keyframe count mismatch: {len(input_.selected_keyframe_paths)} paths "
            f"vs {len(input_.selected_view_ids)} view_ids"
        )

    # keyframe 路径存在性
    for i, path in enumerate(input_.selected_keyframe_paths):
        if not Path(path).exists():
            errors.append(f"Keyframe {i} not found: {path}")

    # BEV 路径
    if input_.bev_image_path and not Path(input_.bev_image_path).exists():
        errors.append(f"BEV image not found: {input_.bev_image_path}")

    # GT 数据完整性
    if not input_.gt_target_obj_ids:
        errors.append("gt_target_obj_ids is empty")
    if not input_.gt_target_categories:
        errors.append("gt_target_categories is empty")

    return errors


def validate_llm_response(
    response: Dict,
    num_keyframes: int,
) -> List[str]:
    """校验 LLM 响应的格式"""
    errors = []

    # 必需字段
    required = ["per_keyframe_evals", "selector_score", "best_keyframe_idx", "can_answer_query"]
    for field in required:
        if field not in response:
            errors.append(f"Missing required field: {field}")

    # per_keyframe_evals 数量
    evals = response.get("per_keyframe_evals", [])
    if len(evals) != num_keyframes:
        errors.append(
            f"per_keyframe_evals count {len(evals)} != expected {num_keyframes}"
        )

    # best_keyframe_idx 范围
    best_idx = response.get("best_keyframe_idx", -1)
    if best_idx < 0 or best_idx >= num_keyframes:
        errors.append(f"best_keyframe_idx {best_idx} out of range [0, {num_keyframes})")

    # 分数范围
    for i, ev in enumerate(evals):
        for key in ["target_visibility", "target_completeness", "image_quality"]:
            val = ev.get(key)
            if val is not None and (val < 0 or val > 10):
                errors.append(f"Keyframe {i} {key}={val} out of range [0, 10]")

    return errors
```

## 11. 使用流程

```python
from conceptgraph.query_scene.llm_evaluator_v2 import LLMEvaluatorV2, BatchEvaluatorConfig
from conceptgraph.query_scene.query_case_generator import EvaluationCase, GenerationBatch
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

# 1. 加载 Ground Truth cases
with open("eval_cases.json") as f:
    batch = GenerationBatch.model_validate_json(f.read())

# 2. 初始化评估器
config = BatchEvaluatorConfig(
    max_workers=4,
    per_case_timeout=120,
    include_bev=True,
)
evaluator = LLMEvaluatorV2(config)
selector = KeyframeSelector(scene_path)

# 3. 对每个 case 运行 pipeline
results = []

for case in batch.cases:
    # 运行 QueryParser
    hypothesis_output = parser.parse(case.query)

    # 选择最佳假设
    best_hypo, hypo_rank, hypo_kind = select_hypothesis_for_evaluation(
        hypothesis_output.hypotheses
    )

    # 运行 KeyframeSelector
    keyframe_result = selector.select_keyframes_v2(case.query)

    # 构建评估输入 (BEV 独立生成，不依赖 keyframe_result)
    bev_path = generate_bev_image(scene_path, keyframe_result.target_objects)

    eval_input = EvaluationInputV2(
        query=case.query,
        gt_target_obj_ids=case.target_obj_ids,
        gt_target_categories=case.target_categories,
        gt_anchor_categories=case.anchor_categories,
        gt_spatial_relation=case.spatial_relation,
        gt_source_view_id=case.source_view_id,
        gt_source_frame_path=scene_path / case.source_frame_path,
        parsed_target_categories=best_hypo.grounding_query.root.categories,
        parsed_anchor_categories=extract_anchor_categories(best_hypo),
        parsed_spatial_relation=extract_spatial_relation(best_hypo),
        hypothesis_kind=hypo_kind,
        hypothesis_rank=hypo_rank,
        raw_hypothesis_json=hypothesis_output.model_dump_json(),
        selected_keyframe_paths=keyframe_result.keyframe_paths,
        selected_view_ids=keyframe_result.keyframe_indices,
        matched_obj_ids=[obj.obj_id for obj in keyframe_result.target_objects],
        bev_image_path=bev_path,
    )

    # 校验输入
    validation_errors = validate_evaluation_input(eval_input)
    if validation_errors:
        logger.warning(f"Validation errors for {case.query}: {validation_errors}")

    results.append(eval_input)

# 4. 批量评估
eval_results = evaluator.evaluate_batch(results)

# 5. 生成报告
report = evaluator.generate_report(eval_results)
```

## 12. 输出报告示例

```json
{
  "summary": {
    "total_cases": 100,
    "avg_parse_score": 8.2,
    "avg_selector_score": 7.5,
    "avg_overall_score": 7.8,
    "gt_coverage": 0.85
  },
  "by_query_type": {
    "direct": {"count": 30, "avg_score": 8.5},
    "spatial": {"count": 50, "avg_score": 7.2},
    "attribute": {"count": 15, "avg_score": 7.8},
    "superlative": {"count": 5, "avg_score": 6.5}
  },
  "by_difficulty": {
    "easy": {"count": 40, "avg_score": 8.8},
    "medium": {"count": 35, "avg_score": 7.5},
    "hard": {"count": 25, "avg_score": 6.2}
  },
  "common_issues": {
    "parse": [
      "Anchor category mismatch (15 cases)",
      "Spatial relation not captured (8 cases)"
    ],
    "selector": [
      "Target partially occluded (12 cases)",
      "Anchor not visible in selected frames (10 cases)"
    ]
  }
}
```

## 13. 实施步骤

### Phase 1: 核心评估器
- [ ] 创建 `llm_evaluator_v2.py`
- [ ] 实现 `CategoryResolver` 类别解析器
- [ ] 实现 `EvaluationInputV2` / `EvaluationResultV2` 数据结构
- [ ] 实现 `validate_evaluation_input()` 输入校验
- [ ] 实现 prompt 构建

### Phase 2: 评估逻辑
- [ ] 实现 Stage 1: `compute_category_match()` + `compute_parse_score()`
- [ ] 实现 Stage 2: `evaluate_selector_blind()` + `compute_selector_score()`
- [ ] 实现 `validate_llm_response()` 响应校验
- [ ] 实现 `create_failure_result()` 失败处理

### Phase 3: 批量评估
- [ ] 实现 `BatchEvaluatorConfig` 配置
- [ ] 实现 `BatchEvaluator` 带限流和重试
- [ ] 实现 `select_hypothesis_for_evaluation()` 假设选择

### Phase 4: 集成脚本
- [ ] 创建 `run_eval_with_gt.py`
- [ ] 实现批量评估流程
- [ ] 实现报告生成
