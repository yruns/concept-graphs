# LLMEvaluator 实现计划 v2

> **设计原则**: 不考虑成本，只考虑效果；无任何 fallback 逻辑，失败即抛出异常。

## 1. 目标

创建一个基于 Gemini 视觉能力的 LLMEvaluator，用于自动评估 KeyframeSelector 选择的关键帧质量。

## 2. 评估维度（动态权重）

| 维度 | 基础权重 | 适用条件 | 说明 |
|------|----------|----------|------|
| **target_visibility** | 30% | 始终 | 目标对象是否清晰可见 |
| **target_completeness** | 20% | 始终 | 目标对象是否完整（无截断/遮挡） |
| **spatial_context** | 20% | 有空间关系时 | 空间关系是否可验证 |
| **anchor_visibility** | 15% | 有锚点时 | 锚点对象是否可见 |
| **image_quality** | 15% | 始终 | 图像整体质量（亮度、清晰度、角度） |

**动态归一化**: 当 `spatial_relation` 或 `anchor_categories` 为空时，对应维度标记为 `null`，剩余维度权重归一化到 1.0。

## 3. 数据结构设计（Pydantic 强类型）

```python
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path

class HypothesisKind(str, Enum):
    DIRECT = "direct"
    PROXY = "proxy"
    CONTEXT = "context"

class DimensionName(str, Enum):
    TARGET_VISIBILITY = "target_visibility"
    TARGET_COMPLETENESS = "target_completeness"
    SPATIAL_CONTEXT = "spatial_context"
    ANCHOR_VISIBILITY = "anchor_visibility"
    IMAGE_QUALITY = "image_quality"

class EvaluationInput(BaseModel):
    """评估输入（Pydantic 验证）"""
    query: str
    keyframe_paths: List[Path]
    target_categories: List[str]
    anchor_categories: List[str] = Field(default_factory=list)
    spatial_relation: Optional[str] = None
    hypothesis_kind: HypothesisKind
    matched_object_count: int
    bev_image_path: Optional[Path] = None

    # 追踪信息
    view_ids: List[int] = Field(default_factory=list)  # 对应 keyframe 的 view_id
    resolved_frame_ids: List[int] = Field(default_factory=list)  # 实际帧号

class FrameEvaluation(BaseModel):
    """单帧评估结果"""
    frame_idx: int
    view_id: Optional[int] = None
    frame_path: str
    target_visibility: float = Field(ge=0, le=10)
    target_completeness: float = Field(ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)  # 可为 null
    anchor_visibility: Optional[float] = Field(default=None, ge=0, le=10)  # 可为 null
    image_quality: float = Field(ge=0, le=10)
    observations: str

def validate_frame_evaluations(
    evaluations: List[FrameEvaluation],
    has_anchor: bool,
    has_spatial: bool,
) -> None:
    """验证 FrameEvaluation 的 null 字段是否符合预期

    规则:
    - 有锚点 (has_anchor=True): anchor_visibility 必须是 float
    - 无锚点 (has_anchor=False): anchor_visibility 必须是 null
    - 有空间关系 (has_spatial=True): spatial_context 必须是 float
    - 无空间关系 (has_spatial=False): spatial_context 必须是 null

    违反规则直接抛出 ValueError。
    """
    for i, fe in enumerate(evaluations):
        # 锚点可见性校验
        if has_anchor:
            if fe.anchor_visibility is None:
                raise ValueError(
                    f"Frame {i}: anchor_visibility must not be null "
                    f"when anchor_categories is provided"
                )
        else:
            if fe.anchor_visibility is not None:
                raise ValueError(
                    f"Frame {i}: anchor_visibility must be null "
                    f"when no anchor_categories (got {fe.anchor_visibility})"
                )

        # 空间关系校验
        if has_spatial:
            if fe.spatial_context is None:
                raise ValueError(
                    f"Frame {i}: spatial_context must not be null "
                    f"when spatial_relation is provided"
                )
        else:
            if fe.spatial_context is not None:
                raise ValueError(
                    f"Frame {i}: spatial_context must be null "
                    f"when no spatial_relation (got {fe.spatial_context})"
                )

class OverallAssessment(BaseModel):
    """整体评估结果"""
    best_frame_idx: int
    overall_score: float = Field(ge=0, le=10)
    can_answer_query: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

class LLMEvaluationResponse(BaseModel):
    """LLM 返回的完整响应结构（用于严格解析）"""
    per_frame_evaluations: List[FrameEvaluation]
    overall_assessment: OverallAssessment

class EvaluationResult(BaseModel):
    """最终评估结果"""
    query: str
    overall_score: float
    dimension_scores: dict[DimensionName, Optional[float]]
    reasoning: str
    issues: List[str]
    suggestions: List[str]
    per_frame_evaluations: List[FrameEvaluation]
    best_frame_idx: int
    raw_llm_response: str

    # 元数据
    model_name: str
    prompt_version: str = "v2"
    timestamp: str
    retry_count: int = 0
```

## 4. Gemini 应看到的信息

| 信息类型 | 展示方式 | 理由 |
|---------|---------|------|
| **关键帧图像** | base64 data URL | 核心评估对象 |
| **BEV 俯视图** | base64 data URL（单独标注） | 展示空间布局，但不计入帧评分 |
| **原始查询** | 文本 | 评估目标 |
| **目标类别列表** | 文本 | 明确要找什么 |
| **锚点类别** | 文本（如有） | 验证空间关系 |
| **空间关系** | 文本 | 验证关系是否可见 |
| **帧标识** | `Frame 1 (view_id=42)` | 追踪评估结果 |

**不展示**: 匹配对象数量、3D 坐标、hypothesis_kind（避免偏见）

## 5. 图像处理流程

```python
def _encode_image_to_data_url(self, image_path: Path) -> str:
    """将图像转换为 base64 data URL

    流程: Path -> PIL.Image -> Resize(max_edge=1024) -> JPEG(quality=90) -> base64
    失败时直接抛出异常，不做任何 fallback。

    使用 context manager 确保文件正确关闭。
    """
    from PIL import Image
    import base64
    import io

    # 使用 context manager 打开图像
    with Image.open(image_path) as img:
        # Resize: 保持宽高比，最大边 1024px（高质量优先）
        max_edge = 1024
        if max(img.size) > max_edge:
            ratio = max_edge / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to RGB (handle RGBA/grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Encode to JPEG base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}"
```

## 6. GeminiClientPool 集成（完整 Key Rotation v3）

```python
from conceptgraph.utils.llm_client import GeminiClientPool, _is_rate_limit_error
from langchain_core.messages import HumanMessage
import time

class LLMEvaluator:
    def __init__(
        self,
        temperature: float = 0.1,  # 极低温度保证一致性
        timeout: int = 180,
        max_rounds: int = 5,  # 最多重试 5 轮（每轮尝试所有 key）
    ):
        self._pool = GeminiClientPool.get_instance()
        self.temperature = temperature
        self.timeout = timeout
        self.max_rounds = max_rounds

    def _invoke_with_images(
        self,
        prompt: str,
        image_data_urls: List[str],
    ) -> str:
        """调用 Gemini 进行视觉评估，带完整 key rotation 重试

        重试策略（嵌套循环）:
        - 外层: max_rounds 轮
        - 内层: 每轮依次尝试 pool 中的所有 key
        - 每轮结束后等待指数退避时间（30s, 60s, 120s...）
        """
        pool_size = self._pool.pool_size
        last_error = None

        for round_idx in range(self.max_rounds):
            # 每轮结束后等待（第一轮不等待）
            if round_idx > 0:
                wait_time = min(30 * (2 ** (round_idx - 1)), 300)  # 30s, 60s, 120s, 240s, 300s
                time.sleep(wait_time)

            # 内层循环：尝试所有 key
            tried_in_round = set()
            attempts_in_round = 0

            while attempts_in_round < pool_size * 2:  # 防止无限循环
                client, config_idx = self._pool.get_next_client(
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                attempts_in_round += 1

                # 本轮已尝试过此 key，跳过
                if config_idx in tried_in_round:
                    continue
                tried_in_round.add(config_idx)

                try:
                    # 构建 multimodal message
                    content_parts = [{"type": "text", "text": prompt}]
                    for img_url in image_data_urls:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": img_url}
                        })

                    messages = [HumanMessage(content=content_parts)]
                    result = client.invoke(messages)

                    self._pool.record_request(config_idx, rate_limited=False)

                    # 安全提取 content（处理 string 或 list）
                    return self._extract_content(result.content)

                except Exception as e:
                    if _is_rate_limit_error(e):
                        self._pool.record_request(config_idx, rate_limited=True)
                        last_error = e
                        # 继续尝试下一个 key
                    else:
                        # 非 rate limit 错误也记录，然后直接抛出
                        self._pool.record_request(config_idx, rate_limited=False)
                        raise

                # 本轮所有 key 都尝试完？
                if len(tried_in_round) >= pool_size:
                    break

        # 所有轮次重试失败
        raise RuntimeError(
            f"All {self.max_rounds} rounds × {pool_size} keys exhausted. "
            f"Last error: {last_error}"
        )

    def _extract_content(self, content: Any) -> str:
        """安全提取 LLM 响应内容为字符串

        处理多种 content 格式:
        - str: 直接返回
        - list[dict]: 拼接所有 text 块
        - 其他: 转为字符串
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif isinstance(block, str):
                    texts.append(block)
                elif hasattr(block, "text"):
                    texts.append(block.text)
            return "".join(texts)
        else:
            return str(content)
```

## 7. Prompt 设计 v2

```python
EVALUATION_PROMPT_TEMPLATE = '''# Keyframe Quality Evaluation

## Task
Evaluate how well the selected keyframes support answering this spatial query.

## Query
"{query}"

## Target Information
- **Target categories to find**: {target_categories}
{anchor_section}
{spatial_section}

## Images
{image_descriptions}

## Evaluation Rubric

For EACH keyframe, score these dimensions (0-10):

### Always Evaluate:
1. **target_visibility**: Can you clearly see object(s) matching the target categories?
   - 0-2: Target not visible or wrong object type
   - 3-4: Partially visible, heavily occluded
   - 5-6: Visible but with issues (small, blurry, edge of frame)
   - 7-8: Clearly visible, minor issues
   - 9-10: Perfectly clear view of target

2. **target_completeness**: Is the target object fully visible without cropping?
   - 0-2: Mostly cropped or occluded
   - 3-4: Significant portion missing
   - 5-6: Minor cropping
   - 7-8: Nearly complete
   - 9-10: Fully visible, no occlusion

3. **image_quality**: Overall image quality
   - Score based on lighting, focus, angle, noise

{conditional_dimensions}

## Important Instructions
- Be strict and objective in scoring
- Look carefully for small objects that match target categories
- Verify spatial relationships by examining relative positions
- Score `spatial_context` and `anchor_visibility` as `null` if no anchor/spatial relation exists

## Response Format (JSON only, no markdown)
{{
  "per_frame_evaluations": [
    {{
      "frame_idx": 0,
      "view_id": <view_id or null>,
      "frame_path": "<path>",
      "target_visibility": <0-10>,
      "target_completeness": <0-10>,
      "spatial_context": <0-10 or null>,
      "anchor_visibility": <0-10 or null>,
      "image_quality": <0-10>,
      "observations": "<detailed description of what you see>"
    }}
  ],
  "overall_assessment": {{
    "best_frame_idx": <index of best frame>,
    "overall_score": <weighted average 0-10>,
    "can_answer_query": <true/false>,
    "reasoning": "<explanation of scores>",
    "issues": ["<issue1>", "<issue2>"],
    "suggestions": ["<suggestion1>"]
  }}
}}'''

ANCHOR_SECTION_TEMPLATE = "- **Anchor categories**: {anchor_categories}"
SPATIAL_SECTION_TEMPLATE = "- **Spatial relation to verify**: \"{spatial_relation}\""

CONDITIONAL_DIMENSIONS_WITH_ANCHOR = '''
### Conditional (when anchor/spatial relation exists):
4. **spatial_context**: Can you verify the spatial relationship between target and anchor?
   - Score how clearly the relationship "{spatial_relation}" is visible

5. **anchor_visibility**: Can you see the anchor object ({anchor_categories})?
   - Score how clearly the anchor is visible'''

CONDITIONAL_DIMENSIONS_WITHOUT_ANCHOR = '''
### Note
Since this query has no anchor object or spatial relation, score `spatial_context` and `anchor_visibility` as `null`.'''
```

## 8. 响应解析（严格 Pydantic，无 fallback）

```python
import json

def _parse_llm_response(
    self,
    response: str,
    input: EvaluationInput,
) -> EvaluationResult:
    """解析 LLM 响应，使用 Pydantic 严格验证 + 业务规则校验

    失败直接抛出异常，不做任何 fallback 或正则提取。
    """
    # 清理可能的 markdown 包装
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 严格 JSON 解析
    data = json.loads(cleaned)

    # Pydantic 验证
    llm_response = LLMEvaluationResponse.model_validate(data)

    # 业务规则校验：null 字段必须符合预期
    has_anchor = bool(input.anchor_categories)
    has_spatial = bool(input.spatial_relation)
    validate_frame_evaluations(
        llm_response.per_frame_evaluations,
        has_anchor=has_anchor,
        has_spatial=has_spatial,
    )

    # 计算动态权重加权平均
    dimension_scores = self._compute_dimension_scores(
        llm_response.per_frame_evaluations,
        has_anchor=has_anchor,
        has_spatial=has_spatial,
    )

    return EvaluationResult(
        query=input.query,
        overall_score=llm_response.overall_assessment.overall_score,
        dimension_scores=dimension_scores,
        reasoning=llm_response.overall_assessment.reasoning,
        issues=llm_response.overall_assessment.issues,
        suggestions=llm_response.overall_assessment.suggestions,
        per_frame_evaluations=llm_response.per_frame_evaluations,
        best_frame_idx=llm_response.overall_assessment.best_frame_idx,
        raw_llm_response=response,
        model_name="gemini-3-pro",
        timestamp=datetime.now().isoformat(),
    )

def _compute_dimension_scores(
    self,
    frame_evals: List[FrameEvaluation],
    has_anchor: bool,
    has_spatial: bool,
) -> dict[DimensionName, Optional[float]]:
    """计算各维度平均分，动态归一化权重"""

    # 收集各维度分数
    scores = {
        DimensionName.TARGET_VISIBILITY: [],
        DimensionName.TARGET_COMPLETENESS: [],
        DimensionName.SPATIAL_CONTEXT: [],
        DimensionName.ANCHOR_VISIBILITY: [],
        DimensionName.IMAGE_QUALITY: [],
    }

    for fe in frame_evals:
        scores[DimensionName.TARGET_VISIBILITY].append(fe.target_visibility)
        scores[DimensionName.TARGET_COMPLETENESS].append(fe.target_completeness)
        if fe.spatial_context is not None:
            scores[DimensionName.SPATIAL_CONTEXT].append(fe.spatial_context)
        if fe.anchor_visibility is not None:
            scores[DimensionName.ANCHOR_VISIBILITY].append(fe.anchor_visibility)
        scores[DimensionName.IMAGE_QUALITY].append(fe.image_quality)

    # 计算平均分
    result = {}
    for dim, vals in scores.items():
        if vals:
            result[dim] = sum(vals) / len(vals)
        else:
            result[dim] = None

    return result
```

## 9. 评估模式

### 9.1 keyframe_only（默认）
只评估关键帧本身的质量，不提供 BEV。

### 9.2 with_bev_context（诊断模式）
同时提供 BEV 图像帮助理解空间布局，但 BEV 的作用是辅助验证，不计入帧评分。

```python
def evaluate_single(
    self,
    input: EvaluationInput,
    mode: str = "keyframe_only",  # or "with_bev_context"
) -> EvaluationResult:
    """评估单个查询的关键帧选择"""

    # 验证输入
    if not input.keyframe_paths:
        raise ValueError("keyframe_paths cannot be empty")
    for path in input.keyframe_paths:
        if not path.exists():
            raise FileNotFoundError(f"Keyframe not found: {path}")

    # 编码图像
    image_data_urls = []
    for path in input.keyframe_paths:
        image_data_urls.append(self._encode_image_to_data_url(path))

    # BEV 图像（诊断模式）
    bev_data_url = None
    if mode == "with_bev_context" and input.bev_image_path:
        if not input.bev_image_path.exists():
            raise FileNotFoundError(f"BEV image not found: {input.bev_image_path}")
        bev_data_url = self._encode_image_to_data_url(input.bev_image_path)

    # 构建 prompt
    prompt = self._build_evaluation_prompt(input, mode, bev_data_url)

    # 调用 LLM
    response = self._invoke_with_images(prompt, image_data_urls)

    # 解析结果
    return self._parse_llm_response(response, input)
```

## 10. 实施步骤

### Phase 1: 核心实现 (Day 1)
- [ ] 创建 `llm_evaluator.py`
- [ ] 实现 Pydantic 数据模型
- [ ] 实现图像编码管道 `_encode_image_to_data_url`
- [ ] 实现 key rotation 调用逻辑 `_invoke_with_images`
- [ ] 实现 prompt 构建 `_build_evaluation_prompt`
- [ ] 实现严格解析 `_parse_llm_response`

### Phase 2: 集成测试 (Day 1-2)
- [ ] 创建 `examples/run_llm_evaluation.py`
- [ ] 在 room0 上运行 5 个样本
- [ ] 验证各维度评分合理性
- [ ] 测试 rate limit 重试逻辑

### Phase 3: 完整评估 (Day 2-3)
- [ ] 与 `e2e_query_test.py` 集成
- [ ] 批量评估支持 `evaluate_batch`
- [ ] 生成评估报告 JSON
- [ ] 添加 HTML 可视化报告

## 11. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| JSON 解析失败 | 评估中断 | 保存 raw response 用于调试，直接抛出详细异常 |
| Rate limit | 评估中断 | 5 key rotation + 指数退避，最长等待 5 分钟 |
| 小对象识别差 | 评分偏低 | Prompt 强调仔细观察小物体 |
| 评分不一致 | 结果波动 | temperature=0.1 + 详细 rubric |
| 图像加载失败 | 评估中断 | 直接抛出 FileNotFoundError |

## 12. 与现有代码的集成点

```python
# 从 KeyframeResult 构建 EvaluationInput
def from_keyframe_result(
    result: KeyframeResult,
    query: str,
    hypothesis: HypothesisOutputV1,
) -> EvaluationInput:
    """从 KeyframeSelector 结果构建评估输入"""
    return EvaluationInput(
        query=query,
        keyframe_paths=result.keyframe_paths,
        target_categories=hypothesis.hypotheses[0].query.target.categories,
        anchor_categories=_extract_anchor_categories(hypothesis),
        spatial_relation=_extract_spatial_relation(hypothesis),
        hypothesis_kind=HypothesisKind(hypothesis.hypotheses[0].kind.value),
        matched_object_count=len(result.matched_object_ids),
        view_ids=result.view_ids,
        resolved_frame_ids=result.resolved_frame_ids,
        bev_image_path=result.bev_image_path if hasattr(result, 'bev_image_path') else None,
    )
```

## 13. Codex Review Round 1 反馈总结

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| 图像必须转为 base64 data URL | ✅ 已修复 | 添加 `_encode_image_to_data_url` |
| 缺少 key rotation 重试 | ✅ 已修复 | 完整实现 `_invoke_with_images` |
| 固定权重对无锚点查询不公平 | ✅ 已修复 | 动态归一化 + null 处理 |
| 弱类型设计 | ✅ 已修复 | 全面使用 Pydantic + Enum |
| per_frame_scores 缺 frame_id | ✅ 已修复 | `FrameEvaluation` 包含 view_id/frame_path |
| JSON fallback 脆弱 | ✅ 已移除 | 严格 Pydantic 验证，失败即异常 |
| BEV 可能影响评分 | ✅ 已修复 | 分离两种模式 |
| 缺少元数据 | ✅ 已修复 | 添加 model_name/timestamp/retry_count |
