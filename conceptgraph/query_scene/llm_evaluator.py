"""
LLM-based Keyframe Evaluator using Gemini Vision.

Uses GeminiClientPool to automatically score keyframe selector results
via Gemini's visual reasoning capabilities.

Design Principles:
- No fallback logic - failures raise exceptions
- Strict Pydantic validation for all data structures
- Dynamic weight normalization for conditional dimensions
"""

from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from conceptgraph.utils.llm_client import GeminiClientPool, _is_rate_limit_error

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage


# =============================================================================
# Enums
# =============================================================================


class HypothesisKind(str, Enum):
    """Hypothesis type for query grounding."""

    DIRECT = "direct"
    PROXY = "proxy"
    CONTEXT = "context"


class DimensionName(str, Enum):
    """Evaluation dimension names."""

    TARGET_VISIBILITY = "target_visibility"
    TARGET_COMPLETENESS = "target_completeness"
    SPATIAL_CONTEXT = "spatial_context"
    ANCHOR_VISIBILITY = "anchor_visibility"
    IMAGE_QUALITY = "image_quality"


# =============================================================================
# Data Models
# =============================================================================


class EvaluationInput(BaseModel):
    """Input for keyframe evaluation."""

    query: str
    keyframe_paths: List[Path]
    target_categories: List[str]
    anchor_categories: List[str] = Field(default_factory=list)
    spatial_relation: Optional[str] = None
    hypothesis_kind: HypothesisKind
    matched_object_count: int
    bev_image_path: Optional[Path] = None

    # Tracking info
    view_ids: List[int] = Field(default_factory=list)
    resolved_frame_ids: List[int] = Field(default_factory=list)

    @property
    def has_anchor(self) -> bool:
        return bool(self.anchor_categories)

    @property
    def has_spatial(self) -> bool:
        return bool(self.spatial_relation)


class FrameEvaluation(BaseModel):
    """Per-frame evaluation result."""

    frame_idx: int
    view_id: Optional[int] = None
    frame_path: str
    target_visibility: float = Field(ge=0, le=10)
    target_completeness: float = Field(ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)
    anchor_visibility: Optional[float] = Field(default=None, ge=0, le=10)
    image_quality: float = Field(ge=0, le=10)
    observations: str


class OverallAssessment(BaseModel):
    """Overall evaluation assessment."""

    best_frame_idx: int
    overall_score: float = Field(ge=0, le=10)
    can_answer_query: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class LLMEvaluationResponse(BaseModel):
    """Complete LLM response structure for strict parsing."""

    per_frame_evaluations: List[FrameEvaluation]
    overall_assessment: OverallAssessment


class EvaluationResult(BaseModel):
    """Final evaluation result."""

    query: str
    overall_score: float
    dimension_scores: dict[DimensionName, Optional[float]]
    reasoning: str
    issues: List[str]
    suggestions: List[str]
    per_frame_evaluations: List[FrameEvaluation]
    best_frame_idx: int
    raw_llm_response: str

    # Metadata
    model_name: str
    prompt_version: str = "v2"
    timestamp: str
    retry_count: int = 0


# =============================================================================
# Validation Functions
# =============================================================================


def validate_frame_evaluations(
    evaluations: List[FrameEvaluation],
    has_anchor: bool,
    has_spatial: bool,
) -> None:
    """Validate that null fields match the expected conditions.

    Rules:
    - has_anchor=True: anchor_visibility must be float
    - has_anchor=False: anchor_visibility must be null
    - has_spatial=True: spatial_context must be float
    - has_spatial=False: spatial_context must be null

    Raises ValueError on violation.
    """
    for i, fe in enumerate(evaluations):
        # Anchor visibility validation
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

        # Spatial context validation
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


def validate_evaluation_integrity(
    response: LLMEvaluationResponse,
    expected_frame_count: int,
) -> None:
    """Validate evaluation integrity.

    Checks:
    - Number of frame evaluations matches expected count
    - best_frame_idx is within range
    """
    actual_count = len(response.per_frame_evaluations)
    if actual_count != expected_frame_count:
        raise ValueError(
            f"Frame count mismatch: expected {expected_frame_count}, "
            f"got {actual_count} frame evaluations"
        )

    best_idx = response.overall_assessment.best_frame_idx
    if best_idx < 0 or best_idx >= expected_frame_count:
        raise ValueError(
            f"best_frame_idx {best_idx} out of range [0, {expected_frame_count})"
        )


# =============================================================================
# Prompt Templates
# =============================================================================


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
{null_instructions}

## Response Format (JSON only, no markdown code blocks)
{{
  "per_frame_evaluations": [
    {{
      "frame_idx": 0,
      "view_id": {example_view_id},
      "frame_path": "<path>",
      "target_visibility": <0-10>,
      "target_completeness": <0-10>,
      "spatial_context": {example_spatial},
      "anchor_visibility": {example_anchor},
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

CONDITIONAL_WITH_ANCHOR = '''
### Conditional (when anchor/spatial relation exists):
4. **spatial_context**: Can you verify the spatial relationship between target and anchor?
   - Score how clearly the relationship "{spatial_relation}" is visible

5. **anchor_visibility**: Can you see the anchor object ({anchor_categories})?
   - Score how clearly the anchor is visible'''

CONDITIONAL_WITHOUT_ANCHOR = '''
### Note
This query has no anchor object or spatial relation.'''


# =============================================================================
# LLMEvaluator
# =============================================================================


class LLMEvaluator:
    """LLM-based keyframe evaluator using Gemini Vision.

    Uses GeminiClientPool for rate limit handling and key rotation.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        timeout: int = 180,
        max_rounds: int = 5,
        max_image_size: int = 1024,
    ):
        """Initialize the evaluator.

        Args:
            temperature: Sampling temperature (lower = more consistent)
            timeout: Request timeout in seconds
            max_rounds: Maximum retry rounds (each round tries all keys)
            max_image_size: Maximum image edge size in pixels
        """
        self._pool = GeminiClientPool.get_instance()
        self.temperature = temperature
        self.timeout = timeout
        self.max_rounds = max_rounds
        self.max_image_size = max_image_size

    def evaluate_single(
        self,
        input: EvaluationInput,
        mode: str = "keyframe_only",
    ) -> EvaluationResult:
        """Evaluate keyframe selection for a single query.

        Args:
            input: Evaluation input with keyframe paths and metadata
            mode: "keyframe_only" (default) or "with_bev_context"

        Returns:
            EvaluationResult with scores and reasoning

        Raises:
            ValueError: If input validation fails
            FileNotFoundError: If keyframe or BEV image not found
            RuntimeError: If all retry attempts exhausted
        """
        # Validate input
        if not input.keyframe_paths:
            raise ValueError("keyframe_paths cannot be empty")
        for path in input.keyframe_paths:
            if not path.exists():
                raise FileNotFoundError(f"Keyframe not found: {path}")

        # Encode keyframe images
        image_data_urls = []
        for path in input.keyframe_paths:
            image_data_urls.append(self._encode_image_to_data_url(path))

        # BEV image (diagnostic mode)
        bev_data_url = None
        if mode == "with_bev_context" and input.bev_image_path:
            if not input.bev_image_path.exists():
                raise FileNotFoundError(f"BEV image not found: {input.bev_image_path}")
            bev_data_url = self._encode_image_to_data_url(input.bev_image_path)
            image_data_urls.append(bev_data_url)

        # Build prompt
        prompt = self._build_evaluation_prompt(input, mode, bev_data_url is not None)

        # Call LLM
        response, retry_count = self._invoke_with_images(prompt, image_data_urls)

        # Parse result
        result = self._parse_llm_response(response, input)
        result.retry_count = retry_count

        return result

    def evaluate_batch(
        self,
        inputs: List[EvaluationInput],
        mode: str = "keyframe_only",
    ) -> List[EvaluationResult]:
        """Evaluate multiple queries.

        Args:
            inputs: List of evaluation inputs
            mode: "keyframe_only" (default) or "with_bev_context"

        Returns:
            List of evaluation results (None for failed evaluations)
        """
        results = []
        for i, input in enumerate(inputs):
            try:
                result = self.evaluate_single(input, mode)
                results.append(result)
                logger.info(f"[{i+1}/{len(inputs)}] {input.query[:50]}... -> {result.overall_score:.1f}")
            except Exception as e:
                logger.error(f"[{i+1}/{len(inputs)}] {input.query[:50]}... -> FAILED: {e}")
                results.append(None)
        return results

    def _encode_image_to_data_url(self, image_path: Path) -> str:
        """Convert image to base64 data URL.

        Pipeline: Path -> PIL.Image -> Resize -> JPEG -> base64

        Uses context manager for proper resource cleanup.
        Raises exception on failure (no fallback).
        """
        with Image.open(image_path) as img:
            # Resize: maintain aspect ratio, max edge = max_image_size
            if max(img.size) > self.max_image_size:
                ratio = self.max_image_size / max(img.size)
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

    def _invoke_with_images(
        self,
        prompt: str,
        image_data_urls: List[str],
    ) -> tuple[str, int]:
        """Call Gemini with images using full key rotation retry.

        Retry strategy (nested loops):
        - Outer: max_rounds rounds
        - Inner: try all keys in pool per round
        - Wait exponential backoff between rounds

        Returns:
            Tuple of (response_text, retry_count)
        """
        pool_size = self._pool.pool_size
        last_error = None
        total_attempts = 0

        for round_idx in range(self.max_rounds):
            # Wait between rounds (not on first round)
            if round_idx > 0:
                wait_time = min(30 * (2 ** (round_idx - 1)), 300)
                logger.warning(f"Round {round_idx + 1}: waiting {wait_time}s before retry")
                time.sleep(wait_time)

            # Inner loop: try all keys in this round
            tried_in_round = set()

            # Use explicit iteration over all config indices
            for _ in range(pool_size * 2):  # Safety limit
                client, config_idx = self._pool.get_next_client(
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                total_attempts += 1

                # Skip if already tried this key in this round
                if config_idx in tried_in_round:
                    continue
                tried_in_round.add(config_idx)

                try:
                    # Build multimodal message
                    content_parts = [{"type": "text", "text": prompt}]
                    for img_url in image_data_urls:
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": img_url}}
                        )

                    messages = [HumanMessage(content=content_parts)]
                    result = client.invoke(messages)

                    self._pool.record_request(config_idx, rate_limited=False)

                    # Safely extract content
                    content = self._extract_content(result.content)
                    return content, total_attempts - 1  # retry_count = attempts - 1

                except Exception as e:
                    if _is_rate_limit_error(e):
                        self._pool.record_request(config_idx, rate_limited=True)
                        last_error = e
                        logger.warning(f"Key {config_idx} rate limited, trying next...")
                    else:
                        # Non-rate-limit error: record and re-raise
                        self._pool.record_request(config_idx, rate_limited=False)
                        raise

                # Break if all keys tried in this round
                if len(tried_in_round) >= pool_size:
                    break

        # All rounds exhausted
        raise RuntimeError(
            f"All {self.max_rounds} rounds × {pool_size} keys exhausted. "
            f"Total attempts: {total_attempts}. Last error: {last_error}"
        )

    def _extract_content(self, content: Any) -> str:
        """Safely extract LLM response content as string.

        Handles multiple content formats:
        - str: return directly
        - list[dict]: join all text blocks with newlines
        - other: convert to string
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        texts.append(block["text"])
                    # Skip non-text blocks (images, etc.)
                elif isinstance(block, str):
                    texts.append(block)
                elif hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts)
        else:
            return str(content)

    def _build_evaluation_prompt(
        self,
        input: EvaluationInput,
        mode: str,
        has_bev: bool,
    ) -> str:
        """Build evaluation prompt from input."""
        # Image descriptions
        image_descs = []
        for i, path in enumerate(input.keyframe_paths):
            view_id = input.view_ids[i] if i < len(input.view_ids) else None
            frame_id = input.resolved_frame_ids[i] if i < len(input.resolved_frame_ids) else None
            desc = f"**Frame {i}**"
            if view_id is not None:
                desc += f" (view_id={view_id})"
            if frame_id is not None:
                desc += f" (frame={frame_id})"
            desc += f": {path.name}"
            image_descs.append(desc)

        if has_bev:
            image_descs.append("**BEV Overview**: Bird's-eye view of the scene (for spatial context only)")

        image_descriptions = "\n".join(image_descs)

        # Anchor/spatial sections
        anchor_section = ""
        spatial_section = ""
        if input.anchor_categories:
            anchor_section = f"- **Anchor categories**: {input.anchor_categories}"
        if input.spatial_relation:
            spatial_section = f'- **Spatial relation to verify**: "{input.spatial_relation}"'

        # Conditional dimensions
        if input.has_anchor and input.has_spatial:
            conditional_dimensions = CONDITIONAL_WITH_ANCHOR.format(
                spatial_relation=input.spatial_relation,
                anchor_categories=input.anchor_categories,
            )
            null_instructions = ""
            example_spatial = "<0-10>"
            example_anchor = "<0-10>"
        else:
            conditional_dimensions = CONDITIONAL_WITHOUT_ANCHOR
            null_instructions = "- Score `spatial_context` and `anchor_visibility` as `null`"
            example_spatial = "null"
            example_anchor = "null"

        # Example view_id
        example_view_id = input.view_ids[0] if input.view_ids else "null"

        return EVALUATION_PROMPT_TEMPLATE.format(
            query=input.query,
            target_categories=input.target_categories,
            anchor_section=anchor_section,
            spatial_section=spatial_section,
            image_descriptions=image_descriptions,
            conditional_dimensions=conditional_dimensions,
            null_instructions=null_instructions,
            example_spatial=example_spatial,
            example_anchor=example_anchor,
            example_view_id=example_view_id,
        )

    def _parse_llm_response(
        self,
        response: str,
        input: EvaluationInput,
    ) -> EvaluationResult:
        """Parse LLM response with strict Pydantic validation.

        Raises exception on failure (no fallback).
        """
        # Clean potential markdown wrapping
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Strict JSON parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed: {e}\nResponse: {response[:500]}")

        # Pydantic validation
        llm_response = LLMEvaluationResponse.model_validate(data)

        # Integrity validation
        validate_evaluation_integrity(llm_response, len(input.keyframe_paths))

        # Business rule validation: null fields must match conditions
        validate_frame_evaluations(
            llm_response.per_frame_evaluations,
            has_anchor=input.has_anchor,
            has_spatial=input.has_spatial,
        )

        # Compute dimension scores
        dimension_scores = self._compute_dimension_scores(
            llm_response.per_frame_evaluations,
            has_anchor=input.has_anchor,
            has_spatial=input.has_spatial,
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
        """Compute average scores per dimension.

        Returns None for dimensions that don't apply (no anchor/spatial).
        """
        scores: dict[DimensionName, list[float]] = {
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

        result: dict[DimensionName, Optional[float]] = {}
        for dim, vals in scores.items():
            if vals:
                result[dim] = sum(vals) / len(vals)
            else:
                result[dim] = None

        return result


# =============================================================================
# Factory Functions
# =============================================================================


def create_evaluator(
    temperature: float = 0.1,
    timeout: int = 180,
    max_rounds: int = 5,
) -> LLMEvaluator:
    """Create an LLMEvaluator instance."""
    return LLMEvaluator(
        temperature=temperature,
        timeout=timeout,
        max_rounds=max_rounds,
    )


def from_keyframe_result(
    keyframe_paths: List[Path],
    query: str,
    target_categories: List[str],
    anchor_categories: Optional[List[str]] = None,
    spatial_relation: Optional[str] = None,
    hypothesis_kind: str = "direct",
    matched_object_count: int = 0,
    view_ids: Optional[List[int]] = None,
    resolved_frame_ids: Optional[List[int]] = None,
    bev_image_path: Optional[Path] = None,
) -> EvaluationInput:
    """Create EvaluationInput from keyframe selector results."""
    return EvaluationInput(
        query=query,
        keyframe_paths=keyframe_paths,
        target_categories=target_categories,
        anchor_categories=anchor_categories or [],
        spatial_relation=spatial_relation,
        hypothesis_kind=HypothesisKind(hypothesis_kind),
        matched_object_count=matched_object_count,
        view_ids=view_ids or [],
        resolved_frame_ids=resolved_frame_ids or [],
        bev_image_path=bev_image_path,
    )
