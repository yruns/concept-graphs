"""LLM Evaluator v2: End-to-end evaluation for QueryParser and KeyframeSelector.

This module implements a two-stage evaluation pipeline:
- Stage 1: Deterministic parse evaluation (code-based category matching)
- Stage 2: Blind selector evaluation (Gemini visual assessment, no GT leakage)
- Stage 3: GT comparison (optional diagnostic mode)
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# BEV Path Helper
# ============================================================================


def get_scene_bev_path(scene_path: Path) -> Optional[Path]:
    """Get the cached BEV image path for a scene.

    Uses ReplicaDefaultBEVConfig (same as KeyframeSelector._generate_scene_images):
    - image_size=1000, perspective=True
    - show_objects=False, show_labels=False

    Args:
        scene_path: Path to scene directory (e.g., /Users/bytedance/Replica/room0)

    Returns:
        Path to cached BEV image, or None if not found
    """
    import hashlib
    from dataclasses import asdict

    from .bev_builder import ReplicaDefaultBEVConfig

    # Use global default config (must match KeyframeSelector)
    config = ReplicaDefaultBEVConfig
    config_dict = asdict(config)
    config_str = str(sorted(config_dict.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    bev_path = Path(scene_path) / "bev" / f"scene_bev_{config_hash}.png"

    if bev_path.exists():
        return bev_path

    # Fallback: find any scene_bev_*.png in the directory
    bev_dir = Path(scene_path) / "bev"
    if bev_dir.exists():
        bev_files = list(bev_dir.glob("scene_bev_*.png"))
        if bev_files:
            # Return the most recently modified one
            return max(bev_files, key=lambda p: p.stat().st_mtime)

    return None


# ============================================================================
# Enums and Constants
# ============================================================================


class EvaluationStatus(str, Enum):
    """Unified evaluation status enum."""

    SUCCESS = "success"
    PARSE_FAILED = "parse_failed"
    SELECTOR_EMPTY = "selector_empty"
    IMAGE_LOAD_ERROR = "image_load_error"
    LLM_ERROR = "llm_error"
    LLM_PARSE_ERROR = "llm_parse_error"


# ============================================================================
# Stage 1: Parse Evaluation Data Models
# ============================================================================


class CategoryMatchResult(BaseModel):
    """Category matching result (deterministic computation)."""

    gt_categories: List[str] = Field(default_factory=list)
    parsed_categories: List[str] = Field(default_factory=list)
    exact_matches: List[str] = Field(default_factory=list)
    alias_matches: List[str] = Field(default_factory=list)
    missing_in_parsed: List[str] = Field(default_factory=list)
    extra_in_parsed: List[str] = Field(default_factory=list)
    match_score: float = 0.0


class ParseMetrics(BaseModel):
    """Parse evaluation metrics (code-based, not Gemini)."""

    target_match: CategoryMatchResult
    anchor_match: Optional[CategoryMatchResult] = None
    spatial_relation_correct: Optional[bool] = None
    hypothesis_kind: str = "unknown"
    hypothesis_rank: int = 0
    parse_score: float = Field(default=0.0, ge=0, le=10)
    weight_breakdown: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Stage 2: Selector Evaluation Data Models
# ============================================================================


class PerKeyframeEval(BaseModel):
    """Per-keyframe evaluation details."""

    keyframe_idx: int
    view_id: int = -1  # Mapped by code, not from LLM
    keyframe_path: str = ""  # Mapped by code, not from LLM
    target_visibility: float = Field(default=0.0, ge=0, le=10)
    target_completeness: float = Field(default=0.0, ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)
    image_quality: float = Field(default=0.0, ge=0, le=10)
    observations: str = ""


class SelectorEvaluation(BaseModel):
    """Selector evaluation result (Gemini visual assessment)."""

    target_visibility: float = Field(default=0.0, ge=0, le=10)
    target_completeness: float = Field(default=0.0, ge=0, le=10)
    spatial_context: Optional[float] = Field(default=None, ge=0, le=10)
    image_quality: float = Field(default=0.0, ge=0, le=10)
    selector_score: float = Field(default=0.0, ge=0, le=10)
    best_keyframe_idx: int = -1
    can_answer_query: bool = False
    reasoning: str = ""
    issues: List[str] = Field(default_factory=list)
    per_keyframe_evals: List[PerKeyframeEval] = Field(default_factory=list)


# ============================================================================
# Stage 3: GT Comparison Data Models
# ============================================================================


class GTComparison(BaseModel):
    """GT comparison result (diagnostic mode)."""

    gt_target_obj_ids: List[int] = Field(default_factory=list)
    matched_obj_ids: List[int] = Field(default_factory=list)
    gt_found: List[int] = Field(default_factory=list)
    gt_missed: List[int] = Field(default_factory=list)
    extra_matched: List[int] = Field(default_factory=list)
    coverage: float = 0.0


# ============================================================================
# Input/Output Data Models
# ============================================================================


class EvaluationInputV2(BaseModel):
    """Evaluation input v2.

    BEV Image Path:
        The bev_image_path should point to the cached BEV generated by
        KeyframeSelector._generate_scene_images(), typically located at:
        {scene_path}/bev/scene_bev_{config_hash}.png

        This is a perspective view with mesh background only (no object markers).
        Do NOT regenerate - use the cached path from KeyframeSelector.
    """

    # === Original Query ===
    query: str

    # === Ground Truth (from EvaluationCase) ===
    gt_target_obj_ids: List[int] = Field(default_factory=list)
    gt_target_categories: List[str] = Field(default_factory=list)
    gt_anchor_categories: List[str] = Field(default_factory=list)
    gt_spatial_relation: Optional[str] = None
    gt_source_view_id: int = -1
    gt_source_frame_path: Optional[Path] = None

    # === QueryParser Output ===
    parsed_target_categories: List[str] = Field(default_factory=list)
    parsed_anchor_categories: List[str] = Field(default_factory=list)
    parsed_spatial_relation: Optional[str] = None
    hypothesis_kind: str = "unknown"
    hypothesis_rank: int = 1
    raw_hypothesis_json: str = ""

    # === KeyframeSelector Output ===
    selected_keyframe_paths: List[Path] = Field(default_factory=list)
    selected_view_ids: List[int] = Field(default_factory=list)
    matched_obj_ids: List[int] = Field(default_factory=list)

    # === BEV ===
    bev_image_path: Optional[Path] = None

    # === Config ===
    enable_diagnostic_mode: bool = False
    include_bev: bool = True  # Set False to skip BEV for cost savings

    class Config:
        arbitrary_types_allowed = True


class EvaluationResultV2(BaseModel):
    """Complete evaluation result v2."""

    query: str

    # === Stage 1: Parse (Deterministic) ===
    parse_metrics: ParseMetrics

    # === Stage 2: Selector (Gemini) ===
    selector_evaluation: SelectorEvaluation

    # === Stage 3: GT Comparison (Diagnostic) ===
    gt_comparison: Optional[GTComparison] = None

    # === Overall ===
    overall_score: float = Field(default=0.0, ge=0, le=10)
    suggestions: List[str] = Field(default_factory=list)

    # === Metadata ===
    raw_llm_response: str = ""
    model_name: str = "gemini-2.5-pro"
    prompt_version: str = "v2"
    timestamp: str = ""
    retry_count: int = 0

    # === Failure Handling ===
    evaluation_status: EvaluationStatus = EvaluationStatus.SUCCESS
    error_message: Optional[str] = None


# ============================================================================
# Category Resolver
# ============================================================================


class CategoryResolver:
    """Unified category resolution with alias handling and UNKNOW filtering."""

    ALIASES = {
        "pillow": ["throw_pillow", "cushion", "decorative_pillow"],
        "couch": ["sofa", "settee", "loveseat"],
        "lamp": ["table_lamp", "floor_lamp", "wall_sconce", "light"],
        "table": ["end_table", "side_table", "coffee_table", "c_table"],
        "chair": ["armchair", "dining_chair", "desk_chair", "office_chair"],
        "cabinet": ["cupboard", "dresser", "wardrobe", "chest"],
        "tv": ["television", "monitor", "screen"],
        "plant": ["potted_plant", "houseplant", "flower"],
        "rug": ["carpet", "mat", "area_rug"],
        "bed": ["mattress", "sleeping_area"],
    }

    IGNORE_CATEGORIES = {"unknow", "unknown", "object", "thing", "item", "stuff"}

    def __init__(self, scene_categories: Optional[Set[str]] = None):
        """Initialize resolver with optional scene-specific categories."""
        self.scene_categories = scene_categories or set()
        self._alias_lookup = self._build_alias_lookup()

    def _build_alias_lookup(self) -> Dict[str, str]:
        """Build reverse alias lookup table."""
        lookup = {}
        for base, aliases in self.ALIASES.items():
            lookup[self._normalize(base)] = base
            for alias in aliases:
                lookup[self._normalize(alias)] = base
        return lookup

    def _normalize(self, cat: str) -> str:
        """Normalize category name."""
        return cat.lower().replace("_", " ").replace("-", " ").strip()

    def resolve(self, category: str) -> Optional[str]:
        """Resolve category to canonical form. Returns None if should be ignored."""
        norm = self._normalize(category)

        # Ignore fuzzy categories
        if norm in self.IGNORE_CATEGORIES:
            return None

        # Look up alias
        if norm in self._alias_lookup:
            return self._alias_lookup[norm]

        # If scene categories provided, filter unknown categories
        if self.scene_categories:
            if norm not in {self._normalize(c) for c in self.scene_categories}:
                # Keep but flag as potential extra
                return norm

        return norm

    def resolve_list(self, categories: List[str]) -> List[str]:
        """Resolve category list, filtering invalid entries."""
        resolved = []
        for cat in categories:
            r = self.resolve(cat)
            if r is not None and r not in resolved:
                resolved.append(r)
        return resolved


# ============================================================================
# Stage 1: Deterministic Parse Evaluation
# ============================================================================


def compute_category_match(
    gt_categories: List[str],
    parsed_categories: List[str],
    resolver: Optional[CategoryResolver] = None,
) -> CategoryMatchResult:
    """Compute category match using IoU-style scoring."""
    resolver = resolver or CategoryResolver()

    gt_resolved = set(resolver.resolve_list(gt_categories))
    parsed_resolved = set(resolver.resolve_list(parsed_categories))

    exact_matches = list(gt_resolved & parsed_resolved)
    intersection = len(exact_matches)
    union = len(gt_resolved | parsed_resolved)
    iou = intersection / union if union > 0 else 0.0

    return CategoryMatchResult(
        gt_categories=list(gt_categories),
        parsed_categories=list(parsed_categories),
        exact_matches=exact_matches,
        alias_matches=[],  # Handled by resolver
        missing_in_parsed=list(gt_resolved - parsed_resolved),
        extra_in_parsed=list(parsed_resolved - gt_resolved),
        match_score=iou,
    )


def compute_parse_score(
    target_match: CategoryMatchResult,
    anchor_match: Optional[CategoryMatchResult],
    spatial_correct: Optional[bool],
) -> Tuple[float, Dict[str, float]]:
    """Compute parse score with dynamic weight normalization."""
    weights = {
        "target": 0.5,
        "anchor": 0.3,
        "spatial": 0.2,
    }

    scores = {"target": target_match.match_score * 10}
    active_weights = {"target": weights["target"]}

    if anchor_match is not None:
        scores["anchor"] = anchor_match.match_score * 10
        active_weights["anchor"] = weights["anchor"]

    if spatial_correct is not None:
        scores["spatial"] = 10.0 if spatial_correct else 0.0
        active_weights["spatial"] = weights["spatial"]

    # Normalize weights
    total_weight = sum(active_weights.values())
    normalized = {k: v / total_weight for k, v in active_weights.items()}

    # Weighted average
    parse_score = sum(scores[k] * normalized[k] for k in scores)

    return parse_score, normalized


# ============================================================================
# Stage 2: Blind Selector Evaluation
# ============================================================================


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


def compute_selector_score(
    target_visibility: float,
    target_completeness: float,
    spatial_context: Optional[float],
    image_quality: float,
) -> float:
    """Compute selector score with dynamic weight normalization."""
    if spatial_context is not None:
        return (
            0.35 * target_visibility
            + 0.25 * target_completeness
            + 0.25 * spatial_context
            + 0.15 * image_quality
        )
    else:
        return (
            0.45 * target_visibility
            + 0.35 * target_completeness
            + 0.20 * image_quality
        )


# ============================================================================
# Stage 3: GT Comparison
# ============================================================================


def compute_gt_comparison(
    gt_target_obj_ids: List[int],
    matched_obj_ids: List[int],
) -> GTComparison:
    """Compute GT coverage (diagnostic mode)."""
    gt_set = set(gt_target_obj_ids)
    matched_set = set(matched_obj_ids)

    gt_found = list(gt_set & matched_set)
    gt_missed = list(gt_set - matched_set)
    extra_matched = list(matched_set - gt_set)

    coverage = len(gt_found) / len(gt_set) if gt_set else 0.0

    return GTComparison(
        gt_target_obj_ids=list(gt_target_obj_ids),
        matched_obj_ids=list(matched_obj_ids),
        gt_found=gt_found,
        gt_missed=gt_missed,
        extra_matched=extra_matched,
        coverage=coverage,
    )


# ============================================================================
# Overall Score Computation
# ============================================================================


def compute_overall_score(
    parse_score: float,
    selector_score: float,
    gt_coverage: Optional[float] = None,
) -> float:
    """Compute overall evaluation score.

    Weights:
    - parse_score: 30% (parsing is prerequisite)
    - selector_score: 50% (visual presentation is core)
    - gt_coverage: 20% (GT coverage as validation, if available)
    """
    if gt_coverage is not None:
        return 0.30 * parse_score + 0.50 * selector_score + 0.20 * (gt_coverage * 10)
    else:
        return 0.375 * parse_score + 0.625 * selector_score


# ============================================================================
# Hypothesis Selection
# ============================================================================


def select_hypothesis_for_evaluation(
    hypotheses: List[Any],
) -> Tuple[Any, int, str]:
    """Select the best hypothesis for evaluation.

    Strategy: direct > proxy > context, same kind by lowest rank.
    Returns: (hypothesis, original_rank, kind)
    """
    if not hypotheses:
        return None, 0, "unknown"

    priority = {"direct": 0, "proxy": 1, "context": 2}

    def sort_key(h: Any) -> Tuple[int, int]:
        kind = getattr(h, "kind", "context")
        if hasattr(kind, "value"):
            kind = kind.value
        rank = getattr(h, "rank", 99) or 99
        return (priority.get(str(kind).lower(), 99), rank)

    sorted_hypos = sorted(hypotheses, key=sort_key)
    best = sorted_hypos[0]

    kind = getattr(best, "kind", "context")
    if hasattr(kind, "value"):
        kind = kind.value

    original_rank = getattr(best, "rank", 1) or 1

    return best, original_rank, str(kind).lower()


# ============================================================================
# Validation Functions
# ============================================================================


def validate_evaluation_input(input_: EvaluationInputV2) -> List[str]:
    """Validate evaluation input completeness."""
    errors = []

    # Keyframe count consistency
    if len(input_.selected_keyframe_paths) != len(input_.selected_view_ids):
        errors.append(
            f"Keyframe count mismatch: {len(input_.selected_keyframe_paths)} paths "
            f"vs {len(input_.selected_view_ids)} view_ids"
        )

    # Keyframe path existence
    for i, path in enumerate(input_.selected_keyframe_paths):
        if not Path(path).exists():
            errors.append(f"Keyframe {i} not found: {path}")

    # BEV path
    if input_.include_bev and input_.bev_image_path:
        if not Path(input_.bev_image_path).exists():
            errors.append(f"BEV image not found: {input_.bev_image_path}")

    # GT data completeness
    if not input_.gt_target_obj_ids:
        errors.append("gt_target_obj_ids is empty")
    if not input_.gt_target_categories:
        errors.append("gt_target_categories is empty")

    return errors


def validate_llm_response(
    response: Dict[str, Any],
    num_keyframes: int,
) -> List[str]:
    """Validate LLM response format and integrity."""
    errors = []

    # Required fields
    required = [
        "per_keyframe_evals",
        "selector_score",
        "best_keyframe_idx",
        "can_answer_query",
    ]
    for field in required:
        if field not in response:
            errors.append(f"Missing required field: {field}")

    # per_keyframe_evals count
    evals = response.get("per_keyframe_evals", [])
    if len(evals) != num_keyframes:
        errors.append(
            f"per_keyframe_evals count {len(evals)} != expected {num_keyframes}"
        )

    # Validate keyframe_idx uniqueness and coverage
    indices = [ev.get("keyframe_idx") for ev in evals]
    expected_indices = set(range(num_keyframes))
    actual_indices = set(indices)
    if actual_indices != expected_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        if missing:
            errors.append(f"Missing keyframe indices: {missing}")
        if extra:
            errors.append(f"Unexpected keyframe indices: {extra}")

    # best_keyframe_idx range
    best_idx = response.get("best_keyframe_idx", -1)
    if best_idx < 0 or best_idx >= num_keyframes:
        errors.append(f"best_keyframe_idx {best_idx} out of range [0, {num_keyframes})")

    # Score ranges
    for i, ev in enumerate(evals):
        for key in ["target_visibility", "target_completeness", "image_quality"]:
            val = ev.get(key)
            if val is not None and (val < 0 or val > 10):
                errors.append(f"Keyframe {i} {key}={val} out of range [0, 10]")

    return errors


# ============================================================================
# Failure Result Factory
# ============================================================================


def create_failure_result(
    query: str,
    status: EvaluationStatus,
    error_message: str,
    parse_metrics: Optional[ParseMetrics] = None,
    model_name: str = "gemini-2.5-pro",
    retry_count: int = 0,
) -> EvaluationResultV2:
    """Create a failure result with complete default objects."""
    empty_match = CategoryMatchResult(
        gt_categories=[],
        parsed_categories=[],
        exact_matches=[],
        alias_matches=[],
        missing_in_parsed=[],
        extra_in_parsed=[],
        match_score=0.0,
    )

    default_parse = parse_metrics or ParseMetrics(
        target_match=empty_match,
        anchor_match=None,
        spatial_relation_correct=None,
        hypothesis_kind="unknown",
        hypothesis_rank=0,
        parse_score=0.0,
        weight_breakdown={},
    )

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
        model_name=model_name,
        prompt_version="v2",
        timestamp=datetime.now().isoformat(),
        retry_count=retry_count,
        evaluation_status=status,
        error_message=error_message,
    )


# ============================================================================
# Batch Evaluator Configuration
# ============================================================================


@dataclass
class BatchEvaluatorConfig:
    """Batch evaluator configuration."""

    max_workers: int = 4
    per_case_timeout: int = 120
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0
    include_bev: bool = True
    bev_fallback_on_missing: bool = True
    temperature: float = 0.2
    model_name: str = "gemini-2.5-pro"


# ============================================================================
# LLM Evaluator V2
# ============================================================================


class LLMEvaluatorV2:
    """End-to-end evaluator for QueryParser and KeyframeSelector."""

    def __init__(
        self,
        config: Optional[BatchEvaluatorConfig] = None,
        scene_categories: Optional[Set[str]] = None,
    ):
        self.config = config or BatchEvaluatorConfig()
        self.resolver = CategoryResolver(scene_categories=scene_categories)
        self._pool = None  # Lazy init

    def _get_pool(self):
        """Lazy initialize GeminiClientPool."""
        if self._pool is None:
            from conceptgraph.utils.llm_client import GeminiClientPool

            self._pool = GeminiClientPool.get_instance()
        return self._pool

    def evaluate(self, input_: EvaluationInputV2) -> EvaluationResultV2:
        """Evaluate a single case through all stages."""
        timestamp = datetime.now().isoformat()

        # Validate input
        validation_errors = validate_evaluation_input(input_)
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")
            # Continue with evaluation despite warnings

        # Stage 1: Deterministic Parse Evaluation
        try:
            parse_metrics = self._evaluate_parse(input_)
        except Exception as e:
            logger.error(f"Parse evaluation failed: {e}")
            return create_failure_result(
                query=input_.query,
                status=EvaluationStatus.PARSE_FAILED,
                error_message=str(e),
                model_name=self.config.model_name,
            )

        # Stage 2: Blind Selector Evaluation (Gemini)
        try:
            selector_eval, raw_response = self._evaluate_selector(input_)
        except Exception as e:
            logger.error(f"Selector evaluation failed: {e}")
            return create_failure_result(
                query=input_.query,
                status=EvaluationStatus.LLM_ERROR,
                error_message=str(e),
                parse_metrics=parse_metrics,
                model_name=self.config.model_name,
            )

        # Stage 3: GT Comparison (if enabled)
        gt_comparison = None
        if input_.enable_diagnostic_mode:
            gt_comparison = compute_gt_comparison(
                input_.gt_target_obj_ids,
                input_.matched_obj_ids,
            )

        # Overall Score
        gt_coverage = gt_comparison.coverage if gt_comparison else None
        overall_score = compute_overall_score(
            parse_metrics.parse_score,
            selector_eval.selector_score,
            gt_coverage,
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            parse_metrics, selector_eval, gt_comparison
        )

        return EvaluationResultV2(
            query=input_.query,
            parse_metrics=parse_metrics,
            selector_evaluation=selector_eval,
            gt_comparison=gt_comparison,
            overall_score=overall_score,
            suggestions=suggestions,
            raw_llm_response=raw_response,
            model_name=self.config.model_name,
            prompt_version="v2",
            timestamp=timestamp,
            retry_count=0,
            evaluation_status=EvaluationStatus.SUCCESS,
            error_message=None,
        )

    def _evaluate_parse(self, input_: EvaluationInputV2) -> ParseMetrics:
        """Stage 1: Deterministic parse evaluation."""
        # Target category match
        target_match = compute_category_match(
            input_.gt_target_categories,
            input_.parsed_target_categories,
            self.resolver,
        )

        # Anchor category match (if available)
        anchor_match = None
        if input_.gt_anchor_categories or input_.parsed_anchor_categories:
            anchor_match = compute_category_match(
                input_.gt_anchor_categories,
                input_.parsed_anchor_categories,
                self.resolver,
            )

        # Spatial relation match (if available)
        spatial_correct = None
        if input_.gt_spatial_relation and input_.parsed_spatial_relation:
            gt_rel = input_.gt_spatial_relation.lower().strip()
            parsed_rel = input_.parsed_spatial_relation.lower().strip()
            spatial_correct = gt_rel == parsed_rel

        # Compute score
        parse_score, weight_breakdown = compute_parse_score(
            target_match, anchor_match, spatial_correct
        )

        return ParseMetrics(
            target_match=target_match,
            anchor_match=anchor_match,
            spatial_relation_correct=spatial_correct,
            hypothesis_kind=input_.hypothesis_kind,
            hypothesis_rank=input_.hypothesis_rank,
            parse_score=parse_score,
            weight_breakdown=weight_breakdown,
        )

    def _evaluate_selector(
        self, input_: EvaluationInputV2
    ) -> Tuple[SelectorEvaluation, str]:
        """Stage 2: Blind selector evaluation via Gemini."""
        # Build prompt
        num_keyframes = len(input_.selected_keyframe_paths)
        include_bev = (
            input_.include_bev
            and input_.bev_image_path
            and Path(input_.bev_image_path).exists()
        )

        if include_bev:
            bev_note = f"The last image (Image {num_keyframes + 1}) is a Bird's Eye View showing the spatial layout."
        else:
            bev_note = "No BEV image provided."

        prompt = BLIND_SELECTOR_PROMPT.format(
            query=input_.query,
            parsed_target_categories=input_.parsed_target_categories,
            parsed_anchor_categories=input_.parsed_anchor_categories or "None",
            parsed_spatial_relation=input_.parsed_spatial_relation or "None",
            hypothesis_kind=input_.hypothesis_kind,
            num_keyframes=num_keyframes,
            max_idx=num_keyframes - 1,
            bev_note=bev_note,
        )

        # Prepare images
        images = self._prepare_images(input_, include_bev)

        # Call Gemini
        raw_response = self._invoke_gemini(prompt, images)

        # Parse response
        response_dict = self._parse_llm_response(raw_response, num_keyframes)

        # Map per-keyframe evals with view_id and path
        per_keyframe_evals = []
        for ev in response_dict.get("per_keyframe_evals", []):
            idx = ev.get("keyframe_idx", 0)
            view_id = (
                input_.selected_view_ids[idx]
                if idx < len(input_.selected_view_ids)
                else -1
            )
            keyframe_path = (
                str(input_.selected_keyframe_paths[idx])
                if idx < len(input_.selected_keyframe_paths)
                else ""
            )

            per_keyframe_evals.append(
                PerKeyframeEval(
                    keyframe_idx=idx,
                    view_id=view_id,
                    keyframe_path=keyframe_path,
                    target_visibility=ev.get("target_visibility", 0.0),
                    target_completeness=ev.get("target_completeness", 0.0),
                    spatial_context=ev.get("spatial_context"),
                    image_quality=ev.get("image_quality", 0.0),
                    observations=ev.get("observations", ""),
                )
            )

        # Compute selector score (recompute for consistency)
        spatial_ctx = response_dict.get("spatial_context")
        selector_score = compute_selector_score(
            response_dict.get("target_visibility", 0.0),
            response_dict.get("target_completeness", 0.0),
            spatial_ctx,
            response_dict.get("image_quality", 0.0),
        )

        return SelectorEvaluation(
            target_visibility=response_dict.get("target_visibility", 0.0),
            target_completeness=response_dict.get("target_completeness", 0.0),
            spatial_context=spatial_ctx,
            image_quality=response_dict.get("image_quality", 0.0),
            selector_score=selector_score,
            best_keyframe_idx=response_dict.get("best_keyframe_idx", 0),
            can_answer_query=response_dict.get("can_answer_query", False),
            reasoning=response_dict.get("reasoning", ""),
            issues=response_dict.get("issues", []),
            per_keyframe_evals=per_keyframe_evals,
        ), raw_response

    def _prepare_images(
        self, input_: EvaluationInputV2, include_bev: bool
    ) -> List[str]:
        """Prepare image list for Gemini (base64 data URIs)."""
        import base64

        images = []

        # Add keyframe images
        for path in input_.selected_keyframe_paths:
            if Path(path).exists():
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                    ext = Path(path).suffix.lower()
                    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                    images.append(f"data:{mime};base64,{data}")

        # Add BEV if enabled
        if include_bev and input_.bev_image_path:
            bev_path = Path(input_.bev_image_path)
            if bev_path.exists():
                with open(bev_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                    ext = bev_path.suffix.lower()
                    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                    images.append(f"data:{mime};base64,{data}")

        return images

    def _invoke_gemini(self, prompt: str, images: List[str]) -> str:
        """Invoke Gemini with images."""
        from langchain_core.messages import HumanMessage

        pool = self._get_pool()
        client, config_idx = pool.get_next_client(
            temperature=self.config.temperature,
            timeout=self.config.per_case_timeout,
        )

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        messages = [HumanMessage(content=content)]

        try:
            result = client.invoke(messages)
            pool.record_request(config_idx, rate_limited=False)
            return result.content
        except Exception as e:
            if self._is_rate_limit_error(e):
                pool.record_request(config_idx, rate_limited=True)
            raise

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Check if exception is a rate limit error."""
        error_str = str(e).lower()
        return any(
            kw in error_str
            for kw in ["rate limit", "quota", "429", "resource exhausted"]
        )

    def _parse_llm_response(
        self, raw_response: str, num_keyframes: int
    ) -> Dict[str, Any]:
        """Parse LLM response JSON."""
        # Extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = raw_response.strip()

        try:
            response_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, attempting repair")
            # Fallback: try to find JSON object
            brace_match = re.search(r"\{[\s\S]*\}", raw_response)
            if brace_match:
                try:
                    response_dict = json.loads(brace_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse LLM response: {raw_response[:200]}")
            else:
                raise ValueError(f"No JSON found in LLM response: {raw_response[:200]}")

        # Validate response
        validation_errors = validate_llm_response(response_dict, num_keyframes)
        if validation_errors:
            logger.warning(f"LLM response validation errors: {validation_errors}")

        return response_dict

    def _generate_suggestions(
        self,
        parse_metrics: ParseMetrics,
        selector_eval: SelectorEvaluation,
        gt_comparison: Optional[GTComparison],
    ) -> List[str]:
        """Generate improvement suggestions based on evaluation."""
        suggestions = []

        # Parse suggestions
        if parse_metrics.target_match.missing_in_parsed:
            suggestions.append(
                f"Parser missed categories: {parse_metrics.target_match.missing_in_parsed}"
            )
        if parse_metrics.target_match.extra_in_parsed:
            suggestions.append(
                f"Parser added extra categories: {parse_metrics.target_match.extra_in_parsed}"
            )

        # Selector suggestions
        if selector_eval.target_visibility is not None and selector_eval.target_visibility < 5:
            suggestions.append("Target objects not visible enough in selected frames")
        if selector_eval.target_completeness is not None and selector_eval.target_completeness < 5:
            suggestions.append("Target objects partially occluded or cropped")
        if selector_eval.spatial_context is not None and selector_eval.spatial_context < 5:
            suggestions.append("Spatial relationship not clearly shown")

        # GT comparison suggestions
        if gt_comparison and gt_comparison.gt_missed:
            suggestions.append(f"GT objects missed by selector: {gt_comparison.gt_missed}")

        return suggestions

    def evaluate_batch(
        self,
        inputs: List[EvaluationInputV2],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EvaluationResultV2]:
        """Batch evaluate with parallel execution and rate limiting."""
        results: List[Tuple[int, EvaluationResultV2]] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_with_retry, inp): i
                for i, inp in enumerate(inputs)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Batch evaluation error for case {idx}: {e}")
                    result = create_failure_result(
                        inputs[idx].query,
                        EvaluationStatus.LLM_ERROR,
                        str(e),
                        model_name=self.config.model_name,
                    )

                results.append((idx, result))

                if progress_callback:
                    progress_callback(len(results), len(inputs))

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _evaluate_with_retry(
        self, input_: EvaluationInputV2
    ) -> EvaluationResultV2:
        """Evaluate single case with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = self.evaluate(input_)
                result.retry_count = attempt
                return result

            except Exception as e:
                last_error = e
                if self._is_rate_limit_error(e):
                    backoff = min(
                        self.config.retry_backoff_base ** attempt,
                        self.config.retry_backoff_max,
                    )
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff:.1f}s (attempt {attempt + 1})"
                    )
                    time.sleep(backoff)
                else:
                    # Non-rate-limit errors don't retry
                    break

        return create_failure_result(
            input_.query,
            EvaluationStatus.LLM_ERROR,
            f"Max retries exceeded: {last_error}",
            model_name=self.config.model_name,
            retry_count=self.config.max_retries,
        )

    def generate_report(
        self, results: List[EvaluationResultV2]
    ) -> Dict[str, Any]:
        """Generate evaluation summary report."""
        successful = [r for r in results if r.evaluation_status == EvaluationStatus.SUCCESS]
        failed = [r for r in results if r.evaluation_status != EvaluationStatus.SUCCESS]

        if not successful:
            return {
                "summary": {
                    "total_cases": len(results),
                    "success_count": 0,
                    "failed_count": len(failed),
                    "avg_parse_score": 0.0,
                    "avg_selector_score": 0.0,
                    "avg_overall_score": 0.0,
                },
                "failures": [
                    {"query": r.query, "status": r.evaluation_status.value, "error": r.error_message}
                    for r in failed
                ],
            }

        avg_parse = sum(r.parse_metrics.parse_score for r in successful) / len(successful)
        avg_selector = sum(r.selector_evaluation.selector_score for r in successful) / len(successful)
        avg_overall = sum(r.overall_score for r in successful) / len(successful)

        # Group by hypothesis kind
        by_kind: Dict[str, List[EvaluationResultV2]] = {}
        for r in successful:
            kind = r.parse_metrics.hypothesis_kind
            by_kind.setdefault(kind, []).append(r)

        kind_stats = {
            kind: {
                "count": len(rs),
                "avg_parse": sum(r.parse_metrics.parse_score for r in rs) / len(rs),
                "avg_selector": sum(r.selector_evaluation.selector_score for r in rs) / len(rs),
                "avg_overall": sum(r.overall_score for r in rs) / len(rs),
            }
            for kind, rs in by_kind.items()
        }

        # Common issues
        all_issues = []
        for r in successful:
            all_issues.extend(r.selector_evaluation.issues)

        issue_counts: Dict[str, int] = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:10]

        return {
            "summary": {
                "total_cases": len(results),
                "success_count": len(successful),
                "failed_count": len(failed),
                "avg_parse_score": round(avg_parse, 2),
                "avg_selector_score": round(avg_selector, 2),
                "avg_overall_score": round(avg_overall, 2),
            },
            "by_hypothesis_kind": kind_stats,
            "common_issues": common_issues,
            "failures": [
                {"query": r.query, "status": r.evaluation_status.value, "error": r.error_message}
                for r in failed
            ],
        }
