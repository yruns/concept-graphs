"""Query Case Generator for creating ground-truth evaluation cases.

Generates EvaluationCase objects with target_obj_ids by:
1. Randomly selecting target objects in a scene
2. Annotating images with red bounding boxes
3. Asking Gemini to generate queries for the marked objects
4. Post-processing to derive anchor/type/difficulty from parsed queries
"""

import base64
import io
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from conceptgraph.query_scene.image_annotator import (
    annotate_image_with_targets,
    build_view_score_dict,
    find_best_view_for_objects,
    get_best_bbox_in_view,
)
from conceptgraph.utils.llm_client import GeminiClientPool, _is_rate_limit_error


class QueryDifficulty(str, Enum):
    """Query difficulty levels based on disambiguation requirements."""

    EASY = "easy"  # Single distinct object
    MEDIUM = "medium"  # Requires spatial relation
    HARD = "hard"  # Multiple candidates, complex disambiguation


class QueryType(str, Enum):
    """Query type classification based on structure."""

    DIRECT = "direct"  # "the sofa"
    SPATIAL = "spatial"  # "the pillow on the sofa"
    ATTRIBUTE = "attribute"  # "the red pillow"
    SUPERLATIVE = "superlative"  # "the largest sofa"
    MULTI_TARGET = "multi"  # "the pillows on the sofa"


class EvaluationCase(BaseModel):
    """Ground-truth evaluation case with target object IDs."""

    # Query
    query: str  # "the throw pillow on the sofa"

    # Ground Truth (core)
    target_obj_ids: List[int]  # [15] - objects marked with red boxes
    target_categories: List[str]  # ["throw_pillow"]

    # Derived fields (inferred in post-processing, not from LLM)
    anchor_obj_ids: List[int] = Field(default_factory=list)
    anchor_categories: List[str] = Field(default_factory=list)
    spatial_relation: Optional[str] = None
    query_type: QueryType = QueryType.DIRECT
    difficulty: QueryDifficulty = QueryDifficulty.EASY

    # Generation context
    source_view_id: int  # View ID used for generation (not frame_idx)
    source_frame_path: str  # Relative path: "results/frame000127.jpg"

    # LLM generation info
    raw_llm_response: str = ""
    generation_timestamp: str = ""

    # Validation flags
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None  # Structured rejection reason


class GenerationBatch(BaseModel):
    """A batch of generated evaluation cases."""

    scene_name: str
    scene_path: str  # Relative path
    cases: List[EvaluationCase] = Field(default_factory=list)
    generation_config: dict = Field(default_factory=dict)
    total_generated: int = 0
    failed_count: int = 0
    validation_passed: int = 0


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are a spatial query generator for 3D scene understanding.

Your task:
1. Look at the image with RED BOUNDING BOXES highlighting specific objects (labeled A, B, C...)
2. Generate a natural language query that would uniquely identify the boxed object(s)
3. The query should be something a human might naturally ask to find these objects

Rules:
- The query MUST target the objects inside the RED BOXES
- Use spatial relations (on, near, next to, between) when needed to disambiguate
- Generate realistic, natural queries a human would ask
- Do NOT use the marker letters (A, B, C) in your query - use object descriptions
"""

GENERATION_PROMPT = """# Generate Query for Highlighted Objects

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
"""


# ============================================================================
# Post-Processing Utilities
# ============================================================================


def _create_parser(scene_categories: List[str]):
    """Factory function: centralized QueryParser initialization."""
    from conceptgraph.query_scene.query_parser import QueryParser

    return QueryParser(
        llm_model="gemini-2.5-pro",
        scene_categories=scene_categories,
        use_pool=True,
    )


def _select_best_hypothesis(hypotheses: List) -> Optional[Any]:
    """Select best hypothesis by kind priority and rank.

    Priority order: direct > proxy > context
    Within same kind: prefer lower rank value
    """
    if not hypotheses:
        return None

    priority = {"direct": 0, "proxy": 1, "context": 2}

    def sort_key(h):
        # Safe enum handling: get string value
        kind = getattr(h, "kind", "context")
        if hasattr(kind, "value"):
            kind = kind.value
        kind_str = str(kind).lower()

        # Get rank (default to 0 if not present)
        rank = getattr(h, "rank", 0) or 0

        return (priority.get(kind_str, 99), rank)

    sorted_hypos = sorted(hypotheses, key=sort_key)
    return sorted_hypos[0]


def _find_anchor_obj_ids_in_view(
    anchor_categories: List[str],
    all_objects: List,
    source_view_id: int,
    object_to_views: Dict[int, List[Tuple[int, float]]],
    min_visibility: float = 0.3,
    max_anchors: int = 3,  # Limit over-inclusion
) -> List[int]:
    """View-local grounding: only return anchors visible in source_view_id.

    Args:
        anchor_categories: Categories to match
        all_objects: All scene objects
        source_view_id: View ID for visibility check
        object_to_views: Visibility index
        min_visibility: Minimum visibility score
        max_anchors: Maximum anchors to return per category

    Returns:
        List of anchor object IDs
    """
    anchor_obj_ids = []
    category_counts: Dict[str, int] = defaultdict(int)

    for obj in all_objects:
        if obj.category not in anchor_categories:
            continue

        # Limit per-category to prevent over-inclusion
        if category_counts[obj.category] >= max_anchors:
            continue

        # Check visibility in source view
        view_scores = build_view_score_dict(obj.obj_id, object_to_views)
        if source_view_id in view_scores:
            score = view_scores[source_view_id]
            if score >= min_visibility:
                anchor_obj_ids.append(obj.obj_id)
                category_counts[obj.category] += 1

    return anchor_obj_ids


def infer_query_type_recursive(node) -> QueryType:
    """Recursively traverse query tree to infer type."""
    # Check current node
    if getattr(node, "select_constraint", None):
        return QueryType.SUPERLATIVE

    spatial_constraints = getattr(node, "spatial_constraints", None)
    if spatial_constraints:
        # Recursively check anchors
        for sc in spatial_constraints:
            anchors = getattr(sc, "anchors", None)
            if anchors:
                for anchor in anchors:
                    sub_type = infer_query_type_recursive(anchor)
                    if sub_type == QueryType.SUPERLATIVE:
                        return QueryType.SUPERLATIVE
        return QueryType.SPATIAL

    if getattr(node, "attributes", None):
        return QueryType.ATTRIBUTE

    return QueryType.DIRECT


def infer_difficulty(
    target_objects: List,
    anchor_obj_ids: List[int],
    all_objects: List,
    scene_categories: List[str],
) -> QueryDifficulty:
    """Infer difficulty based on scene complexity."""
    target_cats = [obj.category for obj in target_objects]

    # Count same-category objects in scene
    same_category_count = sum(
        1 for obj in all_objects if obj.category in target_cats
    )

    # Easy: only one object of this category
    if same_category_count == len(target_objects):
        return QueryDifficulty.EASY

    # Hard: requires complex spatial relation or many candidates
    if len(anchor_obj_ids) > 1 or same_category_count > 5:
        return QueryDifficulty.HARD

    # Medium: requires simple spatial relation
    return QueryDifficulty.MEDIUM


def post_process_case(
    raw_query: str,
    target_obj_ids: List[int],
    target_objects: List,
    all_objects: List,
    scene_categories: List[str],
    source_view_id: int,
    source_frame_path: str,
    object_to_views: Dict[int, List[Tuple[int, float]]],
) -> Tuple[EvaluationCase, List[str]]:
    """Post-process: parse query to infer anchor/type/difficulty, validate.

    Does not rely on LLM-output anchor_obj_ids.
    """
    errors = []
    target_categories = [obj.category for obj in target_objects]
    rejection_reason = None

    # 1. Parse query to get structured info
    parser = _create_parser(scene_categories)

    try:
        hypothesis_output = parser.parse(raw_query)
        hypo = _select_best_hypothesis(hypothesis_output.hypotheses)
        if hypo is None:
            raise ValueError("No valid hypothesis found")
        grounding_query = hypo.grounding_query
    except Exception as e:
        errors.append(f"Query parse failed: {e}")
        rejection_reason = "parse_failed"
        return (
            EvaluationCase(
                query=raw_query,
                target_obj_ids=target_obj_ids,
                target_categories=target_categories,
                query_type=QueryType.DIRECT,
                difficulty=QueryDifficulty.EASY,
                source_view_id=source_view_id,
                source_frame_path=source_frame_path,
                validated=False,
                validation_errors=errors,
                rejection_reason=rejection_reason,
            ),
            errors,
        )

    # 2. Extract anchor info
    anchor_categories = []
    spatial_relation = None
    spatial_constraints = getattr(grounding_query.root, "spatial_constraints", None)
    if spatial_constraints:
        sc = spatial_constraints[0]
        spatial_relation = getattr(sc, "relation", None)
        anchors = getattr(sc, "anchors", None)
        if anchors:
            anchor_categories = getattr(anchors[0], "categories", [])

    # 3. Find anchor_obj_ids (view-local grounding)
    anchor_obj_ids = _find_anchor_obj_ids_in_view(
        anchor_categories, all_objects, source_view_id, object_to_views
    )

    # 4. Validate anchor exists in scene
    for cat in anchor_categories:
        if cat not in scene_categories and cat != "UNKNOW":
            errors.append(f"Anchor category '{cat}' not in scene")

    # 5. Infer query_type (recursive traversal)
    query_type = infer_query_type_recursive(grounding_query.root)

    # 6. Infer difficulty
    difficulty = infer_difficulty(
        target_objects, anchor_obj_ids, all_objects, scene_categories
    )

    # 7. Validate target categories (strict subset match)
    parsed_target_cats = set(getattr(grounding_query.root, "categories", []))
    expected_cats = set(target_categories)

    # Allow UNKNOW as wildcard
    if "UNKNOW" not in parsed_target_cats:
        if not parsed_target_cats.issubset(expected_cats | {"UNKNOW"}):
            errors.append(
                f"Parsed target '{parsed_target_cats}' not subset of GT '{expected_cats}'"
            )
            rejection_reason = "target_mismatch"

    return (
        EvaluationCase(
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
            rejection_reason=rejection_reason,
        ),
        errors,
    )


# ============================================================================
# Sampling Utilities
# ============================================================================


@dataclass
class SamplingQuota:
    """Thread-safe quota tracker for target count distribution."""

    target_distribution: Dict[int, float]  # {1: 0.7, 2: 0.25, 3: 0.05}
    total_target: int
    generated: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def remaining(self) -> Dict[int, int]:
        """Calculate remaining quota for each target count."""
        with self._lock:
            result = {}
            for num_targets, ratio in self.target_distribution.items():
                expected = int(self.total_target * ratio)
                result[num_targets] = max(0, expected - self.generated[num_targets])
            return result

    def should_sample(self, num_targets: int) -> bool:
        """Check if quota allows generating this target count."""
        return self.remaining.get(num_targets, 0) > 0

    def record(self, actual_num_targets: int) -> None:
        """Thread-safe recording of generated case."""
        with self._lock:
            self.generated[actual_num_targets] += 1

    def next_target_count(self) -> Optional[int]:
        """Intelligently select next target count to fill unfilled bins."""
        remaining = self.remaining
        # Sort by remaining count descending
        sorted_bins = sorted(remaining.items(), key=lambda x: -x[1])
        for num_targets, count in sorted_bins:
            if count > 0:
                return num_targets
        return None


def sample_target_objects_with_fallback(
    objects: List,
    object_to_views: Dict[int, List[Tuple[int, float]]],
    num_targets: int,
    max_attempts: int = 100,
    min_bbox_area: int = 500,
    min_conf: float = 0.0,  # Default to 0.0 since conf may not be available
) -> Tuple[List, int, int]:
    """Sample target objects with fallback strategy.

    Strategy:
    1. Try to find common visible view for all targets
    2. On failure, fall back to fewer targets
    3. Ultimate fallback to single target

    Returns:
        Tuple of (sampled objects, actual count, best view_id)

    Raises:
        ValueError: If no valid combination found
    """
    # Filter valid candidates with relaxed criteria
    # Note: num_detections may not be set, use len(xyxy) instead
    valid_objects = []
    for obj in objects:
        if getattr(obj, "is_background", False):
            continue

        xyxy = getattr(obj, "xyxy", [])
        if len(xyxy) < 3:  # Need at least 3 detections
            continue

        # Check for valid detection (meets thresholds)
        conf_arr = getattr(obj, "conf", None)
        has_valid = False
        for i, bbox in enumerate(xyxy):
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            area = (x2 - x1) * (y2 - y1)

            # Get confidence (default 1.0 if not available)
            if conf_arr is not None and i < len(conf_arr) and conf_arr[i] is not None:
                conf = float(conf_arr[i])
            else:
                conf = 1.0

            if area >= min_bbox_area and conf >= min_conf:
                has_valid = True
                break

        if has_valid:
            valid_objects.append(obj)

    if not valid_objects:
        raise ValueError("No valid objects for sampling")

    # Try target counts from num_targets down to 1
    for current_num in range(num_targets, 0, -1):
        if len(valid_objects) < current_num:
            continue

        for _ in range(max_attempts):
            targets = random.sample(valid_objects, current_num)
            view_id = find_best_view_for_objects(targets, object_to_views)
            if view_id is not None:
                return targets, current_num, view_id

    raise ValueError("Could not find any valid object combination with visible view")


# ============================================================================
# Main Generator Class
# ============================================================================


class QueryCaseGenerator:
    """Generate ground-truth evaluation cases with target object IDs."""

    def __init__(
        self,
        scene_path: Path,
        temperature: float = 0.7,
        max_retries: int = 3,
        min_bbox_area: int = 500,
        max_workers: int = 4,
        timeout: int = 120,
    ):
        """Initialize the generator.

        Args:
            scene_path: Path to scene directory
            temperature: LLM temperature for generation
            max_retries: Max retries per case
            min_bbox_area: Minimum bbox area threshold
            max_workers: Parallel worker count (bounded by pool capacity)
            timeout: LLM call timeout in seconds
        """
        self.scene_path = Path(scene_path)
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_bbox_area = min_bbox_area
        self.timeout = timeout

        # Bound workers by pool capacity for backpressure
        self._pool = GeminiClientPool.get_instance()
        self.max_workers = min(max_workers, self._pool.pool_size)

        self._load_scene()

    def _load_scene(self) -> None:
        """Load scene data."""
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
        target_distribution: Optional[Dict[int, float]] = None,
    ) -> GenerationBatch:
        """Generate evaluation cases with parallel execution.

        Args:
            num_cases: Number of cases to generate
            target_distribution: Distribution of target counts {1: 0.7, 2: 0.25, 3: 0.05}

        Returns:
            GenerationBatch with generated cases
        """
        if target_distribution is None:
            target_distribution = {1: 0.7, 2: 0.25, 3: 0.05}

        quota = SamplingQuota(
            target_distribution=target_distribution,
            total_target=num_cases,
        )

        cases = []
        failed = 0
        validation_passed = 0

        # Parallel generation with bounded workers
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i in range(num_cases):
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
        """Thread-safe single case generation with retry."""
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
                # Add jitter backoff between retries
                import time
                time.sleep(0.1 * (attempt + 1) * random.random())
                continue

        raise last_error or ValueError("Generation failed")

    def _generate_single_case(
        self, targets: List, view_id: int
    ) -> EvaluationCase:
        """Generate a single evaluation case for target objects."""
        frame_path = self._get_frame_path_from_view(view_id)

        # Generate annotated image (validates bbox)
        annotated_img, marker_map = annotate_image_with_targets(
            frame_path, targets, view_id, min_bbox_area=self.min_bbox_area
        )

        # Convert to base64
        img_data_url = self._image_to_data_url(annotated_img)

        # Build prompt
        markers = ", ".join(sorted(marker_map.keys()))
        prompt = GENERATION_PROMPT.format(markers=markers)

        # Call Gemini
        response = self._invoke_with_image(prompt, img_data_url)

        # Parse LLM response
        raw_query = self._parse_llm_response(response)

        # Post-process validation
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

    def _get_frame_path_from_view(self, view_id: int) -> Path:
        """Convert view_id to frame path."""
        # Use selector's mapping
        frame_idx = self.selector.map_view_to_frame(view_id)
        # Construct path (assuming results/ directory structure)
        return self.scene_path / "results" / f"frame{frame_idx:06d}.jpg"

    def _image_to_data_url(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffer = io.BytesIO()

        # Resize if too large
        max_edge = 1024
        if max(img.size) > max_edge:
            ratio = max_edge / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(buffer, format="JPEG", quality=90)
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{b64_data}"

    def _invoke_with_image(self, prompt: str, image_data_url: str) -> str:
        """Call Gemini with image using pool."""
        from langchain_core.messages import HumanMessage

        client, config_idx = self._pool.get_next_client(
            temperature=self.temperature,
            timeout=self.timeout,
        )

        try:
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]
                )
            ]
            result = client.invoke(messages)
            self._pool.record_request(config_idx, rate_limited=False)

            # Extract content
            content = result.content
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

        except Exception as e:
            if _is_rate_limit_error(e):
                self._pool.record_request(config_idx, rate_limited=True)
            raise

    def _parse_llm_response(self, response: str) -> str:
        """Parse LLM response to extract query."""
        # Clean markdown wrapping
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return data.get("query", "")
        except json.JSONDecodeError:
            # Try to extract query with regex as fallback
            import re
            match = re.search(r'"query"\s*:\s*"([^"]+)"', response)
            if match:
                return match.group(1)
            raise ValueError(f"Failed to parse LLM response: {response[:200]}")
