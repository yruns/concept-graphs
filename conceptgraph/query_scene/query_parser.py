"""
Query Parser using LangChain Structured Output.

This module implements a natural language query parser that converts
spatial queries into structured HypothesisOutputV1 objects using LLM
with structured output.

The LLM directly decides:
1. Whether to use SINGLE or MULTI parse mode
2. What hypotheses to generate (DIRECT, PROXY, CONTEXT)
3. How to handle unknown categories (UNKNOW)

Usage:
    parser = QueryParser(llm_model="gpt-5.2-2025-12-11", scene_categories=["sofa", "pillow", "door"])
    result = parser.parse("the pillow on the sofa nearest the door")
    # Returns: HypothesisOutputV1 with hypotheses
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, ForwardRef, Literal, Union

from loguru import logger
from pydantic import Field, create_model

from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    ConstraintType,
    HypothesisOutputV1,
    QueryHypothesis,
    HypothesisKind,
    ParseMode,
)

# Lazy import for LLM client to avoid dependency issues when not using LLM
def _get_langchain_chat_model(*args, **kwargs):
    from conceptgraph.utils.llm_client import get_langchain_chat_model
    return get_langchain_chat_model(*args, **kwargs)


def _get_gemini_pool():
    from conceptgraph.utils.llm_client import GeminiClientPool
    return GeminiClientPool.get_instance()


def _is_rate_limit_error(error: Exception) -> bool:
    from conceptgraph.utils.llm_client import _is_rate_limit_error
    return _is_rate_limit_error(error)


# Supported spatial relations (for quick coordinate-based filtering)
# Import from query_structures to ensure consistency
try:
    from .query_structures import SUPPORTED_RELATIONS_STR
except ImportError:
    SUPPORTED_RELATIONS_STR = "on, above, below, left_of, right_of, in_front_of, behind, near, next_to, beside, inside, between"

# System prompt for query parsing
QUERY_PARSER_SYSTEM_PROMPT = f"""You are a spatial query parser for 3D scene understanding.
Your task is to parse natural language queries into a structured HypothesisOutputV1 JSON format.

The output must be a valid HypothesisOutputV1 with the following structure:
- format_version: Always "hypothesis_output_v1"
- parse_mode: "single" or "multi" (see decision rules below)
- hypotheses: List of QueryHypothesis objects (1-3 hypotheses)

Each QueryHypothesis has:
- kind: "direct", "proxy", or "context"
- rank: 1-based priority (1=highest priority)
- grounding_query: A GroundingQuery object
- lexical_hints: List of free-form hints (synonyms, paraphrases) from the query

GroundingQuery structure:
- raw_query: The original query text
- root: A QueryNode representing the target object
- expect_unique: True if the query uses "the" (singular), False otherwise

Each QueryNode has:
- categories: LIST of object types (MUST be EXACT strings from SCENE CATEGORIES, or ["UNKNOW"] if no match)
- attributes: List of adjective attributes like "red", "large", "wooden"
- spatial_constraints: List of spatial relations to other objects (filter phase, AND logic)
- select_constraint: Optional selection like "nearest", "largest", "second" (select phase)

SpatialConstraint structure:
- relation: PREFERRED to be one of these predefined values: {SUPPORTED_RELATIONS_STR}
  (Map synonyms: "on top of"→"on", "under"→"below", "close to"→"near")
- anchors: List of reference QueryNode objects (1 for most relations, 2 for "between")

SelectConstraint structure (for superlative/ordinal):
- constraint_type: "superlative" or "ordinal"
- metric: "distance", "size", "height", "x_position", etc.
- order: "min" (nearest/smallest), "max" (farthest/largest), "asc", "desc"
- reference: QueryNode for distance reference (e.g., "nearest the door" -> door)
- position: Integer for ordinal (1=first, 2=second, etc.)

=== PARSE MODE DECISION RULES ===

Use parse_mode="single" when:
1. Target AND all anchors/references exist in SCENE CATEGORIES
2. Even with semantic expansion (pillow → [pillow, throw_pillow]), if all categories are in scene

Use parse_mode="multi" when:
1. Target category is NOT in scene → output ["UNKNOW"] and add PROXY/CONTEXT fallback
2. Any anchor/reference category is NOT in scene → output ["UNKNOW"] in that anchor and add PROXY fallback
3. Query is ambiguous and may need fallback strategies

=== HYPOTHESIS KIND RULES ===

DIRECT hypothesis (always rank=1):
- Parse the query literally
- Use ["UNKNOW"] for any category not found in scene
- Keep original spatial constraints and select constraints

PROXY hypothesis (rank=2, only in multi mode):
- Created when target or anchor is UNKNOW
- For missing ANCHOR: replace UNKNOW anchor with semantically similar categories from scene
  (e.g., if "bed" is missing but target is "pillow", try "sofa" or "armchair" as proxy anchor)
- For missing TARGET: replace UNKNOW target with related categories from scene

CONTEXT hypothesis (rank=3, used as last resort):
- Created when both direct and proxy may fail
- Remove all spatial constraints and select constraints
- Replace target with the anchor categories (find the context/scene objects)
- Set expect_unique=false

=== CATEGORY RULES ===

1. SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. When the user mentions a general term,
   include ALL semantically related categories from SCENE CATEGORIES:
   - Query "a pillow" with scene [door, pillow, throw_pillow, sofa] → categories: ["pillow", "throw_pillow"]
   - Query "the lamp" with scene [floor_lamp, table_lamp, sofa] → categories: ["floor_lamp", "table_lamp"]

2. Every category MUST be EXACT string from SCENE CATEGORIES (case-sensitive, keep underscores).

3. If no suitable category exists in SCENE CATEGORIES, output ["UNKNOW"].

4. This applies to ALL QueryNode objects: root, anchors in spatial_constraints, references in select_constraint.

5. Map common relation synonyms: "on top of"→"on", "under"/"beneath"→"below", "close to"→"near"

6. "nearest/closest X" uses SelectConstraint with metric="distance", order="min", reference=X

7. "largest/biggest" uses SelectConstraint with metric="size", order="max", reference=null

=== VISUAL CONTEXT (if image provided) ===

When a Bird's Eye View (BEV) image is provided:
- Each object is shown as a labeled circle at its centroid position
- Labels follow format "NNN: category" (e.g., "001: sofa", "002: pillow")
- Object IDs correspond to the SCENE CATEGORIES list
- Use this visual context to understand spatial relationships
- Reference the image to verify "near", "on", "between" relationships
- Use object positions to resolve ambiguous references
- Note: In the BEV, Y-axis increases downward (image coordinates)"""


def get_few_shot_examples() -> str:
    """Get few-shot examples for the parser."""
    return '''
EXAMPLES:

=== EXAMPLE 1: Simple query - target and anchor both exist (SINGLE mode) ===
Query: "the pillow on the sofa" (scene has: pillow, throw_pillow, sofa, door)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the sofa",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}

=== EXAMPLE 2: Superlative with reference (SINGLE mode) ===
Query: "the sofa nearest the door" (scene has: sofa, door, window)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the sofa nearest the door",
        "root": {
          "categories": ["sofa"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "position": null
          }
        },
        "expect_unique": true
      },
      "lexical_hints": ["sofa", "door", "nearest"]
    }
  ]
}

=== EXAMPLE 3: Missing anchor - "bed" not in scene (MULTI mode with PROXY) ===
Query: "the pillow on the bed" (scene has: pillow, throw_pillow, sofa, armchair, door - NO bed)
NOTE: "bed" is NOT in scene, so anchor uses ["UNKNOW"]. Add PROXY hypothesis with "sofa" as proxy anchor.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "bed"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa", "armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy_anchor"]
    }
  ]
}

=== EXAMPLE 4: Missing target - "laptop" not in scene (MULTI mode with PROXY and CONTEXT) ===
Query: "the laptop on the table" (scene has: book, cup, side_table, coffee_table, chair - NO laptop)
NOTE: "laptop" is NOT in scene. PROXY tries related objects like "book". CONTEXT falls back to anchor.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the laptop on the table",
        "root": {
          "categories": ["UNKNOW"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["laptop", "table"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the laptop on the table",
        "root": {
          "categories": ["book", "cup"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy"]
    },
    {
      "kind": "context",
      "rank": 3,
      "grounding_query": {
        "raw_query": "context for: the laptop on the table",
        "root": {
          "categories": ["side_table", "coffee_table"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 5: Semantic expansion only, all exist (SINGLE mode) ===
Query: "the cushion on the couch" (scene has: sofa, sofa_seat_cushion, pillow, throw_pillow, door)
NOTE: "cushion" expands to all cushion-like categories; "couch" maps to "sofa". All exist → SINGLE mode.
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the cushion on the couch",
        "root": {
          "categories": ["sofa_seat_cushion", "pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}
'''


class QueryParser:
    """
    Natural language query parser using LLM structured output.

    Converts queries like "the pillow on the sofa nearest the door" into
    structured HypothesisOutputV1 objects with hypotheses for multi-strategy grounding.

    The LLM directly decides:
    - parse_mode: SINGLE (all categories exist) or MULTI (some categories missing)
    - hypotheses: DIRECT (literal parse), PROXY (proxy anchors/targets), CONTEXT (fallback)

    Supports:
    - Single model mode: Uses a specific LLM model
    - Pool mode: Uses Gemini client pool for concurrent requests

    Attributes:
        llm_model: Name of the LLM model to use
        scene_categories: List of object categories in the scene
        use_pool: Whether to use Gemini client pool (for concurrent requests)
    """

    def __init__(
        self,
        llm_model: str,
        scene_categories: List[str],
        temperature: float = 0.0,
        use_pool: bool = False,
    ):
        """
        Initialize the query parser.

        Args:
            llm_model: LLM model name (e.g., "gemini-2.5-pro")
            scene_categories: List of object categories present in the scene
            temperature: LLM temperature (default 0.0 for deterministic output)
            use_pool: If True, use Gemini pool for load-balanced concurrent requests
        """
        self.llm_model = llm_model
        self.scene_categories = scene_categories
        self.temperature = temperature
        self.use_pool = use_pool

        # Initialize LLM (structured schema is built per-parse)
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of base LLM."""
        if self._llm is None:
            if self.use_pool and "gemini" in self.llm_model.lower():
                # Use pool's get_client_with_config for load balancing
                pool = _get_gemini_pool()
                self._llm = pool.get_client_with_config(temperature=self.temperature)
            else:
                self._llm = _get_langchain_chat_model(
                    deployment_name=self.llm_model,
                    temperature=self.temperature,
                )
        return self._llm

    def _get_fresh_llm(self):
        """Get a fresh LLM instance (useful for pool to rotate clients)."""
        if self.use_pool and "gemini" in self.llm_model.lower():
            pool = _get_gemini_pool()
            return pool.get_client_with_config(temperature=self.temperature)
        return self._get_llm()

    def _image_to_data_url(
        self, image_path: Union[str, Path], max_size: int = 800
    ) -> str:
        """
        Convert image file to base64 data URL for multimodal LLM input.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height) for resizing

        Returns:
            Data URL string in format "data:image/jpeg;base64,..."
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL (Pillow) is required for image encoding. Install with: pip install Pillow")

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Resize if needed
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Encode to JPEG
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        return f"data:image/jpeg;base64,{b64}"

    def _build_dynamic_schema(self):
        """Build a dynamic schema for HypothesisOutputV1 with category enum + UNKNOW."""
        categories = sorted(set(self.scene_categories))
        if "UNKNOW" not in categories:
            categories.append("UNKNOW")

        Category = Literal[tuple(categories)]

        query_node_ref = ForwardRef("QueryNodeDynamic")
        spatial_constraint_ref = ForwardRef("SpatialConstraintDynamic")
        select_constraint_ref = ForwardRef("SelectConstraintDynamic")
        grounding_query_ref = ForwardRef("GroundingQueryDynamic")
        query_hypothesis_ref = ForwardRef("QueryHypothesisDynamic")

        QueryNodeDynamic = create_model(
            "QueryNodeDynamic",
            categories=(List[Category], Field(..., min_length=1)),
            attributes=(List[str], Field(default_factory=list)),
            spatial_constraints=(List[spatial_constraint_ref], Field(default_factory=list)),
            select_constraint=(Optional[select_constraint_ref], None),
            node_id=(str, ""),
        )

        SpatialConstraintDynamic = create_model(
            "SpatialConstraintDynamic",
            relation=(str, Field(...)),
            anchors=(List[query_node_ref], Field(...)),
        )

        SelectConstraintDynamic = create_model(
            "SelectConstraintDynamic",
            constraint_type=(ConstraintType, Field(...)),
            metric=(str, Field(...)),
            order=(str, Field(...)),
            reference=(Optional[query_node_ref], None),
            position=(Optional[int], None),
        )

        GroundingQueryDynamic = create_model(
            "GroundingQueryDynamic",
            raw_query=(str, Field(...)),
            root=(QueryNodeDynamic, Field(...)),
            expect_unique=(bool, Field(...)),
        )

        # Hypothesis kind enum
        HypothesisKindLiteral = Literal["direct", "proxy", "context"]
        ParseModeLiteral = Literal["single", "multi"]

        QueryHypothesisDynamic = create_model(
            "QueryHypothesisDynamic",
            kind=(HypothesisKindLiteral, Field(...)),
            rank=(int, Field(..., ge=1)),
            grounding_query=(grounding_query_ref, Field(...)),
            lexical_hints=(List[str], Field(default_factory=list)),
        )

        HypothesisOutputV1Dynamic = create_model(
            "HypothesisOutputV1Dynamic",
            format_version=(Literal["hypothesis_output_v1"], "hypothesis_output_v1"),
            parse_mode=(ParseModeLiteral, Field(...)),
            hypotheses=(List[query_hypothesis_ref], Field(..., min_length=1, max_length=3)),
        )

        types_namespace = {
            "QueryNodeDynamic": QueryNodeDynamic,
            "SpatialConstraintDynamic": SpatialConstraintDynamic,
            "SelectConstraintDynamic": SelectConstraintDynamic,
            "GroundingQueryDynamic": GroundingQueryDynamic,
            "QueryHypothesisDynamic": QueryHypothesisDynamic,
        }
        QueryNodeDynamic.model_rebuild(_types_namespace=types_namespace)
        SpatialConstraintDynamic.model_rebuild(_types_namespace=types_namespace)
        SelectConstraintDynamic.model_rebuild(_types_namespace=types_namespace)
        GroundingQueryDynamic.model_rebuild(_types_namespace=types_namespace)
        QueryHypothesisDynamic.model_rebuild(_types_namespace=types_namespace)
        HypothesisOutputV1Dynamic.model_rebuild(_types_namespace=types_namespace)

        return HypothesisOutputV1Dynamic
    
    def _build_prompt(self, query: str) -> str:
        """Build the prompt for query parsing."""
        categories_str = ", ".join(sorted(set(self.scene_categories)))

        prompt = f"""{QUERY_PARSER_SYSTEM_PROMPT}

SCENE CATEGORIES: [{categories_str}]

{get_few_shot_examples()}

Now parse this query:
Query: "{query}"

Return ONLY the JSON object matching the HypothesisOutputV1 schema."""

        return prompt

    def _is_gemini_model(self) -> bool:
        """Check if the current LLM model is a Gemini model."""
        return "gemini" in self.llm_model.lower()

    def _parse_json_response(self, response_text: str, query: str) -> HypothesisOutputV1:
        """Parse JSON response text to HypothesisOutputV1."""
        import json
        import re

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()

        # Parse JSON
        data = json.loads(json_str)

        # Ensure format_version is set
        if "format_version" not in data:
            data["format_version"] = "hypothesis_output_v1"

        # Ensure each hypothesis has raw_query set
        for hypo in data.get("hypotheses", []):
            gq = hypo.get("grounding_query", {})
            if not gq.get("raw_query"):
                prefix = ""
                if hypo.get("kind") == "proxy":
                    prefix = "proxy for: "
                elif hypo.get("kind") == "context":
                    prefix = "context for: "
                gq["raw_query"] = f"{prefix}{query}"

        # Validate and convert
        parsed = HypothesisOutputV1.model_validate(data)
        return parsed

    def parse(
        self,
        query: str,
        scene_images: Optional[List[Union[str, Path]]] = None,
    ) -> HypothesisOutputV1:
        """
        Parse a natural language query into a HypothesisOutputV1.

        For Gemini models with pool enabled, automatically retries with different
        API keys on rate limit errors.

        Args:
            query: Natural language query string
            scene_images: Optional list of scene image paths (e.g., BEV images)
                         for multimodal context. Currently supports k=1 images.

        Returns:
            HypothesisOutputV1 object with hypotheses

        Raises:
            ValueError: If parsing fails after retries
        """
        # For pool mode with Gemini, we handle rate limit retries at the key level
        if self.use_pool and self._is_gemini_model():
            return self._parse_with_pool_retry(query, scene_images)

        # Standard retry logic for non-pool mode
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[QueryParser] Parsing query: '{query}' (attempt {attempt + 1})")
                if scene_images:
                    logger.info(f"[QueryParser] Using {len(scene_images)} scene image(s)")
                parsed = self._do_parse(query, scene_images)

                logger.success(f"[QueryParser] Successfully parsed query")
                logger.debug(f"[QueryParser] Result: {parsed.model_dump_json(indent=2)}")
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"[QueryParser] Attempt {attempt + 1} failed: {e}")

        logger.error(f"[QueryParser] All parsing attempts failed: {last_error}")
        raise ValueError(f"Failed to parse query '{query}' after {max_retries} attempts: {last_error}")

    def _parse_with_pool_retry(
        self,
        query: str,
        scene_images: Optional[List[Union[str, Path]]] = None,
    ) -> HypothesisOutputV1:
        """
        Parse with automatic retry across all pool keys on rate limit.

        Tries each key in the pool until one succeeds or all are exhausted.
        """
        pool = _get_gemini_pool()
        tried_indices = set()
        last_error = None
        max_keys = pool.pool_size

        while len(tried_indices) < max_keys:
            # Get next client with config index for tracking
            llm, config_idx = pool.get_next_client(temperature=self.temperature)

            if config_idx in tried_indices:
                continue

            tried_indices.add(config_idx)
            key_id = pool._get_key_id(config_idx)

            try:
                logger.info(f"[QueryParser] Parsing query: '{query}' (key {key_id})")
                if scene_images:
                    logger.info(f"[QueryParser] Using {len(scene_images)} scene image(s)")

                parsed = self._do_parse_with_llm(query, llm, scene_images)
                pool.record_request(config_idx, rate_limited=False)

                logger.success(f"[QueryParser] Successfully parsed query")
                logger.debug(f"[QueryParser] Result: {parsed.model_dump_json(indent=2)}")
                return parsed

            except Exception as e:
                is_rate_limited = _is_rate_limit_error(e)
                pool.record_request(config_idx, rate_limited=is_rate_limited)

                if is_rate_limited:
                    logger.warning(f"[QueryParser] Key {key_id} rate limited, trying next key...")
                else:
                    logger.warning(f"[QueryParser] Key {key_id} failed: {e}")
                last_error = e
                continue

        logger.error(f"[QueryParser] All {max_keys} keys exhausted")
        raise ValueError(f"Failed to parse query '{query}' - all keys exhausted: {last_error}")

    def _do_parse(
        self,
        query: str,
        scene_images: Optional[List[Union[str, Path]]] = None,
    ) -> HypothesisOutputV1:
        """Core parsing logic with fresh LLM."""
        llm = self._get_fresh_llm()
        return self._do_parse_with_llm(query, llm, scene_images)

    def _do_parse_with_llm(
        self,
        query: str,
        llm,
        scene_images: Optional[List[Union[str, Path]]] = None,
    ) -> HypothesisOutputV1:
        """Core parsing logic with provided LLM."""
        prompt = self._build_prompt(query)

        # For Gemini models, use JSON mode (they don't support complex $ref schemas)
        if self._is_gemini_model():
            # Build multimodal message if images provided
            if scene_images:
                from langchain_core.messages import HumanMessage

                content = [{"type": "text", "text": prompt}]
                for img_path in scene_images:
                    data_url = self._image_to_data_url(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "low"},
                    })

                message = HumanMessage(content=content)
                response = llm.invoke([message])
            else:
                response = llm.invoke(prompt)

            content = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_json_response(content, query)
        else:
            # Use structured output for non-Gemini models
            # Note: Multimodal + structured output may not be supported by all models
            if scene_images:
                from langchain_core.messages import HumanMessage

                content = [{"type": "text", "text": prompt}]
                for img_path in scene_images:
                    data_url = self._image_to_data_url(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "low"},
                    })

                message = HumanMessage(content=content)
                response = llm.invoke([message])
                content_str = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse_json_response(content_str, query)
            else:
                schema = self._build_dynamic_schema()
                structured_llm = llm.with_structured_output(schema)
                result = structured_llm.invoke(prompt)

                parsed = HypothesisOutputV1.model_validate(result.model_dump())

        # Assign node IDs for all hypotheses
        for hypo in parsed.hypotheses:
            self._assign_node_ids(hypo.grounding_query.root, f"h{hypo.rank}_root")

        return parsed

    def _assign_node_ids(self, node: QueryNode, prefix: str) -> None:
        """Recursively assign unique IDs to query nodes."""
        node.node_id = prefix

        for i, constraint in enumerate(node.spatial_constraints):
            for j, anchor in enumerate(constraint.anchors):
                self._assign_node_ids(anchor, f"{prefix}_sc{i}_a{j}")

        if node.select_constraint and node.select_constraint.reference:
            self._assign_node_ids(node.select_constraint.reference, f"{prefix}_sel_ref")

    def parse_batch(self, queries: List[str]) -> List[HypothesisOutputV1]:
        """
        Parse multiple queries (sequential).

        Args:
            queries: List of query strings

        Returns:
            List of HypothesisOutputV1 objects
        """
        return [self.parse(q) for q in queries]

    def parse_batch_parallel(
        self,
        queries: List[str],
        max_workers: Optional[int] = None,
    ) -> List[HypothesisOutputV1]:
        """
        Parse multiple queries in parallel using Gemini pool.

        Requires use_pool=True in constructor.

        Args:
            queries: List of query strings
            max_workers: Max concurrent threads (default: pool size)

        Returns:
            List of HypothesisOutputV1 objects in same order as queries
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self.use_pool:
            logger.warning("parse_batch_parallel called without use_pool=True, falling back to sequential")
            return self.parse_batch(queries)

        pool = _get_gemini_pool()
        if max_workers is None:
            max_workers = min(len(queries), pool.pool_size)

        results = [None] * len(queries)

        def parse_single(idx: int, query: str):
            return idx, self.parse(query)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(parse_single, i, q) for i, q in enumerate(queries)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results


# Convenience function
def parse_query(
    query: str,
    scene_categories: List[str],
    llm_model: str,
) -> HypothesisOutputV1:
    """
    Parse a natural language query.

    Args:
        query: Query string
        scene_categories: List of object categories in the scene
        llm_model: LLM model name (required)

    Returns:
        HypothesisOutputV1 object

    Raises:
        ValueError: If parsing fails
    """
    parser = QueryParser(llm_model, scene_categories)
    return parser.parse(query)
