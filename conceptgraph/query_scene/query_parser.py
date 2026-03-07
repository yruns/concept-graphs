"""
Query Parser using LangChain Structured Output.

This module implements a natural language query parser that converts
spatial queries into structured GroundingQuery objects using LLM
with structured output.

Usage:
    parser = QueryParser(llm_model="gpt-5.2-2025-12-11", scene_categories=["sofa", "pillow", "door"])
    query = parser.parse("the pillow on the sofa nearest the door")
    # Returns: GroundingQuery with nested structure
"""

from __future__ import annotations

from typing import List, Optional, ForwardRef, Literal
from loguru import logger
from pydantic import Field, create_model

from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    ConstraintType,
)

# Lazy import for LLM client to avoid dependency issues when not using LLM
def _get_langchain_chat_model(*args, **kwargs):
    from conceptgraph.utils.llm_client import get_langchain_chat_model
    return get_langchain_chat_model(*args, **kwargs)


# Supported spatial relations (for quick coordinate-based filtering)
# Import from query_structures to ensure consistency
try:
    from .query_structures import SUPPORTED_RELATIONS_STR
except ImportError:
    SUPPORTED_RELATIONS_STR = "on, above, below, left_of, right_of, in_front_of, behind, near, next_to, beside, inside, between"

# System prompt for query parsing
QUERY_PARSER_SYSTEM_PROMPT = f"""You are a spatial query parser for 3D scene understanding.
Your task is to parse natural language queries about objects in a scene into a structured JSON format.

The output must be a valid GroundingQuery with the following structure:
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
  (These relations support fast coordinate-based filtering. Map synonyms: "on top of"→"on", "under"→"below", "close to"→"near")
  If the query doesn't contain a clear spatial relation, or uses an uncommon relation (e.g., "hanging from", "leaning against"),
  you may use the original wording - the system will skip quick filtering and use full spatial reasoning.
- anchors: List of reference QueryNode objects (1 for most relations, 2 for "between")

SelectConstraint structure (for superlative/ordinal):
- constraint_type: "superlative" or "ordinal"
- metric: "distance", "size", "height", "x_position", etc.
- order: "min" (nearest/smallest), "max" (farthest/largest), "asc", "desc"
- reference: QueryNode for distance reference (e.g., "nearest the door" -> door)
- position: Integer for ordinal (1=first, 2=second, etc.)

IMPORTANT RULES:
1. SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. When the user mentions a general term (e.g., "pillow", "lamp", "table"),
   include ALL semantically related categories from SCENE CATEGORIES. Examples:
   - Query "a pillow" with scene [door, pillow, throw_pillow, sofa] → categories: ["pillow", "throw_pillow"]
   - Query "the lamp" with scene [floor_lamp, table_lamp, sofa] → categories: ["floor_lamp", "table_lamp"]
   - Query "a table" with scene [side_table, coffee_table, chair] → categories: ["side_table", "coffee_table"]
2. Every category in the list MUST be chosen from SCENE CATEGORIES exactly (case-sensitive, keep underscores).
3. If no suitable category exists in SCENE CATEGORIES, output ["UNKNOW"] (a list with single element).
4. ANCHOR CATEGORIES (CRITICAL): This rule applies to ALL QueryNode objects, including:
   - The root target node
   - Anchor nodes in spatial_constraints
   - Reference nodes in select_constraint
   If an anchor/reference category is not in SCENE CATEGORIES, set its categories to ["UNKNOW"].
   Example: Query "pillow on bed" with scene [pillow, sofa, door] (no bed) →
   anchor categories: ["UNKNOW"] (bed not in scene)
5. Before returning, verify EVERY category string in ALL nodes is present in SCENE CATEGORIES or is exactly "UNKNOW".
6. Map common relation synonyms to predefined values: "on top of"→"on", "under"/"beneath"→"below", "close to"→"near"
7. "nearest/closest X" uses SelectConstraint with metric="distance", order="min", reference=X
8. "largest/biggest" uses SelectConstraint with metric="size", order="max", reference=null
9. "first/second/third from left" uses SelectConstraint with constraint_type="ordinal", metric="x_position"
10. Spatial constraints are filters (AND logic), select_constraint is for final selection
11. Keep structure flat when possible - don't over-nest
12. Prefer predefined relations, but if the query uses uncommon spatial words (e.g., "hanging from", "leaning against"), keep them as-is
13. The `categories` list must have at least one element. Include exact matches first, then semantically related categories."""


def get_few_shot_examples() -> str:
    """Get few-shot examples for the parser."""
    return '''
EXAMPLES:

Query: "the pillow on the sofa" (scene has: pillow, throw_pillow, sofa, door)
{
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
}

Query: "the sofa nearest the door" (scene has: sofa, door, window)
{
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
}

Query: "the pillow on the sofa nearest the door" (scene has: pillow, throw_pillow, sofa, door)
{
  "raw_query": "the pillow on the sofa nearest the door",
  "root": {
    "categories": ["pillow", "throw_pillow"],
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [
          {
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
          }
        ]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the red cup on the table" (scene has: cup, side_table, coffee_table, chair)
{
  "raw_query": "the red cup on the table",
  "root": {
    "categories": ["cup"],
    "attributes": ["red"],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the lamp between the sofa and the TV" (scene has: floor_lamp, table_lamp, sofa, TV)
{
  "raw_query": "the lamp between the sofa and the TV",
  "root": {
    "categories": ["floor_lamp", "table_lamp"],
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "between",
        "anchors": [
          {"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
          {"categories": ["TV"], "attributes": [], "spatial_constraints": [], "select_constraint": null}
        ]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the largest book on the shelf" (scene has: book, shelf, table)
{
  "raw_query": "the largest book on the shelf",
  "root": {
    "categories": ["book"],
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [{"categories": ["shelf"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
      }
    ],
    "select_constraint": {
      "constraint_type": "superlative",
      "metric": "size",
      "order": "max",
      "reference": null,
      "position": null
    }
  },
  "expect_unique": true
}

Query: "the second chair from the left" (scene has: chair, armchair, table)
{
  "raw_query": "the second chair from the left",
  "root": {
    "categories": ["chair", "armchair"],
    "attributes": [],
    "spatial_constraints": [],
    "select_constraint": {
      "constraint_type": "ordinal",
      "metric": "x_position",
      "order": "asc",
      "reference": null,
      "position": 2
    }
  },
  "expect_unique": true
}

Query: "the pillow on the bed" (scene has: pillow, throw_pillow, sofa, door - NO bed)
NOTE: "bed" is NOT in scene categories, so anchor must use ["UNKNOW"]
{
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
}

Query: "the lamp nearest the desk" (scene has: floor_lamp, sofa, door - NO desk)
NOTE: "desk" is NOT in scene categories, so reference must use ["UNKNOW"]
{
  "raw_query": "the lamp nearest the desk",
  "root": {
    "categories": ["floor_lamp"],
    "attributes": [],
    "spatial_constraints": [],
    "select_constraint": {
      "constraint_type": "superlative",
      "metric": "distance",
      "order": "min",
      "reference": {"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
      "position": null
    }
  },
  "expect_unique": true
}
'''


class QueryParser:
    """
    Natural language query parser using LLM structured output.
    
    Converts queries like "the pillow on the sofa nearest the door" into
    structured GroundingQuery objects with nested spatial constraints.
    
    Attributes:
        llm_model: Name of the LLM model to use
        scene_categories: List of object categories in the scene
    """
    
    def __init__(
        self,
        llm_model: str,
        scene_categories: List[str],
        temperature: float = 0.0,
    ):
        """
        Initialize the query parser.
        
        Args:
            llm_model: LLM model name (e.g., "gpt-5.2-2025-12-11", "gemini-2.5-pro")
            scene_categories: List of object categories present in the scene
            temperature: LLM temperature (default 0.0 for deterministic output)
        """
        self.llm_model = llm_model
        self.scene_categories = scene_categories
        self.temperature = temperature
        
        # Initialize LLM (structured schema is built per-parse)
        self._llm = None
    
    def _get_llm(self):
        """Lazy initialization of base LLM."""
        if self._llm is None:
            self._llm = _get_langchain_chat_model(
                deployment_name=self.llm_model,
                temperature=self.temperature,
            )
        return self._llm

    def _build_dynamic_schema(self):
        """Build a dynamic schema with category enum + UNKNOW."""
        categories = sorted(set(self.scene_categories))
        if "UNKNOW" not in categories:
            categories.append("UNKNOW")

        Category = Literal[tuple(categories)]

        query_node_ref = ForwardRef("QueryNodeDynamic")
        spatial_constraint_ref = ForwardRef("SpatialConstraintDynamic")
        select_constraint_ref = ForwardRef("SelectConstraintDynamic")

        QueryNodeDynamic = create_model(
            "QueryNodeDynamic",
            categories=(List[Category], Field(..., min_length=1)),  # Changed: List of categories for semantic expansion
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

        types_namespace = {
            "QueryNodeDynamic": QueryNodeDynamic,
            "SpatialConstraintDynamic": SpatialConstraintDynamic,
            "SelectConstraintDynamic": SelectConstraintDynamic,
        }
        QueryNodeDynamic.model_rebuild(_types_namespace=types_namespace)
        SpatialConstraintDynamic.model_rebuild(_types_namespace=types_namespace)
        SelectConstraintDynamic.model_rebuild(_types_namespace=types_namespace)
        GroundingQueryDynamic.model_rebuild(_types_namespace=types_namespace)

        return GroundingQueryDynamic
    
    def _build_prompt(self, query: str) -> str:
        """Build the prompt for query parsing."""
        categories_str = ", ".join(sorted(set(self.scene_categories)))
        
        prompt = f"""{QUERY_PARSER_SYSTEM_PROMPT}

SCENE CATEGORIES: [{categories_str}]

{get_few_shot_examples()}

Now parse this query:
Query: "{query}"

Return ONLY the JSON object matching the GroundingQuery schema."""
        
        return prompt
    
    def parse(self, query: str) -> GroundingQuery:
        """
        Parse a natural language query into a GroundingQuery.

        Args:
            query: Natural language query string

        Returns:
            GroundingQuery object with parsed structure

        Raises:
            ValueError: If parsing fails after retries
        """
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[QueryParser] Parsing query: '{query}' (attempt {attempt + 1})")

                prompt = self._build_prompt(query)
                schema = self._build_dynamic_schema()
                structured_llm = self._get_llm().with_structured_output(schema)

                # Invoke LLM with structured output
                result = structured_llm.invoke(prompt)

                # Ensure raw_query is set
                if not result.raw_query:
                    result.raw_query = query

                # Convert to standard GroundingQuery for downstream compatibility
                parsed = GroundingQuery.model_validate(result.model_dump())

                # Assign node IDs
                self._assign_node_ids(parsed.root, "root")

                logger.success(f"[QueryParser] Successfully parsed query")
                logger.info(f"[QueryParser] Result: {parsed.model_dump_json(indent=2)}")

                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"[QueryParser] Attempt {attempt + 1} failed: {e}")

        # All retries failed - raise error, no fallback
        logger.error(f"[QueryParser] All parsing attempts failed: {last_error}")
        raise ValueError(f"Failed to parse query '{query}' after {max_retries} attempts: {last_error}")

    def _assign_node_ids(self, node: QueryNode, prefix: str) -> None:
        """Recursively assign unique IDs to query nodes."""
        node.node_id = prefix

        for i, constraint in enumerate(node.spatial_constraints):
            for j, anchor in enumerate(constraint.anchors):
                self._assign_node_ids(anchor, f"{prefix}_sc{i}_a{j}")

        if node.select_constraint and node.select_constraint.reference:
            self._assign_node_ids(node.select_constraint.reference, f"{prefix}_sel_ref")

    def parse_batch(self, queries: List[str]) -> List[GroundingQuery]:
        """
        Parse multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of GroundingQuery objects
        """
        return [self.parse(q) for q in queries]


# Convenience function
def parse_query(
    query: str,
    scene_categories: List[str],
    llm_model: str,
) -> GroundingQuery:
    """
    Parse a natural language query.

    Args:
        query: Query string
        scene_categories: List of object categories in the scene
        llm_model: LLM model name (required)

    Returns:
        GroundingQuery object

    Raises:
        ValueError: If parsing fails
    """
    parser = QueryParser(llm_model, scene_categories)
    return parser.parse(query)
