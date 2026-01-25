"""
Query Parser using LangChain Structured Output.

This module implements a natural language query parser that converts
spatial queries into structured GroundingQuery objects using LLM
with structured output.

Usage:
    parser = QueryParser(llm_model="gpt-4o-2024-08-06", scene_categories=["sofa", "pillow", "door"])
    query = parser.parse("the pillow on the sofa nearest the door")
    # Returns: GroundingQuery with nested structure
"""

from __future__ import annotations

from typing import List, Optional
from loguru import logger

from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    ConstraintType,
)
from conceptgraph.utils.llm_client import get_langchain_chat_model


# System prompt for query parsing
QUERY_PARSER_SYSTEM_PROMPT = """You are a spatial query parser for 3D scene understanding.
Your task is to parse natural language queries about objects in a scene into a structured JSON format.

The output must be a valid GroundingQuery with the following structure:
- raw_query: The original query text
- root: A QueryNode representing the target object
- expect_unique: True if the query uses "the" (singular), False otherwise

Each QueryNode has:
- category: The object type (MUST be from the provided scene categories or a close synonym)
- attributes: List of adjective attributes like "red", "large", "wooden"
- spatial_constraints: List of spatial relations to other objects (filter phase, AND logic)
- select_constraint: Optional selection like "nearest", "largest", "second" (select phase)

SpatialConstraint structure:
- relation: One of: on, near, beside, above, below, in_front_of, behind, left_of, right_of, inside, between
- anchors: List of reference QueryNode objects (1 for most relations, 2 for "between")

SelectConstraint structure (for superlative/ordinal):
- constraint_type: "superlative" or "ordinal"
- metric: "distance", "size", "height", "x_position", etc.
- order: "min" (nearest/smallest), "max" (farthest/largest), "asc", "desc"
- reference: QueryNode for distance reference (e.g., "nearest the door" -> door)
- position: Integer for ordinal (1=first, 2=second, etc.)

IMPORTANT RULES:
1. Map synonyms to scene categories: pillow→throw_pillow, couch→sofa, lamp→table_lamp
2. "nearest/closest X" uses SelectConstraint with metric="distance", order="min", reference=X
3. "largest/biggest" uses SelectConstraint with metric="size", order="max", reference=null
4. "first/second/third from left" uses SelectConstraint with constraint_type="ordinal", metric="x_position"
5. Spatial constraints are filters (AND logic), select_constraint is for final selection
6. Keep structure flat when possible - don't over-nest"""


def get_few_shot_examples() -> str:
    """Get few-shot examples for the parser."""
    return '''
EXAMPLES:

Query: "the pillow on the sofa"
{
  "raw_query": "the pillow on the sofa",
  "root": {
    "category": "pillow",
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [{"category": "sofa", "attributes": [], "spatial_constraints": [], "select_constraint": null}]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the sofa nearest the door"
{
  "raw_query": "the sofa nearest the door",
  "root": {
    "category": "sofa",
    "attributes": [],
    "spatial_constraints": [],
    "select_constraint": {
      "constraint_type": "superlative",
      "metric": "distance",
      "order": "min",
      "reference": {"category": "door", "attributes": [], "spatial_constraints": [], "select_constraint": null},
      "position": null
    }
  },
  "expect_unique": true
}

Query: "the pillow on the sofa nearest the door"
{
  "raw_query": "the pillow on the sofa nearest the door",
  "root": {
    "category": "pillow",
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [
          {
            "category": "sofa",
            "attributes": [],
            "spatial_constraints": [],
            "select_constraint": {
              "constraint_type": "superlative",
              "metric": "distance",
              "order": "min",
              "reference": {"category": "door", "attributes": [], "spatial_constraints": [], "select_constraint": null},
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

Query: "the red cup on the table"
{
  "raw_query": "the red cup on the table",
  "root": {
    "category": "cup",
    "attributes": ["red"],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [{"category": "table", "attributes": [], "spatial_constraints": [], "select_constraint": null}]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the lamp between the sofa and the TV"
{
  "raw_query": "the lamp between the sofa and the TV",
  "root": {
    "category": "lamp",
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "between",
        "anchors": [
          {"category": "sofa", "attributes": [], "spatial_constraints": [], "select_constraint": null},
          {"category": "TV", "attributes": [], "spatial_constraints": [], "select_constraint": null}
        ]
      }
    ],
    "select_constraint": null
  },
  "expect_unique": true
}

Query: "the largest book on the shelf"
{
  "raw_query": "the largest book on the shelf",
  "root": {
    "category": "book",
    "attributes": [],
    "spatial_constraints": [
      {
        "relation": "on",
        "anchors": [{"category": "shelf", "attributes": [], "spatial_constraints": [], "select_constraint": null}]
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

Query: "the second chair from the left"
{
  "raw_query": "the second chair from the left",
  "root": {
    "category": "chair",
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
            llm_model: LLM model name (e.g., "gpt-4o-2024-08-06", "gemini-2.5-pro")
            scene_categories: List of object categories present in the scene
            temperature: LLM temperature (default 0.0 for deterministic output)
        """
        self.llm_model = llm_model
        self.scene_categories = scene_categories
        self.temperature = temperature
        
        # Initialize LLM with structured output
        self._llm = None
        self._structured_llm = None
    
    def _get_llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = get_langchain_chat_model(
                deployment_name=self.llm_model,
                temperature=self.temperature,
            )
            # Use with_structured_output for Pydantic model parsing
            self._structured_llm = self._llm.with_structured_output(GroundingQuery)
        return self._structured_llm
    
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
                structured_llm = self._get_llm()
                
                # Invoke LLM with structured output
                result = structured_llm.invoke(prompt)
                
                # Ensure raw_query is set
                if not result.raw_query:
                    result.raw_query = query
                
                # Assign node IDs
                self._assign_node_ids(result.root, "root")
                
                logger.success(f"[QueryParser] Successfully parsed query")
                logger.debug(f"[QueryParser] Result: {result.model_dump_json(indent=2)}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"[QueryParser] Attempt {attempt + 1} failed: {e}")
        
        # All retries failed - return a simple fallback
        logger.error(f"[QueryParser] All parsing attempts failed: {last_error}")
        return self._fallback_parse(query)
    
    def _assign_node_ids(self, node: QueryNode, prefix: str) -> None:
        """Recursively assign unique IDs to query nodes."""
        node.node_id = prefix
        
        for i, constraint in enumerate(node.spatial_constraints):
            for j, anchor in enumerate(constraint.anchors):
                self._assign_node_ids(anchor, f"{prefix}_sc{i}_a{j}")
        
        if node.select_constraint and node.select_constraint.reference:
            self._assign_node_ids(node.select_constraint.reference, f"{prefix}_sel_ref")
    
    def _fallback_parse(self, query: str) -> GroundingQuery:
        """
        Fallback parsing when LLM fails.
        
        Creates a simple query with the entire query as the category.
        """
        # Try to extract at least the first noun as category
        words = query.lower().replace("the ", "").replace("a ", "").split()
        category = words[0] if words else "object"
        
        return GroundingQuery(
            raw_query=query,
            root=QueryNode(
                category=category,
                node_id="root"
            ),
            expect_unique=query.lower().startswith("the ")
        )
    
    def parse_batch(self, queries: List[str]) -> List[GroundingQuery]:
        """
        Parse multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of GroundingQuery objects
        """
        return [self.parse(q) for q in queries]


class SimpleQueryParser:
    """
    Simple rule-based query parser for basic queries.
    
    Falls back to this when LLM is not available or for simple queries.
    Does not support complex nesting.
    """
    
    SPATIAL_KEYWORDS = ["on", "near", "beside", "above", "below", "in", "behind", "in_front_of"]
    SUPERLATIVE_KEYWORDS = {
        "nearest": ("distance", "min"),
        "closest": ("distance", "min"),
        "farthest": ("distance", "max"),
        "largest": ("size", "max"),
        "biggest": ("size", "max"),
        "smallest": ("size", "min"),
        "highest": ("height", "max"),
        "lowest": ("height", "min"),
    }
    
    def __init__(self, scene_categories: Optional[List[str]] = None):
        self.scene_categories = scene_categories or []
    
    def parse(self, query: str) -> GroundingQuery:
        """Parse a simple query using rules."""
        query_lower = query.lower().strip()
        expect_unique = query_lower.startswith("the ")
        
        # Remove articles
        clean = query_lower.replace("the ", "").replace("a ", "").replace("an ", "")
        
        # Check for superlative
        for keyword, (metric, order) in self.SUPERLATIVE_KEYWORDS.items():
            if keyword in clean:
                parts = clean.split(keyword)
                target = parts[0].strip() or parts[1].strip().split()[0]
                reference = None
                if len(parts) > 1 and parts[1].strip():
                    ref_words = parts[1].strip().split()
                    if ref_words:
                        reference = QueryNode(category=ref_words[-1])
                
                return GroundingQuery(
                    raw_query=query,
                    root=QueryNode(
                        category=target,
                        select_constraint=SelectConstraint(
                            constraint_type=ConstraintType.SUPERLATIVE,
                            metric=metric,
                            order=order,
                            reference=reference,
                        )
                    ),
                    expect_unique=expect_unique
                )
        
        # Check for spatial relation
        for rel in self.SPATIAL_KEYWORDS:
            if f" {rel} " in clean:
                parts = clean.split(f" {rel} ", 1)
                if len(parts) == 2:
                    target = parts[0].strip().split()[-1]
                    anchor = parts[1].strip().split()[0]
                    
                    return GroundingQuery(
                        raw_query=query,
                        root=QueryNode(
                            category=target,
                            spatial_constraints=[
                                SpatialConstraint(
                                    relation=rel,
                                    anchors=[QueryNode(category=anchor)]
                                )
                            ]
                        ),
                        expect_unique=expect_unique
                    )
        
        # Simple single object query
        words = clean.split()
        category = words[-1] if words else "object"
        attributes = words[:-1] if len(words) > 1 else []
        
        return GroundingQuery(
            raw_query=query,
            root=QueryNode(
                category=category,
                attributes=attributes
            ),
            expect_unique=expect_unique
        )


# Convenience function
def parse_query(
    query: str,
    scene_categories: List[str],
    llm_model: Optional[str] = None,
    use_simple: bool = False,
) -> GroundingQuery:
    """
    Parse a natural language query.
    
    Args:
        query: Query string
        scene_categories: List of object categories in the scene
        llm_model: LLM model name (required if not using simple parser)
        use_simple: Use simple rule-based parser instead of LLM
        
    Returns:
        GroundingQuery object
    """
    if use_simple or llm_model is None:
        parser = SimpleQueryParser(scene_categories)
    else:
        parser = QueryParser(llm_model, scene_categories)
    
    return parser.parse(query)
