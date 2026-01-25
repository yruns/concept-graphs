#!/usr/bin/env python3
"""
Test nested query parsing with complex spatial queries.

This script tests the QueryParser with various complex nested queries
to verify that the LLM correctly parses them into GroundingQuery structures.

Usage:
    python -m conceptgraph.query_scene.examples.test_nested_query_parsing \
        --llm_model gpt-4o-2024-08-06
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from loguru import logger

# Add parent to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# Test queries with expected structure descriptions
TEST_QUERIES = [
    # Simple queries
    {
        "query": "the pillow on the sofa",
        "description": "Simple spatial relation: pillow -[on]-> sofa",
        "expected": {
            "target": "pillow",
            "relation": "on",
            "anchor": "sofa",
        }
    },
    {
        "query": "the red cup",
        "description": "Simple object with attribute, no spatial relation",
        "expected": {
            "target": "cup",
            "attributes": ["red"],
            "relation": None,
        }
    },
    
    # Nested queries with superlative
    {
        "query": "the pillow on the sofa nearest the door",
        "description": "Nested: pillow -[on]-> sofa -[nearest]-> door",
        "expected": {
            "target": "pillow",
            "relation": "on",
            "anchor": "sofa",
            "anchor_constraint": "nearest door",
        }
    },
    {
        "query": "the sofa nearest the door",
        "description": "Superlative: sofa -[nearest]-> door",
        "expected": {
            "target": "sofa",
            "select_constraint": "nearest",
            "reference": "door",
        }
    },
    {
        "query": "the largest book on the shelf",
        "description": "Superlative + spatial: book -[on]-> shelf, select largest",
        "expected": {
            "target": "book",
            "relation": "on",
            "anchor": "shelf",
            "select_constraint": "largest",
        }
    },
    
    # Multi-level nesting
    {
        "query": "the lamp on the table near the window",
        "description": "Multi-level: lamp -[on]-> table -[near]-> window",
        "expected": {
            "target": "lamp",
            "relation": "on",
            "anchor": "table",
            "anchor_constraint": "near window",
        }
    },
    {
        "query": "the cup on the table in the kitchen",
        "description": "Multi-level: cup -[on]-> table -[in]-> kitchen",
        "expected": {
            "target": "cup",
            "relation": "on",
            "anchor": "table",
            "anchor_constraint": "in kitchen",
        }
    },
    
    # Multi-anchor (between)
    {
        "query": "the lamp between the sofa and the TV",
        "description": "Multi-anchor: lamp -[between]-> [sofa, TV]",
        "expected": {
            "target": "lamp",
            "relation": "between",
            "anchors": ["sofa", "TV"],
        }
    },
    
    # Ordinal
    {
        "query": "the second chair from the left",
        "description": "Ordinal: select 2nd chair by x_position",
        "expected": {
            "target": "chair",
            "select_constraint": "ordinal",
            "position": 2,
        }
    },
    
    # Complex combined
    {
        "query": "the red book on the shelf above the desk",
        "description": "Complex: book(red) -[on]-> shelf -[above]-> desk",
        "expected": {
            "target": "book",
            "attributes": ["red"],
            "relation": "on",
            "anchor": "shelf",
            "anchor_constraint": "above desk",
        }
    },
    {
        "query": "the smallest plant near the window",
        "description": "Superlative + spatial: plant -[near]-> window, select smallest",
        "expected": {
            "target": "plant",
            "relation": "near",
            "anchor": "window",
            "select_constraint": "smallest",
        }
    },
]

# Simulated scene categories (for testing without actual scene)
MOCK_SCENE_CATEGORIES = [
    "pillow", "throw_pillow", "cushion",
    "sofa", "couch", "armchair", "chair",
    "table", "coffee_table", "side_table", "desk",
    "lamp", "table_lamp", "floor_lamp",
    "book", "bookshelf", "shelf",
    "cup", "mug", "glass",
    "door", "window",
    "TV", "television",
    "plant", "flower", "vase",
    "kitchen", "living_room", "bedroom",
]


def print_query_result(query: str, result, description: str):
    """Pretty print a parsing result."""
    print("\n" + "=" * 70)
    print(f"Query: \"{query}\"")
    print(f"Description: {description}")
    print("-" * 70)
    
    # Print the parsed structure
    print(f"Target: {result.root.category}")
    
    if result.root.attributes:
        print(f"Attributes: {result.root.attributes}")
    
    if result.root.spatial_constraints:
        for i, sc in enumerate(result.root.spatial_constraints):
            anchor_cats = [a.category for a in sc.anchors]
            print(f"Spatial Constraint {i+1}: [{sc.relation}] -> {anchor_cats}")
            
            # Check for nested constraints in anchors
            for anchor in sc.anchors:
                if anchor.spatial_constraints:
                    for asc in anchor.spatial_constraints:
                        nested_anchors = [a.category for a in asc.anchors]
                        print(f"  └─ Anchor constraint: [{asc.relation}] -> {nested_anchors}")
                if anchor.select_constraint:
                    sel = anchor.select_constraint
                    ref = sel.reference.category if sel.reference else "N/A"
                    print(f"  └─ Anchor select: {sel.constraint_type.value} ({sel.metric}, {sel.order}) -> {ref}")
    
    if result.root.select_constraint:
        sel = result.root.select_constraint
        ref = sel.reference.category if sel.reference else "N/A"
        pos = f", position={sel.position}" if sel.position else ""
        print(f"Select Constraint: {sel.constraint_type.value} ({sel.metric}, {sel.order}{pos}) -> {ref}")
    
    print(f"Expect Unique: {result.expect_unique}")
    
    # Print raw JSON (condensed)
    print("-" * 70)
    print("JSON Structure:")
    json_str = result.model_dump_json(indent=2)
    # Truncate if too long
    if len(json_str) > 1000:
        lines = json_str.split('\n')
        print('\n'.join(lines[:30]))
        print("... (truncated)")
    else:
        print(json_str)


def test_simple_parser(queries: List[dict]):
    """Test with SimpleQueryParser (no LLM)."""
    # Direct import to avoid dependency issues
    from conceptgraph.query_scene.query_structures import GroundingQuery, QueryNode
    from conceptgraph.query_scene.query_parser import SimpleQueryParser
    
    print("\n" + "#" * 70)
    print("# Testing SimpleQueryParser (rule-based, no LLM)")
    print("#" * 70)
    
    parser = SimpleQueryParser(MOCK_SCENE_CATEGORIES)
    
    for item in queries[:5]:  # Only first 5 for simple parser
        query = item["query"]
        desc = item["description"]
        
        try:
            result = parser.parse(query)
            print_query_result(query, result, desc)
        except Exception as e:
            print(f"\nError parsing '{query}': {e}")


def test_llm_parser(queries: List[dict], llm_model: str):
    """Test with QueryParser (LLM-based)."""
    try:
        from conceptgraph.query_scene.query_parser import QueryParser
    except ImportError as e:
        print("\n" + "#" * 70)
        print("# ERROR: Cannot import QueryParser")
        print(f"# Missing dependency: {e}")
        print("#")
        print("# To install required dependencies, run:")
        print("#   pip install langchain-openai")
        print("#" * 70)
        return
    
    print("\n" + "#" * 70)
    print(f"# Testing QueryParser with LLM: {llm_model}")
    print("#" * 70)
    
    try:
        parser = QueryParser(
            llm_model=llm_model,
            scene_categories=MOCK_SCENE_CATEGORIES,
        )
    except Exception as e:
        print(f"\nFailed to initialize QueryParser: {e}")
        print("Make sure langchain-openai is installed: pip install langchain-openai")
        return
    
    for item in queries:
        query = item["query"]
        desc = item["description"]
        
        try:
            result = parser.parse(query)
            print_query_result(query, result, desc)
        except Exception as e:
            logger.error(f"Error parsing '{query}': {e}")
            print(f"\nError parsing '{query}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Test nested query parsing")
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="LLM model name. If not provided, only tests SimpleQueryParser."
    )
    parser.add_argument(
        "--simple_only",
        action="store_true",
        help="Only test SimpleQueryParser (no LLM)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Nested Query Parsing Test")
    print("=" * 70)
    print(f"\nTesting {len(TEST_QUERIES)} queries...")
    print(f"Scene categories: {len(MOCK_SCENE_CATEGORIES)} categories")
    
    # Test simple parser
    # test_simple_parser(TEST_QUERIES)
    
    # Test LLM parser if model provided
    if args.llm_model and not args.simple_only:
        test_llm_parser(TEST_QUERIES, args.llm_model)
    elif not args.simple_only:
        print("\n" + "#" * 70)
        print("# Skipping LLM parser test (no --llm_model provided)")
        print("# To test with LLM, run:")
        print("#   python -m conceptgraph.query_scene.examples.test_nested_query_parsing \\")
        print("#       --llm_model gpt-4o-2024-08-06")
        print("#" * 70)
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
