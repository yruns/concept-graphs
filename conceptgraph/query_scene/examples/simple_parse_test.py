#!/usr/bin/env python3
"""
Simple test script for query parsing - runs without external dependencies.
"""

import sys
from loguru import logger
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from conceptgraph.query_scene.query_structures import (
    GroundingQuery, QueryNode, SpatialConstraint, SelectConstraint, ConstraintType
)
from conceptgraph.query_scene.query_parser import SimpleQueryParser

# Test queries - ordered by complexity
TEST_QUERIES = [
    # Simple object
    "the red cup",
    
    # Simple spatial
    "the pillow on the sofa",
    "the lamp near the window",
    
    # Superlative
    "the sofa nearest the door", 
    "the largest book on the shelf",
    "the smallest plant",
    
    # Multi-anchor
    "the lamp between the sofa and the TV",
    
    # Ordinal
    "the second chair from the left",
    "the first book from the right",
    
    # Complex nested (LLM would handle better)
    "the pillow on the sofa nearest the door",
    "the red cup on the table near the window",
]

# Mock scene categories
MOCK_CATEGORIES = [
    "pillow", "sofa", "cup", "door", "book", "shelf", 
    "lamp", "TV", "chair", "table", "window"
]

def main():
    logger.info("=" * 60)
    logger.info("SimpleQueryParser Test")
    logger.info("=" * 60)
    
    parser = SimpleQueryParser(MOCK_CATEGORIES)
    
    for query in TEST_QUERIES:
        logger.info(f"Query: \"{query}\"")
        logger.info("-" * 40)
        
        try:
            result = parser.parse(query)
            logger.info(f"Target: {result.root.category}")
            
            if result.root.attributes:
                logger.info(f"Attributes: {result.root.attributes}")
            
            if result.root.spatial_constraints:
                for sc in result.root.spatial_constraints:
                    anchors = [a.category for a in sc.anchors]
                    logger.info(f"Spatial: [{sc.relation}] -> {anchors}")
            
            if result.root.select_constraint:
                sel = result.root.select_constraint
                ref = sel.reference.category if sel.reference else "N/A"
                logger.info(f"Select: {sel.constraint_type.value} ({sel.metric}, {sel.order}) -> {ref}")
            
            logger.info(f"Expect Unique: {result.expect_unique}")
            
        except Exception as e:
            logger.error(f"{e}")
    
    logger.info("=" * 60)
    logger.success("Test completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
