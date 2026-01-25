"""
Nested Spatial Query Structures.

This module defines Pydantic models for representing nested spatial queries
that support arbitrary depth of spatial constraints and selection operations.

Example query: "the pillow on the sofa nearest the door"
Parsed structure:
    QueryNode(
        category="pillow",
        spatial_constraints=[
            SpatialConstraint(
                relation="on",
                anchors=[
                    QueryNode(
                        category="sofa",
                        select_constraint=SelectConstraint(
                            constraint_type=ConstraintType.SUPERLATIVE,
                            metric="distance",
                            order="min",
                            reference=QueryNode(category="door")
                        )
                    )
                ]
            )
        ]
    )
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, ForwardRef

from pydantic import BaseModel, Field, model_validator


class ConstraintType(str, Enum):
    """Type of selection constraint."""
    
    # Superlative: nearest, largest, highest, smallest
    SUPERLATIVE = "superlative"
    
    # Comparative: closer than, larger than
    COMPARATIVE = "comparative"
    
    # Ordinal: first, second, third
    ORDINAL = "ordinal"


class QueryNode(BaseModel):
    """
    Recursive query node representing an object or object set.
    
    This is the core structure that supports arbitrary nesting depth.
    Each node can have:
    - A category (required): the type of object to find
    - Attributes (optional): adjectives like "red", "large"
    - Spatial constraints (optional): relations to other objects
    - Select constraint (optional): superlative/ordinal selection
    
    Attributes:
        category: Object category to search for (e.g., "pillow", "sofa")
        attributes: List of attribute filters (e.g., ["red", "large"])
        spatial_constraints: List of spatial relation constraints (AND logic)
        select_constraint: Optional selection constraint (nearest, largest, etc.)
        node_id: Unique identifier for execution tracking
    """
    
    category: str = Field(
        ...,
        description="Object category to search for, e.g., 'pillow', 'sofa', 'door'"
    )
    
    attributes: List[str] = Field(
        default_factory=list,
        description="Attribute filters like 'red', 'large', 'wooden'"
    )
    
    spatial_constraints: List["SpatialConstraint"] = Field(
        default_factory=list,
        description="List of spatial constraints (AND logic between them)"
    )
    
    select_constraint: Optional["SelectConstraint"] = Field(
        default=None,
        description="Selection constraint like 'nearest', 'largest', 'second'"
    )
    
    node_id: str = Field(
        default="",
        description="Unique identifier for tracking during execution"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "category": "pillow",
                    "attributes": ["red"],
                    "spatial_constraints": [],
                    "select_constraint": None,
                    "node_id": "target_pillow"
                }
            ]
        }
    }


class SpatialConstraint(BaseModel):
    """
    Spatial relation constraint between objects.
    
    Represents relations like "on", "near", "beside", "between", etc.
    The `anchors` list typically contains one object, but can contain
    multiple for relations like "between A and B".
    
    Attributes:
        relation: Spatial relation word (e.g., "on", "near", "between")
        anchors: List of reference objects (usually 1, can be 2 for "between")
    """
    
    relation: str = Field(
        ...,
        description="Spatial relation: on, near, beside, above, below, between, in_front_of, behind, left_of, right_of, inside"
    )
    
    anchors: List[QueryNode] = Field(
        ...,
        description="Reference objects. Usually 1, can be 2 for 'between'"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "relation": "on",
                    "anchors": [{"category": "table"}]
                }
            ]
        }
    }


class SelectConstraint(BaseModel):
    """
    Selection constraint for choosing from candidates.
    
    Used for superlative (nearest, largest) and ordinal (first, second) selections.
    
    Attributes:
        constraint_type: Type of constraint (superlative, comparative, ordinal)
        metric: What to measure (distance, size, height, x_position, etc.)
        order: Sort order (min, max for superlative; asc, desc for ordinal)
        reference: Reference object for distance-based comparisons (e.g., door for "nearest the door")
        position: Position for ordinal selection (1 = first, 2 = second, etc.)
    """
    
    constraint_type: ConstraintType = Field(
        ...,
        description="Type: superlative (nearest/largest), comparative, or ordinal (first/second)"
    )
    
    metric: str = Field(
        ...,
        description="Metric to compare: 'distance', 'size', 'height', 'x_position', 'y_position'"
    )
    
    order: str = Field(
        ...,
        description="Order: 'min' (nearest/smallest), 'max' (farthest/largest), 'asc', 'desc'"
    )
    
    reference: Optional[QueryNode] = Field(
        default=None,
        description="Reference object for distance comparisons (e.g., door for 'nearest the door')"
    )
    
    position: Optional[int] = Field(
        default=None,
        description="Position for ordinal selection: 1=first, 2=second, etc."
    )
    
    @model_validator(mode='after')
    def validate_constraint(self) -> "SelectConstraint":
        """Validate that ordinal constraints have position set."""
        if self.constraint_type == ConstraintType.ORDINAL and self.position is None:
            raise ValueError("Ordinal constraints require 'position' to be set")
        if self.constraint_type == ConstraintType.SUPERLATIVE and self.metric == "distance" and self.reference is None:
            # This is actually OK - could be "nearest" without explicit reference
            pass
        return self
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "constraint_type": "superlative",
                    "metric": "distance",
                    "order": "min",
                    "reference": {"category": "door"},
                    "position": None
                }
            ]
        }
    }


class GroundingQuery(BaseModel):
    """
    Complete grounding query representation.
    
    This is the top-level structure returned by the query parser.
    
    Attributes:
        raw_query: Original natural language query
        root: Root query node (the target object to find)
        expect_unique: Whether to expect a single result (True for "the X", False for "X" or "Xs")
    """
    
    raw_query: str = Field(
        default="",
        description="Original natural language query"
    )
    
    root: QueryNode = Field(
        ...,
        description="Root query node representing the target object"
    )
    
    expect_unique: bool = Field(
        default=True,
        description="True for 'the X' (expect single result), False for 'X' or 'Xs'"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "raw_query": "the pillow on the sofa",
                    "root": {
                        "category": "pillow",
                        "spatial_constraints": [
                            {
                                "relation": "on",
                                "anchors": [{"category": "sofa"}]
                            }
                        ]
                    },
                    "expect_unique": True
                }
            ]
        }
    }
    
    def get_all_categories(self) -> List[str]:
        """Extract all object categories mentioned in the query."""
        categories = []
        self._collect_categories(self.root, categories)
        return categories
    
    def _collect_categories(self, node: QueryNode, categories: List[str]) -> None:
        """Recursively collect categories from a node."""
        categories.append(node.category)
        
        for constraint in node.spatial_constraints:
            for anchor in constraint.anchors:
                self._collect_categories(anchor, categories)
        
        if node.select_constraint and node.select_constraint.reference:
            self._collect_categories(node.select_constraint.reference, categories)


# Rebuild models to resolve forward references
QueryNode.model_rebuild()
SpatialConstraint.model_rebuild()
SelectConstraint.model_rebuild()
GroundingQuery.model_rebuild()


# Convenience functions for creating queries programmatically
def simple_query(category: str, attributes: Optional[List[str]] = None) -> GroundingQuery:
    """Create a simple query for a single object category."""
    return GroundingQuery(
        raw_query=category,
        root=QueryNode(
            category=category,
            attributes=attributes or []
        )
    )


def spatial_query(
    target: str,
    relation: str,
    anchor: str,
    target_attributes: Optional[List[str]] = None,
    anchor_attributes: Optional[List[str]] = None,
) -> GroundingQuery:
    """Create a simple spatial query: target [relation] anchor."""
    return GroundingQuery(
        raw_query=f"{target} {relation} {anchor}",
        root=QueryNode(
            category=target,
            attributes=target_attributes or [],
            spatial_constraints=[
                SpatialConstraint(
                    relation=relation,
                    anchors=[
                        QueryNode(
                            category=anchor,
                            attributes=anchor_attributes or []
                        )
                    ]
                )
            ]
        )
    )


def superlative_query(
    target: str,
    metric: str,
    order: str,
    reference: Optional[str] = None,
) -> GroundingQuery:
    """Create a superlative query: e.g., 'the nearest chair to the door'."""
    ref_node = QueryNode(category=reference) if reference else None
    
    return GroundingQuery(
        raw_query=f"{order} {target}" + (f" to {reference}" if reference else ""),
        root=QueryNode(
            category=target,
            select_constraint=SelectConstraint(
                constraint_type=ConstraintType.SUPERLATIVE,
                metric=metric,
                order=order,
                reference=ref_node
            )
        )
    )
