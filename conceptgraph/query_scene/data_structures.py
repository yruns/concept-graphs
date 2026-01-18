"""
Core data structures for QueryScene.

This module defines the fundamental data classes used throughout the
QueryScene system, including object representations, view scoring,
and query results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class QueryType(str, Enum):
    """Query type classification."""
    
    SIMPLE_OBJECT = "simple_object"
    SPATIAL_RELATION = "spatial_relation"
    FUNCTIONAL_REGION = "functional_region"
    COUNTING = "counting"
    ATTRIBUTE = "attribute"
    COMPARISON = "comparison"


@dataclass
class ObjectDescriptions:
    """Multi-level natural language descriptions for an object."""
    
    appearance: str = ""
    function: str = ""
    spatial: str = ""
    context: str = ""
    summary: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "appearance": self.appearance,
            "function": self.function,
            "spatial": self.spatial,
            "context": self.context,
            "summary": self.summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> ObjectDescriptions:
        return cls(
            appearance=data.get("appearance", ""),
            function=data.get("function", ""),
            spatial=data.get("spatial", ""),
            context=data.get("context", ""),
            summary=data.get("summary", ""),
        )


@dataclass
class BoundingBox3D:
    """3D axis-aligned bounding box."""
    
    min_point: np.ndarray
    max_point: np.ndarray
    
    def __post_init__(self):
        self.min_point = np.asarray(self.min_point, dtype=np.float32)
        self.max_point = np.asarray(self.max_point, dtype=np.float32)
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) / 2
    
    @property
    def size(self) -> np.ndarray:
        return self.max_point - self.min_point
    
    def to_dict(self) -> Dict[str, List[float]]:
        return {"min": self.min_point.tolist(), "max": self.max_point.tolist()}
    
    @classmethod
    def from_dict(cls, data: Dict) -> BoundingBox3D:
        return cls(min_point=np.array(data["min"]), max_point=np.array(data["max"]))
    
    @classmethod
    def from_points(cls, points: np.ndarray) -> BoundingBox3D:
        points = np.asarray(points)
        return cls(min_point=points.min(axis=0), max_point=points.max(axis=0))


@dataclass
class ObjectNode:
    """Object node in the scene representation."""
    
    obj_id: int
    category: str
    detection_confidence: float = 1.0
    
    # Geometry
    point_cloud: Optional[np.ndarray] = None
    bbox_3d: Optional[BoundingBox3D] = None
    centroid: Optional[np.ndarray] = None
    n_points: int = 0
    
    # Features
    clip_feature: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    
    # Descriptions
    descriptions: ObjectDescriptions = field(default_factory=ObjectDescriptions)
    
    # View information
    best_view_ids: List[int] = field(default_factory=list)
    
    # Spatial relationships
    region_id: Optional[int] = None
    nearby_object_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.centroid is not None:
            self.centroid = np.asarray(self.centroid, dtype=np.float32)
        if self.clip_feature is not None:
            self.clip_feature = np.asarray(self.clip_feature, dtype=np.float32)
    
    def to_dict(self, include_features: bool = False) -> Dict[str, Any]:
        result = {
            "obj_id": self.obj_id,
            "category": self.category,
            "detection_confidence": self.detection_confidence,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "n_points": self.n_points,
            "bbox_3d": self.bbox_3d.to_dict() if self.bbox_3d else None,
            "descriptions": self.descriptions.to_dict(),
            "best_view_ids": self.best_view_ids,
            "region_id": self.region_id,
            "nearby_object_ids": self.nearby_object_ids,
        }
        if include_features and self.clip_feature is not None:
            result["clip_feature"] = self.clip_feature.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectNode:
        return cls(
            obj_id=data["obj_id"],
            category=data["category"],
            detection_confidence=data.get("detection_confidence", 1.0),
            centroid=np.array(data["centroid"]) if data.get("centroid") else None,
            n_points=data.get("n_points", 0),
            bbox_3d=BoundingBox3D.from_dict(data["bbox_3d"]) if data.get("bbox_3d") else None,
            descriptions=ObjectDescriptions.from_dict(data.get("descriptions", {})),
            best_view_ids=data.get("best_view_ids", []),
            region_id=data.get("region_id"),
            nearby_object_ids=data.get("nearby_object_ids", []),
            clip_feature=np.array(data["clip_feature"]) if data.get("clip_feature") else None,
        )
    
    def __repr__(self) -> str:
        return f"ObjectNode(id={self.obj_id}, category='{self.category}')"


@dataclass
class RegionNode:
    """Region node representing a functional area in the scene."""
    
    region_id: int
    region_type: str = "unknown"
    description: str = ""
    boundary_2d: Optional[np.ndarray] = None
    area_m2: float = 0.0
    object_ids: List[int] = field(default_factory=list)
    clip_feature: Optional[np.ndarray] = None
    representative_view_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "description": self.description,
            "area_m2": self.area_m2,
            "object_ids": self.object_ids,
            "representative_view_ids": self.representative_view_ids,
        }


@dataclass
class ViewScore:
    """Quality score for a view observing an object."""
    
    view_id: int
    visible_ratio: float = 0.0
    view_quality: float = 0.0
    resolution: float = 0.0
    occlusion_ratio: float = 0.0
    semantic_score: float = 0.0
    
    def get_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        if weights is None:
            weights = {
                "visible": 0.25, "quality": 0.15, "resolution": 0.20,
                "occlusion": 0.15, "semantic": 0.25,
            }
        norm_resolution = min(self.resolution / 100.0, 1.0)
        return (
            weights["visible"] * self.visible_ratio +
            weights["quality"] * self.view_quality +
            weights["resolution"] * norm_resolution +
            weights["occlusion"] * (1.0 - self.occlusion_ratio) +
            weights["semantic"] * self.semantic_score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_id": self.view_id,
            "visible_ratio": self.visible_ratio,
            "view_quality": self.view_quality,
            "resolution": self.resolution,
            "occlusion_ratio": self.occlusion_ratio,
            "semantic_score": self.semantic_score,
            "composite_score": self.get_composite_score(),
        }


@dataclass
class QueryInfo:
    """Parsed query information."""
    
    original_query: str
    target: str
    anchor: Optional[str] = None
    relation: Optional[str] = None
    query_type: QueryType = QueryType.SIMPLE_OBJECT
    use_bev: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "target": self.target,
            "anchor": self.anchor,
            "relation": self.relation,
            "query_type": self.query_type.value,
            "use_bev": self.use_bev,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryInfo:
        return cls(
            original_query=data["original_query"],
            target=data["target"],
            anchor=data.get("anchor"),
            relation=data.get("relation"),
            query_type=QueryType(data.get("query_type", "simple_object")),
            use_bev=data.get("use_bev", False),
        )


@dataclass
class GroundingResult:
    """Result of grounding a query to a 3D scene location."""
    
    success: bool
    object_id: Optional[int] = None
    object_node: Optional[ObjectNode] = None
    bbox_3d: Optional[BoundingBox3D] = None
    centroid: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    confidence: float = 0.0
    reasoning: str = ""
    evidence_view_ids: List[int] = field(default_factory=list)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "object_id": self.object_id,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "bbox_3d": self.bbox_3d.to_dict() if self.bbox_3d else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence_view_ids": self.evidence_view_ids,
            "reason": self.reason,
        }
    
    @classmethod
    def failure(cls, reason: str) -> GroundingResult:
        return cls(success=False, reason=reason)
    
    @classmethod
    def from_object(
        cls, obj: ObjectNode, confidence: float = 1.0,
        reasoning: str = "", evidence_view_ids: Optional[List[int]] = None,
    ) -> GroundingResult:
        return cls(
            success=True,
            object_id=obj.obj_id,
            object_node=obj,
            bbox_3d=obj.bbox_3d,
            centroid=obj.centroid,
            point_cloud=obj.point_cloud,
            confidence=confidence,
            reasoning=reasoning,
            evidence_view_ids=evidence_view_ids or obj.best_view_ids,
        )


@dataclass
class CameraPose:
    """Camera pose with intrinsics."""
    
    position: np.ndarray
    rotation: np.ndarray
    intrinsics: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (640, 480)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
    
    @property
    def fx(self) -> float:
        return float(self.intrinsics[0, 0]) if self.intrinsics is not None else 600.0
    
    @property
    def fy(self) -> float:
        return float(self.intrinsics[1, 1]) if self.intrinsics is not None else 600.0
