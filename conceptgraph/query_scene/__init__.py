"""
QueryScene: Query-Driven Scene Representation for VLM Reasoning
===============================================================

A novel scene representation optimized for VLM inference, featuring:
- Multi-granularity CLIP indexing (region -> object -> point)
- View-object bidirectional index with semantic scoring  
- Query-adaptive VLM input construction
- Structured output parsing

Example:
    >>> from conceptgraph.query_scene import QueryScenePipeline
    >>> pipeline = QueryScenePipeline.from_scene("/path/to/scene")
    >>> result = pipeline.query("沙发旁边的台灯")
    >>> print(result.object_node.category, result.centroid)
"""

from .data_structures import (
    ObjectNode,
    ObjectDescriptions,
    RegionNode,
    ViewScore,
    GroundingResult,
    QueryInfo,
    QueryType,
    BoundingBox3D,
)
from .scene_representation import QuerySceneRepresentation
from .query_pipeline import QueryScenePipeline, run_query
from .query_parser import QueryParser, parse_query
from .index_builder import CLIPIndex, VisibilityIndex, SpatialIndex, RegionIndex, PointLevelIndex, SceneIndices
from .point_feature_extractor import (
    PointFeatureExtractor, PointFeatureIndex, PointFeatureConfig,
    compute_scene_point_features
)
from .lseg_extractor import (
    LSegFeatureExtractor, DensePointFeatureExtractor, LSegConfig,
    extract_dense_scene_features
)
from .vlm_interface import VLMClient, VLMInputConstructor, VLMOutputParser, VLMInput, STRATEGY_MAP
from .description_generator import DescriptionGenerator, generate_descriptions
from .utils import (
    project_point_to_image,
    project_3d_bbox_to_2d,
    crop_object_from_image,
    annotate_image_with_objects,
    annotate_bev_with_distances,
)

__version__ = "0.1.0"
__all__ = [
    # Data structures
    "ObjectNode",
    "ObjectDescriptions",
    "RegionNode",
    "ViewScore",
    "GroundingResult",
    "QueryInfo",
    "QueryType",
    "BoundingBox3D",
    # Scene
    "QuerySceneRepresentation",
    # Pipeline
    "QueryScenePipeline",
    "run_query",
    # Parser
    "QueryParser",
    "parse_query",
    # Indices (hierarchical)
    "CLIPIndex",
    "VisibilityIndex",
    "SpatialIndex",
    "RegionIndex",
    "PointLevelIndex",
    "SceneIndices",
    # OpenScene-style point features
    "PointFeatureExtractor",
    "PointFeatureIndex",
    "PointFeatureConfig",
    "compute_scene_point_features",
    # LSeg dense features
    "LSegFeatureExtractor",
    "DensePointFeatureExtractor",
    "LSegConfig",
    "extract_dense_scene_features",
    # VLM
    "VLMClient",
    "VLMInputConstructor",
    "VLMOutputParser",
    "VLMInput",
    "STRATEGY_MAP",
    # Description generation
    "DescriptionGenerator",
    "generate_descriptions",
    # Utils
    "project_point_to_image",
    "project_3d_bbox_to_2d",
    "crop_object_from_image",
    "annotate_image_with_objects",
    "annotate_bev_with_distances",
]
