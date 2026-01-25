"""
QueryScene: Query-Driven Scene Representation for VLM Reasoning
===============================================================

A novel scene representation optimized for VLM inference, featuring:
- Multi-granularity CLIP indexing (region -> object -> point)
- View-object bidirectional index with semantic scoring  
- Query-adaptive VLM input construction
- Structured output parsing
- Nested spatial query support (e.g., "pillow on sofa nearest door")

Example:
    >>> from conceptgraph.query_scene import QueryScenePipeline
    >>> pipeline = QueryScenePipeline.from_scene("/path/to/scene")
    >>> result = pipeline.query("沙发旁边的台灯")
    >>> print(result.object_node.category, result.centroid)
    
    # For nested queries:
    >>> from conceptgraph.query_scene import QueryParser, QueryExecutor
    >>> parser = QueryParser(llm_model="gpt-4o", scene_categories=["pillow", "sofa", "door"])
    >>> query = parser.parse("the pillow on the sofa nearest the door")
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

# Query parsing (nested spatial queries)
from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    ConstraintType,
    simple_query,
    spatial_query,
    superlative_query,
)
from .query_parser import QueryParser, SimpleQueryParser, parse_query
from .query_executor import QueryExecutor, ExecutionResult, execute_query
from .spatial_relations import (
    SpatialRelationChecker,
    RelationResult,
    RELATION_ALIASES,
    check_relation,
    get_canonical_relation,
)

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
    # Query structures (nested spatial queries)
    "GroundingQuery",
    "QueryNode",
    "SpatialConstraint",
    "SelectConstraint",
    "ConstraintType",
    "simple_query",
    "spatial_query",
    "superlative_query",
    # Query parser
    "QueryParser",
    "SimpleQueryParser",
    "parse_query",
    # Query executor
    "QueryExecutor",
    "ExecutionResult",
    "execute_query",
    # Spatial relations
    "SpatialRelationChecker",
    "RelationResult",
    "RELATION_ALIASES",
    "check_relation",
    "get_canonical_relation",
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
