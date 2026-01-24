"""Debug runner with loguru logging for QueryScene pipeline."""
import sys
from pathlib import Path
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)
logger.add(
    "query_scene_debug.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
)


def test_imports():
    """Test all module imports."""
    logger.info("=" * 60)
    logger.info("Testing module imports...")
    
    try:
        from conceptgraph.query_scene import (
            ObjectNode, ObjectDescriptions, QueryInfo, QueryType,
            GroundingResult, BoundingBox3D, RegionNode, ViewScore,
        )
        logger.success("Data structures imported OK")
    except Exception as e:
        logger.error(f"Data structures import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import QuerySceneRepresentation
        logger.success("QuerySceneRepresentation imported OK")
    except Exception as e:
        logger.error(f"QuerySceneRepresentation import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import (
            CLIPIndex, VisibilityIndex, SpatialIndex, RegionIndex, SceneIndices
        )
        logger.success("Index classes imported OK")
    except Exception as e:
        logger.error(f"Index classes import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import QueryParser, parse_query
        logger.success("QueryParser imported OK")
    except Exception as e:
        logger.error(f"QueryParser import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import (
            VLMClient, VLMInputConstructor, VLMOutputParser, STRATEGY_MAP
        )
        logger.success("VLM interface imported OK")
        logger.debug(f"STRATEGY_MAP keys: {list(STRATEGY_MAP.keys())}")
    except Exception as e:
        logger.error(f"VLM interface import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import (
            project_3d_bbox_to_2d, annotate_image_with_objects, annotate_bev_with_distances
        )
        logger.success("Utils imported OK")
    except Exception as e:
        logger.error(f"Utils import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import DescriptionGenerator, generate_descriptions
        logger.success("DescriptionGenerator imported OK")
    except Exception as e:
        logger.error(f"DescriptionGenerator import failed: {e}")
        return False
    
    try:
        from conceptgraph.query_scene import QueryScenePipeline, run_query
        logger.success("QueryScenePipeline imported OK")
    except Exception as e:
        logger.error(f"QueryScenePipeline import failed: {e}")
        return False
    
    logger.success("All imports successful!")
    return True


def test_scene_loading(scene_path: str):
    """Test scene loading."""
    logger.info("=" * 60)
    logger.info(f"Testing scene loading from: {scene_path}")
    
    from conceptgraph.query_scene import QuerySceneRepresentation
    
    scene_path = Path(scene_path)
    pcd_dir = scene_path / "pcd_saves"
    
    # Find PCD file
    logger.debug(f"Looking for PCD files in {pcd_dir}")
    pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list(pcd_dir.glob("*.pkl.gz"))
    
    if not pcd_files:
        logger.error(f"No PCD files found in {pcd_dir}")
        return None
    
    pcd_file = str(pcd_files[0])
    logger.info(f"Using PCD file: {pcd_file}")
    
    try:
        scene = QuerySceneRepresentation.from_pcd_file(pcd_file, scene_path, stride=5)
        logger.success(f"Scene loaded successfully")
        logger.info(f"  Objects: {len(scene.objects)}")
        logger.info(f"  Camera poses: {len(scene.camera_poses)}")
        logger.info(f"  Images: {len(scene.image_paths)}")
        
        # Log some object details
        for obj in scene.objects[:5]:
            logger.debug(f"  Object {obj.obj_id}: {obj.category}, centroid={obj.centroid}")
        
        return scene, pcd_file
    except Exception as e:
        logger.exception(f"Scene loading failed: {e}")
        return None


def test_index_building(scene):
    """Test index building."""
    logger.info("=" * 60)
    logger.info("Testing index building...")
    
    from conceptgraph.query_scene import SceneIndices
    
    try:
        indices = SceneIndices.build_all(scene, build_regions=True)
        logger.success("Indices built successfully")
        
        # Check CLIP index
        if indices.clip_index.index is not None:
            logger.info(f"  CLIP index: {len(indices.clip_index.metadata)} entries")
        else:
            logger.warning("  CLIP index is None")
        
        # Check visibility index
        logger.info(f"  Visibility index: {len(indices.visibility_index.object_to_views)} objects")
        
        # Check spatial index
        if indices.spatial_index.tree is not None:
            logger.info(f"  Spatial index: {len(indices.spatial_index.object_ids)} objects")
        
        # Check region index
        if indices.region_index is not None:
            logger.info(f"  Region index: {len(indices.region_index.regions)} regions")
            for r in indices.region_index.regions[:3]:
                logger.debug(f"    Region {r['region_id']}: {len(r['object_ids'])} objects")
        
        return indices
    except Exception as e:
        logger.exception(f"Index building failed: {e}")
        return None


def test_query_parser():
    """Test query parser."""
    logger.info("=" * 60)
    logger.info("Testing query parser...")
    
    from conceptgraph.query_scene import QueryParser, QueryType
    
    parser = QueryParser()
    
    test_queries = [
        "lamp",
        "the chair near the table",
        "沙发旁边的台灯",
        "how many chairs are there",
        "red pillow",
    ]
    
    for query in test_queries:
        logger.info(f"Parsing: '{query}'")
        try:
            result = parser.parse(query)
            logger.success(f"  target={result.target}, anchor={result.anchor}, "
                         f"relation={result.relation}, type={result.query_type.value}, "
                         f"use_bev={result.use_bev}")
        except Exception as e:
            logger.error(f"  Parse failed: {e}")
    
    return True


def test_full_pipeline(scene_path: str, query: str = "lamp"):
    """Test full pipeline."""
    logger.info("=" * 60)
    logger.info(f"Testing full pipeline with query: '{query}'")
    
    from conceptgraph.query_scene import QueryScenePipeline
    
    try:
        logger.info("Creating pipeline...")
        pipeline = QueryScenePipeline.from_scene(
            scene_path,
            # use_vlm defaults to True in QueryScenePipeline
        )
        logger.success("Pipeline created")
        logger.info(f"Pipeline summary: {pipeline.summary()}")
        
        logger.info(f"Running query: '{query}'")
        result = pipeline.query(query)
        
        if result.success:
            logger.success(f"Query successful!")
            logger.info(f"  Object ID: {result.object_id}")
            logger.info(f"  Category: {result.object_node.category if result.object_node else 'N/A'}")
            logger.info(f"  Centroid: {result.centroid}")
            logger.info(f"  Confidence: {result.confidence}")
            logger.info(f"  Reasoning: {result.reasoning}")
        else:
            logger.warning(f"Query failed: {result.reason}")
        
        return result
    except Exception as e:
        logger.exception(f"Pipeline test failed: {e}")
        return None


def main():
    """Main debug function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug QueryScene pipeline")
    parser.add_argument("--scene", type=str, help="Scene path")
    parser.add_argument("--query", type=str, default="lamp", help="Test query")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["imports", "scene", "index", "parser", "pipeline", "all"],
                       help="Test to run")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("QueryScene Debug Runner")
    logger.info("=" * 60)
    
    # Default scene path
    if args.scene is None:
        import os
        replica_root = os.environ.get("REPLICA_ROOT", "/home/shyue/Datasets/Replica/Replica")
        args.scene = f"{replica_root}/room0"
    
    logger.info(f"Scene path: {args.scene}")
    logger.info(f"Test query: {args.query}")
    logger.info(f"Test mode: {args.test}")
    
    if args.test in ["imports", "all"]:
        if not test_imports():
            logger.error("Import test failed, stopping")
            return 1
    
    if args.test in ["parser", "all"]:
        test_query_parser()
    
    if args.test in ["scene", "index", "pipeline", "all"]:
        result = test_scene_loading(args.scene)
        if result is None:
            logger.error("Scene loading failed, stopping")
            return 1
        scene, pcd_file = result
        
        if args.test in ["index", "all"]:
            indices = test_index_building(scene)
        
        if args.test in ["pipeline", "all"]:
            test_full_pipeline(args.scene, args.query)
    
    logger.success("Debug run completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
