#!/usr/bin/env python3
"""Demo script for QueryScene on Replica room0."""
from __future__ import annotations
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conceptgraph.query_scene.query_pipeline import QueryScenePipeline
from conceptgraph.query_scene.utils import format_result


def main():
    """Run demo queries on Replica room0."""
    import os
    
    # Get scene path from environment or default
    replica_root = os.environ.get("REPLICA_ROOT", "/home/shyue/Datasets/Replica/Replica")
    scene_path = Path(replica_root) / "room0"
    
    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        logger.info("Please set REPLICA_ROOT environment variable or ensure the path exists.")
        return
    
    logger.info("=" * 60)
    logger.info("QueryScene Demo - Replica room0")
    logger.info("=" * 60)
    
    # Find pcd file
    pcd_dir = scene_path / "pcd_saves"
    pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list(pcd_dir.glob("*.pkl.gz"))
    
    if not pcd_files:
        logger.error(f"No pcd file found in {pcd_dir}")
        logger.info("Please run the ConceptGraphs pipeline first:")
        logger.info("  bash bashes/1_extract_2d_segmentation.sh")
        logger.info("  bash bashes/2_build_3d_object_map.sh")
        return
    
    pcd_file = pcd_files[0]
    logger.info(f"Using pcd file: {pcd_file.name}")
    
    # Create pipeline
    # LLM model must be explicitly specified
    llm_model = os.environ.get("LLM_MODEL")
    if not llm_model:
        logger.error("LLM_MODEL environment variable must be set.")
        logger.info("Available models: gpt-4o-2024-08-06, gemini-2.5-pro, gemini-3-pro-preview-new, gemini-3-flash-preview")
        return
    
    logger.info(f"Loading scene and building indices (using LLM: {llm_model})...")
    try:
        pipeline = QueryScenePipeline.from_scene(
            str(scene_path),
            str(pcd_file),
            stride=5,
            llm_model=llm_model,
        )
    except Exception as e:
        logger.error(f"Error loading scene: {e}")
        import traceback
        logger.exception("Stack trace:")
        return
    
    # Print scene summary
    summary = pipeline.summary()
    logger.info("Scene loaded:")
    logger.info(f"  Objects: {summary['scene']['n_objects']}")
    logger.info(f"  Views: {summary['scene']['n_views']}")
    logger.info(f"  Categories: {list(summary['scene']['categories'].keys())[:10]}...")
    
    # Demo queries
    queries = [
        "椅子",
        "桌子",
        "沙发旁边的台灯",
        "窗户附近的物体",
    ]
    
    logger.info("=" * 60)
    logger.info("Running queries...")
    logger.info("=" * 60)
    
    for query in queries:
        logger.info(f">>> Query: \"{query}\"")
        try:
            result = pipeline.query(query)
            logger.info(format_result(result))
        except Exception as e:
            logger.error(f"{e}")
    
    # Interactive mode
    logger.info("=" * 60)
    logger.info("Interactive mode (type 'quit' to exit)")
    logger.info("=" * 60)
    
    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            result = pipeline.query(query)
            logger.info(format_result(result))
        except KeyboardInterrupt:
            logger.info("Bye!")
            break
        except Exception as e:
            logger.error(f"{e}")


if __name__ == "__main__":
    main()
