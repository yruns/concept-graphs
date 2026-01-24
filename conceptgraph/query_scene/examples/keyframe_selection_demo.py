#!/usr/bin/env python3
"""
Demo: Query-driven Keyframe Selection

This script demonstrates the keyframe selection pipeline on Replica room0.
It shows how natural language queries are processed to select the most
relevant camera viewpoints.

Usage:
    python keyframe_selection_demo.py
    python keyframe_selection_demo.py --scene /path/to/scene --query "the pillow on the sofa"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from loguru import logger

from conceptgraph.query_scene.keyframe_selector import KeyframeSelector, KeyframeResult


def visualize_keyframes(
    result: KeyframeResult,
    selector: KeyframeSelector,
    output_dir: Path,
    show_objects: bool = True,
):
    """Visualize selected keyframes with object annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Visualizing {len(result.keyframe_indices)} keyframes...")
    
    for i, (view_id, img_path) in enumerate(zip(result.keyframe_indices, result.keyframe_paths)):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load: {img_path}")
            continue
        
        # Add title
        title = f"Keyframe {i+1} (view {view_id})"
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        # Add query info
        query_text = f"Query: {result.query}"
        cv2.putText(img, query_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, query_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)
        
        # Add target objects info
        target_names = [o.object_tag or o.category for o in result.target_objects[:3]]
        target_text = f"Targets: {', '.join(target_names)}"
        cv2.putText(img, target_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, target_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        
        # Save
        output_path = output_dir / f"keyframe_{i+1}_view{view_id}.jpg"
        cv2.imwrite(str(output_path), img)
        logger.info(f"  Saved: {output_path}")
    
    # Create combined view
    if len(result.keyframe_paths) > 1:
        images = []
        for img_path in result.keyframe_paths[:4]:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize for grid
                h, w = img.shape[:2]
                scale = 400 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                images.append(img)
        
        if len(images) >= 2:
            # Create 2x2 grid
            h, w = images[0].shape[:2]
            grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            
            for idx, img in enumerate(images[:4]):
                row, col = idx // 2, idx % 2
                ih, iw = img.shape[:2]
                grid[row*h:row*h+ih, col*w:col*w+iw] = img
            
            combined_path = output_dir / "keyframes_combined.jpg"
            cv2.imwrite(str(combined_path), grid)
            logger.info(f"  Combined view: {combined_path}")


def run_demo(scene_path: str, queries: list, output_dir: str, k: int = 3):
    """Run the demo with multiple queries."""
    
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)
    
    logger.info("=" * 60)
    logger.info("Keyframe Selection Demo")
    logger.info("=" * 60)
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # Initialize selector
    logger.info("Initializing KeyframeSelector...")
    selector = KeyframeSelector.from_scene_path(scene_path, stride=5)
    
    # Print scene summary
    logger.info("")
    logger.info("Scene Summary:")
    logger.info(f"  Objects: {len(selector.objects)}")
    logger.info(f"  Views: {len(selector.camera_poses)}")
    logger.info(f"  Categories: {selector.scene_categories[:15]}...")
    logger.info("")
    
    # Run queries
    results = []
    for query in queries:
        logger.info("-" * 60)
        result = selector.select_keyframes(query, k=k)
        results.append(result)
        
        # Print result
        logger.info("")
        logger.info(result.summary())
        
        # Visualize
        query_dir = output_dir / query.replace(" ", "_")[:30]
        visualize_keyframes(result, selector, query_dir)
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Demo Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Keyframe Selection Demo")
    parser.add_argument(
        "--scene", 
        type=str, 
        default="/home/shyue/Datasets/Replica/Replica/room0",
        help="Path to scene directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to test (if not provided, runs default examples)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of keyframes to select"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(args.scene) / "query_results" / "keyframe_demo"
    
    # Define queries
    if args.query:
        queries = [args.query]
    else:
        # Default example queries
        queries = [
            # Simple object queries
            "table_lamp",
            "ottoman",
            "sideboard",
            
            # Spatial relation queries
            "the pillow on the sofa",
            "lamp near the couch",
            "throw_pillow beside the ottoman",
            
            # More complex queries
            "the storage furniture with books",
        ]
    
    # Run demo
    run_demo(args.scene, queries, output_dir, k=args.k)


if __name__ == "__main__":
    main()
