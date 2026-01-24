#!/usr/bin/env python3
"""
Query-driven keyframe selection demo.

Usage:
    python -m conceptgraph.query_scene.examples.query_keyframes \
        --scene_path /path/to/scene \
        --query "pillow on the sofa" \
        --k 3
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query-driven keyframe selection"
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to scene directory",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query, e.g., 'pillow on the sofa'",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of keyframes to select (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualization (default: scene_path/query_results)",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=None,
        help="LLM server URL (default: LLM_BASE_URL env var)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="LLM model name (default: LLM_MODEL env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import here to avoid slow startup for --help
    from conceptgraph.query_scene.keyframe_selector import KeyframeSelector
    
    scene_path = Path(args.scene_path)
    output_dir = Path(args.output_dir) if args.output_dir else scene_path / "query_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build kwargs for optional LLM config
    kwargs = {}
    if args.llm_url:
        kwargs["llm_url"] = args.llm_url
    if args.llm_model:
        kwargs["llm_model"] = args.llm_model
    
    # Load scene
    logger.info(f"Loading scene from: {scene_path}")
    selector = KeyframeSelector.from_scene_path(scene_path, **kwargs)
    
    # Run query
    logger.info(f"Running query: '{args.query}'")
    result = selector.select_keyframes(args.query, k=args.k)
    
    # Print results
    print()
    print("=" * 50)
    print(f"Query: {args.query}")
    print(f"Target: {result.target_term} -> {len(result.target_objects)} objects")
    if result.anchor_term:
        print(f"Anchor: {result.anchor_term} -> {len(result.anchor_objects)} objects")
    print(f"Selected keyframes: {result.keyframe_indices}")
    print("=" * 50)
    
    # Save visualization
    if result.keyframe_paths:
        images = []
        for i, (idx, path) in enumerate(zip(result.keyframe_indices, result.keyframe_paths)):
            if path.exists():
                img = cv2.imread(str(path))
                cv2.putText(
                    img, f"View {idx} (rank {i+1})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                images.append(img)
        
        if images:
            combined = np.hstack(images[:5])
            safe_name = args.query.replace(" ", "_")[:30]
            out_path = output_dir / f"{safe_name}.jpg"
            cv2.imwrite(str(out_path), combined)
            print(f"Saved: {out_path}")
    
    return result


if __name__ == "__main__":
    main()
