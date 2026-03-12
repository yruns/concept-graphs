#!/usr/bin/env python3
"""Generate evaluation cases with ground truth target annotations.

Usage:
    python -m conceptgraph.query_scene.examples.generate_eval_cases
    python -m conceptgraph.query_scene.examples.generate_eval_cases --num_cases 50
    python -m conceptgraph.query_scene.examples.generate_eval_cases --scene room0 --output eval_cases.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


def get_scene_path(scene_name: str = "room0") -> Path:
    """Get scene path from environment or default."""
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    return Path(replica_root) / scene_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation cases with ground truth"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="room0",
        help="Scene name (default: room0)",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=20,
        help="Number of cases to generate (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <scene>/query_visualizations/eval_cases.json)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )

    args = parser.parse_args()

    scene_path = get_scene_path(args.scene)
    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        sys.exit(1)

    logger.info(f"Scene: {scene_path}")
    logger.info(f"Generating {args.num_cases} evaluation cases...")

    # Import after path setup
    from conceptgraph.query_scene.query_case_generator import (
        QueryCaseGenerator,
    )

    # Initialize generator
    generator = QueryCaseGenerator(
        scene_path=scene_path,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    logger.info(f"Loaded {len(generator.objects)} objects")
    logger.info(f"Categories: {generator.scene_categories}")

    # Generate cases
    batch = generator.generate_cases(
        num_cases=args.num_cases,
        target_distribution={1: 0.7, 2: 0.25, 3: 0.05},
    )

    # Summary
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total generated: {batch.total_generated}")
    logger.info(f"Failed: {batch.failed_count}")
    logger.info(f"Validation passed: {batch.validation_passed}")
    logger.info(f"Pass rate: {batch.validation_passed / max(1, batch.total_generated) * 100:.1f}%")

    # Distribution stats
    actual_dist = batch.generation_config.get("actual_distribution", {})
    logger.info(f"Actual target distribution: {actual_dist}")

    # Difficulty distribution
    difficulty_counts = {}
    type_counts = {}
    for case in batch.cases:
        diff = case.difficulty.value
        qtype = case.query_type.value
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    logger.info(f"Difficulty distribution: {difficulty_counts}")
    logger.info(f"Query type distribution: {type_counts}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_dir = scene_path / "query_visualizations"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "eval_cases.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(batch.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")

    # Print sample cases
    logger.info("\nSample cases:")
    for i, case in enumerate(batch.cases[:5]):
        status = "✓" if case.validated else "⚠"
        logger.info(f"  {i+1}. [{status}] \"{case.query}\"")
        logger.info(f"      targets: {case.target_obj_ids} ({case.target_categories})")
        if case.anchor_obj_ids:
            logger.info(f"      anchors: {case.anchor_obj_ids} ({case.anchor_categories})")
        logger.info(f"      type: {case.query_type.value}, difficulty: {case.difficulty.value}")
        if case.validation_errors:
            logger.info(f"      errors: {case.validation_errors}")


if __name__ == "__main__":
    main()
