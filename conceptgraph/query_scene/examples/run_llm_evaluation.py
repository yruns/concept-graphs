#!/usr/bin/env python3
"""
Run LLM-based keyframe evaluation on room0 scene.

Usage:
    python -m conceptgraph.query_scene.examples.run_llm_evaluation
    python -m conceptgraph.query_scene.examples.run_llm_evaluation --queries 5
    python -m conceptgraph.query_scene.examples.run_llm_evaluation --mode with_bev_context
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


def get_scene_path() -> Path:
    """Get scene path from environment or default."""
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    return Path(replica_root) / "room0"


def run_single_evaluation():
    """Run a single evaluation test."""
    from conceptgraph.query_scene.llm_evaluator import (
        LLMEvaluator,
        EvaluationInput,
        HypothesisKind,
    )
    from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

    scene_path = get_scene_path()
    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        return

    logger.info(f"Loading scene from: {scene_path}")

    # Initialize keyframe selector with LLM model
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gemini-2.5-pro",  # Required for query parsing
    )
    logger.info(f"Loaded {len(selector.objects)} objects")

    # Test query
    query = "the pillow on the sofa"

    # Run keyframe selection (this parses and executes internally)
    logger.info(f"Running keyframe selection for: '{query}'")
    keyframe_result = selector.select_keyframes_v2(query=query, k=3)

    logger.info(f"Selected {len(keyframe_result.keyframe_paths)} keyframes")
    logger.info(f"Status: {keyframe_result.metadata.get('status', 'unknown')}")

    if not keyframe_result.keyframe_paths:
        logger.warning("No keyframes selected, cannot evaluate")
        return

    # Get hypothesis info from metadata
    hypo_dump = keyframe_result.metadata.get("hypothesis_output", {})
    hypotheses = hypo_dump.get("hypotheses", [])

    if not hypotheses:
        logger.error("No hypothesis found in result")
        return

    hypo = hypotheses[0]

    # Extract from grounding_query (v1 format)
    grounding_query = hypo.get("grounding_query", {})
    root = grounding_query.get("root", {})
    target_cats = root.get("categories", [])

    # Extract anchor categories from spatial_constraints
    anchor_cats = []
    spatial_rel = None
    spatial_constraints = root.get("spatial_constraints", [])
    if spatial_constraints:
        sc = spatial_constraints[0]
        spatial_rel = sc.get("relation")
        anchors = sc.get("anchors", [])
        if anchors:
            anchor_cats = anchors[0].get("categories", [])

    eval_input = EvaluationInput(
        query=query,
        keyframe_paths=keyframe_result.keyframe_paths,
        target_categories=target_cats,
        anchor_categories=anchor_cats,
        spatial_relation=spatial_rel,
        hypothesis_kind=HypothesisKind(hypo.get("kind", "direct").lower()),
        matched_object_count=len(keyframe_result.target_objects),
        view_ids=keyframe_result.keyframe_indices,
        resolved_frame_ids=[m["resolved_frame_id"] for m in keyframe_result.metadata.get("frame_mappings", [])],
    )

    logger.info(f"Evaluation input:")
    logger.info(f"  - target_categories: {eval_input.target_categories}")
    logger.info(f"  - anchor_categories: {eval_input.anchor_categories}")
    logger.info(f"  - spatial_relation: {eval_input.spatial_relation}")
    logger.info(f"  - hypothesis_kind: {eval_input.hypothesis_kind}")

    # Create evaluator and run
    evaluator = LLMEvaluator(temperature=0.1, timeout=180, max_rounds=3)

    logger.info("Calling Gemini for evaluation...")
    result = evaluator.evaluate_single(eval_input)

    # Print results
    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Query: {result.query}")
    logger.info(f"Overall Score: {result.overall_score:.1f}/10")
    logger.info(f"Best Frame: {result.best_frame_idx}")
    logger.info(f"Retry Count: {result.retry_count}")

    logger.info("\nDimension Scores:")
    for dim, score in result.dimension_scores.items():
        if score is not None:
            logger.info(f"  - {dim.value}: {score:.1f}")
        else:
            logger.info(f"  - {dim.value}: N/A")

    logger.info(f"\nReasoning: {result.reasoning}")

    if result.issues:
        logger.info("\nIssues:")
        for issue in result.issues:
            logger.info(f"  - {issue}")

    if result.suggestions:
        logger.info("\nSuggestions:")
        for suggestion in result.suggestions:
            logger.info(f"  - {suggestion}")

    logger.info("\nPer-Frame Evaluations:")
    for fe in result.per_frame_evaluations:
        logger.info(f"  Frame {fe.frame_idx} (view_id={fe.view_id}):")
        logger.info(f"    - target_visibility: {fe.target_visibility}")
        logger.info(f"    - target_completeness: {fe.target_completeness}")
        if fe.spatial_context is not None:
            logger.info(f"    - spatial_context: {fe.spatial_context}")
        if fe.anchor_visibility is not None:
            logger.info(f"    - anchor_visibility: {fe.anchor_visibility}")
        logger.info(f"    - image_quality: {fe.image_quality}")
        logger.info(f"    - observations: {fe.observations[:100]}...")

    # Save result
    output_dir = scene_path / "query_visualizations"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "llm_evaluation_result.json"

    with open(output_file, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
    logger.info(f"\nResult saved to: {output_file}")


def run_batch_evaluation(num_queries: int = 5, mode: str = "keyframe_only"):
    """Run batch evaluation on multiple queries."""
    from conceptgraph.query_scene.llm_evaluator import (
        LLMEvaluator,
        EvaluationInput,
        HypothesisKind,
    )
    from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

    scene_path = get_scene_path()
    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        return

    # Test queries
    test_queries = [
        "the pillow on the sofa",
        "the lamp near the desk",
        "the plant on the shelf",
        "the book on the table",
        "the chair next to the window",
        "the picture on the wall",
        "the rug on the floor",
        "the cushion on the couch",
        "the vase on the cabinet",
        "the monitor on the desk",
    ][:num_queries]

    logger.info(f"Running batch evaluation on {len(test_queries)} queries")
    logger.info(f"Mode: {mode}")

    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gemini-2.5-pro",
    )
    evaluator = LLMEvaluator(temperature=0.1, timeout=180, max_rounds=3)

    results = []
    for i, query in enumerate(test_queries):
        logger.info(f"\n[{i+1}/{len(test_queries)}] Processing: '{query}'")

        try:
            # Select keyframes (parses and executes internally)
            keyframe_result = selector.select_keyframes_v2(query=query, k=3)

            if not keyframe_result.keyframe_paths:
                logger.warning(f"  No keyframes selected, skipping")
                results.append({"query": query, "status": "no_keyframes"})
                continue

            # Get hypothesis info from metadata
            hypo_dump = keyframe_result.metadata.get("hypothesis_output", {})
            hypotheses = hypo_dump.get("hypotheses", [])

            if not hypotheses:
                logger.warning(f"  No hypothesis found, skipping")
                results.append({"query": query, "status": "no_hypothesis"})
                continue

            hypo = hypotheses[0]

            # Extract from grounding_query (v1 format)
            grounding_query = hypo.get("grounding_query", {})
            root = grounding_query.get("root", {})
            target_cats = root.get("categories", [])

            # Extract anchor categories
            anchor_cats = []
            spatial_rel = None
            spatial_constraints = root.get("spatial_constraints", [])
            if spatial_constraints:
                sc = spatial_constraints[0]
                spatial_rel = sc.get("relation")
                anchors = sc.get("anchors", [])
                if anchors:
                    anchor_cats = anchors[0].get("categories", [])

            eval_input = EvaluationInput(
                query=query,
                keyframe_paths=keyframe_result.keyframe_paths,
                target_categories=target_cats,
                anchor_categories=anchor_cats,
                spatial_relation=spatial_rel,
                hypothesis_kind=HypothesisKind(hypo.get("kind", "direct").lower()),
                matched_object_count=len(keyframe_result.target_objects),
                view_ids=keyframe_result.keyframe_indices,
                resolved_frame_ids=[
                    m["resolved_frame_id"]
                    for m in keyframe_result.metadata.get("frame_mappings", [])
                ],
            )

            # Evaluate
            result = evaluator.evaluate_single(eval_input, mode=mode)
            logger.info(f"  Score: {result.overall_score:.1f}/10")

            results.append({
                "query": query,
                "status": "success",
                "overall_score": result.overall_score,
                "dimension_scores": {k.value: v for k, v in result.dimension_scores.items()},
                "reasoning": result.reasoning,
                "retry_count": result.retry_count,
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"query": query, "status": "error", "error": str(e)})

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)

    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        avg_score = sum(r["overall_score"] for r in successful) / len(successful)
        logger.info(f"Successful: {len(successful)}/{len(results)}")
        logger.info(f"Average Score: {avg_score:.2f}/10")

        # Score distribution
        bins = {"0-3": 0, "4-6": 0, "7-8": 0, "9-10": 0}
        for r in successful:
            s = r["overall_score"]
            if s <= 3:
                bins["0-3"] += 1
            elif s <= 6:
                bins["4-6"] += 1
            elif s <= 8:
                bins["7-8"] += 1
            else:
                bins["9-10"] += 1
        logger.info(f"Score Distribution: {bins}")

    # Save results
    output_dir = scene_path / "query_visualizations"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "batch_evaluation_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based keyframe evaluation")
    parser.add_argument(
        "--queries",
        type=int,
        default=1,
        help="Number of queries to evaluate (1 = single test, >1 = batch)",
    )
    parser.add_argument(
        "--mode",
        choices=["keyframe_only", "with_bev_context"],
        default="keyframe_only",
        help="Evaluation mode",
    )

    args = parser.parse_args()

    if args.queries == 1:
        run_single_evaluation()
    else:
        run_batch_evaluation(num_queries=args.queries, mode=args.mode)


if __name__ == "__main__":
    main()
