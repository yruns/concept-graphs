#!/usr/bin/env python3
"""
Evaluate KeyframeSelector with Gemini-generated queries.
"""
import sys
import json
import time
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


def main():
    import os
    from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

    # Get scene path
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_name = os.environ.get("SCENE_NAME", "room0")
    scene_path = Path(replica_root) / scene_name

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        return

    # Load generated queries
    queries_file = scene_path / "generated_queries_v2.json"
    if not queries_file.exists():
        logger.error(f"Generated queries not found: {queries_file}")
        logger.info("Run: python -m conceptgraph.query_scene.query_sample_generator_v2 100")
        return

    with open(queries_file) as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries from {queries_file.name}")

    # Create selector
    logger.info("=" * 70)
    logger.info("Initializing KeyframeSelector")
    logger.info("=" * 70)

    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gemini-2.5-pro",
        use_pool=True
    )
    logger.info(f"Loaded {len(selector.objects)} objects, {len(selector.scene_categories)} categories")

    # Run evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("Running Evaluation")
    logger.info("=" * 70)

    results = {"success": 0, "fail": 0, "error": 0}
    by_difficulty = {"easy": [], "medium": [], "hard": [], "expert": []}
    by_type = {}
    details = []

    start_time = time.time()

    for i, q in enumerate(queries):
        query_text = q["query"]
        difficulty = q.get("difficulty", "unknown")
        query_type = q.get("query_type", "unknown")
        special = q.get("special_case", "none")

        try:
            hypo = selector.parse_query_hypotheses(query_text)
            status, winning, result = selector.execute_hypotheses(hypo)

            matched = len(result.matched_objects) if result else 0
            success = matched > 0

            if success:
                results["success"] += 1
                symbol = "OK"
            else:
                results["fail"] += 1
                symbol = "FAIL"

            logger.info(
                f"[{i+1:02d}] {symbol:4} {difficulty:6} {query_type:12} "
                f"-> {matched} obj | {query_text[:40]}..."
            )

            detail = {
                "query": query_text,
                "difficulty": difficulty,
                "query_type": query_type,
                "special_case": special,
                "matched": matched,
                "success": success,
                "objects": [obj.object_tag for obj in result.matched_objects] if result else [],
            }
            details.append(detail)

            # Track by difficulty
            if difficulty in by_difficulty:
                by_difficulty[difficulty].append(success)

            # Track by type
            if query_type not in by_type:
                by_type[query_type] = []
            by_type[query_type].append(success)

        except Exception as e:
            results["error"] += 1
            logger.error(f"[{i+1:02d}] ERR  {difficulty:6} {query_type:12} -> {str(e)[:40]}")
            details.append({
                "query": query_text,
                "difficulty": difficulty,
                "query_type": query_type,
                "matched": 0,
                "success": False,
                "error": str(e),
            })

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)

    total = results["success"] + results["fail"]
    pass_rate = results["success"] / total * 100 if total > 0 else 0

    logger.info(f"Total: {results['success']}/{total} passed ({pass_rate:.1f}%)")
    logger.info(f"Errors: {results['error']}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/len(queries):.2f}s/query)")

    # By difficulty
    logger.info("")
    logger.info("By Difficulty:")
    for diff in ["easy", "medium", "hard", "expert"]:
        successes = by_difficulty.get(diff, [])
        if successes:
            rate = sum(successes) / len(successes) * 100
            logger.info(f"  {diff:8}: {sum(successes)}/{len(successes)} ({rate:.1f}%)")

    # By type
    logger.info("")
    logger.info("By Query Type:")
    for qtype, successes in sorted(by_type.items()):
        if successes:
            rate = sum(successes) / len(successes) * 100
            logger.info(f"  {qtype:12}: {sum(successes)}/{len(successes)} ({rate:.1f}%)")

    # Save results
    output = {
        "summary": {
            "total": total,
            "success": results["success"],
            "fail": results["fail"],
            "error": results["error"],
            "pass_rate": pass_rate,
            "time_seconds": elapsed,
        },
        "by_difficulty": {k: {"total": len(v), "success": sum(v)} for k, v in by_difficulty.items()},
        "by_type": {k: {"total": len(v), "success": sum(v)} for k, v in by_type.items()},
        "details": details,
    }

    output_file = scene_path / "eval_results_v2.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.success(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
