#!/usr/bin/env python3
"""
Test KeyframeSelector parallel parsing with Gemini pool.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


def main():
    import os
    from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

    # Get scene path
    replica_root = os.environ.get("REPLICA_ROOT")
    scene_name = os.environ.get("SCENE_NAME", "room0")

    if replica_root:
        scene_path = Path(replica_root) / scene_name
    else:
        scene_path = project_root / scene_name
        logger.warning(f"REPLICA_ROOT not set, using fallback: {scene_path}")

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        return

    # Test queries
    test_queries = [
        "the coffee_table",
        "the floor_lamp",
        "a sofa",
        "find a couch",
        "the lamp",
        "the throw_pillow on the sofa",
        "the vase on the coffee_table",
        "the sofa nearest the coffee_table",
        "the throw_pillow on the sofa near the coffee_table",
        "the largest sofa",
    ]

    # Create selector with pool
    logger.info("=" * 70)
    logger.info("Parallel Parsing Test with Gemini Pool")
    logger.info("=" * 70)

    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gemini-2.5-pro",
        use_pool=True
    )
    logger.info(f"Loaded scene with {len(selector.objects)} objects")
    logger.info(f"Categories: {len(selector.scene_categories)}")

    # Sequential baseline
    logger.info("\n[Sequential Baseline]")
    start = time.time()
    sequential_results = []
    for q in test_queries[:5]:  # Only 5 for baseline
        try:
            hypo = selector.parse_query_hypotheses(q)
            sequential_results.append((q, True))
        except Exception as e:
            sequential_results.append((q, False))
    seq_time = time.time() - start
    logger.info(f"Sequential (5 queries): {seq_time:.2f}s ({seq_time/5:.2f}s/query)")

    # Parallel execution
    logger.info("\n[Parallel Execution]")
    start = time.time()

    def parse_query(query):
        try:
            hypo = selector.parse_query_hypotheses(query)
            return query, True, hypo.hypotheses[0].grounding_query.root.categories
        except Exception as e:
            return query, False, str(e)

    parallel_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_query, q): q for q in test_queries}
        for future in as_completed(futures):
            result = future.result()
            parallel_results.append(result)

    parallel_time = time.time() - start

    logger.info(f"Parallel ({len(test_queries)} queries): {parallel_time:.2f}s ({parallel_time/len(test_queries):.2f}s/query)")
    logger.info(f"Speedup vs sequential estimate: {(seq_time/5 * len(test_queries)) / parallel_time:.1f}x")

    # Results
    logger.info("\n[Results]")
    for query, success, info in parallel_results:
        status = "OK" if success else "FAIL"
        logger.info(f"  [{status}] {query[:40]:<40} -> {info}")

    success_count = sum(1 for _, s, _ in parallel_results if s)
    logger.info(f"\nSuccess: {success_count}/{len(test_queries)}")

    # Show pool statistics
    from conceptgraph.utils.llm_client import GeminiClientPool
    pool = GeminiClientPool.get_instance()
    stats = pool.get_stats()

    logger.info("\n[Pool Statistics]")
    for key_id, data in stats.items():
        if data["total_requests"] > 0:
            logger.info(
                f"  {key_id}: {data['total_requests']} requests, "
                f"{data['rate_limited']} rate limited ({data['rate_limit_ratio']*100:.1f}%), "
                f"weight={data['weight']}"
            )


if __name__ == "__main__":
    main()
