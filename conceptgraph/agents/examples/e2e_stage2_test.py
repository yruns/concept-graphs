#!/usr/bin/env python
"""End-to-end test of Stage 1 -> Stage 2 pipeline with real Replica scene data.

This script:
1. Loads a Replica scene via KeyframeSelector
2. Runs Stage 1 keyframe retrieval
3. Creates Stage 2 agent with real `request_more_views` callback
4. Runs Stage 2 reasoning task
5. Verifies the full pipeline works with real visual evidence

Run: REPLICA_ROOT=/path/to/Replica .venv/bin/python -m conceptgraph.agents.examples.e2e_stage2_test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

# Ensure we can import from project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from conceptgraph.agents import (
    Stage1BackendCallbacks,
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2TaskSpec,
    Stage2TaskType,
    build_stage2_evidence_bundle,
)
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector


def run_e2e_stage2_test(
    scene_path: Path,
    query: str,
    task_query: Optional[str] = None,
    k: int = 3,
) -> dict:
    """Run end-to-end Stage 1 -> Stage 2 test.

    Args:
        scene_path: Path to Replica scene directory
        query: Stage 1 query for keyframe retrieval
        task_query: Stage 2 task query (defaults to same as query)
        k: Number of initial keyframes

    Returns:
        Dict with test results
    """
    task_query = task_query or f"Describe what you see related to: {query}"

    logger.info("=" * 70)
    logger.info(f"E2E Stage 2 Test")
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Stage 1 query: {query}")
    logger.info(f"Stage 2 task: {task_query}")
    logger.info("=" * 70)

    # Step 1: Initialize KeyframeSelector
    logger.info("[Step 1] Initializing KeyframeSelector...")
    try:
        selector = KeyframeSelector.from_scene_path(
            str(scene_path),
            llm_model="gpt-5.2-2025-12-11",  # Use GPT for parsing to match Stage 2
            use_pool=False,  # Single key for consistency
        )
        logger.info(f"  Loaded {len(selector.objects)} scene objects")
    except Exception as e:
        logger.error(f"  Failed to initialize KeyframeSelector: {e}")
        return {"success": False, "error": f"KeyframeSelector init failed: {e}"}

    # Step 2: Run Stage 1 keyframe retrieval
    logger.info("[Step 2] Running Stage 1 keyframe retrieval...")
    try:
        keyframe_result = selector.select_keyframes_v2(query, k=k)

        logger.info(f"  Status: {keyframe_result.metadata.get('status')}")
        logger.info(f"  Keyframes: {len(keyframe_result.keyframe_paths)}")
        logger.info(f"  Target: {keyframe_result.target_term}")
        logger.info(f"  Anchor: {keyframe_result.anchor_term}")

        if not keyframe_result.keyframe_paths:
            logger.warning("  No keyframes found!")
            return {"success": False, "error": "No keyframes from Stage 1"}

        for i, path in enumerate(keyframe_result.keyframe_paths):
            logger.info(f"    [{i}] {Path(path).name}")

    except Exception as e:
        logger.error(f"  Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Stage 1 failed: {e}"}

    # Step 3: Build Stage 2 evidence bundle
    logger.info("[Step 3] Building Stage 2 evidence bundle...")
    bundle = build_stage2_evidence_bundle(
        keyframe_result,
        scene_id=scene_path.name,
        scene_summary=f"Replica scene {scene_path.name} with {len(selector.objects)} objects.",
    )

    logger.info(f"  Bundle keyframes: {len(bundle.keyframes)}")
    logger.info(f"  Object context keys: {list(bundle.object_context.keys())[:5]}...")

    # Step 4: Create Stage 2 agent with real callbacks
    logger.info("[Step 4] Creating Stage 2 agent with Stage 1 callbacks...")
    callbacks = Stage1BackendCallbacks(
        keyframe_selector=selector,
        scene_id=scene_path.name,
        max_additional_views=2,
    )

    agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(
            include_thoughts=False,
        ),
        more_views_callback=callbacks.more_views,
        crop_callback=callbacks.crops,
        hypothesis_callback=callbacks.hypothesis,
    )

    # Step 5: Run Stage 2 task
    logger.info("[Step 5] Running Stage 2 agent...")
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query=task_query,
        max_reasoning_turns=3,
    )

    try:
        result = agent.run(task, bundle)

        logger.info(f"\n{'=' * 70}")
        logger.info("Stage 2 Result")
        logger.info(f"{'=' * 70}")
        logger.info(f"Status: {result.result.status.value}")
        logger.info(f"Confidence: {result.result.confidence}")
        logger.info(f"Summary: {result.result.summary}")

        if result.result.payload:
            logger.info(f"Payload: {json.dumps(result.result.payload, indent=2)}")

        if result.result.uncertainties:
            logger.info(f"Uncertainties: {result.result.uncertainties}")

        if result.result.cited_frame_indices:
            logger.info(f"Cited frames: {result.result.cited_frame_indices}")

        logger.info(f"\nTool trace: {len(result.tool_trace)} calls")
        for trace in result.tool_trace:
            logger.info(f"  - {trace.tool_name}: {trace.tool_input}")

        # Check if more_views was called and evidence was updated
        more_views_calls = [t for t in result.tool_trace if t.tool_name == "request_more_views"]
        final_keyframe_count = len(result.final_bundle.keyframes)

        logger.info(f"\nFinal bundle keyframes: {final_keyframe_count}")
        if final_keyframe_count > len(bundle.keyframes):
            logger.success(f"Evidence refinement worked! Added {final_keyframe_count - len(bundle.keyframes)} new keyframes.")

        return {
            "success": True,
            "status": result.result.status.value,
            "summary": result.result.summary,
            "confidence": result.result.confidence,
            "tool_calls": len(result.tool_trace),
            "more_views_calls": len(more_views_calls),
            "initial_keyframes": len(bundle.keyframes),
            "final_keyframes": final_keyframe_count,
        }

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Stage 2 failed: {e}"}


def main():
    # Get REPLICA_ROOT from environment
    replica_root = os.environ.get("REPLICA_ROOT")
    if not replica_root:
        # Try common locations
        candidates = [
            Path.home() / "Datasets" / "Replica",
            Path.home() / "Replica",
            Path("/Users/bytedance/Replica"),
        ]
        for candidate in candidates:
            if candidate.exists():
                replica_root = str(candidate)
                break

    if not replica_root:
        logger.error("REPLICA_ROOT not set and no common path found.")
        logger.info("Usage: REPLICA_ROOT=/path/to/Replica python -m conceptgraph.agents.examples.e2e_stage2_test")
        sys.exit(1)

    scene_path = Path(replica_root) / "room0"
    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    # Test cases: (stage1_query, stage2_task_query)
    test_cases = [
        # Simple case - should work without needing more views
        (
            "pillow on the sofa",
            "What color is the pillow? Is it on the sofa?"
        ),
        # Case that might need more views for full answer
        (
            "chair near table",
            "Describe the chair and its relationship to nearby furniture."
        ),
    ]

    results = []
    for query, task_query in test_cases[:1]:  # Run first test case
        logger.info("\n" + "=" * 70)
        result = run_e2e_stage2_test(
            scene_path=scene_path,
            query=query,
            task_query=task_query,
            k=2,  # Start with fewer keyframes to encourage more_views
        )
        results.append(result)
        logger.info(f"\nTest result: {result}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for i, (test, result) in enumerate(zip(test_cases[:1], results)):
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        logger.info(f"Test {i+1}: {status}")
        logger.info(f"  Query: {test[0]}")
        if result.get("success"):
            logger.info(f"  Summary: {result.get('summary', 'N/A')[:80]}...")


if __name__ == "__main__":
    main()
