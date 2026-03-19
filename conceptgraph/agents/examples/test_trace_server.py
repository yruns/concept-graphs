#!/usr/bin/env python
"""Test SQLite-backed trace server with real agent execution.

This script:
1. Runs several agent executions with TracingAgent wrapper
2. Starts the TraceServer
3. Opens the browser to view traces

Usage:
    REPLICA_ROOT=/path/to/Replica .venv/bin/python -m conceptgraph.agents.examples.test_trace_server
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from conceptgraph.agents import (
    Stage1BackendCallbacks,
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2PlanMode,
    Stage2TaskSpec,
    Stage2TaskType,
    build_stage2_evidence_bundle,
)
from conceptgraph.agents.trace_server import TraceDB, TracingAgent, TraceServer
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector


def run_traced_queries(scene_path: Path, db: TraceDB) -> None:
    """Run a few test queries with tracing."""
    logger.info("[Setup] Loading KeyframeSelector...")
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )
    logger.info(f"  Loaded {len(selector.objects)} objects")

    # Create base agent
    callbacks = Stage1BackendCallbacks(
        keyframe_selector=selector,
        scene_id=scene_path.name,
        max_additional_views=3,
    )

    base_agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(include_thoughts=False),
        more_views_callback=callbacks.more_views,
        crop_callback=callbacks.crops,
        hypothesis_callback=callbacks.hypothesis,
    )

    # Wrap with tracing
    agent = TracingAgent(base_agent, db)

    # Test cases with varying difficulty
    test_cases = [
        # Easy: should complete quickly, maybe no tools
        ("pillow on the sofa", "What color is the pillow?", 1),
        # Medium: might need more context
        ("table and chairs", "How many chairs are around the table?", 2),
        # Hard: likely needs multiple views
        (
            "window and furniture",
            "Describe the spatial layout around the window. What furniture is nearby?",
            1,
        ),
    ]

    for i, (stage1_query, task_query, k) in enumerate(test_cases, 1):
        logger.info("=" * 60)
        logger.info(f"[Test {i}/{len(test_cases)}] {task_query[:50]}...")
        logger.info("=" * 60)

        try:
            # Stage 1: retrieve keyframes
            keyframe_result = selector.select_keyframes_v2(stage1_query, k=k)
            logger.info(f"  Stage 1: {len(keyframe_result.keyframe_paths)} keyframes")

            bundle = build_stage2_evidence_bundle(
                keyframe_result,
                scene_id=scene_path.name,
                scene_summary=f"Replica scene with {len(selector.objects)} objects.",
            )

            # Stage 2: run traced agent
            task = Stage2TaskSpec(
                task_type=Stage2TaskType.QA,
                user_query=task_query,
                plan_mode=Stage2PlanMode.BRIEF,
                max_reasoning_turns=4,
            )

            result = agent.run(
                task,
                bundle,
                metadata={"test_case": i, "stage1_query": stage1_query},
            )

            logger.success(
                f"  Result: status={result.result.status.value}, "
                f"conf={result.result.confidence:.2f}, "
                f"tools={len(result.tool_trace)}"
            )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback

            traceback.print_exc()


def main() -> None:
    # Find Replica scene
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_path = Path(replica_root) / "room0"

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        logger.error("Set REPLICA_ROOT environment variable")
        sys.exit(1)

    # Initialize database
    db_path = Path("traces_stage2.db")
    db = TraceDB(db_path)
    logger.info(f"[DB] Using database: {db_path.absolute()}")

    # Check existing stats
    stats = db.get_stats()
    logger.info(f"[DB] Existing traces: {stats['total_traces']}")

    # Ask user if they want to run more traces
    if stats["total_traces"] >= 3:
        logger.info("Database already has traces. Skipping to server.")
    else:
        logger.info("Running test queries to populate database...")
        run_traced_queries(scene_path, db)

    # Show final stats
    stats = db.get_stats()
    logger.info("-" * 40)
    logger.info("Database Statistics:")
    logger.info(f"  Total traces: {stats['total_traces']}")
    logger.info(f"  By status: {stats['by_status']}")
    logger.info(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    logger.info(f"  Avg duration: {stats['avg_duration_ms']:.0f}ms")
    logger.info(f"  Avg tool calls: {stats['avg_tool_calls']:.1f}")
    logger.info("-" * 40)

    # Start server
    server = TraceServer(db, port=8765)
    url = f"http://127.0.0.1:8765/"

    # Open browser
    logger.info(f"Opening browser: {url}")
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False)
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", url], check=False)

    logger.info("Starting trace server... (Ctrl+C to stop)")
    server.run()


if __name__ == "__main__":
    main()
