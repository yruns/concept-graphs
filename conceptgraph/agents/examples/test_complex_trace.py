#!/usr/bin/env python
"""Test trace visualization with a complex query that triggers tool calls.

This deliberately uses a harder query that requires multiple views or tool usage.
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
    save_trace_report,
)
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector


def main():
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_path = Path(replica_root) / "room0"

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Complex Query Test - Should Trigger Tool Usage")
    logger.info("=" * 70)

    # Load selector
    logger.info("[Stage 1] Loading KeyframeSelector...")
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )

    # Use a more complex query that needs verification
    stage1_query = "furniture near the window"
    task_query = (
        "Analyze the spatial layout around the window. "
        "What furniture is near the window? "
        "Is there anything blocking natural light? "
        "Describe the relative positions of objects."
    )

    logger.info(f"Stage 1 query: {stage1_query}")
    logger.info(f"Task query: {task_query}")

    # Retrieve only 1 keyframe to force the agent to request more
    logger.info("[Stage 1] Retrieving minimal keyframes (k=1)...")
    keyframe_result = selector.select_keyframes_v2(stage1_query, k=1)
    logger.info(f"  Found {len(keyframe_result.keyframe_paths)} keyframe")

    bundle = build_stage2_evidence_bundle(
        keyframe_result,
        scene_id=scene_path.name,
        scene_summary=f"Living room scene with {len(selector.objects)} detected objects including furniture and decor.",
    )

    # Create agent with FULL plan mode to encourage tool usage
    logger.info("[Stage 2] Creating agent with BRIEF plan mode...")
    callbacks = Stage1BackendCallbacks(
        keyframe_selector=selector,
        scene_id=scene_path.name,
        max_additional_views=3,
    )

    agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(include_thoughts=False),
        more_views_callback=callbacks.more_views,
        crop_callback=callbacks.crops,
        hypothesis_callback=callbacks.hypothesis,
    )

    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query=task_query,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=5,
    )

    logger.info("[Stage 2] Running agent...")
    result = agent.run(task, bundle)

    logger.info("-" * 40)
    logger.info("RESULT")
    logger.info("-" * 40)
    logger.info(f"Status: {result.result.status.value}")
    logger.info(f"Confidence: {result.result.confidence:.2f}")
    logger.info(f"Tool calls: {len(result.tool_trace)}")
    for tc in result.tool_trace:
        logger.info(f"  - {tc.tool_name}: {tc.tool_input}")
    logger.info(f"Initial keyframes: {len(bundle.keyframes)}")
    logger.info(f"Final keyframes: {len(result.final_bundle.keyframes)}")
    logger.info(f"Summary: {result.result.summary}")

    # Generate trace report
    output_dir = Path("trace_reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"trace_complex_window_{int(__import__('time').time())}.html"

    save_trace_report(
        result=result,
        task=task,
        initial_bundle=bundle,
        output_path=report_path,
    )
    logger.success(f"Report: {report_path}")

    # Open in browser
    if sys.platform == "darwin":
        subprocess.run(["open", str(report_path)], check=False)


if __name__ == "__main__":
    main()
