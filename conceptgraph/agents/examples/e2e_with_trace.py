#!/usr/bin/env python
"""End-to-end Stage 2 test with HTML trace visualization.

This script demonstrates the full pipeline with visual debugging:
1. Stage 1 keyframe retrieval
2. Stage 2 agentic reasoning
3. HTML trace report generation

Run: REPLICA_ROOT=/path/to/Replica .venv/bin/python -m conceptgraph.agents.examples.e2e_with_trace

Output: Opens an HTML report showing the execution trace with embedded images.
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
    Stage2TaskSpec,
    Stage2TaskType,
    build_stage2_evidence_bundle,
    save_trace_report,
)
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector


def run_with_trace(
    scene_path: Path,
    stage1_query: str,
    task_query: str,
    k: int = 2,
    max_turns: int = 4,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run Stage 1 -> Stage 2 pipeline and generate HTML trace report.

    Returns:
        Path to the generated HTML report
    """
    from typing import Optional

    output_dir = output_dir or Path("trace_reports")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("Stage 2 E2E with Trace Visualization")
    logger.info("=" * 70)

    # Stage 1: Keyframe retrieval
    logger.info("[Stage 1] Loading KeyframeSelector...")
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )
    logger.info(f"  Loaded {len(selector.objects)} objects")

    logger.info("[Stage 1] Retrieving keyframes...")
    keyframe_result = selector.select_keyframes_v2(stage1_query, k=k)
    logger.info(f"  Found {len(keyframe_result.keyframe_paths)} keyframes")
    logger.info(f"  Status: {keyframe_result.metadata.get('status')}")

    # Build evidence bundle
    bundle = build_stage2_evidence_bundle(
        keyframe_result,
        scene_id=scene_path.name,
        scene_summary=f"Replica scene {scene_path.name} with {len(selector.objects)} detected objects.",
    )

    # Stage 2: Create agent with real callbacks
    logger.info("[Stage 2] Creating agent with callbacks...")
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

    # Create task
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query=task_query,
        max_reasoning_turns=max_turns,
    )

    # Run agent
    logger.info("[Stage 2] Running agent...")
    result = agent.run(task, bundle)

    logger.info(f"  Status: {result.result.status.value}")
    logger.info(f"  Confidence: {result.result.confidence:.2f}")
    logger.info(f"  Summary: {result.result.summary[:100]}...")
    logger.info(f"  Tool calls: {len(result.tool_trace)}")
    logger.info(f"  Final keyframes: {len(result.final_bundle.keyframes)}")

    # Generate trace report
    timestamp = int(__import__("time").time())
    safe_query = "".join(c if c.isalnum() else "_" for c in task_query[:30])
    report_path = output_dir / f"trace_{safe_query}_{timestamp}.html"

    logger.info(f"[Trace] Generating HTML report: {report_path}")
    save_trace_report(
        result=result,
        task=task,
        initial_bundle=bundle,
        output_path=report_path,
    )

    return report_path


def main():
    # Find Replica scene
    replica_root = os.environ.get("REPLICA_ROOT")
    if not replica_root:
        candidates = [
            Path.home() / "Datasets" / "Replica",
            Path.home() / "Replica",
            Path("/Users/bytedance/Replica"),
        ]
        for c in candidates:
            if c.exists():
                replica_root = str(c)
                break

    if not replica_root:
        logger.error("REPLICA_ROOT not set")
        sys.exit(1)

    scene_path = Path(replica_root) / "room0"
    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    # Test cases that might trigger tool usage
    test_cases = [
        # Simple case - probably won't use tools
        ("pillow on the sofa", "What color is the pillow and is it on the sofa?"),
        # Harder case - might need more views
        (
            "objects on the table",
            "List all objects visible on or near the table. For each, describe its position relative to other objects.",
        ),
    ]

    reports = []
    for stage1_query, task_query in test_cases[:1]:  # Run first case
        try:
            report = run_with_trace(
                scene_path=scene_path,
                stage1_query=stage1_query,
                task_query=task_query,
                k=2,  # Start with fewer keyframes
                max_turns=4,
            )
            reports.append(report)
            logger.success(f"Report generated: {report}")
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    # Open first report in browser
    if reports:
        logger.info(f"\nOpening report in browser: {reports[0]}")
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(reports[0])], check=False)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(reports[0])], check=False)
        except Exception:
            logger.info(f"Please open manually: file://{reports[0].absolute()}")


if __name__ == "__main__":
    main()
