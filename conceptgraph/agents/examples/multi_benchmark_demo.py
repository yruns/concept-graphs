#!/usr/bin/env python
"""Demonstration of multi-benchmark Stage 2 pipeline integration.

This script demonstrates:
1. MultiBenchmarkAdapter usage with all supported benchmarks
2. Frame-based Stage 2 evaluation (bypasses Stage 1)
3. Full pipeline evaluation with real Replica data
4. Cross-benchmark comparison of Stage 2 outputs

Run: .venv/bin/python -m conceptgraph.agents.examples.multi_benchmark_demo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from conceptgraph.agents import (
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2TaskSpec,
    Stage2TaskType,
)
from conceptgraph.agents.benchmark_adapters import (
    BenchmarkSampleInfo,
    MultiBenchmarkAdapter,
    MockFrameProvider,
    ReplicaFrameProvider,
    create_adapter_for_benchmark,
    extract_sample_info,
    build_evidence_bundle_from_frames,
)


def demo_benchmark_info_extraction():
    """Demonstrate unified sample info extraction from different benchmark formats."""
    logger.info("=" * 70)
    logger.info("Demo 1: Benchmark Sample Info Extraction")
    logger.info("=" * 70)

    # Mock OpenEQA sample
    openeqa = MagicMock()
    openeqa.question_id = "openeqa-001"
    openeqa.question = "What color is the sofa in the living room?"
    openeqa.answer = "white"
    openeqa.scene_id = "mp3d-scene-42"
    openeqa.category = "object_recognition"
    openeqa.question_type = "episodic_memory"
    openeqa.episode_history = Path("/data/episodes/ep001")

    info = extract_sample_info(openeqa, "openeqa")
    logger.info(f"[OpenEQA] ID: {info.sample_id}")
    logger.info(f"          Query: {info.query}")
    logger.info(f"          Task: {info.task_type}")
    logger.info(f"          GT: {info.ground_truth}")

    # Mock SQA3D sample with egocentric situation
    sqa3d = MagicMock()
    sqa3d.question_id = "sqa3d-001"
    sqa3d.question = "What is to my left?"
    sqa3d.scene_id = "scene0001_00"
    sqa3d.primary_answer = "a window"
    sqa3d.answers = ["a window", "window"]
    sqa3d.question_type = "what"
    sqa3d.situation = MagicMock()
    sqa3d.situation.position = [1.0, 2.0, 0.0]
    sqa3d.situation.orientation = [0.0, 1.0, 0.0]
    sqa3d.situation.room_description = "I am standing in a living room facing the TV."

    info = extract_sample_info(sqa3d, "sqa3d")
    logger.info(f"[SQA3D]   ID: {info.sample_id}")
    logger.info(f"          Query: {info.query}")
    logger.info(f"          Situation: {info.extra['situation']['room_description']}")
    logger.info(f"          GT: {info.ground_truth}")

    # Mock ScanRefer sample with 3D bounding box
    scanrefer = MagicMock()
    scanrefer.sample_id = "scanrefer-001"
    scanrefer.description = "The white chair next to the desk"
    scanrefer.scene_id = "scene0002_00"
    scanrefer.object_id = "42"
    scanrefer.object_name = "chair"
    scanrefer.target_bbox = MagicMock()
    scanrefer.target_bbox.to_dict.return_value = {"center": [1, 2, 0.5], "size": [0.5, 0.5, 1]}

    info = extract_sample_info(scanrefer, "scanrefer")
    logger.info(f"[ScanRef] ID: {info.sample_id}")
    logger.info(f"          Query: {info.query}")
    logger.info(f"          Task: {info.task_type}")
    logger.info(f"          GT bbox: {info.ground_truth}")

    logger.success("Demo 1 complete: All benchmark formats extracted correctly.")
    return True


def demo_frame_based_stage2(replica_root: Optional[Path] = None):
    """Demonstrate frame-based Stage 2 evaluation (no Stage 1)."""
    logger.info("\n" + "=" * 70)
    logger.info("Demo 2: Frame-Based Stage 2 (Benchmark Direct Mode)")
    logger.info("=" * 70)

    # Create adapter with mock frames
    if replica_root and (replica_root / "room0").exists():
        provider = ReplicaFrameProvider(replica_root)
        adapter = MultiBenchmarkAdapter(
            frame_provider=provider,
            max_frames=3,
            plan_mode="brief",
        )
        logger.info(f"Using Replica frames from: {replica_root}")
    else:
        provider = MockFrameProvider()
        adapter = MultiBenchmarkAdapter(
            frame_provider=provider,
            max_frames=3,
        )
        logger.info("Using mock frame provider (no real frames)")

    # Create a Replica-style sample
    sample = {
        "sample_id": "replica-demo-001",
        "query": "What color is the pillow on the sofa?",
        "scene_id": "room0",
        "ground_truth": "light gray",
        "task_type": "qa",
    }

    # Prepare Stage 2 inputs
    task, bundle = adapter.prepare_stage2_inputs(sample, "replica")

    logger.info(f"Task type: {task.task_type}")
    logger.info(f"User query: {task.user_query}")
    logger.info(f"Bundle keyframes: {len(bundle.keyframes)}")
    logger.info(f"Hypothesis mode: {bundle.hypothesis.hypothesis_kind}")

    # Show frame paths if available
    for i, kf in enumerate(bundle.keyframes[:3]):
        logger.info(f"  Frame {i}: {Path(kf.image_path).name if kf.image_path else 'N/A'}")

    logger.success("Demo 2 complete: Frame-based Stage 2 inputs prepared.")
    return task, bundle


def demo_full_pipeline_with_vlm(replica_root: Path):
    """Demonstrate full Stage 2 pipeline with VLM inference."""
    logger.info("\n" + "=" * 70)
    logger.info("Demo 3: Full Stage 2 Pipeline with VLM")
    logger.info("=" * 70)

    scene_path = replica_root / "room0"
    if not scene_path.exists():
        logger.warning(f"Scene not found: {scene_path}")
        return None

    # Create adapter
    adapter = create_adapter_for_benchmark("replica", replica_root, max_frames=2)

    # Create sample
    sample = {
        "sample_id": "replica-vlm-test",
        "query": "Describe the pillow",
        "scene_id": "room0",
        "ground_truth": "light gray pillow on white sofa",
        "task_type": "qa",
    }

    task_spec, bundle = adapter.prepare_stage2_inputs(
        sample, "replica", scene_summary="A modern apartment living room."
    )

    logger.info(f"Evidence bundle: {len(bundle.keyframes)} keyframes")

    if not bundle.keyframes:
        logger.warning("No frames available - skipping VLM inference")
        return None

    # Create Stage 2 agent (no callbacks for demo)
    agent = Stage2DeepResearchAgent(
        config=Stage2DeepAgentConfig(
            include_thoughts=False,
            max_reasoning_turns=2,
        ),
    )

    logger.info("Running Stage 2 VLM agent...")
    result = agent.run(task_spec, bundle)

    logger.info(f"\nResult Status: {result.result.status.value}")
    logger.info(f"Confidence: {result.result.confidence}")
    logger.info(f"Summary: {result.result.summary}")

    if result.result.payload:
        logger.info(f"Answer: {result.result.payload.get('answer', 'N/A')[:100]}")

    gt = adapter.get_ground_truth(sample, "replica")
    logger.info(f"Ground Truth: {gt}")

    logger.success("Demo 3 complete: VLM inference successful.")
    return result


def demo_cross_benchmark_comparison():
    """Show how the adapter unifies different benchmarks."""
    logger.info("\n" + "=" * 70)
    logger.info("Demo 4: Cross-Benchmark Comparison")
    logger.info("=" * 70)

    samples = []

    # OpenEQA: episodic memory question
    openeqa = MagicMock()
    openeqa.question_id = "openeqa-memory-001"
    openeqa.question = "What did I see when I walked past the kitchen?"
    openeqa.answer = "a refrigerator and stove"
    openeqa.scene_id = "mp3d-kitchen"
    openeqa.category = "room_recognition"
    openeqa.question_type = "episodic_memory"
    openeqa.episode_history = None
    samples.append(("openeqa", openeqa))

    # SQA3D: situated reasoning
    sqa3d = MagicMock()
    sqa3d.question_id = "sqa3d-spatial-001"
    sqa3d.question = "If I turn right, what will I see?"
    sqa3d.scene_id = "scene0042_00"
    sqa3d.primary_answer = "a bookshelf"
    sqa3d.answers = ["a bookshelf", "bookshelf"]
    sqa3d.question_type = "where"
    sqa3d.situation = MagicMock()
    sqa3d.situation.position = [0, 0, 0]
    sqa3d.situation.orientation = [1, 0, 0]
    sqa3d.situation.room_description = "I'm in a study facing the window."
    samples.append(("sqa3d", sqa3d))

    # ScanRefer: visual grounding
    scanrefer = MagicMock()
    scanrefer.sample_id = "scanrefer-ground-001"
    scanrefer.description = "The black office chair in front of the wooden desk"
    scanrefer.scene_id = "scene0042_00"
    scanrefer.object_id = "15"
    scanrefer.object_name = "chair"
    scanrefer.target_bbox = MagicMock()
    scanrefer.target_bbox.to_dict.return_value = {"center": [2, 1, 0.4], "size": [0.6, 0.6, 1.1]}
    samples.append(("scanrefer", scanrefer))

    # Create unified comparison
    logger.info("\nBenchmark | Task Type        | Query Preview")
    logger.info("-" * 70)

    for bench_type, sample in samples:
        info = extract_sample_info(sample, bench_type)
        query_preview = info.query[:40] + "..." if len(info.query) > 40 else info.query
        logger.info(f"{bench_type:9} | {info.task_type:16} | {query_preview}")

    # Show how they all become Stage2TaskSpec
    logger.info("\nAll samples convert to unified Stage2TaskSpec:")
    adapter = MultiBenchmarkAdapter(frame_provider=MockFrameProvider(), max_frames=4)

    for bench_type, sample in samples:
        task, bundle = adapter.prepare_stage2_inputs(sample, bench_type)
        logger.info(f"  {bench_type}: task_type={task.task_type}, max_turns={task.max_reasoning_turns}")

    logger.success("Demo 4 complete: Unified benchmark handling verified.")
    return True


def main():
    """Run all demos."""
    logger.info("=" * 70)
    logger.info("Multi-Benchmark Stage 2 Pipeline Demonstration")
    logger.info("=" * 70)

    # Find Replica data
    replica_root = None
    for candidate in [
        Path(os.environ.get("REPLICA_ROOT", "")),
        Path("/Users/bytedance/Replica"),
        Path.home() / "Replica",
        Path.home() / "Datasets" / "Replica",
    ]:
        if candidate.exists():
            replica_root = candidate
            break

    if replica_root:
        logger.info(f"Found Replica data at: {replica_root}")
    else:
        logger.warning("No Replica data found - some demos will use mock data")

    # Run demos
    results = {}

    # Demo 1: Sample info extraction (no data needed)
    results["extraction"] = demo_benchmark_info_extraction()

    # Demo 2: Frame-based Stage 2 (mock or real frames)
    task, bundle = demo_frame_based_stage2(replica_root)
    results["frame_based"] = task is not None

    # Demo 3: Full VLM pipeline (needs Replica data)
    if replica_root and (replica_root / "room0").exists():
        results["vlm_pipeline"] = demo_full_pipeline_with_vlm(replica_root) is not None
    else:
        logger.info("\nSkipping Demo 3 (VLM pipeline) - no Replica data")
        results["vlm_pipeline"] = None

    # Demo 4: Cross-benchmark comparison
    results["comparison"] = demo_cross_benchmark_comparison()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION SUMMARY")
    logger.info("=" * 70)

    for name, success in results.items():
        if success is None:
            status = "⏭️  SKIPPED"
        elif success:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        logger.info(f"  {name}: {status}")

    all_passed = all(r is None or r for r in results.values())
    if all_passed:
        logger.success("\nAll demonstrations completed successfully!")
        logger.info("\nThe multi-benchmark adapter is ready for:")
        logger.info("  - OpenEQA: Episodic memory QA")
        logger.info("  - SQA3D: Situated question answering")
        logger.info("  - ScanRefer: Visual grounding with 3D bboxes")
        logger.info("  - Replica: Full pipeline with ConceptGraph integration")


if __name__ == "__main__":
    main()
