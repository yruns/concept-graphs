"""Run one-shot VLM baseline on OpenEQA benchmark.

This script evaluates a one-shot VLM baseline where Stage 1 retrieves keyframes
and Stage 2 VLM performs single-pass inference WITHOUT iterative evidence seeking.
This serves as a control condition to measure the value of adaptive evidence
acquisition.

TASK-031: Run one-shot VLM baseline on OpenEQA

Academic Relevance:
- Tests raw VLM reasoning capability on retrieved evidence
- Establishes baseline for "evidence-seeking agents beat one-shot" claim
- Enables comparison: one-shot vs multi-turn adaptive agent
- Demonstrates limitations of single-pass inference on complex queries

Key Differences from Full Agent:
- max_turns = 1 (single inference pass)
- Evidence-seeking tools disabled (no request_more_views, request_crops)
- No hypothesis repair (no switch_or_expand_hypothesis)
- Inspection tools still available (inspect_stage1_metadata, retrieve_object_context)

Usage:
    # With real OpenEQA data:
    python -m conceptgraph.evaluation.scripts.run_openeqa_oneshot \
        --data_root /path/to/open-eqa \
        --output results/baselines/openeqa_oneshot.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_openeqa_oneshot \
        --mock --max_samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from loguru import logger

from conceptgraph.evaluation.ablation_config import get_preset_config
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
    OpenEQASampleAdapter,
    adapt_openeqa_samples,
)


@dataclass
class MockOpenEQASample:
    """Mock OpenEQA sample for testing without real data."""

    question_id: str
    question: str
    answer: str
    category: str
    scene_id: str
    question_type: str = "episodic_memory"


def create_mock_openeqa_samples(n_samples: int = 50) -> List[MockOpenEQASample]:
    """Create synthetic OpenEQA samples for testing.

    These samples cover diverse question types and categories to stress-test
    the one-shot VLM reasoning.

    Args:
        n_samples: Number of mock samples to generate.

    Returns:
        List of mock OpenEQA samples.
    """
    question_templates = [
        ("What is on the {location}?", "{object}", "object_recognition"),
        ("What color is the {object}?", "{color}", "attribute_recognition"),
        ("How many {object}s are there?", "{count}", "counting"),
        ("Where is the {object}?", "on the {location}", "spatial"),
        ("What is next to the {object}?", "{neighbor}", "spatial"),
        ("What is between the {obj1} and {obj2}?", "{middle}", "spatial"),
        ("What did I see in the {room}?", "{object}", "episodic_memory"),
        ("Was there a {object} in the room?", "{yes_no}", "existence"),
        ("Is the {object} open or closed?", "{state}", "state"),
        ("What is inside the {container}?", "{contents}", "containment"),
    ]

    objects = ["chair", "table", "sofa", "lamp", "book", "cup", "plant", "pillow"]
    locations = ["table", "floor", "shelf", "couch", "desk", "counter"]
    colors = ["red", "blue", "green", "white", "brown", "black"]
    rooms = ["living room", "bedroom", "kitchen", "bathroom", "office"]
    scenes = ["scene_001", "scene_002", "scene_003", "room0", "room1"]

    samples = []
    for i in range(n_samples):
        template, answer_template, category = question_templates[
            i % len(question_templates)
        ]
        obj = objects[i % len(objects)]
        loc = locations[i % len(locations)]
        color = colors[i % len(colors)]
        room = rooms[i % len(rooms)]
        scene = scenes[i % len(scenes)]

        question = template.format(
            location=loc,
            object=obj,
            obj1=objects[(i + 1) % len(objects)],
            obj2=objects[(i + 2) % len(objects)],
            room=room,
            container=objects[(i + 3) % len(objects)],
        )
        answer = answer_template.format(
            object=obj,
            color=color,
            count=str((i % 5) + 1),
            location=loc,
            neighbor=objects[(i + 1) % len(objects)],
            middle=objects[(i + 2) % len(objects)],
            yes_no="yes" if i % 2 == 0 else "no",
            state="open" if i % 2 == 0 else "closed",
            contents=objects[(i + 4) % len(objects)],
        )

        samples.append(
            MockOpenEQASample(
                question_id=f"mock_openeqa_{i:04d}",
                question=question,
                answer=answer,
                category=category,
                scene_id=scene,
            )
        )

    return samples


def create_mock_stage1_factory():
    """Create a mock Stage 1 factory for testing without real scene data.

    The mock simulates KeyframeSelector behavior with realistic outputs
    including keyframe paths, hypothesis kinds, and metadata.
    """
    hypothesis_kinds = ["direct", "proxy", "context", "multi_object"]

    def factory(scene_id: str):
        mock_selector = MagicMock()
        call_count = [0]

        def mock_select(query: str, k: int = 3):
            call_count[0] += 1
            idx = call_count[0]

            result = MagicMock()
            result.query = query  # Required for Stage2EvidenceBundle
            result.keyframe_paths = [
                Path(f"/mock/scenes/{scene_id}/frame_{i:04d}.jpg") for i in range(k)
            ]
            result.metadata = {
                "selected_hypothesis_kind": hypothesis_kinds[
                    idx % len(hypothesis_kinds)
                ],
                "query": query,
                "scene_id": scene_id,
                "num_hypotheses_generated": 3,
                "retrieval_method": "visibility_weighted",
                "bev_used": True,
            }
            return result

        mock_selector.select_keyframes_v2 = mock_select
        return mock_selector

    return factory


def create_mock_stage2_factory():
    """Create a mock Stage 2 factory for testing one-shot inference.

    The mock simulates the Stage2DeepResearchAgent behavior in one-shot mode,
    returning structured responses without actual VLM calls.
    """
    from conceptgraph.agents.models import (
        Stage2AgentResult,
        Stage2EvidenceBundle,
        Stage2StructuredResponse,
        Stage2Status,
        Stage2TaskSpec,
        Stage2TaskType,
    )

    call_count = [0]

    def mock_run(task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
        call_count[0] += 1
        idx = call_count[0]

        # Simulate varying confidence levels for one-shot inference
        # One-shot tends to be less reliable, so we simulate lower confidence
        confidences = [0.65, 0.72, 0.58, 0.81, 0.45, 0.69, 0.55, 0.77]
        confidence = confidences[idx % len(confidences)]

        # Simulate some insufficient evidence cases (common in one-shot)
        status = Stage2Status.COMPLETED
        uncertainties = []
        if confidence < 0.5:
            status = Stage2Status.INSUFFICIENT_EVIDENCE
            uncertainties = [
                "Cannot verify object presence in available frames",
                "Limited viewing angles may miss target",
            ]

        # Generate mock answer based on task type
        if task.task_type == Stage2TaskType.QA:
            answer = f"Based on frame 0, the answer appears to be related to the query"
            payload = {
                "answer": answer,
                "supporting_claims": [
                    f"Observed in frame {i}" for i in range(len(bundle.keyframes))
                ],
            }
        else:
            answer = "Visual grounding result from one-shot inference"
            payload = {"result": answer}

        result = Stage2StructuredResponse(
            task_type=task.task_type,
            status=status,
            summary=answer,
            confidence=confidence,
            uncertainties=uncertainties,
            cited_frame_indices=list(range(min(3, len(bundle.keyframes)))),
            evidence_items=[],
            plan=[],
            payload=payload,
        )

        return Stage2AgentResult(
            task=task,
            result=result,
            tool_trace=[],  # No tool use in one-shot mode
            final_bundle=bundle,
            raw_state={},
        )

    class MockStage2Agent:
        def run(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
            return mock_run(task, bundle)

    return lambda: MockStage2Agent()


def run_oneshot_baseline(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/baselines/openeqa_oneshot.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    question_type: Optional[str] = None,
    category: Optional[str] = None,
    verbose: bool = True,
) -> EvalRunResult:
    """Run one-shot VLM evaluation on OpenEQA.

    Args:
        data_root: Path to OpenEQA dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock Stage 1/2.
        question_type: Filter by question type (episodic_memory/active_exploration).
        category: Filter by question category.
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with one-shot baseline metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("One-Shot VLM Baseline: OpenEQA")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("openeqa_oneshot_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock OpenEQA samples")
        mock_samples = create_mock_openeqa_samples(max_samples or 50)
        samples = adapt_openeqa_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
        stage2_factory = create_mock_stage2_factory()
    else:
        if data_root is None:
            raise ValueError("data_root is required when not using mock data")

        logger.info(f"Loading OpenEQA from {data_root}")
        from conceptgraph.benchmarks.openeqa_loader import OpenEQADataset

        dataset = OpenEQADataset.from_path(
            data_root,
            question_type=question_type,
            category=category,
            max_samples=max_samples,
        )
        samples = adapt_openeqa_samples(list(dataset))
        stage1_factory = None
        stage2_factory = None

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Get oneshot preset configuration for reference
    oneshot_preset = get_preset_config("oneshot")

    # Configure batch evaluation for one-shot mode
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="openeqa",
        max_workers=max_workers,
        # Stage 1 configuration (same as full agent)
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 ENABLED but in one-shot mode
        stage2_enabled=True,
        stage2_model="gpt-5.2-2025-12-11",
        stage2_max_turns=1,  # KEY: Single inference pass
        stage2_plan_mode="off",  # No planning in one-shot
        # Disable evidence-seeking tools (KEY ABLATION)
        enable_tool_request_more_views=False,
        enable_tool_request_crops=False,
        enable_tool_hypothesis_repair=False,
        # Keep uncertainty stopping for fair comparison
        enable_uncertainty_stopping=True,
        confidence_threshold=0.4,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        save_tool_traces=True,
        # Limits
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}")
    logger.info(f"Config: stage2_max_turns={config.stage2_max_turns}")
    logger.info(
        f"Config: enable_tool_request_more_views={config.enable_tool_request_more_views}"
    )
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")

    # Create evaluator
    evaluator = BatchEvaluator(
        config,
        stage1_factory=stage1_factory if use_mock else None,
        stage2_factory=stage2_factory if use_mock else None,
    )

    # Scene path provider (mock or real)
    def scene_path_provider(scene_id: str) -> Path:
        if use_mock:
            return Path(f"/mock/scenes/{scene_id}")
        if data_root:
            scene_path = data_root / "data" / "frames" / scene_id
            if scene_path.exists():
                return scene_path
            return data_root / scene_id
        raise ValueError("No scene path available")

    # Run evaluation
    logger.info(f"Starting evaluation with {max_workers} workers...")
    run_result = evaluator.run(samples, scene_path_provider)

    # Save additional output to specified path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create detailed baseline report
    baseline_report = {
        "experiment": "oneshot_vlm_baseline",
        "benchmark": "openeqa",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "stage2_enabled": True,
            "ablation_tag": config.get_ablation_tag(),
            "stage1_k": config.stage1_k,
            "stage1_model": config.stage1_model,
            "stage2_model": config.stage2_model,
            "stage2_max_turns": config.stage2_max_turns,
            "enable_tool_request_more_views": config.enable_tool_request_more_views,
            "enable_tool_request_crops": config.enable_tool_request_crops,
            "enable_tool_hypothesis_repair": config.enable_tool_hypothesis_repair,
            "max_samples": len(samples),
            "max_workers": max_workers,
            "use_mock": use_mock,
        },
        "summary": {
            "total_samples": run_result.total_samples,
            "successful_samples": run_result.successful_samples,
            "stage1_success": run_result.total_samples - run_result.failed_stage1,
            "stage1_failure": run_result.failed_stage1,
            "stage2_success": run_result.successful_samples,
            "stage2_failure": run_result.failed_stage2,
            "success_rate": (
                run_result.successful_samples / run_result.total_samples
                if run_result.total_samples > 0
                else 0.0
            ),
            "avg_stage1_latency_ms": run_result.avg_stage1_latency_ms,
            "avg_stage2_latency_ms": run_result.avg_stage2_latency_ms,
            "total_duration_seconds": run_result.total_duration_seconds,
        },
        "stage2_analysis": {
            "avg_confidence": run_result.avg_stage2_confidence,
            "avg_tool_calls": run_result.avg_tool_calls_per_sample,
            "samples_with_tool_use": run_result.samples_with_tool_use,
            "samples_with_insufficient_evidence": run_result.samples_with_insufficient_evidence,
            "tool_usage_distribution": run_result.tool_usage_distribution,
        },
        "confidence_distribution": _compute_confidence_distribution(run_result),
        "status_distribution": _compute_status_distribution(run_result),
        "per_sample_results": [
            {
                "sample_id": r.sample_id,
                "query": r.query,
                "scene_id": r.scene_id,
                "stage1_success": r.stage1_success,
                "stage1_keyframe_count": r.stage1_keyframe_count,
                "stage1_hypothesis_kind": r.stage1_hypothesis_kind,
                "stage1_latency_ms": r.stage1_latency_ms,
                "stage1_error": r.stage1_error,
                "stage2_success": r.stage2_success,
                "stage2_status": r.stage2_status,
                "stage2_confidence": r.stage2_confidence,
                "stage2_tool_calls": r.stage2_tool_calls,
                "stage2_latency_ms": r.stage2_latency_ms,
                "stage2_error": r.stage2_error,
                "uncertainties": r.uncertainties,
                "cited_frames": r.cited_frames,
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "One-shot VLM baseline for evidence-seeking agent comparison",
            "claim_support": "Demonstrates limitations of single-pass VLM inference",
            "expected_improvement_from": "Iterative evidence acquisition with tool use",
            "key_hypothesis": "Multi-turn agents with evidence-seeking outperform one-shot",
            "ablation_condition": "max_turns=1, no evidence-seeking tools",
        },
    }

    with open(output_path, "w") as f:
        json.dump(baseline_report, f, indent=2)

    logger.success(f"Results saved to {output_path}")

    # Print summary
    if verbose:
        _print_summary(run_result, baseline_report)

    return run_result


def _compute_confidence_distribution(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of confidence scores from Stage 2 results."""
    distribution = {
        "very_low_0_30": 0,
        "low_30_50": 0,
        "medium_50_70": 0,
        "high_70_90": 0,
        "very_high_90_100": 0,
    }
    for r in run_result.results:
        if r.stage2_success:
            conf = r.stage2_confidence
            if conf < 0.3:
                distribution["very_low_0_30"] += 1
            elif conf < 0.5:
                distribution["low_30_50"] += 1
            elif conf < 0.7:
                distribution["medium_50_70"] += 1
            elif conf < 0.9:
                distribution["high_70_90"] += 1
            else:
                distribution["very_high_90_100"] += 1
    return distribution


def _compute_status_distribution(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of Stage 2 status values."""
    distribution: Dict[str, int] = {}
    for r in run_result.results:
        if r.stage2_success and r.stage2_status:
            status = r.stage2_status
            distribution[status] = distribution.get(status, 0) + 1
    return distribution


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: One-Shot VLM Baseline")
    logger.info("=" * 60)
    logger.info("")

    summary = report["summary"]
    logger.info(f"Total Samples:       {summary['total_samples']}")
    logger.info(f"Successful:          {summary['successful_samples']}")
    logger.info(f"Stage 1 Success:     {summary['stage1_success']}")
    logger.info(f"Stage 1 Failure:     {summary['stage1_failure']}")
    logger.info(f"Stage 2 Success:     {summary['stage2_success']}")
    logger.info(f"Stage 2 Failure:     {summary['stage2_failure']}")
    logger.info(f"Success Rate:        {summary['success_rate']:.1%}")
    logger.info("")

    logger.info(f"Avg Stage 1 Latency: {summary['avg_stage1_latency_ms']:.1f}ms")
    logger.info(f"Avg Stage 2 Latency: {summary['avg_stage2_latency_ms']:.1f}ms")
    logger.info(f"Total Duration:      {summary['total_duration_seconds']:.1f}s")
    logger.info("")

    analysis = report["stage2_analysis"]
    logger.info("Stage 2 Analysis (One-Shot Mode):")
    logger.info(f"  Avg Confidence:    {analysis['avg_confidence']:.3f}")
    logger.info(
        f"  Avg Tool Calls:    {analysis['avg_tool_calls']:.2f} (expect ~0 for one-shot)"
    )
    logger.info(
        f"  Insufficient Evidence: {analysis['samples_with_insufficient_evidence']}"
    )
    logger.info("")

    logger.info("Confidence Distribution:")
    for bucket, count in report["confidence_distribution"].items():
        logger.info(f"  {bucket}: {count}")
    logger.info("")

    logger.info("Status Distribution:")
    for status, count in report["status_distribution"].items():
        logger.info(f"  {status}: {count}")
    logger.info("")

    logger.info("Academic Notes:")
    for key, value in report["academic_notes"].items():
        logger.info(f"  {key}: {value}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run one-shot VLM baseline on OpenEQA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        help="Path to OpenEQA dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/openeqa_oneshot.json"),
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing (no real OpenEQA required)",
    )
    parser.add_argument(
        "--question_type",
        choices=["episodic_memory", "active_exploration"],
        help="Filter by question type",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Filter by question category",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.mock and args.data_root is None:
        parser.error("--data_root is required when not using --mock")

    try:
        run_oneshot_baseline(
            data_root=args.data_root,
            output_path=args.output,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_mock=args.mock,
            question_type=args.question_type,
            category=args.category,
            verbose=not args.quiet,
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
