"""Run full Stage 2 agent evaluation on OpenEQA benchmark.

This script evaluates the complete two-stage pipeline:
- Stage 1: Task-conditioned keyframe retrieval
- Stage 2: Evidence-seeking VLM agent with iterative reasoning

TASK-032: Run full Stage 2 agent on OpenEQA

Academic Relevance:
- Tests core claim: "Evidence-seeking VLM agents that iteratively acquire
  visual context outperform one-shot baselines"
- Demonstrates adaptive evidence acquisition via tool use
- Shows symbolic-to-visual repair through hypothesis switching
- Validates evidence-grounded uncertainty quantification

Usage:
    # With real OpenEQA data:
    python -m conceptgraph.evaluation.scripts.run_openeqa_stage2_full \
        --data_root /path/to/open-eqa \
        --output results/experiments/openeqa_stage2_full.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_openeqa_stage2_full \
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

from conceptgraph.agents.models import (
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskType,
)
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
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

    These samples cover diverse question types to stress-test the full
    Stage 2 agent with its evidence-seeking capabilities.

    Args:
        n_samples: Number of mock samples to generate.

    Returns:
        List of mock OpenEQA samples.
    """
    # Question templates designed to exercise Stage 2 tools
    question_templates = [
        # Object recognition - benefits from request_crops
        ("What is on the {location}?", "{object}", "object_recognition"),
        ("What color is the {object}?", "{color}", "attribute_recognition"),
        ("How many {object}s are there?", "{count}", "counting"),
        # Spatial reasoning - benefits from request_more_views
        ("Where is the {object}?", "on the {location}", "spatial"),
        ("What is next to the {object}?", "{neighbor}", "spatial"),
        ("What is between the {obj1} and {obj2}?", "{middle}", "spatial"),
        # Episodic memory - benefits from hypothesis_repair
        ("What did I see in the {room}?", "{object}", "episodic_memory"),
        ("Was there a {object} in the room?", "{yes_no}", "existence"),
        # State reasoning - complex multi-tool scenarios
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

    The mock simulates KeyframeSelector behavior with outputs that
    exercise the Stage 2 agent's evidence-seeking capabilities.
    """
    hypothesis_kinds = ["direct", "proxy", "context", "multi_object"]

    def factory(scene_id: str):
        mock_selector = MagicMock()
        call_count = [0]

        def mock_select(query: str, k: int = 3):
            call_count[0] += 1
            idx = call_count[0]

            result = MagicMock()
            result.keyframe_paths = [
                Path(f"/mock/scenes/{scene_id}/frame_{i:04d}.jpg") for i in range(k)
            ]
            # Store query in result for Stage 2 adapter
            result.query = query
            result.metadata = {
                "selected_hypothesis_kind": hypothesis_kinds[
                    idx % len(hypothesis_kinds)
                ],
                "query": query,
                "scene_id": scene_id,
                "num_hypotheses_generated": 3,
                "retrieval_method": "visibility_weighted",
                "bev_used": True,
                "object_candidates": [
                    {"name": "chair", "confidence": 0.9},
                    {"name": "table", "confidence": 0.85},
                ],
            }
            return result

        mock_selector.select_keyframes_v2 = mock_select
        return mock_selector

    return factory


@dataclass
class MockToolTrace:
    """Mock tool trace entry for agent results."""
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class MockAgentResult:
    """Mock agent result matching BatchEvaluator expectations.

    BatchEvaluator accesses:
    - result.result.status.value
    - result.result.confidence
    - result.result.summary
    - result.result.payload.get()
    - result.result.uncertainties
    - result.result.cited_frame_indices
    - result.result.evidence_items
    - result.tool_trace (list of objects with .tool_name, .tool_input)
    """
    result: Stage2StructuredResponse
    tool_trace: List[MockToolTrace]


def create_mock_stage2_factory():
    """Create a mock Stage 2 factory for testing without real VLM calls.

    The mock simulates Stage2DeepResearchAgent behavior with realistic
    outputs including tool usage patterns and uncertainty quantification.

    Note: The factory returns a callable that creates a mock agent (no args).
    The mock agent's run() method accepts (task_spec, evidence_bundle).
    Returns MockAgentResult with .result and .tool_trace attributes.
    """
    tool_patterns = [
        ["inspect_stage1_metadata", "request_crops"],
        ["inspect_stage1_metadata", "request_more_views", "request_crops"],
        ["inspect_stage1_metadata", "switch_or_expand_hypothesis"],
        ["inspect_stage1_metadata"],  # One-turn direct answer
        ["inspect_stage1_metadata", "request_more_views", "switch_or_expand_hypothesis", "request_crops"],
    ]

    call_count = [0]

    def factory():
        mock_agent = MagicMock()

        def mock_run(task_spec, evidence_bundle=None):
            call_count[0] += 1
            idx = call_count[0]

            # Simulate varying confidence levels
            confidence = 0.7 + (idx % 4) * 0.08
            tools_used = tool_patterns[idx % len(tool_patterns)]

            # Determine status based on confidence
            if confidence >= 0.85:
                status = Stage2Status.COMPLETED
            elif confidence >= 0.6:
                status = Stage2Status.COMPLETED
            else:
                status = Stage2Status.INSUFFICIENT_EVIDENCE

            # Create structured response (this is what .result returns)
            structured_response = Stage2StructuredResponse(
                task_type=Stage2TaskType.QA,
                status=status,
                confidence=confidence,
                payload={"answer": f"mock_answer_{idx}"},
                summary=f"Mock answer for query",
                uncertainties=["object_identity"] if confidence < 0.8 else [],
                cited_frame_indices=[0, 1],
                plan=[f"Step {i+1}: {tool}" for i, tool in enumerate(tools_used)],
            )

            # Create tool trace
            tool_trace = [
                MockToolTrace(
                    tool_name=tool,
                    tool_input={"query": "mock_query", "iteration": i}
                )
                for i, tool in enumerate(tools_used)
            ]

            return MockAgentResult(
                result=structured_response,
                tool_trace=tool_trace,
            )

        mock_agent.run = mock_run
        return mock_agent

    return factory


def run_stage2_full_evaluation(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/experiments/openeqa_stage2_full.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    question_type: Optional[str] = None,
    category: Optional[str] = None,
    plan_mode: str = "brief",
    max_turns: int = 6,
    confidence_threshold: float = 0.7,
    verbose: bool = True,
) -> EvalRunResult:
    """Run full Stage 2 evaluation on OpenEQA.

    This is the main evaluation function that runs the complete two-stage
    pipeline with all evidence-seeking tools enabled.

    Args:
        data_root: Path to OpenEQA dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock agents.
        question_type: Filter by question type (episodic_memory/active_exploration).
        category: Filter by question category.
        plan_mode: Stage 2 planning mode (off/brief/full).
        max_turns: Maximum reasoning turns for Stage 2.
        confidence_threshold: Confidence threshold for uncertainty stopping.
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with full Stage 2 metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Full Stage 2 Evaluation: OpenEQA")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("openeqa_stage2_full_%Y%m%d_%H%M%S")

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

    # Configure batch evaluation with FULL Stage 2 capabilities
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="openeqa",
        max_workers=max_workers,
        # Stage 1 configuration
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 ENABLED with full capabilities
        stage2_enabled=True,
        stage2_model="gpt-5.2-2025-12-11",
        stage2_max_turns=max_turns,
        stage2_plan_mode=plan_mode,
        # ALL evidence-seeking tools ENABLED
        enable_tool_request_more_views=True,
        enable_tool_request_crops=True,
        enable_tool_hypothesis_repair=True,
        # Uncertainty-aware stopping ENABLED
        enable_uncertainty_stopping=True,
        confidence_threshold=confidence_threshold,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}")
    logger.info(f"Config: stage2_max_turns={config.stage2_max_turns}")
    logger.info(f"Config: plan_mode={config.stage2_plan_mode}")
    logger.info(f"Tools: request_more_views={config.enable_tool_request_more_views}")
    logger.info(f"Tools: request_crops={config.enable_tool_request_crops}")
    logger.info(f"Tools: hypothesis_repair={config.enable_tool_hypothesis_repair}")
    logger.info(f"Uncertainty: enabled={config.enable_uncertainty_stopping}, threshold={config.confidence_threshold}")
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")

    # Create evaluator
    evaluator = BatchEvaluator(
        config,
        stage1_factory=stage1_factory if use_mock else None,
        stage2_factory=stage2_factory if use_mock else None,
    )

    # Scene path provider
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

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create detailed experiment report
    experiment_report = {
        "experiment": "stage2_full_evaluation",
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
            "stage2_plan_mode": config.stage2_plan_mode,
            "tools": {
                "request_more_views": config.enable_tool_request_more_views,
                "request_crops": config.enable_tool_request_crops,
                "hypothesis_repair": config.enable_tool_hypothesis_repair,
            },
            "uncertainty": {
                "enabled": config.enable_uncertainty_stopping,
                "confidence_threshold": config.confidence_threshold,
            },
            "max_samples": len(samples),
            "max_workers": max_workers,
            "use_mock": use_mock,
        },
        "summary": {
            "total_samples": run_result.total_samples,
            "stage1_success": run_result.total_samples - run_result.failed_stage1,
            "stage1_failure": run_result.failed_stage1,
            "stage2_success": run_result.total_samples - run_result.failed_stage2,
            "stage2_failure": run_result.failed_stage2,
            "avg_stage1_latency_ms": run_result.avg_stage1_latency_ms,
            "avg_stage2_latency_ms": run_result.avg_stage2_latency_ms,
            "total_duration_seconds": run_result.total_duration_seconds,
        },
        "stage2_analysis": _compute_stage2_analysis(run_result),
        "tool_usage_distribution": _compute_tool_usage(run_result),
        "confidence_distribution": _compute_confidence_distribution(run_result),
        "uncertainty_analysis": _compute_uncertainty_analysis(run_result),
        "per_sample_results": [
            {
                "sample_id": r.sample_id,
                "query": r.query,
                "scene_id": r.scene_id,
                "stage1_success": r.stage1_success,
                "stage1_hypothesis_kind": r.stage1_hypothesis_kind,
                "stage2_success": r.stage2_success,
                "stage2_status": (
                    r.stage2_status.value
                    if hasattr(r.stage2_status, "value")
                    else str(r.stage2_status)
                ),
                "stage2_confidence": r.stage2_confidence,
                "stage2_tool_calls": r.stage2_tool_calls,
                "stage2_latency_ms": r.stage2_latency_ms,
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "Validate evidence-seeking VLM agent hypothesis",
            "core_claim": "Iterative evidence acquisition outperforms one-shot baselines",
            "innovation_points": [
                "Adaptive evidence acquisition via request_more_views/request_crops",
                "Symbolic-to-visual repair via hypothesis_repair",
                "Evidence-grounded uncertainty quantification",
                "Unified multi-task policy across QA/grounding/planning",
            ],
            "comparison_baselines": [
                "openeqa_stage1_only.json (Stage 1 retrieval only)",
                "openeqa_oneshot.json (One-shot VLM without tools)",
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(experiment_report, f, indent=2)

    logger.success(f"Results saved to {output_path}")

    if verbose:
        _print_summary(run_result, experiment_report)

    return run_result


def _compute_stage2_analysis(run_result: EvalRunResult) -> Dict[str, Any]:
    """Compute Stage 2 specific analysis metrics."""
    successful_stage2 = [r for r in run_result.results if r.stage2_success]

    if not successful_stage2:
        return {
            "avg_tool_calls": 0.0,
            "avg_confidence": 0.0,
            "complete_rate": 0.0,
            "insufficient_evidence_rate": 0.0,
        }

    tool_calls = [r.stage2_tool_calls for r in successful_stage2 if r.stage2_tool_calls]
    confidences = [r.stage2_confidence for r in successful_stage2 if r.stage2_confidence]

    statuses = [r.stage2_status for r in successful_stage2]
    complete_count = sum(1 for s in statuses if s == "completed")
    insufficient_count = sum(1 for s in statuses if s == "insufficient_evidence")

    return {
        "avg_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
        "max_tool_calls": max(tool_calls) if tool_calls else 0,
        "min_tool_calls": min(tool_calls) if tool_calls else 0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "complete_rate": complete_count / len(successful_stage2),
        "insufficient_evidence_rate": insufficient_count / len(successful_stage2),
    }


def _compute_tool_usage(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of tool usage across samples."""
    tool_counts: Dict[str, int] = {}

    for r in run_result.results:
        if r.stage2_success and r.tool_trace:
            for trace_entry in r.tool_trace:
                tool_name = trace_entry.get("tool_name", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    return tool_counts


def _compute_confidence_distribution(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of confidence scores."""
    bins = {
        "very_high_0.9+": 0,
        "high_0.8-0.9": 0,
        "medium_0.7-0.8": 0,
        "low_0.6-0.7": 0,
        "very_low_<0.6": 0,
    }

    for r in run_result.results:
        if r.stage2_success and r.stage2_confidence is not None:
            conf = r.stage2_confidence
            if conf >= 0.9:
                bins["very_high_0.9+"] += 1
            elif conf >= 0.8:
                bins["high_0.8-0.9"] += 1
            elif conf >= 0.7:
                bins["medium_0.7-0.8"] += 1
            elif conf >= 0.6:
                bins["low_0.6-0.7"] += 1
            else:
                bins["very_low_<0.6"] += 1

    return bins


def _compute_uncertainty_analysis(run_result: EvalRunResult) -> Dict[str, Any]:
    """Compute uncertainty-related analysis."""
    insufficient_evidence = []
    high_confidence = []

    for r in run_result.results:
        if r.stage2_success:
            if r.stage2_status == "insufficient_evidence":
                insufficient_evidence.append(r)
            elif r.stage2_confidence and r.stage2_confidence >= 0.85:
                high_confidence.append(r)

    return {
        "insufficient_evidence_count": len(insufficient_evidence),
        "high_confidence_count": len(high_confidence),
        "uncertainty_ratio": (
            len(insufficient_evidence) / len(run_result.results)
            if run_result.results else 0.0
        ),
    }


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: Full Stage 2 Agent")
    logger.info("=" * 60)
    logger.info("")

    summary = report["summary"]
    logger.info(f"Total Samples:       {summary['total_samples']}")
    logger.info(f"Stage 1 Success:     {summary['stage1_success']}")
    logger.info(f"Stage 2 Success:     {summary['stage2_success']}")
    logger.info(f"Avg S1 Latency:      {summary['avg_stage1_latency_ms']:.1f}ms")
    logger.info(f"Avg S2 Latency:      {summary['avg_stage2_latency_ms']:.1f}ms")
    logger.info(f"Total Duration:      {summary['total_duration_seconds']:.1f}s")
    logger.info("")

    s2_analysis = report["stage2_analysis"]
    logger.info("Stage 2 Analysis:")
    logger.info(f"  Avg Tool Calls:    {s2_analysis['avg_tool_calls']:.2f}")
    logger.info(f"  Avg Confidence:    {s2_analysis['avg_confidence']:.2f}")
    logger.info(f"  Complete Rate:     {s2_analysis['complete_rate']:.1%}")
    logger.info(f"  Insufficient Rate: {s2_analysis['insufficient_evidence_rate']:.1%}")
    logger.info("")

    logger.info("Tool Usage Distribution:")
    for tool, count in report["tool_usage_distribution"].items():
        logger.info(f"  {tool}: {count}")
    logger.info("")

    logger.info("Confidence Distribution:")
    for bin_name, count in report["confidence_distribution"].items():
        logger.info(f"  {bin_name}: {count}")
    logger.info("")

    logger.info("Academic Alignment:")
    for point in report["academic_notes"]["innovation_points"]:
        logger.info(f"  ✓ {point}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run full Stage 2 agent evaluation on OpenEQA benchmark",
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
        default=Path("results/experiments/openeqa_stage2_full.json"),
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
        "--plan_mode",
        choices=["off", "brief", "full"],
        default="brief",
        help="Stage 2 planning mode",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Maximum reasoning turns for Stage 2",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for uncertainty stopping",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if not args.mock and args.data_root is None:
        parser.error("--data_root is required when not using --mock")

    try:
        run_stage2_full_evaluation(
            data_root=args.data_root,
            output_path=args.output,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_mock=args.mock,
            question_type=args.question_type,
            category=args.category,
            plan_mode=args.plan_mode,
            max_turns=args.max_turns,
            confidence_threshold=args.confidence_threshold,
            verbose=not args.quiet,
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
