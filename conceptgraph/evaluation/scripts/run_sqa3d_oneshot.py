"""Run one-shot VLM baseline evaluation on SQA3D benchmark.

SQA3D (Situated Question Answering in 3D Scenes) evaluates scene understanding
in the context of an agent's position and orientation.

This script runs the one-shot VLM baseline where the VLM receives all keyframes
at once and produces an answer in a single inference pass (no iterative tools).

TASK-033: Run SQA3D experiments (all three conditions)

Academic Relevance:
- One-shot baseline captures "standard VQA" approach on 3D scenes
- No iterative evidence acquisition (contrast with full Stage 2)
- Demonstrates limitation of one-shot approaches for complex spatial reasoning
- Key comparison point: Does iterative evidence-seeking improve SQA3D accuracy?

Usage:
    # With real SQA3D data:
    python -m conceptgraph.evaluation.scripts.run_sqa3d_oneshot \
        --data_root /path/to/SQA3D \
        --output results/baselines/sqa3d_oneshot.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_sqa3d_oneshot \
        --mock --max_samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
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
from conceptgraph.benchmarks.sqa3d_loader import (
    SQA3DDataset,
    SQA3DSituation,
    compute_sqa3d_metrics,
    evaluate_sqa3d,
)
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
    adapt_sqa3d_samples,
)


@dataclass
class MockSQA3DSample:
    """Mock SQA3D sample for testing without real data."""

    question_id: str
    question: str
    answers: List[str]
    situation: SQA3DSituation
    scene_id: str
    question_type: str = "what"
    answer_type: str = "single_word"
    choices: List[str] = field(default_factory=list)

    @property
    def primary_answer(self) -> str:
        return self.answers[0] if self.answers else ""


def create_mock_sqa3d_samples(n_samples: int = 50) -> List[MockSQA3DSample]:
    """Create synthetic SQA3D samples for testing.

    These samples include situation context (position, orientation)
    to simulate the situated QA nature of SQA3D.

    Args:
        n_samples: Number of mock samples to generate.

    Returns:
        List of mock SQA3D samples.
    """
    question_templates = [
        ("What is on my {direction}?", ["chair", "table", "lamp"], "what"),
        ("What color is the object in front of me?", ["red", "blue", "white"], "what_color"),
        ("What is the material of the {object}?", ["wood", "metal", "fabric"], "what_material"),
        ("Where is the {object} relative to me?", ["in front", "behind", "left"], "where"),
        ("Where is the nearest {object}?", ["on the table", "by the wall", "on the floor"], "where"),
        ("How many {object}s can I see?", ["1", "2", "3"], "how_many"),
        ("How many chairs are around the table?", ["2", "4", "6"], "how_many"),
        ("Is there a {object} in this room?", ["yes", "no"], "yes_no"),
        ("Can I reach the {object} from here?", ["yes", "no"], "can"),
        ("Which {object} is closest to me?", ["the red one", "the large one"], "which"),
    ]

    objects = ["chair", "table", "sofa", "lamp", "book", "cup", "plant", "pillow"]
    directions = ["left", "right", "front", "behind"]
    scenes = ["scene0000_00", "scene0011_00", "scene0025_00", "scene0050_00", "scene0100_00"]

    samples = []
    for i in range(n_samples):
        template, answers_pool, q_type = question_templates[i % len(question_templates)]
        obj = objects[i % len(objects)]
        direction = directions[i % len(directions)]

        question = template.format(object=obj, direction=direction)
        answers = [answers_pool[i % len(answers_pool)]]

        position = [
            float((i * 0.5) % 10),
            float((i * 0.3) % 10),
            1.0,
        ]
        orientation_options = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        orientation = orientation_options[i % len(orientation_options)]

        room_descriptions = [
            "A small living room with a couch and TV",
            "A large bedroom with a queen-size bed",
            "A kitchen with modern appliances",
            "An office with a desk and bookshelves",
            "A dining room with a large table",
        ]

        situation = SQA3DSituation(
            position=position,
            orientation=orientation,
            room_description=room_descriptions[i % len(room_descriptions)],
            reference_objects=[obj],
        )

        samples.append(
            MockSQA3DSample(
                question_id=f"mock_sqa3d_{i:04d}",
                question=question,
                answers=answers,
                situation=situation,
                scene_id=scenes[i % len(scenes)],
                question_type=q_type,
            )
        )

    return samples


def create_mock_stage1_factory():
    """Create a mock Stage 1 factory for testing without real scene data."""
    hypothesis_kinds = ["direct", "proxy", "context", "situated"]

    def factory(scene_id: str):
        mock_selector = MagicMock()
        call_count = [0]

        def mock_select(query: str, k: int = 3):
            call_count[0] += 1
            idx = call_count[0]

            result = MagicMock()
            result.keyframe_paths = [
                Path(f"/mock/scannet/{scene_id}/color/{i:06d}.jpg") for i in range(k)
            ]
            result.query = query
            result.metadata = {
                "selected_hypothesis_kind": hypothesis_kinds[idx % len(hypothesis_kinds)],
                "query": query,
                "scene_id": scene_id,
                "num_hypotheses_generated": 3,
                "retrieval_method": "visibility_weighted",
                "bev_used": True,
                "situation_aware": True,
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
    """Mock agent result matching BatchEvaluator expectations."""
    result: Stage2StructuredResponse
    tool_trace: List[MockToolTrace]


def create_mock_stage2_oneshot_factory():
    """Create a mock Stage 2 factory for one-shot baseline.

    The one-shot baseline uses Stage 2 agent but with max_turns=1 and
    no tool use, simulating a single-pass VLM inference.
    """
    call_count = [0]

    def factory():
        mock_agent = MagicMock()

        def mock_run(task_spec, evidence_bundle=None):
            call_count[0] += 1
            idx = call_count[0]

            # One-shot: higher confidence variance, no tools
            confidence = 0.65 + (idx % 5) * 0.07

            if confidence >= 0.75:
                status = Stage2Status.COMPLETED
            elif confidence >= 0.55:
                status = Stage2Status.COMPLETED
            else:
                status = Stage2Status.INSUFFICIENT_EVIDENCE

            structured_response = Stage2StructuredResponse(
                task_type=Stage2TaskType.QA,
                status=status,
                confidence=confidence,
                payload={"answer": f"oneshot_answer_{idx}"},
                summary=f"One-shot answer for situated question",
                uncertainties=["spatial_ambiguity", "viewpoint_dependency"] if confidence < 0.7 else [],
                cited_frame_indices=[0, 1, 2],  # All frames cited at once
                plan=[],  # No planning in one-shot
            )

            # One-shot: minimal tool trace (only inspect metadata)
            tool_trace = [
                MockToolTrace(
                    tool_name="inspect_stage1_metadata",
                    tool_input={"query": "oneshot_query"}
                )
            ]

            return MockAgentResult(
                result=structured_response,
                tool_trace=tool_trace,
            )

        mock_agent.run = mock_run
        return mock_agent

    return factory


def run_sqa3d_oneshot(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/baselines/sqa3d_oneshot.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    question_type: Optional[str] = None,
    scene_id: Optional[str] = None,
    split: str = "val",
    verbose: bool = True,
) -> EvalRunResult:
    """Run one-shot VLM baseline on SQA3D.

    This evaluates a VLM that receives all keyframes at once and produces
    an answer in a single inference pass, without iterative tool use.

    Args:
        data_root: Path to SQA3D dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock agents.
        question_type: Filter by question type.
        scene_id: Filter by specific ScanNet scene.
        split: Dataset split (train/val/test).
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with one-shot baseline metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("One-Shot VLM Baseline: SQA3D")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("sqa3d_oneshot_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock SQA3D samples")
        mock_samples = create_mock_sqa3d_samples(max_samples or 50)
        samples = adapt_sqa3d_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
        stage2_factory = create_mock_stage2_oneshot_factory()
    else:
        if data_root is None:
            raise ValueError("data_root is required when not using mock data")

        logger.info(f"Loading SQA3D from {data_root}")
        dataset = SQA3DDataset.from_path(
            data_root,
            split=split,
            question_type=question_type,
            scene_id=scene_id,
            max_samples=max_samples,
        )
        samples = adapt_sqa3d_samples(list(dataset))
        stage1_factory = None
        stage2_factory = None

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Configure batch evaluation - ONE-SHOT mode (max_turns=1, no tools)
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="sqa3d",
        max_workers=max_workers,
        # Stage 1 configuration
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 ONE-SHOT mode
        stage2_enabled=True,
        stage2_model="gpt-5.2-2025-12-11",
        stage2_max_turns=1,  # ONE-SHOT: single turn only
        stage2_plan_mode="off",  # No planning
        # ALL tools DISABLED for one-shot baseline
        enable_tool_request_more_views=False,
        enable_tool_request_crops=False,
        enable_tool_hypothesis_repair=False,
        # Uncertainty stopping DISABLED
        enable_uncertainty_stopping=False,
        confidence_threshold=0.0,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}")
    logger.info(f"Config: stage2_max_turns={config.stage2_max_turns} (ONE-SHOT)")
    logger.info(f"Config: plan_mode={config.stage2_plan_mode}")
    logger.info(f"Tools: request_more_views={config.enable_tool_request_more_views}")
    logger.info(f"Tools: request_crops={config.enable_tool_request_crops}")
    logger.info(f"Tools: hypothesis_repair={config.enable_tool_hypothesis_repair}")
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")

    # Create evaluator
    evaluator = BatchEvaluator(
        config,
        stage1_factory=stage1_factory if use_mock else None,
        stage2_factory=stage2_factory if use_mock else None,
    )

    # Scene path provider
    def scene_path_provider(sid: str) -> Path:
        if use_mock:
            return Path(f"/mock/scannet/{sid}")
        if data_root:
            scene_path = data_root.parent / "scannet_frames" / sid
            if scene_path.exists():
                return scene_path
            scene_path = data_root / "scans" / sid
            if scene_path.exists():
                return scene_path
        return Path(f"/data/scannet/{sid}")

    # Run evaluation
    logger.info(f"Starting evaluation with {max_workers} workers...")
    run_result = evaluator.run(samples, scene_path_provider)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_report = {
        "experiment": "oneshot_vlm_baseline",
        "benchmark": "sqa3d",
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
                "request_more_views": False,
                "request_crops": False,
                "hypothesis_repair": False,
            },
            "max_samples": len(samples),
            "max_workers": max_workers,
            "use_mock": use_mock,
            "split": split,
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
        "confidence_distribution": _compute_confidence_distribution(run_result),
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
                "stage2_latency_ms": r.stage2_latency_ms,
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "One-shot VLM baseline for SQA3D (no iterative tools)",
            "key_differences": {
                "vs_stage1_only": "Includes VLM reasoning, but single-pass only",
                "vs_stage2_full": "No iterative evidence acquisition or tool use",
            },
            "hypothesis": "One-shot struggle with ambiguous spatial queries requiring viewpoint navigation",
            "comparison_baselines": [
                "sqa3d_stage1_only.json (Stage 1 retrieval only)",
                "sqa3d_stage2_full.json (Full iterative agent)",
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
            "avg_confidence": 0.0,
            "complete_rate": 0.0,
            "insufficient_evidence_rate": 0.0,
        }

    confidences = [r.stage2_confidence for r in successful_stage2 if r.stage2_confidence]

    statuses = [r.stage2_status for r in successful_stage2]
    complete_count = sum(1 for s in statuses if s == "completed")
    insufficient_count = sum(1 for s in statuses if s == "insufficient_evidence")

    return {
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "complete_rate": complete_count / len(successful_stage2),
        "insufficient_evidence_rate": insufficient_count / len(successful_stage2),
        "note": "One-shot mode - no tool calls",
    }


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


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: One-Shot VLM Baseline (SQA3D)")
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
    logger.info("Stage 2 Analysis (One-Shot):")
    logger.info(f"  Avg Confidence:    {s2_analysis['avg_confidence']:.2f}")
    logger.info(f"  Complete Rate:     {s2_analysis['complete_rate']:.1%}")
    logger.info(f"  Insufficient Rate: {s2_analysis['insufficient_evidence_rate']:.1%}")
    logger.info("")

    logger.info("Confidence Distribution:")
    for bin_name, count in report["confidence_distribution"].items():
        logger.info(f"  {bin_name}: {count}")
    logger.info("")

    logger.info("Academic Notes:")
    notes = report["academic_notes"]
    logger.info(f"  Purpose: {notes['purpose']}")
    logger.info(f"  Hypothesis: {notes['hypothesis']}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run one-shot VLM baseline on SQA3D benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        help="Path to SQA3D dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/sqa3d_oneshot.json"),
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
        help="Use mock data for testing (no real SQA3D required)",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        help="Filter by question type",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        help="Filter by specific ScanNet scene ID",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate",
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
        run_sqa3d_oneshot(
            data_root=args.data_root,
            output_path=args.output,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_mock=args.mock,
            question_type=args.question_type,
            scene_id=args.scene_id,
            split=args.split,
            verbose=not args.quiet,
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
