"""Run Stage 1 only baseline on OpenEQA benchmark.

This script evaluates the Stage 1 keyframe retrieval component without Stage 2
VLM agent reasoning. It serves as a baseline to measure how much Stage 2
contributes to overall performance.

TASK-030: Run Stage 1 only baseline on OpenEQA

Academic Relevance:
- Establishes lower bound for keyframe retrieval-only approach
- Provides control condition for "evidence-seeking VLM agent" claim
- Shows Stage 1 retrieval quality before VLM reasoning

Usage:
    # With real OpenEQA data:
    python -m conceptgraph.evaluation.scripts.run_openeqa_stage1_only \
        --data_root /path/to/open-eqa \
        --output results/baselines/openeqa_stage1_only.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_openeqa_stage1_only \
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
    the Stage 1 retrieval system.

    Args:
        n_samples: Number of mock samples to generate.

    Returns:
        List of mock OpenEQA samples.
    """
    # Representative question templates covering OpenEQA categories
    question_templates = [
        # Object recognition
        ("What is on the {location}?", "{object}", "object_recognition"),
        ("What color is the {object}?", "{color}", "attribute_recognition"),
        ("How many {object}s are there?", "{count}", "counting"),
        # Spatial reasoning
        ("Where is the {object}?", "on the {location}", "spatial"),
        ("What is next to the {object}?", "{neighbor}", "spatial"),
        ("What is between the {obj1} and {obj2}?", "{middle}", "spatial"),
        # Episodic memory
        ("What did I see in the {room}?", "{object}", "episodic_memory"),
        ("Was there a {object} in the room?", "{yes_no}", "existence"),
        # State reasoning
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

        # Format question and answer
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

            # Create mock result
            result = MagicMock()
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


def run_stage1_only_baseline(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/baselines/openeqa_stage1_only.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    question_type: Optional[str] = None,
    category: Optional[str] = None,
    verbose: bool = True,
) -> EvalRunResult:
    """Run Stage 1 only evaluation on OpenEQA.

    Args:
        data_root: Path to OpenEQA dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock Stage 1.
        question_type: Filter by question type (episodic_memory/active_exploration).
        category: Filter by question category.
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with Stage 1 only metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Stage 1 Only Baseline: OpenEQA")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("openeqa_stage1_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock OpenEQA samples")
        mock_samples = create_mock_openeqa_samples(max_samples or 50)
        samples = adapt_openeqa_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
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
        stage1_factory = None  # Use default KeyframeSelector

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Configure batch evaluation
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="openeqa",
        max_workers=max_workers,
        # Stage 1 configuration
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 DISABLED for baseline
        stage2_enabled=False,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        # Limits
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}")
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")

    # Create evaluator
    evaluator = BatchEvaluator(
        config,
        stage1_factory=stage1_factory if use_mock else None,
    )

    # Scene path provider (mock or real)
    def scene_path_provider(scene_id: str) -> Path:
        if use_mock:
            return Path(f"/mock/scenes/{scene_id}")
        if data_root:
            # OpenEQA scenes are typically in data/frames/{scene_id}/
            scene_path = data_root / "data" / "frames" / scene_id
            if scene_path.exists():
                return scene_path
            # Fallback to direct scene path
            return data_root / scene_id
        raise ValueError("No scene path available")

    # Run evaluation
    logger.info(f"Starting evaluation with {max_workers} workers...")
    run_result = evaluator.run(samples, scene_path_provider)

    # Save additional output to specified path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create detailed baseline report
    baseline_report = {
        "experiment": "stage1_only_baseline",
        "benchmark": "openeqa",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "stage2_enabled": False,
            "ablation_tag": config.get_ablation_tag(),
            "stage1_k": config.stage1_k,
            "stage1_model": config.stage1_model,
            "max_samples": len(samples),
            "max_workers": max_workers,
            "use_mock": use_mock,
        },
        "summary": {
            "total_samples": run_result.total_samples,
            "stage1_success": run_result.total_samples - run_result.failed_stage1,
            "stage1_failure": run_result.failed_stage1,
            "stage1_success_rate": (
                (run_result.total_samples - run_result.failed_stage1)
                / run_result.total_samples
                if run_result.total_samples > 0
                else 0.0
            ),
            "avg_stage1_latency_ms": run_result.avg_stage1_latency_ms,
            "total_duration_seconds": run_result.total_duration_seconds,
        },
        "hypothesis_distribution": _compute_hypothesis_distribution(run_result),
        "keyframe_statistics": _compute_keyframe_statistics(run_result),
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
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "Baseline for evidence-seeking VLM agent comparison",
            "claim_support": "Shows retrieval-only performance without Stage 2 reasoning",
            "expected_improvement_from": "Stage 2 adaptive evidence acquisition",
        },
    }

    with open(output_path, "w") as f:
        json.dump(baseline_report, f, indent=2)

    logger.success(f"Results saved to {output_path}")

    # Print summary
    if verbose:
        _print_summary(run_result, baseline_report)

    return run_result


def _compute_hypothesis_distribution(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of hypothesis types from Stage 1 results."""
    distribution: Dict[str, int] = {}
    for r in run_result.results:
        if r.stage1_success and r.stage1_hypothesis_kind:
            kind = r.stage1_hypothesis_kind
            distribution[kind] = distribution.get(kind, 0) + 1
    return distribution


def _compute_keyframe_statistics(run_result: EvalRunResult) -> Dict[str, Any]:
    """Compute statistics about retrieved keyframes."""
    keyframe_counts = [
        r.stage1_keyframe_count for r in run_result.results if r.stage1_success
    ]
    if not keyframe_counts:
        return {"avg_keyframes": 0.0, "min_keyframes": 0, "max_keyframes": 0}

    return {
        "avg_keyframes": sum(keyframe_counts) / len(keyframe_counts),
        "min_keyframes": min(keyframe_counts),
        "max_keyframes": max(keyframe_counts),
        "total_keyframes": sum(keyframe_counts),
    }


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: Stage 1 Only Baseline")
    logger.info("=" * 60)
    logger.info("")

    summary = report["summary"]
    logger.info(f"Total Samples:       {summary['total_samples']}")
    logger.info(f"Stage 1 Success:     {summary['stage1_success']}")
    logger.info(f"Stage 1 Failure:     {summary['stage1_failure']}")
    logger.info(f"Success Rate:        {summary['stage1_success_rate']:.1%}")
    logger.info(f"Avg Latency:         {summary['avg_stage1_latency_ms']:.1f}ms")
    logger.info(f"Total Duration:      {summary['total_duration_seconds']:.1f}s")
    logger.info("")

    logger.info("Hypothesis Distribution:")
    for kind, count in report["hypothesis_distribution"].items():
        logger.info(f"  {kind}: {count}")
    logger.info("")

    kf_stats = report["keyframe_statistics"]
    logger.info(f"Avg Keyframes/Sample: {kf_stats.get('avg_keyframes', 0):.1f}")
    logger.info("")

    logger.info("Academic Notes:")
    for key, value in report["academic_notes"].items():
        logger.info(f"  {key}: {value}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 only baseline on OpenEQA benchmark",
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
        default=Path("results/baselines/openeqa_stage1_only.json"),
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
        run_stage1_only_baseline(
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
