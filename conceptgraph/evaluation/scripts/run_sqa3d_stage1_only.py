"""Run Stage 1 only baseline evaluation on SQA3D benchmark.

SQA3D (Situated Question Answering in 3D Scenes) evaluates scene understanding
in the context of an agent's position and orientation.

This script runs the Stage 1 keyframe retrieval only (no Stage 2 VLM agent),
establishing a baseline for comparison with the full two-stage pipeline.

TASK-033: Run SQA3D experiments (all three conditions)

Academic Relevance:
- Establishes Stage 1 capability baseline for SQA3D (situated QA)
- Measures keyframe retrieval quality when situation context is provided
- Provides lower bound for comparison with evidence-seeking approaches

Usage:
    # With real SQA3D data:
    python -m conceptgraph.evaluation.scripts.run_sqa3d_stage1_only \
        --data_root /path/to/SQA3D \
        --output results/baselines/sqa3d_stage1_only.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_sqa3d_stage1_only \
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

from conceptgraph.benchmarks.sqa3d_loader import (
    SQA3DDataset,
    SQA3DEvaluationResult,
    SQA3DSample,
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
    # Question templates with situation-dependency
    question_templates = [
        # What questions - basic object identification
        ("What is on my {direction}?", ["chair", "table", "lamp"], "what"),
        ("What color is the object in front of me?", ["red", "blue", "white"], "what_color"),
        ("What is the material of the {object}?", ["wood", "metal", "fabric"], "what_material"),
        # Where questions - spatial reasoning from agent's perspective
        ("Where is the {object} relative to me?", ["in front", "behind", "left"], "where"),
        ("Where is the nearest {object}?", ["on the table", "by the wall", "on the floor"], "where"),
        # How many questions - counting
        ("How many {object}s can I see?", ["1", "2", "3"], "how_many"),
        ("How many chairs are around the table?", ["2", "4", "6"], "how_many"),
        # Yes/No questions - existence and state
        ("Is there a {object} in this room?", ["yes", "no"], "yes_no"),
        ("Can I reach the {object} from here?", ["yes", "no"], "can"),
        # Which questions - selection
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

        # Create realistic situation context
        position = [
            float((i * 0.5) % 10),
            float((i * 0.3) % 10),
            1.0,  # Standing height
        ]
        # Orientation vector (normalized direction agent is facing)
        orientation_options = [
            [1.0, 0.0, 0.0],   # Facing +X
            [0.0, 1.0, 0.0],   # Facing +Y
            [-1.0, 0.0, 0.0],  # Facing -X
            [0.0, -1.0, 0.0],  # Facing -Y
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
    """Create a mock Stage 1 factory for testing without real scene data.

    The mock simulates KeyframeSelector behavior with situation-aware
    outputs that respect the agent's position and orientation.
    """
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
                "situation_aware": True,  # SQA3D-specific
                "object_candidates": [
                    {"name": "chair", "confidence": 0.9},
                    {"name": "table", "confidence": 0.85},
                ],
            }
            return result

        mock_selector.select_keyframes_v2 = mock_select
        return mock_selector

    return factory


def run_sqa3d_stage1_only(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/baselines/sqa3d_stage1_only.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    question_type: Optional[str] = None,
    scene_id: Optional[str] = None,
    split: str = "val",
    verbose: bool = True,
) -> EvalRunResult:
    """Run Stage 1 only baseline on SQA3D.

    This evaluates keyframe retrieval quality without the Stage 2 agent,
    establishing a baseline for the two-stage pipeline.

    Args:
        data_root: Path to SQA3D dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock Stage 1.
        question_type: Filter by question type (what, where, etc.).
        scene_id: Filter by specific ScanNet scene.
        split: Dataset split (train/val/test).
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with Stage 1 only metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Stage 1 Only Baseline: SQA3D")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("sqa3d_stage1_only_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock SQA3D samples")
        mock_samples = create_mock_sqa3d_samples(max_samples or 50)
        samples = adapt_sqa3d_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
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

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Configure batch evaluation - Stage 2 DISABLED
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="sqa3d",
        max_workers=max_workers,
        # Stage 1 configuration
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 DISABLED for baseline
        stage2_enabled=False,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}")
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")

    # Create evaluator
    evaluator = BatchEvaluator(
        config,
        stage1_factory=stage1_factory if use_mock else None,
    )

    # Scene path provider
    def scene_path_provider(sid: str) -> Path:
        if use_mock:
            return Path(f"/mock/scannet/{sid}")
        if data_root:
            # ScanNet scenes are typically at scannet_frames/<scene_id>
            scene_path = data_root.parent / "scannet_frames" / sid
            if scene_path.exists():
                return scene_path
            # Try alternative path
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
        "experiment": "stage1_only_baseline",
        "benchmark": "sqa3d",
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
            "split": split,
        },
        "summary": {
            "total_samples": run_result.total_samples,
            "stage1_success": run_result.total_samples - run_result.failed_stage1,
            "stage1_failure": run_result.failed_stage1,
            "avg_stage1_latency_ms": run_result.avg_stage1_latency_ms,
            "total_duration_seconds": run_result.total_duration_seconds,
        },
        "hypothesis_distribution": _compute_hypothesis_distribution(run_result),
        "per_sample_results": [
            {
                "sample_id": r.sample_id,
                "query": r.query,
                "scene_id": r.scene_id,
                "stage1_success": r.stage1_success,
                "stage1_hypothesis_kind": r.stage1_hypothesis_kind,
                "stage1_keyframe_count": r.stage1_keyframe_count,
                "stage1_latency_ms": r.stage1_latency_ms,
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "Establish Stage 1 baseline for SQA3D (situated QA)",
            "key_insight": "Stage 1 retrieval with situation context provides evidence frames",
            "limitation": "No reasoning over evidence - just retrieval",
            "comparison_target": "sqa3d_stage2_full.json (full pipeline)",
        },
    }

    with open(output_path, "w") as f:
        json.dump(experiment_report, f, indent=2)

    logger.success(f"Results saved to {output_path}")

    if verbose:
        _print_summary(run_result, experiment_report)

    return run_result


def _compute_hypothesis_distribution(run_result: EvalRunResult) -> Dict[str, int]:
    """Compute distribution of hypothesis kinds."""
    dist: Dict[str, int] = {}
    for r in run_result.results:
        if r.stage1_success and r.stage1_hypothesis_kind:
            kind = r.stage1_hypothesis_kind
            dist[kind] = dist.get(kind, 0) + 1
    return dist


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: Stage 1 Only Baseline (SQA3D)")
    logger.info("=" * 60)
    logger.info("")

    summary = report["summary"]
    logger.info(f"Total Samples:       {summary['total_samples']}")
    logger.info(f"Stage 1 Success:     {summary['stage1_success']}")
    logger.info(f"Stage 1 Failure:     {summary['stage1_failure']}")
    logger.info(f"Avg S1 Latency:      {summary['avg_stage1_latency_ms']:.1f}ms")
    logger.info(f"Total Duration:      {summary['total_duration_seconds']:.1f}s")
    logger.info("")

    logger.info("Hypothesis Distribution:")
    for kind, count in report["hypothesis_distribution"].items():
        logger.info(f"  {kind}: {count}")
    logger.info("")

    logger.info("Academic Notes:")
    logger.info(f"  Purpose: {report['academic_notes']['purpose']}")
    logger.info(f"  Limitation: {report['academic_notes']['limitation']}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 only baseline on SQA3D benchmark",
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
        default=Path("results/baselines/sqa3d_stage1_only.json"),
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
        run_sqa3d_stage1_only(
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
