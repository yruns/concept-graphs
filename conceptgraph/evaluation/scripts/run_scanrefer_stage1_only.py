"""Run Stage 1 only baseline evaluation on ScanRefer benchmark.

ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language
ECCV 2020, https://daveredrum.github.io/ScanRefer/

This script runs the Stage 1 keyframe retrieval only (no Stage 2 VLM agent),
establishing a baseline for 3D visual grounding comparison.

TASK-034: Run ScanRefer experiments

Academic Relevance:
- Establishes Stage 1 capability baseline for visual grounding
- Measures keyframe retrieval quality for object localization tasks
- Provides lower bound for comparison with evidence-seeking approaches
- ScanRefer requires precise 3D bounding box prediction (IoU-based metrics)

Usage:
    # With real ScanRefer data:
    python -m conceptgraph.evaluation.scripts.run_scanrefer_stage1_only \
        --data_root /path/to/ScanRefer \
        --output results/baselines/scanrefer_stage1_only.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_scanrefer_stage1_only \
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

from conceptgraph.benchmarks.scanrefer_loader import (
    BoundingBox3D,
    ScanReferDataset,
    ScanReferEvaluationResult,
    ScanReferSample,
    compute_scanrefer_metrics,
    evaluate_scanrefer,
)
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
    adapt_scanrefer_samples,
)


@dataclass
class MockScanReferSample:
    """Mock ScanRefer sample for testing without real data."""

    sample_id: str
    scene_id: str
    object_id: str
    object_name: str
    description: str
    target_bbox: BoundingBox3D
    ann_id: str = ""
    token: List[str] = field(default_factory=list)

    @property
    def query(self) -> str:
        """Get the referring expression as query."""
        return self.description


def create_mock_scanrefer_samples(n_samples: int = 50) -> List[MockScanReferSample]:
    """Create synthetic ScanRefer samples for testing.

    These samples include referring expressions that describe objects
    using spatial relationships and attributes.

    Args:
        n_samples: Number of mock samples to generate.

    Returns:
        List of mock ScanRefer samples.
    """
    # Object categories commonly found in ScanNet scenes
    object_categories = [
        "chair", "table", "sofa", "bed", "desk", "cabinet", "bookshelf",
        "refrigerator", "toilet", "sink", "lamp", "door", "window", "pillow",
    ]

    # Referring expression templates
    description_templates = [
        # Spatial relationships
        "the {obj} next to the {ref}",
        "the {obj} in front of the {ref}",
        "the {obj} behind the {ref}",
        "the {color} {obj} near the window",
        "the {obj} on the left side of the room",
        # Attribute-based
        "the {size} {color} {obj}",
        "the {obj} with {feature}",
        # Count-based disambiguation
        "the second {obj} from the left",
        "the {obj} closest to the door",
        # Multi-relation
        "the {obj} between the {ref} and the wall",
    ]

    colors = ["brown", "white", "black", "red", "blue", "gray", "beige"]
    sizes = ["small", "large", "tall", "short", "wide"]
    features = ["wooden legs", "cushions", "wheels", "a metal frame", "drawers"]
    reference_objects = ["wall", "door", "window", "table", "desk", "bed"]

    scenes = [
        "scene0000_00", "scene0011_00", "scene0025_00",
        "scene0050_00", "scene0100_00", "scene0200_00",
    ]

    samples = []
    for i in range(n_samples):
        obj_name = object_categories[i % len(object_categories)]
        template = description_templates[i % len(description_templates)]
        color = colors[i % len(colors)]
        size = sizes[i % len(sizes)]
        feature = features[i % len(features)]
        ref = reference_objects[i % len(reference_objects)]

        description = template.format(
            obj=obj_name,
            color=color,
            size=size,
            feature=feature,
            ref=ref,
        )

        # Generate realistic 3D bounding box
        # Center positions in a room-scale coordinate system
        center = [
            float((i * 0.7) % 8 - 4),  # x: -4 to 4 meters
            float((i * 0.5) % 6 - 3),  # y: -3 to 3 meters
            float(0.5 + (i * 0.1) % 1.5),  # z: 0.5 to 2.0 meters (height)
        ]

        # Size varies by object type
        if obj_name in ["chair", "lamp", "toilet"]:
            size_bbox = [0.5, 0.5, 0.8]
        elif obj_name in ["table", "desk", "bed"]:
            size_bbox = [1.2, 0.8, 0.6]
        elif obj_name in ["sofa", "bookshelf"]:
            size_bbox = [1.5, 0.8, 0.9]
        else:
            size_bbox = [0.6, 0.6, 0.7]

        # Add some variation
        size_bbox = [s * (0.8 + (i % 5) * 0.1) for s in size_bbox]

        target_bbox = BoundingBox3D(
            center=center,
            size=size_bbox,
        )

        samples.append(
            MockScanReferSample(
                sample_id=f"mock_scanrefer_{i:04d}",
                scene_id=scenes[i % len(scenes)],
                object_id=str(i % 100),
                object_name=obj_name,
                description=description,
                target_bbox=target_bbox,
                ann_id=str(i),
            )
        )

    return samples


def create_mock_stage1_factory():
    """Create a mock Stage 1 factory for testing without real scene data.

    The mock simulates KeyframeSelector behavior for visual grounding tasks,
    returning keyframes that would likely contain the referred object.
    """
    hypothesis_kinds = ["direct", "proxy", "context", "spatial"]

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
                "task_type": "visual_grounding",
                "object_candidates": [
                    {"name": "chair", "confidence": 0.9, "bbox_estimate": True},
                    {"name": "table", "confidence": 0.85, "bbox_estimate": True},
                ],
            }
            return result

        mock_selector.select_keyframes_v2 = mock_select
        return mock_selector

    return factory


def run_scanrefer_stage1_only(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/baselines/scanrefer_stage1_only.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    object_name: Optional[str] = None,
    scene_id: Optional[str] = None,
    split: str = "val",
    verbose: bool = True,
) -> EvalRunResult:
    """Run Stage 1 only baseline on ScanRefer.

    This evaluates keyframe retrieval quality for visual grounding tasks
    without the Stage 2 agent, establishing a baseline for the two-stage pipeline.

    Args:
        data_root: Path to ScanRefer dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock Stage 1.
        object_name: Filter by object category.
        scene_id: Filter by specific ScanNet scene.
        split: Dataset split (train/val).
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with Stage 1 only metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Stage 1 Only Baseline: ScanRefer (Visual Grounding)")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("scanrefer_stage1_only_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock ScanRefer samples")
        mock_samples = create_mock_scanrefer_samples(max_samples or 50)
        samples = adapt_scanrefer_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
    else:
        if data_root is None:
            raise ValueError("data_root is required when not using mock data")

        logger.info(f"Loading ScanRefer from {data_root}")
        dataset = ScanReferDataset.from_path(
            data_root,
            split=split,
            object_name=object_name,
            scene_id=scene_id,
            max_samples=max_samples,
        )
        samples = adapt_scanrefer_samples(list(dataset))
        stage1_factory = None

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Configure batch evaluation - Stage 2 DISABLED
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="scanrefer",
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
            scene_path = data_root.parent / "scans" / sid
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
        "experiment": "stage1_only_baseline",
        "benchmark": "scanrefer",
        "task_type": "visual_grounding",
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
            "purpose": "Establish Stage 1 baseline for ScanRefer (visual grounding)",
            "key_insight": "Stage 1 retrieves keyframes likely containing referred objects",
            "limitation": "No bounding box prediction - only keyframe retrieval",
            "comparison_target": "scanrefer_stage2_full.json (full pipeline with 3D IoU)",
            "grounding_note": "Visual grounding requires 3D bbox output; Stage 1 only retrieves views",
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
    logger.info("EVALUATION SUMMARY: Stage 1 Only Baseline (ScanRefer)")
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
        description="Run Stage 1 only baseline on ScanRefer benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        help="Path to ScanRefer dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/scanrefer_stage1_only.json"),
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
        help="Use mock data for testing (no real ScanRefer required)",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        help="Filter by object category",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        help="Filter by specific ScanNet scene ID",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
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
        run_scanrefer_stage1_only(
            data_root=args.data_root,
            output_path=args.output,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_mock=args.mock,
            object_name=args.object_name,
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
