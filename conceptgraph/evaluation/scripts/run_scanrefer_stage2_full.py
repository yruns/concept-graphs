"""Run full Stage 2 agent evaluation on ScanRefer benchmark.

ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language
ECCV 2020, https://daveredrum.github.io/ScanRefer/

This script evaluates the complete two-stage pipeline with:
- Stage 1: Keyframe retrieval based on referring expression
- Stage 2: Iterative evidence-seeking agent with tool use

TASK-034: Run ScanRefer experiments

Academic Relevance:
- Full system evaluation for visual grounding (3D object localization)
- Demonstrates adaptive evidence acquisition for spatial reasoning
- Shows symbolic-to-visual repair when Stage 1 hypotheses miss target
- Evidence-grounded uncertainty for bounding box confidence

Key Innovation Points:
1. Iterative Evidence Seeking: Agent can request additional views/crops
2. Hypothesis Repair: Switch between direct/proxy/context hypotheses
3. Uncertainty-Aware Output: Confidence scores for bbox predictions
4. Multi-Turn Reasoning: Refine predictions through evidence gathering

Usage:
    # With real ScanRefer data:
    python -m conceptgraph.evaluation.scripts.run_scanrefer_stage2_full \
        --data_root /path/to/ScanRefer \
        --output results/experiments/scanrefer_stage2_full.json

    # With mock data (for development/testing):
    python -m conceptgraph.evaluation.scripts.run_scanrefer_stage2_full \
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
    compute_iou_3d,
    compute_scanrefer_metrics,
    evaluate_scanrefer,
)
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
    adapt_scanrefer_samples,
)
from conceptgraph.agents.models import Stage2Status


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
    object_categories = [
        "chair", "table", "sofa", "bed", "desk", "cabinet", "bookshelf",
        "refrigerator", "toilet", "sink", "lamp", "door", "window", "pillow",
    ]

    description_templates = [
        "the {obj} next to the {ref}",
        "the {obj} in front of the {ref}",
        "the {obj} behind the {ref}",
        "the {color} {obj} near the window",
        "the {obj} on the left side of the room",
        "the {size} {color} {obj}",
        "the {obj} with {feature}",
        "the second {obj} from the left",
        "the {obj} closest to the door",
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

        center = [
            float((i * 0.7) % 8 - 4),
            float((i * 0.5) % 6 - 3),
            float(0.5 + (i * 0.1) % 1.5),
        ]

        if obj_name in ["chair", "lamp", "toilet"]:
            size_bbox = [0.5, 0.5, 0.8]
        elif obj_name in ["table", "desk", "bed"]:
            size_bbox = [1.2, 0.8, 0.6]
        elif obj_name in ["sofa", "bookshelf"]:
            size_bbox = [1.5, 0.8, 0.9]
        else:
            size_bbox = [0.6, 0.6, 0.7]

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
    """Create a mock Stage 1 factory for testing without real scene data."""
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
                # Additional metadata for tool support
                "available_views": [f"{i:06d}" for i in range(20)],
                "scene_objects": [
                    {"id": "0", "name": "chair", "center": [1.0, 2.0, 0.5]},
                    {"id": "1", "name": "table", "center": [0.0, 1.0, 0.4]},
                    {"id": "2", "name": "sofa", "center": [-1.0, 0.0, 0.5]},
                ],
            }
            return result

        mock_selector.select_keyframes_v2 = mock_select
        return mock_selector

    return factory


def create_mock_stage2_factory():
    """Create a mock Stage 2 factory for full agentic reasoning.

    The mock simulates multi-turn iterative reasoning with tool use:
    - request_more_views: Get additional perspectives
    - request_crops: Zoom in on specific objects
    - switch_or_expand_hypothesis: Try alternative hypotheses

    Returns predictions with varying quality based on tool usage.
    Returns a factory callable that creates MockStage2Agent instances.
    """
    from conceptgraph.agents.models import (
        Stage2AgentResult,
        Stage2EvidenceBundle,
        Stage2StructuredResponse,
        Stage2TaskSpec,
        Stage2TaskType,
        Stage2ToolObservation,
    )

    call_count = [0]

    def mock_run(task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
        call_count[0] += 1
        idx = call_count[0]

        # Simulate full agent with tool use
        # Higher quality predictions due to iterative refinement
        base_iou = 0.4 + (idx % 6) * 0.1  # IoU varies 0.4-0.9

        # Tool usage pattern (varies by sample)
        tool_patterns = [
            ["request_more_views", "request_crops"],
            ["request_crops", "switch_or_expand_hypothesis"],
            ["request_more_views"],
            ["request_crops", "request_more_views", "request_crops"],
            ["switch_or_expand_hypothesis", "request_crops"],
        ]
        tools_used = tool_patterns[idx % len(tool_patterns)]
        num_tool_calls = len(tools_used)

        # Better predictions with more tool use
        improvement_factor = 0.05 * num_tool_calls
        adjusted_iou = min(0.95, base_iou + improvement_factor)

        # Generate predicted bounding box
        noise_factor = 0.5 - (adjusted_iou * 0.5)
        pred_center = [
            1.0 + noise_factor * (idx % 3 - 1),
            2.0 + noise_factor * ((idx + 1) % 3 - 1),
            0.8 + noise_factor * 0.2,
        ]
        pred_size = [
            0.6 * (1 + noise_factor * 0.5),
            0.6 * (1 + noise_factor * 0.5),
            0.8 * (1 + noise_factor * 0.5),
        ]

        confidence = 0.6 + adjusted_iou * 0.35

        # Visual grounding answer with bbox prediction
        answer = (
            f"Located target object at center "
            f"[{pred_center[0]:.2f}, {pred_center[1]:.2f}, {pred_center[2]:.2f}] "
            f"with size [{pred_size[0]:.2f}, {pred_size[1]:.2f}, {pred_size[2]:.2f}]"
        )
        payload = {
            "answer": answer,
            "predicted_bbox": {
                "center": pred_center,
                "size": pred_size,
            },
            "grounding_method": "iterative_evidence_seeking",
            "tools_used": tools_used,
        }

        result = Stage2StructuredResponse(
            task_type=task.task_type,
            status=Stage2Status.COMPLETED,
            summary=answer,
            confidence=confidence,
            uncertainties=[],
            cited_frame_indices=list(range(min(3, len(bundle.keyframes)))),
            evidence_items=[],
            plan=[],
            payload=payload,
        )

        # Create tool trace for the tools used
        tool_trace = [
            Stage2ToolObservation(
                tool_name=tool,
                tool_input={"source": "mock"},
                tool_output=f"Mock {tool} result",
                latency_ms=100.0,
            )
            for tool in tools_used
        ]

        return Stage2AgentResult(
            task=task,
            result=result,
            tool_trace=tool_trace,
            final_bundle=bundle,
            raw_state={},
        )

    class MockStage2Agent:
        def run(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
            return mock_run(task, bundle)

    return lambda: MockStage2Agent()


def run_scanrefer_stage2_full(
    data_root: Optional[Path] = None,
    output_path: Path = Path("results/experiments/scanrefer_stage2_full.json"),
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    use_mock: bool = False,
    object_name: Optional[str] = None,
    scene_id: Optional[str] = None,
    split: str = "val",
    max_turns: int = 5,
    verbose: bool = True,
) -> EvalRunResult:
    """Run full Stage 2 agent on ScanRefer.

    This evaluates the complete two-stage pipeline with iterative
    evidence-seeking agent for visual grounding.

    Args:
        data_root: Path to ScanRefer dataset. Required if use_mock=False.
        output_path: Path to save results JSON.
        max_samples: Maximum number of samples to evaluate.
        max_workers: Number of parallel workers.
        use_mock: If True, use mock data and mock Stage 1/2.
        object_name: Filter by object category.
        scene_id: Filter by specific ScanNet scene.
        split: Dataset split (train/val).
        max_turns: Maximum reasoning turns for Stage 2 agent.
        verbose: Enable verbose logging.

    Returns:
        EvalRunResult with full pipeline metrics.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Full Stage 2 Agent: ScanRefer (Visual Grounding)")
        logger.info("=" * 60)

    run_id = datetime.now().strftime("scanrefer_stage2_full_%Y%m%d_%H%M%S")

    # Load samples
    if use_mock:
        logger.info("Using mock ScanRefer samples")
        mock_samples = create_mock_scanrefer_samples(max_samples or 50)
        samples = adapt_scanrefer_samples(mock_samples)
        stage1_factory = create_mock_stage1_factory()
        stage2_factory = create_mock_stage2_factory()
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
        stage2_factory = None

    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # Configure batch evaluation - Full Stage 2 with ALL tools enabled
    config = BatchEvalConfig(
        run_id=run_id,
        benchmark_name="scanrefer",
        max_workers=max_workers,
        # Stage 1 configuration
        stage1_model="gpt-5.2-2025-12-11",
        stage1_k=3,
        # Stage 2 FULL configuration
        stage2_enabled=True,
        stage2_model="gpt-5.2-2025-12-11",
        stage2_max_turns=max_turns,
        # ALL TOOLS ENABLED for full pipeline
        enable_request_more_views=True,
        enable_request_crops=True,
        enable_hypothesis_repair=True,
        # Output
        output_dir=str(output_path.parent),
        save_raw_outputs=True,
        max_samples=len(samples) if max_samples else None,
    )

    logger.info(f"Config: stage2_enabled={config.stage2_enabled}, max_turns={config.stage2_max_turns}")
    logger.info(f"Ablation tag: {config.get_ablation_tag()}")
    logger.info("Tools: request_more_views, request_crops, hypothesis_repair (ALL ENABLED)")

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

    # Compute detailed statistics
    grounding_stats = _compute_grounding_stats(run_result)
    tool_stats = _compute_tool_usage_stats(run_result)

    experiment_report = {
        "experiment": "stage2_full_pipeline",
        "benchmark": "scanrefer",
        "task_type": "visual_grounding",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "stage2_enabled": True,
            "stage2_max_turns": max_turns,
            "tools_enabled": [
                "request_more_views",
                "request_crops",
                "switch_or_expand_hypothesis",
            ],
            "ablation_tag": config.get_ablation_tag(),
            "stage1_k": config.stage1_k,
            "stage1_model": config.stage1_model,
            "stage2_model": config.stage2_model,
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
        "grounding_metrics": grounding_stats,
        "tool_usage": tool_stats,
        "hypothesis_distribution": _compute_hypothesis_distribution(run_result),
        "per_sample_results": [
            {
                "sample_id": r.sample_id,
                "query": r.query,
                "scene_id": r.scene_id,
                "stage1_success": r.stage1_success,
                "stage1_hypothesis_kind": r.stage1_hypothesis_kind,
                "stage2_success": r.stage2_success,
                "stage2_answer": r.stage2_answer,
                "stage2_confidence": r.stage2_confidence,
                "stage2_tool_calls": r.stage2_tool_calls,
                "stage2_latency_ms": r.stage2_latency_ms,
                "stage2_turns": getattr(r, "stage2_turns", None),
            }
            for r in run_result.results
        ],
        "academic_notes": {
            "purpose": "Full two-stage pipeline evaluation for ScanRefer",
            "innovation_points": [
                "Adaptive Evidence Acquisition: Agent decides what additional evidence to gather",
                "Symbolic-to-Visual Repair: Request crops when objects are ambiguous",
                "Hypothesis Switching: Try alternative retrieval strategies on failure",
                "Evidence-Grounded Uncertainty: Confidence reflects evidence quality",
            ],
            "tools_available": [
                "request_more_views: Get additional viewpoints for spatial reasoning",
                "request_crops: Zoom in on specific objects for detail verification",
                "switch_or_expand_hypothesis: Try alternative retrieval hypotheses",
            ],
            "comparison_targets": {
                "lower_bound_1": "scanrefer_stage1_only.json (retrieval only)",
                "lower_bound_2": "scanrefer_oneshot.json (single-pass VLM)",
            },
            "expected_improvement": "Tool use enables recovery from initial hypothesis failures",
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


def _compute_grounding_stats(run_result: EvalRunResult) -> Dict[str, Any]:
    """Compute visual grounding metrics."""
    successful = sum(1 for r in run_result.results if r.stage2_success)
    with_bbox = sum(
        1 for r in run_result.results
        if r.stage2_success and r.stage2_answer and "center" in str(r.stage2_answer).lower()
    )

    # Confidence distribution
    confidences = [
        r.stage2_confidence for r in run_result.results
        if r.stage2_success and r.stage2_confidence is not None
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "predictions_made": with_bbox,
        "predictions_success_rate": with_bbox / max(successful, 1),
        "avg_confidence": avg_confidence,
        "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
        "low_confidence_count": sum(1 for c in confidences if c < 0.5),
        "note": "IoU metrics require ground truth bbox comparison",
        "estimated_acc_at_025": 0.0,
        "estimated_acc_at_050": 0.0,
    }


def _compute_tool_usage_stats(run_result: EvalRunResult) -> Dict[str, Any]:
    """Compute tool usage statistics."""
    tool_calls = [
        r.stage2_tool_calls for r in run_result.results
        if r.stage2_success and r.stage2_tool_calls is not None
    ]

    if not tool_calls:
        return {
            "avg_tool_calls": 0.0,
            "max_tool_calls": 0,
            "samples_with_tools": 0,
            "samples_no_tools": len(run_result.results),
        }

    return {
        "avg_tool_calls": sum(tool_calls) / len(tool_calls),
        "max_tool_calls": max(tool_calls),
        "min_tool_calls": min(tool_calls),
        "samples_with_tools": sum(1 for t in tool_calls if t > 0),
        "samples_no_tools": sum(1 for t in tool_calls if t == 0),
        "total_tool_calls": sum(tool_calls),
    }


def _print_summary(run_result: EvalRunResult, report: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY: Full Stage 2 Agent (ScanRefer)")
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

    logger.info("Grounding Metrics:")
    gm = report["grounding_metrics"]
    logger.info(f"  Predictions Made:  {gm['predictions_made']}")
    logger.info(f"  Success Rate:      {gm['predictions_success_rate']:.2%}")
    logger.info(f"  Avg Confidence:    {gm['avg_confidence']:.2%}")
    logger.info("")

    logger.info("Tool Usage Statistics:")
    ts = report["tool_usage"]
    logger.info(f"  Avg Tool Calls:    {ts['avg_tool_calls']:.1f}")
    logger.info(f"  With Tools:        {ts['samples_with_tools']}")
    logger.info(f"  Without Tools:     {ts['samples_no_tools']}")
    logger.info("")

    logger.info("Hypothesis Distribution:")
    for kind, count in report["hypothesis_distribution"].items():
        logger.info(f"  {kind}: {count}")
    logger.info("")

    logger.info("Innovation Points:")
    for point in report["academic_notes"]["innovation_points"]:
        logger.info(f"  • {point}")
    logger.info("")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run full Stage 2 agent on ScanRefer benchmark",
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
        default=Path("results/experiments/scanrefer_stage2_full.json"),
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
        "--max_turns",
        type=int,
        default=5,
        help="Maximum reasoning turns for Stage 2 agent",
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
        run_scanrefer_stage2_full(
            data_root=args.data_root,
            output_path=args.output,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_mock=args.mock,
            object_name=args.object_name,
            scene_id=args.scene_id,
            split=args.split,
            max_turns=args.max_turns,
            verbose=not args.quiet,
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
