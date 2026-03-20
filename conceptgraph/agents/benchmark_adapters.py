"""Multi-benchmark adapters for Stage 2 agent evaluation.

This module provides adapters to run Stage 2 on various benchmarks
(OpenEQA, SQA3D, ScanRefer) either with:
1. Real ConceptGraph scene data (like Replica)
2. Benchmark-provided frames/images
3. Mock/simulated Stage 1 outputs for testing

The key insight is that Stage 2 evaluation can proceed in two modes:
- Full pipeline: Stage 1 retrieval -> Stage 2 reasoning (requires preprocessed scenes)
- Frame-based: Direct benchmark frames -> Stage 2 reasoning (no scene preprocessing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Union

from loguru import logger

from .models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)


# Type definitions for benchmark samples
BenchmarkType = Literal["openeqa", "sqa3d", "scanrefer", "replica"]


@dataclass
class BenchmarkSampleInfo:
    """Unified info extracted from any benchmark sample."""

    benchmark_type: BenchmarkType
    sample_id: str
    query: str  # Question or referring expression
    scene_id: str
    ground_truth: Any  # Answer string, bbox, etc.
    task_type: str  # qa, visual_grounding, etc.
    extra: Dict[str, Any] = field(default_factory=dict)


class FrameProvider(Protocol):
    """Protocol for providing frames to Stage 2."""

    def get_frames(self, sample_info: BenchmarkSampleInfo, max_frames: int) -> List[Path]:
        """Get frame paths for a benchmark sample."""
        ...


class MockFrameProvider:
    """Provides mock frames for testing Stage 2 without real data."""

    def __init__(self, mock_frame_dir: Optional[Path] = None):
        self.mock_frame_dir = mock_frame_dir

    def get_frames(self, sample_info: BenchmarkSampleInfo, max_frames: int) -> List[Path]:
        """Return empty or mock frames."""
        if self.mock_frame_dir and self.mock_frame_dir.exists():
            frames = sorted(self.mock_frame_dir.glob("*.png"))[:max_frames]
            return frames
        return []


class OpenEQAFrameProvider:
    """Provides frames from OpenEQA episode history."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.frames_dir = self.data_root / "frames"

    def get_frames(self, sample_info: BenchmarkSampleInfo, max_frames: int) -> List[Path]:
        """Load frames from OpenEQA episode history."""
        if sample_info.benchmark_type != "openeqa":
            return []

        episode_path = sample_info.extra.get("episode_history")
        if not episode_path:
            return []

        episode_dir = Path(episode_path)
        if not episode_dir.exists():
            logger.warning(f"Episode history not found: {episode_dir}")
            return []

        image_extensions = {".jpg", ".jpeg", ".png"}
        all_frames = sorted(
            f for f in episode_dir.iterdir() if f.suffix.lower() in image_extensions
        )

        if len(all_frames) <= max_frames:
            return all_frames

        step = len(all_frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return [all_frames[i] for i in indices]


class ScanNetFrameProvider:
    """Provides frames from ScanNet scenes (for SQA3D/ScanRefer)."""

    def __init__(self, scannet_root: Path):
        self.scannet_root = Path(scannet_root)

    def get_frames(self, sample_info: BenchmarkSampleInfo, max_frames: int) -> List[Path]:
        """Load frames from ScanNet scene."""
        scene_id = sample_info.scene_id
        if not scene_id:
            return []

        scene_dir = self.scannet_root / scene_id / "color"
        if not scene_dir.exists():
            logger.warning(f"ScanNet scene not found: {scene_dir}")
            return []

        frames = sorted(scene_dir.glob("*.jpg"))[:max_frames]
        return frames


class ReplicaFrameProvider:
    """Provides frames from pre-processed Replica scenes."""

    def __init__(self, replica_root: Path):
        self.replica_root = Path(replica_root)

    def get_frames(self, sample_info: BenchmarkSampleInfo, max_frames: int) -> List[Path]:
        """Load frames from Replica scene."""
        scene_id = sample_info.scene_id
        if not scene_id:
            return []

        # Check annotated frames first, then raw rgb
        scene_dir = self.replica_root / scene_id
        annotated_dir = scene_dir / "annotated_frames_debug"
        rgb_dir = scene_dir / "results"

        frames_dir = None
        if annotated_dir.exists():
            frames_dir = annotated_dir
        elif rgb_dir.exists():
            frames_dir = rgb_dir

        if not frames_dir:
            logger.warning(f"Replica scene frames not found: {scene_dir}")
            return []

        # Try multiple patterns - annotated frames use different naming
        patterns = [
            "frame_*_annotated.jpg",  # annotated_frames_debug format
            "frame_*_annotated_v2.jpg",
            "rgb_*.png",  # results directory format
            "rgb_*.jpg",
            "*.jpg",
            "*.png",
        ]

        frames = []
        for pattern in patterns:
            frames = sorted(frames_dir.glob(pattern))
            if frames:
                break

        return frames[:max_frames]


def extract_sample_info(
    sample: Any,
    benchmark_type: BenchmarkType,
) -> BenchmarkSampleInfo:
    """Extract unified sample info from benchmark-specific sample object."""

    if benchmark_type == "openeqa":
        return BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id=sample.question_id,
            query=sample.question,
            scene_id=sample.scene_id,
            ground_truth=sample.answer,
            task_type="qa",
            extra={
                "episode_history": sample.episode_history,
                "category": sample.category,
                "question_type": sample.question_type,
            },
        )

    elif benchmark_type == "sqa3d":
        return BenchmarkSampleInfo(
            benchmark_type="sqa3d",
            sample_id=sample.question_id,
            query=sample.question,
            scene_id=sample.scene_id,
            ground_truth=sample.primary_answer,
            task_type="qa",
            extra={
                "situation": {
                    "position": sample.situation.position,
                    "orientation": sample.situation.orientation,
                    "room_description": sample.situation.room_description,
                },
                "question_type": sample.question_type,
                "all_answers": sample.answers,
            },
        )

    elif benchmark_type == "scanrefer":
        return BenchmarkSampleInfo(
            benchmark_type="scanrefer",
            sample_id=sample.sample_id,
            query=sample.description,
            scene_id=sample.scene_id,
            ground_truth=sample.target_bbox.to_dict(),
            task_type="visual_grounding",
            extra={
                "object_id": sample.object_id,
                "object_name": sample.object_name,
            },
        )

    elif benchmark_type == "replica":
        # Generic replica sample - expects dict with query/scene_id
        if isinstance(sample, dict):
            return BenchmarkSampleInfo(
                benchmark_type="replica",
                sample_id=sample.get("sample_id", "unknown"),
                query=sample.get("query", ""),
                scene_id=sample.get("scene_id", "room0"),
                ground_truth=sample.get("ground_truth"),
                task_type=sample.get("task_type", "qa"),
                extra=sample.get("extra", {}),
            )
        # Named tuple or dataclass
        return BenchmarkSampleInfo(
            benchmark_type="replica",
            sample_id=getattr(sample, "sample_id", "unknown"),
            query=getattr(sample, "query", ""),
            scene_id=getattr(sample, "scene_id", "room0"),
            ground_truth=getattr(sample, "ground_truth", None),
            task_type=getattr(sample, "task_type", "qa"),
            extra=getattr(sample, "extra", {}),
        )

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def build_evidence_bundle_from_frames(
    sample_info: BenchmarkSampleInfo,
    frame_paths: List[Path],
    scene_summary: str = "",
) -> Stage2EvidenceBundle:
    """Build a Stage2EvidenceBundle directly from benchmark frames.

    This is the "frame-based" mode that bypasses Stage 1 retrieval
    and uses benchmark-provided frames directly.
    """
    keyframes = []
    for idx, frame_path in enumerate(frame_paths):
        keyframes.append(
            KeyframeEvidence(
                keyframe_idx=idx,
                image_path=str(frame_path),
                view_id=idx,  # Use index as view_id
                frame_id=idx,
                score=1.0 / (idx + 1),  # Simple decreasing score
                note="benchmark_provided",
            )
        )

    # Build mock hypothesis from sample info
    hypothesis = Stage1HypothesisSummary(
        status="benchmark_direct",
        hypothesis_kind="direct",
        hypothesis_rank=0,
        parse_mode="benchmark",
        raw_query=sample_info.query,
        target_categories=[],  # Not available without Stage 1
        anchor_categories=[],
        metadata={
            "benchmark_type": sample_info.benchmark_type,
            "sample_id": sample_info.sample_id,
        },
    )

    # Build scene summary from situation if available
    if not scene_summary and sample_info.benchmark_type == "sqa3d":
        situation = sample_info.extra.get("situation", {})
        scene_summary = situation.get("room_description", "")

    return Stage2EvidenceBundle(
        scene_id=sample_info.scene_id,
        stage1_query=sample_info.query,
        keyframes=keyframes,
        bev_image_path=None,  # No BEV for benchmark-direct mode
        scene_summary=scene_summary,
        object_context={},  # Not available without Stage 1
        hypothesis=hypothesis,
        extra_metadata={
            "benchmark_type": sample_info.benchmark_type,
            "sample_id": sample_info.sample_id,
            "mode": "benchmark_direct",
        },
    )


def build_task_spec_from_sample(
    sample_info: BenchmarkSampleInfo,
    plan_mode: str = "brief",
    max_reasoning_turns: int = 5,
) -> Stage2TaskSpec:
    """Build a Stage2TaskSpec from benchmark sample info."""
    return Stage2TaskSpec(
        task_type=sample_info.task_type,
        user_query=sample_info.query,
        plan_mode=plan_mode,
        max_reasoning_turns=max_reasoning_turns,
    )


@dataclass
class MultiBenchmarkAdapter:
    """Unified adapter for running Stage 2 on multiple benchmarks.

    Supports two operational modes:
    1. Full pipeline: Uses KeyframeSelector for Stage 1 retrieval
    2. Frame-based: Uses benchmark frames directly (for testing or when no scene data)
    """

    frame_provider: FrameProvider
    max_frames: int = 8
    plan_mode: str = "brief"
    max_reasoning_turns: int = 5

    def prepare_stage2_inputs(
        self,
        sample: Any,
        benchmark_type: BenchmarkType,
        scene_summary: str = "",
    ) -> tuple[Stage2TaskSpec, Stage2EvidenceBundle]:
        """Prepare Stage 2 inputs from a benchmark sample.

        This uses frame-based mode which bypasses Stage 1 retrieval.
        For full pipeline mode, use the BatchEvaluator instead.

        Args:
            sample: Benchmark-specific sample object
            benchmark_type: Type of benchmark
            scene_summary: Optional scene description

        Returns:
            Tuple of (task_spec, evidence_bundle)
        """
        sample_info = extract_sample_info(sample, benchmark_type)

        # Get frames from provider
        frame_paths = self.frame_provider.get_frames(sample_info, self.max_frames)

        # Build evidence bundle
        bundle = build_evidence_bundle_from_frames(
            sample_info, frame_paths, scene_summary
        )

        # Build task spec
        task = build_task_spec_from_sample(
            sample_info,
            plan_mode=self.plan_mode,
            max_reasoning_turns=self.max_reasoning_turns,
        )

        return task, bundle

    def get_ground_truth(
        self,
        sample: Any,
        benchmark_type: BenchmarkType,
    ) -> Any:
        """Extract ground truth from a benchmark sample."""
        sample_info = extract_sample_info(sample, benchmark_type)
        return sample_info.ground_truth


def create_adapter_for_benchmark(
    benchmark_type: BenchmarkType,
    data_root: Path,
    **kwargs,
) -> MultiBenchmarkAdapter:
    """Factory to create appropriate adapter for a benchmark.

    Args:
        benchmark_type: Type of benchmark
        data_root: Root directory for benchmark data
        **kwargs: Additional arguments passed to MultiBenchmarkAdapter

    Returns:
        Configured MultiBenchmarkAdapter
    """
    if benchmark_type == "openeqa":
        provider = OpenEQAFrameProvider(data_root)
    elif benchmark_type in ("sqa3d", "scanrefer"):
        # These need ScanNet data - use mock if not available
        scannet_root = kwargs.pop("scannet_root", None)
        if scannet_root and Path(scannet_root).exists():
            provider = ScanNetFrameProvider(scannet_root)
        else:
            logger.warning(
                f"ScanNet root not provided or not found for {benchmark_type}. "
                "Using mock frame provider."
            )
            provider = MockFrameProvider()
    elif benchmark_type == "replica":
        provider = ReplicaFrameProvider(data_root)
    else:
        provider = MockFrameProvider()

    return MultiBenchmarkAdapter(frame_provider=provider, **kwargs)
