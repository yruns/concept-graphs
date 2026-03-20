"""Tests for multi-benchmark adapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.agents.benchmark_adapters import (
    BenchmarkSampleInfo,
    MockFrameProvider,
    MultiBenchmarkAdapter,
    OpenEQAFrameProvider,
    ReplicaFrameProvider,
    ScanNetFrameProvider,
    build_evidence_bundle_from_frames,
    build_task_spec_from_sample,
    create_adapter_for_benchmark,
    extract_sample_info,
)
from conceptgraph.agents.models import Stage2EvidenceBundle, Stage2TaskSpec


# Test fixtures
@pytest.fixture
def mock_openeqa_sample():
    """Create a mock OpenEQA sample."""
    sample = MagicMock()
    sample.question_id = "test-q-001"
    sample.question = "What color is the sofa?"
    sample.answer = "blue"
    sample.scene_id = "hm3d-scene-001"
    sample.category = "object_recognition"
    sample.question_type = "episodic_memory"
    sample.episode_history = Path("/data/episodes/ep001")
    return sample


@pytest.fixture
def mock_sqa3d_sample():
    """Create a mock SQA3D sample."""
    sample = MagicMock()
    sample.question_id = "sqa-001"
    sample.question = "Which direction should I turn to face the window?"
    sample.scene_id = "scene0001_00"
    sample.primary_answer = "left"
    sample.answers = ["left", "to the left"]
    sample.question_type = "where"
    sample.situation = MagicMock()
    sample.situation.position = [1.0, 2.0, 0.0]
    sample.situation.orientation = [0.0, 1.0, 0.0]
    sample.situation.room_description = "I am standing in a living room near a sofa."
    return sample


@pytest.fixture
def mock_scanrefer_sample():
    """Create a mock ScanRefer sample."""
    sample = MagicMock()
    sample.sample_id = "sr-001"
    sample.description = "The wooden chair to the left of the desk"
    sample.scene_id = "scene0002_00"
    sample.object_id = "42"
    sample.object_name = "chair"
    sample.target_bbox = MagicMock()
    sample.target_bbox.to_dict.return_value = {
        "center": [1.0, 2.0, 0.5],
        "size": [0.5, 0.5, 1.0],
    }
    return sample


class TestExtractSampleInfo:
    """Tests for extract_sample_info function."""

    def test_extract_openeqa(self, mock_openeqa_sample):
        info = extract_sample_info(mock_openeqa_sample, "openeqa")

        assert info.benchmark_type == "openeqa"
        assert info.sample_id == "test-q-001"
        assert info.query == "What color is the sofa?"
        assert info.scene_id == "hm3d-scene-001"
        assert info.ground_truth == "blue"
        assert info.task_type == "qa"
        assert info.extra["category"] == "object_recognition"

    def test_extract_sqa3d(self, mock_sqa3d_sample):
        info = extract_sample_info(mock_sqa3d_sample, "sqa3d")

        assert info.benchmark_type == "sqa3d"
        assert info.sample_id == "sqa-001"
        assert info.query == "Which direction should I turn to face the window?"
        assert info.scene_id == "scene0001_00"
        assert info.ground_truth == "left"
        assert info.task_type == "qa"
        assert "situation" in info.extra
        assert info.extra["situation"]["room_description"].startswith("I am standing")

    def test_extract_scanrefer(self, mock_scanrefer_sample):
        info = extract_sample_info(mock_scanrefer_sample, "scanrefer")

        assert info.benchmark_type == "scanrefer"
        assert info.sample_id == "sr-001"
        assert info.query == "The wooden chair to the left of the desk"
        assert info.scene_id == "scene0002_00"
        assert info.task_type == "visual_grounding"
        assert "center" in info.ground_truth
        assert info.extra["object_name"] == "chair"

    def test_extract_replica_dict(self):
        sample = {
            "sample_id": "replica-001",
            "query": "Find the pillow",
            "scene_id": "room0",
            "ground_truth": "on the sofa",
            "task_type": "qa",
        }
        info = extract_sample_info(sample, "replica")

        assert info.benchmark_type == "replica"
        assert info.sample_id == "replica-001"
        assert info.query == "Find the pillow"
        assert info.scene_id == "room0"


class TestBuildEvidenceBundleFromFrames:
    """Tests for build_evidence_bundle_from_frames function."""

    def test_basic_bundle_creation(self):
        sample_info = BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id="test-001",
            query="What is on the table?",
            scene_id="scene001",
            ground_truth="a book",
            task_type="qa",
        )
        frame_paths = [Path("/frames/f1.png"), Path("/frames/f2.png")]

        bundle = build_evidence_bundle_from_frames(sample_info, frame_paths)

        assert isinstance(bundle, Stage2EvidenceBundle)
        assert bundle.scene_id == "scene001"
        assert bundle.stage1_query == "What is on the table?"
        assert len(bundle.keyframes) == 2
        assert bundle.keyframes[0].image_path == "/frames/f1.png"
        assert bundle.hypothesis.status == "benchmark_direct"

    def test_empty_frames(self):
        sample_info = BenchmarkSampleInfo(
            benchmark_type="sqa3d",
            sample_id="test-002",
            query="Where is the door?",
            scene_id="scene002",
            ground_truth="behind you",
            task_type="qa",
        )

        bundle = build_evidence_bundle_from_frames(sample_info, [])

        assert len(bundle.keyframes) == 0
        assert bundle.hypothesis is not None

    def test_sqa3d_scene_summary_from_situation(self):
        sample_info = BenchmarkSampleInfo(
            benchmark_type="sqa3d",
            sample_id="test-003",
            query="Which way is the window?",
            scene_id="scene003",
            ground_truth="right",
            task_type="qa",
            extra={
                "situation": {
                    "room_description": "I am in a bedroom near the bed.",
                    "position": [0, 0, 0],
                    "orientation": [1, 0, 0],
                }
            },
        )

        bundle = build_evidence_bundle_from_frames(sample_info, [])

        assert bundle.scene_summary == "I am in a bedroom near the bed."


class TestBuildTaskSpecFromSample:
    """Tests for build_task_spec_from_sample function."""

    def test_qa_task(self):
        sample_info = BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id="test-001",
            query="What color is the wall?",
            scene_id="scene001",
            ground_truth="white",
            task_type="qa",
        )

        task = build_task_spec_from_sample(sample_info)

        assert isinstance(task, Stage2TaskSpec)
        assert task.task_type == "qa"
        assert task.user_query == "What color is the wall?"
        assert task.plan_mode == "brief"

    def test_visual_grounding_task(self):
        sample_info = BenchmarkSampleInfo(
            benchmark_type="scanrefer",
            sample_id="test-002",
            query="The chair near the window",
            scene_id="scene002",
            ground_truth={"center": [1, 2, 3], "size": [0.5, 0.5, 1]},
            task_type="visual_grounding",
        )

        task = build_task_spec_from_sample(sample_info, plan_mode="full")

        assert task.task_type == "visual_grounding"
        assert task.user_query == "The chair near the window"
        assert task.plan_mode == "full"


class TestFrameProviders:
    """Tests for frame provider classes."""

    def test_mock_provider_empty(self):
        provider = MockFrameProvider()
        sample_info = BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id="test",
            query="test",
            scene_id="test",
            ground_truth=None,
            task_type="qa",
        )

        frames = provider.get_frames(sample_info, max_frames=5)

        assert frames == []

    def test_mock_provider_with_dir(self, tmp_path):
        # Create mock frame files
        for i in range(3):
            (tmp_path / f"frame_{i}.png").touch()

        provider = MockFrameProvider(mock_frame_dir=tmp_path)
        sample_info = BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id="test",
            query="test",
            scene_id="test",
            ground_truth=None,
            task_type="qa",
        )

        frames = provider.get_frames(sample_info, max_frames=5)

        assert len(frames) == 3

    def test_openeqa_provider_missing_dir(self):
        provider = OpenEQAFrameProvider(Path("/nonexistent"))
        sample_info = BenchmarkSampleInfo(
            benchmark_type="openeqa",
            sample_id="test",
            query="test",
            scene_id="test",
            ground_truth=None,
            task_type="qa",
            extra={"episode_history": Path("/nonexistent/episode")},
        )

        frames = provider.get_frames(sample_info, max_frames=5)

        assert frames == []

    def test_openeqa_provider_wrong_benchmark(self):
        provider = OpenEQAFrameProvider(Path("/data"))
        sample_info = BenchmarkSampleInfo(
            benchmark_type="sqa3d",  # Wrong type
            sample_id="test",
            query="test",
            scene_id="test",
            ground_truth=None,
            task_type="qa",
        )

        frames = provider.get_frames(sample_info, max_frames=5)

        assert frames == []


class TestMultiBenchmarkAdapter:
    """Tests for MultiBenchmarkAdapter class."""

    def test_prepare_stage2_inputs(self, mock_openeqa_sample):
        provider = MockFrameProvider()
        adapter = MultiBenchmarkAdapter(
            frame_provider=provider,
            max_frames=5,
            plan_mode="brief",
        )

        task, bundle = adapter.prepare_stage2_inputs(
            mock_openeqa_sample, "openeqa"
        )

        assert isinstance(task, Stage2TaskSpec)
        assert isinstance(bundle, Stage2EvidenceBundle)
        assert task.task_type == "qa"
        assert bundle.stage1_query == "What color is the sofa?"

    def test_get_ground_truth(self, mock_openeqa_sample):
        provider = MockFrameProvider()
        adapter = MultiBenchmarkAdapter(frame_provider=provider)

        gt = adapter.get_ground_truth(mock_openeqa_sample, "openeqa")

        assert gt == "blue"

    def test_get_ground_truth_scanrefer(self, mock_scanrefer_sample):
        provider = MockFrameProvider()
        adapter = MultiBenchmarkAdapter(frame_provider=provider)

        gt = adapter.get_ground_truth(mock_scanrefer_sample, "scanrefer")

        assert "center" in gt
        assert "size" in gt


class TestCreateAdapterForBenchmark:
    """Tests for create_adapter_for_benchmark factory."""

    def test_create_openeqa_adapter(self, tmp_path):
        adapter = create_adapter_for_benchmark("openeqa", tmp_path)

        assert isinstance(adapter, MultiBenchmarkAdapter)
        assert isinstance(adapter.frame_provider, OpenEQAFrameProvider)

    def test_create_sqa3d_adapter_no_scannet(self, tmp_path):
        adapter = create_adapter_for_benchmark("sqa3d", tmp_path)

        # Should fall back to mock provider
        assert isinstance(adapter, MultiBenchmarkAdapter)
        assert isinstance(adapter.frame_provider, MockFrameProvider)

    def test_create_sqa3d_adapter_with_scannet(self, tmp_path):
        scannet_dir = tmp_path / "scannet"
        scannet_dir.mkdir()

        adapter = create_adapter_for_benchmark(
            "sqa3d", tmp_path, scannet_root=scannet_dir
        )

        assert isinstance(adapter.frame_provider, ScanNetFrameProvider)

    def test_create_replica_adapter(self, tmp_path):
        adapter = create_adapter_for_benchmark("replica", tmp_path)

        assert isinstance(adapter.frame_provider, ReplicaFrameProvider)

    def test_create_with_kwargs(self, tmp_path):
        adapter = create_adapter_for_benchmark(
            "openeqa",
            tmp_path,
            max_frames=10,
            plan_mode="full",
        )

        assert adapter.max_frames == 10
        assert adapter.plan_mode == "full"


class TestIntegrationWithRealBenchmarks:
    """Integration tests using real benchmark loaders (if data available)."""

    @pytest.mark.skipif(
        not Path("data/benchmarks/open-eqa").exists(),
        reason="OpenEQA data not available",
    )
    def test_openeqa_integration(self):
        from conceptgraph.benchmarks.openeqa_loader import OpenEQADataset

        ds = OpenEQADataset.from_path("data/benchmarks/open-eqa", max_samples=1)
        sample = ds[0]

        adapter = create_adapter_for_benchmark(
            "openeqa", Path("data/benchmarks/open-eqa")
        )
        task, bundle = adapter.prepare_stage2_inputs(sample, "openeqa")

        assert task.task_type == "qa"
        assert bundle.scene_id  # Should have scene_id

    @pytest.mark.skipif(
        not Path("data/benchmarks/SQA3D").exists(),
        reason="SQA3D data not available",
    )
    def test_sqa3d_integration(self):
        from conceptgraph.benchmarks.sqa3d_loader import SQA3DDataset

        ds = SQA3DDataset.from_path("data/benchmarks/SQA3D", max_samples=1)
        sample = ds[0]

        adapter = create_adapter_for_benchmark(
            "sqa3d", Path("data/benchmarks/SQA3D")
        )
        task, bundle = adapter.prepare_stage2_inputs(sample, "sqa3d")

        assert task.task_type == "qa"
        assert bundle.scene_summary  # Should have situation description

    @pytest.mark.skipif(
        not Path("data/benchmarks/ScanRefer").exists(),
        reason="ScanRefer data not available",
    )
    def test_scanrefer_integration(self):
        from conceptgraph.benchmarks.scanrefer_loader import ScanReferDataset

        ds = ScanReferDataset.from_path(
            "data/benchmarks/ScanRefer/data", split="test", max_samples=1
        )
        sample = ds[0]

        adapter = create_adapter_for_benchmark(
            "scanrefer", Path("data/benchmarks/ScanRefer")
        )
        task, bundle = adapter.prepare_stage2_inputs(sample, "scanrefer")

        assert task.task_type == "visual_grounding"
        gt = adapter.get_ground_truth(sample, "scanrefer")
        assert "center" in gt
