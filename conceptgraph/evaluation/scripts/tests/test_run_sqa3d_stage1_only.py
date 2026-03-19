"""Tests for run_sqa3d_stage1_only script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.evaluation.scripts.run_sqa3d_stage1_only import (
    MockSQA3DSample,
    create_mock_sqa3d_samples,
    create_mock_stage1_factory,
    run_sqa3d_stage1_only,
    _compute_hypothesis_distribution,
)
from conceptgraph.evaluation.batch_eval import EvalRunResult, EvalSampleResult


class TestMockSQA3DSamples:
    """Tests for mock sample generation."""

    def test_create_mock_samples(self):
        """Test mock sample creation."""
        samples = create_mock_sqa3d_samples(10)
        assert len(samples) == 10
        assert all(isinstance(s, MockSQA3DSample) for s in samples)

    def test_mock_sample_fields(self):
        """Test mock samples have all required fields."""
        samples = create_mock_sqa3d_samples(1)
        sample = samples[0]
        assert sample.question_id
        assert sample.question
        assert sample.answers
        assert sample.scene_id
        assert sample.situation is not None

    def test_mock_sample_situation_context(self):
        """Test mock samples include situation context specific to SQA3D."""
        samples = create_mock_sqa3d_samples(5)
        for sample in samples:
            situation = sample.situation
            assert situation.position is not None
            assert situation.orientation is not None
            assert situation.room_description

    def test_mock_sample_diversity(self):
        """Test mock samples have diverse question types."""
        samples = create_mock_sqa3d_samples(50)
        question_types = set(s.question_type for s in samples)
        assert len(question_types) > 3, "Should have diverse question types"

    def test_mock_sample_scenes(self):
        """Test mock samples cover multiple scenes."""
        samples = create_mock_sqa3d_samples(50)
        scenes = set(s.scene_id for s in samples)
        assert len(scenes) > 1, "Should cover multiple scenes"

    def test_mock_sample_primary_answer(self):
        """Test primary_answer property works correctly."""
        samples = create_mock_sqa3d_samples(5)
        for sample in samples:
            assert sample.primary_answer == sample.answers[0]


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_keyframes(self):
        """Test mock selector returns keyframes."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0011_00")
        result = selector.select_keyframes_v2("What is on my left?", k=3)
        assert len(result.keyframe_paths) == 3
        assert all(isinstance(p, Path) for p in result.keyframe_paths)

    def test_selector_includes_metadata(self):
        """Test mock selector includes SQA3D-specific metadata."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0025_00")
        result = selector.select_keyframes_v2("Where is the chair relative to me?", k=3)
        metadata = result.metadata
        assert "selected_hypothesis_kind" in metadata
        assert "query" in metadata
        assert "scene_id" in metadata
        assert "situation_aware" in metadata
        assert metadata["situation_aware"] is True

    def test_selector_varies_hypothesis_kind(self):
        """Test mock selector produces varied hypothesis kinds."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0050_00")
        kinds = set()
        for i in range(10):
            result = selector.select_keyframes_v2(f"Query {i}", k=3)
            kinds.add(result.metadata["selected_hypothesis_kind"])
        assert len(kinds) > 1, "Should produce varied hypothesis kinds"

    def test_selector_includes_situated_kind(self):
        """Test mock selector can produce 'situated' hypothesis kind."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        kinds = set()
        for i in range(20):
            result = selector.select_keyframes_v2(f"Spatial query {i}", k=3)
            kinds.add(result.metadata["selected_hypothesis_kind"])
        assert "situated" in kinds, "Should produce 'situated' hypothesis kind for SQA3D"


class TestRunSQA3DStage1OnlyBaseline:
    """Tests for the main baseline evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=5,
                max_workers=2,
                use_mock=True,
                verbose=False,
            )

            assert result.total_samples == 5
            assert result.failed_stage1 == 0  # Mock should succeed
            assert output_path.exists()

    def test_output_file_structure(self):
        """Test output file has correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check required top-level keys
            assert "experiment" in data
            assert data["experiment"] == "stage1_only_baseline"
            assert "benchmark" in data
            assert data["benchmark"] == "sqa3d"
            assert "summary" in data
            assert "hypothesis_distribution" in data
            assert "per_sample_results" in data
            assert "academic_notes" in data

    def test_summary_statistics(self):
        """Test summary statistics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=10,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            summary = data["summary"]
            assert summary["total_samples"] == 10
            assert summary["stage1_success"] == 10  # Mock always succeeds
            assert summary["stage1_failure"] == 0
            assert summary["avg_stage1_latency_ms"] > 0

    def test_stage2_disabled(self):
        """Test that Stage 2 is disabled in the baseline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["config"]["stage2_enabled"] is False
            assert data["config"]["ablation_tag"] == "stage1_only"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_hypothesis_distribution(self):
        """Test hypothesis distribution computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage1_hypothesis_kind=kind,
            )
            for i, kind in enumerate(["direct", "direct", "proxy", "context", "situated"])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        dist = _compute_hypothesis_distribution(run_result)
        assert dist["direct"] == 2
        assert dist["proxy"] == 1
        assert dist["context"] == 1
        assert dist["situated"] == 1

    def test_compute_hypothesis_distribution_with_failures(self):
        """Test hypothesis distribution ignores failed Stage 1 results."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q0",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage1_hypothesis_kind="direct",
            ),
            EvalSampleResult(
                sample_id="s1",
                query="Q1",
                task_type="qa",
                scene_id="scene",
                stage1_success=False,
                stage1_hypothesis_kind="proxy",
            ),
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        dist = _compute_hypothesis_distribution(run_result)
        assert dist.get("direct") == 1
        assert "proxy" not in dist  # Failed, should be ignored


class TestAcademicAlignment:
    """Tests to verify academic alignment of the baseline."""

    def test_academic_notes_present(self):
        """Test academic notes are included in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "SQA3D" in notes["purpose"] or "situated" in notes["purpose"].lower()

    def test_baseline_supports_comparison(self):
        """Test baseline output supports comparison with Stage 2 results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Per-sample results should have fields needed for comparison
            for sample in data["per_sample_results"]:
                assert "sample_id" in sample
                assert "query" in sample
                assert "stage1_success" in sample
                assert "stage1_hypothesis_kind" in sample
                assert "stage1_latency_ms" in sample

    def test_sqa3d_specific_fields(self):
        """Test SQA3D-specific fields in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage1_only(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Benchmark should be sqa3d
            assert data["benchmark"] == "sqa3d"
