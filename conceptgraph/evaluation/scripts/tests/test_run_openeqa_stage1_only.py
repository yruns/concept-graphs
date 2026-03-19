"""Tests for run_openeqa_stage1_only script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.evaluation.scripts.run_openeqa_stage1_only import (
    MockOpenEQASample,
    create_mock_openeqa_samples,
    create_mock_stage1_factory,
    run_stage1_only_baseline,
    _compute_hypothesis_distribution,
    _compute_keyframe_statistics,
)
from conceptgraph.evaluation.batch_eval import EvalRunResult, EvalSampleResult


class TestMockOpenEQASamples:
    """Tests for mock sample generation."""

    def test_create_mock_samples(self):
        """Test mock sample creation."""
        samples = create_mock_openeqa_samples(10)
        assert len(samples) == 10
        assert all(isinstance(s, MockOpenEQASample) for s in samples)

    def test_mock_sample_fields(self):
        """Test mock samples have all required fields."""
        samples = create_mock_openeqa_samples(1)
        sample = samples[0]
        assert sample.question_id
        assert sample.question
        assert sample.answer
        assert sample.category
        assert sample.scene_id

    def test_mock_sample_diversity(self):
        """Test mock samples have diverse categories."""
        samples = create_mock_openeqa_samples(50)
        categories = set(s.category for s in samples)
        assert len(categories) > 3, "Should have diverse categories"

    def test_mock_sample_scenes(self):
        """Test mock samples cover multiple scenes."""
        samples = create_mock_openeqa_samples(50)
        scenes = set(s.scene_id for s in samples)
        assert len(scenes) > 1, "Should cover multiple scenes"


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_keyframes(self):
        """Test mock selector returns keyframes."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        result = selector.select_keyframes_v2("What is on the table?", k=3)
        assert len(result.keyframe_paths) == 3
        assert all(isinstance(p, Path) for p in result.keyframe_paths)

    def test_selector_includes_metadata(self):
        """Test mock selector includes metadata."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        result = selector.select_keyframes_v2("Test query", k=3)
        metadata = result.metadata
        assert "selected_hypothesis_kind" in metadata
        assert "query" in metadata
        assert "scene_id" in metadata

    def test_selector_varies_hypothesis_kind(self):
        """Test mock selector produces varied hypothesis kinds."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        kinds = set()
        for i in range(10):
            result = selector.select_keyframes_v2(f"Query {i}", k=3)
            kinds.add(result.metadata["selected_hypothesis_kind"])
        assert len(kinds) > 1, "Should produce varied hypothesis kinds"


class TestRunStage1OnlyBaseline:
    """Tests for the main baseline evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_stage1_only_baseline(
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
            run_stage1_only_baseline(
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
            assert data["benchmark"] == "openeqa"
            assert "summary" in data
            assert "hypothesis_distribution" in data
            assert "keyframe_statistics" in data
            assert "per_sample_results" in data
            assert "academic_notes" in data

    def test_summary_statistics(self):
        """Test summary statistics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_stage1_only_baseline(
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
            assert summary["stage1_success_rate"] == 1.0
            assert summary["avg_stage1_latency_ms"] > 0

    def test_stage2_disabled(self):
        """Test that Stage 2 is disabled in the baseline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_stage1_only_baseline(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            # Stage 2 should not be executed
            for r in result.results:
                assert not r.stage2_success or r.stage2_answer == ""

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
            for i, kind in enumerate(["direct", "direct", "proxy", "context", "direct"])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        dist = _compute_hypothesis_distribution(run_result)
        assert dist["direct"] == 3
        assert dist["proxy"] == 1
        assert dist["context"] == 1

    def test_compute_keyframe_statistics(self):
        """Test keyframe statistics computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage1_keyframe_count=count,
            )
            for i, count in enumerate([3, 5, 4, 2, 6])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        stats = _compute_keyframe_statistics(run_result)
        assert stats["avg_keyframes"] == 4.0
        assert stats["min_keyframes"] == 2
        assert stats["max_keyframes"] == 6
        assert stats["total_keyframes"] == 20

    def test_compute_keyframe_statistics_empty(self):
        """Test keyframe statistics with no successful results."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q",
                task_type="qa",
                scene_id="scene",
                stage1_success=False,
            )
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        stats = _compute_keyframe_statistics(run_result)
        assert stats["avg_keyframes"] == 0.0


class TestAcademicAlignment:
    """Tests to verify academic alignment of the baseline."""

    def test_academic_notes_present(self):
        """Test academic notes are included in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_stage1_only_baseline(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "claim_support" in notes
            assert "expected_improvement_from" in notes

    def test_baseline_supports_comparison(self):
        """Test baseline output supports comparison with Stage 2 results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_stage1_only_baseline(
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
                assert "stage1_keyframe_count" in sample
                assert "stage1_hypothesis_kind" in sample
                assert "stage1_latency_ms" in sample
