"""Tests for run_sqa3d_oneshot script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.evaluation.scripts.run_sqa3d_oneshot import (
    MockSQA3DSample,
    MockAgentResult,
    MockToolTrace,
    create_mock_sqa3d_samples,
    create_mock_stage1_factory,
    create_mock_stage2_oneshot_factory,
    run_sqa3d_oneshot,
    _compute_stage2_analysis,
    _compute_confidence_distribution,
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
            assert len(situation.position) == 3
            assert situation.orientation is not None
            assert len(situation.orientation) == 3
            assert situation.room_description


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_keyframes(self):
        """Test mock selector returns keyframes with query attribute."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0011_00")
        result = selector.select_keyframes_v2("What color is the chair?", k=3)
        assert len(result.keyframe_paths) == 3
        assert result.query == "What color is the chair?"


class TestMockStage2OneshotFactory:
    """Tests for mock Stage 2 one-shot factory."""

    def test_factory_creates_agent(self):
        """Test factory creates mock agent."""
        factory = create_mock_stage2_oneshot_factory()
        agent = factory()
        assert agent is not None
        assert hasattr(agent, "run")

    def test_agent_returns_result(self):
        """Test mock agent returns properly structured result."""
        factory = create_mock_stage2_oneshot_factory()
        agent = factory()

        # Create minimal mock input
        from unittest.mock import MagicMock
        mock_task_spec = MagicMock()
        mock_evidence = MagicMock()

        result = agent.run(mock_task_spec, mock_evidence)
        assert isinstance(result, MockAgentResult)
        assert result.result is not None
        assert result.tool_trace is not None

    def test_agent_minimal_tool_trace(self):
        """Test one-shot agent has minimal tool trace (no iterative tools)."""
        factory = create_mock_stage2_oneshot_factory()
        agent = factory()

        from unittest.mock import MagicMock
        mock_task_spec = MagicMock()
        mock_evidence = MagicMock()

        result = agent.run(mock_task_spec, mock_evidence)
        # One-shot should only have inspect metadata, no iterative tools
        tool_names = [t.tool_name for t in result.tool_trace]
        assert "request_more_views" not in tool_names
        assert "request_crops" not in tool_names
        assert "switch_or_expand_hypothesis" not in tool_names

    def test_agent_varying_confidence(self):
        """Test one-shot agent produces varying confidence levels."""
        factory = create_mock_stage2_oneshot_factory()
        agent = factory()

        from unittest.mock import MagicMock
        confidences = []
        for i in range(10):
            mock_task_spec = MagicMock()
            mock_evidence = MagicMock()
            result = agent.run(mock_task_spec, mock_evidence)
            confidences.append(result.result.confidence)

        # Should have some variance in confidence
        assert max(confidences) > min(confidences)


class TestRunSQA3DOneshot:
    """Tests for the main one-shot evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=5,
                max_workers=2,
                use_mock=True,
                verbose=False,
            )

            assert result.total_samples == 5
            assert result.failed_stage1 == 0
            assert output_path.exists()

    def test_output_file_structure(self):
        """Test output file has correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check required top-level keys
            assert "experiment" in data
            assert data["experiment"] == "oneshot_vlm_baseline"
            assert "benchmark" in data
            assert data["benchmark"] == "sqa3d"
            assert "summary" in data
            assert "stage2_analysis" in data
            assert "confidence_distribution" in data
            assert "per_sample_results" in data
            assert "academic_notes" in data

    def test_oneshot_config(self):
        """Test one-shot configuration is correct."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            config = data["config"]
            assert config["stage2_enabled"] is True
            assert config["stage2_max_turns"] == 1  # ONE-SHOT
            assert config["stage2_plan_mode"] == "off"

            # All tools should be disabled for one-shot
            tools = config["tools"]
            assert tools["request_more_views"] is False
            assert tools["request_crops"] is False
            assert tools["hypothesis_repair"] is False

    def test_stage2_enabled(self):
        """Test that Stage 2 is enabled but limited to one turn."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            summary = data["summary"]
            # Stage 2 should be executed
            assert summary["stage2_success"] > 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_stage2_analysis(self):
        """Test Stage 2 analysis computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_confidence=0.7 + i * 0.05,
                stage2_status="completed",
            )
            for i in range(5)
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        analysis = _compute_stage2_analysis(run_result)
        assert analysis["avg_confidence"] > 0.7
        assert analysis["complete_rate"] == 1.0
        assert analysis["insufficient_evidence_rate"] == 0.0

    def test_compute_stage2_analysis_empty(self):
        """Test Stage 2 analysis with no successful results."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=False,
            )
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        analysis = _compute_stage2_analysis(run_result)
        assert analysis["avg_confidence"] == 0.0

    def test_compute_confidence_distribution(self):
        """Test confidence distribution computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_confidence=conf,
            )
            for i, conf in enumerate([0.95, 0.85, 0.75, 0.65, 0.55])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        dist = _compute_confidence_distribution(run_result)
        assert dist["very_high_0.9+"] == 1
        assert dist["high_0.8-0.9"] == 1
        assert dist["medium_0.7-0.8"] == 1
        assert dist["low_0.6-0.7"] == 1
        assert dist["very_low_<0.6"] == 1


class TestAcademicAlignment:
    """Tests to verify academic alignment of the one-shot baseline."""

    def test_academic_notes_present(self):
        """Test academic notes are included in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "key_differences" in notes
            assert "hypothesis" in notes

    def test_comparison_baselines_listed(self):
        """Test comparison baselines are listed in academic notes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            baselines = notes["comparison_baselines"]
            assert any("stage1_only" in b for b in baselines)
            assert any("stage2_full" in b for b in baselines)

    def test_oneshot_supports_comparison(self):
        """Test one-shot output supports comparison with full Stage 2 results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_oneshot(
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
                assert "stage2_success" in sample
                assert "stage2_confidence" in sample
                assert "stage2_latency_ms" in sample
