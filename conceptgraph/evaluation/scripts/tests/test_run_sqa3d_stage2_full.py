"""Tests for run_sqa3d_stage2_full script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.evaluation.scripts.run_sqa3d_stage2_full import (
    MockSQA3DSample,
    MockAgentResult,
    MockToolTrace,
    create_mock_sqa3d_samples,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_sqa3d_stage2_full,
    _compute_stage2_analysis,
    _compute_tool_usage,
    _compute_confidence_distribution,
    _compute_uncertainty_analysis,
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
        assert sample.question_type

    def test_mock_sample_situation_context(self):
        """Test mock samples include situation context specific to SQA3D."""
        samples = create_mock_sqa3d_samples(5)
        for sample in samples:
            situation = sample.situation
            assert situation.position is not None
            assert len(situation.position) == 3  # x, y, z
            assert situation.orientation is not None
            assert len(situation.orientation) == 3  # facing direction
            assert situation.room_description
            assert situation.reference_objects

    def test_mock_sample_question_types(self):
        """Test mock samples cover various SQA3D question types."""
        samples = create_mock_sqa3d_samples(50)
        question_types = set(s.question_type for s in samples)
        # Should have spatial and object-centric question types
        assert len(question_types) >= 5


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_keyframes_with_query(self):
        """Test mock selector returns keyframes with query attribute."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0011_00")
        query = "Where is the table relative to me?"
        result = selector.select_keyframes_v2(query, k=3)
        assert len(result.keyframe_paths) == 3
        assert result.query == query

    def test_selector_metadata_situation_aware(self):
        """Test mock selector metadata indicates situation awareness."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0025_00")
        result = selector.select_keyframes_v2("What is on my left?", k=3)
        assert result.metadata["situation_aware"] is True


class TestMockStage2Factory:
    """Tests for mock Stage 2 factory."""

    def test_factory_creates_agent(self):
        """Test factory creates mock agent."""
        factory = create_mock_stage2_factory()
        agent = factory()
        assert agent is not None
        assert hasattr(agent, "run")

    def test_agent_returns_result(self):
        """Test mock agent returns properly structured result."""
        factory = create_mock_stage2_factory()
        agent = factory()

        from unittest.mock import MagicMock
        mock_task_spec = MagicMock()
        mock_evidence = MagicMock()

        result = agent.run(mock_task_spec, mock_evidence)
        assert isinstance(result, MockAgentResult)
        assert result.result is not None
        assert result.tool_trace is not None

    def test_agent_uses_multiple_tools(self):
        """Test full Stage 2 agent uses multiple tools (unlike one-shot)."""
        factory = create_mock_stage2_factory()
        agent = factory()

        from unittest.mock import MagicMock
        all_tool_names = set()
        for i in range(20):
            mock_task_spec = MagicMock()
            mock_evidence = MagicMock()
            result = agent.run(mock_task_spec, mock_evidence)
            for trace in result.tool_trace:
                all_tool_names.add(trace.tool_name)

        # Full agent should use various tools
        assert "inspect_stage1_metadata" in all_tool_names
        assert "request_more_views" in all_tool_names or "request_crops" in all_tool_names

    def test_agent_varying_confidence(self):
        """Test agent produces varying confidence levels."""
        factory = create_mock_stage2_factory()
        agent = factory()

        from unittest.mock import MagicMock
        confidences = []
        for i in range(10):
            mock_task_spec = MagicMock()
            mock_evidence = MagicMock()
            result = agent.run(mock_task_spec, mock_evidence)
            confidences.append(result.result.confidence)

        assert max(confidences) > min(confidences)

    def test_agent_includes_plan(self):
        """Test full agent includes execution plan."""
        factory = create_mock_stage2_factory()
        agent = factory()

        from unittest.mock import MagicMock
        mock_task_spec = MagicMock()
        mock_evidence = MagicMock()
        result = agent.run(mock_task_spec, mock_evidence)

        # Full agent should have a plan
        assert result.result.plan is not None
        assert len(result.result.plan) > 0


class TestRunSQA3DStage2Full:
    """Tests for the main full evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_sqa3d_stage2_full(
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
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check required top-level keys
            assert "experiment" in data
            assert data["experiment"] == "stage2_full_evaluation"
            assert "benchmark" in data
            assert data["benchmark"] == "sqa3d"
            assert "summary" in data
            assert "stage2_analysis" in data
            assert "tool_usage_distribution" in data
            assert "confidence_distribution" in data
            assert "uncertainty_analysis" in data
            assert "per_sample_results" in data
            assert "academic_notes" in data

    def test_full_stage2_config(self):
        """Test full Stage 2 configuration is correct."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
                max_turns=6,
                plan_mode="brief",
            )

            with open(output_path) as f:
                data = json.load(f)

            config = data["config"]
            assert config["stage2_enabled"] is True
            assert config["stage2_max_turns"] == 6
            assert config["stage2_plan_mode"] == "brief"

            # All tools should be ENABLED for full evaluation
            tools = config["tools"]
            assert tools["request_more_views"] is True
            assert tools["request_crops"] is True
            assert tools["hypothesis_repair"] is True

            # Uncertainty should be enabled
            uncertainty = config["uncertainty"]
            assert uncertainty["enabled"] is True

    def test_uncertainty_config(self):
        """Test uncertainty configuration is respected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
                confidence_threshold=0.85,
            )

            with open(output_path) as f:
                data = json.load(f)

            uncertainty = data["config"]["uncertainty"]
            assert uncertainty["confidence_threshold"] == 0.85


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
                stage2_confidence=0.75 + i * 0.05,
                stage2_status="completed",
                stage2_tool_calls=2 + i,
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
        assert analysis["avg_tool_calls"] > 2
        assert analysis["max_tool_calls"] == 6
        assert analysis["min_tool_calls"] == 2
        assert analysis["avg_confidence"] > 0.75
        assert analysis["complete_rate"] == 1.0

    def test_compute_tool_usage(self):
        """Test tool usage distribution computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                tool_trace=[
                    {"tool_name": "inspect_stage1_metadata"},
                    {"tool_name": "request_more_views"},
                ],
            )
            for i in range(3)
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        usage = _compute_tool_usage(run_result)
        assert usage["inspect_stage1_metadata"] == 3
        assert usage["request_more_views"] == 3

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
            for i, conf in enumerate([0.92, 0.88, 0.82, 0.72, 0.62, 0.52])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        dist = _compute_confidence_distribution(run_result)
        assert dist["very_high_0.9+"] == 1
        assert dist["high_0.8-0.9"] == 2
        assert dist["medium_0.7-0.8"] == 1
        assert dist["low_0.6-0.7"] == 1
        assert dist["very_low_<0.6"] == 1

    def test_compute_uncertainty_analysis(self):
        """Test uncertainty analysis computation."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q0",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_status="completed",
                stage2_confidence=0.9,
            ),
            EvalSampleResult(
                sample_id="s1",
                query="Q1",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_status="insufficient_evidence",
                stage2_confidence=0.4,
            ),
            EvalSampleResult(
                sample_id="s2",
                query="Q2",
                task_type="qa",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_status="completed",
                stage2_confidence=0.88,
            ),
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="sqa3d",
            config={},
            results=results,
        )

        analysis = _compute_uncertainty_analysis(run_result)
        assert analysis["insufficient_evidence_count"] == 1
        assert analysis["high_confidence_count"] == 2  # >= 0.85
        assert abs(analysis["uncertainty_ratio"] - 1 / 3) < 0.01


class TestAcademicAlignment:
    """Tests to verify academic alignment of the full Stage 2 evaluation."""

    def test_academic_notes_present(self):
        """Test academic notes are included in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "core_claim" in notes
            assert "innovation_points" in notes
            assert "sqa3d_specific_insights" in notes

    def test_core_claim_present(self):
        """Test core academic claim is stated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            claim = data["academic_notes"]["core_claim"]
            assert "evidence" in claim.lower() or "iterative" in claim.lower()

    def test_innovation_points_present(self):
        """Test innovation points are documented."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            innovations = data["academic_notes"]["innovation_points"]
            assert len(innovations) >= 3
            # Check key innovation areas are mentioned
            all_text = " ".join(innovations).lower()
            assert "evidence" in all_text or "adaptive" in all_text
            assert "uncertainty" in all_text or "repair" in all_text

    def test_sqa3d_specific_insights(self):
        """Test SQA3D-specific insights are documented."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            insights = data["academic_notes"]["sqa3d_specific_insights"]
            assert len(insights) >= 2
            # Should mention spatial/situated aspects
            all_text = " ".join(insights).lower()
            assert "spatial" in all_text or "situated" in all_text or "position" in all_text or "view" in all_text

    def test_comparison_baselines_listed(self):
        """Test comparison baselines are listed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            baselines = data["academic_notes"]["comparison_baselines"]
            assert any("stage1_only" in b for b in baselines)
            assert any("oneshot" in b for b in baselines)

    def test_full_supports_comparison(self):
        """Test full Stage 2 output supports comparison with baselines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Per-sample results should have fields for comparison
            for sample in data["per_sample_results"]:
                assert "sample_id" in sample
                assert "query" in sample
                assert "stage1_success" in sample
                assert "stage2_success" in sample
                assert "stage2_confidence" in sample
                assert "stage2_tool_calls" in sample
                assert "stage2_latency_ms" in sample


class TestIntegration:
    """Integration tests for the full evaluation pipeline."""

    def test_end_to_end_mock_evaluation(self):
        """Test full end-to-end evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sqa3d_stage2_full.json"
            result = run_sqa3d_stage2_full(
                output_path=output_path,
                max_samples=10,
                max_workers=2,
                use_mock=True,
                verbose=False,
                plan_mode="brief",
                max_turns=4,
                confidence_threshold=0.7,
            )

            # Verify run result
            assert result.total_samples == 10
            assert result.failed_stage1 == 0
            assert len(result.results) == 10

            # Verify output file
            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)

            assert len(data["per_sample_results"]) == 10
            assert data["summary"]["total_samples"] == 10

    def test_output_reproducibility(self):
        """Test that output structure is consistent across runs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            required_keys = {
                "experiment", "benchmark", "run_id", "timestamp",
                "config", "summary", "stage2_analysis",
                "tool_usage_distribution", "confidence_distribution",
                "uncertainty_analysis", "per_sample_results", "academic_notes"
            }

            for i in range(2):
                output_path = Path(tmp_dir) / f"results_{i}.json"
                run_sqa3d_stage2_full(
                    output_path=output_path,
                    max_samples=3,
                    use_mock=True,
                    verbose=False,
                )

                with open(output_path) as f:
                    data = json.load(f)

                # Check all required keys are present
                assert required_keys.issubset(set(data.keys()))
