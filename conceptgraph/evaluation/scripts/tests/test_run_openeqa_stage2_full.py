"""Tests for run_openeqa_stage2_full.py script.

TASK-032: Run full Stage 2 agent on OpenEQA

Tests cover:
- Mock sample creation
- Mock Stage 1 and Stage 2 factories
- Evaluation workflow
- Report generation
- CLI interface
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.evaluation.scripts.run_openeqa_stage2_full import (
    MockOpenEQASample,
    _compute_confidence_distribution,
    _compute_stage2_analysis,
    _compute_tool_usage,
    _compute_uncertainty_analysis,
    create_mock_openeqa_samples,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    main,
    run_stage2_full_evaluation,
)


class TestMockSampleCreation:
    """Test mock sample creation for development."""

    def test_create_mock_samples_default_count(self):
        """Test default sample creation."""
        samples = create_mock_openeqa_samples()
        assert len(samples) == 50

    def test_create_mock_samples_custom_count(self):
        """Test custom sample count."""
        samples = create_mock_openeqa_samples(n_samples=25)
        assert len(samples) == 25

    def test_mock_sample_structure(self):
        """Test mock sample has required fields."""
        samples = create_mock_openeqa_samples(n_samples=1)
        sample = samples[0]

        assert isinstance(sample, MockOpenEQASample)
        assert hasattr(sample, "question_id")
        assert hasattr(sample, "question")
        assert hasattr(sample, "answer")
        assert hasattr(sample, "category")
        assert hasattr(sample, "scene_id")
        assert hasattr(sample, "question_type")

    def test_mock_samples_unique_ids(self):
        """Test all samples have unique IDs."""
        samples = create_mock_openeqa_samples(n_samples=50)
        ids = [s.question_id for s in samples]
        assert len(ids) == len(set(ids))

    def test_mock_samples_diverse_categories(self):
        """Test samples cover diverse categories."""
        samples = create_mock_openeqa_samples(n_samples=50)
        categories = set(s.category for s in samples)
        # Should have at least 5 different categories
        assert len(categories) >= 5

    def test_mock_samples_diverse_scenes(self):
        """Test samples spread across scenes."""
        samples = create_mock_openeqa_samples(n_samples=50)
        scenes = set(s.scene_id for s in samples)
        assert len(scenes) >= 3


class TestMockStage1Factory:
    """Test mock Stage 1 factory."""

    def test_factory_returns_callable(self):
        """Test factory returns a callable."""
        factory = create_mock_stage1_factory()
        assert callable(factory)

    def test_factory_creates_mock_selector(self):
        """Test factory creates mock selector for scene."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_mock_selector_returns_keyframes(self):
        """Test mock selector returns keyframe paths."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")

        result = selector.select_keyframes_v2("What is on the table?", k=3)

        assert hasattr(result, "keyframe_paths")
        assert len(result.keyframe_paths) == 3
        assert all(isinstance(p, Path) for p in result.keyframe_paths)

    def test_mock_selector_returns_metadata(self):
        """Test mock selector returns metadata with hypothesis info."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")

        result = selector.select_keyframes_v2("What is on the table?")

        assert hasattr(result, "metadata")
        assert "selected_hypothesis_kind" in result.metadata
        assert "query" in result.metadata
        assert "scene_id" in result.metadata
        assert result.metadata["scene_id"] == "scene_001"

    def test_mock_selector_cycles_hypothesis_kinds(self):
        """Test mock selector cycles through hypothesis kinds."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")

        kinds = []
        for _ in range(5):
            result = selector.select_keyframes_v2("query")
            kinds.append(result.metadata["selected_hypothesis_kind"])

        # Should see variety in hypothesis kinds
        assert len(set(kinds)) >= 2


class TestMockStage2Factory:
    """Test mock Stage 2 factory."""

    def test_factory_returns_callable(self):
        """Test factory returns a callable."""
        factory = create_mock_stage2_factory()
        assert callable(factory)

    def test_factory_creates_mock_agent(self):
        """Test factory creates mock agent."""
        factory = create_mock_stage2_factory()

        # Factory takes no arguments, returns an agent
        agent = factory()

        assert agent is not None
        assert hasattr(agent, "run")

    def test_mock_agent_run_returns_result(self):
        """Test mock agent run returns expected result structure."""
        factory = create_mock_stage2_factory()

        agent = factory()

        # Create mock task spec
        task_spec = MagicMock()
        task_spec.user_query = "What is on the table?"

        result = agent.run(task_spec)

        # Result should have .result (Stage2StructuredResponse) and .tool_trace
        assert hasattr(result, "result")
        assert hasattr(result, "tool_trace")

    def test_mock_agent_response_has_confidence(self):
        """Test mock agent response includes confidence."""
        factory = create_mock_stage2_factory()
        agent = factory()

        task_spec = MagicMock()
        task_spec.user_query = "query"

        result = agent.run(task_spec)

        assert result.result.confidence is not None
        assert 0.0 <= result.result.confidence <= 1.0

    def test_mock_agent_response_has_tools_used(self):
        """Test mock agent response includes plan with tools used."""
        factory = create_mock_stage2_factory()
        agent = factory()

        task_spec = MagicMock()
        task_spec.user_query = "query"

        result = agent.run(task_spec)

        # Plan contains the tools used as steps
        assert result.result.plan is not None
        assert isinstance(result.result.plan, list)
        # Tool trace contains tool names and inputs
        assert result.tool_trace is not None
        assert len(result.tool_trace) > 0


class TestAnalysisFunctions:
    """Test analysis helper functions."""

    def test_compute_stage2_analysis_empty(self):
        """Test analysis with no results."""
        mock_result = MagicMock()
        mock_result.results = []

        analysis = _compute_stage2_analysis(mock_result)

        assert analysis["avg_tool_calls"] == 0.0
        assert analysis["avg_confidence"] == 0.0

    def test_compute_stage2_analysis_with_results(self):
        """Test analysis with mock results."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                stage2_success=True,
                stage2_tool_calls=3,
                stage2_confidence=0.85,
                stage2_status="completed",
            ),
            MagicMock(
                stage2_success=True,
                stage2_tool_calls=5,
                stage2_confidence=0.72,
                stage2_status="completed",
            ),
        ]

        analysis = _compute_stage2_analysis(mock_result)

        assert analysis["avg_tool_calls"] == 4.0
        assert abs(analysis["avg_confidence"] - 0.785) < 0.01

    def test_compute_tool_usage_empty(self):
        """Test tool usage with no results."""
        mock_result = MagicMock()
        mock_result.results = []

        usage = _compute_tool_usage(mock_result)

        assert usage == {}

    def test_compute_tool_usage_with_tools(self):
        """Test tool usage computation."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                stage2_success=True,
                tool_trace=[
                    {"tool_name": "inspect_stage1_metadata", "tool_input": {}},
                    {"tool_name": "request_crops", "tool_input": {}},
                ],
            ),
            MagicMock(
                stage2_success=True,
                tool_trace=[
                    {"tool_name": "inspect_stage1_metadata", "tool_input": {}},
                    {"tool_name": "request_more_views", "tool_input": {}},
                ],
            ),
        ]

        usage = _compute_tool_usage(mock_result)

        assert usage["inspect_stage1_metadata"] == 2
        assert usage["request_crops"] == 1
        assert usage["request_more_views"] == 1

    def test_compute_confidence_distribution(self):
        """Test confidence distribution computation."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(stage2_success=True, stage2_confidence=0.95),
            MagicMock(stage2_success=True, stage2_confidence=0.85),
            MagicMock(stage2_success=True, stage2_confidence=0.75),
            MagicMock(stage2_success=True, stage2_confidence=0.65),
            MagicMock(stage2_success=True, stage2_confidence=0.55),
        ]

        dist = _compute_confidence_distribution(mock_result)

        assert dist["very_high_0.9+"] == 1
        assert dist["high_0.8-0.9"] == 1
        assert dist["medium_0.7-0.8"] == 1
        assert dist["low_0.6-0.7"] == 1
        assert dist["very_low_<0.6"] == 1

    def test_compute_uncertainty_analysis(self):
        """Test uncertainty analysis computation."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(stage2_success=True, stage2_status="complete", stage2_confidence=0.9),
            MagicMock(stage2_success=True, stage2_status="insufficient_evidence", stage2_confidence=0.5),
            MagicMock(stage2_success=True, stage2_status="complete", stage2_confidence=0.75),
        ]

        analysis = _compute_uncertainty_analysis(mock_result)

        assert analysis["insufficient_evidence_count"] == 1
        assert analysis["high_confidence_count"] == 1
        assert abs(analysis["uncertainty_ratio"] - 1/3) < 0.01


class TestEvaluationWorkflow:
    """Test the main evaluation workflow."""

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.adapt_openeqa_samples")
    def test_run_with_mock_data(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test evaluation with mock data."""
        # Setup mocks
        mock_adapt.return_value = [
            MagicMock(sample_id="s1", query="q1", scene_id="scene_001"),
        ]

        mock_evaluator = MagicMock()
        mock_run_result = MagicMock()
        mock_run_result.total_samples = 1
        mock_run_result.failed_stage1 = 0
        mock_run_result.failed_stage2 = 0
        mock_run_result.avg_stage1_latency_ms = 100.0
        mock_run_result.avg_stage2_latency_ms = 500.0
        mock_run_result.total_duration_seconds = 10.0
        # Create proper tool traces that can be serialized
        mock_tool_trace = [
            {"tool_name": "inspect_stage1_metadata", "tool_input": {}},
            {"tool_name": "request_crops", "tool_input": {"object_ids": [1, 2]}},
        ]
        mock_result_item = MagicMock()
        mock_result_item.sample_id = "s1"
        mock_result_item.query = "q1"
        mock_result_item.scene_id = "scene_001"
        mock_result_item.stage1_success = True
        mock_result_item.stage1_hypothesis_kind = "direct"
        mock_result_item.stage2_success = True
        mock_result_item.stage2_status = MagicMock(value="complete")
        mock_result_item.stage2_confidence = 0.85
        mock_result_item.stage2_tool_calls = 2
        mock_result_item.tool_trace = mock_tool_trace
        mock_result_item.stage2_latency_ms = 500.0
        mock_run_result.results = [mock_result_item]
        mock_evaluator.run.return_value = mock_run_result
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "results" / "test_output.json"

        result = run_stage2_full_evaluation(
            use_mock=True,
            max_samples=1,
            output_path=output_path,
            verbose=False,
        )

        assert result.total_samples == 1
        assert output_path.exists()

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.adapt_openeqa_samples")
    def test_output_json_structure(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test output JSON has expected structure."""
        mock_adapt.return_value = [MagicMock(sample_id="s1")]

        mock_evaluator = MagicMock()
        mock_result = MagicMock()
        mock_result.total_samples = 1
        mock_result.failed_stage1 = 0
        mock_result.failed_stage2 = 0
        mock_result.avg_stage1_latency_ms = 100.0
        mock_result.avg_stage2_latency_ms = 500.0
        mock_result.total_duration_seconds = 10.0
        mock_result.results = []
        mock_evaluator.run.return_value = mock_result
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "output.json"
        run_stage2_full_evaluation(
            use_mock=True,
            max_samples=1,
            output_path=output_path,
            verbose=False,
        )

        with open(output_path) as f:
            report = json.load(f)

        assert report["experiment"] == "stage2_full_evaluation"
        assert report["benchmark"] == "openeqa"
        assert "config" in report
        assert "summary" in report
        assert "stage2_analysis" in report
        assert "tool_usage_distribution" in report
        assert "academic_notes" in report

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.adapt_openeqa_samples")
    def test_config_enables_all_tools(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test config enables all Stage 2 tools."""
        mock_adapt.return_value = [MagicMock()]
        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = MagicMock(
            total_samples=1,
            failed_stage1=0,
            failed_stage2=0,
            avg_stage1_latency_ms=100.0,
            avg_stage2_latency_ms=500.0,
            total_duration_seconds=10.0,
            results=[],
        )
        mock_evaluator_class.return_value = mock_evaluator

        run_stage2_full_evaluation(
            use_mock=True,
            max_samples=1,
            output_path=tmp_path / "out.json",
            verbose=False,
        )

        # Verify BatchEvalConfig was created with all tools enabled
        config_call = mock_evaluator_class.call_args
        config = config_call[0][0]  # First positional arg is config

        assert config.stage2_enabled is True
        assert config.enable_tool_request_more_views is True
        assert config.enable_tool_request_crops is True
        assert config.enable_tool_hypothesis_repair is True
        assert config.enable_uncertainty_stopping is True

    def test_requires_data_root_without_mock(self):
        """Test validation that data_root is required when not using mock."""
        with pytest.raises(ValueError, match="data_root is required"):
            run_stage2_full_evaluation(use_mock=False, data_root=None)


class TestCLI:
    """Test CLI interface."""

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.run_stage2_full_evaluation")
    def test_cli_with_mock(self, mock_run_eval):
        """Test CLI with --mock flag."""
        mock_run_eval.return_value = MagicMock()

        with patch("sys.argv", ["script", "--mock", "--max_samples", "5"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["use_mock"] is True
        assert call_kwargs["max_samples"] == 5

    @patch("sys.argv", ["script"])
    def test_cli_requires_data_root_or_mock(self):
        """Test CLI requires --data_root or --mock."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code != 0

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.run_stage2_full_evaluation")
    def test_cli_plan_mode_options(self, mock_run_eval):
        """Test CLI plan mode options."""
        mock_run_eval.return_value = MagicMock()

        with patch("sys.argv", ["script", "--mock", "--plan_mode", "full"]):
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["plan_mode"] == "full"

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.run_stage2_full_evaluation")
    def test_cli_max_turns_option(self, mock_run_eval):
        """Test CLI max turns option."""
        mock_run_eval.return_value = MagicMock()

        with patch("sys.argv", ["script", "--mock", "--max_turns", "10"]):
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["max_turns"] == 10

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.run_stage2_full_evaluation")
    def test_cli_confidence_threshold_option(self, mock_run_eval):
        """Test CLI confidence threshold option."""
        mock_run_eval.return_value = MagicMock()

        with patch("sys.argv", ["script", "--mock", "--confidence_threshold", "0.8"]):
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_run_eval.call_args[1]
        assert call_kwargs["confidence_threshold"] == 0.8


class TestAcademicAlignment:
    """Test academic alignment of the evaluation script."""

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.adapt_openeqa_samples")
    def test_academic_notes_in_output(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test academic notes are included in output."""
        mock_adapt.return_value = [MagicMock()]
        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = MagicMock(
            total_samples=1,
            failed_stage1=0,
            failed_stage2=0,
            avg_stage1_latency_ms=100.0,
            avg_stage2_latency_ms=500.0,
            total_duration_seconds=10.0,
            results=[],
        )
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "output.json"
        run_stage2_full_evaluation(
            use_mock=True,
            max_samples=1,
            output_path=output_path,
            verbose=False,
        )

        with open(output_path) as f:
            report = json.load(f)

        notes = report["academic_notes"]
        assert "purpose" in notes
        assert "core_claim" in notes
        assert "innovation_points" in notes
        assert "comparison_baselines" in notes

        # Verify innovation points align with research direction
        innovation_points = notes["innovation_points"]
        assert any("evidence acquisition" in p.lower() for p in innovation_points)
        assert any("repair" in p.lower() or "hypothesis" in p.lower() for p in innovation_points)
        assert any("uncertainty" in p.lower() for p in innovation_points)

    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_stage2_full.adapt_openeqa_samples")
    def test_comparison_baselines_listed(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test comparison baselines are listed for academic context."""
        mock_adapt.return_value = [MagicMock()]
        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = MagicMock(
            total_samples=1,
            failed_stage1=0,
            failed_stage2=0,
            avg_stage1_latency_ms=100.0,
            avg_stage2_latency_ms=500.0,
            total_duration_seconds=10.0,
            results=[],
        )
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "output.json"
        run_stage2_full_evaluation(
            use_mock=True,
            max_samples=1,
            output_path=output_path,
            verbose=False,
        )

        with open(output_path) as f:
            report = json.load(f)

        baselines = report["academic_notes"]["comparison_baselines"]
        assert any("stage1_only" in b for b in baselines)
        assert any("oneshot" in b for b in baselines)
