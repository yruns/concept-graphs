"""Tests for run_openeqa_oneshot.py baseline script.

TASK-031: One-shot VLM baseline on OpenEQA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.evaluation.scripts.run_openeqa_oneshot import (
    MockOpenEQASample,
    create_mock_openeqa_samples,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_oneshot_baseline,
    _compute_confidence_distribution,
    _compute_status_distribution,
)


class TestMockOpenEQASample:
    """Tests for MockOpenEQASample dataclass."""

    def test_sample_creation(self):
        """Test creating a mock sample."""
        sample = MockOpenEQASample(
            question_id="test_001",
            question="What color is the chair?",
            answer="blue",
            category="attribute_recognition",
            scene_id="scene_001",
        )
        assert sample.question_id == "test_001"
        assert sample.question == "What color is the chair?"
        assert sample.answer == "blue"
        assert sample.category == "attribute_recognition"
        assert sample.scene_id == "scene_001"
        assert sample.question_type == "episodic_memory"  # default

    def test_sample_with_custom_question_type(self):
        """Test creating sample with custom question type."""
        sample = MockOpenEQASample(
            question_id="test_002",
            question="Navigate to the kitchen",
            answer="turn left",
            category="navigation",
            scene_id="scene_002",
            question_type="active_exploration",
        )
        assert sample.question_type == "active_exploration"


class TestCreateMockSamples:
    """Tests for mock sample generation."""

    def test_default_sample_count(self):
        """Test generating default number of samples."""
        samples = create_mock_openeqa_samples()
        assert len(samples) == 50

    def test_custom_sample_count(self):
        """Test generating custom number of samples."""
        samples = create_mock_openeqa_samples(n_samples=10)
        assert len(samples) == 10

    def test_unique_question_ids(self):
        """Test that all question IDs are unique."""
        samples = create_mock_openeqa_samples(n_samples=100)
        ids = [s.question_id for s in samples]
        assert len(ids) == len(set(ids))

    def test_diverse_categories(self):
        """Test that samples cover multiple categories."""
        samples = create_mock_openeqa_samples(n_samples=50)
        categories = set(s.category for s in samples)
        assert len(categories) > 3

    def test_diverse_scenes(self):
        """Test that samples cover multiple scenes."""
        samples = create_mock_openeqa_samples(n_samples=50)
        scenes = set(s.scene_id for s in samples)
        assert len(scenes) >= 3


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_returns_selector(self):
        """Test that factory returns a mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_results(self):
        """Test that selector returns valid results."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")
        result = selector.select_keyframes_v2("pillow on sofa", k=3)

        assert len(result.keyframe_paths) == 3
        assert all(isinstance(p, Path) for p in result.keyframe_paths)
        assert "selected_hypothesis_kind" in result.metadata

    def test_selector_varies_hypothesis_kinds(self):
        """Test that hypothesis kinds vary across calls."""
        factory = create_mock_stage1_factory()
        selector = factory("scene_001")

        kinds = set()
        for i in range(10):
            result = selector.select_keyframes_v2(f"query {i}")
            kinds.add(result.metadata["selected_hypothesis_kind"])

        assert len(kinds) > 1


class TestMockStage2Factory:
    """Tests for mock Stage 2 factory."""

    def test_factory_returns_agent(self):
        """Test that factory returns a mock agent."""
        factory = create_mock_stage2_factory()
        agent = factory()
        assert hasattr(agent, "run")

    def test_agent_returns_structured_response(self):
        """Test that agent returns valid structured response."""
        from conceptgraph.agents.models import (
            Stage2TaskSpec,
            Stage2TaskType,
            Stage2EvidenceBundle,
        )

        factory = create_mock_stage2_factory()
        agent = factory()

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="What color is the chair?",
        )
        bundle = Stage2EvidenceBundle(
            keyframes=[],
            hypothesis_kind="direct",
            metadata={},
        )

        result = agent.run(task, bundle)
        assert result.result.summary is not None
        assert result.result.confidence is not None
        assert result.result.status is not None

    def test_agent_varies_confidence(self):
        """Test that confidence varies across calls."""
        from conceptgraph.agents.models import (
            Stage2TaskSpec,
            Stage2TaskType,
            Stage2EvidenceBundle,
        )

        factory = create_mock_stage2_factory()
        agent = factory()

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="What color is the chair?",
        )
        bundle = Stage2EvidenceBundle(
            keyframes=[],
            hypothesis_kind="direct",
            metadata={},
        )

        confidences = set()
        for i in range(8):
            result = agent.run(task, bundle)
            confidences.add(round(result.result.confidence, 1))

        # Should have some variety
        assert len(confidences) >= 2

    def test_agent_models_oneshot_weakness(self):
        """Test that mock agent models one-shot weaknesses."""
        from conceptgraph.agents.models import (
            Stage2TaskSpec,
            Stage2TaskType,
            Stage2EvidenceBundle,
            Stage2Status,
        )

        factory = create_mock_stage2_factory()
        agent = factory()

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="What color is the chair?",
        )
        bundle = Stage2EvidenceBundle(
            keyframes=[],
            hypothesis_kind="direct",
            metadata={},
        )

        # One-shot should have some insufficient evidence cases
        has_low_conf = False
        for i in range(10):
            result = agent.run(task, bundle)
            if result.result.confidence < 0.6:
                has_low_conf = True

        assert has_low_conf


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_confidence_distribution(self):
        """Test confidence distribution computation."""
        mock_result = MagicMock()

        results = []
        for conf in [0.2, 0.4, 0.6, 0.8, 0.95]:
            r = MagicMock()
            r.stage2_success = True
            r.stage2_confidence = conf
            results.append(r)

        mock_result.results = results

        dist = _compute_confidence_distribution(mock_result)
        assert dist["very_low_0_30"] == 1  # 0.2
        assert dist["low_30_50"] == 1  # 0.4
        assert dist["medium_50_70"] == 1  # 0.6
        assert dist["high_70_90"] == 1  # 0.8
        assert dist["very_high_90_100"] == 1  # 0.95

    def test_compute_status_distribution(self):
        """Test status distribution computation."""
        mock_result = MagicMock()

        results = []
        for status in [
            "high_confidence",
            "high_confidence",
            "low_confidence",
            "needs_evidence",
        ]:
            r = MagicMock()
            r.stage2_success = True
            r.stage2_status = status
            results.append(r)

        mock_result.results = results

        dist = _compute_status_distribution(mock_result)
        assert dist["high_confidence"] == 2
        assert dist["low_confidence"] == 1
        assert dist["needs_evidence"] == 1


class TestRunOneshotBaseline:
    """Integration tests for the main baseline function."""

    @patch("conceptgraph.evaluation.scripts.run_openeqa_oneshot.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_oneshot.adapt_openeqa_samples")
    def test_mock_mode_runs_successfully(
        self, mock_adapt, mock_evaluator_class, tmp_path
    ):
        """Test that mock mode runs without errors."""
        # Setup mocks
        mock_adapt.return_value = [MagicMock(scene_id="scene_001")]

        mock_run_result = MagicMock()
        mock_run_result.total_samples = 1
        mock_run_result.successful_samples = 1
        mock_run_result.failed_stage1 = 0
        mock_run_result.failed_stage2 = 0
        mock_run_result.avg_stage1_latency_ms = 100.0
        mock_run_result.avg_stage2_latency_ms = 500.0
        mock_run_result.avg_stage2_confidence = 0.75
        mock_run_result.avg_tool_calls_per_sample = 0.0
        mock_run_result.samples_with_tool_use = 0
        mock_run_result.samples_with_insufficient_evidence = 0
        mock_run_result.tool_usage_distribution = {}
        mock_run_result.total_duration_seconds = 1.0
        mock_run_result.results = []

        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = mock_run_result
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "output.json"

        result = run_oneshot_baseline(
            use_mock=True,
            max_samples=5,
            output_path=output_path,
            verbose=False,
        )

        assert result == mock_run_result
        assert output_path.exists()

    @patch("conceptgraph.evaluation.scripts.run_openeqa_oneshot.BatchEvaluator")
    @patch("conceptgraph.evaluation.scripts.run_openeqa_oneshot.adapt_openeqa_samples")
    def test_output_json_format(self, mock_adapt, mock_evaluator_class, tmp_path):
        """Test that output JSON has correct format."""
        mock_adapt.return_value = [MagicMock(scene_id="scene_001")]

        mock_run_result = MagicMock()
        mock_run_result.total_samples = 1
        mock_run_result.successful_samples = 1
        mock_run_result.failed_stage1 = 0
        mock_run_result.failed_stage2 = 0
        mock_run_result.avg_stage1_latency_ms = 100.0
        mock_run_result.avg_stage2_latency_ms = 500.0
        mock_run_result.avg_stage2_confidence = 0.75
        mock_run_result.avg_tool_calls_per_sample = 0.0
        mock_run_result.samples_with_tool_use = 0
        mock_run_result.samples_with_insufficient_evidence = 0
        mock_run_result.tool_usage_distribution = {}
        mock_run_result.total_duration_seconds = 1.0
        mock_run_result.results = []

        mock_evaluator = MagicMock()
        mock_evaluator.run.return_value = mock_run_result
        mock_evaluator_class.return_value = mock_evaluator

        output_path = tmp_path / "output.json"

        run_oneshot_baseline(
            use_mock=True,
            max_samples=5,
            output_path=output_path,
            verbose=False,
        )

        with open(output_path) as f:
            data = json.load(f)

        # Check required fields
        assert data["experiment"] == "oneshot_vlm_baseline"
        assert data["benchmark"] == "openeqa"
        assert "config" in data
        assert data["config"]["stage2_enabled"] is True
        assert data["config"]["stage2_max_turns"] == 1
        assert "academic_notes" in data

    def test_requires_data_root_without_mock(self):
        """Test that data_root is required when not using mock."""
        with pytest.raises(ValueError, match="data_root is required"):
            run_oneshot_baseline(use_mock=False, verbose=False)


class TestCLIInterface:
    """Tests for command-line interface."""

    @patch("conceptgraph.evaluation.scripts.run_openeqa_oneshot.run_oneshot_baseline")
    def test_mock_flag(self, mock_run):
        """Test --mock flag."""
        mock_run.return_value = MagicMock()

        from conceptgraph.evaluation.scripts.run_openeqa_oneshot import main

        with patch.object(sys, "argv", ["prog", "--mock", "--max_samples", "5"]):
            with patch.object(sys, "exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(0)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["use_mock"] is True
        assert call_kwargs["max_samples"] == 5

    def test_requires_data_root_or_mock(self):
        """Test that CLI requires --data_root or --mock."""
        from conceptgraph.evaluation.scripts.run_openeqa_oneshot import main

        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
