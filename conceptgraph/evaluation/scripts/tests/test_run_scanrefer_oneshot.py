"""Tests for run_scanrefer_oneshot script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.benchmarks.scanrefer_loader import BoundingBox3D
from conceptgraph.evaluation.scripts.run_scanrefer_oneshot import (
    MockScanReferSample,
    create_mock_scanrefer_samples,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_scanrefer_oneshot,
    _compute_hypothesis_distribution,
    _compute_grounding_stats,
)
from conceptgraph.evaluation.batch_eval import EvalRunResult, EvalSampleResult


class TestMockScanReferSamples:
    """Tests for mock sample generation."""

    def test_create_mock_samples(self):
        """Test mock sample creation."""
        samples = create_mock_scanrefer_samples(10)
        assert len(samples) == 10
        assert all(isinstance(s, MockScanReferSample) for s in samples)

    def test_mock_sample_bounding_box(self):
        """Test mock samples have valid 3D bounding boxes."""
        samples = create_mock_scanrefer_samples(5)
        for sample in samples:
            bbox = sample.target_bbox
            assert isinstance(bbox, BoundingBox3D)
            assert len(bbox.center) == 3
            assert len(bbox.size) == 3
            assert all(s > 0 for s in bbox.size)


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
        result = selector.select_keyframes_v2("the chair next to the table", k=3)
        assert len(result.keyframe_paths) == 3


class TestMockStage2Factory:
    """Tests for mock Stage 2 factory (one-shot mode)."""

    def _create_mock_task_and_bundle(self):
        """Helper to create mock task and bundle for testing."""
        from conceptgraph.agents.models import (
            KeyframeEvidence,
            Stage2EvidenceBundle,
            Stage2PlanMode,
            Stage2TaskSpec,
            Stage2TaskType,
        )

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query="the chair next to the table",
            plan_mode=Stage2PlanMode.OFF,
            max_reasoning_turns=1,
        )
        bundle = Stage2EvidenceBundle(
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/mock/frame0.jpg"),
                KeyframeEvidence(keyframe_idx=1, image_path="/mock/frame1.jpg"),
            ],
            scene_id="scene0000_00",
            stage1_query="the chair next to the table",
        )
        return task, bundle

    def test_factory_creates_agent(self):
        """Test factory creates mock agent."""
        factory = create_mock_stage2_factory()
        agent = factory()
        assert agent is not None
        assert hasattr(agent, "run")

    def test_agent_returns_response(self):
        """Test agent returns structured response."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        assert result.result.status is not None
        assert result.result.summary is not None
        assert result.result.confidence is not None

    def test_oneshot_no_tool_calls(self):
        """Test one-shot agent makes zero tool calls."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        # One-shot baseline should not use tools
        assert len(result.tool_trace) == 0

    def test_response_includes_bbox_prediction(self):
        """Test response includes bounding box prediction."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        payload = result.result.payload
        assert "predicted_bbox" in payload
        assert "center" in payload["predicted_bbox"]
        assert "size" in payload["predicted_bbox"]

    def test_response_confidence_range(self):
        """Test response confidence is in valid range."""
        factory = create_mock_stage2_factory()

        for i in range(10):
            agent = factory()
            task, bundle = self._create_mock_task_and_bundle()
            result = agent.run(task, bundle)
            assert 0.0 <= result.result.confidence <= 1.0


class TestRunScanReferOneshot:
    """Tests for the main one-shot evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_scanrefer_oneshot(
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
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "experiment" in data
            assert data["experiment"] == "oneshot_vlm_baseline"
            assert "benchmark" in data
            assert data["benchmark"] == "scanrefer"
            assert "task_type" in data
            assert data["task_type"] == "visual_grounding"
            assert "summary" in data
            assert "grounding_metrics" in data
            assert "per_sample_results" in data

    def test_stage2_enabled_single_turn(self):
        """Test Stage 2 is enabled but limited to single turn."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            config = data["config"]
            assert config["stage2_enabled"] is True
            assert config["stage2_max_turns"] == 1
            assert config["tools_enabled"] == []

    def test_summary_includes_both_stages(self):
        """Test summary includes both Stage 1 and Stage 2 metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=5,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            summary = data["summary"]
            assert "stage1_success" in summary
            assert "stage2_success" in summary
            assert "avg_stage1_latency_ms" in summary
            assert "avg_stage2_latency_ms" in summary


class TestGroundingStats:
    """Tests for grounding statistics computation."""

    def test_compute_grounding_stats(self):
        """Test grounding stats computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_answer="Located object at center [1.0, 2.0, 0.8]",
                stage2_confidence=0.85,
            )
            for i in range(5)
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        stats = _compute_grounding_stats(run_result)
        assert "predictions_made" in stats
        assert stats["predictions_made"] == 5  # All have 'center' in answer


class TestAcademicAlignment:
    """Tests to verify academic alignment of one-shot baseline."""

    def test_academic_notes_present(self):
        """Test academic notes are included."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "one-shot" in notes["purpose"].lower() or "oneshot" in notes["purpose"].lower()

    def test_comparison_targets_specified(self):
        """Test comparison targets are specified for context."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            # Should reference both lower (stage1_only) and upper (stage2_full) bounds
            assert "comparison_target_lower" in notes or "stage1" in str(notes).lower()
            assert "comparison_target_upper" in notes or "stage2" in str(notes).lower()

    def test_per_sample_stage2_fields(self):
        """Test per-sample results have Stage 2 fields for comparison."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_oneshot(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            for sample in data["per_sample_results"]:
                assert "stage2_success" in sample
                assert "stage2_answer" in sample
                assert "stage2_confidence" in sample
                assert "stage2_tool_calls" in sample
                # One-shot should have 0 tool calls
                assert sample["stage2_tool_calls"] == 0
