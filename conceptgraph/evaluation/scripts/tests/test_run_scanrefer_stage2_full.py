"""Tests for run_scanrefer_stage2_full script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.benchmarks.scanrefer_loader import BoundingBox3D
from conceptgraph.evaluation.scripts.run_scanrefer_stage2_full import (
    MockScanReferSample,
    create_mock_scanrefer_samples,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_scanrefer_stage2_full,
    _compute_hypothesis_distribution,
    _compute_grounding_stats,
    _compute_tool_usage_stats,
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


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_includes_tool_support_metadata(self):
        """Test selector metadata includes tool support info."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        result = selector.select_keyframes_v2("the chair", k=3)
        metadata = result.metadata

        # Full Stage 2 needs additional metadata for tools
        assert "available_views" in metadata
        assert "scene_objects" in metadata


class TestMockStage2Factory:
    """Tests for mock Stage 2 factory (full agent mode)."""

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
            plan_mode=Stage2PlanMode.BRIEF,
            max_reasoning_turns=5,
        )
        bundle = Stage2EvidenceBundle(
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/mock/frame0.jpg"),
                KeyframeEvidence(keyframe_idx=1, image_path="/mock/frame1.jpg"),
                KeyframeEvidence(keyframe_idx=2, image_path="/mock/frame2.jpg"),
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

    def test_agent_uses_tools(self):
        """Test full agent uses tools (unlike one-shot)."""
        factory = create_mock_stage2_factory()

        # Run multiple times to check tool usage
        tool_call_counts = []
        for _ in range(10):
            agent = factory()
            task, bundle = self._create_mock_task_and_bundle()
            result = agent.run(task, bundle)
            tool_call_counts.append(len(result.tool_trace))

        # Should have some tool calls (unlike one-shot which has 0)
        assert any(t > 0 for t in tool_call_counts), "Full agent should use tools"

    def test_response_includes_tools_used(self):
        """Test response includes which tools were used."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        payload = result.result.payload
        assert "tools_used" in payload
        assert isinstance(payload["tools_used"], list)

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

    def test_multi_turn_reasoning(self):
        """Test agent produces multi-turn reasoning (tool usage indicates turns)."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        # Multi-turn should have tool trace entries
        assert len(result.tool_trace) >= 1

    def test_iterative_evidence_seeking_method(self):
        """Test grounding method is iterative evidence seeking."""
        factory = create_mock_stage2_factory()
        agent = factory()
        task, bundle = self._create_mock_task_and_bundle()
        result = agent.run(task, bundle)

        payload = result.result.payload
        assert payload["grounding_method"] == "iterative_evidence_seeking"


class TestRunScanReferStage2Full:
    """Tests for the main full Stage 2 evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_scanrefer_stage2_full(
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
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert "experiment" in data
            assert data["experiment"] == "stage2_full_pipeline"
            assert "benchmark" in data
            assert data["benchmark"] == "scanrefer"
            assert "task_type" in data
            assert data["task_type"] == "visual_grounding"
            assert "summary" in data
            assert "grounding_metrics" in data
            assert "tool_usage" in data
            assert "per_sample_results" in data

    def test_all_tools_enabled(self):
        """Test all Stage 2 tools are enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            config = data["config"]
            assert config["stage2_enabled"] is True
            assert "request_more_views" in config["tools_enabled"]
            assert "request_crops" in config["tools_enabled"]
            assert "switch_or_expand_hypothesis" in config["tools_enabled"]

    def test_multi_turn_enabled(self):
        """Test multi-turn reasoning is enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=3,
                max_turns=5,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            config = data["config"]
            assert config["stage2_max_turns"] == 5

    def test_tool_usage_stats_present(self):
        """Test tool usage statistics are in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=5,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            tool_usage = data["tool_usage"]
            assert "avg_tool_calls" in tool_usage
            assert "samples_with_tools" in tool_usage
            assert tool_usage["avg_tool_calls"] > 0  # Full agent uses tools


class TestToolUsageStats:
    """Tests for tool usage statistics computation."""

    def test_compute_tool_usage_stats(self):
        """Test tool usage stats computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_tool_calls=i + 1,  # 1, 2, 3, 4, 5 tool calls
            )
            for i in range(5)
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        stats = _compute_tool_usage_stats(run_result)
        assert stats["avg_tool_calls"] == 3.0  # (1+2+3+4+5)/5
        assert stats["max_tool_calls"] == 5
        assert stats["min_tool_calls"] == 1
        assert stats["samples_with_tools"] == 5
        assert stats["samples_no_tools"] == 0

    def test_compute_tool_usage_stats_with_no_tools(self):
        """Test tool usage stats with some no-tool samples."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q0",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_tool_calls=0,
            ),
            EvalSampleResult(
                sample_id="s1",
                query="Q1",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage2_success=True,
                stage2_tool_calls=3,
            ),
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        stats = _compute_tool_usage_stats(run_result)
        assert stats["samples_with_tools"] == 1
        assert stats["samples_no_tools"] == 1


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
                stage2_confidence=0.7 + i * 0.05,
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
        assert stats["predictions_made"] == 5
        assert stats["avg_confidence"] > 0.7


class TestAcademicAlignment:
    """Tests to verify academic alignment of full Stage 2 pipeline."""

    def test_academic_notes_present(self):
        """Test academic notes are included."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "innovation_points" in notes

    def test_innovation_points_documented(self):
        """Test innovation points are documented."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            innovations = data["academic_notes"]["innovation_points"]
            assert len(innovations) >= 3, "Should document multiple innovations"

            # Check for key innovation themes
            innovations_text = " ".join(innovations).lower()
            assert "evidence" in innovations_text or "adaptive" in innovations_text
            assert "uncertainty" in innovations_text or "confidence" in innovations_text

    def test_comparison_targets_documented(self):
        """Test comparison targets are documented."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "comparison_targets" in notes
            targets = notes["comparison_targets"]
            assert "lower_bound_1" in targets or "stage1" in str(targets).lower()
            assert "lower_bound_2" in targets or "oneshot" in str(targets).lower()

    def test_tools_documented(self):
        """Test available tools are documented."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "tools_available" in notes
            tools = notes["tools_available"]
            assert len(tools) >= 3

    def test_per_sample_full_fields(self):
        """Test per-sample results have full Stage 2 fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
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
                assert "stage2_latency_ms" in sample


class TestCLIArguments:
    """Tests for CLI argument handling."""

    def test_max_turns_argument(self):
        """Test max_turns argument is respected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                max_turns=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["config"]["stage2_max_turns"] == 3

    def test_default_max_turns(self):
        """Test default max_turns value."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage2_full(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Default should be 5
            assert data["config"]["stage2_max_turns"] == 5
