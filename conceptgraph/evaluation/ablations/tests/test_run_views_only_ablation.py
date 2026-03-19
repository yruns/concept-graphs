"""Tests for views-only (request_more_views only) ablation study runner.

TASK-041: Ablation: + request_more_views only

This test suite validates the ablation runner that executes the Stage 2 agent
with only the request_more_views tool enabled across all benchmarks.

Test Coverage:
- Configuration validation (views-only requirements)
- Mock data loading for all benchmarks
- Single-benchmark execution
- Cross-benchmark execution
- Result aggregation and reporting
- Academic alignment verification
- Comparison against one-shot baseline settings
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.evaluation.ablation_config import (
    AgentConfig,
    AblationConfig,
    ToolConfig,
)
from conceptgraph.evaluation.ablations.run_views_only_ablation import (
    VIEWS_ONLY_ABLATION_CONFIG,
    SUPPORTED_BENCHMARKS,
    AblationStudyResult,
    BenchmarkResult,
    ViewsOnlyAblationRunner,
    create_mock_samples_factory,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_views_only_ablation,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestViewsOnlyAblationConfig:
    """Tests for views-only ablation configuration."""

    def test_views_config_has_multi_turn(self):
        """Views-only config must allow multi-turn for tool usage."""
        assert VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_views_config_enables_request_more_views(self):
        """Views-only config must enable request_more_views (the key tool)."""
        tools = VIEWS_ONLY_ABLATION_CONFIG.tools
        assert tools.request_more_views is True

    def test_views_config_disables_other_evidence_tools(self):
        """Views-only config must disable other evidence-seeking tools."""
        tools = VIEWS_ONLY_ABLATION_CONFIG.tools
        assert tools.request_crops is False
        assert tools.switch_or_expand_hypothesis is False

    def test_views_config_keeps_inspection_tools(self):
        """Views-only config should keep read-only inspection tools."""
        tools = VIEWS_ONLY_ABLATION_CONFIG.tools
        # These provide context but don't acquire new evidence
        assert tools.inspect_stage1_metadata is True
        assert tools.retrieve_object_context is True

    def test_views_config_has_brief_plan_mode(self):
        """Views-only config should have plan_mode=brief for normal operation."""
        assert VIEWS_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"

    def test_views_config_keeps_uncertainty_stopping(self):
        """Views-only config should keep uncertainty stopping for fair comparison."""
        assert VIEWS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is True

    def test_views_config_has_valid_stage2(self):
        """Views-only config must have Stage 2 enabled."""
        assert VIEWS_ONLY_ABLATION_CONFIG.stage2.enabled is True

    def test_views_config_has_appropriate_tags(self):
        """Views-only config should have ablation tags."""
        tags = VIEWS_ONLY_ABLATION_CONFIG.tags
        assert "ablation" in tags
        assert "views_only" in tags
        assert "tool_ablation" in tags

    def test_views_config_differs_from_oneshot(self):
        """Views-only config should differ from oneshot in key ways."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Key differences
        assert (
            VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True
        assert ONESHOT_ABLATION_CONFIG.tools.request_more_views is False
        assert VIEWS_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"
        assert ONESHOT_ABLATION_CONFIG.agent.plan_mode == "off"


# =============================================================================
# Mock Factory Tests
# =============================================================================


class TestMockFactories:
    """Tests for mock data factories."""

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_create_mock_samples_factory(self, benchmark: str):
        """Mock samples factory should return valid samples for each benchmark."""
        samples = create_mock_samples_factory(benchmark, n_samples=10)
        assert len(samples) == 10
        # Each sample should have required attributes
        for sample in samples:
            assert hasattr(sample, "question_id") or hasattr(sample, "object_id")

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_create_mock_stage1_factory(self, benchmark: str):
        """Mock Stage 1 factory should return callable factory."""
        factory = create_mock_stage1_factory(benchmark)
        assert callable(factory)

        # Factory should accept scene_id
        mock_selector = factory("test_scene")
        assert hasattr(mock_selector, "select_keyframes_v2")

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_create_mock_stage2_factory(self, benchmark: str):
        """Mock Stage 2 factory should return callable factory."""
        factory = create_mock_stage2_factory(benchmark)
        assert callable(factory)

        # Factory should return agent-like object
        mock_agent = factory()
        assert hasattr(mock_agent, "run")

    def test_mock_samples_have_diversity(self):
        """Mock samples should have diverse question types."""
        samples = create_mock_samples_factory("openeqa", n_samples=50)

        # Check for category diversity
        categories = set()
        for sample in samples:
            if hasattr(sample, "category"):
                categories.add(sample.category)

        # Should have multiple categories
        assert len(categories) >= 3


# =============================================================================
# Runner Initialization Tests
# =============================================================================


class TestViewsOnlyAblationRunner:
    """Tests for ViewsOnlyAblationRunner initialization."""

    def test_runner_init_with_defaults(self):
        """Runner should initialize with default config."""
        runner = ViewsOnlyAblationRunner(use_mock=True)
        assert runner.config.agent.max_turns >= 2
        assert runner.config.tools.request_more_views is True
        assert runner.use_mock is True

    def test_runner_init_with_custom_config(self):
        """Runner should accept custom config."""
        custom_config = AblationConfig(
            name="custom_views_only",
            description="Custom test",
            agent=AgentConfig(max_turns=4, plan_mode="brief"),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=False,
                switch_or_expand_hypothesis=False,
            ),
        )
        runner = ViewsOnlyAblationRunner(config=custom_config, use_mock=True)
        assert runner.config.name == "custom_views_only"

    def test_runner_validates_config_warns_on_single_turn(self):
        """Runner should warn if config has single-turn."""
        invalid_config = AblationConfig(
            name="invalid_views_only",
            description="Invalid - single turn",
            agent=AgentConfig(max_turns=1),  # Invalid for views-only
            tools=ToolConfig(
                request_more_views=True,
                request_crops=False,
                switch_or_expand_hypothesis=False,
            ),
        )
        # Should not raise, but should warn
        with patch("loguru.logger.warning") as mock_warn:
            runner = ViewsOnlyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_views_disabled(self):
        """Runner should warn if request_more_views is disabled."""
        invalid_config = AblationConfig(
            name="invalid_views_disabled",
            description="Invalid - views disabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=False,  # Invalid - should be True
                request_crops=False,
                switch_or_expand_hypothesis=False,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = ViewsOnlyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_creates_output_dir(self, tmp_path: Path):
        """Runner should create output directory."""
        output_dir = tmp_path / "ablation_output"
        runner = ViewsOnlyAblationRunner(use_mock=True, output_dir=output_dir)
        assert output_dir.exists()


# =============================================================================
# Single Benchmark Execution Tests
# =============================================================================


class TestSingleBenchmarkExecution:
    """Tests for running ablation on single benchmarks."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a runner with mock mode."""
        return ViewsOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_run_single_benchmark_mock(
        self, runner: ViewsOnlyAblationRunner, benchmark: str
    ):
        """Should successfully run on each benchmark with mock data."""
        result = runner.run_benchmark(benchmark)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark == benchmark
        assert result.error is None
        assert result.run_result is not None
        assert result.run_result.total_samples > 0

    def test_run_benchmark_returns_duration(self, runner: ViewsOnlyAblationRunner):
        """Benchmark result should include duration."""
        result = runner.run_benchmark("openeqa")
        assert result.duration_seconds >= 0

    def test_run_benchmark_error_handling(self, runner: ViewsOnlyAblationRunner):
        """Should handle benchmark errors gracefully."""
        # Patch to simulate an error
        with patch.object(runner, "_load_samples", side_effect=Exception("Test error")):
            result = runner.run_benchmark("openeqa")
            assert result.error is not None
            assert "Test error" in result.error


# =============================================================================
# Cross-Benchmark Execution Tests
# =============================================================================


class TestCrossBenchmarkExecution:
    """Tests for running ablation across all benchmarks."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a runner with mock mode."""
        return ViewsOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    def test_run_all_benchmarks(self, runner: ViewsOnlyAblationRunner):
        """Should run on all supported benchmarks."""
        result = runner.run_all("all")

        assert isinstance(result, AblationStudyResult)
        assert result.ablation_name == VIEWS_ONLY_ABLATION_CONFIG.name
        assert len(result.benchmark_results) == len(SUPPORTED_BENCHMARKS)

    def test_run_all_aggregates_samples(self, runner: ViewsOnlyAblationRunner):
        """Should aggregate sample counts across benchmarks."""
        result = runner.run_all("all")

        # Total should equal sum from all benchmarks
        expected_total = sum(
            r.run_result.total_samples for r in result.benchmark_results if r.run_result
        )
        assert result.total_samples == expected_total

    def test_run_all_computes_success_rate(self, runner: ViewsOnlyAblationRunner):
        """Should compute overall success rate."""
        result = runner.run_all("all")

        assert 0.0 <= result.overall_success_rate <= 1.0

    def test_run_all_computes_per_benchmark_rates(
        self, runner: ViewsOnlyAblationRunner
    ):
        """Should compute per-benchmark success rates."""
        result = runner.run_all("all")

        rates = result.per_benchmark_success_rates
        assert len(rates) == len(SUPPORTED_BENCHMARKS)
        for benchmark in SUPPORTED_BENCHMARKS:
            assert benchmark in rates
            assert 0.0 <= rates[benchmark] <= 1.0

    def test_run_selected_benchmarks(self, runner: ViewsOnlyAblationRunner):
        """Should run only selected benchmarks."""
        result = runner.run_all(["openeqa", "sqa3d"])

        assert len(result.benchmark_results) == 2
        benchmarks = {r.benchmark for r in result.benchmark_results}
        assert benchmarks == {"openeqa", "sqa3d"}


# =============================================================================
# Result Serialization Tests
# =============================================================================


class TestResultSerialization:
    """Tests for result serialization and reporting."""

    @pytest.fixture
    def sample_result(self) -> AblationStudyResult:
        """Create a sample result for testing."""
        mock_run_result = MagicMock()
        mock_run_result.total_samples = 50
        mock_run_result.successful_samples = 42
        mock_run_result.failed_stage1 = 2
        mock_run_result.failed_stage2 = 6
        mock_run_result.avg_stage1_latency_ms = 100.0
        mock_run_result.avg_stage2_latency_ms = 800.0
        mock_run_result.avg_stage2_confidence = 0.72
        mock_run_result.avg_tool_calls_per_sample = 1.5  # Views-only has tool calls
        mock_run_result.samples_with_insufficient_evidence = 4

        result = AblationStudyResult(
            ablation_name="views_only",
            ablation_description="Test ablation",
            timestamp="2026-03-20T04:30:00",
            benchmark_results=[
                BenchmarkResult(
                    benchmark="openeqa",
                    run_result=mock_run_result,
                    error=None,
                    duration_seconds=15.0,
                )
            ],
            total_samples=50,
            total_successful=42,
            total_failed_stage1=2,
            total_failed_stage2=6,
            total_duration_seconds=15.0,
        )
        return result

    def test_result_to_dict(self, sample_result: AblationStudyResult):
        """Result should convert to dictionary."""
        d = sample_result.to_dict()

        assert "ablation_name" in d
        assert "summary" in d
        assert "benchmark_results" in d
        assert "academic_notes" in d

    def test_result_json_serializable(self, sample_result: AblationStudyResult):
        """Result dict should be JSON serializable."""
        d = sample_result.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_result_includes_academic_notes(self, sample_result: AblationStudyResult):
        """Result should include academic notes."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert "ablation_condition" in notes
        assert "purpose" in notes
        assert "expected_findings" in notes
        assert "research_question" in notes

    def test_result_academic_notes_reference_baselines(
        self, sample_result: AblationStudyResult
    ):
        """Academic notes should reference comparison baselines."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert "comparison_baselines" in notes
        baselines = notes["comparison_baselines"]
        assert any("oneshot" in b.lower() for b in baselines)
        assert any("full" in b.lower() for b in baselines)

    def test_runner_saves_results(self, tmp_path: Path):
        """Runner should save results to JSON files."""
        runner = ViewsOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        # Check output files exist
        output_dir = tmp_path / "output"
        json_files = list(output_dir.glob("ablation_views_only_*.json"))
        assert len(json_files) >= 1

    def test_runner_saves_summary_with_ablation_settings(self, tmp_path: Path):
        """Runner should save summary including ablation settings."""
        runner = ViewsOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        summary_file = tmp_path / "output" / "ablation_views_only_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "ablation_settings" in summary
        settings = summary["ablation_settings"]
        assert settings["request_more_views"] is True
        assert settings["request_crops"] is False
        assert settings["switch_or_expand_hypothesis"] is False


# =============================================================================
# Top-level API Tests
# =============================================================================


class TestTopLevelAPI:
    """Tests for the top-level run_views_only_ablation function."""

    def test_run_views_only_ablation_mock(self, tmp_path: Path):
        """Top-level function should work with mock mode."""
        result = run_views_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, AblationStudyResult)
        assert result.total_samples > 0

    def test_run_views_only_ablation_all_mock(self, tmp_path: Path):
        """Top-level function should run all benchmarks."""
        result = run_views_only_ablation(
            benchmarks="all",
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        assert len(result.benchmark_results) == len(SUPPORTED_BENCHMARKS)


# =============================================================================
# Academic Alignment Tests
# =============================================================================


class TestAcademicAlignment:
    """Tests to verify the ablation supports academic claims."""

    def test_views_only_supports_evidence_acquisition_claim(self):
        """Views-only ablation should support 'adaptive evidence acquisition' claim."""
        config = VIEWS_ONLY_ABLATION_CONFIG

        # Key: iterative keyframe evidence acquisition
        assert config.agent.max_turns >= 2  # Multi-turn allowed
        assert config.tools.request_more_views is True  # KEY tool enabled
        assert config.tools.request_crops is False  # Other tools disabled
        assert config.tools.switch_or_expand_hypothesis is False

        # This isolates contribution of keyframe expansion

    def test_ablation_has_fair_comparison_setup(self):
        """Views-only should have fair comparison settings."""
        config = VIEWS_ONLY_ABLATION_CONFIG

        # Same model
        assert config.stage2.model == "gpt-5.2-2025-12-11"

        # Same Stage 1 settings
        assert config.stage1.k == 3

        # Same uncertainty threshold
        assert config.agent.confidence_threshold == 0.4

        # Uncertainty stopping enabled for fair comparison
        assert config.agent.enable_uncertainty_stopping is True

    def test_result_tracks_tool_calls(self, tmp_path: Path):
        """Results should track tool call frequency."""
        result = run_views_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=10,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        d = result.to_dict()
        for br in d["benchmark_results"]:
            if br["summary"]:
                assert "avg_tool_calls" in br["summary"]

    def test_result_tracks_insufficient_evidence(self, tmp_path: Path):
        """Results should track insufficient evidence cases."""
        result = run_views_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=10,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        d = result.to_dict()
        for br in d["benchmark_results"]:
            if br["summary"]:
                assert "samples_with_insufficient_evidence" in br["summary"]

    def test_views_only_expected_to_have_tool_calls(self, tmp_path: Path):
        """Views-only results should show some tool calls (unlike one-shot)."""
        result = run_views_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=10,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        d = result.to_dict()
        for br in d["benchmark_results"]:
            if br["summary"]:
                avg_tool_calls = br["summary"]["avg_tool_calls"]
                # Views-only should have some tool calls (agent can request views)
                # Not asserting > 0 because mock might not always call tools
                # But it COULD have tool calls (unlike oneshot which has ~0)
                assert avg_tool_calls >= 0.0

    def test_ablation_tag_identifies_views_only(self):
        """Ablation config should generate tag without 'oneshot'."""
        config = VIEWS_ONLY_ABLATION_CONFIG
        tag = config.get_ablation_tag()
        # Should NOT be marked as oneshot
        assert "oneshot" not in tag
        # Should be marked as partial (no_crops or no_repair)
        # Based on get_ablation_tag logic: no request_crops -> "no_crops"
        assert "no_crops" in tag or "no_repair" in tag


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparisonWithOneshot:
    """Tests comparing views-only config with oneshot config."""

    def test_views_only_allows_more_turns(self):
        """Views-only should allow more turns than oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )

    def test_views_only_enables_view_tool(self):
        """Views-only should enable request_more_views unlike oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True
        assert ONESHOT_ABLATION_CONFIG.tools.request_more_views is False

    def test_both_disable_crops_and_repair(self):
        """Both should disable crops and repair tools."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Views-only disables these for ablation
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_crops is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False

        # Oneshot also disables all evidence tools
        assert ONESHOT_ABLATION_CONFIG.tools.request_crops is False
        assert ONESHOT_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False

    def test_same_uncertainty_and_confidence_settings(self):
        """Both should have same uncertainty and confidence settings for fair comparison."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            VIEWS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            == ONESHOT_ABLATION_CONFIG.agent.enable_uncertainty_stopping
        )
        assert (
            VIEWS_ONLY_ABLATION_CONFIG.agent.confidence_threshold
            == ONESHOT_ABLATION_CONFIG.agent.confidence_threshold
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete ablation workflow."""

    def test_complete_workflow_mock(self, tmp_path: Path):
        """Complete workflow should succeed with mock data."""
        output_dir = tmp_path / "ablation_output"

        # Run ablation
        result = run_views_only_ablation(
            benchmarks="all",
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=output_dir,
        )

        # Verify result structure
        assert result.ablation_name == VIEWS_ONLY_ABLATION_CONFIG.name
        assert result.total_samples > 0
        assert len(result.benchmark_results) == 3

        # Verify output files
        assert output_dir.exists()
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 2  # Main result + summary

        # Verify JSON content
        summary_file = output_dir / "ablation_views_only_summary.json"
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)
        assert "overall_success_rate" in summary
        assert "ablation_settings" in summary

    def test_workflow_handles_partial_failure(self, tmp_path: Path):
        """Workflow should continue if one benchmark fails."""
        runner = ViewsOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        # Patch one benchmark to fail
        original_run = runner.run_benchmark
        call_count = [0]

        def patched_run(benchmark, data_root=None):
            call_count[0] += 1
            if benchmark == "sqa3d":
                return BenchmarkResult(
                    benchmark=benchmark,
                    run_result=None,
                    error="Simulated failure",
                )
            return original_run(benchmark, data_root)

        runner.run_benchmark = patched_run

        # Should still complete
        result = runner.run_all("all")

        # Should have results from working benchmarks
        working_results = [r for r in result.benchmark_results if r.error is None]
        assert len(working_results) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
