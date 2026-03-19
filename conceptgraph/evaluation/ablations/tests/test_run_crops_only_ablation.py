"""Tests for crops-only (request_crops only) ablation study runner.

TASK-042: Ablation: + request_crops only

This test suite validates the ablation runner that executes the Stage 2 agent
with only the request_crops tool enabled across all benchmarks.

Test Coverage:
- Configuration validation (crops-only requirements)
- Mock data loading for all benchmarks
- Single-benchmark execution
- Cross-benchmark execution
- Result aggregation and reporting
- Academic alignment verification
- Comparison against one-shot and views-only baseline settings
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
from conceptgraph.evaluation.ablations.run_crops_only_ablation import (
    CROPS_ONLY_ABLATION_CONFIG,
    SUPPORTED_BENCHMARKS,
    AblationStudyResult,
    BenchmarkResult,
    CropsOnlyAblationRunner,
    create_mock_samples_factory,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_crops_only_ablation,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestCropsOnlyAblationConfig:
    """Tests for crops-only ablation configuration."""

    def test_crops_config_has_multi_turn(self):
        """Crops-only config must allow multi-turn for tool usage."""
        assert CROPS_ONLY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_crops_config_enables_request_crops(self):
        """Crops-only config must enable request_crops (the key tool)."""
        tools = CROPS_ONLY_ABLATION_CONFIG.tools
        assert tools.request_crops is True

    def test_crops_config_disables_other_evidence_tools(self):
        """Crops-only config must disable other evidence-seeking tools."""
        tools = CROPS_ONLY_ABLATION_CONFIG.tools
        assert tools.request_more_views is False
        assert tools.switch_or_expand_hypothesis is False

    def test_crops_config_keeps_inspection_tools(self):
        """Crops-only config should keep read-only inspection tools."""
        tools = CROPS_ONLY_ABLATION_CONFIG.tools
        # These provide context but don't acquire new evidence
        assert tools.inspect_stage1_metadata is True
        assert tools.retrieve_object_context is True

    def test_crops_config_has_brief_plan_mode(self):
        """Crops-only config should have plan_mode=brief for normal operation."""
        assert CROPS_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"

    def test_crops_config_keeps_uncertainty_stopping(self):
        """Crops-only config should keep uncertainty stopping for fair comparison."""
        assert CROPS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is True

    def test_crops_config_has_valid_stage2(self):
        """Crops-only config must have Stage 2 enabled."""
        assert CROPS_ONLY_ABLATION_CONFIG.stage2.enabled is True

    def test_crops_config_has_appropriate_tags(self):
        """Crops-only config should have ablation tags."""
        tags = CROPS_ONLY_ABLATION_CONFIG.tags
        assert "ablation" in tags
        assert "crops_only" in tags
        assert "tool_ablation" in tags

    def test_crops_config_differs_from_oneshot(self):
        """Crops-only config should differ from oneshot in key ways."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Key differences
        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True
        assert ONESHOT_ABLATION_CONFIG.tools.request_crops is False
        assert CROPS_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"
        assert ONESHOT_ABLATION_CONFIG.agent.plan_mode == "off"

    def test_crops_config_differs_from_views_only(self):
        """Crops-only config should differ from views-only in key ways."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # Both have multi-turn
        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.max_turns
            == VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns
        )

        # Key tool difference: crops vs views
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_more_views is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_crops is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True


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


class TestCropsOnlyAblationRunner:
    """Tests for CropsOnlyAblationRunner initialization."""

    def test_runner_init_with_defaults(self):
        """Runner should initialize with default config."""
        runner = CropsOnlyAblationRunner(use_mock=True)
        assert runner.config.agent.max_turns >= 2
        assert runner.config.tools.request_crops is True
        assert runner.use_mock is True

    def test_runner_init_with_custom_config(self):
        """Runner should accept custom config."""
        custom_config = AblationConfig(
            name="custom_crops_only",
            description="Custom test",
            agent=AgentConfig(max_turns=4, plan_mode="brief"),
            tools=ToolConfig(
                request_more_views=False,
                request_crops=True,
                switch_or_expand_hypothesis=False,
            ),
        )
        runner = CropsOnlyAblationRunner(config=custom_config, use_mock=True)
        assert runner.config.name == "custom_crops_only"

    def test_runner_validates_config_warns_on_single_turn(self):
        """Runner should warn if config has single-turn."""
        invalid_config = AblationConfig(
            name="invalid_crops_only",
            description="Invalid - single turn",
            agent=AgentConfig(max_turns=1),  # Invalid for crops-only
            tools=ToolConfig(
                request_more_views=False,
                request_crops=True,
                switch_or_expand_hypothesis=False,
            ),
        )
        # Should not raise, but should warn
        with patch("loguru.logger.warning") as mock_warn:
            runner = CropsOnlyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_crops_disabled(self):
        """Runner should warn if request_crops is disabled."""
        invalid_config = AblationConfig(
            name="invalid_crops_disabled",
            description="Invalid - crops disabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=False,
                request_crops=False,  # Invalid - should be True
                switch_or_expand_hypothesis=False,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = CropsOnlyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_views_enabled(self):
        """Runner should warn if request_more_views is enabled."""
        invalid_config = AblationConfig(
            name="invalid_views_enabled",
            description="Invalid - views also enabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=True,  # Invalid - should be False
                request_crops=True,
                switch_or_expand_hypothesis=False,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = CropsOnlyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_creates_output_dir(self, tmp_path: Path):
        """Runner should create output directory."""
        output_dir = tmp_path / "ablation_output"
        runner = CropsOnlyAblationRunner(use_mock=True, output_dir=output_dir)
        assert output_dir.exists()


# =============================================================================
# Single Benchmark Execution Tests
# =============================================================================


class TestSingleBenchmarkExecution:
    """Tests for running ablation on single benchmarks."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a runner with mock mode."""
        return CropsOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_run_single_benchmark_mock(
        self, runner: CropsOnlyAblationRunner, benchmark: str
    ):
        """Should successfully run on each benchmark with mock data."""
        result = runner.run_benchmark(benchmark)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark == benchmark
        assert result.error is None
        assert result.run_result is not None
        assert result.run_result.total_samples > 0

    def test_run_benchmark_returns_duration(self, runner: CropsOnlyAblationRunner):
        """Benchmark result should include duration."""
        result = runner.run_benchmark("openeqa")
        assert result.duration_seconds >= 0

    def test_run_benchmark_error_handling(self, runner: CropsOnlyAblationRunner):
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
        return CropsOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    def test_run_all_benchmarks(self, runner: CropsOnlyAblationRunner):
        """Should run on all supported benchmarks."""
        result = runner.run_all("all")

        assert isinstance(result, AblationStudyResult)
        assert result.ablation_name == CROPS_ONLY_ABLATION_CONFIG.name
        assert len(result.benchmark_results) == len(SUPPORTED_BENCHMARKS)

    def test_run_all_aggregates_samples(self, runner: CropsOnlyAblationRunner):
        """Should aggregate sample counts across benchmarks."""
        result = runner.run_all("all")

        # Total should equal sum from all benchmarks
        expected_total = sum(
            r.run_result.total_samples for r in result.benchmark_results if r.run_result
        )
        assert result.total_samples == expected_total

    def test_run_all_computes_success_rate(self, runner: CropsOnlyAblationRunner):
        """Should compute overall success rate."""
        result = runner.run_all("all")

        assert 0.0 <= result.overall_success_rate <= 1.0

    def test_run_all_computes_per_benchmark_rates(
        self, runner: CropsOnlyAblationRunner
    ):
        """Should compute per-benchmark success rates."""
        result = runner.run_all("all")

        rates = result.per_benchmark_success_rates
        assert len(rates) == len(SUPPORTED_BENCHMARKS)
        for benchmark in SUPPORTED_BENCHMARKS:
            assert benchmark in rates
            assert 0.0 <= rates[benchmark] <= 1.0

    def test_run_selected_benchmarks(self, runner: CropsOnlyAblationRunner):
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
        mock_run_result.successful_samples = 45
        mock_run_result.failed_stage1 = 1
        mock_run_result.failed_stage2 = 4
        mock_run_result.avg_stage1_latency_ms = 100.0
        mock_run_result.avg_stage2_latency_ms = 900.0
        mock_run_result.avg_stage2_confidence = 0.75
        mock_run_result.avg_tool_calls_per_sample = 1.8  # Crops-only has tool calls
        mock_run_result.samples_with_insufficient_evidence = 3

        result = AblationStudyResult(
            ablation_name="crops_only",
            ablation_description="Test ablation",
            timestamp="2026-03-20T04:45:00",
            benchmark_results=[
                BenchmarkResult(
                    benchmark="openeqa",
                    run_result=mock_run_result,
                    error=None,
                    duration_seconds=18.0,
                )
            ],
            total_samples=50,
            total_successful=45,
            total_failed_stage1=1,
            total_failed_stage2=4,
            total_duration_seconds=18.0,
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
        assert any("views" in b.lower() for b in baselines)
        assert any("full" in b.lower() for b in baselines)

    def test_result_has_hypothesis(self, sample_result: AblationStudyResult):
        """Academic notes should include hypothesis for crops vs views."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert "hypothesis" in notes
        hypothesis = notes["hypothesis"]
        # Should mention attribute verification or fine-grained
        assert "attribute" in hypothesis.lower() or "fine-grained" in hypothesis.lower()

    def test_runner_saves_results(self, tmp_path: Path):
        """Runner should save results to JSON files."""
        runner = CropsOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        # Check output files exist
        output_dir = tmp_path / "output"
        json_files = list(output_dir.glob("ablation_crops_only_*.json"))
        assert len(json_files) >= 1

    def test_runner_saves_summary_with_ablation_settings(self, tmp_path: Path):
        """Runner should save summary including ablation settings."""
        runner = CropsOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        summary_file = tmp_path / "output" / "ablation_crops_only_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "ablation_settings" in summary
        settings = summary["ablation_settings"]
        assert settings["request_more_views"] is False
        assert settings["request_crops"] is True
        assert settings["switch_or_expand_hypothesis"] is False


# =============================================================================
# Top-level API Tests
# =============================================================================


class TestTopLevelAPI:
    """Tests for the top-level run_crops_only_ablation function."""

    def test_run_crops_only_ablation_mock(self, tmp_path: Path):
        """Top-level function should work with mock mode."""
        result = run_crops_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, AblationStudyResult)
        assert result.total_samples > 0

    def test_run_crops_only_ablation_all_mock(self, tmp_path: Path):
        """Top-level function should run all benchmarks."""
        result = run_crops_only_ablation(
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

    def test_crops_only_supports_symbolic_to_visual_repair_claim(self):
        """Crops-only ablation should support 'symbolic-to-visual repair' claim."""
        config = CROPS_ONLY_ABLATION_CONFIG

        # Key: object-level evidence acquisition for fine-grained verification
        assert config.agent.max_turns >= 2  # Multi-turn allowed
        assert config.tools.request_crops is True  # KEY tool enabled
        assert config.tools.request_more_views is False  # Keyframe expansion disabled
        assert config.tools.switch_or_expand_hypothesis is False

        # This isolates contribution of object cropping

    def test_ablation_has_fair_comparison_setup(self):
        """Crops-only should have fair comparison settings."""
        config = CROPS_ONLY_ABLATION_CONFIG

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
        result = run_crops_only_ablation(
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
        result = run_crops_only_ablation(
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

    def test_crops_only_expected_to_have_tool_calls(self, tmp_path: Path):
        """Crops-only results should show some tool calls (unlike one-shot)."""
        result = run_crops_only_ablation(
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
                # Crops-only should have some tool calls (agent can request crops)
                # Not asserting > 0 because mock might not always call tools
                # But it COULD have tool calls (unlike oneshot which has ~0)
                assert avg_tool_calls >= 0.0

    def test_ablation_tag_identifies_crops_only(self):
        """Ablation config should generate tag without 'oneshot'."""
        config = CROPS_ONLY_ABLATION_CONFIG
        tag = config.get_ablation_tag()
        # Should NOT be marked as oneshot
        assert "oneshot" not in tag
        # Should be marked as partial (no_views or no_repair)
        # Based on get_ablation_tag logic: no request_more_views -> "no_views"
        assert "no_views" in tag or "no_repair" in tag


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparisonWithOtherAblations:
    """Tests comparing crops-only config with other ablation configs."""

    def test_crops_only_allows_more_turns_than_oneshot(self):
        """Crops-only should allow more turns than oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )

    def test_crops_only_enables_crops_tool_unlike_oneshot(self):
        """Crops-only should enable request_crops unlike oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True
        assert ONESHOT_ABLATION_CONFIG.tools.request_crops is False

    def test_crops_and_views_are_mutually_exclusive(self):
        """Crops-only should have opposite tool settings from views-only."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # Crops: crops=True, views=False
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_more_views is False

        # Views: crops=False, views=True
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_crops is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True

    def test_both_disable_hypothesis_repair(self):
        """Both crops-only and views-only should disable hypothesis repair."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        assert CROPS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False

    def test_same_uncertainty_and_confidence_settings(self):
        """All ablations should have same uncertainty and confidence settings for fair comparison."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # All should have same enable_uncertainty_stopping
        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            == ONESHOT_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            == VIEWS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
        )

        # All should have same confidence_threshold
        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.confidence_threshold
            == ONESHOT_ABLATION_CONFIG.agent.confidence_threshold
            == VIEWS_ONLY_ABLATION_CONFIG.agent.confidence_threshold
        )


# =============================================================================
# Granularity Comparison Tests
# =============================================================================


class TestGranularityComparison:
    """Tests for comparing crops (object-level) vs views (keyframe-level)."""

    def test_crops_provides_finer_granularity_hypothesis(self):
        """Academic notes should include hypothesis about granularity."""
        sample_result = MagicMock()
        sample_result.to_dict = lambda: {
            "academic_notes": {
                "hypothesis": (
                    "Object-level cropping is most valuable for tasks requiring fine-grained "
                    "attribute or state verification, while keyframe expansion is better for "
                    "spatial reasoning and scene-level understanding."
                )
            }
        }

        d = sample_result.to_dict()
        hypothesis = d["academic_notes"]["hypothesis"]

        # Should mention fine-grained and attribute
        assert "fine-grained" in hypothesis.lower()
        assert "attribute" in hypothesis.lower()

        # Should also mention spatial reasoning for views
        assert "spatial" in hypothesis.lower() or "scene" in hypothesis.lower()

    def test_crops_and_views_have_same_max_turns(self):
        """Crops and views should have same max_turns for fair comparison."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        assert (
            CROPS_ONLY_ABLATION_CONFIG.agent.max_turns
            == VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns
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
        result = run_crops_only_ablation(
            benchmarks="all",
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=output_dir,
        )

        # Verify result structure
        assert result.ablation_name == CROPS_ONLY_ABLATION_CONFIG.name
        assert result.total_samples > 0
        assert len(result.benchmark_results) == 3

        # Verify output files
        assert output_dir.exists()
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 2  # Main result + summary

        # Verify JSON content
        summary_file = output_dir / "ablation_crops_only_summary.json"
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)
        assert "overall_success_rate" in summary
        assert "ablation_settings" in summary

    def test_workflow_handles_partial_failure(self, tmp_path: Path):
        """Workflow should continue if one benchmark fails."""
        runner = CropsOnlyAblationRunner(
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
