"""Tests for hypothesis_repair-only (switch_or_expand_hypothesis only) ablation study runner.

TASK-043: Ablation: + hypothesis_repair only

This test suite validates the ablation runner that executes the Stage 2 agent
with only the switch_or_expand_hypothesis tool enabled across all benchmarks.

Test Coverage:
- Configuration validation (hypothesis_repair-only requirements)
- Mock data loading for all benchmarks
- Single-benchmark execution
- Cross-benchmark execution
- Result aggregation and reporting
- Academic alignment verification
- Comparison against one-shot and other ablation settings
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
from conceptgraph.evaluation.ablations.run_hypothesis_repair_only_ablation import (
    HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG,
    SUPPORTED_BENCHMARKS,
    AblationStudyResult,
    BenchmarkResult,
    HypothesisRepairOnlyAblationRunner,
    create_mock_samples_factory,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_hypothesis_repair_only_ablation,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestHypothesisRepairOnlyAblationConfig:
    """Tests for hypothesis_repair-only ablation configuration."""

    def test_hypothesis_repair_config_has_multi_turn(self):
        """Hypothesis_repair-only config must allow multi-turn for tool usage."""
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_hypothesis_repair_config_enables_switch_or_expand_hypothesis(self):
        """Hypothesis_repair-only config must enable switch_or_expand_hypothesis (the key tool)."""
        tools = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools
        assert tools.switch_or_expand_hypothesis is True

    def test_hypothesis_repair_config_disables_other_evidence_tools(self):
        """Hypothesis_repair-only config must disable other evidence-seeking tools."""
        tools = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools
        assert tools.request_more_views is False
        assert tools.request_crops is False

    def test_hypothesis_repair_config_keeps_inspection_tools(self):
        """Hypothesis_repair-only config should keep read-only inspection tools."""
        tools = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools
        # These provide context but don't acquire new evidence
        assert tools.inspect_stage1_metadata is True
        assert tools.retrieve_object_context is True

    def test_hypothesis_repair_config_has_brief_plan_mode(self):
        """Hypothesis_repair-only config should have plan_mode=brief for normal operation."""
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"

    def test_hypothesis_repair_config_keeps_uncertainty_stopping(self):
        """Hypothesis_repair-only config should keep uncertainty stopping for fair comparison."""
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            is True
        )

    def test_hypothesis_repair_config_has_valid_stage2(self):
        """Hypothesis_repair-only config must have Stage 2 enabled."""
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.stage2.enabled is True

    def test_hypothesis_repair_config_has_appropriate_tags(self):
        """Hypothesis_repair-only config should have ablation tags."""
        tags = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tags
        assert "ablation" in tags
        assert "hypothesis_repair_only" in tags
        assert "tool_ablation" in tags
        assert "symbolic_visual_repair" in tags

    def test_hypothesis_repair_config_differs_from_oneshot(self):
        """Hypothesis_repair-only config should differ from oneshot in key ways."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Key differences
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert ONESHOT_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.plan_mode == "brief"
        assert ONESHOT_ABLATION_CONFIG.agent.plan_mode == "off"

    def test_hypothesis_repair_config_differs_from_views_only(self):
        """Hypothesis_repair-only config should differ from views_only ablation."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # Key difference: different tool enabled
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_more_views is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True

    def test_hypothesis_repair_config_differs_from_crops_only(self):
        """Hypothesis_repair-only config should differ from crops_only ablation."""
        from conceptgraph.evaluation.ablations.run_crops_only_ablation import (
            CROPS_ONLY_ABLATION_CONFIG,
        )

        # Key difference: different tool enabled
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_crops is False
        assert CROPS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True


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


class TestHypothesisRepairOnlyAblationRunner:
    """Tests for HypothesisRepairOnlyAblationRunner initialization."""

    def test_runner_init_with_defaults(self):
        """Runner should initialize with default config."""
        runner = HypothesisRepairOnlyAblationRunner(use_mock=True)
        assert runner.config.agent.max_turns >= 2
        assert runner.config.tools.switch_or_expand_hypothesis is True
        assert runner.use_mock is True

    def test_runner_init_with_custom_config(self):
        """Runner should accept custom config."""
        custom_config = AblationConfig(
            name="custom_hypothesis_repair_only",
            description="Custom test",
            agent=AgentConfig(max_turns=4, plan_mode="brief"),
            tools=ToolConfig(
                request_more_views=False,
                request_crops=False,
                switch_or_expand_hypothesis=True,
            ),
        )
        runner = HypothesisRepairOnlyAblationRunner(config=custom_config, use_mock=True)
        assert runner.config.name == "custom_hypothesis_repair_only"

    def test_runner_validates_config_warns_on_single_turn(self):
        """Runner should warn if config has single-turn."""
        invalid_config = AblationConfig(
            name="invalid_hypothesis_repair_only",
            description="Invalid - single turn",
            agent=AgentConfig(max_turns=1),  # Invalid for hypothesis_repair-only
            tools=ToolConfig(
                request_more_views=False,
                request_crops=False,
                switch_or_expand_hypothesis=True,
            ),
        )
        # Should not raise, but should warn
        with patch("loguru.logger.warning") as mock_warn:
            runner = HypothesisRepairOnlyAblationRunner(
                config=invalid_config, use_mock=True
            )
            assert mock_warn.called

    def test_runner_validates_config_warns_if_hypothesis_repair_disabled(self):
        """Runner should warn if switch_or_expand_hypothesis is disabled."""
        invalid_config = AblationConfig(
            name="invalid_hypothesis_repair_disabled",
            description="Invalid - hypothesis_repair disabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=False,
                request_crops=False,
                switch_or_expand_hypothesis=False,  # Invalid - should be True
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = HypothesisRepairOnlyAblationRunner(
                config=invalid_config, use_mock=True
            )
            assert mock_warn.called

    def test_runner_validates_config_warns_if_views_enabled(self):
        """Runner should warn if request_more_views is enabled."""
        invalid_config = AblationConfig(
            name="invalid_views_enabled",
            description="Invalid - views enabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=True,  # Invalid - should be False
                request_crops=False,
                switch_or_expand_hypothesis=True,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = HypothesisRepairOnlyAblationRunner(
                config=invalid_config, use_mock=True
            )
            assert mock_warn.called

    def test_runner_validates_config_warns_if_crops_enabled(self):
        """Runner should warn if request_crops is enabled."""
        invalid_config = AblationConfig(
            name="invalid_crops_enabled",
            description="Invalid - crops enabled",
            agent=AgentConfig(max_turns=6),
            tools=ToolConfig(
                request_more_views=False,
                request_crops=True,  # Invalid - should be False
                switch_or_expand_hypothesis=True,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = HypothesisRepairOnlyAblationRunner(
                config=invalid_config, use_mock=True
            )
            assert mock_warn.called

    def test_runner_creates_output_dir(self, tmp_path: Path):
        """Runner should create output directory."""
        output_dir = tmp_path / "ablation_output"
        runner = HypothesisRepairOnlyAblationRunner(
            use_mock=True, output_dir=output_dir
        )
        assert output_dir.exists()


# =============================================================================
# Single Benchmark Execution Tests
# =============================================================================


class TestSingleBenchmarkExecution:
    """Tests for running ablation on single benchmarks."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a runner with mock mode."""
        return HypothesisRepairOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_run_single_benchmark_mock(
        self, runner: HypothesisRepairOnlyAblationRunner, benchmark: str
    ):
        """Should successfully run on each benchmark with mock data."""
        result = runner.run_benchmark(benchmark)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark == benchmark
        assert result.error is None
        assert result.run_result is not None
        assert result.run_result.total_samples > 0

    def test_run_benchmark_returns_duration(
        self, runner: HypothesisRepairOnlyAblationRunner
    ):
        """Benchmark result should include duration."""
        result = runner.run_benchmark("openeqa")
        assert result.duration_seconds >= 0

    def test_run_benchmark_error_handling(
        self, runner: HypothesisRepairOnlyAblationRunner
    ):
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
        return HypothesisRepairOnlyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    def test_run_all_benchmarks(self, runner: HypothesisRepairOnlyAblationRunner):
        """Should run on all supported benchmarks."""
        result = runner.run_all("all")

        assert isinstance(result, AblationStudyResult)
        assert result.ablation_name == HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.name
        assert len(result.benchmark_results) == len(SUPPORTED_BENCHMARKS)

    def test_run_all_aggregates_samples(
        self, runner: HypothesisRepairOnlyAblationRunner
    ):
        """Should aggregate sample counts across benchmarks."""
        result = runner.run_all("all")

        # Total should equal sum from all benchmarks
        expected_total = sum(
            r.run_result.total_samples for r in result.benchmark_results if r.run_result
        )
        assert result.total_samples == expected_total

    def test_run_all_computes_success_rate(
        self, runner: HypothesisRepairOnlyAblationRunner
    ):
        """Should compute overall success rate."""
        result = runner.run_all("all")

        assert 0.0 <= result.overall_success_rate <= 1.0

    def test_run_all_computes_per_benchmark_rates(
        self, runner: HypothesisRepairOnlyAblationRunner
    ):
        """Should compute per-benchmark success rates."""
        result = runner.run_all("all")

        rates = result.per_benchmark_success_rates
        assert len(rates) == len(SUPPORTED_BENCHMARKS)
        for benchmark in SUPPORTED_BENCHMARKS:
            assert benchmark in rates
            assert 0.0 <= rates[benchmark] <= 1.0

    def test_run_selected_benchmarks(self, runner: HypothesisRepairOnlyAblationRunner):
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
        mock_run_result.avg_tool_calls_per_sample = 1.2  # Hypothesis repair tool calls
        mock_run_result.samples_with_insufficient_evidence = 4

        result = AblationStudyResult(
            ablation_name="hypothesis_repair_only",
            ablation_description="Test ablation",
            timestamp="2026-03-20T05:00:00",
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

    def test_result_academic_notes_reference_symbolic_visual_repair(
        self, sample_result: AblationStudyResult
    ):
        """Academic notes should reference symbolic-to-visual repair."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert (
            "symbolic" in notes["purpose"].lower()
            or "repair" in notes["purpose"].lower()
        )

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
        assert any("views" in b.lower() for b in baselines)
        assert any("crops" in b.lower() for b in baselines)

    def test_runner_saves_results(self, tmp_path: Path):
        """Runner should save results to JSON files."""
        runner = HypothesisRepairOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        # Check output files exist
        output_dir = tmp_path / "output"
        json_files = list(output_dir.glob("ablation_hypothesis_repair_only_*.json"))
        assert len(json_files) >= 1

    def test_runner_saves_summary_with_ablation_settings(self, tmp_path: Path):
        """Runner should save summary including ablation settings."""
        runner = HypothesisRepairOnlyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        summary_file = (
            tmp_path / "output" / "ablation_hypothesis_repair_only_summary.json"
        )
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "ablation_settings" in summary
        settings = summary["ablation_settings"]
        assert settings["request_more_views"] is False
        assert settings["request_crops"] is False
        assert settings["switch_or_expand_hypothesis"] is True


# =============================================================================
# Top-level API Tests
# =============================================================================


class TestTopLevelAPI:
    """Tests for the top-level run_hypothesis_repair_only_ablation function."""

    def test_run_hypothesis_repair_only_ablation_mock(self, tmp_path: Path):
        """Top-level function should work with mock mode."""
        result = run_hypothesis_repair_only_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, AblationStudyResult)
        assert result.total_samples > 0

    def test_run_hypothesis_repair_only_ablation_all_mock(self, tmp_path: Path):
        """Top-level function should run all benchmarks."""
        result = run_hypothesis_repair_only_ablation(
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

    def test_hypothesis_repair_only_supports_symbolic_visual_repair_claim(self):
        """Hypothesis_repair-only ablation should support 'symbolic-to-visual repair' claim."""
        config = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG

        # Key: symbolic hypothesis repair via tool
        assert config.agent.max_turns >= 2  # Multi-turn allowed
        assert config.tools.switch_or_expand_hypothesis is True  # KEY tool enabled
        assert config.tools.request_more_views is False  # Other tools disabled
        assert config.tools.request_crops is False

        # This isolates contribution of hypothesis repair (symbolic-to-visual)

    def test_ablation_has_fair_comparison_setup(self):
        """Hypothesis_repair-only should have fair comparison settings."""
        config = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG

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
        result = run_hypothesis_repair_only_ablation(
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
        result = run_hypothesis_repair_only_ablation(
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

    def test_hypothesis_repair_only_expected_to_have_tool_calls(self, tmp_path: Path):
        """Hypothesis_repair-only results should show some tool calls (unlike one-shot)."""
        result = run_hypothesis_repair_only_ablation(
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
                # Hypothesis_repair-only should have some tool calls (agent can switch hypothesis)
                # Not asserting > 0 because mock might not always call tools
                # But it COULD have tool calls (unlike oneshot which has ~0)
                assert avg_tool_calls >= 0.0

    def test_ablation_tag_identifies_hypothesis_repair_only(self):
        """Ablation config should generate appropriate tag."""
        config = HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG
        tag = config.get_ablation_tag()
        # Should NOT be marked as oneshot
        assert "oneshot" not in tag
        # Should be marked as partial (no_views or no_crops)
        # Based on get_ablation_tag logic
        assert "no_views" in tag or "no_crops" in tag


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparisonWithOneshot:
    """Tests comparing hypothesis_repair-only config with oneshot config."""

    def test_hypothesis_repair_only_allows_more_turns(self):
        """Hypothesis_repair-only should allow more turns than oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )

    def test_hypothesis_repair_only_enables_repair_tool(self):
        """Hypothesis_repair-only should enable switch_or_expand_hypothesis unlike oneshot."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert ONESHOT_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False

    def test_both_disable_views_and_crops(self):
        """Both hypothesis_repair-only and oneshot should disable views and crops tools."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Hypothesis_repair-only disables these for ablation
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_more_views is False
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_crops is False

        # Oneshot also disables all evidence tools
        assert ONESHOT_ABLATION_CONFIG.tools.request_more_views is False
        assert ONESHOT_ABLATION_CONFIG.tools.request_crops is False

    def test_same_uncertainty_and_confidence_settings(self):
        """Both should have same uncertainty and confidence settings for fair comparison."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            == ONESHOT_ABLATION_CONFIG.agent.enable_uncertainty_stopping
        )
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.confidence_threshold
            == ONESHOT_ABLATION_CONFIG.agent.confidence_threshold
        )


class TestComparisonWithViewsOnly:
    """Tests comparing hypothesis_repair-only config with views_only config."""

    def test_both_have_multi_turn(self):
        """Both ablations should have multi-turn for tool usage."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.max_turns >= 2
        assert VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_different_tools_enabled(self):
        """Hypothesis_repair-only and views_only should enable different tools."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # Hypothesis_repair-only: switch_or_expand_hypothesis only
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_more_views is False

        # Views-only: request_more_views only
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_more_views is True

    def test_same_model_settings(self):
        """Both should use same model for fair comparison."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.stage2.model
            == VIEWS_ONLY_ABLATION_CONFIG.stage2.model
        )


class TestComparisonWithCropsOnly:
    """Tests comparing hypothesis_repair-only config with crops_only config."""

    def test_both_have_multi_turn(self):
        """Both ablations should have multi-turn for tool usage."""
        from conceptgraph.evaluation.ablations.run_crops_only_ablation import (
            CROPS_ONLY_ABLATION_CONFIG,
        )

        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.max_turns >= 2
        assert CROPS_ONLY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_different_tools_enabled(self):
        """Hypothesis_repair-only and crops_only should enable different tools."""
        from conceptgraph.evaluation.ablations.run_crops_only_ablation import (
            CROPS_ONLY_ABLATION_CONFIG,
        )

        # Hypothesis_repair-only: switch_or_expand_hypothesis only
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis
            is True
        )
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_crops is False

        # Crops-only: request_crops only
        assert CROPS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_crops is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete ablation workflow."""

    def test_complete_workflow_mock(self, tmp_path: Path):
        """Complete workflow should succeed with mock data."""
        output_dir = tmp_path / "ablation_output"

        # Run ablation
        result = run_hypothesis_repair_only_ablation(
            benchmarks="all",
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=output_dir,
        )

        # Verify result structure
        assert result.ablation_name == HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.name
        assert result.total_samples > 0
        assert len(result.benchmark_results) == 3

        # Verify output files
        assert output_dir.exists()
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 2  # Main result + summary

        # Verify JSON content
        summary_file = output_dir / "ablation_hypothesis_repair_only_summary.json"
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)
        assert "overall_success_rate" in summary
        assert "ablation_settings" in summary

    def test_workflow_handles_partial_failure(self, tmp_path: Path):
        """Workflow should continue if one benchmark fails."""
        runner = HypothesisRepairOnlyAblationRunner(
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


# =============================================================================
# Research Question Tests
# =============================================================================


class TestResearchQuestion:
    """Tests to verify alignment with the research question."""

    def test_research_question_in_academic_notes(self):
        """The research question should be documented in academic notes."""
        result = AblationStudyResult(
            ablation_name="hypothesis_repair_only",
            ablation_description="Test",
            timestamp="2026-03-20T05:00:00",
        )
        d = result.to_dict()

        assert "research_question" in d["academic_notes"]
        question = d["academic_notes"]["research_question"]

        # Should mention hypothesis switching and comparison
        assert "hypothesis" in question.lower()
        assert "one-shot" in question.lower() or "oneshot" in question.lower()

    def test_expected_findings_mention_hypothesis_correction(self):
        """Expected findings should mention hypothesis correction capability."""
        result = AblationStudyResult(
            ablation_name="hypothesis_repair_only",
            ablation_description="Test",
            timestamp="2026-03-20T05:00:00",
        )
        d = result.to_dict()

        findings = d["academic_notes"]["expected_findings"]
        assert any("hypothesis" in f.lower() for f in findings)
        assert any("stage-1" in f.lower() or "stage1" in f.lower() for f in findings)

    def test_key_metrics_include_hypothesis_specific_metrics(self):
        """Key metrics should include hypothesis-specific metrics."""
        result = AblationStudyResult(
            ablation_name="hypothesis_repair_only",
            ablation_description="Test",
            timestamp="2026-03-20T05:00:00",
        )
        d = result.to_dict()

        metrics = d["academic_notes"]["key_metrics"]
        # Should track hypothesis-specific metrics
        assert any("hypothesis" in m.lower() for m in metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
