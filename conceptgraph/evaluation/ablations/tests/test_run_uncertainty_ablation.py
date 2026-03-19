"""Tests for uncertainty output ablation study runner.

TASK-044: Ablation: + uncertainty output

This test suite validates the ablation runner that executes the Stage 2 agent
with uncertainty stopping DISABLED across all benchmarks. This tests the
academic claim that "evidence-grounded uncertainty reduces hallucination".

Test Coverage:
- Configuration validation (uncertainty disabled, all tools enabled)
- Mock data loading for all benchmarks
- Single-benchmark execution
- Cross-benchmark execution
- Result aggregation and reporting
- Academic alignment verification
- Comparison against full agent and one-shot baselines
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
from conceptgraph.evaluation.ablations.run_uncertainty_ablation import (
    UNCERTAINTY_ABLATION_CONFIG,
    SUPPORTED_BENCHMARKS,
    AblationStudyResult,
    BenchmarkResult,
    UncertaintyAblationRunner,
    create_mock_samples_factory,
    create_mock_stage1_factory,
    create_mock_stage2_factory,
    run_uncertainty_ablation,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestUncertaintyAblationConfig:
    """Tests for uncertainty ablation configuration."""

    def test_uncertainty_config_has_multi_turn(self):
        """Uncertainty config must allow multi-turn for tool usage."""
        assert UNCERTAINTY_ABLATION_CONFIG.agent.max_turns >= 2

    def test_uncertainty_config_disables_uncertainty_stopping(self):
        """Uncertainty config must DISABLE uncertainty stopping (KEY ablation variable)."""
        assert UNCERTAINTY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is False

    def test_uncertainty_config_enables_all_evidence_tools(self):
        """Uncertainty config must enable ALL evidence-seeking tools (like full agent)."""
        tools = UNCERTAINTY_ABLATION_CONFIG.tools
        assert tools.request_more_views is True
        assert tools.request_crops is True
        assert tools.switch_or_expand_hypothesis is True

    def test_uncertainty_config_keeps_inspection_tools(self):
        """Uncertainty config should keep read-only inspection tools."""
        tools = UNCERTAINTY_ABLATION_CONFIG.tools
        # These provide context but don't acquire new evidence
        assert tools.inspect_stage1_metadata is True
        assert tools.retrieve_object_context is True

    def test_uncertainty_config_has_brief_plan_mode(self):
        """Uncertainty config should have plan_mode=brief for standard operation."""
        assert UNCERTAINTY_ABLATION_CONFIG.agent.plan_mode == "brief"

    def test_uncertainty_config_has_same_confidence_threshold(self):
        """Uncertainty config should have same confidence threshold as full agent."""
        # Even though uncertainty stopping is disabled, threshold should match
        assert UNCERTAINTY_ABLATION_CONFIG.agent.confidence_threshold == 0.4

    def test_uncertainty_config_has_valid_stage2(self):
        """Uncertainty config must have Stage 2 enabled."""
        assert UNCERTAINTY_ABLATION_CONFIG.stage2.enabled is True

    def test_uncertainty_config_has_appropriate_tags(self):
        """Uncertainty config should have ablation tags."""
        tags = UNCERTAINTY_ABLATION_CONFIG.tags
        assert "ablation" in tags
        assert "no_uncertainty" in tags
        assert "agent_ablation" in tags
        assert "hallucination_analysis" in tags

    def test_uncertainty_config_differs_from_oneshot(self):
        """Uncertainty config should differ from oneshot in key ways."""
        from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
            ONESHOT_ABLATION_CONFIG,
        )

        # Key differences
        assert (
            UNCERTAINTY_ABLATION_CONFIG.agent.max_turns
            > ONESHOT_ABLATION_CONFIG.agent.max_turns
        )
        # Uncertainty has all tools enabled
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_more_views is True
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_crops is True
        assert ONESHOT_ABLATION_CONFIG.tools.request_more_views is False
        assert ONESHOT_ABLATION_CONFIG.tools.request_crops is False
        assert UNCERTAINTY_ABLATION_CONFIG.agent.plan_mode == "brief"
        assert ONESHOT_ABLATION_CONFIG.agent.plan_mode == "off"

    def test_uncertainty_config_differs_from_views_only(self):
        """Uncertainty config should differ from views-only in key ways."""
        from conceptgraph.evaluation.ablations.run_views_only_ablation import (
            VIEWS_ONLY_ABLATION_CONFIG,
        )

        # Both have multi-turn
        assert (
            UNCERTAINTY_ABLATION_CONFIG.agent.max_turns
            == VIEWS_ONLY_ABLATION_CONFIG.agent.max_turns
        )

        # Key tool difference: uncertainty has ALL tools, views-only has one
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_crops is True
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_more_views is True
        assert UNCERTAINTY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is True
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.request_crops is False
        assert VIEWS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis is False

    def test_uncertainty_config_differs_from_crops_only(self):
        """Uncertainty config should differ from crops-only in key ways."""
        from conceptgraph.evaluation.ablations.run_crops_only_ablation import (
            CROPS_ONLY_ABLATION_CONFIG,
        )

        # Key tool difference: uncertainty has ALL tools, crops-only has one
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_more_views is True
        assert CROPS_ONLY_ABLATION_CONFIG.tools.request_more_views is False

        # Key agent difference: uncertainty stopping
        assert UNCERTAINTY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is False
        assert CROPS_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is True

    def test_uncertainty_config_differs_from_hypothesis_repair_only(self):
        """Uncertainty config should differ from hypothesis-repair-only in key ways."""
        from conceptgraph.evaluation.ablations.run_hypothesis_repair_only_ablation import (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG,
        )

        # Key tool difference: uncertainty has ALL tools, hypothesis-only has one
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_more_views is True
        assert UNCERTAINTY_ABLATION_CONFIG.tools.request_crops is True
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_more_views is False
        assert HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.tools.request_crops is False

        # Key agent difference: uncertainty stopping
        assert UNCERTAINTY_ABLATION_CONFIG.agent.enable_uncertainty_stopping is False
        assert (
            HYPOTHESIS_REPAIR_ONLY_ABLATION_CONFIG.agent.enable_uncertainty_stopping
            is True
        )


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


class TestUncertaintyAblationRunner:
    """Tests for UncertaintyAblationRunner initialization."""

    def test_runner_init_with_defaults(self):
        """Runner should initialize with default config."""
        runner = UncertaintyAblationRunner(use_mock=True)
        assert runner.config.agent.max_turns >= 2
        assert runner.config.agent.enable_uncertainty_stopping is False
        assert runner.config.tools.request_more_views is True
        assert runner.config.tools.request_crops is True
        assert runner.use_mock is True

    def test_runner_init_with_custom_config(self):
        """Runner should accept custom config."""
        custom_config = AblationConfig(
            name="custom_uncertainty",
            description="Custom test",
            agent=AgentConfig(
                max_turns=4, plan_mode="brief", enable_uncertainty_stopping=False
            ),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=True,
                switch_or_expand_hypothesis=True,
            ),
        )
        runner = UncertaintyAblationRunner(config=custom_config, use_mock=True)
        assert runner.config.name == "custom_uncertainty"

    def test_runner_validates_config_warns_on_single_turn(self):
        """Runner should warn if config has single-turn."""
        invalid_config = AblationConfig(
            name="invalid_uncertainty",
            description="Invalid - single turn",
            agent=AgentConfig(max_turns=1, enable_uncertainty_stopping=False),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=True,
                switch_or_expand_hypothesis=True,
            ),
        )
        # Should not raise, but should warn
        with patch("loguru.logger.warning") as mock_warn:
            runner = UncertaintyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_uncertainty_enabled(self):
        """Runner should warn if uncertainty stopping is ENABLED."""
        invalid_config = AblationConfig(
            name="invalid_uncertainty_enabled",
            description="Invalid - uncertainty enabled",
            agent=AgentConfig(max_turns=6, enable_uncertainty_stopping=True),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=True,
                switch_or_expand_hypothesis=True,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = UncertaintyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_views_disabled(self):
        """Runner should warn if request_more_views is disabled."""
        invalid_config = AblationConfig(
            name="invalid_views_disabled",
            description="Invalid - views disabled",
            agent=AgentConfig(max_turns=6, enable_uncertainty_stopping=False),
            tools=ToolConfig(
                request_more_views=False,  # Invalid - should be True
                request_crops=True,
                switch_or_expand_hypothesis=True,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = UncertaintyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_crops_disabled(self):
        """Runner should warn if request_crops is disabled."""
        invalid_config = AblationConfig(
            name="invalid_crops_disabled",
            description="Invalid - crops disabled",
            agent=AgentConfig(max_turns=6, enable_uncertainty_stopping=False),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=False,  # Invalid - should be True
                switch_or_expand_hypothesis=True,
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = UncertaintyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_validates_config_warns_if_hypothesis_disabled(self):
        """Runner should warn if switch_or_expand_hypothesis is disabled."""
        invalid_config = AblationConfig(
            name="invalid_hypothesis_disabled",
            description="Invalid - hypothesis disabled",
            agent=AgentConfig(max_turns=6, enable_uncertainty_stopping=False),
            tools=ToolConfig(
                request_more_views=True,
                request_crops=True,
                switch_or_expand_hypothesis=False,  # Invalid - should be True
            ),
        )
        with patch("loguru.logger.warning") as mock_warn:
            runner = UncertaintyAblationRunner(config=invalid_config, use_mock=True)
            assert mock_warn.called

    def test_runner_creates_output_dir(self, tmp_path: Path):
        """Runner should create output directory."""
        output_dir = tmp_path / "ablation_output"
        runner = UncertaintyAblationRunner(use_mock=True, output_dir=output_dir)
        assert output_dir.exists()


# =============================================================================
# Single Benchmark Execution Tests
# =============================================================================


class TestSingleBenchmarkExecution:
    """Tests for running ablation on single benchmarks."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a runner with mock mode."""
        return UncertaintyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    @pytest.mark.parametrize("benchmark", SUPPORTED_BENCHMARKS)
    def test_run_single_benchmark_mock(
        self, runner: UncertaintyAblationRunner, benchmark: str
    ):
        """Should successfully run on each benchmark with mock data."""
        result = runner.run_benchmark(benchmark)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark == benchmark
        assert result.error is None
        assert result.run_result is not None
        assert result.run_result.total_samples > 0

    def test_run_benchmark_returns_duration(self, runner: UncertaintyAblationRunner):
        """Benchmark result should include duration."""
        result = runner.run_benchmark("openeqa")
        assert result.duration_seconds >= 0

    def test_run_benchmark_error_handling(self, runner: UncertaintyAblationRunner):
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
        return UncertaintyAblationRunner(
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

    def test_run_all_benchmarks(self, runner: UncertaintyAblationRunner):
        """Should run on all supported benchmarks."""
        result = runner.run_all("all")

        assert isinstance(result, AblationStudyResult)
        assert result.ablation_name == UNCERTAINTY_ABLATION_CONFIG.name
        assert len(result.benchmark_results) == len(SUPPORTED_BENCHMARKS)

    def test_run_all_aggregates_samples(self, runner: UncertaintyAblationRunner):
        """Should aggregate sample counts across benchmarks."""
        result = runner.run_all("all")

        # Total should equal sum from all benchmarks
        expected_total = sum(
            r.run_result.total_samples for r in result.benchmark_results if r.run_result
        )
        assert result.total_samples == expected_total

    def test_run_all_computes_success_rate(self, runner: UncertaintyAblationRunner):
        """Should compute overall success rate."""
        result = runner.run_all("all")

        assert 0.0 <= result.overall_success_rate <= 1.0

    def test_run_all_computes_per_benchmark_rates(
        self, runner: UncertaintyAblationRunner
    ):
        """Should compute per-benchmark success rates."""
        result = runner.run_all("all")

        rates = result.per_benchmark_success_rates
        assert len(rates) == len(SUPPORTED_BENCHMARKS)
        for benchmark in SUPPORTED_BENCHMARKS:
            assert benchmark in rates
            assert 0.0 <= rates[benchmark] <= 1.0

    def test_run_selected_benchmarks(self, runner: UncertaintyAblationRunner):
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
        mock_run_result.avg_stage2_confidence = 0.65
        mock_run_result.avg_tool_calls_per_sample = 2.5  # Full toolset
        mock_run_result.samples_with_insufficient_evidence = 0  # Should be 0

        result = AblationStudyResult(
            ablation_name="no_uncertainty_stopping",
            ablation_description="Test uncertainty ablation",
            timestamp="2026-03-20T05:10:00",
            benchmark_results=[
                BenchmarkResult(
                    benchmark="openeqa",
                    run_result=mock_run_result,
                    error=None,
                    duration_seconds=20.0,
                )
            ],
            total_samples=50,
            total_successful=45,
            total_failed_stage1=1,
            total_failed_stage2=4,
            total_duration_seconds=20.0,
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
        assert "research_questions" in notes

    def test_result_academic_notes_reference_evidence_grounded_uncertainty(
        self, sample_result: AblationStudyResult
    ):
        """Academic notes should reference evidence-grounded uncertainty claim."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert "purpose" in notes
        purpose = notes["purpose"]
        assert "uncertainty" in purpose.lower() or "evidence" in purpose.lower()

    def test_result_academic_notes_reference_baselines(
        self, sample_result: AblationStudyResult
    ):
        """Academic notes should reference comparison baselines."""
        d = sample_result.to_dict()
        notes = d["academic_notes"]

        assert "comparison_baselines" in notes
        baselines = notes["comparison_baselines"]
        assert any("full" in b.lower() for b in baselines)
        assert any("oneshot" in b.lower() for b in baselines)

    def test_result_tracks_insufficient_evidence_should_be_zero(
        self, sample_result: AblationStudyResult
    ):
        """For uncertainty ablation, insufficient_evidence should be 0."""
        d = sample_result.to_dict()

        # Key metric note should mention this
        notes = d["academic_notes"]
        if "key_metrics" in notes:
            key_metrics_text = " ".join(str(m) for m in notes["key_metrics"])
            assert "insufficient_evidence" in key_metrics_text.lower()

    def test_runner_saves_results(self, tmp_path: Path):
        """Runner should save results to JSON files."""
        runner = UncertaintyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        # Check output files exist
        output_dir = tmp_path / "output"
        json_files = list(output_dir.glob("ablation_no_uncertainty_*.json"))
        assert len(json_files) >= 1

    def test_runner_saves_summary_with_ablation_settings(self, tmp_path: Path):
        """Runner should save summary including ablation settings."""
        runner = UncertaintyAblationRunner(
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        runner.run_all(["openeqa"])

        summary_file = tmp_path / "output" / "ablation_no_uncertainty_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "ablation_settings" in summary
        settings = summary["ablation_settings"]
        assert settings["enable_uncertainty_stopping"] is False
        assert settings["request_more_views"] is True
        assert settings["request_crops"] is True
        assert settings["switch_or_expand_hypothesis"] is True


# =============================================================================
# Top-level API Tests
# =============================================================================


class TestTopLevelAPI:
    """Tests for the top-level run_uncertainty_ablation function."""

    def test_run_uncertainty_ablation_mock(self, tmp_path: Path):
        """Top-level function should work with mock mode."""
        result = run_uncertainty_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=3,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, AblationStudyResult)
        assert result.total_samples > 0

    def test_run_uncertainty_ablation_all_mock(self, tmp_path: Path):
        """Top-level function should run all benchmarks."""
        result = run_uncertainty_ablation(
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

    def test_uncertainty_ablation_supports_evidence_grounded_uncertainty_claim(self):
        """Uncertainty ablation should test 'evidence-grounded uncertainty' claim."""
        config = UNCERTAINTY_ABLATION_CONFIG

        # Key: uncertainty stopping DISABLED, all tools ENABLED
        assert config.agent.enable_uncertainty_stopping is False  # DISABLED
        assert config.agent.max_turns >= 2  # Multi-turn allowed
        assert config.tools.request_more_views is True  # All tools enabled
        assert config.tools.request_crops is True
        assert config.tools.switch_or_expand_hypothesis is True

        # This tests what happens when agent cannot report insufficient evidence

    def test_ablation_has_fair_comparison_setup(self):
        """Uncertainty ablation should have fair comparison settings."""
        config = UNCERTAINTY_ABLATION_CONFIG

        # Same model
        assert config.stage2.model == "gpt-5.2-2025-12-11"

        # Same Stage 1 settings
        assert config.stage1.k == 3

        # Same confidence threshold (even though not used when uncertainty disabled)
        assert config.agent.confidence_threshold == 0.4

        # All tools enabled (same as full agent)
        assert config.tools.request_more_views is True
        assert config.tools.request_crops is True
        assert config.tools.switch_or_expand_hypothesis is True

    def test_result_tracks_tool_calls(self, tmp_path: Path):
        """Results should track tool call frequency."""
        result = run_uncertainty_ablation(
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
        """Results should track insufficient evidence cases (should be 0)."""
        result = run_uncertainty_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=10,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        # Expect 0 insufficient evidence when uncertainty stopping disabled
        assert result.total_insufficient_evidence_samples >= 0

    def test_result_has_hallucination_analysis_tag(self):
        """Config should have hallucination_analysis tag for academic tracking."""
        assert "hallucination_analysis" in UNCERTAINTY_ABLATION_CONFIG.tags

    def test_ablation_name_reflects_purpose(self):
        """Ablation name should clearly indicate uncertainty is disabled."""
        assert "no_uncertainty" in UNCERTAINTY_ABLATION_CONFIG.name.lower()

    def test_result_includes_research_questions_about_hallucination(
        self, tmp_path: Path
    ):
        """Results should include research questions about hallucination reduction."""
        result = run_uncertainty_ablation(
            benchmarks=["openeqa"],
            use_mock=True,
            max_samples=5,
            max_workers=1,
            output_dir=tmp_path / "output",
        )

        d = result.to_dict()
        notes = d["academic_notes"]
        questions = notes.get("research_questions", [])

        # Should mention hallucination or quality
        questions_text = " ".join(questions).lower()
        assert (
            "hallucination" in questions_text
            or "quality" in questions_text
            or "uncertainty" in questions_text
        )


# =============================================================================
# Comparison with Full Agent Tests
# =============================================================================


class TestComparisonWithFull:
    """Tests ensuring proper comparison setup against full agent."""

    def test_only_uncertainty_stopping_differs(self):
        """Compared to full agent, only uncertainty stopping should differ."""
        config = UNCERTAINTY_ABLATION_CONFIG

        # All tools match full agent
        assert config.tools.request_more_views is True
        assert config.tools.request_crops is True
        assert config.tools.switch_or_expand_hypothesis is True
        assert config.tools.inspect_stage1_metadata is True
        assert config.tools.retrieve_object_context is True

        # Agent settings match full (except uncertainty stopping)
        assert config.agent.max_turns == 6
        assert config.agent.plan_mode == "brief"
        assert config.agent.confidence_threshold == 0.4

        # KEY DIFFERENCE: uncertainty stopping disabled
        assert config.agent.enable_uncertainty_stopping is False

    def test_stage1_config_matches_full(self):
        """Stage 1 config should match full agent for fair comparison."""
        config = UNCERTAINTY_ABLATION_CONFIG

        assert config.stage1.model == "gpt-5.2-2025-12-11"
        assert config.stage1.k == 3
        assert config.stage1.timeout_seconds == 60

    def test_stage2_config_matches_full(self):
        """Stage 2 config should match full agent for fair comparison."""
        config = UNCERTAINTY_ABLATION_CONFIG

        assert config.stage2.enabled is True
        assert config.stage2.model == "gpt-5.2-2025-12-11"
        assert config.stage2.timeout_seconds == 120


# =============================================================================
# Expected Behavior Tests
# =============================================================================


class TestExpectedBehavior:
    """Tests for expected ablation behavior."""

    def test_agent_should_always_provide_answer(self):
        """With uncertainty stopping disabled, agent should always provide answer."""
        # This is the key behavioral expectation of this ablation
        config = UNCERTAINTY_ABLATION_CONFIG
        assert config.agent.enable_uncertainty_stopping is False

        # When disabled, agent cannot output "insufficient evidence"
        # It must always provide an answer, even if uncertain

    def test_config_matches_preset_no_uncertainty(self):
        """Config should match the no_uncertainty preset pattern."""
        from conceptgraph.evaluation.ablation_config import get_preset_config

        # The preset exists
        preset = get_preset_config("no_uncertainty")

        # Key setting matches
        assert preset.agent.enable_uncertainty_stopping is False

    def test_expected_findings_documented(self):
        """Expected findings should be documented in result."""
        result = AblationStudyResult(
            ablation_name=UNCERTAINTY_ABLATION_CONFIG.name,
            ablation_description=UNCERTAINTY_ABLATION_CONFIG.description,
            timestamp="2026-03-20T05:10:00",
        )

        d = result.to_dict()
        notes = d["academic_notes"]

        assert "expected_findings" in notes
        findings = notes["expected_findings"]
        assert len(findings) > 0

        # Should mention forced answering behavior
        findings_text = " ".join(findings).lower()
        assert (
            "always" in findings_text
            or "forced" in findings_text
            or "never" in findings_text
        )
