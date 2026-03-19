"""Unit tests for metrics aggregation module.

Tests cover:
- BenchmarkMetrics dataclass properties and calculations
- AblationGroup and AggregatedResults organization
- aggregate_run_result function with various inputs
- aggregate_multiple_runs function for cross-benchmark aggregation
- LaTeX table export with formatting and highlighting
- Tool usage statistics export
- Summary statistics generation
"""

from __future__ import annotations

import tempfile
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from conceptgraph.evaluation.batch_eval import EvalRunResult, EvalSampleResult
from conceptgraph.evaluation.metrics import (
    AblationGroup,
    AggregatedResults,
    BenchmarkMetrics,
    _describe_ablation,
    _format_ablation_name,
    _format_benchmark_name,
    _format_metric_name,
    _format_tool_name,
    _format_value,
    _is_better,
    aggregate_multiple_runs,
    aggregate_run_result,
    export_summary_statistics,
    export_to_latex_table,
    export_tool_usage_table,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_results() -> List[EvalSampleResult]:
    """Create sample evaluation results for testing."""
    return [
        EvalSampleResult(
            sample_id=f"test_{i:03d}",
            query=f"Query {i}",
            task_type="qa",
            scene_id="scene001",
            stage1_success=i % 5 != 4,  # 80% Stage 1 success
            stage1_keyframe_count=3 if i % 5 != 4 else 0,
            stage1_latency_ms=100 + i * 10,
            stage2_success=i % 5 < 3,  # 60% Stage 2 success
            stage2_status="completed" if i % 5 < 3 else "failed",
            stage2_confidence=0.7 + (i % 3) * 0.1,
            stage2_tool_calls=i % 4,
            stage2_latency_ms=2000 + i * 100,
            metrics={"accuracy": 0.8 + (i % 3) * 0.05, "score": 0.75 + (i % 4) * 0.05},
            tool_trace=[
                {"tool_name": "request_crops", "tool_input": {}} for _ in range(i % 3)
            ],
            timestamp=f"2026-03-20T10:{i:02d}:00",
        )
        for i in range(10)
    ]


@pytest.fixture
def run_result_full(sample_results) -> EvalRunResult:
    """Create a run result with full configuration."""
    return EvalRunResult(
        run_id="test_run_001",
        benchmark_name="openeqa",
        config={
            "stage2_enabled": True,
            "enable_tool_request_more_views": True,
            "enable_tool_request_crops": True,
            "enable_tool_hypothesis_repair": True,
            "enable_uncertainty_stopping": True,
            "stage2_max_turns": 6,
        },
        total_samples=10,
        successful_samples=6,
        failed_stage1=2,
        failed_stage2=2,
        avg_stage1_latency_ms=145.0,
        avg_stage2_latency_ms=2450.0,
        avg_stage2_confidence=0.8,
        avg_tool_calls_per_sample=1.5,
        samples_with_tool_use=7,
        samples_with_insufficient_evidence=1,
        tool_usage_distribution={
            "request_crops": 10,
            "request_more_views": 5,
            "hypothesis_repair": 2,
        },
        results=sample_results,
        start_time="2026-03-20T10:00:00",
        end_time="2026-03-20T10:10:00",
        total_duration_seconds=600.0,
    )


@pytest.fixture
def run_result_oneshot(sample_results) -> EvalRunResult:
    """Create a run result with one-shot configuration."""
    return EvalRunResult(
        run_id="test_run_002",
        benchmark_name="openeqa",
        config={
            "stage2_enabled": True,
            "enable_tool_request_more_views": True,
            "enable_tool_request_crops": True,
            "enable_tool_hypothesis_repair": True,
            "enable_uncertainty_stopping": True,
            "stage2_max_turns": 1,  # One-shot
        },
        total_samples=10,
        successful_samples=4,  # Lower success rate
        failed_stage1=2,
        failed_stage2=4,
        avg_stage1_latency_ms=145.0,
        avg_stage2_latency_ms=1200.0,  # Faster but less accurate
        avg_stage2_confidence=0.6,
        avg_tool_calls_per_sample=0.0,  # No tool calls in one-shot
        samples_with_tool_use=0,
        samples_with_insufficient_evidence=3,
        tool_usage_distribution={},
        results=sample_results,
        start_time="2026-03-20T10:00:00",
        end_time="2026-03-20T10:05:00",
        total_duration_seconds=300.0,
    )


@pytest.fixture
def run_result_sqa3d(sample_results) -> EvalRunResult:
    """Create a run result for SQA3D benchmark."""
    return EvalRunResult(
        run_id="test_run_003",
        benchmark_name="sqa3d",
        config={
            "stage2_enabled": True,
            "enable_tool_request_more_views": True,
            "enable_tool_request_crops": True,
            "enable_tool_hypothesis_repair": True,
            "enable_uncertainty_stopping": True,
            "stage2_max_turns": 6,
        },
        total_samples=10,
        successful_samples=7,
        failed_stage1=1,
        failed_stage2=2,
        avg_stage1_latency_ms=130.0,
        avg_stage2_latency_ms=2200.0,
        avg_stage2_confidence=0.75,
        avg_tool_calls_per_sample=2.0,
        samples_with_tool_use=8,
        samples_with_insufficient_evidence=0,
        tool_usage_distribution={
            "request_crops": 12,
            "request_more_views": 8,
        },
        results=sample_results,
        start_time="2026-03-20T11:00:00",
        end_time="2026-03-20T11:12:00",
        total_duration_seconds=720.0,
    )


# =============================================================================
# BenchmarkMetrics Tests
# =============================================================================


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_initialization_default(self):
        """Test creation with minimal required fields."""
        metrics = BenchmarkMetrics(benchmark_name="test")
        assert metrics.benchmark_name == "test"
        assert metrics.ablation_tag == "full"
        assert metrics.total_samples == 0
        assert metrics.accuracy == 0.0
        assert metrics.tool_usage_distribution == {}

    def test_initialization_full(self):
        """Test creation with all fields populated."""
        metrics = BenchmarkMetrics(
            benchmark_name="openeqa",
            ablation_tag="no_crops",
            total_samples=100,
            successful_samples=85,
            failed_stage1=5,
            failed_stage2=10,
            accuracy=0.82,
            exact_match=0.75,
            avg_confidence=0.78,
            avg_tool_calls=2.3,
        )

        assert metrics.benchmark_name == "openeqa"
        assert metrics.ablation_tag == "no_crops"
        assert metrics.total_samples == 100
        assert metrics.accuracy == 0.82

    def test_success_rate_property(self):
        """Test success rate calculation."""
        metrics = BenchmarkMetrics(
            benchmark_name="test",
            total_samples=100,
            successful_samples=75,
        )
        assert metrics.success_rate == 0.75

    def test_success_rate_zero_samples(self):
        """Test success rate with zero samples."""
        metrics = BenchmarkMetrics(benchmark_name="test", total_samples=0)
        assert metrics.success_rate == 0.0

    def test_stage1_success_rate(self):
        """Test Stage 1 success rate calculation."""
        metrics = BenchmarkMetrics(
            benchmark_name="test",
            total_samples=100,
            failed_stage1=20,
        )
        assert metrics.stage1_success_rate == 0.8

    def test_stage2_success_rate(self):
        """Test Stage 2 success rate (given Stage 1 passed)."""
        metrics = BenchmarkMetrics(
            benchmark_name="test",
            total_samples=100,
            failed_stage1=20,  # 80 passed Stage 1
            failed_stage2=16,  # 64 passed Stage 2
        )
        # Stage 2 success rate = (80 - 16) / 80 = 0.8
        assert metrics.stage2_success_rate == 0.8

    def test_stage2_success_rate_all_stage1_failed(self):
        """Test Stage 2 success rate when all Stage 1 failed."""
        metrics = BenchmarkMetrics(
            benchmark_name="test",
            total_samples=100,
            failed_stage1=100,
        )
        assert metrics.stage2_success_rate == 0.0


# =============================================================================
# AblationGroup Tests
# =============================================================================


class TestAblationGroup:
    """Tests for AblationGroup dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        group = AblationGroup(ablation_tag="full", description="Full configuration")
        assert group.ablation_tag == "full"
        assert group.description == "Full configuration"
        assert group.benchmarks == {}

    def test_get_metric(self):
        """Test getting specific metric from group."""
        group = AblationGroup(ablation_tag="test")
        group.benchmarks["openeqa"] = BenchmarkMetrics(
            benchmark_name="openeqa",
            accuracy=0.85,
        )

        assert group.get_metric("openeqa", "accuracy") == 0.85
        assert group.get_metric("openeqa", "nonexistent") is None
        assert group.get_metric("nonexistent", "accuracy") is None


# =============================================================================
# AggregatedResults Tests
# =============================================================================


class TestAggregatedResults:
    """Tests for AggregatedResults dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        results = AggregatedResults()
        assert results.ablation_groups == {}
        assert results.benchmarks == []

    def test_get_group(self):
        """Test getting ablation group."""
        results = AggregatedResults()
        results.ablation_groups["full"] = AblationGroup(ablation_tag="full")

        assert results.get_group("full") is not None
        assert results.get_group("nonexistent") is None

    def test_get_benchmark_metrics(self):
        """Test getting benchmark metrics from results."""
        results = AggregatedResults()
        group = AblationGroup(ablation_tag="full")
        group.benchmarks["openeqa"] = BenchmarkMetrics(
            benchmark_name="openeqa",
            accuracy=0.85,
        )
        results.ablation_groups["full"] = group

        metrics = results.get_benchmark_metrics("full", "openeqa")
        assert metrics is not None
        assert metrics.accuracy == 0.85

        assert results.get_benchmark_metrics("full", "nonexistent") is None
        assert results.get_benchmark_metrics("nonexistent", "openeqa") is None

    def test_list_ablations_ordering(self):
        """Test ablation listing with proper ordering."""
        results = AggregatedResults()
        results.ablation_groups["no_crops"] = AblationGroup(ablation_tag="no_crops")
        results.ablation_groups["full"] = AblationGroup(ablation_tag="full")
        results.ablation_groups["oneshot"] = AblationGroup(ablation_tag="oneshot")
        results.ablation_groups["stage1_only"] = AblationGroup(
            ablation_tag="stage1_only"
        )

        ablations = results.list_ablations()

        # Full should be first, stage1_only last
        assert ablations[0] == "full"
        assert ablations[-1] == "stage1_only"
        assert ablations[-2] == "oneshot"


# =============================================================================
# aggregate_run_result Tests
# =============================================================================


class TestAggregateRunResult:
    """Tests for aggregate_run_result function."""

    def test_basic_aggregation(self, run_result_full):
        """Test basic aggregation of run result."""
        metrics = aggregate_run_result(run_result_full)

        assert metrics.benchmark_name == "openeqa"
        assert metrics.ablation_tag == "full"
        assert metrics.total_samples == 10
        assert metrics.successful_samples == 6
        assert metrics.avg_confidence == 0.8
        assert metrics.avg_tool_calls == 1.5

    def test_ablation_tag_detection_oneshot(self, run_result_oneshot):
        """Test ablation tag detection for one-shot config."""
        metrics = aggregate_run_result(run_result_oneshot)
        assert metrics.ablation_tag == "oneshot"

    def test_tool_usage_distribution(self, run_result_full):
        """Test tool usage distribution aggregation."""
        metrics = aggregate_run_result(run_result_full)

        assert "request_crops" in metrics.tool_usage_distribution
        assert metrics.tool_usage_distribution["request_crops"] == 10
        assert metrics.tool_usage_distribution["request_more_views"] == 5

    def test_rate_calculations(self, run_result_full):
        """Test rate calculations."""
        metrics = aggregate_run_result(run_result_full)

        # tool_use_rate = 7 / 10 = 0.7
        assert metrics.tool_use_rate == 0.7
        # insufficient_evidence_rate = 1 / 10 = 0.1
        assert metrics.insufficient_evidence_rate == 0.1

    def test_latency_total(self, run_result_full):
        """Test total latency calculation."""
        metrics = aggregate_run_result(run_result_full)

        expected_total = 145.0 + 2450.0
        assert metrics.total_latency_ms == expected_total

    def test_sample_accuracies_extraction(self, run_result_full):
        """Test extraction of per-sample accuracies."""
        metrics = aggregate_run_result(run_result_full)

        # samples have accuracy in their metrics
        assert len(metrics.sample_accuracies) == 10
        assert all(0 <= a <= 1 for a in metrics.sample_accuracies)

    def test_accuracy_statistics(self, run_result_full):
        """Test accuracy mean and std calculation."""
        metrics = aggregate_run_result(run_result_full)

        assert metrics.accuracy > 0
        assert metrics.accuracy_std >= 0

    def test_stage1_only_config(self):
        """Test aggregation with Stage 1 only config."""
        result = EvalRunResult(
            run_id="test",
            benchmark_name="openeqa",
            config={"stage2_enabled": False},
            total_samples=10,
        )

        metrics = aggregate_run_result(result)
        assert metrics.ablation_tag == "stage1_only"

    def test_combined_ablation_tags(self):
        """Test combined ablation tag generation."""
        result = EvalRunResult(
            run_id="test",
            benchmark_name="openeqa",
            config={
                "stage2_enabled": True,
                "enable_tool_request_crops": False,
                "enable_uncertainty_stopping": False,
            },
            total_samples=10,
        )

        metrics = aggregate_run_result(result)
        assert "no_crops" in metrics.ablation_tag
        assert "no_uncertainty" in metrics.ablation_tag


# =============================================================================
# aggregate_multiple_runs Tests
# =============================================================================


class TestAggregateMultipleRuns:
    """Tests for aggregate_multiple_runs function."""

    def test_empty_results(self):
        """Test with empty results list."""
        results = aggregate_multiple_runs([])
        assert results.ablation_groups == {}
        assert results.benchmarks == []

    def test_single_run(self, run_result_full):
        """Test aggregation with single run."""
        results = aggregate_multiple_runs([run_result_full])

        assert "full" in results.ablation_groups
        assert "openeqa" in results.benchmarks
        assert results.get_benchmark_metrics("full", "openeqa") is not None

    def test_multiple_ablations(self, run_result_full, run_result_oneshot):
        """Test aggregation with multiple ablation configurations."""
        results = aggregate_multiple_runs([run_result_full, run_result_oneshot])

        assert "full" in results.ablation_groups
        assert "oneshot" in results.ablation_groups
        assert len(results.benchmarks) == 1  # Both are openeqa

    def test_multiple_benchmarks(self, run_result_full, run_result_sqa3d):
        """Test aggregation with multiple benchmarks."""
        results = aggregate_multiple_runs([run_result_full, run_result_sqa3d])

        assert "openeqa" in results.benchmarks
        assert "sqa3d" in results.benchmarks
        assert len(results.ablation_groups) == 1  # Both are full config

        group = results.get_group("full")
        assert "openeqa" in group.benchmarks
        assert "sqa3d" in group.benchmarks

    def test_cross_benchmark_ablation(
        self, run_result_full, run_result_oneshot, run_result_sqa3d
    ):
        """Test aggregation across multiple benchmarks and ablations."""
        results = aggregate_multiple_runs(
            [run_result_full, run_result_oneshot, run_result_sqa3d]
        )

        assert len(results.ablation_groups) == 2  # full, oneshot
        assert len(results.benchmarks) == 2  # openeqa, sqa3d


# =============================================================================
# LaTeX Export Tests
# =============================================================================


class TestExportToLatexTable:
    """Tests for export_to_latex_table function."""

    def test_empty_results(self):
        """Test export with empty results."""
        results = AggregatedResults()
        output = export_to_latex_table(results)
        assert "No data" in output

    def test_basic_export(self, run_result_full):
        """Test basic LaTeX table export."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_to_latex_table(results)

        assert "\\begin{table}" in output
        assert "\\end{table}" in output
        assert "\\toprule" in output
        assert "\\bottomrule" in output
        assert "OpenEQA" in output  # Formatted benchmark name

    def test_custom_caption_label(self, run_result_full):
        """Test custom caption and label."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_to_latex_table(
            results,
            caption="Custom Caption",
            label="tab:custom",
        )

        assert "\\caption{Custom Caption}" in output
        assert "\\label{tab:custom}" in output

    def test_multiple_benchmarks(self, run_result_full, run_result_sqa3d):
        """Test export with multiple benchmarks."""
        results = aggregate_multiple_runs([run_result_full, run_result_sqa3d])
        output = export_to_latex_table(results)

        assert "OpenEQA" in output
        assert "SQA3D" in output
        assert "\\multicolumn" in output  # For spanning headers

    def test_multiple_ablations(self, run_result_full, run_result_oneshot):
        """Test export with multiple ablation configurations."""
        results = aggregate_multiple_runs([run_result_full, run_result_oneshot])
        output = export_to_latex_table(results)

        assert "Full" in output
        assert "One-shot" in output

    def test_highlight_best(self, run_result_full, run_result_oneshot):
        """Test best value highlighting."""
        results = aggregate_multiple_runs([run_result_full, run_result_oneshot])
        output = export_to_latex_table(results, highlight_best=True)

        # Best values should be bolded
        assert "\\textbf{" in output

    def test_no_highlight(self, run_result_full, run_result_oneshot):
        """Test without best value highlighting."""
        results = aggregate_multiple_runs([run_result_full, run_result_oneshot])
        output = export_to_latex_table(results, highlight_best=False)

        # Very unlikely to have textbf without highlighting
        # (Could still appear in other contexts, so just check it runs)
        assert "\\begin{table}" in output

    def test_custom_metrics(self, run_result_full):
        """Test with custom metrics selection."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_to_latex_table(
            results,
            metrics=["accuracy", "avg_tool_calls"],
        )

        assert "Acc" in output  # accuracy formatted
        assert "Tools" in output  # avg_tool_calls formatted

    def test_percentage_formatting(self, run_result_full):
        """Test percentage formatting for metrics."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_to_latex_table(
            results,
            metrics=["accuracy"],
            percentage_metrics=["accuracy"],
        )

        # Should contain percentage symbol (escaped)
        assert "\\%" in output


class TestExportToolUsageTable:
    """Tests for export_tool_usage_table function."""

    def test_empty_tool_usage(self):
        """Test export with no tool usage data."""
        results = AggregatedResults()
        output = export_tool_usage_table(results)
        assert "No tool usage data" in output

    def test_basic_tool_export(self, run_result_full):
        """Test basic tool usage export."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_tool_usage_table(results)

        assert "\\begin{table}" in output
        assert "request_crops" in output.lower() or "Crops" in output

    def test_multiple_tools(self, run_result_full):
        """Test export with multiple tools."""
        results = aggregate_multiple_runs([run_result_full])
        output = export_tool_usage_table(results)

        # Should show multiple tool columns
        assert "Views" in output or "request_more_views" in output.lower()


class TestExportSummaryStatistics:
    """Tests for export_summary_statistics function."""

    def test_basic_summary(self, run_result_full):
        """Test basic summary generation."""
        results = aggregate_multiple_runs([run_result_full])
        summary = export_summary_statistics(results)

        assert "EVALUATION SUMMARY" in summary
        assert "openeqa" in summary.lower()
        assert "Success Rate" in summary
        assert "Accuracy" in summary

    def test_write_to_file(self, run_result_full):
        """Test writing summary to file."""
        results = aggregate_multiple_runs([run_result_full])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.txt"
            summary = export_summary_statistics(results, output_path=output_path)

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
            assert content == summary


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper formatting functions."""

    def test_format_value_decimal(self):
        """Test decimal value formatting."""
        assert _format_value(0.85, 3, False) == "0.850"
        assert _format_value(0.1234, 2, False) == "0.12"

    def test_format_value_percentage(self):
        """Test percentage value formatting."""
        assert _format_value(0.85, 3, True) == "85.0\\%"
        assert _format_value(0.1234, 2, True) == "12\\%"

    def test_format_metric_name(self):
        """Test metric name formatting."""
        assert _format_metric_name("accuracy") == "Acc"
        assert _format_metric_name("exact_match") == "EM"
        assert _format_metric_name("avg_confidence") == "Conf"
        assert _format_metric_name("unknown_metric") == "Unknown Metric"

    def test_format_benchmark_name(self):
        """Test benchmark name formatting."""
        assert _format_benchmark_name("openeqa") == "OpenEQA"
        assert _format_benchmark_name("sqa3d") == "SQA3D"
        assert _format_benchmark_name("OPENEQA") == "OpenEQA"
        assert _format_benchmark_name("unknown") == "unknown"

    def test_format_ablation_name(self):
        """Test ablation name formatting."""
        assert _format_ablation_name("full") == "Full"
        assert _format_ablation_name("oneshot") == "One-shot"
        assert _format_ablation_name("stage1_only") == "Stage 1"
        assert _format_ablation_name("no_crops") == "$-$Crops"

    def test_format_tool_name(self):
        """Test tool name formatting."""
        assert _format_tool_name("request_crops") == "Crops"
        assert _format_tool_name("request_more_views") == "Views"
        assert _format_tool_name("unknown_tool") == "Unknown Tool"

    def test_is_better_higher_better(self):
        """Test comparison for higher-is-better metrics."""
        assert _is_better("accuracy", 0.9, 0.8) is True
        assert _is_better("accuracy", 0.8, 0.9) is False
        assert _is_better("exact_match", 0.75, 0.70) is True

    def test_is_better_lower_better(self):
        """Test comparison for lower-is-better metrics."""
        assert _is_better("avg_stage1_latency_ms", 100, 150) is True
        assert _is_better("avg_stage1_latency_ms", 150, 100) is False
        assert _is_better("insufficient_evidence_rate", 0.1, 0.2) is True

    def test_describe_ablation(self):
        """Test ablation description generation."""
        assert "all tools" in _describe_ablation("full").lower()
        assert "stage 1" in _describe_ablation("stage1_only").lower()
        assert "one-shot" in _describe_ablation("oneshot").lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, run_result_full, run_result_oneshot, run_result_sqa3d):
        """Test complete aggregation and export workflow."""
        # Aggregate multiple runs
        results = aggregate_multiple_runs(
            [run_result_full, run_result_oneshot, run_result_sqa3d]
        )

        # Generate all exports
        main_table = export_to_latex_table(
            results,
            metrics=["accuracy", "exact_match", "avg_tool_calls"],
            caption="Main Results",
            label="tab:main",
        )
        tool_table = export_tool_usage_table(results)
        summary = export_summary_statistics(results)

        # Verify all outputs are valid
        assert "\\begin{table}" in main_table
        assert "\\begin{table}" in tool_table
        assert "EVALUATION SUMMARY" in summary

        # Verify data integrity
        assert len(results.ablation_groups) == 2  # full, oneshot
        assert len(results.benchmarks) == 2  # openeqa, sqa3d

    def test_ablation_comparison_table(self, run_result_full, run_result_oneshot):
        """Test generating an ablation comparison table."""
        results = aggregate_multiple_runs([run_result_full, run_result_oneshot])

        table = export_to_latex_table(
            results,
            metrics=["success_rate", "accuracy", "avg_confidence", "avg_tool_calls"],
            caption="Ablation Study: Tool Usage Impact",
            label="tab:ablation",
            highlight_best=True,
        )

        # Should have both ablations
        assert "Full" in table
        assert "One-shot" in table

        # Should have highlighting
        assert "\\textbf" in table


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_samples(self):
        """Test aggregation with zero samples."""
        result = EvalRunResult(
            run_id="empty",
            benchmark_name="test",
            config={},
            total_samples=0,
        )

        metrics = aggregate_run_result(result)
        assert metrics.total_samples == 0
        assert metrics.success_rate == 0.0
        assert metrics.accuracy == 0.0

    def test_all_failures(self):
        """Test aggregation when all samples failed."""
        result = EvalRunResult(
            run_id="failures",
            benchmark_name="test",
            config={},
            total_samples=10,
            successful_samples=0,
            failed_stage1=10,
        )

        metrics = aggregate_run_result(result)
        assert metrics.success_rate == 0.0
        assert metrics.stage1_success_rate == 0.0

    def test_missing_metrics_in_samples(self):
        """Test handling samples without accuracy metrics."""
        samples = [
            EvalSampleResult(
                sample_id="no_metrics",
                query="test",
                task_type="qa",
                scene_id="scene",
                stage2_success=True,
                stage2_confidence=0.8,
                metrics={},  # No accuracy
            )
        ]

        result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            total_samples=1,
            results=samples,
        )

        metrics = aggregate_run_result(result)
        assert metrics.sample_accuracies == []
        assert metrics.accuracy == 0.0

    def test_single_sample_statistics(self):
        """Test statistics with single sample (no std possible)."""
        samples = [
            EvalSampleResult(
                sample_id="single",
                query="test",
                task_type="qa",
                scene_id="scene",
                stage2_success=True,
                metrics={"accuracy": 0.9},
            )
        ]

        result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            total_samples=1,
            results=samples,
        )

        metrics = aggregate_run_result(result)
        assert metrics.accuracy == 0.9
        assert metrics.accuracy_std == 0.0  # Can't compute std with 1 sample
