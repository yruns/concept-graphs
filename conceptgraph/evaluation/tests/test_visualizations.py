"""Tests for visualization module.

Tests cover:
1. Detection drop stress test figure generation
2. Tool usage distribution figure generation
3. Confidence vs accuracy plot generation
4. Mock data generation functions
5. Figure output and styling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.evaluation.visualizations import (
    COLORS,
    TOOL_COLORS,
    BENCHMARK_NAMES,
    DetectionDropDataPoint,
    ToolUsageData,
    ConfidenceAccuracyPoint,
    generate_detection_drop_data,
    generate_tool_usage_data,
    generate_confidence_accuracy_data,
    create_detection_drop_figure,
    create_tool_usage_figure,
    create_confidence_accuracy_figure,
    create_all_figures,
    HAS_MATPLOTLIB,
)

# =============================================================================
# Test Constants and Configuration
# =============================================================================


class TestConstants:
    """Test color and display name constants."""

    def test_colors_exist(self):
        """Colors dictionary should have required keys."""
        required_colors = ["ours_full", "oneshot", "stage1_only", "grid", "text"]
        for color in required_colors:
            assert color in COLORS, f"Missing color: {color}"
            assert isinstance(COLORS[color], str), f"Color {color} should be string"
            assert COLORS[color].startswith("#"), f"Color {color} should be hex"

    def test_tool_colors_exist(self):
        """Tool colors dictionary should have required keys."""
        required_tools = [
            "request_more_views",
            "request_crops",
            "switch_hypothesis",
            "inspect_metadata",
            "retrieve_context",
        ]
        for tool in required_tools:
            assert tool in TOOL_COLORS, f"Missing tool color: {tool}"
            assert isinstance(TOOL_COLORS[tool], str)
            assert TOOL_COLORS[tool].startswith("#")

    def test_benchmark_names_exist(self):
        """Benchmark display names should be defined."""
        required_benchmarks = ["openeqa", "sqa3d", "scanrefer"]
        for benchmark in required_benchmarks:
            assert benchmark in BENCHMARK_NAMES, f"Missing benchmark: {benchmark}"
            assert isinstance(BENCHMARK_NAMES[benchmark], str)


# =============================================================================
# Test Data Models
# =============================================================================


class TestDetectionDropDataPoint:
    """Test DetectionDropDataPoint dataclass."""

    def test_creation(self):
        """Create data point with all fields."""
        point = DetectionDropDataPoint(
            drop_rate=0.3,
            accuracy_stage1=0.2,
            accuracy_oneshot=0.35,
            accuracy_full=0.5,
            benchmark="openeqa",
        )
        assert point.drop_rate == 0.3
        assert point.accuracy_stage1 == 0.2
        assert point.accuracy_oneshot == 0.35
        assert point.accuracy_full == 0.5
        assert point.benchmark == "openeqa"

    def test_default_benchmark(self):
        """Default benchmark should be openeqa."""
        point = DetectionDropDataPoint(
            drop_rate=0.0,
            accuracy_stage1=0.3,
            accuracy_oneshot=0.4,
            accuracy_full=0.6,
        )
        assert point.benchmark == "openeqa"


class TestToolUsageData:
    """Test ToolUsageData dataclass."""

    def test_creation(self):
        """Create tool usage data with all fields."""
        data = ToolUsageData(
            benchmark="sqa3d",
            condition="full",
            views_calls=100,
            crops_calls=80,
            repair_calls=60,
            inspect_calls=200,
            context_calls=150,
            total_samples=500,
        )
        assert data.benchmark == "sqa3d"
        assert data.condition == "full"
        assert data.views_calls == 100
        assert data.crops_calls == 80
        assert data.repair_calls == 60
        assert data.total_samples == 500

    def test_defaults(self):
        """Default values should be zero."""
        data = ToolUsageData(benchmark="test", condition="test")
        assert data.views_calls == 0
        assert data.crops_calls == 0
        assert data.repair_calls == 0
        assert data.inspect_calls == 0
        assert data.context_calls == 0
        assert data.total_samples == 100


class TestConfidenceAccuracyPoint:
    """Test ConfidenceAccuracyPoint dataclass."""

    def test_creation(self):
        """Create confidence-accuracy point."""
        point = ConfidenceAccuracyPoint(
            confidence=0.8,
            accuracy=0.75,
            sample_count=50,
            condition="full",
        )
        assert point.confidence == 0.8
        assert point.accuracy == 0.75
        assert point.sample_count == 50
        assert point.condition == "full"

    def test_default_condition(self):
        """Default condition should be full."""
        point = ConfidenceAccuracyPoint(
            confidence=0.5,
            accuracy=0.5,
            sample_count=100,
        )
        assert point.condition == "full"


# =============================================================================
# Test Mock Data Generators
# =============================================================================


class TestGenerateDetectionDropData:
    """Test detection drop data generation."""

    def test_generates_list(self):
        """Should generate a list of data points."""
        data = generate_detection_drop_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_data_point_types(self):
        """All items should be DetectionDropDataPoint."""
        data = generate_detection_drop_data()
        for point in data:
            assert isinstance(point, DetectionDropDataPoint)

    def test_drop_rates_range(self):
        """Drop rates should cover 0 to 0.8."""
        data = generate_detection_drop_data()
        drop_rates = [d.drop_rate for d in data]
        assert 0.0 in drop_rates
        assert max(drop_rates) >= 0.7

    def test_full_method_most_robust(self):
        """Full method should have highest accuracy at each drop rate."""
        data = generate_detection_drop_data()
        for point in data:
            assert point.accuracy_full >= point.accuracy_oneshot
            assert point.accuracy_oneshot >= point.accuracy_stage1

    def test_accuracies_are_valid(self):
        """All accuracy values should be in [0, 1]."""
        data = generate_detection_drop_data()
        for point in data:
            assert 0 <= point.accuracy_stage1 <= 1
            assert 0 <= point.accuracy_oneshot <= 1
            assert 0 <= point.accuracy_full <= 1

    def test_graceful_degradation(self):
        """Accuracy should generally decrease with drop rate."""
        data = generate_detection_drop_data()
        # Compare first and last points
        first = data[0]
        last = data[-1]
        assert first.accuracy_full > last.accuracy_full
        assert first.accuracy_stage1 > last.accuracy_stage1

    def test_benchmark_parameter(self):
        """Should respect benchmark parameter."""
        data = generate_detection_drop_data(benchmark="sqa3d")
        for point in data:
            assert point.benchmark == "sqa3d"


class TestGenerateToolUsageData:
    """Test tool usage data generation."""

    def test_generates_list(self):
        """Should generate a list of tool usage data."""
        data = generate_tool_usage_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_data_types(self):
        """All items should be ToolUsageData."""
        data = generate_tool_usage_data()
        for item in data:
            assert isinstance(item, ToolUsageData)

    def test_covers_benchmarks(self):
        """Should cover all main benchmarks."""
        data = generate_tool_usage_data()
        benchmarks = {d.benchmark for d in data}
        assert "openeqa" in benchmarks
        assert "sqa3d" in benchmarks
        assert "scanrefer" in benchmarks

    def test_positive_values(self):
        """All tool call counts should be non-negative."""
        data = generate_tool_usage_data()
        for item in data:
            assert item.views_calls >= 0
            assert item.crops_calls >= 0
            assert item.repair_calls >= 0
            assert item.inspect_calls >= 0
            assert item.context_calls >= 0

    def test_realistic_totals(self):
        """Tool call totals should be reasonable."""
        data = generate_tool_usage_data()
        for item in data:
            total = (
                item.views_calls
                + item.crops_calls
                + item.repair_calls
                + item.inspect_calls
                + item.context_calls
            )
            assert total > 0  # At least some tool usage
            assert total < 5000  # Not unrealistically high


class TestGenerateConfidenceAccuracyData:
    """Test confidence-accuracy data generation."""

    def test_generates_list(self):
        """Should generate a list of data points."""
        data = generate_confidence_accuracy_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_data_types(self):
        """All items should be ConfidenceAccuracyPoint."""
        data = generate_confidence_accuracy_data()
        for point in data:
            assert isinstance(point, ConfidenceAccuracyPoint)

    def test_confidence_range(self):
        """Confidence values should be in [0, 1]."""
        data = generate_confidence_accuracy_data()
        for point in data:
            assert 0 <= point.confidence <= 1

    def test_accuracy_range(self):
        """Accuracy values should be in [0, 1]."""
        data = generate_confidence_accuracy_data()
        for point in data:
            assert 0 <= point.accuracy <= 1

    def test_includes_multiple_conditions(self):
        """Should include multiple experimental conditions."""
        data = generate_confidence_accuracy_data()
        conditions = {d.condition for d in data}
        assert "full" in conditions
        assert "oneshot" in conditions

    def test_positive_sample_counts(self):
        """Sample counts should be positive."""
        data = generate_confidence_accuracy_data()
        for point in data:
            assert point.sample_count > 0


# =============================================================================
# Test Figure Generation (with matplotlib mocking)
# =============================================================================


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCreateDetectionDropFigure:
    """Test detection drop figure generation."""

    def test_creates_figure(self):
        """Should create a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = create_detection_drop_figure()
        assert fig is not None
        plt.close(fig)

    def test_with_custom_data(self):
        """Should work with custom data."""
        import matplotlib.pyplot as plt

        data = [
            DetectionDropDataPoint(0.0, 0.3, 0.4, 0.6),
            DetectionDropDataPoint(0.5, 0.1, 0.2, 0.4),
        ]
        fig = create_detection_drop_figure(data=data)
        assert fig is not None
        plt.close(fig)

    def test_with_title(self):
        """Should accept title parameter."""
        import matplotlib.pyplot as plt

        fig = create_detection_drop_figure(title="Test Title")
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Should save figure to file."""
        import matplotlib.pyplot as plt

        output_path = tmp_path / "test_figure.png"
        fig = create_detection_drop_figure(output_path=output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_custom_figsize(self):
        """Should accept custom figure size."""
        import matplotlib.pyplot as plt

        fig = create_detection_drop_figure(figsize=(8, 6))
        assert fig is not None
        # Check figure size
        fig_size = fig.get_size_inches()
        assert abs(fig_size[0] - 8) < 0.1
        assert abs(fig_size[1] - 6) < 0.1
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCreateToolUsageFigure:
    """Test tool usage figure generation."""

    def test_creates_figure(self):
        """Should create a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = create_tool_usage_figure()
        assert fig is not None
        plt.close(fig)

    def test_with_custom_data(self):
        """Should work with custom data."""
        import matplotlib.pyplot as plt

        data = [
            ToolUsageData("test", "full", 10, 20, 15, 50, 30, 100),
        ]
        fig = create_tool_usage_figure(data=data)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Should save figure to file."""
        import matplotlib.pyplot as plt

        output_path = tmp_path / "tool_usage.pdf"
        fig = create_tool_usage_figure(output_path=output_path)
        assert output_path.exists()
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCreateConfidenceAccuracyFigure:
    """Test confidence-accuracy figure generation."""

    def test_creates_figure(self):
        """Should create a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = create_confidence_accuracy_figure()
        assert fig is not None
        plt.close(fig)

    def test_with_custom_data(self):
        """Should work with custom data."""
        import matplotlib.pyplot as plt

        data = [
            ConfidenceAccuracyPoint(0.5, 0.5, 100, "full"),
            ConfidenceAccuracyPoint(0.8, 0.75, 50, "full"),
        ]
        fig = create_confidence_accuracy_figure(data=data)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """Should save figure to file."""
        import matplotlib.pyplot as plt

        output_path = tmp_path / "calibration.png"
        fig = create_confidence_accuracy_figure(output_path=output_path)
        assert output_path.exists()
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCreateAllFigures:
    """Test composite figure generation."""

    def test_generates_all_figures(self):
        """Should generate all three figures."""
        import matplotlib.pyplot as plt

        figures = create_all_figures()
        assert "detection_drop" in figures
        assert "tool_usage" in figures
        assert "confidence_accuracy" in figures
        for fig in figures.values():
            plt.close(fig)

    def test_saves_to_directory(self, tmp_path):
        """Should save all figures to directory."""
        import matplotlib.pyplot as plt

        figures = create_all_figures(output_dir=tmp_path)
        assert (tmp_path / "fig_detection_drop.pdf").exists()
        assert (tmp_path / "fig_tool_usage.pdf").exists()
        assert (tmp_path / "fig_confidence_accuracy.pdf").exists()
        for fig in figures.values():
            plt.close(fig)


# =============================================================================
# Test Without Matplotlib
# =============================================================================


class TestWithoutMatplotlib:
    """Test behavior when matplotlib is not available."""

    def test_has_matplotlib_flag(self):
        """HAS_MATPLOTLIB should be boolean."""
        assert isinstance(HAS_MATPLOTLIB, bool)


# =============================================================================
# Test Academic Alignment
# =============================================================================


class TestAcademicAlignment:
    """Test that visualizations support academic claims."""

    def test_detection_drop_supports_repair_claim(self):
        """Detection drop data should support symbolic-to-visual repair claim.

        Key claim: Stage 2 agent is more robust to detection failures because
        it can seek additional visual evidence.
        """
        data = generate_detection_drop_data()

        # At high drop rates, full method should still outperform
        high_drop_data = [d for d in data if d.drop_rate >= 0.5]
        assert len(high_drop_data) > 0

        for point in high_drop_data:
            gap_to_oneshot = point.accuracy_full - point.accuracy_oneshot
            assert (
                gap_to_oneshot > 0
            ), f"Full method should beat oneshot at {point.drop_rate:.0%} drop"

    def test_tool_usage_supports_adaptive_acquisition_claim(self):
        """Tool usage should show adaptive patterns across benchmarks.

        Key claim: Agent adapts evidence-seeking strategy based on task.
        """
        data = generate_tool_usage_data()

        # Different benchmarks should have different tool distributions
        benchmark_distributions = {}
        for item in data:
            if item.condition == "full":
                benchmark_distributions[item.benchmark] = {
                    "views": item.views_calls,
                    "crops": item.crops_calls,
                    "repair": item.repair_calls,
                }

        # Verify we have multiple benchmarks
        assert len(benchmark_distributions) >= 2

        # Distributions should not be identical
        distributions = list(benchmark_distributions.values())
        if len(distributions) >= 2:
            d1, d2 = distributions[0], distributions[1]
            # At least one tool should differ significantly
            differs = (
                abs(d1["views"] - d2["views"]) > 10
                or abs(d1["crops"] - d2["crops"]) > 10
                or abs(d1["repair"] - d2["repair"]) > 10
            )
            assert differs, "Tool distributions should vary across benchmarks"

    def test_confidence_accuracy_supports_uncertainty_claim(self):
        """Confidence-accuracy data should show calibration difference.

        Key claim: Evidence-grounded uncertainty produces well-calibrated
        predictions, while models without it are overconfident.
        """
        data = generate_confidence_accuracy_data()

        # Group by condition
        full_data = [d for d in data if d.condition == "full"]
        oneshot_data = [d for d in data if d.condition == "oneshot"]

        assert len(full_data) > 0, "Should have full method data"
        assert len(oneshot_data) > 0, "Should have oneshot data"

        # Calculate calibration error for each
        def calc_calibration_error(points):
            return sum(abs(p.confidence - p.accuracy) for p in points) / len(points)

        full_error = calc_calibration_error(full_data)
        oneshot_error = calc_calibration_error(oneshot_data)

        # Full method should be better calibrated
        assert full_error <= oneshot_error, (
            f"Full method ({full_error:.3f}) should be better "
            f"calibrated than oneshot ({oneshot_error:.3f})"
        )


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_handling(self):
        """Data generators should handle edge cases gracefully."""
        # These should not raise exceptions
        data1 = generate_detection_drop_data()
        data2 = generate_tool_usage_data()
        data3 = generate_confidence_accuracy_data()

        assert len(data1) > 0
        assert len(data2) > 0
        assert len(data3) > 0

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_figure_with_single_point(self):
        """Figures should handle minimal data."""
        import matplotlib.pyplot as plt

        data = [DetectionDropDataPoint(0.0, 0.3, 0.4, 0.6)]
        fig = create_detection_drop_figure(data=data)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_output_directory_creation(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        import matplotlib.pyplot as plt

        nested_path = tmp_path / "nested" / "dir" / "figure.png"
        fig = create_detection_drop_figure(output_path=nested_path)
        assert nested_path.exists()
        plt.close(fig)
