"""Tests for result table generator module.

Tests cover:
- Data model creation and manipulation
- Mock result generation
- Table 1 (main results) generation
- Table 2 (ablation study) generation
- LaTeX formatting correctness
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List

from conceptgraph.evaluation.result_tables import (
    MethodResult,
    BenchmarkResultSet,
    PaperResults,
    create_mock_results,
    generate_table1_main_results,
    generate_table2_ablation_results,
    generate_all_tables,
    load_result_json,
    parse_result_to_method,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_method_result() -> MethodResult:
    """Create a sample method result for testing."""
    return MethodResult(
        method_name="test_method",
        benchmark="openeqa",
        ablation_tag="full",
        accuracy=0.75,
        exact_match=0.65,
        avg_confidence=0.82,
        avg_tool_calls=2.1,
        tool_use_rate=0.78,
        total_samples=100,
        successful_samples=95,
    )


@pytest.fixture
def sample_benchmark_result_set() -> BenchmarkResultSet:
    """Create a sample benchmark result set."""
    result_set = BenchmarkResultSet(benchmark="openeqa")
    result_set.add_result(
        MethodResult(
            method_name="stage1_only",
            benchmark="openeqa",
            ablation_tag="stage1_only",
            accuracy=0.31,
        )
    )
    result_set.add_result(
        MethodResult(
            method_name="oneshot",
            benchmark="openeqa",
            ablation_tag="oneshot",
            accuracy=0.48,
        )
    )
    result_set.add_result(
        MethodResult(
            method_name="full",
            benchmark="openeqa",
            ablation_tag="full",
            accuracy=0.62,
        )
    )
    return result_set


@pytest.fixture
def mock_results() -> PaperResults:
    """Create mock results for testing."""
    return create_mock_results()


# =============================================================================
# Test: Data Models
# =============================================================================


class TestMethodResult:
    """Tests for MethodResult data class."""

    def test_create_method_result(self, sample_method_result: MethodResult) -> None:
        """Test basic method result creation."""
        assert sample_method_result.method_name == "test_method"
        assert sample_method_result.benchmark == "openeqa"
        assert sample_method_result.accuracy == 0.75

    def test_default_values(self) -> None:
        """Test default values for method result."""
        result = MethodResult(method_name="test", benchmark="test")
        assert result.accuracy == 0.0
        assert result.avg_tool_calls == 0.0
        assert result.ablation_tag == "full"

    def test_scanrefer_metrics(self) -> None:
        """Test ScanRefer-specific metrics."""
        result = MethodResult(
            method_name="scanrefer_test",
            benchmark="scanrefer",
            accuracy=0.55,
            acc_at_025=0.55,
            acc_at_050=0.40,
        )
        assert result.acc_at_025 == 0.55
        assert result.acc_at_050 == 0.40


class TestBenchmarkResultSet:
    """Tests for BenchmarkResultSet data class."""

    def test_add_result(self, sample_benchmark_result_set: BenchmarkResultSet) -> None:
        """Test adding results to benchmark set."""
        assert len(sample_benchmark_result_set.methods) == 3
        assert "stage1_only" in sample_benchmark_result_set.methods
        assert "full" in sample_benchmark_result_set.methods

    def test_empty_benchmark_set(self) -> None:
        """Test empty benchmark result set."""
        result_set = BenchmarkResultSet(benchmark="empty")
        assert len(result_set.methods) == 0
        assert result_set.benchmark == "empty"


class TestPaperResults:
    """Tests for PaperResults data class."""

    def test_add_result(self) -> None:
        """Test adding results to paper results."""
        results = PaperResults()
        results.add_result(
            MethodResult(
                method_name="test",
                benchmark="openeqa",
                ablation_tag="full",
            )
        )
        assert "openeqa" in results.benchmarks
        assert "full" in results.ablation_order

    def test_multiple_benchmarks(self) -> None:
        """Test adding results for multiple benchmarks."""
        results = PaperResults()
        results.add_result(
            MethodResult(method_name="openeqa_full", benchmark="openeqa")
        )
        results.add_result(MethodResult(method_name="sqa3d_full", benchmark="sqa3d"))
        results.add_result(
            MethodResult(method_name="scanrefer_full", benchmark="scanrefer")
        )

        assert len(results.benchmarks) == 3
        assert "openeqa" in results.benchmark_order
        assert "sqa3d" in results.benchmark_order
        assert "scanrefer" in results.benchmark_order


# =============================================================================
# Test: Mock Results
# =============================================================================


class TestMockResults:
    """Tests for mock result generation."""

    def test_mock_results_created(self, mock_results: PaperResults) -> None:
        """Test that mock results are created."""
        assert len(mock_results.benchmarks) > 0

    def test_mock_results_have_openeqa(self, mock_results: PaperResults) -> None:
        """Test mock results include OpenEQA."""
        assert "openeqa" in mock_results.benchmarks
        openeqa = mock_results.benchmarks["openeqa"]
        assert len(openeqa.methods) > 0

    def test_mock_results_have_sqa3d(self, mock_results: PaperResults) -> None:
        """Test mock results include SQA3D."""
        assert "sqa3d" in mock_results.benchmarks

    def test_mock_results_have_scanrefer(self, mock_results: PaperResults) -> None:
        """Test mock results include ScanRefer."""
        assert "scanrefer" in mock_results.benchmarks

    def test_mock_results_have_ablations(self, mock_results: PaperResults) -> None:
        """Test mock results include all ablation tags."""
        expected_tags = {"stage1_only", "oneshot", "full", "views_only", "crops_only"}
        actual_tags = set(mock_results.ablation_order)
        assert expected_tags.issubset(actual_tags)

    def test_mock_accuracy_values_realistic(self, mock_results: PaperResults) -> None:
        """Test mock accuracy values are realistic (0-1 range)."""
        for benchmark_set in mock_results.benchmarks.values():
            for method in benchmark_set.methods.values():
                assert 0.0 <= method.accuracy <= 1.0

    def test_mock_full_method_beats_oneshot(self, mock_results: PaperResults) -> None:
        """Test that full method outperforms oneshot."""
        for benchmark_set in mock_results.benchmarks.values():
            full_result = None
            oneshot_result = None
            for method in benchmark_set.methods.values():
                if method.ablation_tag == "full":
                    full_result = method
                elif method.ablation_tag == "oneshot":
                    oneshot_result = method

            if full_result and oneshot_result:
                assert (
                    full_result.accuracy > oneshot_result.accuracy
                ), f"Full method should beat oneshot on {benchmark_set.benchmark}"


# =============================================================================
# Test: Table 1 Generation
# =============================================================================


class TestTable1Generation:
    """Tests for Table 1 (main results) generation."""

    def test_table1_generates_latex(self, mock_results: PaperResults) -> None:
        """Test that Table 1 generates valid LaTeX."""
        latex = generate_table1_main_results(mock_results)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_table1_has_caption(self, mock_results: PaperResults) -> None:
        """Test that Table 1 has a caption."""
        latex = generate_table1_main_results(mock_results)
        assert "\\caption{" in latex

    def test_table1_has_label(self, mock_results: PaperResults) -> None:
        """Test that Table 1 has a label."""
        latex = generate_table1_main_results(mock_results)
        assert "\\label{tab:main-results}" in latex

    def test_table1_includes_all_benchmarks(self, mock_results: PaperResults) -> None:
        """Test that Table 1 includes all benchmarks."""
        latex = generate_table1_main_results(mock_results)
        assert "OpenEQA" in latex
        assert "SQA3D" in latex
        assert "ScanRefer" in latex

    def test_table1_includes_methods(self, mock_results: PaperResults) -> None:
        """Test that Table 1 includes all methods."""
        latex = generate_table1_main_results(mock_results)
        assert "Stage 1 Only" in latex
        assert "One-shot" in latex
        assert "Ours" in latex

    def test_table1_custom_caption(self, mock_results: PaperResults) -> None:
        """Test custom caption in Table 1."""
        caption = "Custom Caption for Table 1"
        latex = generate_table1_main_results(mock_results, caption=caption)
        assert caption in latex

    def test_table1_custom_label(self, mock_results: PaperResults) -> None:
        """Test custom label in Table 1."""
        label = "tab:custom-label"
        latex = generate_table1_main_results(mock_results, label=label)
        assert f"\\label{{{label}}}" in latex


# =============================================================================
# Test: Table 2 Generation
# =============================================================================


class TestTable2Generation:
    """Tests for Table 2 (ablation study) generation."""

    def test_table2_generates_latex(self, mock_results: PaperResults) -> None:
        """Test that Table 2 generates valid LaTeX."""
        latex = generate_table2_ablation_results(mock_results)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex

    def test_table2_has_caption(self, mock_results: PaperResults) -> None:
        """Test that Table 2 has a caption."""
        latex = generate_table2_ablation_results(mock_results)
        assert "\\caption{" in latex

    def test_table2_has_label(self, mock_results: PaperResults) -> None:
        """Test that Table 2 has a label."""
        latex = generate_table2_ablation_results(mock_results)
        assert "\\label{tab:ablation}" in latex

    def test_table2_includes_ablations(self, mock_results: PaperResults) -> None:
        """Test that Table 2 includes ablation conditions."""
        latex = generate_table2_ablation_results(mock_results)
        assert "Views" in latex  # "+ Views" ablation
        assert "Crops" in latex  # "+ Crops" ablation
        assert "Repair" in latex  # "+ Repair" ablation

    def test_table2_includes_avg_tools(self, mock_results: PaperResults) -> None:
        """Test that Table 2 includes average tool calls column."""
        latex = generate_table2_ablation_results(mock_results)
        assert "Avg. Tools" in latex

    def test_table2_custom_benchmarks(self, mock_results: PaperResults) -> None:
        """Test Table 2 with custom benchmark list."""
        latex = generate_table2_ablation_results(mock_results, benchmarks=["openeqa"])
        assert "OpenEQA" in latex
        # Should not have other benchmarks in header (they might appear in data)


# =============================================================================
# Test: LaTeX Formatting
# =============================================================================


class TestLatexFormatting:
    """Tests for LaTeX formatting correctness."""

    def test_table1_formatting(self, mock_results: PaperResults) -> None:
        """Test Table 1 has proper LaTeX formatting."""
        latex = generate_table1_main_results(mock_results)

        # Check for balanced braces
        assert latex.count("{") == latex.count("}")

        # Check for required LaTeX commands
        assert "\\centering" in latex
        assert "\\small" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_table2_formatting(self, mock_results: PaperResults) -> None:
        """Test Table 2 has proper LaTeX formatting."""
        latex = generate_table2_ablation_results(mock_results)

        # Check for balanced braces
        assert latex.count("{") == latex.count("}")

        # Check for required LaTeX commands
        assert "\\centering" in latex
        assert "\\midrule" in latex

    def test_percentage_formatting(self, mock_results: PaperResults) -> None:
        """Test that percentages are formatted correctly."""
        latex = generate_table1_main_results(mock_results)
        # Should have percentage values (e.g., 62.3)
        import re

        numbers = re.findall(r"\d+\.\d+", latex)
        assert len(numbers) > 0  # Should have numeric values


# =============================================================================
# Test: File Operations
# =============================================================================


class TestFileOperations:
    """Tests for file loading and writing."""

    def test_generate_all_tables_mock(self) -> None:
        """Test generating all tables with mock data."""
        table1, table2 = generate_all_tables(use_mock=True)
        assert "\\begin{table}" in table1
        assert "\\begin{table}" in table2

    def test_generate_all_tables_to_file(self) -> None:
        """Test writing tables to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_all_tables(output_dir=output_dir, use_mock=True)

            table1_path = output_dir / "table1_main_results.tex"
            table2_path = output_dir / "table2_ablation.tex"

            assert table1_path.exists()
            assert table2_path.exists()

            # Check file contents
            with open(table1_path) as f:
                content = f.read()
                assert "\\begin{table}" in content

    def test_load_result_json_invalid_path(self) -> None:
        """Test loading from invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            load_result_json(Path("/nonexistent/path.json"))


# =============================================================================
# Test: Result Parsing
# =============================================================================


class TestResultParsing:
    """Tests for parsing result JSON files."""

    def test_parse_stage1_only_result(self) -> None:
        """Test parsing a Stage 1 only result."""
        data = {
            "benchmark": "openeqa",
            "config": {
                "stage2_enabled": False,
                "ablation_tag": "stage1_only",
            },
            "summary": {
                "total_samples": 50,
                "stage1_success": 50,
            },
            "per_sample_results": [],
        }
        result = parse_result_to_method(data, "test")
        assert result.ablation_tag == "stage1_only"
        assert result.benchmark == "openeqa"

    def test_parse_full_result(self) -> None:
        """Test parsing a full Stage 2 result."""
        data = {
            "benchmark": "openeqa",
            "config": {
                "stage2_enabled": True,
                "ablation_tag": "full",
            },
            "summary": {
                "total_samples": 50,
                "stage1_success": 48,
            },
            "per_sample_results": [
                {
                    "stage2_success": True,
                    "stage2_confidence": 0.85,
                    "stage2_tool_calls": 2,
                    "metrics": {"accuracy": 0.8},
                }
            ],
        }
        result = parse_result_to_method(data, "test")
        assert result.ablation_tag == "full"
        assert result.accuracy == 0.8


# =============================================================================
# Test: Academic Alignment
# =============================================================================


class TestAcademicAlignment:
    """Tests for academic alignment of result tables."""

    def test_table1_shows_improvement(self, mock_results: PaperResults) -> None:
        """Test that Table 1 shows improvement of our method."""
        latex = generate_table1_main_results(mock_results)
        # Should have an improvement row
        assert "Improvement" in latex

    def test_ablation_shows_component_contribution(
        self, mock_results: PaperResults
    ) -> None:
        """Test that ablation table shows component contributions."""
        latex = generate_table2_ablation_results(mock_results)
        # Views, Crops, Repair should all be present
        assert "Views" in latex
        assert "Crops" in latex
        assert "Repair" in latex

    def test_results_support_evidence_seeking_claim(
        self, mock_results: PaperResults
    ) -> None:
        """Test that results support evidence-seeking claim."""
        # Full method should have more tool calls than oneshot
        openeqa = mock_results.benchmarks.get("openeqa")
        if openeqa:
            full = oneshot = None
            for m in openeqa.methods.values():
                if m.ablation_tag == "full":
                    full = m
                elif m.ablation_tag == "oneshot":
                    oneshot = m

            if full and oneshot:
                assert full.avg_tool_calls > oneshot.avg_tool_calls
