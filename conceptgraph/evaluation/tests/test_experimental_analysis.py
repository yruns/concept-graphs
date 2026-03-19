"""Tests for experimental analysis section generator module.

Tests cover:
- Analysis data structure creation
- Benchmark analysis computation
- Ablation analysis computation
- Full analysis aggregation
- Text generation for each subsection
- Complete section generation
- LaTeX output formatting
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List

from conceptgraph.evaluation.experimental_analysis import (
    BenchmarkAnalysis,
    AblationAnalysis,
    ExperimentalAnalysis,
    compute_benchmark_analysis,
    compute_ablation_analysis,
    compute_full_analysis,
    generate_main_results_analysis,
    generate_ablation_analysis_text,
    generate_robustness_analysis,
    generate_tool_usage_analysis,
    generate_calibration_analysis,
    generate_experimental_analysis_section,
)
from conceptgraph.evaluation.result_tables import (
    MethodResult,
    BenchmarkResultSet,
    PaperResults,
    create_mock_results,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_benchmark_analysis() -> BenchmarkAnalysis:
    """Create a sample benchmark analysis for testing."""
    return BenchmarkAnalysis(
        benchmark="openeqa",
        stage1_accuracy=0.31,
        oneshot_accuracy=0.478,
        full_accuracy=0.623,
        avg_tool_calls=2.1,
        tool_use_rate=0.78,
    )


@pytest.fixture
def sample_ablation_analysis() -> AblationAnalysis:
    """Create a sample ablation analysis for testing."""
    return AblationAnalysis(
        ablation_name="oneshot",
        description="Baseline without any tool-based evidence seeking",
        accuracy_delta=-0.15,
        tool_calls_delta=-2.1,
        claim_tested="Adaptive Evidence Acquisition",
        claim_supported=True,
    )


@pytest.fixture
def mock_results() -> PaperResults:
    """Create mock results for testing."""
    return create_mock_results()


@pytest.fixture
def full_analysis() -> ExperimentalAnalysis:
    """Create a full analysis from mock results."""
    return compute_full_analysis()


# =============================================================================
# Test: Data Models
# =============================================================================


class TestBenchmarkAnalysis:
    """Tests for BenchmarkAnalysis data class."""

    def test_create_benchmark_analysis(
        self, sample_benchmark_analysis: BenchmarkAnalysis
    ) -> None:
        """Test basic benchmark analysis creation."""
        assert sample_benchmark_analysis.benchmark == "openeqa"
        assert sample_benchmark_analysis.full_accuracy == 0.623
        assert sample_benchmark_analysis.avg_tool_calls == 2.1

    def test_improvement_computation(
        self, sample_benchmark_analysis: BenchmarkAnalysis
    ) -> None:
        """Test automatically computed improvement metrics."""
        # improvement_over_stage1 = 0.623 - 0.31 = 0.313
        assert abs(sample_benchmark_analysis.improvement_over_stage1 - 0.313) < 0.001
        # improvement_over_oneshot = 0.623 - 0.478 = 0.145
        assert abs(sample_benchmark_analysis.improvement_over_oneshot - 0.145) < 0.001

    def test_relative_improvement_computation(
        self, sample_benchmark_analysis: BenchmarkAnalysis
    ) -> None:
        """Test relative improvement percentage calculation."""
        # relative_improvement_pct = (0.145 / 0.478) * 100 = 30.33%
        assert sample_benchmark_analysis.relative_improvement_pct > 30.0
        assert sample_benchmark_analysis.relative_improvement_pct < 31.0

    def test_zero_oneshot_handling(self) -> None:
        """Test handling of zero oneshot accuracy."""
        analysis = BenchmarkAnalysis(
            benchmark="test",
            stage1_accuracy=0.0,
            oneshot_accuracy=0.0,
            full_accuracy=0.5,
        )
        # Should not raise division by zero
        assert analysis.relative_improvement_pct == 0.0


class TestAblationAnalysis:
    """Tests for AblationAnalysis data class."""

    def test_create_ablation_analysis(
        self, sample_ablation_analysis: AblationAnalysis
    ) -> None:
        """Test basic ablation analysis creation."""
        assert sample_ablation_analysis.ablation_name == "oneshot"
        assert sample_ablation_analysis.claim_tested == "Adaptive Evidence Acquisition"
        assert sample_ablation_analysis.claim_supported is True

    def test_accuracy_delta_negative(
        self, sample_ablation_analysis: AblationAnalysis
    ) -> None:
        """Test that removing component shows negative delta."""
        assert sample_ablation_analysis.accuracy_delta < 0


class TestExperimentalAnalysis:
    """Tests for ExperimentalAnalysis data class."""

    def test_create_empty_analysis(self) -> None:
        """Test creating empty analysis object."""
        analysis = ExperimentalAnalysis()
        assert len(analysis.benchmark_analyses) == 0
        assert len(analysis.ablation_analyses) == 0
        assert analysis.avg_improvement_over_oneshot == 0.0

    def test_claims_supported_default(self) -> None:
        """Test default claims supported dict."""
        analysis = ExperimentalAnalysis()
        assert len(analysis.claims_supported) == 0


# =============================================================================
# Test: Computation Functions
# =============================================================================


class TestComputeBenchmarkAnalysis:
    """Tests for compute_benchmark_analysis function."""

    def test_compute_benchmark_analysis_openeqa(
        self, mock_results: PaperResults
    ) -> None:
        """Test benchmark analysis for OpenEQA."""
        analysis = compute_benchmark_analysis(mock_results, "openeqa")
        assert analysis is not None
        assert analysis.benchmark == "openeqa"
        assert analysis.full_accuracy > analysis.oneshot_accuracy

    def test_compute_benchmark_analysis_sqa3d(self, mock_results: PaperResults) -> None:
        """Test benchmark analysis for SQA3D."""
        analysis = compute_benchmark_analysis(mock_results, "sqa3d")
        assert analysis is not None
        assert analysis.benchmark == "sqa3d"
        assert analysis.full_accuracy > analysis.stage1_accuracy

    def test_compute_benchmark_analysis_scanrefer(
        self, mock_results: PaperResults
    ) -> None:
        """Test benchmark analysis for ScanRefer."""
        analysis = compute_benchmark_analysis(mock_results, "scanrefer")
        assert analysis is not None
        assert analysis.benchmark == "scanrefer"

    def test_compute_benchmark_analysis_nonexistent(
        self, mock_results: PaperResults
    ) -> None:
        """Test handling of nonexistent benchmark."""
        analysis = compute_benchmark_analysis(mock_results, "nonexistent")
        assert analysis is None

    def test_compute_benchmark_analysis_tool_metrics(
        self, mock_results: PaperResults
    ) -> None:
        """Test that tool metrics are captured."""
        analysis = compute_benchmark_analysis(mock_results, "openeqa")
        assert analysis is not None
        assert analysis.avg_tool_calls >= 0
        assert 0 <= analysis.tool_use_rate <= 1


class TestComputeAblationAnalysis:
    """Tests for compute_ablation_analysis function."""

    def test_compute_ablation_analysis_oneshot(
        self, mock_results: PaperResults
    ) -> None:
        """Test ablation analysis for oneshot condition."""
        analysis = compute_ablation_analysis(mock_results, "oneshot")
        assert analysis is not None
        assert analysis.ablation_name == "oneshot"
        assert analysis.claim_tested == "Adaptive Evidence Acquisition"
        # Removing tools should hurt performance
        assert analysis.accuracy_delta < 0

    def test_compute_ablation_analysis_views_only(
        self, mock_results: PaperResults
    ) -> None:
        """Test ablation analysis for views_only condition."""
        analysis = compute_ablation_analysis(mock_results, "views_only")
        assert analysis is not None
        assert analysis.ablation_name == "views_only"

    def test_compute_ablation_analysis_crops_only(
        self, mock_results: PaperResults
    ) -> None:
        """Test ablation analysis for crops_only condition."""
        analysis = compute_ablation_analysis(mock_results, "crops_only")
        assert analysis is not None
        assert analysis.ablation_name == "crops_only"

    def test_compute_ablation_analysis_hypothesis_repair(
        self, mock_results: PaperResults
    ) -> None:
        """Test ablation analysis for hypothesis_repair condition."""
        analysis = compute_ablation_analysis(mock_results, "hypothesis_repair_only")
        assert analysis is not None
        assert analysis.claim_tested == "Symbolic-to-Visual Repair"

    def test_compute_ablation_analysis_no_uncertainty(
        self, mock_results: PaperResults
    ) -> None:
        """Test ablation analysis for no_uncertainty condition."""
        analysis = compute_ablation_analysis(mock_results, "no_uncertainty")
        assert analysis is not None
        assert analysis.claim_tested == "Evidence-Grounded Uncertainty"

    def test_compute_ablation_analysis_nonexistent(
        self, mock_results: PaperResults
    ) -> None:
        """Test handling of nonexistent ablation tag."""
        analysis = compute_ablation_analysis(mock_results, "nonexistent")
        assert analysis is None


class TestComputeFullAnalysis:
    """Tests for compute_full_analysis function."""

    def test_compute_full_analysis_benchmarks(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that all benchmarks are analyzed."""
        assert len(full_analysis.benchmark_analyses) == 3
        assert "openeqa" in full_analysis.benchmark_analyses
        assert "sqa3d" in full_analysis.benchmark_analyses
        assert "scanrefer" in full_analysis.benchmark_analyses

    def test_compute_full_analysis_ablations(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that ablations are analyzed."""
        assert len(full_analysis.ablation_analyses) >= 4
        ablation_names = [a.ablation_name for a in full_analysis.ablation_analyses]
        assert "oneshot" in ablation_names

    def test_compute_full_analysis_overall_stats(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test overall statistics computation."""
        assert full_analysis.avg_improvement_over_oneshot > 0
        assert full_analysis.avg_tool_calls_full > 0

    def test_compute_full_analysis_robustness(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test robustness advantage computation."""
        assert full_analysis.robustness_advantage_at_50pct_drop > 0

    def test_compute_full_analysis_claims_supported(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test claims supported summary."""
        assert len(full_analysis.claims_supported) == 4
        assert "Adaptive Evidence Acquisition" in full_analysis.claims_supported
        assert "Symbolic-to-Visual Repair" in full_analysis.claims_supported
        assert "Evidence-Grounded Uncertainty" in full_analysis.claims_supported
        assert "Unified Multi-Task Policy" in full_analysis.claims_supported

    def test_compute_full_analysis_with_none_results(self) -> None:
        """Test analysis with None results uses mock data."""
        analysis = compute_full_analysis(None)
        assert len(analysis.benchmark_analyses) == 3
        assert len(analysis.ablation_analyses) >= 4


# =============================================================================
# Test: Text Generation Functions
# =============================================================================


class TestGenerateMainResultsAnalysis:
    """Tests for generate_main_results_analysis function."""

    def test_generate_main_results_contains_subsection(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output contains subsection header."""
        text = generate_main_results_analysis(full_analysis)
        assert "\\subsection{Main Results}" in text

    def test_generate_main_results_references_table(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output references Table 1."""
        text = generate_main_results_analysis(full_analysis)
        assert "Table~\\ref{tab:main-results}" in text

    def test_generate_main_results_mentions_benchmarks(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output mentions all three benchmarks."""
        text = generate_main_results_analysis(full_analysis)
        assert "OpenEQA" in text
        assert "SQA3D" in text
        assert "ScanRefer" in text

    def test_generate_main_results_describes_baselines(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output describes baseline methods."""
        text = generate_main_results_analysis(full_analysis)
        assert "Stage 1 Only" in text or "Stage 1 only" in text
        assert "One-shot" in text or "one-shot" in text

    def test_generate_main_results_contains_percentages(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output contains accuracy percentages."""
        text = generate_main_results_analysis(full_analysis)
        assert "%" in text or "\\%" in text

    def test_generate_main_results_itemized(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output contains itemized breakdown."""
        text = generate_main_results_analysis(full_analysis)
        assert "\\begin{itemize}" in text
        assert "\\end{itemize}" in text
        assert "\\item" in text


class TestGenerateAblationAnalysisText:
    """Tests for generate_ablation_analysis_text function."""

    def test_generate_ablation_contains_subsection(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output contains subsection header."""
        text = generate_ablation_analysis_text(full_analysis)
        assert "\\subsection{Ablation Study}" in text

    def test_generate_ablation_references_table(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output references Table 2."""
        text = generate_ablation_analysis_text(full_analysis)
        assert "Table~\\ref{tab:ablation}" in text

    def test_generate_ablation_covers_all_claims(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output covers all four claims."""
        text = generate_ablation_analysis_text(full_analysis)
        assert "Claim 1" in text or "Adaptive Evidence Acquisition" in text
        assert "Claim 2" in text or "Symbolic-to-Visual Repair" in text
        assert "Claim 3" in text or "Evidence-Grounded Uncertainty" in text
        assert "Claim 4" in text or "Unified Multi-Task" in text

    def test_generate_ablation_mentions_tools(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output mentions specific tools."""
        text = generate_ablation_analysis_text(full_analysis)
        # At least one tool should be mentioned
        has_tool_mention = (
            "request\\_more\\_views" in text
            or "request\\_crops" in text
            or "hypothesis" in text
        )
        assert has_tool_mention


class TestGenerateRobustnessAnalysis:
    """Tests for generate_robustness_analysis function."""

    def test_generate_robustness_contains_subsection(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output contains subsection header."""
        text = generate_robustness_analysis(full_analysis)
        assert "\\subsection{Robustness" in text

    def test_generate_robustness_references_figure(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output references detection drop figure."""
        text = generate_robustness_analysis(full_analysis)
        assert "Figure~\\ref{fig:detection-drop}" in text

    def test_generate_robustness_mentions_drop_rates(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output mentions specific drop rates."""
        text = generate_robustness_analysis(full_analysis)
        assert "50\\%" in text or "50%" in text

    def test_generate_robustness_explains_mechanisms(
        self, full_analysis: ExperimentalAnalysis
    ) -> None:
        """Test that output explains robustness mechanisms."""
        text = generate_robustness_analysis(full_analysis)
        # Should explain either additional views or hypothesis repair
        has_mechanism = "views" in text.lower() or "repair" in text.lower()
        assert has_mechanism


class TestGenerateToolUsageAnalysis:
    """Tests for generate_tool_usage_analysis function."""

    def test_generate_tool_usage_contains_subsection(self) -> None:
        """Test that output contains subsection header."""
        text = generate_tool_usage_analysis()
        assert "\\subsection{Tool Usage" in text

    def test_generate_tool_usage_references_figure(self) -> None:
        """Test that output references tool usage figure."""
        text = generate_tool_usage_analysis()
        assert "Figure~\\ref{fig:tool-usage}" in text

    def test_generate_tool_usage_contains_itemize(self) -> None:
        """Test that output contains itemized list."""
        text = generate_tool_usage_analysis()
        assert "\\begin{itemize}" in text
        assert "\\end{itemize}" in text


class TestGenerateCalibrationAnalysis:
    """Tests for generate_calibration_analysis function."""

    def test_generate_calibration_contains_subsection(self) -> None:
        """Test that output contains subsection header."""
        text = generate_calibration_analysis()
        assert "\\subsection{Confidence Calibration}" in text

    def test_generate_calibration_references_figure(self) -> None:
        """Test that output references calibration figure."""
        text = generate_calibration_analysis()
        assert "Figure~\\ref{fig:calibration}" in text

    def test_generate_calibration_explains_concept(self) -> None:
        """Test that output explains calibration concept."""
        text = generate_calibration_analysis()
        has_explanation = "diagonal" in text.lower() or "confidence" in text.lower()
        assert has_explanation


# =============================================================================
# Test: Complete Section Generation
# =============================================================================


class TestGenerateExperimentalAnalysisSection:
    """Tests for generate_experimental_analysis_section function."""

    def test_generate_section_contains_header(self) -> None:
        """Test that output contains section header."""
        text = generate_experimental_analysis_section()
        assert "\\section{Experimental Analysis}" in text

    def test_generate_section_contains_label(self) -> None:
        """Test that output contains section label."""
        text = generate_experimental_analysis_section()
        assert "\\label{sec:experiments}" in text

    def test_generate_section_contains_all_subsections(self) -> None:
        """Test that output contains all expected subsections."""
        text = generate_experimental_analysis_section()
        expected = [
            "\\subsection{Main Results}",
            "\\subsection{Ablation Study}",
            "\\subsection{Robustness",
            "\\subsection{Tool Usage",
            "\\subsection{Confidence Calibration}",
            "\\subsection{Summary}",
        ]
        for subsection in expected:
            assert subsection in text, f"Missing: {subsection}"

    def test_generate_section_contains_summary(self) -> None:
        """Test that output contains summary section."""
        text = generate_experimental_analysis_section()
        assert "\\subsection{Summary}" in text

    def test_generate_section_validates_claims(self) -> None:
        """Test that summary validates all claims."""
        text = generate_experimental_analysis_section()
        # Summary should mention claim validation
        assert "validate" in text.lower() or "claim" in text.lower()

    def test_generate_section_save_to_file(self) -> None:
        """Test saving generated section to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "experimental_analysis.tex"
            text = generate_experimental_analysis_section(output_path=output_path)

            assert output_path.exists()
            with open(output_path) as f:
                saved_text = f.read()
            assert saved_text == text

    def test_generate_section_creates_parent_dirs(self) -> None:
        """Test that output path parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "analysis.tex"
            generate_experimental_analysis_section(output_path=output_path)
            assert output_path.exists()

    def test_generate_section_with_mock_results(self) -> None:
        """Test generation with explicitly provided mock results."""
        results = create_mock_results()
        text = generate_experimental_analysis_section(results=results)
        assert "\\section{Experimental Analysis}" in text


# =============================================================================
# Test: LaTeX Formatting
# =============================================================================


class TestLatexFormatting:
    """Tests for LaTeX formatting correctness."""

    def test_latex_escaping_percentages(self) -> None:
        """Test that percentages are properly escaped."""
        text = generate_experimental_analysis_section()
        # Should have escaped percentages (\\%)
        assert "\\%" in text

    def test_latex_textbf_usage(self) -> None:
        """Test proper use of bold formatting."""
        text = generate_experimental_analysis_section()
        assert "\\textbf{" in text

    def test_latex_no_unmatched_braces(self) -> None:
        """Test that all braces are matched."""
        text = generate_experimental_analysis_section()
        open_braces = text.count("{") - text.count("\\{")
        close_braces = text.count("}") - text.count("\\}")
        assert open_braces == close_braces

    def test_latex_no_unmatched_itemize(self) -> None:
        """Test that all itemize environments are matched."""
        text = generate_experimental_analysis_section()
        open_items = text.count("\\begin{itemize}")
        close_items = text.count("\\end{itemize}")
        assert open_items == close_items

    def test_latex_valid_table_references(self) -> None:
        """Test that table references use proper format."""
        text = generate_experimental_analysis_section()
        # Table references should use ref command
        assert "Table~\\ref{" in text

    def test_latex_valid_figure_references(self) -> None:
        """Test that figure references use proper format."""
        text = generate_experimental_analysis_section()
        # Figure references should use ref command
        assert "Figure~\\ref{" in text


# =============================================================================
# Test: Academic Claims Alignment
# =============================================================================


class TestAcademicClaimsAlignment:
    """Tests for alignment with academic innovation claims."""

    def test_claim1_adaptive_evidence_mentioned(self) -> None:
        """Test that Claim 1 (Adaptive Evidence Acquisition) is addressed."""
        text = generate_experimental_analysis_section()
        assert "adaptive" in text.lower() or "evidence" in text.lower()

    def test_claim2_symbolic_visual_repair_mentioned(self) -> None:
        """Test that Claim 2 (Symbolic-to-Visual Repair) is addressed."""
        text = generate_experimental_analysis_section()
        assert "repair" in text.lower() or "hypothesis" in text.lower()

    def test_claim3_uncertainty_mentioned(self) -> None:
        """Test that Claim 3 (Evidence-Grounded Uncertainty) is addressed."""
        text = generate_experimental_analysis_section()
        assert "uncertainty" in text.lower() or "calibration" in text.lower()

    def test_claim4_unified_policy_mentioned(self) -> None:
        """Test that Claim 4 (Unified Multi-Task Policy) is addressed."""
        text = generate_experimental_analysis_section()
        assert "unified" in text.lower() or "multi-task" in text.lower()


# =============================================================================
# Test: Integration with Other Modules
# =============================================================================


class TestModuleIntegration:
    """Tests for integration with other evaluation modules."""

    def test_imports_from_result_tables(self) -> None:
        """Test that module correctly imports from result_tables."""
        from conceptgraph.evaluation.experimental_analysis import create_mock_results

        results = create_mock_results()
        assert results is not None

    def test_imports_from_visualizations(self) -> None:
        """Test that module correctly imports from visualizations."""
        from conceptgraph.evaluation.experimental_analysis import (
            generate_detection_drop_data,
            generate_tool_usage_data,
        )

        drop_data = generate_detection_drop_data()
        assert len(drop_data) > 0

        tool_data = generate_tool_usage_data()
        assert len(tool_data) > 0

    def test_consistent_mock_data(self) -> None:
        """Test that mock data is consistent across modules."""
        results = create_mock_results()
        openeqa = results.benchmarks.get("openeqa")
        assert openeqa is not None
        # Verify mock data has expected structure
        full_method = openeqa.methods.get("openeqa_stage2_full")
        assert full_method is not None
        # Full method should have reasonable accuracy
        assert 0.5 < full_method.accuracy < 0.8
