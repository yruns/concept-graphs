"""Tests for related work comparison module.

Tests cover:
- Data model creation and validation
- Comparison table generation
- Related work section text generation
- File output functionality
- Academic alignment verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest

from conceptgraph.evaluation.related_work import (
    # Enums
    TaskType,
    EvidenceAcquisition,
    RepresentationType,
    Venue,
    # Data structures
    BenchmarkResult,
    RelatedMethod,
    DifferentiationPoint,
    RelatedWorkSection,
    # Method factories
    create_3dgraphllm,
    create_sg_nav,
    create_scene_vlm,
    create_leo,
    create_openground,
    create_probe_and_ground,
    create_ovigo_3dhsg,
    create_our_method,
    create_all_methods,
    create_differentiation_points,
    # Table generation
    generate_comparison_table,
    generate_benchmark_comparison_table,
    # Section generation
    generate_related_work_intro,
    generate_3d_vlm_subsection,
    generate_scene_graph_subsection,
    generate_iterative_reasoning_subsection,
    generate_multitask_subsection,
    generate_related_work_section,
    # Output
    save_related_work_section,
    create_related_work_summary,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enumeration types."""

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.QA.value == "qa"
        assert TaskType.GROUNDING.value == "grounding"
        assert TaskType.NAVIGATION.value == "navigation"
        assert TaskType.MANIPULATION.value == "manipulation"
        assert TaskType.CAPTIONING.value == "captioning"

    def test_evidence_acquisition_values(self):
        """Test EvidenceAcquisition enum values."""
        assert EvidenceAcquisition.STATIC.value == "static"
        assert EvidenceAcquisition.ITERATIVE.value == "iterative"
        assert EvidenceAcquisition.ADAPTIVE.value == "adaptive"

    def test_representation_type_values(self):
        """Test RepresentationType enum values."""
        assert RepresentationType.COORDINATES.value == "coordinates"
        assert RepresentationType.POINT_CLOUD.value == "point_cloud"
        assert RepresentationType.SCENE_GRAPH.value == "scene_graph"
        assert RepresentationType.MULTI_VIEW.value == "multi_view"
        assert RepresentationType.HYBRID.value == "hybrid"

    def test_venue_values(self):
        """Test Venue enum values."""
        assert Venue.CVPR.value == "CVPR"
        assert Venue.ICCV.value == "ICCV"
        assert Venue.NEURIPS.value == "NeurIPS"
        assert Venue.ICML.value == "ICML"


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = BenchmarkResult(
            benchmark="ScanRefer",
            metric="Acc@0.25",
            value=62.4,
        )
        assert result.benchmark == "ScanRefer"
        assert result.metric == "Acc@0.25"
        assert result.value == 62.4
        assert result.notes is None

    def test_with_notes(self):
        """Test result with notes."""
        result = BenchmarkResult(
            benchmark="OpenEQA",
            metric="MNAS",
            value=46.7,
            notes="State-of-the-art for open-source models",
        )
        assert "open-source" in result.notes


class TestRelatedMethod:
    """Tests for RelatedMethod dataclass."""

    def test_basic_creation(self):
        """Test basic method creation."""
        method = RelatedMethod(
            name="TestMethod",
            venue=Venue.CVPR,
            year=2025,
            title="Test Paper Title",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        assert method.name == "TestMethod"
        assert method.venue == Venue.CVPR
        assert method.year == 2025

    def test_venue_str_property(self):
        """Test venue string formatting."""
        method = RelatedMethod(
            name="Test",
            venue=Venue.ICCV,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        assert method.venue_str == "ICCV 2025"

    def test_is_iterative_property(self):
        """Test is_iterative property."""
        static_method = RelatedMethod(
            name="Static",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        assert not static_method.is_iterative

        iterative_method = RelatedMethod(
            name="Iterative",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.ITERATIVE,
        )
        assert iterative_method.is_iterative

        adaptive_method = RelatedMethod(
            name="Adaptive",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.ADAPTIVE,
        )
        assert adaptive_method.is_iterative

    def test_default_empty_collections(self):
        """Test default empty collections."""
        method = RelatedMethod(
            name="Test",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        assert method.tasks_supported == set()
        assert method.results == []
        assert method.key_contributions == []
        assert method.limitations == []


class TestDifferentiationPoint:
    """Tests for DifferentiationPoint dataclass."""

    def test_basic_creation(self):
        """Test basic differentiation point creation."""
        point = DifferentiationPoint(
            axis="Evidence Acquisition",
            our_approach="Adaptive evidence seeking",
            alternatives={"3DGraphLLM": "Static input"},
            academic_claim="Adaptive Evidence Acquisition",
        )
        assert point.axis == "Evidence Acquisition"
        assert "Adaptive" in point.our_approach
        assert "3DGraphLLM" in point.alternatives

    def test_with_empirical_support(self):
        """Test differentiation point with empirical support."""
        point = DifferentiationPoint(
            axis="Test",
            our_approach="Test",
            alternatives={},
            academic_claim="Test",
            empirical_support="Table 1 shows +14.3% improvement",
        )
        assert "14.3%" in point.empirical_support


# =============================================================================
# Method Factory Tests
# =============================================================================


class TestMethodFactories:
    """Tests for predefined method creation factories."""

    def test_create_3dgraphllm(self):
        """Test 3DGraphLLM method creation."""
        method = create_3dgraphllm()
        assert method.name == "3DGraphLLM"
        assert method.venue == Venue.ICCV
        assert method.year == 2025
        assert method.uses_scene_graph
        assert method.evidence_acquisition == EvidenceAcquisition.STATIC
        assert not method.supports_recovery_from_detection_failure
        assert len(method.results) > 0  # Has benchmark results
        assert len(method.key_contributions) > 0
        assert len(method.limitations) > 0

    def test_create_sg_nav(self):
        """Test SG-Nav method creation."""
        method = create_sg_nav()
        assert method.name == "SG-Nav"
        assert method.venue == Venue.NEURIPS
        assert method.year == 2024
        assert method.uses_scene_graph
        assert TaskType.NAVIGATION in method.tasks_supported
        assert not method.supports_multi_task_unified_policy

    def test_create_scene_vlm(self):
        """Test Scene-VLM method creation."""
        method = create_scene_vlm()
        assert method.name == "Scene-VLM"
        assert method.evidence_acquisition == EvidenceAcquisition.ITERATIVE
        assert not method.uses_scene_graph
        assert "Three-module" in method.key_contributions[0]

    def test_create_leo(self):
        """Test LEO method creation."""
        method = create_leo()
        assert method.name == "LEO"
        assert method.venue == Venue.ICML
        assert method.year == 2024
        assert method.supports_multi_task_unified_policy
        assert len(method.tasks_supported) >= 4  # QA, captioning, nav, manipulation

    def test_create_openground(self):
        """Test OpenGround method creation."""
        method = create_openground()
        assert method.name == "OpenGround"
        assert method.venue == Venue.CVPR
        assert method.year == 2025
        assert TaskType.GROUNDING in method.tasks_supported

    def test_create_probe_and_ground(self):
        """Test Probe-and-Ground method creation."""
        method = create_probe_and_ground()
        assert method.name == "Probe-and-Ground"
        assert method.venue == Venue.CVPR
        assert method.year == 2026
        assert method.evidence_acquisition == EvidenceAcquisition.ADAPTIVE
        assert method.supports_uncertainty_output

    def test_create_ovigo_3dhsg(self):
        """Test OVIGo-3DHSG method creation."""
        method = create_ovigo_3dhsg()
        assert method.name == "OVIGo-3DHSG"
        assert method.uses_scene_graph
        assert TaskType.NAVIGATION in method.tasks_supported

    def test_create_our_method(self):
        """Test our method creation with all key capabilities."""
        method = create_our_method()
        assert method.name == "Two-Stage Evidence-Seeking Agent"
        assert method.year == 2026

        # Must have all 4 academic innovation capabilities
        assert method.evidence_acquisition == EvidenceAcquisition.ADAPTIVE
        assert method.supports_recovery_from_detection_failure
        assert method.supports_uncertainty_output
        assert method.supports_multi_task_unified_policy

        # Must support all 4 task types
        assert TaskType.QA in method.tasks_supported
        assert TaskType.GROUNDING in method.tasks_supported
        assert TaskType.NAVIGATION in method.tasks_supported
        assert TaskType.MANIPULATION in method.tasks_supported

        # Must have benchmark results
        assert len(method.results) >= 3

    def test_create_all_methods(self):
        """Test all methods creation."""
        methods = create_all_methods()
        assert len(methods) >= 7
        names = {m.name for m in methods}
        assert "3DGraphLLM" in names
        assert "SG-Nav" in names
        assert "LEO" in names


class TestDifferentiationPointsFactory:
    """Tests for differentiation points factory."""

    def test_create_differentiation_points(self):
        """Test differentiation points creation."""
        points = create_differentiation_points()
        assert len(points) == 4  # 4 academic claims

        # Check each academic claim is represented
        claims = {p.academic_claim for p in points}
        assert "Adaptive Evidence Acquisition" in claims
        assert "Symbolic-to-Visual Repair" in claims
        assert "Evidence-Grounded Uncertainty" in claims
        assert "Unified Multi-Task Policy" in claims

    def test_each_point_has_alternatives(self):
        """Test each differentiation point has competitor alternatives."""
        points = create_differentiation_points()
        for point in points:
            assert len(point.alternatives) > 0
            assert point.our_approach  # Not empty
            assert point.axis  # Not empty


# =============================================================================
# Comparison Table Tests
# =============================================================================


class TestComparisonTable:
    """Tests for comparison table generation."""

    def test_generate_comparison_table(self):
        """Test comparison table generation."""
        methods = create_all_methods()
        our_method = create_our_method()
        table = generate_comparison_table(methods, our_method)

        # Check LaTeX structure
        assert "\\begin{table*}" in table
        assert "\\end{table*}" in table
        assert "\\toprule" in table
        assert "\\midrule" in table
        assert "\\bottomrule" in table

        # Check our method is highlighted
        assert "\\textbf{Two-Stage" in table

        # Check caption and label
        assert "\\caption{" in table
        assert "\\label{tab:related-work-comparison}" in table

    def test_table_contains_all_methods(self):
        """Test table contains all methods."""
        methods = create_all_methods()
        our_method = create_our_method()
        table = generate_comparison_table(methods, our_method)

        for method in methods:
            assert method.name in table

    def test_table_has_correct_columns(self):
        """Test table has expected column headers."""
        methods = create_all_methods()
        our_method = create_our_method()
        table = generate_comparison_table(methods, our_method)

        assert "Evidence" in table
        assert "Scene" in table
        assert "Detection" in table
        assert "Uncertainty" in table
        assert "Multi-Task" in table
        assert "Tasks" in table

    def test_adaptive_is_bold(self):
        """Test adaptive evidence acquisition is bold in table."""
        methods = create_all_methods()
        our_method = create_our_method()
        table = generate_comparison_table(methods, our_method)

        assert "\\textbf{Adaptive}" in table


class TestBenchmarkComparisonTable:
    """Tests for benchmark-specific comparison table."""

    def test_scanrefer_comparison(self):
        """Test ScanRefer comparison table."""
        methods = create_all_methods()
        our_method = create_our_method()
        table = generate_benchmark_comparison_table(methods, our_method, "ScanRefer")

        assert "\\begin{table}" in table
        assert "ScanRefer" in table
        assert "Acc@0.25" in table
        assert "Acc@0.50" in table

    def test_missing_benchmark_returns_comment(self):
        """Test that missing benchmark returns comment."""
        methods = [create_sg_nav()]  # No ScanRefer results
        our_method = RelatedMethod(
            name="Test",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        table = generate_benchmark_comparison_table(methods, our_method, "ScanRefer")
        assert "% No results" in table


# =============================================================================
# Section Generation Tests
# =============================================================================


class TestSectionGeneration:
    """Tests for LaTeX section text generation."""

    def test_generate_related_work_intro(self):
        """Test intro paragraph generation."""
        intro = generate_related_work_intro()

        assert "\\section{Related Work}" in intro
        assert "\\label{sec:related}" in intro
        assert "adaptive evidence acquisition" in intro
        assert "symbolic-to-visual repair" in intro
        assert "evidence-grounded uncertainty" in intro
        assert "unified multi-task policy" in intro

    def test_generate_3d_vlm_subsection(self):
        """Test 3D VLM subsection generation."""
        methods = create_all_methods()
        text = generate_3d_vlm_subsection(methods)

        assert "\\subsection{3D Scene Understanding" in text
        assert "3DGraphLLM" in text
        assert "Scene-VLM" in text
        assert "OpenGround" in text

    def test_generate_scene_graph_subsection(self):
        """Test scene graph subsection generation."""
        methods = create_all_methods()
        text = generate_scene_graph_subsection(methods)

        assert "\\subsection{Scene Graphs for Embodied AI}" in text
        assert "SG-Nav" in text
        assert "soft priors" in text  # Key differentiation

    def test_generate_iterative_reasoning_subsection(self):
        """Test iterative reasoning subsection generation."""
        methods = create_all_methods()
        text = generate_iterative_reasoning_subsection(methods)

        assert "\\subsection{Iterative Reasoning" in text
        assert "Probe-and-Ground" in text or "evidence-seeking" in text
        assert "ReAct" in text

    def test_generate_multitask_subsection(self):
        """Test multi-task subsection generation."""
        methods = create_all_methods()
        text = generate_multitask_subsection(methods)

        assert "\\subsection{Multi-Task Embodied Agents}" in text
        assert "LEO" in text
        assert "unified" in text.lower()

    def test_generate_related_work_section_complete(self):
        """Test complete related work section generation."""
        section = generate_related_work_section()

        # Check all sections present
        assert "\\section{Related Work}" in section
        assert "\\subsection{3D Scene Understanding" in section
        assert "\\subsection{Scene Graphs for Embodied AI}" in section
        assert "\\subsection{Iterative Reasoning" in section
        assert "\\subsection{Multi-Task Embodied Agents}" in section

        # Check comparison table included
        assert "\\begin{table*}" in section
        assert "\\label{tab:related-work-comparison}" in section


# =============================================================================
# Output Function Tests
# =============================================================================


class TestOutputFunctions:
    """Tests for file output functions."""

    def test_save_related_work_section(self):
        """Test saving related work section to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "related_work.tex"
            result_path = save_related_work_section(output_path)

            assert result_path == output_path
            assert output_path.exists()

            content = output_path.read_text()
            assert "\\section{Related Work}" in content
            assert "Auto-generated" in content

    def test_save_creates_parent_directories(self):
        """Test save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "related_work.tex"
            result_path = save_related_work_section(output_path)

            assert result_path.exists()
            assert result_path.parent.exists()

    def test_create_related_work_summary(self):
        """Test summary dictionary creation."""
        summary = create_related_work_summary()

        # Check structure
        assert "our_method" in summary
        assert "competitors" in summary
        assert "differentiation_axes" in summary
        assert "statistics" in summary

        # Check our method details
        assert summary["our_method"]["name"] == "Two-Stage Evidence-Seeking Agent"
        assert len(summary["our_method"]["tasks"]) == 4
        assert len(summary["our_method"]["key_contributions"]) == 4

        # Check competitors
        assert len(summary["competitors"]) >= 7

        # Check statistics
        assert summary["statistics"]["total_competitors"] >= 7


# =============================================================================
# Academic Alignment Tests
# =============================================================================


class TestAcademicAlignment:
    """Tests verifying academic alignment with research claims."""

    def test_our_method_has_all_innovation_points(self):
        """Test our method has all 4 academic innovation points."""
        method = create_our_method()

        # Claim 1: Adaptive Evidence Acquisition
        assert method.evidence_acquisition == EvidenceAcquisition.ADAPTIVE

        # Claim 2: Symbolic-to-Visual Repair
        assert method.supports_recovery_from_detection_failure

        # Claim 3: Evidence-Grounded Uncertainty
        assert method.supports_uncertainty_output

        # Claim 4: Unified Multi-Task Policy
        assert method.supports_multi_task_unified_policy
        assert len(method.tasks_supported) >= 4

    def test_no_competitor_has_all_capabilities(self):
        """Test no single competitor has all our capabilities."""
        methods = create_all_methods()

        for method in methods:
            has_all = (
                method.evidence_acquisition == EvidenceAcquisition.ADAPTIVE
                and method.supports_recovery_from_detection_failure
                and method.supports_uncertainty_output
                and method.supports_multi_task_unified_policy
            )
            assert not has_all, f"{method.name} should not have all our capabilities"

    def test_differentiation_covers_all_claims(self):
        """Test differentiation points cover all 4 academic claims."""
        points = create_differentiation_points()
        claims = {p.academic_claim for p in points}

        required_claims = {
            "Adaptive Evidence Acquisition",
            "Symbolic-to-Visual Repair",
            "Evidence-Grounded Uncertainty",
            "Unified Multi-Task Policy",
        }

        assert claims == required_claims

    def test_key_contributions_match_claims(self):
        """Test our method's key contributions align with claims."""
        method = create_our_method()
        contributions = " ".join(method.key_contributions).lower()

        # Each claim should be reflected in contributions
        assert "adaptive" in contributions
        assert "repair" in contributions or "symbolic" in contributions
        assert "uncertainty" in contributions
        assert "unified" in contributions or "multi-task" in contributions

    def test_section_references_tables_and_figures(self):
        """Test generated section references experimental evidence."""
        section = generate_related_work_section()

        # Should reference comparison table
        assert "Table~\\ref{tab:related-work-comparison}" in section


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_methods_list(self):
        """Test with empty methods list."""
        table = generate_comparison_table([], create_our_method())
        assert "\\begin{table*}" in table  # Table still generated

    def test_method_without_results(self):
        """Test method without benchmark results."""
        method = RelatedMethod(
            name="NoResults",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        methods = [method]
        our_method = create_our_method()
        table = generate_benchmark_comparison_table(methods, our_method, "ScanRefer")
        # Should not crash, returns comment
        assert "%" in table or "NoResults" not in table

    def test_method_with_empty_tasks(self):
        """Test table generation with method having no tasks."""
        method = RelatedMethod(
            name="EmptyTasks",
            venue=Venue.CVPR,
            year=2025,
            title="Test",
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
            tasks_supported=set(),
        )
        table = generate_comparison_table([method], create_our_method())
        assert "EmptyTasks" in table

    def test_special_characters_in_method_name(self):
        """Test method names with LaTeX special characters are handled."""
        method = RelatedMethod(
            name="Method-With_Special",
            venue=Venue.CVPR,
            year=2025,
            title="Test & Title",  # Ampersand is special
            representation=RepresentationType.SCENE_GRAPH,
            evidence_acquisition=EvidenceAcquisition.STATIC,
        )
        table = generate_comparison_table([method], create_our_method())
        # Should not crash, method name appears
        assert "Special" in table


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete related work generation workflow."""
        # Create methods and differentiation
        methods = create_all_methods()
        differentiation = create_differentiation_points()

        # Generate section
        section = generate_related_work_section(methods, differentiation)

        # Verify completeness
        assert len(section) > 5000  # Substantial content
        assert section.count("\\subsection{") == 4  # All subsections

        # Verify all methods mentioned
        for method in methods:
            # Method should appear somewhere
            assert method.name in section or method.cite_key.split("2")[0].lower() in section.lower()

    def test_summary_matches_methods(self):
        """Test summary accurately reflects methods."""
        methods = create_all_methods()
        summary = create_related_work_summary()

        # Competitor count should match
        assert summary["statistics"]["total_competitors"] == len(methods)

        # Check adaptive count
        adaptive_count = sum(
            1 for m in methods if m.evidence_acquisition == EvidenceAcquisition.ADAPTIVE
        )
        assert summary["statistics"]["competitors_with_adaptive_evidence"] == adaptive_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
