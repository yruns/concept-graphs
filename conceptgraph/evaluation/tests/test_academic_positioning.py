"""
Tests for Academic Positioning Module.

Tests the research claim definitions, competitive landscape analysis,
publication strategy, and document generation functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from conceptgraph.evaluation.academic_positioning import (
    # Enums
    NoveltyLevel,
    ContributionType,
    PublicationVenue,
    # Dataclasses
    ResearchClaim,
    CompetingMethod,
    PublicationStrategy,
    AcademicPositioning,
    # Claim creators
    create_adaptive_evidence_claim,
    create_symbolic_repair_claim,
    create_uncertainty_claim,
    create_unified_policy_claim,
    create_all_claims,
    # Competitor creators
    create_probe_and_ground_competitor,
    create_3dgraphllm_competitor,
    create_leo_competitor,
    create_sg_nav_competitor,
    create_all_competitors,
    # Strategy creators
    create_cvpr_strategy,
    create_neurips_strategy,
    # Document generation
    generate_contribution_summary,
    generate_novelty_gap_analysis,
    generate_positioning_header,
    generate_claims_section,
    generate_competitor_section,
    generate_strategy_section,
    generate_gap_analysis_section,
    generate_action_items_section,
    generate_positioning_document,
    create_academic_positioning,
    save_positioning_document,
    create_positioning_summary,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestNoveltyLevel:
    """Tests for NoveltyLevel enum."""

    def test_all_levels_exist(self):
        """All expected novelty levels should exist."""
        assert NoveltyLevel.FIRST.value == "first"
        assert NoveltyLevel.UNIFIED.value == "unified"
        assert NoveltyLevel.IMPROVED.value == "improved"
        assert NoveltyLevel.ALTERNATIVE.value == "alternative"

    def test_level_count(self):
        """Should have exactly 4 novelty levels."""
        assert len(NoveltyLevel) == 4


class TestContributionType:
    """Tests for ContributionType enum."""

    def test_all_types_exist(self):
        """All expected contribution types should exist."""
        assert ContributionType.METHOD.value == "method"
        assert ContributionType.FRAMEWORK.value == "framework"
        assert ContributionType.ANALYSIS.value == "analysis"
        assert ContributionType.BENCHMARK.value == "benchmark"

    def test_type_count(self):
        """Should have exactly 4 contribution types."""
        assert len(ContributionType) == 4


class TestPublicationVenue:
    """Tests for PublicationVenue enum."""

    def test_major_venues_exist(self):
        """All major venues should exist."""
        assert PublicationVenue.CVPR.value == "cvpr"
        assert PublicationVenue.NEURIPS.value == "neurips"
        assert PublicationVenue.ICLR.value == "iclr"
        assert PublicationVenue.ICML.value == "icml"
        assert PublicationVenue.ECCV.value == "eccv"
        assert PublicationVenue.ICCV.value == "iccv"
        assert PublicationVenue.ARXIV.value == "arxiv"


# =============================================================================
# Research Claim Tests
# =============================================================================


class TestResearchClaim:
    """Tests for ResearchClaim dataclass."""

    def test_claim_creation(self):
        """Should create claim with all fields."""
        claim = ResearchClaim(
            claim_id="TEST_001",
            title="Test Claim",
            statement="This is a test claim statement.",
            novelty_level=NoveltyLevel.FIRST,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1", "Exp 2"],
            competing_claims=["Competitor A"],
            key_metrics={"accuracy": "+10%"},
        )
        assert claim.claim_id == "TEST_001"
        assert claim.title == "Test Claim"
        assert claim.novelty_level == NoveltyLevel.FIRST

    def test_strength_score_first_novelty(self):
        """FIRST novelty should have highest base strength."""
        claim = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.FIRST,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1"],
            competing_claims=[],
            key_metrics={},
        )
        assert claim.strength_score > 0.9

    def test_strength_score_alternative_novelty(self):
        """ALTERNATIVE novelty should have lower base strength."""
        claim = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.ALTERNATIVE,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=[],
            competing_claims=[],
            key_metrics={},
        )
        assert claim.strength_score < 0.6

    def test_evidence_bonus(self):
        """More evidence should increase strength."""
        claim_few = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.UNIFIED,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1"],
            competing_claims=[],
            key_metrics={},
        )
        claim_many = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.UNIFIED,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1", "Exp 2", "Exp 3", "Exp 4"],
            competing_claims=[],
            key_metrics={},
        )
        assert claim_many.strength_score > claim_few.strength_score

    def test_risk_penalty(self):
        """Risk factors should decrease strength."""
        claim_safe = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.UNIFIED,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1"],
            competing_claims=[],
            key_metrics={},
            risk_factors=[],
        )
        claim_risky = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.UNIFIED,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["Exp 1"],
            competing_claims=[],
            key_metrics={},
            risk_factors=["Risk 1", "Risk 2", "Risk 3"],
        )
        assert claim_safe.strength_score > claim_risky.strength_score

    def test_max_strength_cap(self):
        """Strength should not exceed 1.0."""
        claim = ResearchClaim(
            claim_id="TEST",
            title="Test",
            statement="Test",
            novelty_level=NoveltyLevel.FIRST,
            contribution_type=ContributionType.METHOD,
            supporting_experiments=["E1", "E2", "E3", "E4", "E5", "E6"],
            competing_claims=[],
            key_metrics={},
        )
        assert claim.strength_score <= 1.0


# =============================================================================
# Claim Creator Tests
# =============================================================================


class TestClaimCreators:
    """Tests for individual claim creator functions."""

    def test_adaptive_evidence_claim(self):
        """Should create valid adaptive evidence claim."""
        claim = create_adaptive_evidence_claim()
        assert claim.claim_id == "CLAIM_01"
        assert "Adaptive" in claim.title
        assert claim.novelty_level == NoveltyLevel.UNIFIED
        assert len(claim.supporting_experiments) >= 4
        assert "improvement_over_oneshot" in claim.key_metrics

    def test_symbolic_repair_claim(self):
        """Should create valid symbolic repair claim."""
        claim = create_symbolic_repair_claim()
        assert claim.claim_id == "CLAIM_02"
        assert "Repair" in claim.title
        assert claim.novelty_level == NoveltyLevel.FIRST  # Strongest novelty
        assert len(claim.supporting_experiments) >= 3
        assert "repair_contribution" in claim.key_metrics

    def test_uncertainty_claim(self):
        """Should create valid uncertainty claim."""
        claim = create_uncertainty_claim()
        assert claim.claim_id == "CLAIM_03"
        assert "Uncertainty" in claim.title
        assert claim.contribution_type == ContributionType.ANALYSIS
        assert "calibration_improvement" in claim.key_metrics

    def test_unified_policy_claim(self):
        """Should create valid unified policy claim."""
        claim = create_unified_policy_claim()
        assert claim.claim_id == "CLAIM_04"
        assert "Multi-Task" in claim.title or "Unified" in claim.title
        assert claim.contribution_type == ContributionType.FRAMEWORK
        assert "task_coverage" in claim.key_metrics

    def test_all_claims_returns_four(self):
        """Should return exactly 4 claims."""
        claims = create_all_claims()
        assert len(claims) == 4

    def test_all_claims_unique_ids(self):
        """All claims should have unique IDs."""
        claims = create_all_claims()
        ids = [c.claim_id for c in claims]
        assert len(ids) == len(set(ids))

    def test_symbolic_repair_strongest_novelty(self):
        """Symbolic repair should be our strongest novelty claim."""
        claims = create_all_claims()
        repair_claim = next(c for c in claims if c.claim_id == "CLAIM_02")

        # Should be FIRST novelty (strongest)
        assert repair_claim.novelty_level == NoveltyLevel.FIRST

        # Should have high strength score
        assert repair_claim.strength_score >= 0.9


# =============================================================================
# Competitor Tests
# =============================================================================


class TestCompetingMethod:
    """Tests for CompetingMethod dataclass."""

    def test_competitor_creation(self):
        """Should create competitor with all fields."""
        comp = CompetingMethod(
            name="Test Method",
            venue="CVPR 2025",
            year=2025,
            key_claims=["Claim 1"],
            limitations=["Limitation 1"],
            overlap_with_ours=0.5,
            differentiation="Our method differs by X",
        )
        assert comp.name == "Test Method"
        assert comp.overlap_with_ours == 0.5


class TestCompetitorCreators:
    """Tests for competitor creator functions."""

    def test_probe_and_ground_competitor(self):
        """Should create Probe-and-Ground competitor."""
        comp = create_probe_and_ground_competitor()
        assert "Probe" in comp.name
        assert comp.year == 2026
        assert comp.overlap_with_ours > 0.5  # Most similar competitor

    def test_3dgraphllm_competitor(self):
        """Should create 3DGraphLLM competitor."""
        comp = create_3dgraphllm_competitor()
        assert "3DGraphLLM" in comp.name
        assert "ICCV" in comp.venue
        assert any("detection" in lim.lower() for lim in comp.limitations)

    def test_leo_competitor(self):
        """Should create LEO competitor."""
        comp = create_leo_competitor()
        assert "LEO" in comp.name
        assert "ICML" in comp.venue
        assert any("training" in lim.lower() for lim in comp.limitations)

    def test_sg_nav_competitor(self):
        """Should create SG-Nav competitor."""
        comp = create_sg_nav_competitor()
        assert "Nav" in comp.name
        assert "NeurIPS" in comp.venue

    def test_all_competitors_returns_four(self):
        """Should return exactly 4 competitors."""
        competitors = create_all_competitors()
        assert len(competitors) == 4

    def test_overlap_ordering(self):
        """Probe-and-Ground should have highest overlap."""
        competitors = create_all_competitors()
        pag = next(c for c in competitors if "Probe" in c.name)
        max_overlap = max(c.overlap_with_ours for c in competitors)
        assert pag.overlap_with_ours == max_overlap


# =============================================================================
# Strategy Tests
# =============================================================================


class TestPublicationStrategy:
    """Tests for PublicationStrategy dataclass."""

    def test_strategy_creation(self):
        """Should create strategy with all fields."""
        strategy = PublicationStrategy(
            primary_venue=PublicationVenue.CVPR,
            backup_venues=[PublicationVenue.ECCV],
            submission_deadline="Nov 2026",
            positioning_angle="Test angle",
            reviewer_concerns=["Concern 1"],
            rebuttal_preparation=["Prep 1"],
        )
        assert strategy.primary_venue == PublicationVenue.CVPR


class TestStrategyCreators:
    """Tests for strategy creator functions."""

    def test_cvpr_strategy(self):
        """Should create valid CVPR strategy."""
        strategy = create_cvpr_strategy()
        assert strategy.primary_venue == PublicationVenue.CVPR
        assert PublicationVenue.ECCV in strategy.backup_venues
        assert len(strategy.reviewer_concerns) >= 3
        assert len(strategy.rebuttal_preparation) >= 3

    def test_neurips_strategy(self):
        """Should create valid NeurIPS strategy."""
        strategy = create_neurips_strategy()
        assert strategy.primary_venue == PublicationVenue.NEURIPS
        assert PublicationVenue.ICLR in strategy.backup_venues
        assert "algorithmic" in strategy.positioning_angle.lower()

    def test_cvpr_strategy_visual_focus(self):
        """CVPR strategy should emphasize visual/grounding aspects."""
        strategy = create_cvpr_strategy()
        angle_lower = strategy.positioning_angle.lower()
        assert "visual" in angle_lower or "grounding" in angle_lower

    def test_neurips_strategy_algorithmic_focus(self):
        """NeurIPS strategy should emphasize algorithmic aspects."""
        strategy = create_neurips_strategy()
        angle_lower = strategy.positioning_angle.lower()
        assert "policy" in angle_lower or "algorithmic" in angle_lower


# =============================================================================
# Document Generation Tests
# =============================================================================


class TestDocumentGeneration:
    """Tests for LaTeX document generation functions."""

    def test_contribution_summary(self):
        """Should generate valid contribution summary."""
        claims = create_all_claims()
        summary = generate_contribution_summary(claims)

        assert "Core Contributions" in summary
        assert "\\begin{enumerate}" in summary
        assert "Adaptive Evidence" in summary
        assert "14.3" in summary  # Expected improvement number

    def test_positioning_header(self):
        """Should generate valid header with timestamp."""
        header = generate_positioning_header()

        assert "Auto-generated" in header
        assert "2026" in header  # Current year
        assert "Target Venues" in header

    def test_claims_section(self):
        """Should generate all claim subsections."""
        claims = create_all_claims()
        section = generate_claims_section(claims)

        assert "\\section{Research Claims}" in section
        for i in range(1, 5):
            assert f"Claim {i}" in section
        assert "\\textbf{Statement:}" in section
        assert "\\textbf{Supporting Evidence:}" in section

    def test_competitor_section(self):
        """Should generate all competitor subsections."""
        competitors = create_all_competitors()
        section = generate_competitor_section(competitors)

        assert "\\section{Competitive Landscape}" in section
        for comp in competitors:
            assert comp.name in section
        assert "Their Claims:" in section
        assert "Their Limitations:" in section
        assert "Our Differentiation:" in section

    def test_strategy_section(self):
        """Should generate strategy section."""
        strategy = create_cvpr_strategy()
        section = generate_strategy_section(strategy)

        assert "\\section{Publication Strategy}" in section
        assert "CVPR" in section
        assert "Reviewer Concerns" in section
        assert "Rebuttal Preparation" in section

    def test_gap_analysis_section(self):
        """Should generate gap analysis."""
        claims = create_all_claims()
        competitors = create_all_competitors()
        gaps = generate_novelty_gap_analysis(claims, competitors)
        section = generate_gap_analysis_section(gaps)

        assert "\\section{Novelty Gap Analysis}" in section
        assert "Adaptive Evidence" in section
        assert "Symbolic" in section

    def test_action_items_section(self):
        """Should generate action items."""
        section = generate_action_items_section()

        assert "\\section{Action Items" in section
        assert "Before Submission" in section
        assert "Supplementary Material" in section
        assert "Video Demo" in section


class TestFullDocumentGeneration:
    """Tests for complete document generation."""

    def test_generate_cvpr_document(self):
        """Should generate complete CVPR document."""
        doc = generate_positioning_document(venue="cvpr")

        # Check all sections present
        assert "Academic Positioning" in doc
        assert "Research Claims" in doc
        assert "Competitive Landscape" in doc
        assert "Publication Strategy" in doc
        assert "Novelty Gap Analysis" in doc
        assert "Action Items" in doc

        # Check CVPR-specific content
        assert "CVPR" in doc

    def test_generate_neurips_document(self):
        """Should generate complete NeurIPS document."""
        doc = generate_positioning_document(venue="neurips")

        assert "NEURIPS" in doc or "NeurIPS" in doc

    def test_document_latex_validity(self):
        """Document should have balanced LaTeX commands."""
        doc = generate_positioning_document()

        # Check balanced environments
        begin_count = doc.count("\\begin{")
        end_count = doc.count("\\end{")
        assert begin_count == end_count

        # Check section commands exist
        assert doc.count("\\section{") >= 4


# =============================================================================
# Academic Positioning Tests
# =============================================================================


class TestAcademicPositioning:
    """Tests for AcademicPositioning dataclass."""

    def test_positioning_creation(self):
        """Should create positioning with all components."""
        positioning = create_academic_positioning()

        assert positioning.title
        assert positioning.tagline
        assert len(positioning.claims) == 4
        assert len(positioning.competitors) >= 3
        assert positioning.strategy is not None
        assert positioning.contribution_summary

    def test_overall_strength(self):
        """Should compute reasonable overall strength."""
        positioning = create_academic_positioning()

        assert 0.6 < positioning.overall_strength < 1.0

    def test_empty_claims_strength(self):
        """Empty claims should give 0 strength."""
        positioning = AcademicPositioning(
            title="Test",
            tagline="Test",
            claims=[],
            competitors=[],
            strategy=create_cvpr_strategy(),
            contribution_summary="Test",
            novelty_gap_analysis={},
        )
        assert positioning.overall_strength == 0.0

    def test_generated_timestamp(self):
        """Should have valid timestamp."""
        positioning = create_academic_positioning()

        # Should be recent timestamp (within last minute)
        timestamp = datetime.fromisoformat(positioning.generated_at)
        now = datetime.now()
        delta = (now - timestamp).total_seconds()
        assert delta < 60


# =============================================================================
# Summary Generation Tests
# =============================================================================


class TestPositioningSummary:
    """Tests for positioning summary generation."""

    def test_summary_structure(self):
        """Summary should have expected structure."""
        summary = create_positioning_summary()

        assert "title" in summary
        assert "tagline" in summary
        assert "overall_strength" in summary
        assert "claims" in summary
        assert "competitors" in summary
        assert "primary_venue" in summary
        assert "generated_at" in summary

    def test_summary_claims(self):
        """Summary should have all 4 claims."""
        summary = create_positioning_summary()

        assert len(summary["claims"]) == 4
        for claim in summary["claims"]:
            assert "id" in claim
            assert "title" in claim
            assert "novelty" in claim
            assert "strength" in claim

    def test_summary_competitors(self):
        """Summary should have competitor info."""
        summary = create_positioning_summary()

        assert len(summary["competitors"]) >= 3
        for comp in summary["competitors"]:
            assert "name" in comp
            assert "venue" in comp
            assert "overlap" in comp

    def test_summary_json_serializable(self):
        """Summary should be JSON serializable."""
        summary = create_positioning_summary()
        serialized = json.dumps(summary)
        deserialized = json.loads(serialized)

        assert deserialized["title"] == summary["title"]


# =============================================================================
# File Save Tests
# =============================================================================


class TestFileSave:
    """Tests for file saving functionality."""

    def test_save_to_directory(self, tmp_path: Path):
        """Should save document to directory with auto-generated name."""
        output = save_positioning_document(tmp_path, venue="cvpr")

        assert output.exists()
        assert output.suffix == ".tex"
        assert "cvpr" in output.name

    def test_save_to_file(self, tmp_path: Path):
        """Should save document to specific file."""
        output_file = tmp_path / "test_position.tex"
        output = save_positioning_document(output_file, venue="neurips")

        assert output == output_file
        assert output.exists()

    def test_saved_content(self, tmp_path: Path):
        """Saved file should contain valid document."""
        output = save_positioning_document(tmp_path, venue="cvpr")
        content = output.read_text()

        assert "Academic Positioning" in content
        assert "Research Claims" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Should create parent directories if needed."""
        output_path = tmp_path / "nested" / "dirs" / "position.tex"
        output = save_positioning_document(output_path)

        assert output.exists()
        assert output.parent.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestAcademicPositioningIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, tmp_path: Path):
        """Should complete full workflow successfully."""
        # 1. Create positioning
        positioning = create_academic_positioning(venue="cvpr")

        # 2. Generate document
        doc = generate_positioning_document(venue="cvpr")

        # 3. Save document
        output = save_positioning_document(tmp_path, venue="cvpr")

        # 4. Create summary
        summary = create_positioning_summary()

        # Verify all steps succeeded
        assert positioning.overall_strength > 0.6
        assert len(doc) > 5000  # Substantial document
        assert output.exists()
        assert summary["overall_strength"] > 0.6

    def test_claim_coverage_in_document(self):
        """All claims should appear in generated document."""
        claims = create_all_claims()
        doc = generate_positioning_document()

        for claim in claims:
            assert claim.title in doc

    def test_competitor_coverage_in_document(self):
        """All competitors should appear in generated document."""
        competitors = create_all_competitors()
        doc = generate_positioning_document()

        for comp in competitors:
            assert comp.name in doc

    def test_claim_metrics_in_document(self):
        """Key metrics should appear in document."""
        doc = generate_positioning_document()

        # Check specific metrics appear
        assert "14.3" in doc  # Overall improvement
        assert "8.0" in doc or "8%" in doc  # Detection drop advantage
        assert "6.2" in doc  # Repair contribution


class TestAcademicAlignment:
    """Tests for alignment with academic innovation points."""

    def test_adaptive_evidence_alignment(self):
        """Claim should align with adaptive evidence acquisition point."""
        claim = create_adaptive_evidence_claim()

        # Should mention dynamic/adaptive behavior
        assert (
            "dynamic" in claim.statement.lower()
            or "adaptive" in claim.statement.lower()
        )

    def test_symbolic_repair_alignment(self):
        """Claim should align with symbolic-to-visual repair point."""
        claim = create_symbolic_repair_claim()

        # Should mention validation/correction
        assert (
            "validate" in claim.statement.lower()
            or "correct" in claim.statement.lower()
        )

    def test_uncertainty_alignment(self):
        """Claim should align with evidence-grounded uncertainty point."""
        claim = create_uncertainty_claim()

        # Should mention calibration/hallucination
        assert (
            "calibrat" in claim.statement.lower()
            or "hallucin" in claim.statement.lower()
        )

    def test_unified_policy_alignment(self):
        """Claim should align with unified multi-task policy point."""
        claim = create_unified_policy_claim()

        # Should mention multiple task types
        statement_lower = claim.statement.lower()
        assert "qa" in statement_lower or "grounding" in statement_lower

    def test_core_research_claim_present(self):
        """The core research claim should be supported by positioning."""
        doc = generate_positioning_document()
        core_claim = (
            "evidence-seeking VLM agents that iteratively acquire visual context "
            "outperform one-shot baselines"
        ).lower()

        # Key phrases from core claim should appear
        doc_lower = doc.lower()
        assert "evidence-seeking" in doc_lower or "evidence seeking" in doc_lower
        assert "one-shot" in doc_lower or "oneshot" in doc_lower
