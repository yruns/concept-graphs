"""Tests for run_scanrefer_stage1_only script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from conceptgraph.benchmarks.scanrefer_loader import BoundingBox3D
from conceptgraph.evaluation.scripts.run_scanrefer_stage1_only import (
    MockScanReferSample,
    create_mock_scanrefer_samples,
    create_mock_stage1_factory,
    run_scanrefer_stage1_only,
    _compute_hypothesis_distribution,
)
from conceptgraph.evaluation.batch_eval import EvalRunResult, EvalSampleResult


class TestMockScanReferSamples:
    """Tests for mock sample generation."""

    def test_create_mock_samples(self):
        """Test mock sample creation."""
        samples = create_mock_scanrefer_samples(10)
        assert len(samples) == 10
        assert all(isinstance(s, MockScanReferSample) for s in samples)

    def test_mock_sample_fields(self):
        """Test mock samples have all required fields."""
        samples = create_mock_scanrefer_samples(1)
        sample = samples[0]
        assert sample.sample_id
        assert sample.scene_id
        assert sample.object_id
        assert sample.object_name
        assert sample.description
        assert sample.target_bbox is not None

    def test_mock_sample_bounding_box(self):
        """Test mock samples have valid 3D bounding boxes."""
        samples = create_mock_scanrefer_samples(5)
        for sample in samples:
            bbox = sample.target_bbox
            assert isinstance(bbox, BoundingBox3D)
            assert len(bbox.center) == 3
            assert len(bbox.size) == 3
            assert all(s > 0 for s in bbox.size), "Bbox size should be positive"

    def test_mock_sample_diversity(self):
        """Test mock samples have diverse object categories."""
        samples = create_mock_scanrefer_samples(50)
        categories = set(s.object_name for s in samples)
        assert len(categories) > 5, "Should have diverse object categories"

    def test_mock_sample_scenes(self):
        """Test mock samples cover multiple scenes."""
        samples = create_mock_scanrefer_samples(50)
        scenes = set(s.scene_id for s in samples)
        assert len(scenes) > 1, "Should cover multiple scenes"

    def test_mock_sample_query_property(self):
        """Test query property returns description."""
        samples = create_mock_scanrefer_samples(5)
        for sample in samples:
            assert sample.query == sample.description

    def test_mock_sample_referring_expressions(self):
        """Test referring expressions are varied and descriptive."""
        samples = create_mock_scanrefer_samples(20)
        descriptions = [s.description for s in samples]
        unique_descriptions = set(descriptions)
        # Should have some variation (not all unique due to template reuse)
        assert len(unique_descriptions) >= 5, "Should have varied descriptions"


class TestMockStage1Factory:
    """Tests for mock Stage 1 factory."""

    def test_factory_creates_selector(self):
        """Test factory creates mock selector."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0000_00")
        assert selector is not None
        assert hasattr(selector, "select_keyframes_v2")

    def test_selector_returns_keyframes(self):
        """Test mock selector returns keyframes."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0011_00")
        result = selector.select_keyframes_v2("the chair next to the table", k=3)
        assert len(result.keyframe_paths) == 3
        assert all(isinstance(p, Path) for p in result.keyframe_paths)

    def test_selector_includes_metadata(self):
        """Test mock selector includes visual grounding metadata."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0025_00")
        result = selector.select_keyframes_v2("the brown chair near the wall", k=3)
        metadata = result.metadata
        assert "selected_hypothesis_kind" in metadata
        assert "query" in metadata
        assert "scene_id" in metadata
        assert "task_type" in metadata
        assert metadata["task_type"] == "visual_grounding"

    def test_selector_varies_hypothesis_kind(self):
        """Test mock selector produces varied hypothesis kinds."""
        factory = create_mock_stage1_factory()
        selector = factory("scene0050_00")
        kinds = set()
        for i in range(10):
            result = selector.select_keyframes_v2(f"Query {i}", k=3)
            kinds.add(result.metadata["selected_hypothesis_kind"])
        assert len(kinds) > 1, "Should produce varied hypothesis kinds"


class TestBoundingBox3D:
    """Tests for 3D bounding box in mock samples."""

    def test_bbox_center_range(self):
        """Test bounding box centers are in reasonable range."""
        samples = create_mock_scanrefer_samples(50)
        for sample in samples:
            center = sample.target_bbox.center
            # Typical room-scale coordinates
            assert -10 <= center[0] <= 10, f"X center {center[0]} out of range"
            assert -10 <= center[1] <= 10, f"Y center {center[1]} out of range"
            assert 0 <= center[2] <= 3, f"Z center {center[2]} out of range"

    def test_bbox_size_by_category(self):
        """Test bounding box sizes vary by object category."""
        samples = create_mock_scanrefer_samples(50)
        sizes_by_cat = {}
        for sample in samples:
            cat = sample.object_name
            vol = sample.target_bbox.volume()
            if cat not in sizes_by_cat:
                sizes_by_cat[cat] = []
            sizes_by_cat[cat].append(vol)

        # Check that different categories have different average sizes
        if len(sizes_by_cat) >= 2:
            avg_sizes = {cat: sum(vols) / len(vols) for cat, vols in sizes_by_cat.items()}
            # At least some variation expected
            all_same = all(v == list(avg_sizes.values())[0] for v in avg_sizes.values())
            assert not all_same or len(avg_sizes) == 1, "Different categories should have different sizes"


class TestRunScanReferStage1OnlyBaseline:
    """Tests for the main baseline evaluation function."""

    def test_run_with_mock_data(self):
        """Test running evaluation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=5,
                max_workers=2,
                use_mock=True,
                verbose=False,
            )

            assert result.total_samples == 5
            assert result.failed_stage1 == 0  # Mock should succeed
            assert output_path.exists()

    def test_output_file_structure(self):
        """Test output file has correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Check required top-level keys
            assert "experiment" in data
            assert data["experiment"] == "stage1_only_baseline"
            assert "benchmark" in data
            assert data["benchmark"] == "scanrefer"
            assert "task_type" in data
            assert data["task_type"] == "visual_grounding"
            assert "summary" in data
            assert "hypothesis_distribution" in data
            assert "per_sample_results" in data
            assert "academic_notes" in data

    def test_summary_statistics(self):
        """Test summary statistics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=10,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            summary = data["summary"]
            assert summary["total_samples"] == 10
            assert summary["stage1_success"] == 10  # Mock always succeeds
            assert summary["stage1_failure"] == 0
            assert summary["avg_stage1_latency_ms"] > 0

    def test_stage2_disabled(self):
        """Test that Stage 2 is disabled in the baseline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            result = run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["config"]["stage2_enabled"] is False
            assert data["config"]["ablation_tag"] == "stage1_only"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_hypothesis_distribution(self):
        """Test hypothesis distribution computation."""
        results = [
            EvalSampleResult(
                sample_id=f"s{i}",
                query=f"Q{i}",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage1_hypothesis_kind=kind,
            )
            for i, kind in enumerate(["direct", "direct", "proxy", "context", "spatial"])
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        dist = _compute_hypothesis_distribution(run_result)
        assert dist["direct"] == 2
        assert dist["proxy"] == 1
        assert dist["context"] == 1
        assert dist["spatial"] == 1

    def test_compute_hypothesis_distribution_with_failures(self):
        """Test hypothesis distribution ignores failed Stage 1 results."""
        results = [
            EvalSampleResult(
                sample_id="s0",
                query="Q0",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=True,
                stage1_hypothesis_kind="direct",
            ),
            EvalSampleResult(
                sample_id="s1",
                query="Q1",
                task_type="visual_grounding",
                scene_id="scene",
                stage1_success=False,
                stage1_hypothesis_kind="proxy",
            ),
        ]
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            results=results,
        )

        dist = _compute_hypothesis_distribution(run_result)
        assert dist.get("direct") == 1
        assert "proxy" not in dist  # Failed, should be ignored


class TestAcademicAlignment:
    """Tests to verify academic alignment of the baseline."""

    def test_academic_notes_present(self):
        """Test academic notes are included in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            assert "purpose" in notes
            assert "ScanRefer" in notes["purpose"] or "grounding" in notes["purpose"].lower()

    def test_baseline_supports_comparison(self):
        """Test baseline output supports comparison with Stage 2 results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=3,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            # Per-sample results should have fields needed for comparison
            for sample in data["per_sample_results"]:
                assert "sample_id" in sample
                assert "query" in sample
                assert "stage1_success" in sample
                assert "stage1_hypothesis_kind" in sample
                assert "stage1_latency_ms" in sample

    def test_scanrefer_grounding_note(self):
        """Test ScanRefer-specific grounding note in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            run_scanrefer_stage1_only(
                output_path=output_path,
                max_samples=2,
                use_mock=True,
                verbose=False,
            )

            with open(output_path) as f:
                data = json.load(f)

            notes = data["academic_notes"]
            # Should mention that Stage 1 only does retrieval, not bbox prediction
            assert "grounding_note" in notes or "limitation" in notes
