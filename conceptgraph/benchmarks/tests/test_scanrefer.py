"""Unit tests for ScanRefer benchmark loader."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conceptgraph.benchmarks.scanrefer_loader import (
    BoundingBox3D,
    ScanReferDataset,
    ScanReferEvaluationResult,
    ScanReferSample,
    compute_iou_3d,
    compute_scanrefer_metrics,
    compute_scanrefer_metrics_by_category,
    download_scanrefer,
    evaluate_scanrefer,
)


class TestBoundingBox3D:
    """Tests for BoundingBox3D dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic bounding box creation."""
        bbox = BoundingBox3D(
            center=[1.0, 2.0, 3.0],
            size=[2.0, 4.0, 6.0],
        )

        assert bbox.center == [1.0, 2.0, 3.0]
        assert bbox.size == [2.0, 4.0, 6.0]
        assert bbox.orientation is None

    def test_with_orientation(self) -> None:
        """Test bounding box with orientation."""
        bbox = BoundingBox3D(
            center=[0.0, 0.0, 0.0],
            size=[1.0, 1.0, 1.0],
            orientation=[0.0, 0.0, 1.57],
        )

        assert bbox.orientation == [0.0, 0.0, 1.57]

    def test_min_corner(self) -> None:
        """Test min corner calculation."""
        bbox = BoundingBox3D(center=[5.0, 5.0, 5.0], size=[2.0, 4.0, 6.0])

        expected = np.array([4.0, 3.0, 2.0])
        np.testing.assert_array_almost_equal(bbox.min_corner, expected)

    def test_max_corner(self) -> None:
        """Test max corner calculation."""
        bbox = BoundingBox3D(center=[5.0, 5.0, 5.0], size=[2.0, 4.0, 6.0])

        expected = np.array([6.0, 7.0, 8.0])
        np.testing.assert_array_almost_equal(bbox.max_corner, expected)

    def test_volume(self) -> None:
        """Test volume calculation."""
        bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[2.0, 3.0, 4.0])

        assert bbox.volume() == pytest.approx(24.0)

    def test_volume_unit_cube(self) -> None:
        """Test unit cube volume."""
        bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        assert bbox.volume() == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        bbox = BoundingBox3D(center=[1.0, 2.0, 3.0], size=[4.0, 5.0, 6.0])

        result = bbox.to_dict()
        assert result == {"center": [1.0, 2.0, 3.0], "size": [4.0, 5.0, 6.0]}

    def test_to_dict_with_orientation(self) -> None:
        """Test conversion to dictionary with orientation."""
        bbox = BoundingBox3D(
            center=[0.0, 0.0, 0.0],
            size=[1.0, 1.0, 1.0],
            orientation=[0.0, 0.0, 0.0],
        )

        result = bbox.to_dict()
        assert "orientation" in result
        assert result["orientation"] == [0.0, 0.0, 0.0]


class TestComputeIoU3D:
    """Tests for 3D IoU computation."""

    def test_identical_boxes(self) -> None:
        """Test IoU of identical boxes."""
        bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[2.0, 2.0, 2.0])

        iou = compute_iou_3d(bbox, bbox)
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Test IoU of non-overlapping boxes."""
        bbox1 = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])
        bbox2 = BoundingBox3D(center=[10.0, 10.0, 10.0], size=[1.0, 1.0, 1.0])

        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Test IoU of partially overlapping boxes."""
        # Two unit cubes with centers 0.5 apart
        bbox1 = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])
        bbox2 = BoundingBox3D(center=[0.5, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        # Intersection: 0.5 x 1 x 1 = 0.5
        # Union: 1 + 1 - 0.5 = 1.5
        # IoU: 0.5 / 1.5 = 0.333...
        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(1.0 / 3.0)

    def test_contained_box(self) -> None:
        """Test IoU when one box contains the other."""
        bbox1 = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[4.0, 4.0, 4.0])
        bbox2 = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[2.0, 2.0, 2.0])

        # Intersection = smaller box volume = 8
        # Union = larger box volume = 64
        # IoU = 8 / 64 = 0.125
        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(0.125)

    def test_touching_boxes(self) -> None:
        """Test IoU of boxes that just touch (edge contact)."""
        bbox1 = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])
        bbox2 = BoundingBox3D(center=[1.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        # Boxes touch at x=0.5, no volume overlap
        iou = compute_iou_3d(bbox1, bbox2)
        assert iou == pytest.approx(0.0)


class TestScanReferSample:
    """Tests for ScanReferSample dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic sample creation."""
        bbox = BoundingBox3D(center=[1.0, 2.0, 0.5], size=[0.6, 0.6, 0.8])

        sample = ScanReferSample(
            sample_id="scene0000_00_chair_001",
            scene_id="scene0000_00",
            object_id="15",
            object_name="chair",
            description="the brown wooden chair near the window",
            target_bbox=bbox,
        )

        assert sample.sample_id == "scene0000_00_chair_001"
        assert sample.scene_id == "scene0000_00"
        assert sample.object_id == "15"
        assert sample.object_name == "chair"
        assert sample.description == "the brown wooden chair near the window"
        assert sample.ann_id == ""
        assert sample.token == []

    def test_with_optional_fields(self) -> None:
        """Test sample with optional fields."""
        bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        sample = ScanReferSample(
            sample_id="test_001",
            scene_id="scene0001_00",
            object_id="1",
            object_name="table",
            description="the table",
            target_bbox=bbox,
            ann_id="ann_001",
            token=["the", "table"],
        )

        assert sample.ann_id == "ann_001"
        assert sample.token == ["the", "table"]

    def test_query_property(self) -> None:
        """Test query property returns description."""
        bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        sample = ScanReferSample(
            sample_id="test",
            scene_id="scene0000_00",
            object_id="1",
            object_name="lamp",
            description="the tall lamp by the sofa",
            target_bbox=bbox,
        )

        assert sample.query == "the tall lamp by the sofa"


class TestScanReferDataset:
    """Tests for ScanReferDataset class."""

    @pytest.fixture
    def sample_data(self) -> List[dict]:
        """Create sample ScanRefer JSON data."""
        return [
            {
                "scene_id": "scene0000_00",
                "object_id": "1",
                "object_name": "chair",
                "ann_id": "0",
                "description": "a wooden chair near the window",
                "object_bbox": [1.0, 2.0, 0.5, 0.6, 0.6, 0.8],
            },
            {
                "scene_id": "scene0000_00",
                "object_id": "2",
                "object_name": "table",
                "ann_id": "1",
                "description": "the large table in the center",
                "object_bbox": [0.0, 0.0, 0.0, 2.0, 1.5, 0.75],
            },
            {
                "scene_id": "scene0011_00",
                "object_id": "5",
                "object_name": "chair",
                "ann_id": "2",
                "description": "the red chair",
                "object_bbox": [-1.0, 3.0, 0.4, 0.5, 0.5, 0.9],
            },
            {
                "scene_id": "scene0011_00",
                "object_id": "10",
                "object_name": "lamp",
                "ann_id": "3",
                "description": "a floor lamp in the corner",
                "object_bbox": [2.0, 2.0, 0.7, 0.3, 0.3, 1.5],
            },
        ]

    @pytest.fixture
    def temp_dataset_dir(self, sample_data: List[dict]) -> Path:
        """Create temporary directory with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "ScanRefer_filtered_val.json"
            with open(data_path, "w") as f:
                json.dump(sample_data, f)
            yield Path(tmpdir)

    def test_from_path(self, temp_dataset_dir: Path) -> None:
        """Test loading dataset from path."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        assert len(dataset) == 4
        assert dataset.split == "val"

    def test_from_path_with_filter_scene(self, temp_dataset_dir: Path) -> None:
        """Test filtering by scene ID."""
        dataset = ScanReferDataset.from_path(
            temp_dataset_dir, split="val", scene_id="scene0000_00"
        )

        assert len(dataset) == 2
        assert all(s.scene_id == "scene0000_00" for s in dataset)

    def test_from_path_with_filter_object(self, temp_dataset_dir: Path) -> None:
        """Test filtering by object name."""
        dataset = ScanReferDataset.from_path(
            temp_dataset_dir, split="val", object_name="chair"
        )

        assert len(dataset) == 2
        assert all(s.object_name == "chair" for s in dataset)

    def test_from_path_with_max_samples(self, temp_dataset_dir: Path) -> None:
        """Test limiting number of samples."""
        dataset = ScanReferDataset.from_path(
            temp_dataset_dir, split="val", max_samples=2
        )

        assert len(dataset) == 2

    def test_from_path_not_found(self) -> None:
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError, match="ScanRefer data file not found"):
            ScanReferDataset.from_path("/nonexistent/path")

    def test_iteration(self, temp_dataset_dir: Path) -> None:
        """Test dataset iteration."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        samples = list(dataset)
        assert len(samples) == 4
        assert all(isinstance(s, ScanReferSample) for s in samples)

    def test_getitem(self, temp_dataset_dir: Path) -> None:
        """Test indexing into dataset."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        sample = dataset[0]
        assert isinstance(sample, ScanReferSample)
        assert sample.scene_id == "scene0000_00"

    def test_get_scenes(self, temp_dataset_dir: Path) -> None:
        """Test getting unique scene IDs."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        scenes = dataset.get_scenes()
        assert scenes == ["scene0000_00", "scene0011_00"]

    def test_get_object_names(self, temp_dataset_dir: Path) -> None:
        """Test getting unique object names."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        objects = dataset.get_object_names()
        assert set(objects) == {"chair", "table", "lamp"}

    def test_filter_by_scene(self, temp_dataset_dir: Path) -> None:
        """Test filtering dataset by scene."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")
        filtered = dataset.filter_by_scene("scene0011_00")

        assert len(filtered) == 2
        assert all(s.scene_id == "scene0011_00" for s in filtered)

    def test_filter_by_object(self, temp_dataset_dir: Path) -> None:
        """Test filtering dataset by object."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")
        filtered = dataset.filter_by_object("chair")

        assert len(filtered) == 2
        assert all(s.object_name == "chair" for s in filtered)

    def test_bbox_parsing(self, temp_dataset_dir: Path) -> None:
        """Test that bounding boxes are parsed correctly."""
        dataset = ScanReferDataset.from_path(temp_dataset_dir, split="val")

        sample = dataset[0]
        assert sample.target_bbox.center == [1.0, 2.0, 0.5]
        assert sample.target_bbox.size == [0.6, 0.6, 0.8]


class TestEvaluateScanRefer:
    """Tests for ScanRefer evaluation."""

    def create_sample(
        self, sample_id: str, center: List[float], size: List[float]
    ) -> ScanReferSample:
        """Helper to create samples."""
        return ScanReferSample(
            sample_id=sample_id,
            scene_id="scene0000_00",
            object_id="1",
            object_name="chair",
            description="a chair",
            target_bbox=BoundingBox3D(center=center, size=size),
        )

    def test_perfect_prediction(self) -> None:
        """Test evaluation with perfect prediction."""
        sample = self.create_sample("s1", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        pred_bbox = BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        results = evaluate_scanrefer([(sample, pred_bbox)])

        assert len(results) == 1
        assert results[0].iou == pytest.approx(1.0)
        assert results[0].acc_at_025 is True
        assert results[0].acc_at_050 is True

    def test_no_prediction(self) -> None:
        """Test evaluation with no prediction (None)."""
        sample = self.create_sample("s1", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        results = evaluate_scanrefer([(sample, None)])

        assert len(results) == 1
        assert results[0].iou == pytest.approx(0.0)
        assert results[0].acc_at_025 is False
        assert results[0].acc_at_050 is False

    def test_partial_overlap_above_025(self) -> None:
        """Test evaluation with IoU above 0.25 but below 0.5."""
        sample = self.create_sample("s1", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        # Shift prediction to get IoU around 0.33
        pred_bbox = BoundingBox3D(center=[0.5, 0.0, 0.0], size=[1.0, 1.0, 1.0])

        results = evaluate_scanrefer([(sample, pred_bbox)])

        assert len(results) == 1
        assert 0.25 <= results[0].iou < 0.5
        assert results[0].acc_at_025 is True
        assert results[0].acc_at_050 is False

    def test_multiple_predictions(self) -> None:
        """Test evaluation with multiple samples."""
        samples = [
            self.create_sample("s1", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            self.create_sample("s2", [5.0, 5.0, 5.0], [2.0, 2.0, 2.0]),
            self.create_sample("s3", [10.0, 10.0, 10.0], [1.0, 1.0, 1.0]),
        ]
        predictions = [
            BoundingBox3D(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0]),  # Perfect
            BoundingBox3D(center=[5.5, 5.0, 5.0], size=[2.0, 2.0, 2.0]),  # Good
            None,  # No prediction
        ]

        results = evaluate_scanrefer(list(zip(samples, predictions)))

        assert len(results) == 3
        assert results[0].acc_at_050 is True
        assert results[1].acc_at_025 is True  # Shifted, should still work
        assert results[2].iou == pytest.approx(0.0)


class TestComputeScanReferMetrics:
    """Tests for aggregate metrics computation."""

    def test_empty_results(self) -> None:
        """Test with empty results."""
        metrics = compute_scanrefer_metrics([])

        assert metrics["accuracy_at_025"] == 0.0
        assert metrics["accuracy_at_050"] == 0.0
        assert metrics["mean_iou"] == 0.0
        assert metrics["total"] == 0

    def test_all_perfect(self) -> None:
        """Test with all perfect predictions."""
        results = [
            ScanReferEvaluationResult(
                sample_id=f"s{i}",
                scene_id="scene0000_00",
                object_id="1",
                description="test",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                iou=1.0,
                acc_at_025=True,
                acc_at_050=True,
            )
            for i in range(10)
        ]

        metrics = compute_scanrefer_metrics(results)

        assert metrics["accuracy_at_025"] == pytest.approx(1.0)
        assert metrics["accuracy_at_050"] == pytest.approx(1.0)
        assert metrics["mean_iou"] == pytest.approx(1.0)
        assert metrics["total"] == 10

    def test_mixed_results(self) -> None:
        """Test with mixed results."""
        results = [
            ScanReferEvaluationResult(
                sample_id="s1",
                scene_id="scene0000_00",
                object_id="1",
                description="test",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                iou=1.0,
                acc_at_025=True,
                acc_at_050=True,
            ),
            ScanReferEvaluationResult(
                sample_id="s2",
                scene_id="scene0000_00",
                object_id="2",
                description="test",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=BoundingBox3D([0.5, 0, 0], [1, 1, 1]),
                iou=0.33,
                acc_at_025=True,
                acc_at_050=False,
            ),
            ScanReferEvaluationResult(
                sample_id="s3",
                scene_id="scene0000_00",
                object_id="3",
                description="test",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=None,
                iou=0.0,
                acc_at_025=False,
                acc_at_050=False,
            ),
        ]

        metrics = compute_scanrefer_metrics(results)

        assert metrics["accuracy_at_025"] == pytest.approx(2 / 3)
        assert metrics["accuracy_at_050"] == pytest.approx(1 / 3)
        assert metrics["mean_iou"] == pytest.approx((1.0 + 0.33 + 0.0) / 3)
        assert metrics["total"] == 3


class TestComputeMetricsByCategory:
    """Tests for per-category metrics computation."""

    def test_metrics_by_category(self) -> None:
        """Test computing metrics grouped by object category."""
        samples = [
            ScanReferSample(
                sample_id="s1",
                scene_id="scene0000_00",
                object_id="1",
                object_name="chair",
                description="a chair",
                target_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
            ),
            ScanReferSample(
                sample_id="s2",
                scene_id="scene0000_00",
                object_id="2",
                object_name="chair",
                description="another chair",
                target_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
            ),
            ScanReferSample(
                sample_id="s3",
                scene_id="scene0000_00",
                object_id="3",
                object_name="table",
                description="a table",
                target_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
            ),
        ]

        results = [
            ScanReferEvaluationResult(
                sample_id="s1",
                scene_id="scene0000_00",
                object_id="1",
                description="a chair",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                iou=1.0,
                acc_at_025=True,
                acc_at_050=True,
            ),
            ScanReferEvaluationResult(
                sample_id="s2",
                scene_id="scene0000_00",
                object_id="2",
                description="another chair",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=None,
                iou=0.0,
                acc_at_025=False,
                acc_at_050=False,
            ),
            ScanReferEvaluationResult(
                sample_id="s3",
                scene_id="scene0000_00",
                object_id="3",
                description="a table",
                ground_truth_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                predicted_bbox=BoundingBox3D([0, 0, 0], [1, 1, 1]),
                iou=1.0,
                acc_at_025=True,
                acc_at_050=True,
            ),
        ]

        category_metrics = compute_scanrefer_metrics_by_category(results, samples)

        assert "chair" in category_metrics
        assert "table" in category_metrics
        assert category_metrics["chair"]["total"] == 2
        assert category_metrics["chair"]["accuracy_at_050"] == pytest.approx(0.5)
        assert category_metrics["table"]["total"] == 1
        assert category_metrics["table"]["accuracy_at_050"] == pytest.approx(1.0)


class TestDownloadScanRefer:
    """Tests for download function."""

    def test_download_already_exists(self) -> None:
        """Test download when directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "ScanRefer"
            repo_dir.mkdir()

            result = download_scanrefer(tmpdir)

            assert result == repo_dir

    @patch("subprocess.run")
    def test_download_clones_repo(self, mock_run: MagicMock) -> None:
        """Test that download clones the repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_scanrefer(tmpdir)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "git" in call_args
            assert "clone" in call_args
            assert "daveredrum/ScanRefer" in call_args[4]

    @patch("subprocess.run")
    def test_download_with_scannet_warning(self, mock_run: MagicMock) -> None:
        """Test warning when ScanNet download requested."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise, just log warning
            download_scanrefer(tmpdir, include_scannet=True)


class TestAlternativeDataPaths:
    """Tests for alternative data file locations."""

    def test_data_subdir_path(self) -> None:
        """Test loading from data/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            data_file = data_dir / "ScanRefer_filtered_val.json"
            with open(data_file, "w") as f:
                json.dump(
                    [
                        {
                            "scene_id": "scene0000_00",
                            "object_id": "1",
                            "object_name": "chair",
                            "ann_id": "0",
                            "description": "test",
                            "object_bbox": [0, 0, 0, 1, 1, 1],
                        }
                    ],
                    f,
                )

            dataset = ScanReferDataset.from_path(tmpdir, split="val")
            assert len(dataset) == 1

    def test_scanrefer_subdir_path(self) -> None:
        """Test loading from ScanRefer/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "ScanRefer"
            data_dir.mkdir()

            data_file = data_dir / "ScanRefer_filtered_val.json"
            with open(data_file, "w") as f:
                json.dump(
                    [
                        {
                            "scene_id": "scene0000_00",
                            "object_id": "1",
                            "object_name": "chair",
                            "ann_id": "0",
                            "description": "test",
                            "object_bbox": [0, 0, 0, 1, 1, 1],
                        }
                    ],
                    f,
                )

            dataset = ScanReferDataset.from_path(tmpdir, split="val")
            assert len(dataset) == 1
