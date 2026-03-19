"""Unit tests for SQA3D benchmark loader."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.benchmarks.sqa3d_loader import (
    SQA3DDataset,
    SQA3DEvaluationResult,
    SQA3DSample,
    SQA3DSituation,
    _infer_question_type,
    _normalize_answer,
    compute_sqa3d_metrics,
    download_sqa3d,
    evaluate_sqa3d,
)


class TestSQA3DSituation:
    """Tests for SQA3DSituation dataclass."""

    def test_situation_creation(self) -> None:
        """Test basic situation creation."""
        situation = SQA3DSituation(
            position=[1.0, 2.0, 3.0],
            orientation=[0.0, 0.0, 1.0],
            room_description="A living room with a sofa.",
            reference_objects=["sofa", "table"],
        )

        assert situation.position == [1.0, 2.0, 3.0]
        assert situation.orientation == [0.0, 0.0, 1.0]
        assert situation.room_description == "A living room with a sofa."
        assert situation.reference_objects == ["sofa", "table"]

    def test_situation_default_reference_objects(self) -> None:
        """Test default empty reference objects."""
        situation = SQA3DSituation(
            position=[0.0, 0.0, 0.0],
            orientation=[1.0, 0.0, 0.0],
            room_description="Empty room",
        )

        assert situation.reference_objects == []


class TestSQA3DSample:
    """Tests for SQA3DSample dataclass."""

    def test_sample_creation(self) -> None:
        """Test basic sample creation."""
        situation = SQA3DSituation(
            position=[1.0, 2.0, 0.5],
            orientation=[0.0, 1.0, 0.0],
            room_description="Office room",
        )

        sample = SQA3DSample(
            question_id="q001",
            question="What is on the desk?",
            answers=["laptop", "computer"],
            situation=situation,
            scene_id="scene0000_00",
            question_type="what",
            answer_type="single_word",
        )

        assert sample.question_id == "q001"
        assert sample.question == "What is on the desk?"
        assert sample.answers == ["laptop", "computer"]
        assert sample.scene_id == "scene0000_00"
        assert sample.question_type == "what"
        assert sample.answer_type == "single_word"
        assert sample.choices == []

    def test_primary_answer(self) -> None:
        """Test primary answer property."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What color?",
            answers=["red", "crimson", "scarlet"],
            situation=situation,
            scene_id="scene001",
            question_type="what_color",
            answer_type="single_word",
        )

        assert sample.primary_answer == "red"

    def test_primary_answer_empty(self) -> None:
        """Test primary answer with no answers."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What?",
            answers=[],
            situation=situation,
            scene_id="scene001",
            question_type="what",
            answer_type="single_word",
        )

        assert sample.primary_answer == ""

    def test_multiple_choice_sample(self) -> None:
        """Test multiple choice sample."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="Which object is closest?",
            answers=["chair"],
            situation=situation,
            scene_id="scene001",
            question_type="which",
            answer_type="multiple_choice",
            choices=["chair", "table", "lamp", "sofa"],
        )

        assert sample.answer_type == "multiple_choice"
        assert len(sample.choices) == 4


class TestInferQuestionType:
    """Tests for question type inference."""

    def test_what_question(self) -> None:
        """Test 'what' question type."""
        assert _infer_question_type("What is this?") == "what"

    def test_what_color(self) -> None:
        """Test 'what color' question type."""
        assert _infer_question_type("What color is the chair?") == "what_color"

    def test_what_shape(self) -> None:
        """Test 'what shape' question type."""
        assert _infer_question_type("What shape is the table?") == "what_shape"

    def test_what_material(self) -> None:
        """Test 'what material' question type."""
        assert _infer_question_type("What material is the floor?") == "what_material"

    def test_where_question(self) -> None:
        """Test 'where' question type."""
        assert _infer_question_type("Where is the lamp?") == "where"

    def test_how_many_question(self) -> None:
        """Test 'how many' question type."""
        assert _infer_question_type("How many chairs are there?") == "how_many"

    def test_yes_no_is(self) -> None:
        """Test yes/no question starting with 'is'."""
        assert _infer_question_type("Is there a window?") == "yes_no"

    def test_yes_no_are(self) -> None:
        """Test yes/no question starting with 'are'."""
        assert _infer_question_type("Are the chairs blue?") == "yes_no"

    def test_which_question(self) -> None:
        """Test 'which' question type."""
        assert _infer_question_type("Which object is largest?") == "which"

    def test_can_question(self) -> None:
        """Test 'can' question type."""
        assert _infer_question_type("Can you see the door?") == "can"

    def test_other_question(self) -> None:
        """Test fallback to 'other' type."""
        assert _infer_question_type("Please describe the room.") == "other"


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_basic_normalization(self) -> None:
        """Test basic case normalization."""
        assert _normalize_answer("RED") == "red"

    def test_strip_whitespace(self) -> None:
        """Test whitespace stripping."""
        assert _normalize_answer("  red  ") == "red"

    def test_remove_punctuation(self) -> None:
        """Test punctuation removal."""
        assert _normalize_answer("red!") == "red"
        assert _normalize_answer("red.") == "red"

    def test_normalize_whitespace(self) -> None:
        """Test internal whitespace normalization."""
        assert _normalize_answer("the   big    chair") == "the big chair"

    def test_combined_normalization(self) -> None:
        """Test combined normalization."""
        assert _normalize_answer("  The RED Chair!  ") == "the red chair"


class TestSQA3DDataset:
    """Tests for SQA3DDataset class."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a mock SQA3D dataset directory."""
        data_dir = tmp_path / "data" / "sqa_task"
        data_dir.mkdir(parents=True)

        # Create mock JSON data
        mock_data = [
            {
                "question_id": "q001",
                "question": "What color is the sofa?",
                "answers": ["blue", "navy"],
                "situation": {
                    "position": [1.0, 2.0, 0.5],
                    "orientation": [0.0, 1.0, 0.0],
                    "room_description": "Living room",
                    "reference_objects": ["sofa"],
                },
                "scene_id": "scene0001_00",
            },
            {
                "question_id": "q002",
                "question": "Where is the table?",
                "answer": "in the corner",  # Single answer format
                "situation": {
                    "position": [0.0, 0.0, 0.5],
                    "orientation": [1.0, 0.0, 0.0],
                    "room_description": "Dining area",
                },
                "scene_id": "scene0001_00",
            },
            {
                "question_id": "q003",
                "question": "How many chairs are there?",
                "answers": ["four", "4"],
                "situation": {
                    "position": [2.0, 2.0, 0.5],
                    "orientation": [0.0, -1.0, 0.0],
                    "room_description": "Conference room",
                },
                "scene_id": "scene0002_00",
            },
        ]

        json_path = data_dir / "balanced_val_set.json"
        with open(json_path, "w") as f:
            json.dump(mock_data, f)

        return tmp_path

    def test_from_path(self, mock_dataset_dir: Path) -> None:
        """Test loading dataset from path."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")

        assert len(dataset) == 3
        assert dataset.data_root == mock_dataset_dir
        assert dataset.split == "val"

    def test_from_path_file_not_found(self, tmp_path: Path) -> None:
        """Test error when dataset file not found."""
        with pytest.raises(FileNotFoundError):
            SQA3DDataset.from_path(tmp_path)

    def test_filter_by_scene(self, mock_dataset_dir: Path) -> None:
        """Test filtering by scene ID."""
        dataset = SQA3DDataset.from_path(
            mock_dataset_dir, split="val", scene_id="scene0001_00"
        )

        assert len(dataset) == 2
        assert all(s.scene_id == "scene0001_00" for s in dataset)

    def test_max_samples(self, mock_dataset_dir: Path) -> None:
        """Test limiting number of samples."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val", max_samples=2)

        assert len(dataset) == 2

    def test_iteration(self, mock_dataset_dir: Path) -> None:
        """Test iteration over dataset."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")

        samples = list(dataset)
        assert len(samples) == 3
        assert all(isinstance(s, SQA3DSample) for s in samples)

    def test_getitem(self, mock_dataset_dir: Path) -> None:
        """Test indexing into dataset."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")

        sample = dataset[0]
        assert isinstance(sample, SQA3DSample)
        assert sample.question_id == "q001"

    def test_get_question_types(self, mock_dataset_dir: Path) -> None:
        """Test getting unique question types."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")

        types = dataset.get_question_types()
        assert "what_color" in types
        assert "where" in types
        assert "how_many" in types

    def test_get_scenes(self, mock_dataset_dir: Path) -> None:
        """Test getting unique scene IDs."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")

        scenes = dataset.get_scenes()
        assert "scene0001_00" in scenes
        assert "scene0002_00" in scenes

    def test_filter_by_scene_method(self, mock_dataset_dir: Path) -> None:
        """Test filter_by_scene method."""
        dataset = SQA3DDataset.from_path(mock_dataset_dir, split="val")
        filtered = dataset.filter_by_scene("scene0002_00")

        assert len(filtered) == 1
        assert filtered[0].scene_id == "scene0002_00"


class TestEvaluateSQA3D:
    """Tests for SQA3D evaluation function."""

    def test_exact_match(self) -> None:
        """Test exact match evaluation."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What color?",
            answers=["red"],
            situation=situation,
            scene_id="scene001",
            question_type="what_color",
            answer_type="single_word",
        )

        results = evaluate_sqa3d([(sample, "red")])

        assert len(results) == 1
        assert results[0].exact_match is True
        assert results[0].contains_match is True

    def test_no_match(self) -> None:
        """Test no match evaluation."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What color?",
            answers=["red"],
            situation=situation,
            scene_id="scene001",
            question_type="what_color",
            answer_type="single_word",
        )

        results = evaluate_sqa3d([(sample, "blue")])

        assert results[0].exact_match is False
        assert results[0].contains_match is False

    def test_contains_match(self) -> None:
        """Test contains match (partial match)."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What color?",
            answers=["red"],
            situation=situation,
            scene_id="scene001",
            question_type="what_color",
            answer_type="single_word",
        )

        results = evaluate_sqa3d([(sample, "it is red in color")])

        assert results[0].exact_match is False
        assert results[0].contains_match is True

    def test_case_insensitive_match(self) -> None:
        """Test case insensitive matching."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="What color?",
            answers=["Red"],
            situation=situation,
            scene_id="scene001",
            question_type="what_color",
            answer_type="single_word",
        )

        results = evaluate_sqa3d([(sample, "RED")])

        assert results[0].exact_match is True

    def test_multiple_valid_answers(self) -> None:
        """Test matching against multiple valid answers."""
        situation = SQA3DSituation([0, 0, 0], [1, 0, 0], "Room")
        sample = SQA3DSample(
            question_id="q1",
            question="How many?",
            answers=["four", "4"],
            situation=situation,
            scene_id="scene001",
            question_type="how_many",
            answer_type="single_word",
        )

        results = evaluate_sqa3d([(sample, "4")])

        assert results[0].exact_match is True


class TestComputeSQA3DMetrics:
    """Tests for metric computation."""

    def test_empty_results(self) -> None:
        """Test metrics with empty results."""
        metrics = compute_sqa3d_metrics([])

        assert metrics["exact_match_accuracy"] == 0.0
        assert metrics["contains_accuracy"] == 0.0
        assert metrics["total"] == 0

    def test_perfect_scores(self) -> None:
        """Test metrics with all correct."""
        results = [
            SQA3DEvaluationResult("q1", "Q1", ["a1"], "a1", True, True),
            SQA3DEvaluationResult("q2", "Q2", ["a2"], "a2", True, True),
        ]

        metrics = compute_sqa3d_metrics(results)

        assert metrics["exact_match_accuracy"] == 1.0
        assert metrics["contains_accuracy"] == 1.0
        assert metrics["total"] == 2

    def test_mixed_scores(self) -> None:
        """Test metrics with mixed results."""
        results = [
            SQA3DEvaluationResult("q1", "Q1", ["a1"], "a1", True, True),
            SQA3DEvaluationResult("q2", "Q2", ["a2"], "it is a2", False, True),
            SQA3DEvaluationResult("q3", "Q3", ["a3"], "wrong", False, False),
        ]

        metrics = compute_sqa3d_metrics(results)

        assert metrics["exact_match_accuracy"] == pytest.approx(1 / 3)
        assert metrics["contains_accuracy"] == pytest.approx(2 / 3)
        assert metrics["total"] == 3


class TestDownloadSQA3D:
    """Tests for download function."""

    @patch("subprocess.run")
    def test_download_new(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test downloading to new directory."""
        mock_run.return_value = MagicMock(returncode=0)

        result = download_sqa3d(tmp_path)

        assert result == tmp_path / "SQA3D"
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "git" in call_args
        assert "clone" in call_args

    def test_download_existing(self, tmp_path: Path) -> None:
        """Test skipping download when already exists."""
        repo_dir = tmp_path / "SQA3D"
        repo_dir.mkdir()

        result = download_sqa3d(tmp_path)

        assert result == repo_dir
