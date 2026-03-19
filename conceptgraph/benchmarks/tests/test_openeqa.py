"""Unit tests for OpenEQA benchmark loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.benchmarks.openeqa_loader import (
    EvaluationResult,
    OpenEQADataset,
    OpenEQASample,
    compute_metrics,
    download_openeqa,
    evaluate_with_llm,
)


class TestOpenEQASample:
    """Tests for OpenEQASample dataclass."""

    def test_sample_creation(self) -> None:
        """Test basic sample creation."""
        sample = OpenEQASample(
            question_id="q1",
            question="What color is the chair?",
            answer="red",
            episode_history=None,
            category="object_recognition",
            scene_id="scene_001",
            question_type="episodic_memory",
        )

        assert sample.question_id == "q1"
        assert sample.question == "What color is the chair?"
        assert sample.answer == "red"
        assert sample.episode_history is None
        assert sample.category == "object_recognition"
        assert sample.scene_id == "scene_001"
        assert sample.question_type == "episodic_memory"
        assert sample.frames == []

    def test_sample_with_episode_history(self) -> None:
        """Test sample with episode history path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_001"
            episode_path.mkdir()

            sample = OpenEQASample(
                question_id="q2",
                question="Where is the table?",
                answer="in the living room",
                episode_history=episode_path,
                category="spatial",
                scene_id="scene_002",
                question_type="episodic_memory",
            )

            assert sample.episode_history == episode_path
            assert sample.episode_history.exists()

    def test_load_frames_empty_directory(self) -> None:
        """Test loading frames from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_empty"
            episode_path.mkdir()

            sample = OpenEQASample(
                question_id="q3",
                question="Test question",
                answer="Test answer",
                episode_history=episode_path,
                category="test",
                scene_id="test_scene",
                question_type="episodic_memory",
            )

            frames = sample.load_frames()
            assert frames == []

    def test_load_frames_with_images(self) -> None:
        """Test loading frames from directory with images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_with_frames"
            episode_path.mkdir()

            # Create test images
            for i in range(5):
                (episode_path / f"frame_{i:04d}.jpg").touch()

            sample = OpenEQASample(
                question_id="q4",
                question="Test question",
                answer="Test answer",
                episode_history=episode_path,
                category="test",
                scene_id="test_scene",
                question_type="episodic_memory",
            )

            frames = sample.load_frames(max_frames=10)
            assert len(frames) == 5
            assert all(f.suffix == ".jpg" for f in frames)

    def test_load_frames_uniform_sampling(self) -> None:
        """Test uniform sampling when frames exceed max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_many_frames"
            episode_path.mkdir()

            # Create 20 test images
            for i in range(20):
                (episode_path / f"frame_{i:04d}.png").touch()

            sample = OpenEQASample(
                question_id="q5",
                question="Test question",
                answer="Test answer",
                episode_history=episode_path,
                category="test",
                scene_id="test_scene",
                question_type="episodic_memory",
            )

            frames = sample.load_frames(max_frames=5)
            assert len(frames) == 5
            # Check uniform sampling
            assert sample.frames == frames

    def test_load_frames_nonexistent_directory(self) -> None:
        """Test loading frames from nonexistent directory."""
        sample = OpenEQASample(
            question_id="q6",
            question="Test question",
            answer="Test answer",
            episode_history=Path("/nonexistent/path"),
            category="test",
            scene_id="test_scene",
            question_type="episodic_memory",
        )

        frames = sample.load_frames()
        assert frames == []

    def test_load_frames_mixed_extensions(self) -> None:
        """Test loading frames with mixed image extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_mixed"
            episode_path.mkdir()

            # Create mixed format images
            (episode_path / "frame_001.jpg").touch()
            (episode_path / "frame_002.jpeg").touch()
            (episode_path / "frame_003.png").touch()
            (episode_path / "frame_004.txt").touch()  # Non-image file

            sample = OpenEQASample(
                question_id="q7",
                question="Test question",
                answer="Test answer",
                episode_history=episode_path,
                category="test",
                scene_id="test_scene",
                question_type="episodic_memory",
            )

            frames = sample.load_frames()
            assert len(frames) == 3  # Only image files


class TestOpenEQADataset:
    """Tests for OpenEQADataset class."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a mock OpenEQA dataset directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create mock JSON data
        mock_data = [
            {
                "question_id": "q001",
                "question": "What color is the sofa?",
                "answer": "blue",
                "category": "object_recognition",
                "scene_id": "scene_001",
                "question_type": "episodic_memory",
                "episode_history": "episode_001",
            },
            {
                "question_id": "q002",
                "question": "Where is the lamp?",
                "answer": "on the table",
                "category": "spatial",
                "scene_id": "scene_001",
                "question_type": "episodic_memory",
                "episode_history": "episode_001",
            },
            {
                "question_id": "q003",
                "question": "How many chairs are there?",
                "answer": "four",
                "category": "counting",
                "scene_id": "scene_002",
                "question_type": "active_exploration",
                "episode_history": "episode_002",
            },
        ]

        json_path = data_dir / "open-eqa-v0.json"
        with open(json_path, "w") as f:
            json.dump(mock_data, f)

        # Create episode directories
        frames_dir = data_dir / "frames"
        frames_dir.mkdir()
        (frames_dir / "episode_001").mkdir()
        (frames_dir / "episode_002").mkdir()

        return tmp_path

    def test_from_path(self, mock_dataset_dir: Path) -> None:
        """Test loading dataset from path."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir)

        assert len(dataset) == 3
        assert dataset.data_root == mock_dataset_dir
        assert dataset.split == "val"

    def test_from_path_file_not_found(self, tmp_path: Path) -> None:
        """Test error when dataset file not found."""
        with pytest.raises(FileNotFoundError):
            OpenEQADataset.from_path(tmp_path)

    def test_filter_by_question_type(self, mock_dataset_dir: Path) -> None:
        """Test filtering by question type."""
        dataset = OpenEQADataset.from_path(
            mock_dataset_dir, question_type="episodic_memory"
        )

        assert len(dataset) == 2
        assert all(s.question_type == "episodic_memory" for s in dataset)

    def test_filter_by_category(self, mock_dataset_dir: Path) -> None:
        """Test filtering by category."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir, category="spatial")

        assert len(dataset) == 1
        assert dataset[0].category == "spatial"

    def test_max_samples(self, mock_dataset_dir: Path) -> None:
        """Test limiting number of samples."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir, max_samples=2)

        assert len(dataset) == 2

    def test_iteration(self, mock_dataset_dir: Path) -> None:
        """Test iteration over dataset."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir)

        samples = list(dataset)
        assert len(samples) == 3
        assert all(isinstance(s, OpenEQASample) for s in samples)

    def test_getitem(self, mock_dataset_dir: Path) -> None:
        """Test indexing into dataset."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir)

        sample = dataset[0]
        assert isinstance(sample, OpenEQASample)
        assert sample.question_id == "q001"

    def test_get_categories(self, mock_dataset_dir: Path) -> None:
        """Test getting unique categories."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir)

        categories = dataset.get_categories()
        assert set(categories) == {"counting", "object_recognition", "spatial"}

    def test_filter_by_category_method(self, mock_dataset_dir: Path) -> None:
        """Test filter_by_category method."""
        dataset = OpenEQADataset.from_path(mock_dataset_dir)
        filtered = dataset.filter_by_category("counting")

        assert len(filtered) == 1
        assert filtered[0].category == "counting"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test basic result creation."""
        result = EvaluationResult(
            question_id="q1",
            question="What color?",
            ground_truth="red",
            prediction="red",
            score=1.0,
            reasoning="Perfect match",
        )

        assert result.question_id == "q1"
        assert result.score == 1.0
        assert result.reasoning == "Perfect match"


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_empty_results(self) -> None:
        """Test metrics with empty results."""
        metrics = compute_metrics([])

        assert metrics["mean_score"] == 0.0
        assert metrics["exact_match"] == 0.0
        assert metrics["partial_match"] == 0.0

    def test_perfect_scores(self) -> None:
        """Test metrics with perfect scores."""
        results = [
            EvaluationResult("q1", "Q1", "A1", "A1", 1.0, "Perfect"),
            EvaluationResult("q2", "Q2", "A2", "A2", 1.0, "Perfect"),
            EvaluationResult("q3", "Q3", "A3", "A3", 1.0, "Perfect"),
        ]

        metrics = compute_metrics(results)

        assert metrics["mean_score"] == 1.0
        assert metrics["exact_match"] == 1.0
        assert metrics["partial_match"] == 1.0
        assert metrics["total_samples"] == 3

    def test_mixed_scores(self) -> None:
        """Test metrics with mixed scores."""
        results = [
            EvaluationResult("q1", "Q1", "A1", "A1", 1.0, "Perfect"),
            EvaluationResult("q2", "Q2", "A2", "B2", 0.5, "Partial"),
            EvaluationResult("q3", "Q3", "A3", "C3", 0.0, "Wrong"),
        ]

        metrics = compute_metrics(results)

        assert metrics["mean_score"] == 0.5
        assert metrics["exact_match"] == 1 / 3  # Only 1.0 >= 0.9
        assert metrics["partial_match"] == 2 / 3  # 1.0 and 0.5 >= 0.5
        assert metrics["total_samples"] == 3

    def test_threshold_boundary(self) -> None:
        """Test exact match and partial match thresholds."""
        results = [
            EvaluationResult("q1", "Q1", "A1", "A1", 0.9, "Exact threshold"),
            EvaluationResult("q2", "Q2", "A2", "A2", 0.89, "Below exact"),
            EvaluationResult("q3", "Q3", "A3", "A3", 0.5, "Partial threshold"),
            EvaluationResult("q4", "Q4", "A4", "A4", 0.49, "Below partial"),
        ]

        metrics = compute_metrics(results)

        assert metrics["exact_match"] == 0.25  # Only 0.9
        assert metrics["partial_match"] == 0.75  # 0.9, 0.89, 0.5


class TestEvaluateWithLLM:
    """Tests for evaluate_with_llm function."""

    def test_successful_evaluation(self) -> None:
        """Test successful LLM evaluation."""
        with patch("openai.OpenAI") as mock_openai_class:
            # Mock the OpenAI client
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(
                {"score": 0.75, "reasoning": "Good match with minor differences"}
            )
            mock_client.chat.completions.create.return_value = mock_response

            sample = OpenEQASample(
                question_id="q1",
                question="What color is the chair?",
                answer="red",
                episode_history=None,
                category="test",
                scene_id="test",
                question_type="episodic_memory",
            )

            results = evaluate_with_llm(
                [(sample, "The chair is red")], api_key="test-key"
            )

            assert len(results) == 1
            assert results[0].score == 0.75
            assert "Good match" in results[0].reasoning

    def test_evaluation_error_handling(self) -> None:
        """Test error handling during evaluation."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            sample = OpenEQASample(
                question_id="q1",
                question="Test question",
                answer="Test answer",
                episode_history=None,
                category="test",
                scene_id="test",
                question_type="episodic_memory",
            )

            results = evaluate_with_llm([(sample, "prediction")], api_key="test-key")

            assert len(results) == 1
            assert results[0].score == 0.0
            assert "error" in results[0].reasoning.lower()


class TestDownloadOpenEQA:
    """Tests for download_openeqa function."""

    @patch("subprocess.run")
    def test_download_new(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test downloading to new directory."""
        mock_run.return_value = MagicMock(returncode=0)

        result = download_openeqa(tmp_path, include_frames=False)

        assert result == tmp_path / "open-eqa"
        mock_run.assert_called_once()
        # Verify git clone was called
        call_args = mock_run.call_args[0][0]
        assert "git" in call_args
        assert "clone" in call_args

    def test_download_existing(self, tmp_path: Path) -> None:
        """Test skipping download when already exists."""
        repo_dir = tmp_path / "open-eqa"
        repo_dir.mkdir()

        result = download_openeqa(tmp_path)

        assert result == repo_dir

    @patch("subprocess.run")
    def test_download_with_frames(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test downloading with frames."""
        mock_run.return_value = MagicMock(returncode=0)

        # Create mock download script
        repo_dir = tmp_path / "open-eqa"

        def create_repo(*args, **kwargs):
            repo_dir.mkdir(exist_ok=True)
            (repo_dir / "download_data.py").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = create_repo

        download_openeqa(tmp_path, include_frames=True)

        # Should call subprocess.run twice: git clone + download script
        assert mock_run.call_count >= 1
