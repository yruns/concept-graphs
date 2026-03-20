"""SQA3D benchmark loader.

SQA3D: Situated Question Answering in 3D Scenes
CVPR 2023, https://sqa3d.github.io/

This loader provides:
- Dataset loading from the official SQA3D JSON format
- Situation context (position, orientation, room description)
- Multi-choice and free-form answer handling
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

from loguru import logger


@dataclass
class SQA3DSituation:
    """Situation context for a question.

    Attributes:
        position: Agent's position as [x, y, z].
        orientation: Agent's view direction as [x, y, z].
        room_description: Natural language description of the room.
        reference_objects: Objects explicitly referenced in the situation.
    """

    position: list[float]
    orientation: list[float]
    room_description: str
    reference_objects: list[str] = field(default_factory=list)


@dataclass
class SQA3DSample:
    """A single SQA3D question-answer pair with situation context.

    Attributes:
        question_id: Unique identifier for the question.
        question: The natural language question.
        answers: List of valid answers (for multiple annotators).
        situation: The situated context of the question.
        scene_id: ScanNet scene identifier (e.g., "scene0000_00").
        question_type: Type of question (what, where, how many, etc.).
        answer_type: "single_word", "multi_word", or "multiple_choice".
        choices: Answer choices if multiple_choice type.
    """

    question_id: str
    question: str
    answers: list[str]
    situation: SQA3DSituation
    scene_id: str
    question_type: str
    answer_type: Literal["single_word", "multi_word", "multiple_choice"]
    choices: list[str] = field(default_factory=list)

    @property
    def primary_answer(self) -> str:
        """Get the most common answer from annotators."""
        if not self.answers:
            return ""
        # Return first answer (usually the primary one)
        return self.answers[0]


class SQA3DDataset:
    """SQA3D dataset loader.

    Usage:
        dataset = SQA3DDataset.from_path("/path/to/sqa3d", split="val")
        for sample in dataset:
            print(f"Q: {sample.question}")
            print(f"A: {sample.primary_answer}")
    """

    def __init__(
        self,
        samples: list[SQA3DSample],
        data_root: Path,
        split: str = "val",
    ) -> None:
        """Initialize dataset.

        Args:
            samples: List of SQA3D samples.
            data_root: Root directory containing the dataset.
            split: Dataset split (train/val/test).
        """
        self.samples = samples
        self.data_root = data_root
        self.split = split

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        split: str = "val",
        question_type: Optional[str] = None,
        scene_id: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> "SQA3DDataset":
        """Load SQA3D dataset from directory.

        Args:
            data_root: Path to the SQA3D repository root.
            split: Dataset split to load (train/val/test).
            question_type: Filter by question type.
            scene_id: Filter by specific scene.
            max_samples: Maximum number of samples to load.

        Returns:
            SQA3DDataset instance.
        """
        data_root = Path(data_root)

        # Try official SQA3D format first (separate questions + annotations files)
        questions_path = (
            data_root
            / "assets"
            / "data"
            / "sqa_task"
            / "balanced"
            / f"v1_balanced_questions_{split}_scannetv2.json"
        )
        annotations_path = (
            data_root
            / "assets"
            / "data"
            / "sqa_task"
            / "balanced"
            / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
        )

        if questions_path.exists() and annotations_path.exists():
            return cls._load_official_format(
                data_root,
                questions_path,
                annotations_path,
                split,
                question_type,
                scene_id,
                max_samples,
            )

        # Fallback to simplified single-file format
        json_path = data_root / "data" / "sqa_task" / f"balanced_{split}_set.json"

        # Try alternative paths
        if not json_path.exists():
            json_path = data_root / "data" / f"balanced_{split}_set.json"
        if not json_path.exists():
            json_path = data_root / f"balanced_{split}_set.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"SQA3D data file not found. Tried:\n"
                f"  - {questions_path}\n"
                f"  - {json_path}\n"
                "Please download the dataset first using download_sqa3d()."
            )

        logger.info(f"Loading SQA3D from {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        # Handle wrapped format (dict with 'annotations' or 'questions' key)
        if isinstance(data, dict):
            if "annotations" in data:
                data = data["annotations"]
            elif "questions" in data:
                data = data["questions"]

        samples: list[SQA3DSample] = []

        for item in data:
            # Apply filters
            item_scene_id = item.get("scene_id", "")
            if scene_id and item_scene_id != scene_id:
                continue

            item_question_type = _infer_question_type(item.get("question", ""))
            if question_type and item_question_type != question_type:
                continue

            # Parse situation context
            situation_data = item.get("situation", {})
            situation = SQA3DSituation(
                position=situation_data.get("position", [0.0, 0.0, 0.0]),
                orientation=situation_data.get("orientation", [0.0, 0.0, 1.0]),
                room_description=situation_data.get("room_description", ""),
                reference_objects=situation_data.get("reference_objects", []),
            )

            # Handle answers (can be list or single value)
            answers_raw = item.get("answers", item.get("answer", []))
            if isinstance(answers_raw, str):
                answers = [answers_raw]
            elif isinstance(answers_raw, list):
                answers = [str(a) for a in answers_raw]
            else:
                answers = [str(answers_raw)]

            # Determine answer type
            choices = item.get("choices", [])
            if choices:
                answer_type = "multiple_choice"
            elif len(answers[0].split()) == 1:
                answer_type = "single_word"
            else:
                answer_type = "multi_word"

            sample = SQA3DSample(
                question_id=item.get("question_id", str(len(samples))),
                question=item["question"],
                answers=answers,
                situation=situation,
                scene_id=item_scene_id,
                question_type=item_question_type,
                answer_type=answer_type,
                choices=choices,
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} SQA3D samples")
        return cls(samples, data_root, split)

    @classmethod
    def _load_official_format(
        cls,
        data_root: Path,
        questions_path: Path,
        annotations_path: Path,
        split: str,
        question_type: Optional[str],
        scene_id: Optional[str],
        max_samples: Optional[int],
    ) -> "SQA3DDataset":
        """Load from official SQA3D format (separate questions + annotations).

        The official format has:
        - questions file: contains question_id, question, situation, scene_id
        - annotations file: contains question_id, answers, position, rotation
        """
        logger.info(f"Loading SQA3D official format from {questions_path.parent}")

        with open(questions_path) as f:
            questions_data = json.load(f)
        with open(annotations_path) as f:
            annotations_data = json.load(f)

        # Build lookup from question_id to annotation
        annotations_by_id = {
            ann["question_id"]: ann for ann in annotations_data.get("annotations", [])
        }

        samples: list[SQA3DSample] = []

        for q in questions_data.get("questions", []):
            q_id = q.get("question_id")
            item_scene_id = q.get("scene_id", "")

            # Apply scene filter
            if scene_id and item_scene_id != scene_id:
                continue

            # Get question text and infer type
            question_text = q.get("question", "")
            item_question_type = _infer_question_type(question_text)
            if question_type and item_question_type != question_type:
                continue

            # Get annotation for this question
            ann = annotations_by_id.get(q_id, {})

            # Parse situation - use the text description from questions file
            situation_text = q.get("situation", "")
            position = ann.get("position", {})
            rotation = ann.get("rotation", {})

            situation = SQA3DSituation(
                position=[
                    position.get("x", 0.0),
                    position.get("y", 0.0),
                    position.get("z", 0.0),
                ],
                orientation=[
                    rotation.get("_x", 0.0),
                    rotation.get("_y", 0.0),
                    rotation.get("_z", 0.0),
                ],
                room_description=situation_text,
                reference_objects=[],
            )

            # Parse answers from annotation
            answers_list = ann.get("answers", [])
            if isinstance(answers_list, list) and answers_list:
                if isinstance(answers_list[0], dict):
                    answers = [a.get("answer", "") for a in answers_list]
                else:
                    answers = [str(a) for a in answers_list]
            else:
                answers = [""]

            # Filter empty answers
            answers = [a for a in answers if a] or [""]

            # Determine answer type
            answer_type_raw = ann.get("answer_type", "other")
            if len(answers[0].split()) == 1:
                answer_type = "single_word"
            else:
                answer_type = "multi_word"

            sample = SQA3DSample(
                question_id=str(q_id),
                question=question_text,
                answers=answers,
                situation=situation,
                scene_id=item_scene_id,
                question_type=item_question_type,
                answer_type=answer_type,
                choices=[],
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} SQA3D samples (official format)")
        return cls(samples, data_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SQA3DSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[SQA3DSample]:
        return iter(self.samples)

    def get_question_types(self) -> list[str]:
        """Get list of unique question types."""
        return sorted(set(s.question_type for s in self.samples))

    def get_scenes(self) -> list[str]:
        """Get list of unique scene IDs."""
        return sorted(set(s.scene_id for s in self.samples))

    def filter_by_scene(self, scene_id: str) -> "SQA3DDataset":
        """Create a new dataset filtered by scene."""
        filtered = [s for s in self.samples if s.scene_id == scene_id]
        return SQA3DDataset(filtered, self.data_root, self.split)


def _infer_question_type(question: str) -> str:
    """Infer question type from the question text.

    Args:
        question: The question text.

    Returns:
        Question type string.
    """
    question_lower = question.lower().strip()

    if question_lower.startswith("what"):
        if "color" in question_lower:
            return "what_color"
        elif "shape" in question_lower:
            return "what_shape"
        elif "material" in question_lower:
            return "what_material"
        return "what"
    elif question_lower.startswith("where"):
        return "where"
    elif question_lower.startswith("how many"):
        return "how_many"
    elif question_lower.startswith("is ") or question_lower.startswith("are "):
        return "yes_no"
    elif question_lower.startswith("which"):
        return "which"
    elif question_lower.startswith("can"):
        return "can"
    else:
        return "other"


def download_sqa3d(
    output_dir: str | Path,
    include_scannet: bool = False,
) -> Path:
    """Download SQA3D dataset.

    Args:
        output_dir: Directory to save the dataset.
        include_scannet: Whether to download ScanNet scenes (requires agreement).

    Returns:
        Path to the downloaded dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / "SQA3D"

    if repo_dir.exists():
        logger.info(f"SQA3D already exists at {repo_dir}")
        return repo_dir

    logger.info("Cloning SQA3D repository...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/SilongYong/SQA3D.git",
            str(repo_dir),
        ],
        check=True,
    )

    if include_scannet:
        logger.warning(
            "ScanNet download requires manual agreement at "
            "http://www.scan-net.org/. Please download manually."
        )

    logger.success(f"SQA3D downloaded to {repo_dir}")
    return repo_dir


# Evaluation utilities


@dataclass
class SQA3DEvaluationResult:
    """Result of evaluating a single SQA3D prediction."""

    question_id: str
    question: str
    ground_truths: list[str]
    prediction: str
    exact_match: bool
    contains_match: bool  # Prediction contains any ground truth


def evaluate_sqa3d(
    predictions: list[tuple[SQA3DSample, str]],
) -> list[SQA3DEvaluationResult]:
    """Evaluate SQA3D predictions.

    Following the official SQA3D evaluation:
    - Exact match: prediction exactly matches any valid answer
    - Contains match: prediction contains any valid answer

    Args:
        predictions: List of (sample, prediction) tuples.

    Returns:
        List of evaluation results.
    """
    results = []

    for sample, prediction in predictions:
        pred_normalized = _normalize_answer(prediction)
        gt_normalized = [_normalize_answer(a) for a in sample.answers]

        exact_match = pred_normalized in gt_normalized
        contains_match = any(gt in pred_normalized for gt in gt_normalized)

        results.append(
            SQA3DEvaluationResult(
                question_id=sample.question_id,
                question=sample.question,
                ground_truths=sample.answers,
                prediction=prediction,
                exact_match=exact_match,
                contains_match=contains_match,
            )
        )

    return results


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized answer string.
    """
    # Lowercase, strip whitespace, remove punctuation
    import re

    normalized = answer.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = " ".join(normalized.split())  # Normalize whitespace
    return normalized


def compute_sqa3d_metrics(results: list[SQA3DEvaluationResult]) -> dict:
    """Compute aggregate metrics from SQA3D evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        Dictionary containing accuracy metrics.
    """
    if not results:
        return {"exact_match_accuracy": 0.0, "contains_accuracy": 0.0, "total": 0}

    exact_matches = sum(1 for r in results if r.exact_match)
    contains_matches = sum(1 for r in results if r.contains_match)

    return {
        "exact_match_accuracy": exact_matches / len(results),
        "contains_accuracy": contains_matches / len(results),
        "total": len(results),
    }
