"""OpenEQA benchmark loader.

OpenEQA: Embodied Question Answering in the Era of Foundation Models
CVPR 2024, Facebook Research

This loader provides:
- Dataset loading from the official OpenEQA JSON format
- Episode history (images/video frames) loading
- LLM-powered evaluation with GPT-4 protocol
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

from loguru import logger


@dataclass
class OpenEQASample:
    """A single OpenEQA question-answer pair with episode context.

    Attributes:
        question_id: Unique identifier for the question.
        question: The natural language question.
        answer: Ground truth answer.
        episode_history: Path to episode directory containing frames.
        category: Question category (e.g., "object_recognition", "spatial").
        scene_id: Identifier for the underlying scene.
        question_type: "episodic_memory" or "active_exploration".
        frames: List of frame paths loaded from episode history.
    """

    question_id: str
    question: str
    answer: str
    episode_history: Optional[Path]
    category: str
    scene_id: str
    question_type: Literal["episodic_memory", "active_exploration"]
    frames: list[Path] = field(default_factory=list)

    def load_frames(self, max_frames: int = 16) -> list[Path]:
        """Load frame paths from episode history.

        Args:
            max_frames: Maximum number of frames to load (uniformly sampled).

        Returns:
            List of paths to frame images.
        """
        if self.episode_history is None or not self.episode_history.exists():
            logger.warning(f"Episode history not found: {self.episode_history}")
            return []

        # Find all image files in episode directory
        image_extensions = {".jpg", ".jpeg", ".png"}
        all_frames = sorted(
            [
                f
                for f in self.episode_history.iterdir()
                if f.suffix.lower() in image_extensions
            ]
        )

        if not all_frames:
            logger.warning(f"No frames found in {self.episode_history}")
            return []

        # Uniformly sample if we have more frames than max
        if len(all_frames) <= max_frames:
            self.frames = all_frames
        else:
            step = len(all_frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            self.frames = [all_frames[i] for i in indices]

        return self.frames


class OpenEQADataset:
    """OpenEQA dataset loader.

    Usage:
        dataset = OpenEQADataset.from_path("/path/to/open-eqa")
        for sample in dataset:
            frames = sample.load_frames()
            # Run your model
    """

    def __init__(
        self,
        samples: list[OpenEQASample],
        data_root: Path,
        split: str = "val",
    ) -> None:
        """Initialize dataset.

        Args:
            samples: List of OpenEQA samples.
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
        category: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> "OpenEQADataset":
        """Load OpenEQA dataset from directory.

        Args:
            data_root: Path to the open-eqa repository root.
            split: Dataset split to load.
            question_type: Filter by question type ("episodic_memory" or "active_exploration").
            category: Filter by question category.
            max_samples: Maximum number of samples to load.

        Returns:
            OpenEQADataset instance.
        """
        data_root = Path(data_root)
        json_path = data_root / "data" / "open-eqa-v0.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"OpenEQA data file not found: {json_path}. "
                "Please download the dataset first using download_openeqa()."
            )

        logger.info(f"Loading OpenEQA from {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        samples: list[OpenEQASample] = []
        for item in data:
            # Apply filters
            if question_type and item.get("question_type") != question_type:
                continue
            if category and item.get("category") != category:
                continue

            # Determine episode history path
            episode_id = item.get("episode_history", "")
            episode_path = None
            if episode_id:
                # OpenEQA stores episodes in data/frames/<episode_id>/
                episode_path = data_root / "data" / "frames" / episode_id
                if not episode_path.exists():
                    # Try alternative path structure
                    episode_path = data_root / "frames" / episode_id

            sample = OpenEQASample(
                question_id=item.get("question_id", str(len(samples))),
                question=item["question"],
                answer=item["answer"],
                episode_history=episode_path,
                category=item.get("category", "unknown"),
                scene_id=item.get("scene_id", "unknown"),
                question_type=item.get("question_type", "episodic_memory"),
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} OpenEQA samples")
        return cls(samples, data_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> OpenEQASample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[OpenEQASample]:
        return iter(self.samples)

    def get_categories(self) -> list[str]:
        """Get list of unique question categories."""
        return sorted(set(s.category for s in self.samples))

    def filter_by_category(self, category: str) -> "OpenEQADataset":
        """Create a new dataset filtered by category."""
        filtered = [s for s in self.samples if s.category == category]
        return OpenEQADataset(filtered, self.data_root, self.split)


def download_openeqa(
    output_dir: str | Path,
    include_frames: bool = False,
) -> Path:
    """Download OpenEQA dataset.

    Args:
        output_dir: Directory to save the dataset.
        include_frames: Whether to download episode frames (large, ~50GB).

    Returns:
        Path to the downloaded dataset.
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / "open-eqa"

    if repo_dir.exists():
        logger.info(f"OpenEQA already exists at {repo_dir}")
        return repo_dir

    logger.info("Cloning OpenEQA repository...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/facebookresearch/open-eqa.git",
            str(repo_dir),
        ],
        check=True,
    )

    if include_frames:
        logger.info("Downloading episode frames (this may take a while)...")
        # OpenEQA provides download scripts for frames
        download_script = repo_dir / "download_data.py"
        if download_script.exists():
            subprocess.run(
                ["python", str(download_script), "--frames"],
                cwd=repo_dir,
                check=True,
            )
        else:
            logger.warning("Frame download script not found. Manual download required.")

    logger.success(f"OpenEQA downloaded to {repo_dir}")
    return repo_dir


# Evaluation utilities


@dataclass
class EvaluationResult:
    """Result of evaluating a single prediction."""

    question_id: str
    question: str
    ground_truth: str
    prediction: str
    score: float  # 0-1 score from LLM evaluation
    reasoning: str  # LLM's explanation for the score


def evaluate_with_llm(
    predictions: list[tuple[OpenEQASample, str]],
    model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> list[EvaluationResult]:
    """Evaluate predictions using LLM-based scoring (OpenEQA protocol).

    The OpenEQA evaluation protocol uses GPT-4 to score predictions
    based on semantic similarity to ground truth answers.

    Args:
        predictions: List of (sample, prediction) tuples.
        model: LLM model to use for evaluation.
        api_key: OpenAI API key (uses environment variable if not provided).

    Returns:
        List of evaluation results.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # Load evaluation prompt from OpenEQA
    eval_prompt_template = """You are evaluating a response to a question about an environment.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Rate how well the predicted answer matches the ground truth on a scale from 0 to 1:
- 1.0: Perfect match - answers are semantically equivalent
- 0.75: Good match - small differences that don't change the meaning
- 0.5: Partial match - some correct information but missing key details
- 0.25: Poor match - mostly incorrect but has some relevant information
- 0.0: No match - completely wrong or irrelevant

Respond in JSON format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""

    results = []
    for sample, prediction in predictions:
        prompt = eval_prompt_template.format(
            question=sample.question,
            ground_truth=sample.answer,
            prediction=prediction,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200,
            )

            result_json = json.loads(response.choices[0].message.content)
            results.append(
                EvaluationResult(
                    question_id=sample.question_id,
                    question=sample.question,
                    ground_truth=sample.answer,
                    prediction=prediction,
                    score=float(result_json.get("score", 0)),
                    reasoning=result_json.get("reasoning", ""),
                )
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {sample.question_id}: {e}")
            results.append(
                EvaluationResult(
                    question_id=sample.question_id,
                    question=sample.question,
                    ground_truth=sample.answer,
                    prediction=prediction,
                    score=0.0,
                    reasoning=f"Evaluation error: {e}",
                )
            )

    return results


def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute aggregate metrics from evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        Dictionary containing:
        - mean_score: Average score across all samples
        - exact_match: Fraction of samples with score >= 0.9
        - partial_match: Fraction of samples with score >= 0.5
    """
    if not results:
        return {"mean_score": 0.0, "exact_match": 0.0, "partial_match": 0.0}

    scores = [r.score for r in results]
    return {
        "mean_score": sum(scores) / len(scores),
        "exact_match": sum(1 for s in scores if s >= 0.9) / len(scores),
        "partial_match": sum(1 for s in scores if s >= 0.5) / len(scores),
        "total_samples": len(results),
    }
