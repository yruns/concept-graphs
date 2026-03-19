"""Unit tests for batch_eval module.

Tests cover:
- EvalSampleResult dataclass initialization and field access
- EvalRunResult aggregate calculation and serialization
- BatchEvalConfig Pydantic validation and ablation tags
- CheckpointManager save/load functionality
- BatchEvaluator parallel evaluation with mocked stages
- Sample adapters for OpenEQA and SQA3D
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    CheckpointManager,
    EvalRunResult,
    EvalSample,
    EvalSampleResult,
    OpenEQASampleAdapter,
    SQA3DSampleAdapter,
    adapt_openeqa_samples,
    adapt_sqa3d_samples,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockEvalSample:
    """Mock sample conforming to EvalSample protocol."""

    _sample_id: str
    _query: str
    _task_type: str = "qa"
    _ground_truth: Any = "test answer"
    _scene_id: str = "scene001"

    @property
    def sample_id(self) -> str:
        return self._sample_id

    @property
    def query(self) -> str:
        return self._query

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def ground_truth(self) -> Any:
        return self._ground_truth

    @property
    def scene_id(self) -> str:
        return self._scene_id


@pytest.fixture
def sample_result() -> EvalSampleResult:
    """Create a sample evaluation result."""
    return EvalSampleResult(
        sample_id="test_001",
        query="What color is the chair?",
        task_type="qa",
        scene_id="scene001",
        stage1_success=True,
        stage1_keyframe_count=3,
        stage1_hypothesis_kind="direct",
        stage1_latency_ms=150.0,
        stage2_success=True,
        stage2_status="completed",
        stage2_confidence=0.85,
        stage2_answer="The chair is blue",
        stage2_tool_calls=2,
        stage2_latency_ms=2500.0,
        ground_truth="blue",
        prediction="The chair is blue",
        metrics={"accuracy": 1.0},
        tool_trace=[
            {"tool_name": "request_more_views", "tool_input": {"count": 2}},
            {"tool_name": "request_crops", "tool_input": {"object_ids": [1, 2]}},
        ],
        uncertainties=[],
        cited_frames=[0, 1, 2],
        timestamp="2026-03-20T10:00:00",
    )


@pytest.fixture
def sample_results() -> List[EvalSampleResult]:
    """Create multiple evaluation results for aggregate testing."""
    return [
        EvalSampleResult(
            sample_id=f"test_{i:03d}",
            query=f"Query {i}",
            task_type="qa",
            scene_id="scene001",
            stage1_success=i % 5 != 4,  # 80% Stage 1 success
            stage1_keyframe_count=3 if i % 5 != 4 else 0,
            stage1_latency_ms=100 + i * 10,
            stage2_success=i % 5 < 3,  # 60% Stage 2 success
            stage2_status="completed" if i % 5 < 3 else "failed",
            stage2_confidence=0.7 + (i % 3) * 0.1,
            stage2_tool_calls=i % 4,
            stage2_latency_ms=2000 + i * 100,
            timestamp=f"2026-03-20T10:{i:02d}:00",
        )
        for i in range(10)
    ]


@pytest.fixture
def batch_config() -> BatchEvalConfig:
    """Create a test batch configuration."""
    return BatchEvalConfig(
        run_id="test_run_001",
        benchmark_name="openeqa",
        max_workers=2,
        batch_size=5,
        checkpoint_interval=3,
        stage1_k=3,
        stage2_enabled=True,
        max_samples=10,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# =============================================================================
# EvalSampleResult Tests
# =============================================================================


class TestEvalSampleResult:
    """Tests for EvalSampleResult dataclass."""

    def test_initialization_minimal(self):
        """Test creation with minimal required fields."""
        result = EvalSampleResult(
            sample_id="test_001",
            query="What is this?",
            task_type="qa",
            scene_id="scene001",
        )
        assert result.sample_id == "test_001"
        assert result.stage1_success is False
        assert result.stage2_success is False
        assert result.metrics == {}

    def test_initialization_full(self, sample_result):
        """Test creation with all fields populated."""
        assert sample_result.sample_id == "test_001"
        assert sample_result.stage1_success is True
        assert sample_result.stage2_confidence == 0.85
        assert len(sample_result.tool_trace) == 2
        assert len(sample_result.cited_frames) == 3

    def test_default_values(self):
        """Test default values for optional fields."""
        result = EvalSampleResult(
            sample_id="test",
            query="test",
            task_type="qa",
            scene_id="scene",
        )
        assert result.stage1_error is None
        assert result.stage2_error is None
        assert result.raw_stage1_output is None
        assert result.raw_stage2_output is None
        assert result.tool_trace == []
        assert result.uncertainties == []


# =============================================================================
# EvalRunResult Tests
# =============================================================================


class TestEvalRunResult:
    """Tests for EvalRunResult aggregate results."""

    def test_to_dict_empty(self):
        """Test serialization with no results."""
        run_result = EvalRunResult(
            run_id="test_run",
            benchmark_name="openeqa",
            config={},
        )
        data = run_result.to_dict()
        assert data["run_id"] == "test_run"
        assert data["summary"]["total_samples"] == 0
        assert data["summary"]["success_rate"] == 0.0

    def test_to_dict_with_results(self, sample_results):
        """Test serialization with populated results."""
        run_result = EvalRunResult(
            run_id="test_run",
            benchmark_name="openeqa",
            config={"max_workers": 4},
            total_samples=10,
            successful_samples=6,
            failed_stage1=2,
            failed_stage2=2,
            avg_stage1_latency_ms=150.0,
            avg_stage2_latency_ms=2500.0,
            avg_stage2_confidence=0.75,
            avg_tool_calls_per_sample=1.5,
            samples_with_tool_use=8,
            samples_with_insufficient_evidence=1,
            results=sample_results,
            start_time="2026-03-20T10:00:00",
            end_time="2026-03-20T10:05:00",
            total_duration_seconds=300.0,
        )

        data = run_result.to_dict()

        assert data["summary"]["total_samples"] == 10
        assert data["summary"]["success_rate"] == 0.6
        assert data["latency"]["avg_stage1_latency_ms"] == 150.0
        assert data["stage2_analysis"]["avg_confidence"] == 0.75
        assert len(data["results"]) == 10

    def test_success_rate_calculation(self):
        """Test success rate with various scenarios."""
        run_result = EvalRunResult(
            run_id="test",
            benchmark_name="test",
            config={},
            total_samples=100,
            successful_samples=75,
        )
        data = run_result.to_dict()
        assert data["summary"]["success_rate"] == 0.75


# =============================================================================
# BatchEvalConfig Tests
# =============================================================================


class TestBatchEvalConfig:
    """Tests for BatchEvalConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchEvalConfig()
        assert config.max_workers == 4
        assert config.stage1_k == 3
        assert config.stage2_enabled is True
        assert config.enable_uncertainty_stopping is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchEvalConfig(
            max_workers=8,
            stage1_k=5,
            stage2_enabled=False,
            confidence_threshold=0.6,
        )
        assert config.max_workers == 8
        assert config.stage1_k == 5
        assert config.stage2_enabled is False
        assert config.confidence_threshold == 0.6

    def test_validation_max_workers(self):
        """Test max_workers validation bounds."""
        config = BatchEvalConfig(max_workers=1)
        assert config.max_workers == 1

        config = BatchEvalConfig(max_workers=32)
        assert config.max_workers == 32

        with pytest.raises(ValueError):
            BatchEvalConfig(max_workers=0)

        with pytest.raises(ValueError):
            BatchEvalConfig(max_workers=100)

    def test_validation_confidence_threshold(self):
        """Test confidence_threshold validation bounds."""
        config = BatchEvalConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0

        config = BatchEvalConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

        with pytest.raises(ValueError):
            BatchEvalConfig(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            BatchEvalConfig(confidence_threshold=1.1)

    def test_ablation_tag_full(self):
        """Test ablation tag when all features enabled."""
        config = BatchEvalConfig()
        assert config.get_ablation_tag() == "full"

    def test_ablation_tag_stage1_only(self):
        """Test ablation tag when Stage 2 disabled."""
        config = BatchEvalConfig(stage2_enabled=False)
        assert config.get_ablation_tag() == "stage1_only"

    def test_ablation_tag_no_views(self):
        """Test ablation tag when views disabled."""
        config = BatchEvalConfig(enable_tool_request_more_views=False)
        assert config.get_ablation_tag() == "no_views"

    def test_ablation_tag_oneshot(self):
        """Test ablation tag for one-shot (max_turns=1)."""
        config = BatchEvalConfig(stage2_max_turns=1)
        assert config.get_ablation_tag() == "oneshot"

    def test_ablation_tag_combined(self):
        """Test ablation tag with multiple features disabled."""
        config = BatchEvalConfig(
            enable_tool_request_crops=False,
            enable_uncertainty_stopping=False,
        )
        tag = config.get_ablation_tag()
        assert "no_crops" in tag
        assert "no_uncertainty" in tag


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager save/load functionality."""

    def test_save_and_load(self, temp_dir, sample_result):
        """Test basic save and load cycle."""
        manager = CheckpointManager(temp_dir, "test_run")

        completed_ids = {"test_001", "test_002"}
        results = [sample_result]

        manager.save(completed_ids, results)

        loaded_ids, loaded_results = manager.load()

        assert loaded_ids == completed_ids
        assert len(loaded_results) == 1
        assert loaded_results[0].sample_id == "test_001"
        assert loaded_results[0].stage1_success is True

    def test_load_nonexistent(self, temp_dir):
        """Test loading from nonexistent checkpoint."""
        manager = CheckpointManager(temp_dir, "nonexistent_run")
        completed_ids, results = manager.load()
        assert completed_ids == set()
        assert results == []

    def test_incremental_save(self, temp_dir):
        """Test incremental checkpoint saves."""
        manager = CheckpointManager(temp_dir, "inc_run")

        # First save
        manager.save({"id1"}, [])

        # Second save with more IDs
        manager.save({"id1", "id2", "id3"}, [])

        loaded_ids, _ = manager.load()
        assert loaded_ids == {"id1", "id2", "id3"}

    def test_thread_safety(self, temp_dir):
        """Test thread-safe save operations."""
        manager = CheckpointManager(temp_dir, "thread_run")
        errors = []

        def save_task(n):
            try:
                manager.save({f"id_{n}"}, [])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_task, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_result_serialization(self, temp_dir, sample_result):
        """Test full result serialization/deserialization."""
        manager = CheckpointManager(temp_dir, "serial_run")

        # Add complex data to result
        sample_result.metrics = {"accuracy": 1.0, "f1": 0.9}
        sample_result.uncertainties = ["low confidence on object color"]

        manager.save({"test_001"}, [sample_result])

        _, loaded = manager.load()

        assert loaded[0].metrics == {"accuracy": 1.0, "f1": 0.9}
        assert loaded[0].uncertainties == ["low confidence on object color"]
        assert loaded[0].stage2_confidence == 0.85


# =============================================================================
# BatchEvaluator Tests
# =============================================================================


class TestBatchEvaluator:
    """Tests for BatchEvaluator main class."""

    def test_initialization(self, batch_config, temp_dir):
        """Test evaluator initialization."""
        batch_config.output_dir = str(temp_dir / "output")
        batch_config.checkpoint_dir = str(temp_dir / "checkpoints")

        evaluator = BatchEvaluator(batch_config)

        assert evaluator.config == batch_config
        assert evaluator.output_dir.exists()

    def test_run_with_mocked_stages(self, batch_config, temp_dir):
        """Test full evaluation run with mocked Stage 1 and Stage 2."""
        batch_config.output_dir = str(temp_dir / "output")
        batch_config.checkpoint_dir = str(temp_dir / "checkpoints")
        batch_config.stage2_enabled = False  # Skip Stage 2 for simpler test
        batch_config.max_samples = 3

        samples = [
            MockEvalSample(_sample_id=f"s{i}", _query=f"Query {i}") for i in range(3)
        ]

        # Mock Stage 1 factory
        mock_selector = MagicMock()
        mock_result = MagicMock()
        mock_result.keyframe_paths = [Path("/tmp/kf1.jpg"), Path("/tmp/kf2.jpg")]
        mock_result.metadata = {"selected_hypothesis_kind": "direct"}
        mock_selector.select_keyframes_v2.return_value = mock_result

        def mock_stage1_factory(scene_id):
            return mock_selector

        evaluator = BatchEvaluator(
            batch_config,
            stage1_factory=mock_stage1_factory,
        )

        def scene_path_provider(scene_id):
            return temp_dir / scene_id

        run_result = evaluator.run(samples, scene_path_provider)

        assert run_result.total_samples == 3
        assert run_result.failed_stage1 == 0

    def test_parallel_execution(self, batch_config, temp_dir):
        """Test that samples are processed in parallel."""
        batch_config.output_dir = str(temp_dir / "output")
        batch_config.max_workers = 3
        batch_config.max_samples = 6
        batch_config.stage2_enabled = False

        execution_times = []
        lock = threading.Lock()

        def slow_stage1_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = []
            result.metadata = {}

            def slow_select(*args, **kwargs):
                with lock:
                    execution_times.append(time.time())
                time.sleep(0.1)
                return result

            mock.select_keyframes_v2 = slow_select
            return mock

        samples = [MockEvalSample(_sample_id=f"s{i}", _query=f"Q{i}") for i in range(6)]

        evaluator = BatchEvaluator(
            batch_config,
            stage1_factory=slow_stage1_factory,
        )

        start = time.time()
        evaluator.run(samples, lambda s: temp_dir)
        elapsed = time.time() - start

        # With parallel execution, 6 samples at 0.1s each with 3 workers
        # should take ~0.2s, not 0.6s
        assert elapsed < 0.5, f"Parallel execution too slow: {elapsed:.2f}s"

    def test_checkpoint_resume(self, batch_config, temp_dir):
        """Test checkpoint-based resumption."""
        batch_config.output_dir = str(temp_dir / "output")
        batch_config.checkpoint_dir = str(temp_dir / "checkpoints")
        batch_config.checkpoint_interval = 2
        batch_config.max_samples = 5
        batch_config.stage2_enabled = False

        # Create checkpoint directory before saving
        checkpoint_path = Path(batch_config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Pre-populate checkpoint
        checkpoint_manager = CheckpointManager(checkpoint_path, batch_config.run_id)
        checkpoint_manager.save(
            {"s0", "s1"},
            [
                EvalSampleResult(
                    sample_id="s0",
                    query="Q0",
                    task_type="qa",
                    scene_id="scene",
                    stage1_success=True,
                ),
                EvalSampleResult(
                    sample_id="s1",
                    query="Q1",
                    task_type="qa",
                    scene_id="scene",
                    stage1_success=True,
                ),
            ],
        )

        processed_ids = []
        lock = threading.Lock()

        def tracking_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = []
            result.metadata = {}

            def track_select(query, **kwargs):
                with lock:
                    # Extract sample ID from context (not directly available)
                    processed_ids.append(query)
                return result

            mock.select_keyframes_v2 = track_select
            return mock

        samples = [MockEvalSample(_sample_id=f"s{i}", _query=f"Q{i}") for i in range(5)]

        evaluator = BatchEvaluator(
            batch_config,
            stage1_factory=tracking_factory,
        )

        run_result = evaluator.run(samples, lambda s: temp_dir)

        # Should only process s2, s3, s4 (s0, s1 from checkpoint)
        assert len(processed_ids) == 3
        assert "Q0" not in processed_ids
        assert "Q1" not in processed_ids

    def test_error_handling(self, batch_config, temp_dir):
        """Test graceful error handling during evaluation."""
        batch_config.output_dir = str(temp_dir / "output")
        batch_config.max_samples = 3
        batch_config.stage2_enabled = False

        def failing_factory(scene_id):
            mock = MagicMock()
            mock.select_keyframes_v2.side_effect = RuntimeError("Stage 1 error")
            return mock

        samples = [MockEvalSample(_sample_id=f"s{i}", _query=f"Q{i}") for i in range(3)]

        evaluator = BatchEvaluator(
            batch_config,
            stage1_factory=failing_factory,
        )

        run_result = evaluator.run(samples, lambda s: temp_dir)

        assert run_result.total_samples == 3
        assert run_result.failed_stage1 == 3
        assert all(r.stage1_error is not None for r in run_result.results)


# =============================================================================
# Sample Adapter Tests
# =============================================================================


class TestOpenEQASampleAdapter:
    """Tests for OpenEQA sample adapter."""

    def test_adapter_properties(self):
        """Test adapter property mapping."""

        @dataclass
        class MockOpenEQASample:
            question_id: str = "q001"
            question: str = "What is on the table?"
            answer: str = "A book"
            scene_id: str = "scene_001"

        mock_sample = MockOpenEQASample()
        adapter = OpenEQASampleAdapter(_sample=mock_sample)

        assert adapter.sample_id == "q001"
        assert adapter.query == "What is on the table?"
        assert adapter.task_type == "qa"
        assert adapter.ground_truth == "A book"
        assert adapter.scene_id == "scene_001"

    def test_adapt_multiple_samples(self):
        """Test adapting a list of samples."""

        @dataclass
        class MockOpenEQASample:
            question_id: str
            question: str
            answer: str
            scene_id: str

        mock_samples = [
            MockOpenEQASample(f"q{i}", f"Question {i}", f"Answer {i}", "scene")
            for i in range(3)
        ]

        adapted = adapt_openeqa_samples(mock_samples)

        assert len(adapted) == 3
        assert adapted[0].sample_id == "q0"
        assert adapted[2].query == "Question 2"


class TestSQA3DSampleAdapter:
    """Tests for SQA3D sample adapter with situation context."""

    def test_adapter_with_situation(self):
        """Test adapter includes situation context in query."""

        @dataclass
        class MockSituation:
            position: str = "(1.5, 2.0, 0.5)"
            orientation: str = "facing north"
            room_description: str = "living room"

        @dataclass
        class MockSQA3DSample:
            question_id: str = "sq001"
            question: str = "What is in front of me?"
            answers: List[str] = None
            scene_id: str = "scene_001"
            situation: MockSituation = None

            def __post_init__(self):
                if self.answers is None:
                    self.answers = ["sofa", "couch"]
                if self.situation is None:
                    self.situation = MockSituation()

        mock_sample = MockSQA3DSample()
        adapter = SQA3DSampleAdapter(_sample=mock_sample)

        assert adapter.sample_id == "sq001"
        assert "(1.5, 2.0, 0.5)" in adapter.query
        assert "facing north" in adapter.query
        assert "living room" in adapter.query
        assert "What is in front of me?" in adapter.query
        assert adapter.ground_truth == ["sofa", "couch"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for batch evaluation pipeline."""

    def test_full_pipeline_mock(self, temp_dir):
        """Test complete pipeline with fully mocked components."""
        config = BatchEvalConfig(
            run_id="integration_test",
            output_dir=str(temp_dir / "output"),
            checkpoint_dir=str(temp_dir / "checkpoints"),
            max_workers=2,
            max_samples=3,
            stage2_enabled=False,  # Simplify for integration test
            save_raw_outputs=True,
            save_tool_traces=True,
        )

        # Create mock Stage 1
        def mock_stage1_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = [
                Path(f"/tmp/{scene_id}/kf1.jpg"),
                Path(f"/tmp/{scene_id}/kf2.jpg"),
            ]
            result.metadata = {"selected_hypothesis_kind": "proxy"}
            mock.select_keyframes_v2.return_value = result
            return mock

        samples = [
            MockEvalSample(
                _sample_id=f"int_{i}",
                _query=f"Integration query {i}",
                _scene_id=f"scene_{i % 2}",
            )
            for i in range(3)
        ]

        evaluator = BatchEvaluator(
            config,
            stage1_factory=mock_stage1_factory,
        )

        run_result = evaluator.run(
            samples,
            lambda scene_id: temp_dir / scene_id,
        )

        # Verify results
        assert run_result.total_samples == 3
        assert run_result.successful_samples == 3
        assert run_result.failed_stage1 == 0
        assert run_result.benchmark_name == "openeqa"

        # Verify output files
        results_file = temp_dir / "output" / f"eval_{config.run_id}.json"
        assert results_file.exists()

        with open(results_file) as f:
            saved_data = json.load(f)
        assert saved_data["summary"]["total_samples"] == 3

        summary_file = temp_dir / "output" / f"summary_{config.run_id}.txt"
        assert summary_file.exists()

    def test_aggregate_statistics(self, temp_dir):
        """Test aggregate statistics computation."""
        config = BatchEvalConfig(
            run_id="stats_test",
            output_dir=str(temp_dir / "output"),
            max_samples=5,
            stage2_enabled=False,
        )

        # Track latencies
        call_count = [0]

        def variadic_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = [Path("/tmp/kf.jpg")]
            result.metadata = {}

            def timed_call(*args, **kwargs):
                call_count[0] += 1
                time.sleep(0.05)  # 50ms per call
                return result

            mock.select_keyframes_v2 = timed_call
            return mock

        samples = [MockEvalSample(_sample_id=f"s{i}", _query=f"Q{i}") for i in range(5)]

        evaluator = BatchEvaluator(config, stage1_factory=variadic_factory)
        run_result = evaluator.run(samples, lambda s: temp_dir)

        # All should have measurable latency
        assert all(r.stage1_latency_ms > 0 for r in run_result.results)
        assert run_result.avg_stage1_latency_ms > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_samples(self, temp_dir):
        """Test handling of empty sample list."""
        config = BatchEvalConfig(
            output_dir=str(temp_dir / "output"),
        )
        evaluator = BatchEvaluator(config)
        run_result = evaluator.run([], lambda s: temp_dir)

        assert run_result.total_samples == 0
        assert run_result.successful_samples == 0

    def test_skip_samples(self, temp_dir):
        """Test skip_samples configuration."""
        config = BatchEvalConfig(
            output_dir=str(temp_dir / "output"),
            skip_samples=3,
            max_samples=2,
            stage2_enabled=False,
        )

        def tracking_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = []
            result.metadata = {}
            mock.select_keyframes_v2.return_value = result
            return mock

        samples = [
            MockEvalSample(_sample_id=f"s{i}", _query=f"Q{i}") for i in range(10)
        ]

        evaluator = BatchEvaluator(config, stage1_factory=tracking_factory)
        run_result = evaluator.run(samples, lambda s: temp_dir)

        # Should skip first 3, then take 2
        assert run_result.total_samples == 2

    def test_single_sample(self, temp_dir):
        """Test evaluation with single sample."""
        config = BatchEvalConfig(
            output_dir=str(temp_dir / "output"),
            max_workers=1,
            stage2_enabled=False,
        )

        def mock_factory(scene_id):
            mock = MagicMock()
            result = MagicMock()
            result.keyframe_paths = [Path("/tmp/kf.jpg")]
            result.metadata = {"selected_hypothesis_kind": "context"}
            mock.select_keyframes_v2.return_value = result
            return mock

        samples = [MockEvalSample(_sample_id="single", _query="Single query")]

        evaluator = BatchEvaluator(config, stage1_factory=mock_factory)
        run_result = evaluator.run(samples, lambda s: temp_dir)

        assert run_result.total_samples == 1
        assert run_result.results[0].stage1_hypothesis_kind == "context"
