"""Tests for trace server integration with evaluation pipeline.

Covers:
- EvalTraceMetadata creation and serialization
- EvalTraceManager lifecycle (start/finish traces)
- Trace-sample linking
- Statistics export
- CSV export
- Integration with BatchEvaluator via mixin
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from conceptgraph.agents.trace_server import TraceDB, TraceRecord
from conceptgraph.evaluation.trace_integration import (
    EvalTraceMetadata,
    EvalTraceManager,
    TracingBatchEvaluatorMixin,
    create_tracing_evaluator,
    export_run_trace_report,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary path for test database."""
    return tmp_path / "test_traces.db"


@pytest.fixture
def trace_db(temp_db_path: Path) -> TraceDB:
    """Create a TraceDB instance for testing."""
    return TraceDB(temp_db_path)


@pytest.fixture
def eval_metadata() -> EvalTraceMetadata:
    """Create sample evaluation metadata."""
    return EvalTraceMetadata(
        benchmark_name="openeqa",
        sample_id="sample_001",
        scene_id="scene_kitchen",
        task_type="qa",
        query="What color is the chair?",
        ground_truth="brown",
        ablation_tag="full_tools",
        run_id="test_run_001",
    )


@pytest.fixture
def trace_manager(trace_db: TraceDB) -> EvalTraceManager:
    """Create an EvalTraceManager for testing."""
    return EvalTraceManager(
        db=trace_db,
        benchmark_name="openeqa",
        run_id="test_run_001",
        ablation_tag="full_tools",
    )


# ============================================================================
# EvalTraceMetadata Tests
# ============================================================================


class TestEvalTraceMetadata:
    """Tests for EvalTraceMetadata dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test creating metadata with all fields populated."""
        metadata = EvalTraceMetadata(
            benchmark_name="sqa3d",
            sample_id="sample_123",
            scene_id="scene_0001",
            task_type="visual_grounding",
            query="Find the red cube",
            ground_truth={"bbox": [1, 2, 3, 4]},
            ablation_tag="no_crops",
            run_id="run_20260320",
        )

        assert metadata.benchmark_name == "sqa3d"
        assert metadata.sample_id == "sample_123"
        assert metadata.scene_id == "scene_0001"
        assert metadata.task_type == "visual_grounding"
        assert metadata.query == "Find the red cube"
        assert metadata.ground_truth == {"bbox": [1, 2, 3, 4]}
        assert metadata.ablation_tag == "no_crops"
        assert metadata.run_id == "run_20260320"

    def test_create_with_defaults(self) -> None:
        """Test creating metadata with default optional fields."""
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="sample_001",
            scene_id="scene_001",
            task_type="qa",
            query="Test query",
            ground_truth="answer",
        )

        assert metadata.ablation_tag == ""
        assert metadata.run_id == ""

    def test_ground_truth_types(self) -> None:
        """Test that ground_truth accepts various types."""
        # String
        m1 = EvalTraceMetadata(
            benchmark_name="test",
            sample_id="s1",
            scene_id="sc1",
            task_type="qa",
            query="q",
            ground_truth="yes",
        )
        assert m1.ground_truth == "yes"

        # Dict
        m2 = EvalTraceMetadata(
            benchmark_name="test",
            sample_id="s2",
            scene_id="sc2",
            task_type="grounding",
            query="q",
            ground_truth={"x": 1, "y": 2},
        )
        assert m2.ground_truth == {"x": 1, "y": 2}

        # List
        m3 = EvalTraceMetadata(
            benchmark_name="test",
            sample_id="s3",
            scene_id="sc3",
            task_type="multi",
            query="q",
            ground_truth=["a", "b", "c"],
        )
        assert m3.ground_truth == ["a", "b", "c"]


# ============================================================================
# EvalTraceManager Tests
# ============================================================================


class TestEvalTraceManager:
    """Tests for EvalTraceManager class."""

    def test_init_with_auto_run_id(self, trace_db: TraceDB) -> None:
        """Test that run_id is auto-generated if not provided."""
        manager = EvalTraceManager(db=trace_db, benchmark_name="openeqa")

        assert manager.run_id is not None
        assert len(manager.run_id) > 0
        assert "_" in manager.run_id  # Format: YYYYMMDD_HHMMSS

    def test_init_with_explicit_values(self, trace_db: TraceDB) -> None:
        """Test initialization with explicit values."""
        manager = EvalTraceManager(
            db=trace_db,
            benchmark_name="sqa3d",
            run_id="custom_run_123",
            ablation_tag="ablation_v2",
        )

        assert manager.benchmark_name == "sqa3d"
        assert manager.run_id == "custom_run_123"
        assert manager.ablation_tag == "ablation_v2"

    def test_start_trace(
        self, trace_manager: EvalTraceManager, eval_metadata: EvalTraceMetadata
    ) -> None:
        """Test starting a new trace."""
        trace_id = trace_manager.start_trace(eval_metadata)

        assert trace_id is not None
        assert trace_id.startswith("eval_")
        assert trace_id in trace_manager._active_traces

        # Verify trace was inserted to DB
        trace = trace_manager.db.get_trace(trace_id)
        assert trace is not None
        assert trace.task_type == "qa"
        assert trace.user_query == "What color is the chair?"

    def test_start_trace_metadata_merging(
        self, trace_db: TraceDB, eval_metadata: EvalTraceMetadata
    ) -> None:
        """Test that metadata is properly merged with manager defaults."""
        manager = EvalTraceManager(
            db=trace_db,
            benchmark_name="default_bench",
            run_id="default_run",
            ablation_tag="default_ablation",
        )

        # Metadata has its own values
        trace_id = manager.start_trace(eval_metadata)
        trace = manager.db.get_trace(trace_id)
        metadata = json.loads(trace.metadata_json)

        # Metadata values should override manager defaults
        assert metadata["benchmark_name"] == "openeqa"  # From metadata
        assert metadata["ablation_tag"] == "full_tools"  # From metadata
        assert metadata["run_id"] == "test_run_001"  # From metadata

    def test_start_trace_uses_manager_defaults(self, trace_db: TraceDB) -> None:
        """Test that manager defaults are used when metadata fields are empty."""
        manager = EvalTraceManager(
            db=trace_db,
            benchmark_name="manager_bench",
            run_id="manager_run",
            ablation_tag="manager_ablation",
        )

        metadata = EvalTraceMetadata(
            benchmark_name="",  # Empty - should use manager default
            sample_id="s1",
            scene_id="sc1",
            task_type="qa",
            query="test",
            ground_truth="answer",
            ablation_tag="",  # Empty - should use manager default
            run_id="",  # Empty - should use manager default
        )

        trace_id = manager.start_trace(metadata)
        trace = manager.db.get_trace(trace_id)
        stored_meta = json.loads(trace.metadata_json)

        assert stored_meta["benchmark_name"] == "manager_bench"
        assert stored_meta["ablation_tag"] == "manager_ablation"
        assert stored_meta["run_id"] == "manager_run"

    def test_finish_trace(
        self, trace_manager: EvalTraceManager, eval_metadata: EvalTraceMetadata
    ) -> None:
        """Test finishing a trace with results."""
        trace_id = trace_manager.start_trace(eval_metadata)

        trace_manager.finish_trace(
            trace_id=trace_id,
            status="completed",
            confidence=0.85,
            summary="The chair is brown.",
            tool_trace=[{"tool_name": "request_crops", "success": True}],
            keyframe_count=5,
            prediction="brown",
            metrics={"exact_match": 1.0, "llm_score": 0.9},
        )

        # Verify trace was updated
        trace = trace_manager.db.get_trace(trace_id)
        assert trace.status == "completed"
        assert trace.confidence == 0.85
        assert trace.summary == "The chair is brown."
        assert trace.num_tool_calls == 1
        assert trace.final_keyframe_count == 5

        # Verify metadata was updated with prediction and metrics
        metadata = json.loads(trace.metadata_json)
        assert metadata["prediction"] == "brown"
        assert metadata["metrics"]["exact_match"] == 1.0
        assert metadata["metrics"]["llm_score"] == 0.9

    def test_finish_trace_with_error(
        self, trace_manager: EvalTraceManager, eval_metadata: EvalTraceMetadata
    ) -> None:
        """Test finishing a trace with an error."""
        trace_id = trace_manager.start_trace(eval_metadata)

        trace_manager.finish_trace(
            trace_id=trace_id,
            status="failed",
            error="API timeout after 30s",
        )

        trace = trace_manager.db.get_trace(trace_id)
        assert trace.status == "failed"

        metadata = json.loads(trace.metadata_json)
        assert metadata["error"] == "API timeout after 30s"

    def test_finish_unknown_trace(self, trace_manager: EvalTraceManager) -> None:
        """Test finishing a trace that doesn't exist."""
        # Should not raise, just log warning
        trace_manager.finish_trace(
            trace_id="nonexistent_trace_id",
            status="completed",
        )

    def test_finish_trace_removes_from_active(
        self, trace_manager: EvalTraceManager, eval_metadata: EvalTraceMetadata
    ) -> None:
        """Test that finished traces are removed from active traces."""
        trace_id = trace_manager.start_trace(eval_metadata)
        assert trace_id in trace_manager._active_traces

        trace_manager.finish_trace(trace_id=trace_id, status="completed")
        assert trace_id not in trace_manager._active_traces

    def test_get_run_traces(self, trace_manager: EvalTraceManager) -> None:
        """Test retrieving all traces for a run."""
        # Create multiple traces
        for i in range(5):
            metadata = EvalTraceMetadata(
                benchmark_name="openeqa",
                sample_id=f"sample_{i}",
                scene_id=f"scene_{i}",
                task_type="qa",
                query=f"Query {i}",
                ground_truth=f"answer_{i}",
            )
            trace_id = trace_manager.start_trace(metadata)
            trace_manager.finish_trace(trace_id=trace_id, status="completed")

        traces = trace_manager.get_run_traces()
        assert len(traces) == 5

    def test_get_run_traces_filters_by_run_id(self, trace_db: TraceDB) -> None:
        """Test that get_run_traces only returns traces for specified run."""
        manager1 = EvalTraceManager(db=trace_db, run_id="run_A")
        manager2 = EvalTraceManager(db=trace_db, run_id="run_B")

        # Create traces for run_A
        for i in range(3):
            metadata = EvalTraceMetadata(
                benchmark_name="test",
                sample_id=f"a_{i}",
                scene_id="sc",
                task_type="qa",
                query="q",
                ground_truth="a",
            )
            tid = manager1.start_trace(metadata)
            manager1.finish_trace(tid, status="completed")

        # Create traces for run_B
        for i in range(2):
            metadata = EvalTraceMetadata(
                benchmark_name="test",
                sample_id=f"b_{i}",
                scene_id="sc",
                task_type="qa",
                query="q",
                ground_truth="a",
            )
            tid = manager2.start_trace(metadata)
            manager2.finish_trace(tid, status="completed")

        assert len(manager1.get_run_traces()) == 3
        assert len(manager2.get_run_traces()) == 2

    def test_get_sample_trace(
        self, trace_manager: EvalTraceManager
    ) -> None:
        """Test retrieving trace by sample ID."""
        # Create metadata with sample_id in the user_query so search can find it
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="sample_findme_001",
            scene_id="scene_kitchen",
            task_type="qa",
            query="sample_findme_001: What color is the chair?",  # Include sample_id in query
            ground_truth="brown",
        )
        trace_id = trace_manager.start_trace(metadata)
        trace_manager.finish_trace(trace_id=trace_id, status="completed")

        result = trace_manager.get_sample_trace("sample_findme_001")
        assert result is not None
        assert result.trace_id == trace_id

    def test_get_sample_trace_not_found(self, trace_manager: EvalTraceManager) -> None:
        """Test retrieving non-existent sample returns None."""
        result = trace_manager.get_sample_trace("nonexistent_sample")
        assert result is None


# ============================================================================
# Statistics Export Tests
# ============================================================================


class TestExportStatistics:
    """Tests for statistics export functionality."""

    def test_export_empty_run(self, trace_manager: EvalTraceManager) -> None:
        """Test exporting statistics for run with no traces."""
        stats = trace_manager.export_statistics()

        assert stats["run_id"] == "test_run_001"
        assert stats["total_traces"] == 0
        assert stats["status_distribution"] == {}
        assert stats["avg_confidence"] == 0.0
        assert stats["avg_duration_ms"] == 0.0

    def test_export_statistics_aggregation(
        self, trace_manager: EvalTraceManager
    ) -> None:
        """Test that statistics are correctly aggregated."""
        # Create traces with various statuses
        statuses = ["completed", "completed", "failed", "insufficient_evidence"]
        for i, status in enumerate(statuses):
            metadata = EvalTraceMetadata(
                benchmark_name="openeqa",
                sample_id=f"sample_{i}",
                scene_id=f"scene_{i}",
                task_type="qa",
                query=f"Query {i}",
                ground_truth="answer",
            )
            trace_id = trace_manager.start_trace(metadata)
            trace_manager.finish_trace(
                trace_id=trace_id,
                status=status,
                confidence=0.8 if status == "completed" else 0.0,
                metrics={"accuracy": 1.0 if status == "completed" else 0.0},
            )

        stats = trace_manager.export_statistics()

        assert stats["total_traces"] == 4
        assert stats["completed_count"] == 2
        assert stats["failed_count"] == 1
        assert stats["insufficient_evidence_count"] == 1
        assert stats["status_distribution"]["completed"] == 2
        assert stats["status_distribution"]["failed"] == 1

    def test_export_tool_usage_distribution(
        self, trace_manager: EvalTraceManager
    ) -> None:
        """Test that tool usage is correctly tracked."""
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="sample_1",
            scene_id="scene_1",
            task_type="qa",
            query="Test query",
            ground_truth="answer",
        )
        trace_id = trace_manager.start_trace(metadata)
        trace_manager.finish_trace(
            trace_id=trace_id,
            status="completed",
            tool_trace=[
                {"tool_name": "request_crops"},
                {"tool_name": "request_more_views"},
                {"tool_name": "request_crops"},
            ],
        )

        stats = trace_manager.export_statistics()

        # Tool usage is tracked from tool_trace
        assert stats["tool_usage_distribution"].get("request_crops", 0) == 2
        assert stats["tool_usage_distribution"].get("request_more_views", 0) == 1


# ============================================================================
# CSV Export Tests
# ============================================================================


class TestCsvExport:
    """Tests for CSV export functionality."""

    def test_export_to_csv(
        self, trace_manager: EvalTraceManager, tmp_path: Path
    ) -> None:
        """Test exporting traces to CSV."""
        # Create some traces
        for i in range(3):
            metadata = EvalTraceMetadata(
                benchmark_name="openeqa",
                sample_id=f"sample_{i}",
                scene_id=f"scene_{i}",
                task_type="qa",
                query=f"Query {i}",
                ground_truth="answer",
            )
            trace_id = trace_manager.start_trace(metadata)
            trace_manager.finish_trace(
                trace_id=trace_id,
                status="completed",
                confidence=0.85,
            )

        csv_path = tmp_path / "traces.csv"
        result_path = trace_manager.export_to_csv(csv_path)

        assert result_path.exists()

        # Verify CSV content
        with open(result_path) as f:
            lines = f.readlines()

        assert len(lines) == 4  # Header + 3 traces
        assert "trace_id" in lines[0]
        assert "sample_id" in lines[0]
        assert "status" in lines[0]

    def test_export_csv_empty_run(
        self, trace_manager: EvalTraceManager, tmp_path: Path
    ) -> None:
        """Test exporting empty run creates valid CSV with header only."""
        csv_path = tmp_path / "empty_traces.csv"
        result_path = trace_manager.export_to_csv(csv_path)

        assert result_path.exists()
        with open(result_path) as f:
            lines = f.readlines()

        assert len(lines) == 1  # Header only


# ============================================================================
# TracingBatchEvaluatorMixin Tests
# ============================================================================


class TestTracingBatchEvaluatorMixin:
    """Tests for the tracing mixin."""

    def test_init_tracing(self, trace_db: TraceDB) -> None:
        """Test initializing tracing on a mixin instance."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.config.run_id = "test_run"
                self.config.benchmark_name = "openeqa"
                self.config.get_ablation_tag.return_value = "full"

        evaluator = MockEvaluator()
        evaluator.init_tracing(db=trace_db, benchmark_name="openeqa")

        assert evaluator._trace_manager is not None
        assert evaluator._trace_manager.benchmark_name == "openeqa"

    def test_record_sample_trace(self, trace_db: TraceDB) -> None:
        """Test recording a sample trace via mixin."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.config.run_id = "test_run"
                self.config.get_ablation_tag.return_value = ""

        evaluator = MockEvaluator()
        evaluator.init_tracing(db=trace_db, benchmark_name="test")

        trace_id = evaluator._record_sample_trace(
            sample_id="s1",
            query="What color?",
            task_type="qa",
            scene_id="scene1",
            ground_truth="red",
        )

        assert trace_id != ""
        assert trace_id.startswith("eval_")

    def test_record_sample_trace_without_init(self) -> None:
        """Test that recording without init returns empty string."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            pass

        evaluator = MockEvaluator()
        trace_id = evaluator._record_sample_trace(
            sample_id="s1",
            query="What color?",
            task_type="qa",
            scene_id="scene1",
            ground_truth="red",
        )

        assert trace_id == ""

    def test_finish_sample_trace(self, trace_db: TraceDB) -> None:
        """Test finishing a sample trace via mixin."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.config.run_id = "test_run"
                self.config.get_ablation_tag.return_value = ""

        evaluator = MockEvaluator()
        evaluator.init_tracing(db=trace_db, benchmark_name="test")

        trace_id = evaluator._record_sample_trace(
            sample_id="s1",
            query="What color?",
            task_type="qa",
            scene_id="scene1",
            ground_truth="red",
        )

        # Mock result object
        result = MagicMock()
        result.stage2_status = "completed"
        result.stage2_success = True
        result.stage2_confidence = 0.9
        result.stage2_answer = "The color is red"
        result.tool_trace = []
        result.stage1_keyframe_count = 3
        result.prediction = "red"
        result.metrics = {"accuracy": 1.0}
        result.stage1_error = None
        result.stage2_error = None

        evaluator._finish_sample_trace(trace_id, result)

        # Verify trace was finished
        trace = trace_db.get_trace(trace_id)
        assert trace.status == "completed"
        assert trace.confidence == 0.9

    def test_get_trace_statistics(self, trace_db: TraceDB) -> None:
        """Test getting trace statistics via mixin."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.config.run_id = "test_run"
                self.config.get_ablation_tag.return_value = ""

        evaluator = MockEvaluator()
        evaluator.init_tracing(db=trace_db, benchmark_name="test")

        stats = evaluator.get_trace_statistics()
        assert stats is not None
        assert "total_traces" in stats

    def test_get_trace_statistics_without_init(self) -> None:
        """Test getting statistics without init returns None."""

        class MockEvaluator(TracingBatchEvaluatorMixin):
            pass

        evaluator = MockEvaluator()
        assert evaluator.get_trace_statistics() is None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateTracingEvaluator:
    """Tests for create_tracing_evaluator factory function."""

    def test_create_tracing_evaluator(self, tmp_path: Path) -> None:
        """Test creating evaluator with tracing enabled."""
        db_path = tmp_path / "traces.db"

        # Mock BatchEvalConfig
        config = MagicMock()
        config.benchmark_name = "openeqa"
        config.run_id = "test_run_123"
        config.get_ablation_tag.return_value = "full_tools"

        # The BatchEvaluator is imported inside the function with
        # `from .batch_eval import BatchEvaluator`, so we patch it there
        with patch(
            "conceptgraph.evaluation.batch_eval.BatchEvaluator"
        ) as MockEvaluator:
            # Setup mock
            MockEvaluator.return_value = MagicMock()

            evaluator, manager = create_tracing_evaluator(
                config=config,
                trace_db_path=db_path,
            )

            assert manager is not None
            assert manager.benchmark_name == "openeqa"
            assert manager.run_id == "test_run_123"
            assert manager.ablation_tag == "full_tools"

            # Verify BatchEvaluator was called
            MockEvaluator.assert_called_once()


# ============================================================================
# Report Export Tests
# ============================================================================


class TestExportRunTraceReport:
    """Tests for comprehensive report export."""

    def test_export_run_trace_report(self, trace_db: TraceDB, tmp_path: Path) -> None:
        """Test exporting full report for a run."""
        # Create a manager and some traces
        manager = EvalTraceManager(
            db=trace_db,
            benchmark_name="openeqa",
            run_id="report_test_run",
        )

        # Add traces
        for i in range(3):
            metadata = EvalTraceMetadata(
                benchmark_name="openeqa",
                sample_id=f"sample_{i}",
                scene_id=f"scene_{i}",
                task_type="qa",
                query=f"Query {i}",
                ground_truth="answer",
            )
            trace_id = manager.start_trace(metadata)
            manager.finish_trace(trace_id=trace_id, status="completed")

        # Export report
        outputs = export_run_trace_report(
            db=trace_db,
            run_id="report_test_run",
            output_dir=tmp_path / "report",
            benchmark_name="openeqa",
        )

        # Verify all outputs exist
        assert "statistics" in outputs
        assert "csv" in outputs
        assert "traces" in outputs

        assert outputs["statistics"].exists()
        assert outputs["csv"].exists()
        assert outputs["traces"].exists()

        # Verify statistics content
        with open(outputs["statistics"]) as f:
            stats = json.load(f)
        assert stats["total_traces"] == 3

        # Verify full traces content
        with open(outputs["traces"]) as f:
            traces = json.load(f)
        assert len(traces) == 3

    def test_export_report_creates_output_dir(
        self, trace_db: TraceDB, tmp_path: Path
    ) -> None:
        """Test that export creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "nested" / "dir"
        assert not output_dir.exists()

        outputs = export_run_trace_report(
            db=trace_db,
            run_id="test_run",
            output_dir=output_dir,
        )

        assert output_dir.exists()
        assert outputs["statistics"].exists()


# ============================================================================
# Academic Alignment Tests
# ============================================================================


class TestAcademicAlignment:
    """Tests verifying academic research alignment."""

    def test_trace_captures_evidence_acquisition_pattern(
        self, trace_manager: EvalTraceManager
    ) -> None:
        """Test that traces capture evidence acquisition patterns for paper claims."""
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="academic_test_1",
            scene_id="scene_1",
            task_type="qa",
            query="What is on the table?",
            ground_truth="book",
        )

        trace_id = trace_manager.start_trace(metadata)
        trace_manager.finish_trace(
            trace_id=trace_id,
            status="completed",
            tool_trace=[
                {"tool_name": "request_more_views", "success": True},
                {"tool_name": "request_crops", "object_id": "obj_123"},
                {"tool_name": "switch_or_expand_hypothesis", "action": "expand"},
            ],
        )

        trace = trace_manager.db.get_trace(trace_id)
        assert trace.num_tool_calls == 3  # Evidence acquisition actions tracked

    def test_trace_captures_uncertainty(self, trace_manager: EvalTraceManager) -> None:
        """Test that traces capture uncertainty output for calibration analysis."""
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="uncertainty_test",
            scene_id="scene_1",
            task_type="qa",
            query="Is there a cat in the room?",
            ground_truth="yes",
        )

        trace_id = trace_manager.start_trace(metadata)
        trace_manager.finish_trace(
            trace_id=trace_id,
            status="insufficient_evidence",
            confidence=0.35,  # Low confidence
            summary="Cannot determine with available evidence",
        )

        trace = trace_manager.db.get_trace(trace_id)
        assert trace.status == "insufficient_evidence"
        assert trace.confidence == 0.35

        # This supports "evidence-grounded uncertainty" claim
        stats = trace_manager.export_statistics()
        assert stats["insufficient_evidence_count"] == 1

    def test_metrics_support_benchmark_comparison(
        self, trace_manager: EvalTraceManager
    ) -> None:
        """Test that exported metrics support cross-benchmark comparison."""
        # Create traces with benchmark-specific metrics
        metadata = EvalTraceMetadata(
            benchmark_name="openeqa",
            sample_id="metric_test",
            scene_id="scene_1",
            task_type="qa",
            query="Test",
            ground_truth="answer",
        )

        trace_id = trace_manager.start_trace(metadata)
        trace_manager.finish_trace(
            trace_id=trace_id,
            status="completed",
            metrics={
                "exact_match": 1.0,
                "llm_score": 0.87,
                "f1_score": 0.92,
            },
        )

        stats = trace_manager.export_statistics()
        assert "avg_metrics" in stats
        assert stats["avg_metrics"]["exact_match"] == 1.0
        assert stats["avg_metrics"]["llm_score"] == 0.87
