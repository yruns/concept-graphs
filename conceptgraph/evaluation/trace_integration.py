"""Trace server integration for evaluation pipeline.

This module bridges the evaluation system with the trace server, enabling:
- Auto-save traces during evaluation runs
- Link traces to benchmark samples
- Export trace statistics for analysis

Academic Alignment:
- Evidence acquisition patterns are captured in traces (supports "adaptive evidence" claim)
- Tool traces enable "symbolic-to-visual repair" analysis
- Uncertainty outputs are persisted for calibration analysis
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger

from conceptgraph.agents.trace_server import TraceDB, TraceRecord


@dataclass
class EvalTraceMetadata:
    """Metadata linking a trace to a benchmark sample.

    This enables correlating agent behavior (traces) with
    benchmark performance (metrics) for academic analysis.
    """

    benchmark_name: str
    sample_id: str
    scene_id: str
    task_type: str
    query: str
    ground_truth: Any
    ablation_tag: str = ""
    run_id: str = ""


class EvalTraceManager:
    """Manages trace recording during evaluation runs.

    Integrates with TraceDB to auto-save traces and link them
    to benchmark samples for post-hoc analysis.

    Usage:
        db = TraceDB("eval_traces.db")
        manager = EvalTraceManager(db, benchmark_name="openeqa", run_id="20260320")

        # During evaluation
        trace_id = manager.start_trace(metadata)
        # ... agent execution ...
        manager.finish_trace(trace_id, result_data)

        # Export statistics
        stats = manager.export_statistics()
    """

    def __init__(
        self,
        db: TraceDB,
        benchmark_name: str = "",
        run_id: str = "",
        ablation_tag: str = "",
    ) -> None:
        """Initialize the trace manager.

        Args:
            db: TraceDB instance for persistence.
            benchmark_name: Name of the benchmark being evaluated.
            run_id: Unique identifier for this evaluation run.
            ablation_tag: Tag describing the ablation configuration.
        """
        self.db = db
        self.benchmark_name = benchmark_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ablation_tag = ablation_tag
        self._active_traces: Dict[str, float] = {}  # trace_id -> start_time

    def start_trace(self, metadata: EvalTraceMetadata) -> str:
        """Start recording a new trace for a benchmark sample.

        Args:
            metadata: Metadata linking trace to sample.

        Returns:
            Unique trace ID for this trace.
        """
        trace_id = f"eval_{self.run_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        self._active_traces[trace_id] = start_time

        # Merge metadata
        full_metadata = {
            "benchmark_name": metadata.benchmark_name or self.benchmark_name,
            "sample_id": metadata.sample_id,
            "scene_id": metadata.scene_id,
            "task_type": metadata.task_type,
            "ablation_tag": metadata.ablation_tag or self.ablation_tag,
            "run_id": metadata.run_id or self.run_id,
            "ground_truth": metadata.ground_truth,
        }

        # Create initial trace record
        trace_record = TraceRecord(
            trace_id=trace_id,
            created_at=start_time,
            task_type=metadata.task_type,
            user_query=metadata.query,
            scene_id=metadata.scene_id,
            metadata_json=json.dumps(full_metadata),
        )

        self.db.insert_trace(trace_record)
        logger.debug(
            f"[EvalTraceManager] Started trace {trace_id} for {metadata.sample_id}"
        )

        return trace_id

    def finish_trace(
        self,
        trace_id: str,
        status: str = "completed",
        confidence: float = 0.0,
        summary: str = "",
        tool_trace: Optional[List[Dict[str, Any]]] = None,
        keyframe_count: int = 0,
        prediction: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Finish recording a trace with final results.

        Args:
            trace_id: Trace ID from start_trace.
            status: Final status (completed, failed, insufficient_evidence).
            confidence: Agent's confidence score.
            summary: Agent's summary response.
            tool_trace: List of tool invocations.
            keyframe_count: Number of keyframes used.
            prediction: Agent's prediction/answer.
            metrics: Benchmark-specific evaluation metrics.
            error: Error message if failed.
        """
        if trace_id not in self._active_traces:
            logger.warning(f"[EvalTraceManager] Unknown trace_id: {trace_id}")
            return

        start_time = self._active_traces.pop(trace_id)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Update trace record
        updates: Dict[str, Any] = {
            "finished_at": end_time,
            "status": status,
            "confidence": confidence,
            "summary": summary or str(prediction) if prediction else "",
            "num_tool_calls": len(tool_trace) if tool_trace else 0,
            "duration_ms": duration_ms,
            "final_keyframe_count": keyframe_count,
        }

        if tool_trace:
            updates["tool_trace_json"] = json.dumps(tool_trace)

        # Store metrics and prediction in metadata
        existing = self.db.get_trace(trace_id)
        if existing:
            try:
                existing_meta = json.loads(existing.metadata_json or "{}")
            except json.JSONDecodeError:
                existing_meta = {}
            existing_meta["prediction"] = prediction
            existing_meta["metrics"] = metrics or {}
            if error:
                existing_meta["error"] = error
            updates["metadata_json"] = json.dumps(existing_meta)

        self.db.update_trace(trace_id, **updates)
        logger.debug(
            f"[EvalTraceManager] Finished trace {trace_id}: {status}, {duration_ms:.0f}ms"
        )

    def get_run_traces(
        self,
        run_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[TraceRecord]:
        """Get all traces for a specific run.

        Args:
            run_id: Run ID to filter by (defaults to current run).
            limit: Maximum number of traces to return.

        Returns:
            List of trace records for the run.
        """
        target_run_id = run_id or self.run_id

        # Use search to filter by run_id in metadata
        all_traces = self.db.list_traces(limit=limit)
        run_traces = []

        for trace in all_traces:
            try:
                metadata = json.loads(trace.metadata_json or "{}")
                if metadata.get("run_id") == target_run_id:
                    run_traces.append(trace)
            except json.JSONDecodeError:
                continue

        return run_traces

    def get_sample_trace(self, sample_id: str) -> Optional[TraceRecord]:
        """Get trace for a specific benchmark sample.

        Args:
            sample_id: Benchmark sample ID.

        Returns:
            Trace record if found, None otherwise.
        """
        # Search through recent traces
        traces = self.db.list_traces(limit=1000, search=sample_id)

        for trace in traces:
            try:
                metadata = json.loads(trace.metadata_json or "{}")
                if metadata.get("sample_id") == sample_id:
                    return trace
            except json.JSONDecodeError:
                continue

        return None

    def export_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Export statistics for traces in a run.

        Args:
            run_id: Run ID to get stats for (defaults to current run).

        Returns:
            Dictionary with aggregate statistics.
        """
        traces = self.get_run_traces(run_id)

        if not traces:
            return {
                "run_id": run_id or self.run_id,
                "total_traces": 0,
                "status_distribution": {},
                "avg_confidence": 0.0,
                "avg_duration_ms": 0.0,
                "avg_tool_calls": 0.0,
                "tool_usage_distribution": {},
                "task_type_distribution": {},
            }

        # Compute statistics
        status_dist: Dict[str, int] = {}
        task_type_dist: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}
        confidences: List[float] = []
        durations: List[float] = []
        tool_call_counts: List[int] = []

        for trace in traces:
            # Status distribution
            status_dist[trace.status] = status_dist.get(trace.status, 0) + 1

            # Task type distribution
            task_type_dist[trace.task_type] = task_type_dist.get(trace.task_type, 0) + 1

            # Confidence and duration
            if trace.status == "completed":
                confidences.append(trace.confidence)
            durations.append(trace.duration_ms)
            tool_call_counts.append(trace.num_tool_calls)

            # Tool usage from traces
            for tc in trace.tool_trace:
                tool_name = tc.get("tool_name", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Metrics with benchmark annotations
        metrics_agg: Dict[str, List[float]] = {}
        for trace in traces:
            try:
                metadata = json.loads(trace.metadata_json or "{}")
                trace_metrics = metadata.get("metrics", {})
                for key, value in trace_metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in metrics_agg:
                            metrics_agg[key] = []
                        metrics_agg[key].append(value)
            except json.JSONDecodeError:
                continue

        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in metrics_agg.items()
            if values
        }

        return {
            "run_id": run_id or self.run_id,
            "benchmark_name": self.benchmark_name,
            "ablation_tag": self.ablation_tag,
            "total_traces": len(traces),
            "status_distribution": status_dist,
            "task_type_distribution": task_type_dist,
            "avg_confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "avg_tool_calls": (
                sum(tool_call_counts) / len(tool_call_counts)
                if tool_call_counts
                else 0.0
            ),
            "tool_usage_distribution": tool_counts,
            "avg_metrics": avg_metrics,
            "completed_count": status_dist.get("completed", 0),
            "failed_count": status_dist.get("failed", 0),
            "insufficient_evidence_count": status_dist.get("insufficient_evidence", 0),
        }

    def export_to_csv(
        self, output_path: Union[str, Path], run_id: Optional[str] = None
    ) -> Path:
        """Export trace data to CSV for external analysis.

        Args:
            output_path: Path to write CSV file.
            run_id: Run ID to export (defaults to current run).

        Returns:
            Path to the written CSV file.
        """
        import csv

        traces = self.get_run_traces(run_id)
        output_path = Path(output_path)

        fieldnames = [
            "trace_id",
            "sample_id",
            "scene_id",
            "task_type",
            "user_query",
            "status",
            "confidence",
            "num_tool_calls",
            "duration_ms",
            "benchmark_name",
            "ablation_tag",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trace in traces:
                try:
                    metadata = json.loads(trace.metadata_json or "{}")
                except json.JSONDecodeError:
                    metadata = {}

                writer.writerow(
                    {
                        "trace_id": trace.trace_id,
                        "sample_id": metadata.get("sample_id", ""),
                        "scene_id": trace.scene_id,
                        "task_type": trace.task_type,
                        "user_query": trace.user_query[:200],  # Truncate for CSV
                        "status": trace.status,
                        "confidence": trace.confidence,
                        "num_tool_calls": trace.num_tool_calls,
                        "duration_ms": trace.duration_ms,
                        "benchmark_name": metadata.get("benchmark_name", ""),
                        "ablation_tag": metadata.get("ablation_tag", ""),
                    }
                )

        logger.info(f"Exported {len(traces)} traces to {output_path}")
        return output_path


class TracingBatchEvaluatorMixin:
    """Mixin that adds trace recording to BatchEvaluator.

    Provides hooks to auto-save traces during evaluation runs
    and link them to benchmark samples.

    Usage:
        class TracingBatchEvaluator(TracingBatchEvaluatorMixin, BatchEvaluator):
            pass

        evaluator = TracingBatchEvaluator(config, trace_db=TraceDB("traces.db"))
    """

    _trace_manager: Optional[EvalTraceManager] = None

    def init_tracing(
        self,
        db: TraceDB,
        benchmark_name: str = "",
        ablation_tag: str = "",
    ) -> None:
        """Initialize trace recording.

        Call this before running evaluation to enable auto-tracing.

        Args:
            db: TraceDB instance for persistence.
            benchmark_name: Name of the benchmark.
            ablation_tag: Tag for ablation configuration.
        """
        config = getattr(self, "config", None)
        run_id = getattr(config, "run_id", "") if config else ""

        self._trace_manager = EvalTraceManager(
            db=db,
            benchmark_name=benchmark_name or getattr(config, "benchmark_name", ""),
            run_id=run_id,
            ablation_tag=ablation_tag or (config.get_ablation_tag() if config else ""),
        )

    def _record_sample_trace(
        self,
        sample_id: str,
        query: str,
        task_type: str,
        scene_id: str,
        ground_truth: Any,
    ) -> str:
        """Start trace recording for a sample (called by evaluate_single)."""
        if self._trace_manager is None:
            return ""

        metadata = EvalTraceMetadata(
            benchmark_name=self._trace_manager.benchmark_name,
            sample_id=sample_id,
            scene_id=scene_id,
            task_type=task_type,
            query=query,
            ground_truth=ground_truth,
        )
        return self._trace_manager.start_trace(metadata)

    def _finish_sample_trace(
        self,
        trace_id: str,
        result: Any,  # EvalSampleResult
    ) -> None:
        """Finish trace recording for a sample."""
        if self._trace_manager is None or not trace_id:
            return

        self._trace_manager.finish_trace(
            trace_id=trace_id,
            status=result.stage2_status
            or ("completed" if result.stage2_success else "failed"),
            confidence=result.stage2_confidence,
            summary=result.stage2_answer,
            tool_trace=result.tool_trace,
            keyframe_count=result.stage1_keyframe_count,
            prediction=result.prediction,
            metrics=result.metrics,
            error=result.stage1_error or result.stage2_error,
        )

    def get_trace_statistics(self) -> Optional[Dict[str, Any]]:
        """Get trace statistics for the current run."""
        if self._trace_manager is None:
            return None
        return self._trace_manager.export_statistics()


def create_tracing_evaluator(
    config: Any,  # BatchEvalConfig
    trace_db_path: Union[str, Path] = "eval_traces.db",
    stage1_factory: Optional[Callable] = None,
    stage2_factory: Optional[Callable] = None,
) -> Tuple[Any, EvalTraceManager]:
    """Create a BatchEvaluator with tracing enabled.

    This is a convenience function that sets up the evaluator
    with trace recording automatically configured.

    Args:
        config: BatchEvalConfig for the evaluation run.
        trace_db_path: Path to SQLite trace database.
        stage1_factory: Optional Stage 1 factory.
        stage2_factory: Optional Stage 2 factory.

    Returns:
        Tuple of (evaluator, trace_manager) for access to both.
    """
    from .batch_eval import BatchEvaluator

    # Create trace database
    db = TraceDB(trace_db_path)

    # Create trace manager
    manager = EvalTraceManager(
        db=db,
        benchmark_name=config.benchmark_name,
        run_id=config.run_id,
        ablation_tag=config.get_ablation_tag(),
    )

    # Create evaluator
    evaluator = BatchEvaluator(
        config=config,
        stage1_factory=stage1_factory,
        stage2_factory=stage2_factory,
    )

    # Attach trace manager to evaluator
    evaluator._trace_manager = manager

    return evaluator, manager


def export_run_trace_report(
    db: TraceDB,
    run_id: str,
    output_dir: Union[str, Path],
    benchmark_name: str = "",
) -> Dict[str, Path]:
    """Export comprehensive trace report for a run.

    Generates multiple output formats for analysis:
    - JSON summary with all statistics
    - CSV for spreadsheet analysis
    - Per-trace JSON for detailed debugging

    Args:
        db: TraceDB with trace data.
        run_id: Run ID to export.
        output_dir: Directory to write outputs.
        benchmark_name: Optional benchmark name filter.

    Returns:
        Dictionary mapping output type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = EvalTraceManager(
        db=db,
        benchmark_name=benchmark_name,
        run_id=run_id,
    )

    outputs: Dict[str, Path] = {}

    # Export statistics JSON
    stats = manager.export_statistics()
    stats_path = output_dir / f"trace_stats_{run_id}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    outputs["statistics"] = stats_path

    # Export CSV
    csv_path = output_dir / f"traces_{run_id}.csv"
    manager.export_to_csv(csv_path)
    outputs["csv"] = csv_path

    # Export full traces JSON
    traces = manager.get_run_traces()
    traces_data = []
    for trace in traces:
        trace_dict = {
            "trace_id": trace.trace_id,
            "created_at": trace.created_at,
            "finished_at": trace.finished_at,
            "task_type": trace.task_type,
            "user_query": trace.user_query,
            "scene_id": trace.scene_id,
            "status": trace.status,
            "confidence": trace.confidence,
            "summary": trace.summary,
            "num_tool_calls": trace.num_tool_calls,
            "duration_ms": trace.duration_ms,
            "tool_trace": trace.tool_trace,
        }
        try:
            trace_dict["metadata"] = json.loads(trace.metadata_json or "{}")
        except json.JSONDecodeError:
            trace_dict["metadata"] = {}
        traces_data.append(trace_dict)

    traces_path = output_dir / f"traces_full_{run_id}.json"
    with open(traces_path, "w") as f:
        json.dump(traces_data, f, indent=2)
    outputs["traces"] = traces_path

    logger.info(f"Exported trace report for run {run_id}: {len(traces)} traces")
    return outputs
