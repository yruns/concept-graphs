"""Batch evaluation script for two-stage 3D scene understanding.

This module runs Stage 1 (keyframe retrieval) + Stage 2 (VLM agent reasoning) on benchmark
datasets with support for parallel evaluation, progress tracking, and resumption.

Key Features:
- Parallel evaluation of benchmark samples
- Checkpoint-based progress tracking and resumption
- Structured results JSON output
- Support for multiple benchmark types (OpenEQA, SQA3D, ScanRefer, etc.)

Academic Alignment:
- Tracks evidence acquisition patterns (supports "adaptive evidence" claim)
- Records tool usage traces (supports "symbolic-to-visual repair" analysis)
- Captures confidence/uncertainty outputs (supports "evidence-grounded uncertainty" claim)
- Unified interface across task types (supports "unified multi-task policy" claim)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from loguru import logger
from pydantic import BaseModel, Field


class EvalSample(Protocol):
    """Protocol for evaluation samples from any benchmark.

    Benchmark adapters should implement this protocol to enable
    unified evaluation across different datasets.
    """

    @property
    def sample_id(self) -> str:
        """Unique identifier for the sample."""
        ...

    @property
    def query(self) -> str:
        """The natural language query/question."""
        ...

    @property
    def task_type(self) -> str:
        """Task type identifier (qa, visual_grounding, etc.)."""
        ...

    @property
    def ground_truth(self) -> Any:
        """Ground truth answer or annotations."""
        ...

    @property
    def scene_id(self) -> str:
        """Scene identifier for keyframe retrieval."""
        ...


@dataclass
class EvalSampleResult:
    """Result of evaluating a single sample through both stages.

    This structure captures all information needed for academic analysis:
    - Stage 1 retrieval metrics (for keyframe quality analysis)
    - Stage 2 agent metrics (for reasoning quality analysis)
    - Tool traces (for evidence acquisition pattern analysis)
    - Uncertainty outputs (for calibration analysis)
    """

    sample_id: str
    query: str
    task_type: str
    scene_id: str

    # Stage 1 results
    stage1_success: bool = False
    stage1_keyframe_count: int = 0
    stage1_hypothesis_kind: str = ""
    stage1_latency_ms: float = 0.0
    stage1_error: Optional[str] = None

    # Stage 2 results
    stage2_success: bool = False
    stage2_status: str = ""  # completed, insufficient_evidence, failed
    stage2_confidence: float = 0.0
    stage2_answer: str = ""
    stage2_tool_calls: int = 0
    stage2_latency_ms: float = 0.0
    stage2_error: Optional[str] = None

    # Ground truth comparison
    ground_truth: Any = None
    prediction: Any = None

    # Evaluation metrics (benchmark-specific)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Academic analysis data
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    cited_frames: List[int] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    raw_stage1_output: Optional[Dict[str, Any]] = None
    raw_stage2_output: Optional[Dict[str, Any]] = None


@dataclass
class EvalRunResult:
    """Aggregate results from a batch evaluation run.

    Contains both summary statistics and per-sample results
    for comprehensive academic analysis.
    """

    run_id: str
    benchmark_name: str
    config: Dict[str, Any]

    # Summary statistics
    total_samples: int = 0
    successful_samples: int = 0
    failed_stage1: int = 0
    failed_stage2: int = 0

    # Aggregate metrics
    avg_stage1_latency_ms: float = 0.0
    avg_stage2_latency_ms: float = 0.0
    avg_stage2_confidence: float = 0.0
    avg_tool_calls_per_sample: float = 0.0

    # Evidence acquisition statistics (for academic claims)
    samples_with_tool_use: int = 0
    samples_with_insufficient_evidence: int = 0
    tool_usage_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-sample results
    results: List[EvalSampleResult] = field(default_factory=list)

    # Timing
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "config": self.config,
            "summary": {
                "total_samples": self.total_samples,
                "successful_samples": self.successful_samples,
                "failed_stage1": self.failed_stage1,
                "failed_stage2": self.failed_stage2,
                "success_rate": (
                    self.successful_samples / self.total_samples
                    if self.total_samples > 0
                    else 0.0
                ),
            },
            "latency": {
                "avg_stage1_latency_ms": round(self.avg_stage1_latency_ms, 2),
                "avg_stage2_latency_ms": round(self.avg_stage2_latency_ms, 2),
                "total_duration_seconds": round(self.total_duration_seconds, 2),
            },
            "stage2_analysis": {
                "avg_confidence": round(self.avg_stage2_confidence, 3),
                "avg_tool_calls": round(self.avg_tool_calls_per_sample, 2),
                "samples_with_tool_use": self.samples_with_tool_use,
                "samples_with_insufficient_evidence": self.samples_with_insufficient_evidence,
                "tool_usage_distribution": self.tool_usage_distribution,
            },
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "results": [
                {
                    "sample_id": r.sample_id,
                    "query": r.query,
                    "task_type": r.task_type,
                    "stage1_success": r.stage1_success,
                    "stage2_success": r.stage2_success,
                    "stage2_status": r.stage2_status,
                    "stage2_confidence": r.stage2_confidence,
                    "stage2_tool_calls": r.stage2_tool_calls,
                    "metrics": r.metrics,
                    "uncertainties": r.uncertainties,
                    "stage1_error": r.stage1_error,
                    "stage2_error": r.stage2_error,
                }
                for r in self.results
            ],
        }


class BatchEvalConfig(BaseModel):
    """Configuration for batch evaluation runs.

    Supports ablation studies by enabling/disabling various
    Stage 2 capabilities.
    """

    # Run identification
    run_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    benchmark_name: str = "openeqa"

    # Parallelism
    max_workers: int = Field(default=4, ge=1, le=32)
    batch_size: int = Field(default=10, ge=1)

    # Checkpoint/Resume
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = Field(default=10, ge=1)
    resume_from_checkpoint: bool = True

    # Stage 1 configuration
    stage1_model: str = "gpt-5.2-2025-12-11"
    stage1_k: int = Field(default=3, ge=1, le=10)
    stage1_timeout_seconds: int = Field(default=60, ge=10)

    # Stage 2 configuration
    stage2_enabled: bool = True
    stage2_model: str = "gpt-5.2-2025-12-11"
    stage2_max_turns: int = Field(default=6, ge=1, le=12)
    stage2_plan_mode: str = "brief"  # off, brief, full
    stage2_timeout_seconds: int = Field(default=120, ge=30)

    # Ablation controls (for academic experiments)
    enable_tool_request_more_views: bool = True
    enable_tool_request_crops: bool = True
    enable_tool_hypothesis_repair: bool = True
    enable_uncertainty_stopping: bool = True
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # Output configuration
    output_dir: str = "./eval_results"
    save_raw_outputs: bool = True
    save_tool_traces: bool = True

    # Limits (for debugging)
    max_samples: Optional[int] = None
    skip_samples: int = 0

    def get_ablation_tag(self) -> str:
        """Generate a tag describing the ablation configuration."""
        tags = []
        if not self.stage2_enabled:
            return "stage1_only"
        if not self.enable_tool_request_more_views:
            tags.append("no_views")
        if not self.enable_tool_request_crops:
            tags.append("no_crops")
        if not self.enable_tool_hypothesis_repair:
            tags.append("no_repair")
        if not self.enable_uncertainty_stopping:
            tags.append("no_uncertainty")
        if self.stage2_max_turns == 1:
            tags.append("oneshot")
        return "_".join(tags) if tags else "full"


class CheckpointManager:
    """Manages evaluation checkpoints for progress tracking and resumption.

    Checkpoints are saved after each batch to enable:
    - Progress tracking during long runs
    - Resumption after interrupts or failures
    - Incremental result accumulation
    """

    def __init__(self, checkpoint_dir: Path, run_id: str):
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.checkpoint_file = checkpoint_dir / f"checkpoint_{run_id}.json"
        self._lock = threading.Lock()

    def save(self, completed_ids: set[str], results: List[EvalSampleResult]) -> None:
        """Save checkpoint with completed sample IDs and results."""
        with self._lock:
            checkpoint_data = {
                "run_id": self.run_id,
                "completed_ids": list(completed_ids),
                "timestamp": datetime.now().isoformat(),
                "num_results": len(results),
            }

            # Save main checkpoint
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Save results separately (for append-friendly format)
            results_file = self.checkpoint_dir / f"results_{self.run_id}.jsonl"
            with open(results_file, "w") as f:
                for res in results:
                    f.write(json.dumps(self._result_to_dict(res)) + "\n")

            logger.debug(f"Checkpoint saved: {len(completed_ids)} samples completed")

    def load(self) -> Tuple[set[str], List[EvalSampleResult]]:
        """Load checkpoint if exists, return (completed_ids, results)."""
        if not self.checkpoint_file.exists():
            return set(), []

        try:
            with open(self.checkpoint_file) as f:
                checkpoint_data = json.load(f)

            completed_ids = set(checkpoint_data.get("completed_ids", []))

            # Load results
            results_file = self.checkpoint_dir / f"results_{self.run_id}.jsonl"
            results = []
            if results_file.exists():
                with open(results_file) as f:
                    for line in f:
                        if line.strip():
                            results.append(self._dict_to_result(json.loads(line)))

            logger.info(
                f"Loaded checkpoint: {len(completed_ids)} completed, {len(results)} results"
            )
            return completed_ids, results
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return set(), []

    def _result_to_dict(self, res: EvalSampleResult) -> Dict[str, Any]:
        """Convert EvalSampleResult to serializable dict."""
        return {
            "sample_id": res.sample_id,
            "query": res.query,
            "task_type": res.task_type,
            "scene_id": res.scene_id,
            "stage1_success": res.stage1_success,
            "stage1_keyframe_count": res.stage1_keyframe_count,
            "stage1_hypothesis_kind": res.stage1_hypothesis_kind,
            "stage1_latency_ms": res.stage1_latency_ms,
            "stage1_error": res.stage1_error,
            "stage2_success": res.stage2_success,
            "stage2_status": res.stage2_status,
            "stage2_confidence": res.stage2_confidence,
            "stage2_answer": res.stage2_answer,
            "stage2_tool_calls": res.stage2_tool_calls,
            "stage2_latency_ms": res.stage2_latency_ms,
            "stage2_error": res.stage2_error,
            "ground_truth": res.ground_truth,
            "prediction": res.prediction,
            "metrics": res.metrics,
            "tool_trace": res.tool_trace,
            "uncertainties": res.uncertainties,
            "cited_frames": res.cited_frames,
            "evidence_items": res.evidence_items,
            "timestamp": res.timestamp,
        }

    def _dict_to_result(self, data: Dict[str, Any]) -> EvalSampleResult:
        """Convert dict back to EvalSampleResult."""
        return EvalSampleResult(
            sample_id=data.get("sample_id", ""),
            query=data.get("query", ""),
            task_type=data.get("task_type", ""),
            scene_id=data.get("scene_id", ""),
            stage1_success=data.get("stage1_success", False),
            stage1_keyframe_count=data.get("stage1_keyframe_count", 0),
            stage1_hypothesis_kind=data.get("stage1_hypothesis_kind", ""),
            stage1_latency_ms=data.get("stage1_latency_ms", 0.0),
            stage1_error=data.get("stage1_error"),
            stage2_success=data.get("stage2_success", False),
            stage2_status=data.get("stage2_status", ""),
            stage2_confidence=data.get("stage2_confidence", 0.0),
            stage2_answer=data.get("stage2_answer", ""),
            stage2_tool_calls=data.get("stage2_tool_calls", 0),
            stage2_latency_ms=data.get("stage2_latency_ms", 0.0),
            stage2_error=data.get("stage2_error"),
            ground_truth=data.get("ground_truth"),
            prediction=data.get("prediction"),
            metrics=data.get("metrics", {}),
            tool_trace=data.get("tool_trace", []),
            uncertainties=data.get("uncertainties", []),
            cited_frames=data.get("cited_frames", []),
            evidence_items=data.get("evidence_items", []),
            timestamp=data.get("timestamp", ""),
        )


class BatchEvaluator:
    """Batch evaluator for two-stage 3D scene understanding.

    Runs Stage 1 (keyframe retrieval) + Stage 2 (VLM agent) evaluation
    on benchmark datasets with parallel execution and checkpoint support.

    Usage:
        config = BatchEvalConfig(benchmark_name="openeqa", max_workers=4)
        evaluator = BatchEvaluator(config)
        results = evaluator.run(samples, scene_provider)
    """

    def __init__(
        self,
        config: BatchEvalConfig,
        stage1_factory: Optional[Callable[[str], Any]] = None,
        stage2_factory: Optional[Callable[[], Any]] = None,
    ):
        """Initialize the batch evaluator.

        Args:
            config: Evaluation configuration.
            stage1_factory: Factory function that creates a KeyframeSelector
                           given a scene_id. If None, uses default factory.
            stage2_factory: Factory function that creates a Stage2DeepResearchAgent.
                           If None, uses default factory.
        """
        self.config = config
        self._stage1_factory = stage1_factory
        self._stage2_factory = stage2_factory

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint manager
        checkpoint_dir = (
            Path(config.checkpoint_dir)
            if config.checkpoint_dir
            else self.output_dir / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_manager = CheckpointManager(checkpoint_dir, config.run_id)

        # Cache for KeyframeSelectors (one per scene)
        self._selector_cache: Dict[str, Any] = {}
        self._selector_lock = threading.Lock()

        # Stage 2 agent (shared instance)
        self._stage2_agent: Optional[Any] = None

        # Progress tracking
        self._completed_count = 0
        self._total_count = 0
        self._progress_lock = threading.Lock()

    def _get_or_create_selector(self, scene_id: str, scene_path: Path) -> Any:
        """Get or create a KeyframeSelector for a scene (thread-safe, cached)."""
        with self._selector_lock:
            if scene_id not in self._selector_cache:
                if self._stage1_factory:
                    self._selector_cache[scene_id] = self._stage1_factory(scene_id)
                else:
                    # Default factory using KeyframeSelector
                    from conceptgraph.query_scene.keyframe_selector import (
                        KeyframeSelector,
                    )

                    self._selector_cache[scene_id] = KeyframeSelector.from_scene_path(
                        scene_path,
                        llm_model=self.config.stage1_model,
                        use_pool=True,
                    )
            return self._selector_cache[scene_id]

    def _get_or_create_stage2_agent(self) -> Any:
        """Get or create the Stage 2 agent (lazy initialization)."""
        if self._stage2_agent is None:
            if self._stage2_factory:
                self._stage2_agent = self._stage2_factory()
            else:
                # Default factory using Stage2DeepResearchAgent
                from conceptgraph.agents.models import Stage2DeepAgentConfig
                from conceptgraph.agents.stage2_deep_agent import (
                    Stage2DeepResearchAgent,
                )

                agent_config = Stage2DeepAgentConfig(
                    model_name=self.config.stage2_model,
                    confidence_threshold=self.config.confidence_threshold,
                    enable_uncertainty_stopping=self.config.enable_uncertainty_stopping,
                )
                self._stage2_agent = Stage2DeepResearchAgent(config=agent_config)
        return self._stage2_agent

    def _run_stage1(
        self, sample: EvalSample, scene_path: Path
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Run Stage 1 keyframe retrieval for a sample.

        Returns:
            (success, result_dict, error_message)
        """
        start_time = time.time()
        try:
            selector = self._get_or_create_selector(sample.scene_id, scene_path)
            result = selector.select_keyframes_v2(sample.query, k=self.config.stage1_k)

            latency_ms = (time.time() - start_time) * 1000

            # Extract metadata
            metadata = getattr(result, "metadata", {}) or {}
            hypothesis_kind = metadata.get("selected_hypothesis_kind", "unknown")

            return (
                True,
                {
                    "keyframe_paths": [str(p) for p in result.keyframe_paths],
                    "keyframe_count": len(result.keyframe_paths),
                    "hypothesis_kind": hypothesis_kind,
                    "latency_ms": latency_ms,
                    "raw_result": result,
                    "metadata": metadata,
                },
                "",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Stage 1 failed for {sample.sample_id}: {e}")
            return (
                False,
                {"latency_ms": latency_ms},
                str(e),
            )

    def _run_stage2(
        self,
        sample: EvalSample,
        stage1_result: Dict[str, Any],
        scene_path: Path,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Run Stage 2 VLM agent for a sample.

        Returns:
            (success, result_dict, error_message)
        """
        if not self.config.stage2_enabled:
            return (True, {"skipped": True}, "")

        start_time = time.time()
        try:
            from conceptgraph.agents.adapters import build_stage2_evidence_bundle
            from conceptgraph.agents.models import (
                Stage2PlanMode,
                Stage2TaskSpec,
                Stage2TaskType,
            )

            # Map task type string to enum
            task_type_map = {
                "qa": Stage2TaskType.QA,
                "visual_grounding": Stage2TaskType.VISUAL_GROUNDING,
                "nav_plan": Stage2TaskType.NAV_PLAN,
                "manipulation": Stage2TaskType.MANIPULATION,
            }
            task_type = task_type_map.get(sample.task_type, Stage2TaskType.QA)

            plan_mode_map = {
                "off": Stage2PlanMode.OFF,
                "brief": Stage2PlanMode.BRIEF,
                "full": Stage2PlanMode.FULL,
            }
            plan_mode = plan_mode_map.get(
                self.config.stage2_plan_mode, Stage2PlanMode.BRIEF
            )

            # Build evidence bundle from Stage 1 result
            raw_result = stage1_result.get("raw_result")
            if raw_result is None:
                return (False, {}, "No Stage 1 result to pass to Stage 2")

            bundle = build_stage2_evidence_bundle(
                raw_result,
                scene_id=sample.scene_id,
            )

            # Create task spec
            task = Stage2TaskSpec(
                task_type=task_type,
                user_query=sample.query,
                plan_mode=plan_mode,
                max_reasoning_turns=self.config.stage2_max_turns,
            )

            # Run agent
            agent = self._get_or_create_stage2_agent()
            agent_result = agent.run(task, bundle)

            latency_ms = (time.time() - start_time) * 1000

            # Extract results
            result = agent_result.result
            return (
                True,
                {
                    "status": result.status.value,
                    "confidence": result.confidence,
                    "summary": result.summary,
                    "answer": result.payload.get("answer", result.summary),
                    "tool_calls": len(agent_result.tool_trace),
                    "tool_trace": [
                        {
                            "tool_name": obs.tool_name,
                            "tool_input": obs.tool_input,
                        }
                        for obs in agent_result.tool_trace
                    ],
                    "uncertainties": list(result.uncertainties),
                    "cited_frames": list(result.cited_frame_indices),
                    "evidence_items": [
                        item.model_dump() for item in result.evidence_items
                    ],
                    "latency_ms": latency_ms,
                    "raw_result": agent_result,
                },
                "",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Stage 2 failed for {sample.sample_id}: {e}")
            return (
                False,
                {"latency_ms": latency_ms},
                str(e),
            )

    def _evaluate_single(
        self,
        sample: EvalSample,
        scene_path: Path,
        eval_fn: Optional[Callable[[EvalSample, Any], Dict[str, float]]] = None,
    ) -> EvalSampleResult:
        """Evaluate a single sample through both stages."""
        result = EvalSampleResult(
            sample_id=sample.sample_id,
            query=sample.query,
            task_type=sample.task_type,
            scene_id=sample.scene_id,
            ground_truth=sample.ground_truth,
            timestamp=datetime.now().isoformat(),
        )

        # Stage 1
        s1_success, s1_data, s1_error = self._run_stage1(sample, scene_path)
        result.stage1_success = s1_success
        result.stage1_latency_ms = s1_data.get("latency_ms", 0.0)
        result.stage1_error = s1_error if s1_error else None

        if s1_success:
            result.stage1_keyframe_count = s1_data.get("keyframe_count", 0)
            result.stage1_hypothesis_kind = s1_data.get("hypothesis_kind", "")
            if self.config.save_raw_outputs:
                result.raw_stage1_output = {
                    k: v for k, v in s1_data.items() if k != "raw_result"
                }
        else:
            return result  # Skip Stage 2 if Stage 1 failed

        # Stage 2
        s2_success, s2_data, s2_error = self._run_stage2(sample, s1_data, scene_path)
        result.stage2_success = s2_success
        result.stage2_latency_ms = s2_data.get("latency_ms", 0.0)
        result.stage2_error = s2_error if s2_error else None

        if s2_success and not s2_data.get("skipped"):
            result.stage2_status = s2_data.get("status", "")
            result.stage2_confidence = s2_data.get("confidence", 0.0)
            result.stage2_answer = s2_data.get("answer", "")
            result.stage2_tool_calls = s2_data.get("tool_calls", 0)
            result.prediction = result.stage2_answer

            if self.config.save_tool_traces:
                result.tool_trace = s2_data.get("tool_trace", [])
            result.uncertainties = s2_data.get("uncertainties", [])
            result.cited_frames = s2_data.get("cited_frames", [])
            result.evidence_items = s2_data.get("evidence_items", [])

            if self.config.save_raw_outputs:
                result.raw_stage2_output = {
                    k: v for k, v in s2_data.items() if k != "raw_result"
                }

        # Run benchmark-specific evaluation if provided
        if eval_fn and result.stage2_success:
            try:
                result.metrics = eval_fn(sample, result.prediction)
            except Exception as e:
                logger.warning(
                    f"Evaluation function failed for {sample.sample_id}: {e}"
                )

        return result

    def run(
        self,
        samples: List[EvalSample],
        scene_path_provider: Callable[[str], Path],
        eval_fn: Optional[Callable[[EvalSample, Any], Dict[str, float]]] = None,
    ) -> EvalRunResult:
        """Run batch evaluation on samples.

        Args:
            samples: List of evaluation samples.
            scene_path_provider: Function that returns scene path given scene_id.
            eval_fn: Optional benchmark-specific evaluation function.

        Returns:
            Aggregate evaluation results.
        """
        # Apply limits
        if self.config.skip_samples > 0:
            samples = samples[self.config.skip_samples :]
        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        self._total_count = len(samples)
        start_time = datetime.now()

        logger.info(
            f"Starting batch evaluation: {len(samples)} samples, "
            f"max_workers={self.config.max_workers}, "
            f"stage2_enabled={self.config.stage2_enabled}"
        )

        # Load checkpoint if resuming
        completed_ids: set[str] = set()
        results: List[EvalSampleResult] = []

        if self.config.resume_from_checkpoint:
            completed_ids, results = self._checkpoint_manager.load()
            self._completed_count = len(completed_ids)

        # Filter out already completed samples
        remaining_samples = [s for s in samples if s.sample_id not in completed_ids]
        logger.info(
            f"Resuming: {len(completed_ids)} completed, {len(remaining_samples)} remaining"
        )

        # Process samples in parallel
        if remaining_samples:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(
                        self._evaluate_single,
                        sample,
                        scene_path_provider(sample.scene_id),
                        eval_fn,
                    ): sample
                    for sample in remaining_samples
                }

                # Process completed tasks
                checkpoint_batch: List[EvalSampleResult] = []
                for future in concurrent.futures.as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_ids.add(sample.sample_id)
                        checkpoint_batch.append(result)

                        with self._progress_lock:
                            self._completed_count += 1
                            progress = self._completed_count / self._total_count
                            logger.info(
                                f"[{self._completed_count}/{self._total_count}] "
                                f"{progress:.1%} - {sample.sample_id}: "
                                f"s1={'✓' if result.stage1_success else '✗'}, "
                                f"s2={'✓' if result.stage2_success else '✗'}"
                            )

                        # Checkpoint periodically
                        if len(checkpoint_batch) >= self.config.checkpoint_interval:
                            self._checkpoint_manager.save(completed_ids, results)
                            checkpoint_batch.clear()

                    except Exception as e:
                        logger.error(f"Failed to process {sample.sample_id}: {e}")
                        results.append(
                            EvalSampleResult(
                                sample_id=sample.sample_id,
                                query=sample.query,
                                task_type=sample.task_type,
                                scene_id=sample.scene_id,
                                stage1_error=str(e),
                                timestamp=datetime.now().isoformat(),
                            )
                        )

                # Final checkpoint
                if checkpoint_batch:
                    self._checkpoint_manager.save(completed_ids, results)

        end_time = datetime.now()

        # Compute aggregate statistics
        run_result = self._compute_aggregate_results(
            results,
            start_time,
            end_time,
        )

        # Save final results
        self._save_results(run_result)

        return run_result

    def _compute_aggregate_results(
        self,
        results: List[EvalSampleResult],
        start_time: datetime,
        end_time: datetime,
    ) -> EvalRunResult:
        """Compute aggregate statistics from results."""
        run_result = EvalRunResult(
            run_id=self.config.run_id,
            benchmark_name=self.config.benchmark_name,
            config=self.config.model_dump(),
            total_samples=len(results),
            results=results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=(end_time - start_time).total_seconds(),
        )

        if not results:
            return run_result

        # Count successes and failures
        successful = [r for r in results if r.stage1_success and r.stage2_success]
        run_result.successful_samples = len(successful)
        run_result.failed_stage1 = sum(1 for r in results if not r.stage1_success)
        run_result.failed_stage2 = sum(
            1 for r in results if r.stage1_success and not r.stage2_success
        )

        # Latency statistics
        s1_latencies = [r.stage1_latency_ms for r in results if r.stage1_success]
        s2_latencies = [r.stage2_latency_ms for r in results if r.stage2_success]

        if s1_latencies:
            run_result.avg_stage1_latency_ms = sum(s1_latencies) / len(s1_latencies)
        if s2_latencies:
            run_result.avg_stage2_latency_ms = sum(s2_latencies) / len(s2_latencies)

        # Stage 2 analysis
        s2_results = [r for r in results if r.stage2_success]
        if s2_results:
            confidences = [r.stage2_confidence for r in s2_results]
            tool_calls = [r.stage2_tool_calls for r in s2_results]

            run_result.avg_stage2_confidence = sum(confidences) / len(confidences)
            run_result.avg_tool_calls_per_sample = sum(tool_calls) / len(tool_calls)
            run_result.samples_with_tool_use = sum(
                1 for r in s2_results if r.stage2_tool_calls > 0
            )
            run_result.samples_with_insufficient_evidence = sum(
                1 for r in s2_results if r.stage2_status == "insufficient_evidence"
            )

            # Tool usage distribution
            tool_counts: Dict[str, int] = {}
            for r in s2_results:
                for trace in r.tool_trace:
                    tool_name = trace.get("tool_name", "unknown")
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            run_result.tool_usage_distribution = tool_counts

        return run_result

    def _save_results(self, run_result: EvalRunResult) -> None:
        """Save results to output directory."""
        # Main results file
        results_file = self.output_dir / f"eval_{self.config.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(run_result.to_dict(), f, indent=2)
        logger.info(f"Results saved: {results_file}")

        # Summary report
        summary_file = self.output_dir / f"summary_{self.config.run_id}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Evaluation Run: {run_result.run_id}\n")
            f.write(f"Benchmark: {run_result.benchmark_name}\n")
            f.write(f"Ablation: {self.config.get_ablation_tag()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Samples: {run_result.total_samples}\n")
            f.write(f"Successful: {run_result.successful_samples}\n")
            f.write(f"Failed Stage 1: {run_result.failed_stage1}\n")
            f.write(f"Failed Stage 2: {run_result.failed_stage2}\n")
            f.write(
                f"Success Rate: {run_result.successful_samples / run_result.total_samples:.1%}\n"
                if run_result.total_samples > 0
                else "Success Rate: N/A\n"
            )
            f.write("\n")
            f.write(f"Avg Stage 1 Latency: {run_result.avg_stage1_latency_ms:.1f}ms\n")
            f.write(f"Avg Stage 2 Latency: {run_result.avg_stage2_latency_ms:.1f}ms\n")
            f.write(f"Avg Stage 2 Confidence: {run_result.avg_stage2_confidence:.3f}\n")
            f.write(f"Avg Tool Calls: {run_result.avg_tool_calls_per_sample:.2f}\n")
            f.write(f"Samples with Tool Use: {run_result.samples_with_tool_use}\n")
            f.write(
                f"Samples with Insufficient Evidence: {run_result.samples_with_insufficient_evidence}\n"
            )
            f.write("\n")
            f.write(f"Total Duration: {run_result.total_duration_seconds:.1f}s\n")
        logger.info(f"Summary saved: {summary_file}")


# =============================================================================
# Benchmark Adapters
# =============================================================================


@dataclass
class OpenEQASampleAdapter:
    """Adapter to make OpenEQA samples conform to EvalSample protocol."""

    _sample: Any  # OpenEQASample from benchmarks module

    @property
    def sample_id(self) -> str:
        return self._sample.question_id

    @property
    def query(self) -> str:
        return self._sample.question

    @property
    def task_type(self) -> str:
        return "qa"

    @property
    def ground_truth(self) -> str:
        return self._sample.answer

    @property
    def scene_id(self) -> str:
        return self._sample.scene_id


@dataclass
class SQA3DSampleAdapter:
    """Adapter to make SQA3D samples conform to EvalSample protocol."""

    _sample: Any  # SQA3DSample from benchmarks module

    @property
    def sample_id(self) -> str:
        return self._sample.question_id

    @property
    def query(self) -> str:
        # Include situation context in query for situated QA
        situation = self._sample.situation
        context = (
            f"[Position: {situation.position}, Orientation: {situation.orientation}] "
        )
        if situation.room_description:
            context += f"[Room: {situation.room_description}] "
        return context + self._sample.question

    @property
    def task_type(self) -> str:
        return "qa"

    @property
    def ground_truth(self) -> List[str]:
        return self._sample.answers

    @property
    def scene_id(self) -> str:
        return self._sample.scene_id


def adapt_openeqa_samples(samples: List[Any]) -> List[OpenEQASampleAdapter]:
    """Adapt OpenEQA samples to EvalSample protocol."""
    return [OpenEQASampleAdapter(_sample=s) for s in samples]


def adapt_sqa3d_samples(samples: List[Any]) -> List[SQA3DSampleAdapter]:
    """Adapt SQA3D samples to EvalSample protocol."""
    return [SQA3DSampleAdapter(_sample=s) for s in samples]
