"""Unified ablation study runner: request_crops tool only.

TASK-042: Ablation: + request_crops only

This module implements the third ablation condition in Phase 5, which runs
the Stage 2 agent with ONLY the request_crops tool enabled (disabling
request_more_views and switch_or_expand_hypothesis) to isolate the contribution
of object-level cropping to task performance.

Academic Relevance:
- Isolates contribution of "zooming into objects" to evidence-seeking
- Tests whether object-level detail acquisition alone is sufficient
- Compares against one-shot (no tools), views-only, and full agent
- Validates the "symbolic-to-visual repair" claim component

Key Ablation Settings:
- max_turns = 6 (allow multi-turn reasoning)
- plan_mode = "brief" (standard planning)
- request_more_views = False (disabled)
- request_crops = True (KEY: ENABLED)
- switch_or_expand_hypothesis = False (disabled)
- uncertainty_stopping = True (enabled for fair comparison)

Research Question:
"Does enabling object-level cropping alone improve over one-shot,
and is finer-grained visual evidence more valuable than having more keyframes?"

Comparison Matrix:
- oneshot (TASK-040): No tools, single-turn → baseline
- views_only (TASK-041): request_more_views only → keyframe-level evidence
- crops_only (TASK-042): request_crops only → object-level evidence [THIS]
- hypothesis_repair_only (TASK-043): hypothesis tools only → symbolic repair
- full: All tools → combined effect

Usage:
    # Run crops-only ablation on all benchmarks:
    python -m conceptgraph.evaluation.ablations.run_crops_only_ablation \
        --benchmarks all --mock

    # Run on specific benchmark:
    python -m conceptgraph.evaluation.ablations.run_crops_only_ablation \
        --benchmarks openeqa --data_root /path/to/openeqa

    # Dry run (show config only):
    python -m conceptgraph.evaluation.ablations.run_crops_only_ablation \
        --benchmarks all --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from loguru import logger

from conceptgraph.evaluation.ablation_config import (
    AblationConfig,
    AgentConfig,
    EvaluationConfig,
    Stage1Config,
    Stage2Config,
    ToolConfig,
    get_preset_config,
)
from conceptgraph.evaluation.batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalRunResult,
    adapt_openeqa_samples,
)

# =============================================================================
# Ablation Configuration
# =============================================================================

CROPS_ONLY_ABLATION_CONFIG = AblationConfig(
    name="crops_only",
    description="Stage 2 agent with only request_crops tool - tests object-level evidence acquisition",
    tools=ToolConfig(
        request_more_views=False,  # Disabled for ablation
        request_crops=True,  # KEY: ENABLED - isolate object cropping
        switch_or_expand_hypothesis=False,  # Disabled for ablation
        inspect_stage1_metadata=True,  # Keep for context (read-only)
        retrieve_object_context=True,  # Keep for context (read-only)
    ),
    agent=AgentConfig(
        max_turns=6,  # Allow multi-turn reasoning
        plan_mode="brief",  # Standard planning mode
        confidence_threshold=0.4,  # Same as full agent
        enable_uncertainty_stopping=True,  # Keep for fair comparison
        enable_subagents=False,  # No subagents for simpler ablation
        temperature=0.1,  # Same as full agent
        max_images=6,  # Same as full agent
        image_max_size=900,  # Same as full agent
    ),
    stage1=Stage1Config(
        model="gpt-5.2-2025-12-11",
        k=3,
        timeout_seconds=60,
    ),
    stage2=Stage2Config(
        enabled=True,
        model="gpt-5.2-2025-12-11",
        timeout_seconds=120,
        base_url="https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
    ),
    evaluation=EvaluationConfig(
        max_workers=4,
        batch_size=10,
        checkpoint_interval=10,
    ),
    tags=["ablation", "crops_only", "tool_ablation", "object_level_evidence"],
)


# =============================================================================
# Benchmark Support
# =============================================================================

SUPPORTED_BENCHMARKS = ["openeqa", "sqa3d", "scanrefer"]
BenchmarkType = Literal["openeqa", "sqa3d", "scanrefer", "all"]


@dataclass
class BenchmarkResult:
    """Result from running ablation on a single benchmark."""

    benchmark: str
    run_result: Optional[EvalRunResult]
    error: Optional[str] = None
    duration_seconds: float = 0.0
    config: Optional[Dict[str, Any]] = None


@dataclass
class AblationStudyResult:
    """Aggregated result from running ablation across all benchmarks."""

    ablation_name: str
    ablation_description: str
    timestamp: str
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    total_samples: int = 0
    total_successful: int = 0
    total_failed_stage1: int = 0
    total_failed_stage2: int = 0
    total_duration_seconds: float = 0.0

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all benchmarks."""
        if self.total_samples == 0:
            return 0.0
        return self.total_successful / self.total_samples

    @property
    def per_benchmark_success_rates(self) -> Dict[str, float]:
        """Success rate per benchmark."""
        rates = {}
        for result in self.benchmark_results:
            if result.run_result and result.run_result.total_samples > 0:
                rates[result.benchmark] = (
                    result.run_result.successful_samples
                    / result.run_result.total_samples
                )
            else:
                rates[result.benchmark] = 0.0
        return rates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ablation_name": self.ablation_name,
            "ablation_description": self.ablation_description,
            "timestamp": self.timestamp,
            "summary": {
                "total_samples": self.total_samples,
                "total_successful": self.total_successful,
                "total_failed_stage1": self.total_failed_stage1,
                "total_failed_stage2": self.total_failed_stage2,
                "overall_success_rate": self.overall_success_rate,
                "total_duration_seconds": self.total_duration_seconds,
            },
            "per_benchmark_success_rates": self.per_benchmark_success_rates,
            "benchmark_results": [
                {
                    "benchmark": r.benchmark,
                    "error": r.error,
                    "duration_seconds": r.duration_seconds,
                    "summary": (
                        {
                            "total_samples": r.run_result.total_samples,
                            "successful_samples": r.run_result.successful_samples,
                            "failed_stage1": r.run_result.failed_stage1,
                            "failed_stage2": r.run_result.failed_stage2,
                            "avg_stage1_latency_ms": r.run_result.avg_stage1_latency_ms,
                            "avg_stage2_latency_ms": r.run_result.avg_stage2_latency_ms,
                            "avg_stage2_confidence": r.run_result.avg_stage2_confidence,
                            "avg_tool_calls": r.run_result.avg_tool_calls_per_sample,
                            "samples_with_insufficient_evidence": r.run_result.samples_with_insufficient_evidence,
                        }
                        if r.run_result
                        else None
                    ),
                    "config": r.config,
                }
                for r in self.benchmark_results
            ],
            "academic_notes": {
                "ablation_condition": "request_crops=True only, other evidence tools disabled",
                "purpose": "Isolate object-level evidence acquisition contribution",
                "research_question": (
                    "Does enabling object-level cropping alone improve over one-shot, "
                    "and is finer-grained visual evidence more valuable than having more keyframes?"
                ),
                "expected_findings": [
                    "Crops-only should outperform one-shot (validates object-level detail value)",
                    "Crops vs views comparison reveals granularity preference",
                    "Crops-only may excel for attribute verification tasks",
                    "Tool call frequency indicates when agent needs object detail",
                ],
                "key_metrics": [
                    "success_rate_delta_vs_oneshot",
                    "success_rate_delta_vs_views_only",
                    "success_rate_delta_vs_full",
                    "avg_tool_calls_per_sample",
                    "insufficient_evidence_rate",
                ],
                "comparison_baselines": [
                    "oneshot_no_tools (TASK-040)",
                    "views_only (TASK-041)",
                    "full (all tools enabled)",
                ],
                "hypothesis": (
                    "Object-level cropping is most valuable for tasks requiring fine-grained "
                    "attribute or state verification, while keyframe expansion is better for "
                    "spatial reasoning and scene-level understanding."
                ),
            },
        }


# =============================================================================
# Mock Data Factories (Shared)
# =============================================================================


def create_mock_samples_factory(benchmark: str, n_samples: int = 50):
    """Create mock samples for a specific benchmark.

    Creates diverse samples that stress-test object cropping capabilities.
    """
    if benchmark == "openeqa":
        return _create_mock_openeqa_samples(n_samples)
    elif benchmark == "sqa3d":
        return _create_mock_sqa3d_samples(n_samples)
    elif benchmark == "scanrefer":
        return _create_mock_scanrefer_samples(n_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _create_mock_openeqa_samples(n_samples: int):
    """Create mock OpenEQA samples."""
    from conceptgraph.evaluation.scripts.run_openeqa_oneshot import (
        create_mock_openeqa_samples,
    )

    return create_mock_openeqa_samples(n_samples)


def _create_mock_sqa3d_samples(n_samples: int):
    """Create mock SQA3D samples."""
    from conceptgraph.evaluation.scripts.run_sqa3d_oneshot import (
        create_mock_sqa3d_samples,
    )

    return create_mock_sqa3d_samples(n_samples)


def _create_mock_scanrefer_samples(n_samples: int):
    """Create mock ScanRefer samples."""
    from conceptgraph.evaluation.scripts.run_scanrefer_oneshot import (
        create_mock_scanrefer_samples,
    )

    return create_mock_scanrefer_samples(n_samples)


def create_mock_stage1_factory(benchmark: str):
    """Create mock Stage 1 factory for a specific benchmark."""
    if benchmark == "openeqa":
        from conceptgraph.evaluation.scripts.run_openeqa_oneshot import (
            create_mock_stage1_factory,
        )

        return create_mock_stage1_factory()
    elif benchmark == "sqa3d":
        from conceptgraph.evaluation.scripts.run_sqa3d_oneshot import (
            create_mock_stage1_factory,
        )

        return create_mock_stage1_factory()
    elif benchmark == "scanrefer":
        from conceptgraph.evaluation.scripts.run_scanrefer_oneshot import (
            create_mock_stage1_factory,
        )

        return create_mock_stage1_factory()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def create_mock_stage2_factory(benchmark: str):
    """Create mock Stage 2 factory for crops-only ablation.

    This factory creates a mock agent that can use request_crops
    but not request_more_views or hypothesis_repair.
    """
    if benchmark == "openeqa":
        from conceptgraph.evaluation.scripts.run_openeqa_stage2_full import (
            create_mock_stage2_factory,
        )

        return create_mock_stage2_factory()
    elif benchmark == "sqa3d":
        from conceptgraph.evaluation.scripts.run_sqa3d_stage2_full import (
            create_mock_stage2_factory,
        )

        return create_mock_stage2_factory()
    elif benchmark == "scanrefer":
        from conceptgraph.evaluation.scripts.run_scanrefer_stage2_full import (
            create_mock_stage2_factory,
        )

        return create_mock_stage2_factory()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# =============================================================================
# Crops-Only Ablation Runner
# =============================================================================


class CropsOnlyAblationRunner:
    """Runner for crops-only ablation study across benchmarks.

    This runner executes the Stage 2 agent with only the request_crops
    tool enabled, isolating the contribution of object-level evidence
    acquisition (cropping/zooming) to task performance.

    Attributes:
        config: The ablation configuration to use.
        use_mock: Whether to use mock data (for testing).
        max_samples: Maximum samples per benchmark.
        max_workers: Number of parallel workers.
        output_dir: Directory for saving results.
    """

    def __init__(
        self,
        config: Optional[AblationConfig] = None,
        use_mock: bool = False,
        max_samples: Optional[int] = None,
        max_workers: int = 4,
        output_dir: Path = Path("results/ablations/crops_only"),
    ):
        """Initialize the ablation runner.

        Args:
            config: Ablation configuration (defaults to CROPS_ONLY_ABLATION_CONFIG).
            use_mock: Use mock data for testing.
            max_samples: Maximum samples per benchmark.
            max_workers: Number of parallel workers.
            output_dir: Output directory for results.
        """
        self.config = config or CROPS_ONLY_ABLATION_CONFIG
        self.use_mock = use_mock
        self.max_samples = max_samples
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate config matches crops-only ablation requirements
        self._validate_crops_only_config()

    def _validate_crops_only_config(self):
        """Validate that config matches crops-only ablation requirements."""
        issues = []

        # Must have multi-turn for tool usage
        if self.config.agent.max_turns < 2:
            issues.append(
                f"max_turns should be >= 2 for crops-only ablation, got {self.config.agent.max_turns}"
            )

        # request_crops MUST be enabled (the key tool)
        if not self.config.tools.request_crops:
            issues.append("request_crops MUST be True for crops-only ablation")

        # Other evidence-seeking tools should be disabled
        if self.config.tools.request_more_views:
            issues.append("request_more_views should be False for crops-only ablation")
        if self.config.tools.switch_or_expand_hypothesis:
            issues.append(
                "switch_or_expand_hypothesis should be False for crops-only ablation"
            )

        if issues:
            logger.warning(f"Config validation issues: {issues}")
            # Don't error - allow override for experimentation

    def run_benchmark(
        self,
        benchmark: str,
        data_root: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Run crops-only ablation on a single benchmark.

        Args:
            benchmark: Benchmark name (openeqa, sqa3d, scanrefer).
            data_root: Root path to benchmark data.

        Returns:
            BenchmarkResult with evaluation metrics.
        """
        logger.info(f"Running crops-only ablation on {benchmark}...")
        start_time = datetime.now()

        try:
            # Create batch eval config from ablation config
            batch_config = self._create_batch_config(benchmark)

            # Load samples
            samples = self._load_samples(benchmark, data_root)
            if not samples:
                return BenchmarkResult(
                    benchmark=benchmark,
                    run_result=None,
                    error="No samples loaded",
                    duration_seconds=0.0,
                )

            # Get stage factories (mock or real)
            stage1_factory = None
            stage2_factory = None
            if self.use_mock:
                stage1_factory = create_mock_stage1_factory(benchmark)
                stage2_factory = create_mock_stage2_factory(benchmark)

            # Create evaluator
            evaluator = BatchEvaluator(
                batch_config,
                stage1_factory=stage1_factory,
                stage2_factory=stage2_factory,
            )

            # Scene path provider
            def scene_path_provider(scene_id: str) -> Path:
                if self.use_mock:
                    return Path(f"/mock/scenes/{scene_id}")
                if data_root:
                    return data_root / "data" / "frames" / scene_id
                return Path(f"/data/{benchmark}/{scene_id}")

            # Run evaluation
            run_result = evaluator.run(samples, scene_path_provider)

            duration = (datetime.now() - start_time).total_seconds()

            return BenchmarkResult(
                benchmark=benchmark,
                run_result=run_result,
                error=None,
                duration_seconds=duration,
                config=(
                    batch_config.__dict__ if hasattr(batch_config, "__dict__") else {}
                ),
            )

        except Exception as e:
            logger.exception(f"Error running {benchmark}: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            return BenchmarkResult(
                benchmark=benchmark,
                run_result=None,
                error=str(e),
                duration_seconds=duration,
            )

    def _create_batch_config(self, benchmark: str) -> BatchEvalConfig:
        """Create BatchEvalConfig from ablation config for a specific benchmark."""
        run_id = f"crops_only_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return BatchEvalConfig(
            run_id=run_id,
            benchmark_name=benchmark,
            max_workers=self.max_workers,
            # Stage 1 configuration
            stage1_model=self.config.stage1.model,
            stage1_k=self.config.stage1.k,
            stage1_timeout_seconds=self.config.stage1.timeout_seconds,
            # Stage 2 configuration (multi-turn with crops only)
            stage2_enabled=self.config.stage2.enabled,
            stage2_model=self.config.stage2.model,
            stage2_max_turns=self.config.agent.max_turns,
            stage2_plan_mode=self.config.agent.plan_mode,
            stage2_timeout_seconds=self.config.stage2.timeout_seconds,
            # Tool ablations (ONLY request_crops enabled)
            enable_tool_request_more_views=self.config.tools.request_more_views,  # False
            enable_tool_request_crops=self.config.tools.request_crops,  # True
            enable_tool_hypothesis_repair=self.config.tools.switch_or_expand_hypothesis,  # False
            # Uncertainty
            enable_uncertainty_stopping=self.config.agent.enable_uncertainty_stopping,
            confidence_threshold=self.config.agent.confidence_threshold,
            # Output
            output_dir=str(self.output_dir / benchmark),
            save_raw_outputs=True,
            save_tool_traces=True,
            # Limits
            max_samples=self.max_samples,
        )

    def _load_samples(self, benchmark: str, data_root: Optional[Path]):
        """Load samples for a specific benchmark."""
        if self.use_mock:
            mock_samples = create_mock_samples_factory(
                benchmark, self.max_samples or 50
            )
            # Use benchmark-specific adapters for mock samples
            if benchmark == "openeqa":
                return adapt_openeqa_samples(mock_samples)
            elif benchmark == "sqa3d":
                from conceptgraph.evaluation.batch_eval import adapt_sqa3d_samples

                return adapt_sqa3d_samples(mock_samples)
            elif benchmark == "scanrefer":
                from conceptgraph.evaluation.batch_eval import adapt_scanrefer_samples

                return adapt_scanrefer_samples(mock_samples)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")

        if data_root is None:
            raise ValueError(f"data_root required for {benchmark} with real data")

        if benchmark == "openeqa":
            from conceptgraph.benchmarks.openeqa_loader import OpenEQADataset

            dataset = OpenEQADataset.from_path(data_root, max_samples=self.max_samples)
            return adapt_openeqa_samples(list(dataset))
        elif benchmark == "sqa3d":
            from conceptgraph.benchmarks.sqa3d_loader import SQA3DDataset

            dataset = SQA3DDataset.from_path(data_root, max_samples=self.max_samples)
            from conceptgraph.evaluation.batch_eval import adapt_sqa3d_samples

            return adapt_sqa3d_samples(list(dataset))
        elif benchmark == "scanrefer":
            from conceptgraph.benchmarks.scanrefer_loader import ScanReferDataset

            dataset = ScanReferDataset.from_path(
                data_root, max_samples=self.max_samples
            )
            from conceptgraph.evaluation.batch_eval import adapt_scanrefer_samples

            return adapt_scanrefer_samples(list(dataset))
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    def run_all(
        self,
        benchmarks: Union[str, List[str]] = "all",
        data_roots: Optional[Dict[str, Path]] = None,
    ) -> AblationStudyResult:
        """Run crops-only ablation across specified benchmarks.

        Args:
            benchmarks: "all" or list of benchmark names.
            data_roots: Dict mapping benchmark names to data root paths.

        Returns:
            AblationStudyResult with aggregated metrics.
        """
        if benchmarks == "all":
            benchmarks_list = SUPPORTED_BENCHMARKS
        else:
            benchmarks_list = (
                [benchmarks] if isinstance(benchmarks, str) else benchmarks
            )

        data_roots = data_roots or {}

        logger.info(f"Starting crops-only ablation study: {benchmarks_list}")
        logger.info(f"Ablation config: {self.config.name}")
        logger.info(f"  max_turns: {self.config.agent.max_turns}")
        logger.info(
            f"  tools.request_more_views: {self.config.tools.request_more_views}"
        )
        logger.info(f"  tools.request_crops: {self.config.tools.request_crops}")
        logger.info(
            f"  tools.switch_or_expand_hypothesis: {self.config.tools.switch_or_expand_hypothesis}"
        )
        logger.info(f"  use_mock: {self.use_mock}")

        study_result = AblationStudyResult(
            ablation_name=self.config.name,
            ablation_description=self.config.description,
            timestamp=datetime.now().isoformat(),
        )

        for benchmark in benchmarks_list:
            data_root = data_roots.get(benchmark)
            result = self.run_benchmark(benchmark, data_root)
            study_result.benchmark_results.append(result)

            # Aggregate stats
            if result.run_result:
                study_result.total_samples += result.run_result.total_samples
                study_result.total_successful += result.run_result.successful_samples
                study_result.total_failed_stage1 += result.run_result.failed_stage1
                study_result.total_failed_stage2 += result.run_result.failed_stage2
            study_result.total_duration_seconds += result.duration_seconds

        # Save aggregated results
        self._save_results(study_result)

        return study_result

    def _save_results(self, result: AblationStudyResult) -> None:
        """Save ablation study results to JSON."""
        output_path = (
            self.output_dir
            / f"ablation_crops_only_{result.timestamp.replace(':', '-')}.json"
        )

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.success(f"Saved ablation results to {output_path}")

        # Also save a summary table
        summary_path = self.output_dir / "ablation_crops_only_summary.json"
        summary = {
            "ablation_name": result.ablation_name,
            "timestamp": result.timestamp,
            "overall_success_rate": result.overall_success_rate,
            "per_benchmark_success_rates": result.per_benchmark_success_rates,
            "total_samples": result.total_samples,
            "total_successful": result.total_successful,
            "ablation_settings": {
                "request_more_views": False,
                "request_crops": True,
                "switch_or_expand_hypothesis": False,
                "max_turns": self.config.agent.max_turns,
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Top-level API
# =============================================================================


def run_crops_only_ablation(
    benchmarks: Union[str, List[str]] = "all",
    use_mock: bool = False,
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    output_dir: Path = Path("results/ablations/crops_only"),
    data_roots: Optional[Dict[str, Path]] = None,
) -> AblationStudyResult:
    """Run crops-only (request_crops only) ablation study.

    This is the main entry point for TASK-042.

    Args:
        benchmarks: "all" or list of benchmark names.
        use_mock: Use mock data for testing.
        max_samples: Maximum samples per benchmark.
        max_workers: Number of parallel workers.
        output_dir: Output directory for results.
        data_roots: Dict mapping benchmark names to data root paths.

    Returns:
        AblationStudyResult with cross-benchmark metrics.
    """
    runner = CropsOnlyAblationRunner(
        use_mock=use_mock,
        max_samples=max_samples,
        max_workers=max_workers,
        output_dir=output_dir,
    )
    return runner.run_all(benchmarks, data_roots)


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run crops-only (request_crops only) ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Benchmarks to run: 'all' or comma-separated list (openeqa,sqa3d,scanrefer)",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        help="Data root (used if only one benchmark specified)",
    )
    parser.add_argument(
        "--openeqa_root",
        type=Path,
        help="Path to OpenEQA dataset",
    )
    parser.add_argument(
        "--sqa3d_root",
        type=Path,
        help="Path to SQA3D dataset",
    )
    parser.add_argument(
        "--scanrefer_root",
        type=Path,
        help="Path to ScanRefer dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/ablations/crops_only"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show config and exit without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Parse benchmarks
    if args.benchmarks == "all":
        benchmarks = "all"
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Build data_roots dict
    data_roots = {}
    if args.openeqa_root:
        data_roots["openeqa"] = args.openeqa_root
    if args.sqa3d_root:
        data_roots["sqa3d"] = args.sqa3d_root
    if args.scanrefer_root:
        data_roots["scanrefer"] = args.scanrefer_root
    if args.data_root and isinstance(benchmarks, list) and len(benchmarks) == 1:
        data_roots[benchmarks[0]] = args.data_root

    # Dry run - show config only
    if args.dry_run:
        logger.info("DRY RUN: Configuration")
        logger.info(f"  benchmarks: {benchmarks}")
        logger.info(f"  use_mock: {args.mock}")
        logger.info(f"  max_samples: {args.max_samples}")
        logger.info(f"  max_workers: {args.max_workers}")
        logger.info(f"  output_dir: {args.output_dir}")
        logger.info("")
        logger.info("Ablation Config:")
        logger.info(f"  name: {CROPS_ONLY_ABLATION_CONFIG.name}")
        logger.info(f"  max_turns: {CROPS_ONLY_ABLATION_CONFIG.agent.max_turns}")
        logger.info(f"  plan_mode: {CROPS_ONLY_ABLATION_CONFIG.agent.plan_mode}")
        logger.info(
            f"  request_more_views: {CROPS_ONLY_ABLATION_CONFIG.tools.request_more_views}"
        )
        logger.info(
            f"  request_crops: {CROPS_ONLY_ABLATION_CONFIG.tools.request_crops}"
        )
        logger.info(
            f"  switch_or_expand_hypothesis: {CROPS_ONLY_ABLATION_CONFIG.tools.switch_or_expand_hypothesis}"
        )
        return

    # Validate data requirements
    if not args.mock:
        missing = []
        resolved_benchmarks = (
            SUPPORTED_BENCHMARKS if benchmarks == "all" else benchmarks
        )
        for b in resolved_benchmarks:
            if b not in data_roots:
                missing.append(b)
        if missing:
            parser.error(
                f"Data roots required for benchmarks: {missing}. "
                "Use --mock for testing or provide --<benchmark>_root paths."
            )

    try:
        result = run_crops_only_ablation(
            benchmarks=benchmarks,
            use_mock=args.mock,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            data_roots=data_roots,
        )

        # Print summary
        _print_summary(result)
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Ablation study failed: {e}")
        sys.exit(1)


def _print_summary(result: AblationStudyResult) -> None:
    """Print ablation study summary to console."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION STUDY SUMMARY: Crops-Only (request_crops only)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Ablation:          {result.ablation_name}")
    logger.info(f"Description:       {result.ablation_description}")
    logger.info(f"Timestamp:         {result.timestamp}")
    logger.info("")
    logger.info(f"Total Samples:     {result.total_samples}")
    logger.info(f"Total Successful:  {result.total_successful}")
    logger.info(f"Failed Stage 1:    {result.total_failed_stage1}")
    logger.info(f"Failed Stage 2:    {result.total_failed_stage2}")
    logger.info(f"Overall Success:   {result.overall_success_rate:.1%}")
    logger.info(f"Total Duration:    {result.total_duration_seconds:.1f}s")
    logger.info("")
    logger.info("Per-Benchmark Success Rates:")
    for benchmark, rate in result.per_benchmark_success_rates.items():
        status = "✓" if rate > 0.5 else "✗"
        logger.info(f"  {benchmark}: {rate:.1%} {status}")
    logger.info("")
    logger.info("Academic Notes:")
    logger.info("  - This ablation isolates object-level evidence acquisition")
    logger.info("  - request_crops=True, other evidence tools disabled")
    logger.info("  - Compare: oneshot < crops_only <= full (expected ordering)")
    logger.info("  - Crops vs views_only reveals granularity preference")
    logger.info("  - Best for attribute verification and fine-grained tasks")
    logger.info("")


if __name__ == "__main__":
    main()
