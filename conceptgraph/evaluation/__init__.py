"""Evaluation module for two-stage 3D scene understanding.

This module provides batch evaluation infrastructure for running Stage 1 (keyframe retrieval)
and Stage 2 (VLM agent reasoning) on benchmark datasets.

Academic Innovation Points:
- Adaptive Evidence Acquisition: VLM agent dynamically decides when to request more evidence
- Symbolic-to-Visual Repair: Stage 2 validates and corrects Stage 1 scene graph hypotheses
- Evidence-Grounded Uncertainty: Explicit uncertainty output when evidence is insufficient
- Unified Multi-Task Policy: Single agent architecture handles QA, grounding, navigation, manipulation
"""

from .batch_eval import (
    BatchEvalConfig,
    BatchEvaluator,
    EvalSampleResult,
    EvalRunResult,
)
from .metrics import (
    BenchmarkMetrics,
    AblationGroup,
    AggregatedResults,
    aggregate_run_result,
    aggregate_multiple_runs,
    export_to_latex_table,
    export_tool_usage_table,
    export_summary_statistics,
)

__all__ = [
    # Batch evaluation
    "BatchEvalConfig",
    "BatchEvaluator",
    "EvalSampleResult",
    "EvalRunResult",
    # Metrics aggregation
    "BenchmarkMetrics",
    "AblationGroup",
    "AggregatedResults",
    "aggregate_run_result",
    "aggregate_multiple_runs",
    "export_to_latex_table",
    "export_tool_usage_table",
    "export_summary_statistics",
]
