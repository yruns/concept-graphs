"""Ablation study runners for two-stage 3D scene understanding evaluation.

This package provides unified runners for systematic ablation studies
across all benchmarks (OpenEQA, SQA3D, ScanRefer).

TASK-040 through TASK-044 implement the following ablation conditions:
- TASK-040: No tool calls (one-shot) - baseline for evidence-seeking comparison
- TASK-041: + request_more_views only
- TASK-042: + request_crops only
- TASK-043: + hypothesis_repair only
- TASK-044: + uncertainty output

Academic Support:
- Systematic ablation enables isolated component analysis
- Cross-benchmark comparison reveals generalization patterns
- Unified reporting supports academic table generation
"""

from conceptgraph.evaluation.ablations.run_oneshot_ablation import (
    OneshotAblationRunner,
    run_oneshot_ablation,
)
from conceptgraph.evaluation.ablations.run_views_only_ablation import (
    ViewsOnlyAblationRunner,
    run_views_only_ablation,
    VIEWS_ONLY_ABLATION_CONFIG,
)

__all__ = [
    "OneshotAblationRunner",
    "run_oneshot_ablation",
    "ViewsOnlyAblationRunner",
    "run_views_only_ablation",
    "VIEWS_ONLY_ABLATION_CONFIG",
]
