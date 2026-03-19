"""Metrics aggregation module for two-stage 3D scene understanding evaluation.

This module provides tools for:
- Aggregating evaluation results across multiple benchmarks
- Grouping results by ablation configurations for comparative analysis
- Exporting results to LaTeX tables for academic papers

Academic Innovation Support:
- Tracks evidence acquisition patterns (tool usage statistics)
- Quantifies symbolic-to-visual repair effectiveness
- Measures uncertainty calibration across conditions
- Enables cross-benchmark comparison for unified multi-task claims
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from loguru import logger

from .batch_eval import BatchEvalConfig, EvalRunResult, EvalSampleResult

# =============================================================================
# Data Structures for Aggregated Results
# =============================================================================


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a single benchmark.

    Contains both standard evaluation metrics and Stage 2 agent-specific
    metrics that support academic claims about evidence-seeking behavior.
    """

    benchmark_name: str
    ablation_tag: str = "full"

    # Sample counts
    total_samples: int = 0
    successful_samples: int = 0
    failed_stage1: int = 0
    failed_stage2: int = 0

    # Primary evaluation metrics (benchmark-specific)
    accuracy: float = 0.0
    accuracy_std: float = 0.0
    exact_match: float = 0.0
    partial_match: float = 0.0

    # Stage 2 agent metrics (for academic claims)
    avg_confidence: float = 0.0
    avg_tool_calls: float = 0.0
    tool_use_rate: float = 0.0
    insufficient_evidence_rate: float = 0.0

    # Latency statistics
    avg_stage1_latency_ms: float = 0.0
    avg_stage2_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Tool usage breakdown
    tool_usage_distribution: Dict[str, int] = field(default_factory=dict)

    # Raw per-sample metrics for statistical analysis
    sample_accuracies: List[float] = field(default_factory=list)
    sample_confidences: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Overall success rate (both stages passed)."""
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples

    @property
    def stage1_success_rate(self) -> float:
        """Stage 1 only success rate."""
        if self.total_samples == 0:
            return 0.0
        return (self.total_samples - self.failed_stage1) / self.total_samples

    @property
    def stage2_success_rate(self) -> float:
        """Stage 2 success rate (given Stage 1 passed)."""
        stage1_passed = self.total_samples - self.failed_stage1
        if stage1_passed == 0:
            return 0.0
        return (stage1_passed - self.failed_stage2) / stage1_passed


@dataclass
class AblationGroup:
    """A group of metrics for a specific ablation configuration.

    Used for organizing results in comparative ablation studies.
    """

    ablation_tag: str
    description: str = ""
    benchmarks: Dict[str, BenchmarkMetrics] = field(default_factory=dict)

    def get_metric(self, benchmark: str, metric_name: str) -> Optional[float]:
        """Get a specific metric for a benchmark."""
        if benchmark not in self.benchmarks:
            return None
        bm = self.benchmarks[benchmark]
        return getattr(bm, metric_name, None)


@dataclass
class AggregatedResults:
    """Complete aggregated results across all ablations and benchmarks.

    This is the top-level container for metrics aggregation, enabling:
    - Cross-ablation comparison (e.g., full vs no_crops vs oneshot)
    - Cross-benchmark comparison (e.g., OpenEQA vs SQA3D)
    - Statistical significance analysis
    """

    ablation_groups: Dict[str, AblationGroup] = field(default_factory=dict)
    benchmarks: List[str] = field(default_factory=list)

    def get_group(self, ablation_tag: str) -> Optional[AblationGroup]:
        """Get an ablation group by its tag."""
        return self.ablation_groups.get(ablation_tag)

    def get_benchmark_metrics(
        self, ablation_tag: str, benchmark: str
    ) -> Optional[BenchmarkMetrics]:
        """Get metrics for a specific ablation and benchmark."""
        group = self.get_group(ablation_tag)
        if group is None:
            return None
        return group.benchmarks.get(benchmark)

    def list_ablations(self) -> List[str]:
        """List all ablation tags in a meaningful order."""
        # Order: full first, then alphabetically, oneshot/stage1_only last
        tags = list(self.ablation_groups.keys())
        priority = {"full": 0, "stage1_only": 100, "oneshot": 99}
        return sorted(tags, key=lambda x: (priority.get(x, 50), x))


# =============================================================================
# Metrics Aggregation Functions
# =============================================================================


def aggregate_run_result(result: EvalRunResult) -> BenchmarkMetrics:
    """Aggregate metrics from a single evaluation run.

    Args:
        result: Evaluation run result from BatchEvaluator.

    Returns:
        Aggregated benchmark metrics.
    """
    # Extract config for ablation tag
    config_dict = result.config
    ablation_tag = "full"
    if isinstance(config_dict, dict):
        # Reconstruct ablation tag from config
        if not config_dict.get("stage2_enabled", True):
            ablation_tag = "stage1_only"
        else:
            tags = []
            if not config_dict.get("enable_tool_request_more_views", True):
                tags.append("no_views")
            if not config_dict.get("enable_tool_request_crops", True):
                tags.append("no_crops")
            if not config_dict.get("enable_tool_hypothesis_repair", True):
                tags.append("no_repair")
            if not config_dict.get("enable_uncertainty_stopping", True):
                tags.append("no_uncertainty")
            if config_dict.get("stage2_max_turns", 6) == 1:
                tags.append("oneshot")
            ablation_tag = "_".join(tags) if tags else "full"

    metrics = BenchmarkMetrics(
        benchmark_name=result.benchmark_name,
        ablation_tag=ablation_tag,
        total_samples=result.total_samples,
        successful_samples=result.successful_samples,
        failed_stage1=result.failed_stage1,
        failed_stage2=result.failed_stage2,
        avg_confidence=result.avg_stage2_confidence,
        avg_tool_calls=result.avg_tool_calls_per_sample,
        avg_stage1_latency_ms=result.avg_stage1_latency_ms,
        avg_stage2_latency_ms=result.avg_stage2_latency_ms,
        tool_usage_distribution=dict(result.tool_usage_distribution),
    )

    # Compute success-based rates
    if result.total_samples > 0:
        metrics.tool_use_rate = result.samples_with_tool_use / result.total_samples
        metrics.insufficient_evidence_rate = (
            result.samples_with_insufficient_evidence / result.total_samples
        )

    # Compute latency total
    metrics.total_latency_ms = (
        metrics.avg_stage1_latency_ms + metrics.avg_stage2_latency_ms
    )

    # Extract per-sample metrics for statistical analysis
    for sample_result in result.results:
        # Extract accuracy from sample metrics if available
        if "accuracy" in sample_result.metrics:
            metrics.sample_accuracies.append(sample_result.metrics["accuracy"])
        elif "score" in sample_result.metrics:
            metrics.sample_accuracies.append(sample_result.metrics["score"])

        if sample_result.stage2_success:
            metrics.sample_confidences.append(sample_result.stage2_confidence)

    # Compute aggregate accuracy metrics
    if metrics.sample_accuracies:
        metrics.accuracy = sum(metrics.sample_accuracies) / len(
            metrics.sample_accuracies
        )
        # Compute standard deviation
        if len(metrics.sample_accuracies) > 1:
            mean = metrics.accuracy
            variance = sum((x - mean) ** 2 for x in metrics.sample_accuracies) / len(
                metrics.sample_accuracies
            )
            metrics.accuracy_std = variance**0.5

        # Count exact and partial matches
        metrics.exact_match = sum(
            1 for x in metrics.sample_accuracies if x >= 0.9
        ) / len(metrics.sample_accuracies)
        metrics.partial_match = sum(
            1 for x in metrics.sample_accuracies if x >= 0.5
        ) / len(metrics.sample_accuracies)

    return metrics


def aggregate_multiple_runs(
    results: List[EvalRunResult],
) -> AggregatedResults:
    """Aggregate metrics from multiple evaluation runs.

    Automatically groups results by benchmark and ablation configuration.

    Args:
        results: List of evaluation run results.

    Returns:
        Aggregated results organized by ablation and benchmark.
    """
    aggregated = AggregatedResults()
    benchmark_set = set()

    for run_result in results:
        metrics = aggregate_run_result(run_result)
        benchmark_set.add(metrics.benchmark_name)

        # Get or create ablation group
        if metrics.ablation_tag not in aggregated.ablation_groups:
            aggregated.ablation_groups[metrics.ablation_tag] = AblationGroup(
                ablation_tag=metrics.ablation_tag,
                description=_describe_ablation(metrics.ablation_tag),
            )

        group = aggregated.ablation_groups[metrics.ablation_tag]
        group.benchmarks[metrics.benchmark_name] = metrics

    aggregated.benchmarks = sorted(benchmark_set)
    return aggregated


def _describe_ablation(tag: str) -> str:
    """Generate human-readable description for ablation tag."""
    descriptions = {
        "full": "Full Stage 2 agent with all tools",
        "stage1_only": "Stage 1 only (no VLM agent)",
        "oneshot": "One-shot VLM (no tool calls)",
        "no_views": "Without request_more_views tool",
        "no_crops": "Without request_crops tool",
        "no_repair": "Without hypothesis_repair tool",
        "no_uncertainty": "Without uncertainty stopping",
    }

    if tag in descriptions:
        return descriptions[tag]

    # Handle combined tags
    parts = tag.split("_")
    components = []
    i = 0
    while i < len(parts):
        if parts[i] == "no" and i + 1 < len(parts):
            key = f"no_{parts[i+1]}"
            if key in descriptions:
                components.append(descriptions[key])
            i += 2
        else:
            i += 1

    return "; ".join(components) if components else tag


# =============================================================================
# LaTeX Table Export Functions
# =============================================================================


def export_to_latex_table(
    aggregated: AggregatedResults,
    metrics: Sequence[str] = ("accuracy", "exact_match", "avg_tool_calls"),
    caption: str = "Evaluation Results",
    label: str = "tab:results",
    precision: int = 3,
    highlight_best: bool = True,
    percentage_metrics: Sequence[str] = ("accuracy", "exact_match", "partial_match"),
) -> str:
    """Export aggregated results to a LaTeX table.

    Generates a publication-ready LaTeX table comparing ablation conditions
    across benchmarks.

    Args:
        aggregated: Aggregated results from multiple runs.
        metrics: List of metric names to include in columns.
        caption: Table caption.
        label: LaTeX label for referencing.
        precision: Decimal precision for numeric values.
        highlight_best: Whether to bold the best value in each column.
        percentage_metrics: Metrics to display as percentages.

    Returns:
        LaTeX table string.
    """
    ablations = aggregated.list_ablations()
    benchmarks = aggregated.benchmarks

    if not ablations or not benchmarks:
        return "% No data to export"

    # Build header
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")

    # Column specification: ablation | benchmark metrics...
    num_cols = 1 + len(benchmarks) * len(metrics)
    col_spec = "l" + "c" * (num_cols - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # First header row: empty + benchmark names (spanning metric columns)
    header1_parts = [""]
    for benchmark in benchmarks:
        header1_parts.append(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{_format_benchmark_name(benchmark)}}}"
        )
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Add cmidrule under each benchmark
    col_idx = 2
    cmidrules = []
    for _ in benchmarks:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + len(metrics) - 1}}}")
        col_idx += len(metrics)
    lines.append(" ".join(cmidrules))

    # Second header row: Ablation + metric names
    header2_parts = ["Ablation"]
    for _ in benchmarks:
        for metric in metrics:
            header2_parts.append(_format_metric_name(metric))
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # Find best values for highlighting
    best_values: Dict[Tuple[str, str], float] = {}
    if highlight_best:
        for benchmark in benchmarks:
            for metric in metrics:
                best_val = None
                for ablation in ablations:
                    bm = aggregated.get_benchmark_metrics(ablation, benchmark)
                    if bm is not None:
                        val = getattr(bm, metric, None)
                        if val is not None:
                            if best_val is None or _is_better(metric, val, best_val):
                                best_val = val
                if best_val is not None:
                    best_values[(benchmark, metric)] = best_val

    # Data rows
    for ablation in ablations:
        row_parts = [_format_ablation_name(ablation)]
        for benchmark in benchmarks:
            bm = aggregated.get_benchmark_metrics(ablation, benchmark)
            for metric in metrics:
                if bm is None:
                    row_parts.append("--")
                else:
                    val = getattr(bm, metric, None)
                    if val is None:
                        row_parts.append("--")
                    else:
                        # Format value
                        is_percentage = metric in percentage_metrics
                        formatted = _format_value(val, precision, is_percentage)

                        # Bold if best
                        if (
                            highlight_best
                            and (benchmark, metric) in best_values
                            and abs(val - best_values[(benchmark, metric)]) < 1e-9
                        ):
                            formatted = f"\\textbf{{{formatted}}}"

                        row_parts.append(formatted)

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def export_tool_usage_table(
    aggregated: AggregatedResults,
    caption: str = "Tool Usage Distribution",
    label: str = "tab:tool-usage",
) -> str:
    """Export tool usage statistics to a LaTeX table.

    Generates a table showing how often each tool was used across
    ablation conditions, supporting the "adaptive evidence acquisition" claim.

    Args:
        aggregated: Aggregated results from multiple runs.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    # Collect all unique tools
    all_tools = set()
    for group in aggregated.ablation_groups.values():
        for bm in group.benchmarks.values():
            all_tools.update(bm.tool_usage_distribution.keys())

    tools = sorted(all_tools)
    if not tools:
        return "% No tool usage data"

    ablations = aggregated.list_ablations()
    benchmarks = aggregated.benchmarks

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")

    # Columns: Ablation | Benchmark | Tool1 | Tool2 | ...
    col_spec = "ll" + "r" * len(tools)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header_parts = ["Ablation", "Benchmark"] + [_format_tool_name(t) for t in tools]
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for ablation in ablations:
        group = aggregated.get_group(ablation)
        if group is None:
            continue

        first_row = True
        for benchmark in benchmarks:
            bm = group.benchmarks.get(benchmark)
            if bm is None:
                continue

            row_parts = []
            # Use multirow for ablation name
            if first_row:
                num_benchmarks = len([b for b in benchmarks if b in group.benchmarks])
                row_parts.append(
                    f"\\multirow{{{num_benchmarks}}}{{*}}{{{_format_ablation_name(ablation)}}}"
                )
                first_row = False
            else:
                row_parts.append("")

            row_parts.append(_format_benchmark_name(benchmark))
            for tool in tools:
                count = bm.tool_usage_distribution.get(tool, 0)
                row_parts.append(str(count))

            lines.append(" & ".join(row_parts) + " \\\\")

        if not first_row:
            lines.append("\\midrule")

    # Remove the last midrule and replace with bottomrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def export_summary_statistics(
    aggregated: AggregatedResults,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a human-readable summary of all results.

    Args:
        aggregated: Aggregated results.
        output_path: Optional path to write summary file.

    Returns:
        Summary text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    for ablation_tag in aggregated.list_ablations():
        group = aggregated.get_group(ablation_tag)
        if group is None:
            continue

        lines.append(f"## {ablation_tag.upper()}")
        if group.description:
            lines.append(f"   {group.description}")
        lines.append("")

        for benchmark in aggregated.benchmarks:
            bm = group.benchmarks.get(benchmark)
            if bm is None:
                continue

            lines.append(f"   ### {benchmark}")
            lines.append(f"       Samples: {bm.total_samples}")
            lines.append(f"       Success Rate: {bm.success_rate:.1%}")
            lines.append(f"       Accuracy: {bm.accuracy:.3f} ± {bm.accuracy_std:.3f}")
            lines.append(f"       Exact Match: {bm.exact_match:.1%}")
            lines.append(f"       Avg Confidence: {bm.avg_confidence:.3f}")
            lines.append(f"       Avg Tool Calls: {bm.avg_tool_calls:.2f}")
            lines.append(
                f"       Insufficient Evidence: {bm.insufficient_evidence_rate:.1%}"
            )
            lines.append(f"       Stage 1 Latency: {bm.avg_stage1_latency_ms:.0f}ms")
            lines.append(f"       Stage 2 Latency: {bm.avg_stage2_latency_ms:.0f}ms")
            lines.append("")

    summary = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary)
        logger.info(f"Summary written to {output_path}")

    return summary


# =============================================================================
# Helper Functions for LaTeX Formatting
# =============================================================================


def _format_value(value: float, precision: int, as_percentage: bool) -> str:
    """Format a numeric value for LaTeX."""
    if as_percentage:
        return f"{value * 100:.{precision - 2}f}\\%"
    return f"{value:.{precision}f}"


def _format_metric_name(metric: str) -> str:
    """Format metric name for table header."""
    names = {
        "accuracy": "Acc",
        "exact_match": "EM",
        "partial_match": "PM",
        "avg_confidence": "Conf",
        "avg_tool_calls": "Tools",
        "tool_use_rate": "TU\\%",
        "insufficient_evidence_rate": "IE\\%",
        "success_rate": "Succ",
    }
    return names.get(metric, metric.replace("_", " ").title())


def _format_benchmark_name(benchmark: str) -> str:
    """Format benchmark name for table header."""
    names = {
        "openeqa": "OpenEQA",
        "sqa3d": "SQA3D",
        "scanrefer": "ScanRefer",
        "eai": "EAI",
    }
    return names.get(benchmark.lower(), benchmark)


def _format_ablation_name(tag: str) -> str:
    """Format ablation tag for table row."""
    names = {
        "full": "Full",
        "stage1_only": "Stage 1",
        "oneshot": "One-shot",
        "no_views": "$-$Views",
        "no_crops": "$-$Crops",
        "no_repair": "$-$Repair",
        "no_uncertainty": "$-$Uncert.",
    }

    if tag in names:
        return names[tag]

    # Handle combined tags
    parts = []
    for component in tag.split("_"):
        if f"no_{component}" in names:
            parts.append(names[f"no_{component}"])

    return ", ".join(parts) if parts else tag


def _format_tool_name(tool: str) -> str:
    """Format tool name for table header."""
    names = {
        "request_more_views": "Views",
        "request_crops": "Crops",
        "hypothesis_repair": "Repair",
        "inspect_stage1_metadata": "Meta",
        "retrieve_object_context": "Context",
    }
    return names.get(tool, tool.replace("_", " ").title())


def _is_better(metric: str, val1: float, val2: float) -> bool:
    """Determine if val1 is better than val2 for a given metric.

    Most metrics are "higher is better", but some (latency, error rates)
    are "lower is better".
    """
    lower_is_better = {
        "avg_stage1_latency_ms",
        "avg_stage2_latency_ms",
        "total_latency_ms",
        "insufficient_evidence_rate",
        "failed_stage1",
        "failed_stage2",
        "accuracy_std",
    }

    if metric in lower_is_better:
        return val1 < val2
    return val1 > val2
