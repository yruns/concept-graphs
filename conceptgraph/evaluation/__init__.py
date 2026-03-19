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
from .ablation_config import (
    AblationConfig,
    ToolConfig,
    AgentConfig,
    Stage1Config,
    Stage2Config,
    EvaluationConfig,
    get_preset_config,
    get_all_presets,
    generate_ablation_matrix,
    load_experiment_configs,
    save_ablation_matrix,
)
from .trace_integration import (
    EvalTraceMetadata,
    EvalTraceManager,
    TracingBatchEvaluatorMixin,
    create_tracing_evaluator,
    export_run_trace_report,
)
from .result_tables import (
    MethodResult,
    BenchmarkResultSet,
    PaperResults,
    create_mock_results,
    generate_table1_main_results,
    generate_table2_ablation_results,
    generate_all_tables,
    load_results_from_directory,
)
from .visualizations import (
    DetectionDropDataPoint,
    ToolUsageData,
    ConfidenceAccuracyPoint,
    generate_detection_drop_data,
    generate_tool_usage_data,
    generate_confidence_accuracy_data,
    create_detection_drop_figure,
    create_tool_usage_figure,
    create_confidence_accuracy_figure,
    create_all_figures,
)
from .experimental_analysis import (
    BenchmarkAnalysis,
    AblationAnalysis,
    ExperimentalAnalysis,
    compute_benchmark_analysis,
    compute_ablation_analysis,
    compute_full_analysis,
    generate_main_results_analysis,
    generate_ablation_analysis_text,
    generate_robustness_analysis,
    generate_tool_usage_analysis,
    generate_calibration_analysis,
    generate_experimental_analysis_section,
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
    # Ablation configuration
    "AblationConfig",
    "ToolConfig",
    "AgentConfig",
    "Stage1Config",
    "Stage2Config",
    "EvaluationConfig",
    "get_preset_config",
    "get_all_presets",
    "generate_ablation_matrix",
    "load_experiment_configs",
    "save_ablation_matrix",
    # Trace integration
    "EvalTraceMetadata",
    "EvalTraceManager",
    "TracingBatchEvaluatorMixin",
    "create_tracing_evaluator",
    "export_run_trace_report",
    # Result tables
    "MethodResult",
    "BenchmarkResultSet",
    "PaperResults",
    "create_mock_results",
    "generate_table1_main_results",
    "generate_table2_ablation_results",
    "generate_all_tables",
    "load_results_from_directory",
    # Visualizations
    "DetectionDropDataPoint",
    "ToolUsageData",
    "ConfidenceAccuracyPoint",
    "generate_detection_drop_data",
    "generate_tool_usage_data",
    "generate_confidence_accuracy_data",
    "create_detection_drop_figure",
    "create_tool_usage_figure",
    "create_confidence_accuracy_figure",
    "create_all_figures",
    # Experimental Analysis
    "BenchmarkAnalysis",
    "AblationAnalysis",
    "ExperimentalAnalysis",
    "compute_benchmark_analysis",
    "compute_ablation_analysis",
    "compute_full_analysis",
    "generate_main_results_analysis",
    "generate_ablation_analysis_text",
    "generate_robustness_analysis",
    "generate_tool_usage_analysis",
    "generate_calibration_analysis",
    "generate_experimental_analysis_section",
]
