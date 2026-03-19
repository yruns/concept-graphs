"""Stage-2 agent package built on LangChain v1 and DeepAgents."""

from .adapters import build_object_context, build_stage2_evidence_bundle
from .stage1_callbacks import (
    Stage1BackendCallbacks,
    create_crop_callback,
    create_hypothesis_callback,
    create_more_views_callback,
)
from .models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2EvidenceCitation,
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
    Stage2TaskType,
    Stage2ToolObservation,
    Stage2ToolResult,
)
from .stage2_deep_agent import Stage2DeepResearchAgent
from .trace import (
    ExecutionTrace,
    HTMLTraceRenderer,
    TraceRecorder,
    save_trace_report,
)
from .trace_server import (
    TraceDB,
    TracingAgent,
    TraceServer,
)

__all__ = [
    "ExecutionTrace",
    "HTMLTraceRenderer",
    "KeyframeEvidence",
    "Stage1BackendCallbacks",
    "Stage1HypothesisSummary",
    "Stage2AgentResult",
    "Stage2DeepAgentConfig",
    "Stage2DeepResearchAgent",
    "Stage2EvidenceBundle",
    "Stage2EvidenceCitation",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2StructuredResponse",
    "Stage2TaskSpec",
    "Stage2TaskType",
    "Stage2ToolObservation",
    "Stage2ToolResult",
    "TraceDB",
    "TraceRecorder",
    "TraceServer",
    "TracingAgent",
    "build_object_context",
    "build_stage2_evidence_bundle",
    "create_crop_callback",
    "create_hypothesis_callback",
    "create_more_views_callback",
    "save_trace_report",
]
