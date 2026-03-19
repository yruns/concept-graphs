"""Stage-2 agent package built on LangChain v1 and DeepAgents."""

from .adapters import build_object_context, build_stage2_evidence_bundle
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

__all__ = [
    "KeyframeEvidence",
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
    "build_object_context",
    "build_stage2_evidence_bundle",
]
