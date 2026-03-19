"""Shared schemas for the Stage-2 research agent."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Stage2TaskType(str, Enum):
    """Supported downstream task families."""

    QA = "qa"
    VISUAL_GROUNDING = "visual_grounding"
    NAV_PLAN = "nav_plan"
    MANIPULATION = "manipulation"
    GENERAL = "general"


class Stage2PlanMode(str, Enum):
    """How much explicit planning the DeepAgent should do."""

    OFF = "off"
    BRIEF = "brief"
    FULL = "full"


class Stage2Status(str, Enum):
    """Unified status values for Stage-2 outputs."""

    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    FAILED = "failed"


class Stage2TaskSpec(BaseModel):
    """Downstream task specification for the Stage-2 agent."""

    task_type: Stage2TaskType = Stage2TaskType.GENERAL
    user_query: str = Field(..., min_length=1)
    output_instruction: str = ""
    expected_output_schema: Dict[str, Any] = Field(default_factory=dict)
    plan_mode: Stage2PlanMode = Stage2PlanMode.BRIEF
    max_reasoning_turns: int = Field(default=6, ge=1, le=12)


class KeyframeEvidence(BaseModel):
    """One visual evidence item produced by Stage 1."""

    keyframe_idx: int = Field(..., ge=0)
    image_path: str
    view_id: Optional[int] = None
    frame_id: Optional[int] = None
    score: Optional[float] = None
    note: str = ""


class Stage1HypothesisSummary(BaseModel):
    """Compact summary of Stage-1 query grounding metadata."""

    status: str = ""
    hypothesis_kind: str = ""
    hypothesis_rank: Optional[int] = None
    parse_mode: str = ""
    raw_query: str = ""
    target_categories: List[str] = Field(default_factory=list)
    anchor_categories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Stage2EvidenceBundle(BaseModel):
    """Evidence package passed from Stage 1 into the agent."""

    scene_id: str = ""
    stage1_query: str = ""
    keyframes: List[KeyframeEvidence] = Field(default_factory=list)
    bev_image_path: Optional[str] = None
    scene_summary: str = ""
    object_context: Dict[str, str] = Field(default_factory=dict)
    hypothesis: Optional[Stage1HypothesisSummary] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class Stage2EvidenceCitation(BaseModel):
    """One evidence-backed claim in the final response."""

    claim: str = ""
    frame_indices: List[int] = Field(default_factory=list)
    object_terms: List[str] = Field(default_factory=list)


class Stage2ToolObservation(BaseModel):
    """Recorded tool usage during a single agent run."""

    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    response_text: str = ""


class Stage2ToolResult(BaseModel):
    """Normalized result returned by Stage-2 evidence tools."""

    response_text: str
    updated_bundle: Optional[Stage2EvidenceBundle] = None


class Stage2StructuredResponse(BaseModel):
    """Unified structured output envelope returned by Stage 2."""

    task_type: Stage2TaskType
    status: Stage2Status = Stage2Status.COMPLETED
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)
    cited_frame_indices: List[int] = Field(default_factory=list)
    evidence_items: List[Stage2EvidenceCitation] = Field(default_factory=list)
    plan: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)


class Stage2AgentResult(BaseModel):
    """End-to-end Stage-2 execution result."""

    task: Stage2TaskSpec
    result: Stage2StructuredResponse
    tool_trace: List[Stage2ToolObservation] = Field(default_factory=list)
    final_bundle: Stage2EvidenceBundle
    raw_state: Dict[str, Any] = Field(default_factory=dict)


class Stage2DeepAgentConfig(BaseModel):
    """Runtime configuration for the DeepAgents-backed Stage-2 agent."""

    base_url: str = "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl"
    model_name: str = "gemini-2.5-pro"
    api_key: str = "cD6AGSVHrzftqONPxsFmgkVEuVlBynRb_GPT_AK"
    api_version: str = "2024-03-01-preview"
    max_tokens: int = Field(default=10000, ge=1)
    temperature: float = 0.1
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=2, ge=0)
    include_thoughts: bool = True
    session_id: str = Field(default_factory=lambda: str(time.time()))
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    max_images: int = Field(default=6, ge=1, le=12)
    image_max_size: int = Field(default=900, ge=256, le=2048)
    enable_subagents: bool = True
