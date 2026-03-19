"""DeepAgents-backed Stage-2 research agent."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import AzureChatOpenAI
from loguru import logger

from .models import (
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
    Stage2TaskType,
    Stage2ToolObservation,
    Stage2ToolResult,
)

ToolCallback = Callable[[Stage2EvidenceBundle, Dict[str, Any]], Any]


class GeminiCompatibleAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI variant that avoids Gemini-incompatible required tool mode."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice in ("any", "required", True):
            tool_choice = "auto"
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


@dataclass
class _Stage2RuntimeState:
    """Mutable per-run state shared by DeepAgent tools."""

    bundle: Stage2EvidenceBundle
    tool_trace: List[Stage2ToolObservation] = field(default_factory=list)

    def record(self, tool_name: str, tool_input: Dict[str, Any], response_text: str) -> None:
        self.tool_trace.append(
            Stage2ToolObservation(
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=response_text,
            )
        )


def _default_output_instruction(task_type: Stage2TaskType) -> str:
    if task_type == Stage2TaskType.QA:
        return "Answer the question and keep the answer grounded in cited frames."
    if task_type == Stage2TaskType.VISUAL_GROUNDING:
        return "Identify the best supporting frame(s) and explain the grounding evidence."
    if task_type == Stage2TaskType.NAV_PLAN:
        return "Produce a navigation plan grounded in visible landmarks and uncertainty."
    if task_type == Stage2TaskType.MANIPULATION:
        return "Produce a manipulation plan with visible preconditions and missing evidence."
    return "Produce an evidence-grounded answer with explicit uncertainty."


def _default_payload_schema(task_type: Stage2TaskType) -> Dict[str, Any]:
    if task_type == Stage2TaskType.QA:
        return {"answer": "str", "supporting_claims": ["str"]}
    if task_type == Stage2TaskType.VISUAL_GROUNDING:
        return {
            "best_frames": ["int"],
            "target_description": "str",
            "grounding_rationale": "str",
        }
    if task_type == Stage2TaskType.NAV_PLAN:
        return {
            "subgoals": ["str"],
            "landmarks": ["str"],
            "risks": ["str"],
        }
    if task_type == Stage2TaskType.MANIPULATION:
        return {
            "target_object": "str",
            "preconditions": ["str"],
            "action_sequence": ["str"],
            "failure_checks": ["str"],
        }
    return {"result": "str"}


class Stage2DeepResearchAgent:
    """Stage-2 VLM agent built on LangChain v1 and DeepAgents."""

    def __init__(
        self,
        config: Optional[Stage2DeepAgentConfig] = None,
        more_views_callback: Optional[ToolCallback] = None,
        crop_callback: Optional[ToolCallback] = None,
        hypothesis_callback: Optional[ToolCallback] = None,
    ) -> None:
        self.config = config or Stage2DeepAgentConfig()
        self.more_views_callback = more_views_callback
        self.crop_callback = crop_callback
        self.hypothesis_callback = hypothesis_callback
        self._llm = None
        self._session_id = self.config.session_id

    def _build_extra_body(self) -> Dict[str, Any]:
        """Build the provider-specific extra_body payload for prompt caching."""
        extra_body = dict(self.config.extra_body)
        thinking = dict(extra_body.get("thinking", {}))
        if self.config.include_thoughts:
            thinking["include_thoughts"] = True
        if thinking:
            extra_body["thinking"] = thinking
        extra_body["session_id"] = self._session_id
        return extra_body

    def _get_llm(self) -> AzureChatOpenAI:
        """Return a single-key AzureOpenAI-compatible Gemini chat model."""
        if self._llm is None:
            # Use a stable single-key client instead of GeminiClientPool so the
            # runtime can keep a consistent session_id for provider-side prompt caching.
            self._llm = GeminiCompatibleAzureChatOpenAI(
                azure_deployment=self.config.model_name,
                model=self.config.model_name,
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                extra_body=self._build_extra_body(),
            )
        return self._llm

    def _image_to_data_url(self, image_path: Union[str, Path]) -> str:
        """Convert an image file into a data URL for multimodal chat models."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for Stage-2 image encoding.") from exc

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        if max(width, height) > self.config.image_max_size:
            ratio = self.config.image_max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _collect_image_paths(self, bundle: Stage2EvidenceBundle) -> List[str]:
        """Collect keyframes and optional BEV images for a run."""
        images: List[str] = []
        for keyframe in bundle.keyframes[: self.config.max_images]:
            if Path(keyframe.image_path).exists():
                images.append(keyframe.image_path)

        if (
            bundle.bev_image_path
            and Path(bundle.bev_image_path).exists()
            and len(images) < self.config.max_images
        ):
            images.append(bundle.bev_image_path)

        return images

    def _coerce_callback_result(self, result: Any) -> Stage2ToolResult:
        """Normalize external callback payloads for tool responses."""
        if isinstance(result, Stage2ToolResult):
            return result
        if isinstance(result, Stage2EvidenceBundle):
            return Stage2ToolResult(
                response_text="Received updated evidence bundle.",
                updated_bundle=result,
            )
        if isinstance(result, str):
            return Stage2ToolResult(response_text=result)
        if isinstance(result, dict):
            updated_bundle = result.get("updated_bundle")
            if isinstance(updated_bundle, Stage2EvidenceBundle):
                payload = dict(result)
                payload.pop("updated_bundle", None)
                return Stage2ToolResult(
                    response_text=json.dumps(payload, indent=2, ensure_ascii=False),
                    updated_bundle=updated_bundle,
                )
            return Stage2ToolResult(
                response_text=json.dumps(result, indent=2, ensure_ascii=False)
            )
        return Stage2ToolResult(response_text=str(result))

    def _select_object_context(
        self,
        bundle: Stage2EvidenceBundle,
        object_terms: Optional[Sequence[str]],
    ) -> str:
        """Return the requested subset of object context."""
        if not bundle.object_context:
            return bundle.scene_summary or "No object context or scene summary available."

        if not object_terms:
            return json.dumps(bundle.object_context, indent=2, ensure_ascii=False)

        lowered = [term.lower() for term in object_terms]
        selected: Dict[str, str] = {}
        for key, value in bundle.object_context.items():
            key_lower = key.lower()
            if any(term in key_lower or key_lower in term for term in lowered):
                selected[key] = value

        if not selected:
            return "No matching object context found for requested terms."
        return json.dumps(selected, indent=2, ensure_ascii=False)

    def _build_runtime_tools(self, runtime: _Stage2RuntimeState) -> List[BaseTool]:
        """Create Stage-2 evidence tools bound to one runtime state."""

        @tool
        def inspect_stage1_metadata() -> str:
            """Inspect the Stage-1 hypothesis, selector status, and frame mapping metadata."""

            payload = {
                "hypothesis": runtime.bundle.hypothesis.model_dump()
                if runtime.bundle.hypothesis
                else None,
                "extra_metadata": runtime.bundle.extra_metadata,
                "num_keyframes": len(runtime.bundle.keyframes),
            }
            response = json.dumps(payload, indent=2, ensure_ascii=False)
            runtime.record("inspect_stage1_metadata", {}, response)
            return response

        @tool
        def retrieve_object_context(object_terms: Optional[List[str]] = None) -> str:
            """Retrieve scene-level or object-specific context summaries."""

            request = {"object_terms": object_terms or []}
            response = self._select_object_context(runtime.bundle, object_terms)
            runtime.record("retrieve_object_context", request, response)
            return response

        @tool
        def request_more_views(
            request_text: str,
            frame_indices: Optional[List[int]] = None,
            object_terms: Optional[List[str]] = None,
        ) -> str:
            """Request additional keyframes or neighboring views from Stage 1."""

            request = {
                "request_text": request_text,
                "frame_indices": frame_indices or [],
                "object_terms": object_terms or [],
            }
            if self.more_views_callback is None:
                response = Stage2ToolResult(
                    response_text="request_more_views callback is not configured."
                )
            else:
                response = self._coerce_callback_result(
                    self.more_views_callback(runtime.bundle, request)
                )
                if response.updated_bundle is not None:
                    runtime.bundle = response.updated_bundle
            runtime.record("request_more_views", request, response.response_text)
            return response.response_text

        @tool
        def request_crops(
            request_text: str,
            frame_indices: Optional[List[int]] = None,
            object_terms: Optional[List[str]] = None,
        ) -> str:
            """Request object-centric or region-centric crops from the current evidence."""

            request = {
                "request_text": request_text,
                "frame_indices": frame_indices or [],
                "object_terms": object_terms or [],
            }
            if self.crop_callback is None:
                response = Stage2ToolResult(
                    response_text="request_crops callback is not configured."
                )
            else:
                response = self._coerce_callback_result(
                    self.crop_callback(runtime.bundle, request)
                )
                if response.updated_bundle is not None:
                    runtime.bundle = response.updated_bundle
            runtime.record("request_crops", request, response.response_text)
            return response.response_text

        @tool
        def switch_or_expand_hypothesis(
            request_text: str,
            preferred_kind: Optional[str] = None,
        ) -> str:
            """Request Stage-1 hypothesis expansion or direct/proxy/context switching."""

            request = {
                "request_text": request_text,
                "preferred_kind": preferred_kind or "",
            }
            if self.hypothesis_callback is None:
                response = Stage2ToolResult(
                    response_text="switch_or_expand_hypothesis callback is not configured."
                )
            else:
                response = self._coerce_callback_result(
                    self.hypothesis_callback(runtime.bundle, request)
                )
                if response.updated_bundle is not None:
                    runtime.bundle = response.updated_bundle
            runtime.record("switch_or_expand_hypothesis", request, response.response_text)
            return response.response_text

        return [
            inspect_stage1_metadata,
            retrieve_object_context,
            request_more_views,
            request_crops,
            switch_or_expand_hypothesis,
        ]

    def _build_system_prompt(self, task: Stage2TaskSpec) -> str:
        """Build the DeepAgents system prompt."""
        plan_instructions = {
            Stage2PlanMode.OFF: (
                "Plan mode is OFF. Only use the todo list if the task is unexpectedly complex."
            ),
            Stage2PlanMode.BRIEF: (
                "Plan mode is BRIEF. Before major evidence collection, keep a short todo list "
                "with 2-4 items covering evidence acquisition and answer synthesis."
            ),
            Stage2PlanMode.FULL: (
                "Plan mode is FULL. Maintain an explicit todo list throughout execution and "
                "decompose work into evidence acquisition, verification, and task synthesis."
            ),
        }

        payload_schema = task.expected_output_schema or _default_payload_schema(task.task_type)
        instruction = task.output_instruction or _default_output_instruction(task.task_type)

        return (
            "You are the Stage-2 research agent for query-scene.\n\n"
            "Research role:\n"
            "- Stage 1 is a high-recall evidence retriever, not ground truth.\n"
            "- Stage 2 must verify, repair, or reject Stage-1 hypotheses using pixels.\n"
            "- Prefer evidence-seeking behavior over one-shot answering.\n"
            "- Use tools when keyframes are insufficient; do not hallucinate missing evidence.\n"
            "- Explicitly surface uncertainty when the necessary evidence is absent.\n\n"
            "Framework constraints:\n"
            "- This runtime is built with LangChain v1 and DeepAgents.\n"
            "- Use the built-in todo planning capability according to the selected plan mode.\n"
            "- Subagents may be used in FULL mode when decomposition is useful.\n\n"
            f"{plan_instructions[task.plan_mode]}\n\n"
            "Unified output contract:\n"
            f"- task_type must be `{task.task_type.value}`.\n"
            "- status must reflect whether the task is complete or evidence-limited.\n"
            "- summary must be concise and evidence-grounded.\n"
            "- confidence must stay calibrated.\n"
            "- uncertainties must list missing or ambiguous evidence.\n"
            "- cited_frame_indices must only cite visible supporting frames.\n"
            "- evidence_items should map concrete claims to frames and objects.\n"
            "- payload should follow the expected task-specific schema below.\n\n"
            f"Task-specific instruction: {instruction}\n"
            f"Expected payload schema: {json.dumps(payload_schema, indent=2, ensure_ascii=False)}"
        )

    def _build_subagents(self, task: Stage2TaskSpec) -> List[Dict[str, Any]]:
        """Build optional DeepAgents subagents for richer decomposition."""
        if not self.config.enable_subagents or task.plan_mode != Stage2PlanMode.FULL:
            return []

        return [
            {
                "name": "evidence_scout",
                "description": "Diagnose evidence gaps and decide which view/crop/hypothesis tool to call next.",
                "system_prompt": (
                    "You are the evidence scout. Focus only on whether current keyframes are "
                    "sufficient, which missing views or crops are needed, and what uncertainty "
                    "remains. Do not produce the final user-facing answer."
                ),
            },
            {
                "name": "task_head",
                "description": "Synthesize the final task-specific payload from collected evidence.",
                "system_prompt": (
                    "You are the task head. Use the collected evidence to assemble the final "
                    "task-specific payload. Stay faithful to cited frames and explicit uncertainty."
                ),
            },
        ]

    def _build_user_message(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> HumanMessage:
        """Assemble the multimodal task message for the DeepAgent."""
        keyframe_lines = []
        for keyframe in bundle.keyframes:
            keyframe_lines.append(
                f"- idx={keyframe.keyframe_idx}, view_id={keyframe.view_id}, "
                f"frame_id={keyframe.frame_id}, note={keyframe.note or 'N/A'}"
            )
        if not keyframe_lines:
            keyframe_lines.append("- no keyframes available")

        hypothesis_text = (
            json.dumps(bundle.hypothesis.model_dump(), indent=2, ensure_ascii=False)
            if bundle.hypothesis
            else "{}"
        )
        payload_schema = task.expected_output_schema or _default_payload_schema(task.task_type)
        instruction = task.output_instruction or _default_output_instruction(task.task_type)

        prompt = (
            f"Task type: {task.task_type.value}\n"
            f"User query: {task.user_query}\n"
            f"Plan mode: {task.plan_mode.value}\n"
            f"Output instruction: {instruction}\n"
            f"Expected payload schema: {json.dumps(payload_schema, indent=2, ensure_ascii=False)}\n\n"
            f"Stage-1 query: {bundle.stage1_query or task.user_query}\n"
            f"Scene id: {bundle.scene_id or 'unknown'}\n\n"
            f"Current keyframes:\n{chr(10).join(keyframe_lines)}\n\n"
            f"Stage-1 hypothesis summary:\n{hypothesis_text}\n\n"
            f"Scene summary:\n{bundle.scene_summary or 'N/A'}\n\n"
            f"Available object context keys:\n"
            f"{sorted(bundle.object_context.keys()) if bundle.object_context else []}\n\n"
            "Use tools when evidence is missing. Return the final answer through the "
            "structured response schema."
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in self._collect_image_paths(bundle):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_path)},
                }
            )
        return HumanMessage(content=content)

    def build_agent(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
        """Compile a DeepAgent and return it with runtime state."""
        runtime = _Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        tools = self._build_runtime_tools(runtime)
        graph = create_deep_agent(
            model=self._get_llm(),
            tools=tools,
            system_prompt=self._build_system_prompt(task),
            subagents=self._build_subagents(task),
            response_format=Stage2StructuredResponse,
            name="query_scene_stage2_agent",
        )
        return graph, runtime

    def _normalize_final_response(
        self,
        task: Stage2TaskSpec,
        raw_state: Dict[str, Any],
    ) -> Stage2StructuredResponse:
        """Convert DeepAgents final state into the unified Stage-2 schema."""
        structured = raw_state.get("structured_response")
        if structured is not None:
            response = Stage2StructuredResponse.model_validate(structured)
            if response.task_type != task.task_type:
                response.task_type = task.task_type
            return response

        return Stage2StructuredResponse(
            task_type=task.task_type,
            status=Stage2Status.FAILED,
            summary="The agent returned without a structured response.",
            confidence=0.0,
            uncertainties=["Missing structured_response in DeepAgents final state."],
            payload={},
        )

    def run(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle) -> Stage2AgentResult:
        """Execute the Stage-2 DeepAgent end to end."""
        graph, runtime = self.build_agent(task, bundle)
        message = self._build_user_message(task, runtime.bundle)
        logger.info(
            "[Stage2DeepResearchAgent] task={} plan_mode={} keyframes={}",
            task.task_type.value,
            task.plan_mode.value,
            len(runtime.bundle.keyframes),
        )
        raw_state = graph.invoke({"messages": [message]})
        final_response = self._normalize_final_response(task, raw_state)
        return Stage2AgentResult(
            task=task,
            result=final_response,
            tool_trace=runtime.tool_trace,
            final_bundle=runtime.bundle,
            raw_state={k: v for k, v in raw_state.items() if k != "messages"},
        )
