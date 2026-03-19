from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from conceptgraph.agents import (
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2PlanMode,
    Stage2Status,
    Stage2TaskSpec,
    Stage2TaskType,
    build_stage2_evidence_bundle,
)
from conceptgraph.agents.stage2_deep_agent import (
    ToolChoiceCompatibleAzureChatOpenAI,
    _Stage2RuntimeState,
)
from conceptgraph.agents.models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
)
from conceptgraph.query_scene.keyframe_selector import KeyframeResult, SceneObject


class _FakeGraph:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def invoke(self, payload: dict) -> dict:
        self.calls.append(payload)
        return self.response


class TestStage2DeepAgent(unittest.TestCase):
    def test_tool_choice_compatible_client_maps_required_tool_choice_to_auto(self) -> None:
        with patch("langchain_openai.AzureChatOpenAI.bind_tools") as bind_tools_mock:
            model = object.__new__(ToolChoiceCompatibleAzureChatOpenAI)
            ToolChoiceCompatibleAzureChatOpenAI.bind_tools(
                model,
                tools=[],
                tool_choice="any",
            )

        self.assertEqual(bind_tools_mock.call_args.kwargs["tool_choice"], "auto")

    def test_get_llm_uses_single_key_azure_client_with_prompt_caching_payload(self) -> None:
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                session_id="stage2-session",
                include_thoughts=True,
                extra_body={"custom_flag": "x"},
            )
        )

        with patch("conceptgraph.agents.stage2_deep_agent.ToolChoiceCompatibleAzureChatOpenAI") as azure_mock:
            fake_llm = object()
            azure_mock.return_value = fake_llm
            llm = agent._get_llm()

        self.assertIs(llm, fake_llm)
        kwargs = azure_mock.call_args.kwargs
        self.assertEqual(kwargs["azure_endpoint"], agent.config.base_url)
        self.assertEqual(kwargs["api_key"], agent.config.api_key)
        self.assertEqual(kwargs["api_version"], agent.config.api_version)
        self.assertEqual(kwargs["max_tokens"], 10000)
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["extra_body"]["session_id"], "stage2-session")
        self.assertTrue(kwargs["extra_body"]["thinking"]["include_thoughts"])
        self.assertEqual(kwargs["extra_body"]["custom_flag"], "x")

    def test_get_llm_omits_thinking_block_when_disabled(self) -> None:
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                session_id="stage2-session",
                include_thoughts=False,
            )
        )

        with patch("conceptgraph.agents.stage2_deep_agent.ToolChoiceCompatibleAzureChatOpenAI") as azure_mock:
            fake_llm = object()
            azure_mock.return_value = fake_llm
            agent._get_llm()

        kwargs = azure_mock.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"session_id": "stage2-session"})

    def test_build_stage2_evidence_bundle_extracts_hypothesis_and_context(self) -> None:
        result = KeyframeResult(
            query="pillow on the sofa",
            target_term="pillow",
            anchor_term="sofa",
            keyframe_indices=[12],
            keyframe_paths=[Path("/tmp/frame000012.jpg")],
            target_objects=[
                SceneObject(
                    obj_id=1,
                    category="pillow",
                    object_tag="throw_pillow",
                    summary="A small white pillow on the sofa.",
                    affordances={"soft": True},
                )
            ],
            anchor_objects=[
                SceneObject(
                    obj_id=2,
                    category="sofa",
                    object_tag="sofa",
                    summary="A large fabric sofa near the wall.",
                    co_objects=["throw_pillow"],
                )
            ],
            metadata={
                "status": "direct_grounded",
                "selected_hypothesis_kind": "direct",
                "selected_hypothesis_rank": 1,
                "frame_mappings": [
                    {
                        "requested_view_id": 12,
                        "requested_frame_id": 60,
                        "resolved_view_id": 12,
                        "resolved_frame_id": 60,
                    }
                ],
                "hypothesis_output": {
                    "parse_mode": "multi",
                    "hypotheses": [
                        {
                            "rank": 1,
                            "kind": "direct",
                            "grounding_query": {
                                "root": {
                                    "category": ["pillow", "throw_pillow"],
                                    "spatial_constraints": [
                                        {"anchors": [{"category": ["sofa"]}]}
                                    ],
                                }
                            },
                        }
                    ],
                },
            },
        )

        bundle = build_stage2_evidence_bundle(
            result,
            scene_id="room0",
            scene_summary="Living room with a sofa and pillows.",
        )

        self.assertEqual(bundle.scene_id, "room0")
        self.assertEqual(bundle.hypothesis.hypothesis_kind, "direct")
        self.assertEqual(bundle.hypothesis.target_categories, ["pillow", "throw_pillow"])
        self.assertEqual(bundle.hypothesis.anchor_categories, ["sofa"])
        self.assertIn("throw_pillow", bundle.object_context)
        self.assertIn("A small white pillow", bundle.object_context["throw_pillow"])

    def test_runtime_tools_filter_context_and_apply_callback_updates(self) -> None:
        updated_bundle = build_stage2_evidence_bundle(
            KeyframeResult(
                query="lamp beside sofa",
                target_term="lamp",
                anchor_term="sofa",
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={},
            ),
            scene_id="room0",
        )
        updated_bundle.keyframes = [
            updated_bundle.keyframes[0]
        ] if updated_bundle.keyframes else []

        agent = Stage2DeepResearchAgent(
            more_views_callback=lambda bundle, request: {
                "response": "added new view",
                "updated_bundle": bundle.model_copy(
                    update={
                        "scene_summary": "updated",
                    },
                    deep=True,
                ),
            }
        )
        runtime = _Stage2RuntimeState(
            bundle=updated_bundle.model_copy(
                update={
                    "object_context": {
                        "sofa": "large sofa",
                        "lamp": "floor lamp",
                    }
                },
                deep=True,
            )
        )
        tools = {tool.name: tool for tool in agent._build_runtime_tools(runtime)}

        context_text = tools["retrieve_object_context"].invoke({"object_terms": ["sofa"]})
        self.assertIn("sofa", context_text)
        self.assertNotIn("floor lamp", context_text)

        tools["request_more_views"].invoke(
            {"request_text": "Need a wider view", "frame_indices": [0], "object_terms": ["sofa"]}
        )
        self.assertEqual(runtime.bundle.scene_summary, "updated")
        self.assertEqual(len(runtime.tool_trace), 2)

    def test_build_agent_uses_deepagents_response_format_and_full_mode_subagents(self) -> None:
        agent = Stage2DeepResearchAgent(config=Stage2DeepAgentConfig(enable_subagents=True))
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.NAV_PLAN,
            user_query="Navigate to the sofa.",
            plan_mode=Stage2PlanMode.FULL,
        )
        bundle = build_stage2_evidence_bundle(
            KeyframeResult(
                query="sofa",
                target_term="sofa",
                anchor_term=None,
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={},
            ),
            scene_id="room0",
        )

        with patch("conceptgraph.agents.stage2_deep_agent.create_deep_agent") as create_agent_mock:
            create_agent_mock.return_value = _FakeGraph({"structured_response": {}})
            with patch.object(agent, "_get_llm", return_value=object()):
                agent.build_agent(task, bundle)

        kwargs = create_agent_mock.call_args.kwargs
        self.assertEqual(kwargs["response_format"].__name__, "Stage2StructuredResponse")
        self.assertEqual(len(kwargs["subagents"]), 2)
        self.assertIn("LangChain v1 and DeepAgents", kwargs["system_prompt"])

    def test_run_returns_structured_stage2_result(self) -> None:
        agent = Stage2DeepResearchAgent()
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the pillow?",
        )
        bundle = build_stage2_evidence_bundle(
            KeyframeResult(
                query="pillow on sofa",
                target_term="pillow",
                anchor_term="sofa",
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={},
            ),
            scene_id="room0",
        )
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "The pillow is on the sofa.",
                    "confidence": 0.93,
                    "uncertainties": [],
                    "cited_frame_indices": [0],
                    "evidence_items": [
                        {
                            "claim": "The pillow rests on the sofa.",
                            "frame_indices": [0],
                            "object_terms": ["pillow", "sofa"],
                        }
                    ],
                    "plan": ["Inspect current frames", "Answer the question"],
                    "payload": {
                        "answer": "The pillow is on the sofa.",
                        "supporting_claims": ["Visible in frame 0."],
                    },
                }
            }
        )

        with patch.object(agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))):
            result = agent.run(task, bundle)

        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertEqual(result.result.payload["answer"], "The pillow is on the sofa.")
        self.assertEqual(result.result.cited_frame_indices, [0])

    def test_evidence_update_flag_triggers_image_injection_tracking(self) -> None:
        """Verify that evidence updates are tracked for iterative refinement."""
        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg"),
            ],
        )
        runtime = _Stage2RuntimeState(bundle=bundle)

        # Initially no evidence update
        self.assertFalse(runtime.evidence_updated)
        self.assertFalse(runtime.consume_evidence_update())

        # Mark evidence updated
        runtime.mark_evidence_updated()
        self.assertTrue(runtime.evidence_updated)

        # Consume resets the flag
        self.assertTrue(runtime.consume_evidence_update())
        self.assertFalse(runtime.evidence_updated)
        self.assertFalse(runtime.consume_evidence_update())

    def test_iterative_run_respects_max_reasoning_turns(self) -> None:
        """Verify that the iterative loop respects max_reasoning_turns budget."""
        agent = Stage2DeepResearchAgent()
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the lamp?",
            max_reasoning_turns=2,
        )
        bundle = build_stage2_evidence_bundle(
            KeyframeResult(
                query="lamp",
                target_term="lamp",
                anchor_term=None,
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={},
            ),
            scene_id="room0",
        )

        call_count = [0]

        class _CountingGraph:
            def __init__(self, response: dict):
                self.response = response

            def invoke(self, payload: dict) -> dict:
                call_count[0] += 1
                # Return incomplete status to force loop continuation
                return {
                    "structured_response": {
                        "task_type": "qa",
                        "status": "needs_more_evidence",
                        "summary": "Need more views",
                        "confidence": 0.3,
                        "uncertainties": ["Lamp not visible"],
                        "cited_frame_indices": [],
                        "evidence_items": [],
                        "plan": [],
                        "payload": {},
                    },
                    "messages": [],
                }

        fake_graph = _CountingGraph({})

        with patch.object(agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))):
            result = agent.run(task, bundle)

        # Should stop at 1 invoke since no evidence_updated flag is set
        # (the loop only continues when new evidence is injected)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(result.result.status, Stage2Status.NEEDS_MORE_EVIDENCE)

    def test_runtime_marks_evidence_updated_when_callback_returns_bundle(self) -> None:
        """Verify tools mark evidence_updated when callbacks return new bundles."""
        original_bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg"),
            ],
        )
        updated_bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg"),
                KeyframeEvidence(keyframe_idx=1, image_path="/tmp/frame1.jpg"),
            ],
        )

        agent = Stage2DeepResearchAgent(
            more_views_callback=lambda bundle, request: updated_bundle,
        )
        runtime = _Stage2RuntimeState(bundle=original_bundle.model_copy(deep=True))
        tools = {tool.name: tool for tool in agent._build_runtime_tools(runtime)}

        # Before calling the tool
        self.assertFalse(runtime.evidence_updated)
        self.assertEqual(len(runtime.bundle.keyframes), 1)

        # Call the tool
        tools["request_more_views"].invoke(
            {"request_text": "Need wider angle", "frame_indices": [], "object_terms": []}
        )

        # After calling the tool
        self.assertTrue(runtime.evidence_updated)
        self.assertEqual(len(runtime.bundle.keyframes), 2)


if __name__ == "__main__":
    unittest.main()
