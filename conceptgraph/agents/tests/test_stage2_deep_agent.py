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
    Stage2StructuredResponse,
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
    def test_tool_choice_compatible_client_maps_required_tool_choice_to_auto(
        self,
    ) -> None:
        with patch("langchain_openai.AzureChatOpenAI.bind_tools") as bind_tools_mock:
            model = object.__new__(ToolChoiceCompatibleAzureChatOpenAI)
            ToolChoiceCompatibleAzureChatOpenAI.bind_tools(
                model,
                tools=[],
                tool_choice="any",
            )

        self.assertEqual(bind_tools_mock.call_args.kwargs["tool_choice"], "auto")

    def test_get_llm_uses_single_key_azure_client_with_prompt_caching_payload(
        self,
    ) -> None:
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                session_id="stage2-session",
                include_thoughts=True,
                extra_body={"custom_flag": "x"},
            )
        )

        with patch(
            "conceptgraph.agents.stage2_deep_agent.ToolChoiceCompatibleAzureChatOpenAI"
        ) as azure_mock:
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

        with patch(
            "conceptgraph.agents.stage2_deep_agent.ToolChoiceCompatibleAzureChatOpenAI"
        ) as azure_mock:
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
        self.assertEqual(
            bundle.hypothesis.target_categories, ["pillow", "throw_pillow"]
        )
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
        updated_bundle.keyframes = (
            [updated_bundle.keyframes[0]] if updated_bundle.keyframes else []
        )

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

        context_text = tools["retrieve_object_context"].invoke(
            {"object_terms": ["sofa"]}
        )
        self.assertIn("sofa", context_text)
        self.assertNotIn("floor lamp", context_text)

        tools["request_more_views"].invoke(
            {
                "request_text": "Need a wider view",
                "frame_indices": [0],
                "object_terms": ["sofa"],
            }
        )
        self.assertEqual(runtime.bundle.scene_summary, "updated")
        self.assertEqual(len(runtime.tool_trace), 2)

    def test_build_agent_uses_deepagents_response_format_and_full_mode_subagents(
        self,
    ) -> None:
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_subagents=True)
        )
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

        with patch(
            "conceptgraph.agents.stage2_deep_agent.create_deep_agent"
        ) as create_agent_mock:
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

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
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
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=False)
        )
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

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
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
            {
                "request_text": "Need wider angle",
                "frame_indices": [],
                "object_terms": [],
            }
        )

        # After calling the tool
        self.assertTrue(runtime.evidence_updated)
        self.assertEqual(len(runtime.bundle.keyframes), 2)


if __name__ == "__main__":
    unittest.main()


class TestUncertaintyAwareStopping(unittest.TestCase):
    """Tests for uncertainty-aware stopping logic (TASK-013)."""

    def test_config_has_confidence_threshold_with_default(self) -> None:
        """Verify confidence_threshold is configurable with a sensible default."""
        config = Stage2DeepAgentConfig()
        self.assertEqual(config.confidence_threshold, 0.4)
        self.assertTrue(config.enable_uncertainty_stopping)

    def test_config_confidence_threshold_custom_value(self) -> None:
        """Verify confidence_threshold can be set to custom values."""
        config = Stage2DeepAgentConfig(
            confidence_threshold=0.7,
            enable_uncertainty_stopping=True,
        )
        self.assertEqual(config.confidence_threshold, 0.7)

    def test_config_confidence_threshold_bounds_validation(self) -> None:
        """Verify confidence_threshold is bounded between 0 and 1."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            Stage2DeepAgentConfig(confidence_threshold=-0.1)

        with self.assertRaises(ValidationError):
            Stage2DeepAgentConfig(confidence_threshold=1.1)

        # Valid edge cases
        config_zero = Stage2DeepAgentConfig(confidence_threshold=0.0)
        self.assertEqual(config_zero.confidence_threshold, 0.0)

        config_one = Stage2DeepAgentConfig(confidence_threshold=1.0)
        self.assertEqual(config_one.confidence_threshold, 1.0)

    def test_low_confidence_completed_becomes_insufficient_evidence(self) -> None:
        """Verify that low-confidence COMPLETED responses are downgraded."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                confidence_threshold=0.5,
                enable_uncertainty_stopping=True,
            )
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the lamp?",
            max_reasoning_turns=1,
        )
        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg")],
        )

        # Mock response with low confidence
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "The lamp might be on the table.",
                    "confidence": 0.3,  # Below threshold of 0.5
                    "uncertainties": ["Object partially occluded"],
                    "cited_frame_indices": [0],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {"answer": "On the table", "supporting_claims": []},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Should be downgraded to INSUFFICIENT_EVIDENCE
        self.assertEqual(result.result.status, Stage2Status.INSUFFICIENT_EVIDENCE)
        self.assertIn("below threshold", result.result.uncertainties[-1])
        # Original payload should be preserved
        self.assertEqual(result.result.payload["answer"], "On the table")

    def test_high_confidence_completed_remains_completed(self) -> None:
        """Verify that high-confidence responses remain COMPLETED."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                confidence_threshold=0.5,
                enable_uncertainty_stopping=True,
            )
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the pillow?",
        )
        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg")],
        )

        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "The pillow is on the sofa.",
                    "confidence": 0.95,  # Above threshold
                    "uncertainties": [],
                    "cited_frame_indices": [0],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {"answer": "On the sofa", "supporting_claims": []},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Should remain COMPLETED
        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertEqual(result.result.confidence, 0.95)

    def test_needs_more_evidence_becomes_insufficient_when_exhausted(self) -> None:
        """Verify NEEDS_MORE_EVIDENCE becomes INSUFFICIENT_EVIDENCE when no callbacks."""
        # Agent without any evidence callbacks
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                enable_uncertainty_stopping=True,
            ),
            more_views_callback=None,
            crop_callback=None,
            hypothesis_callback=None,
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the hidden object?",
            max_reasoning_turns=1,
        )
        bundle = Stage2EvidenceBundle(scene_id="room0", keyframes=[])

        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "needs_more_evidence",
                    "summary": "Cannot find the object in current views.",
                    "confidence": 0.2,
                    "uncertainties": ["Object not visible in any frame"],
                    "cited_frame_indices": [],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Should be upgraded to INSUFFICIENT_EVIDENCE
        self.assertEqual(result.result.status, Stage2Status.INSUFFICIENT_EVIDENCE)
        self.assertIn(
            "Unable to acquire additional evidence", result.result.uncertainties[-1]
        )

    def test_uncertainty_stopping_disabled_preserves_original_status(self) -> None:
        """Verify that disabling uncertainty stopping preserves original status."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                confidence_threshold=0.5,
                enable_uncertainty_stopping=False,  # Disabled
            )
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the lamp?",
        )
        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg")],
        )

        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "The lamp is somewhere.",
                    "confidence": 0.2,  # Below threshold, but stopping disabled
                    "uncertainties": [],
                    "cited_frame_indices": [],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {"answer": "Somewhere", "supporting_claims": []},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Should remain COMPLETED since stopping is disabled
        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertEqual(result.result.confidence, 0.2)

    def test_agent_self_reported_insufficient_evidence_preserved(self) -> None:
        """Verify agent's self-reported INSUFFICIENT_EVIDENCE status is preserved."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=True)
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Is there a red balloon?",
        )
        bundle = Stage2EvidenceBundle(scene_id="room0", keyframes=[])

        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "insufficient_evidence",
                    "summary": "Cannot determine if there is a red balloon.",
                    "confidence": 0.1,
                    "uncertainties": ["No keyframes showing balloons"],
                    "cited_frame_indices": [],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Agent's self-reported status should be preserved
        self.assertEqual(result.result.status, Stage2Status.INSUFFICIENT_EVIDENCE)
        self.assertEqual(
            result.result.summary, "Cannot determine if there is a red balloon."
        )

    def test_system_prompt_includes_confidence_threshold(self) -> None:
        """Verify the system prompt instructs agent about confidence threshold."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(confidence_threshold=0.6)
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Test query",
        )
        prompt = agent._build_system_prompt(task)

        self.assertIn("0.60", prompt)
        self.assertIn("insufficient_evidence", prompt)
        self.assertIn("Uncertainty-aware stopping", prompt)

    def test_can_acquire_more_evidence_with_callbacks(self) -> None:
        """Verify evidence acquisition is possible when callbacks are configured."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                confidence_threshold=0.5,
                enable_uncertainty_stopping=True,
            ),
            more_views_callback=lambda b, r: b,  # Dummy callback
        )
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="Where is the item?",
            max_reasoning_turns=3,
        )
        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[KeyframeEvidence(keyframe_idx=0, image_path="/tmp/frame0.jpg")],
        )

        # Simulate low confidence with status COMPLETED
        # When callbacks are available but loop exits due to COMPLETED status,
        # we're at turn 1 out of 3, so can_acquire_more_evidence is True
        # -> status should remain COMPLETED (not downgraded)
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",  # Agent says completed
                    "summary": "Uncertain answer",
                    "confidence": 0.3,  # Below threshold
                    "uncertainties": ["Needs more views"],
                    "cited_frame_indices": [],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {},
                },
                "messages": [],
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # With callbacks available and turns remaining, the low-confidence COMPLETED
        # should remain COMPLETED (can still acquire more evidence in principle)
        self.assertEqual(result.result.status, Stage2Status.COMPLETED)


class TestApplyUncertaintyStopping(unittest.TestCase):
    """Direct tests for _apply_uncertainty_stopping method."""

    def setUp(self) -> None:
        self.agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                confidence_threshold=0.5,
                enable_uncertainty_stopping=True,
            )
        )

    def _make_response(
        self,
        status: Stage2Status,
        confidence: float,
        **kwargs,
    ) -> Stage2StructuredResponse:
        return Stage2StructuredResponse(
            task_type=Stage2TaskType.QA,
            status=status,
            summary="Test summary",
            confidence=confidence,
            uncertainties=kwargs.get("uncertainties", []),
            cited_frame_indices=kwargs.get("cited_frame_indices", []),
            evidence_items=kwargs.get("evidence_items", []),
            plan=kwargs.get("plan", []),
            payload=kwargs.get("payload", {}),
        )

    def test_completed_above_threshold_unchanged(self) -> None:
        """COMPLETED with confidence above threshold remains unchanged."""
        response = self._make_response(Stage2Status.COMPLETED, 0.8)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.COMPLETED)

    def test_completed_below_threshold_no_evidence_downgraded(self) -> None:
        """COMPLETED below threshold without more evidence becomes INSUFFICIENT_EVIDENCE."""
        response = self._make_response(Stage2Status.COMPLETED, 0.3)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.INSUFFICIENT_EVIDENCE)

    def test_completed_below_threshold_with_evidence_unchanged(self) -> None:
        """COMPLETED below threshold but with evidence available remains unchanged."""
        response = self._make_response(Stage2Status.COMPLETED, 0.3)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=True
        )
        # Can still acquire evidence, so don't downgrade
        self.assertEqual(result.status, Stage2Status.COMPLETED)

    def test_needs_more_evidence_no_callbacks_upgraded(self) -> None:
        """NEEDS_MORE_EVIDENCE without callbacks becomes INSUFFICIENT_EVIDENCE."""
        response = self._make_response(Stage2Status.NEEDS_MORE_EVIDENCE, 0.3)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.INSUFFICIENT_EVIDENCE)

    def test_needs_more_evidence_with_callbacks_unchanged(self) -> None:
        """NEEDS_MORE_EVIDENCE with callbacks remains unchanged."""
        response = self._make_response(Stage2Status.NEEDS_MORE_EVIDENCE, 0.3)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=True
        )
        self.assertEqual(result.status, Stage2Status.NEEDS_MORE_EVIDENCE)

    def test_failed_status_unchanged(self) -> None:
        """FAILED status is never changed."""
        response = self._make_response(Stage2Status.FAILED, 0.0)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.FAILED)

    def test_insufficient_evidence_preserved(self) -> None:
        """INSUFFICIENT_EVIDENCE status is preserved."""
        response = self._make_response(Stage2Status.INSUFFICIENT_EVIDENCE, 0.2)
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.INSUFFICIENT_EVIDENCE)

    def test_downgrade_preserves_payload(self) -> None:
        """Downgrading status preserves the original payload."""
        response = self._make_response(
            Stage2Status.COMPLETED,
            0.2,
            payload={"answer": "test answer", "details": [1, 2, 3]},
            cited_frame_indices=[0, 1],
        )
        result = self.agent._apply_uncertainty_stopping(
            response, can_acquire_more_evidence=False
        )
        self.assertEqual(result.status, Stage2Status.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.payload["answer"], "test answer")
        self.assertEqual(result.cited_frame_indices, [0, 1])
