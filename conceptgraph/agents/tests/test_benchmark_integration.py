"""Integration tests: benchmark samples → Stage1 simulate → Stage2 agent.

This module validates that benchmark samples can flow through the two-stage
pipeline architecture using mocked Stage 1 results. It demonstrates:
1. Benchmark loaders produce well-formed questions
2. Stage 1 → Stage 2 evidence bundle construction works
3. Stage 2 agent can process each benchmark task type
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from conceptgraph.agents.models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2StructuredResponse,
)
from conceptgraph.agents.stage2_deep_agent import _Stage2RuntimeState
from conceptgraph.benchmarks import (
    BoundingBox3D,
    OpenEQADataset,
    OpenEQASample,
    ScanReferSample,
    SQA3DSample,
    SQA3DSituation,
)
from conceptgraph.query_scene.keyframe_selector import KeyframeResult, SceneObject


# ---------------------------------------------------------------------------
# Helper: mock Stage 1 keyframe retrieval
# ---------------------------------------------------------------------------


def mock_stage1_retrieval(
    query: str,
    scene_id: str,
    target_objects: Optional[List[SceneObject]] = None,
    anchor_objects: Optional[List[SceneObject]] = None,
) -> KeyframeResult:
    """Simulate Stage 1 keyframe selector output for a given query."""
    if target_objects is None:
        target_objects = [
            SceneObject(
                obj_id=1,
                category="object",
                object_tag="generic_object",
                summary=f"Object relevant to: {query}",
            )
        ]

    return KeyframeResult(
        query=query,
        target_term=target_objects[0].category if target_objects else "unknown",
        anchor_term=anchor_objects[0].category if anchor_objects else None,
        keyframe_indices=[0, 1, 2],
        keyframe_paths=[
            Path(f"/mock/{scene_id}/frame_{i:06d}.jpg") for i in [0, 50, 100]
        ],
        target_objects=target_objects,
        anchor_objects=anchor_objects or [],
        metadata={
            "status": "direct_grounded",
            "selected_hypothesis_kind": "direct",
            "selected_hypothesis_rank": 1,
            "hypothesis_output": {
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "rank": 1,
                        "kind": "direct",
                        "grounding_query": {
                            "root": {
                                "category": [target_objects[0].category],
                                "spatial_constraints": [],
                            }
                        },
                    }
                ],
            },
        },
    )


class _FakeGraph:
    """Mock DeepAgents graph that returns canned structured responses."""

    def __init__(self, response: Dict[str, Any]):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(payload)
        return self.response


# ---------------------------------------------------------------------------
# OpenEQA Integration Tests
# ---------------------------------------------------------------------------


class TestOpenEQAIntegration(unittest.TestCase):
    """Test OpenEQA benchmark → Stage2 agent integration."""

    def _create_mock_sample(self) -> OpenEQASample:
        """Create a representative OpenEQA sample."""
        return OpenEQASample(
            question_id="test_q1",
            question="What color is the pillow on the sofa?",
            answer="blue",
            episode_history=None,
            scene_id="scene0001",
            question_type="episodic_memory",
            category="object_attributes",
        )

    def test_openeqa_sample_converts_to_stage2_qa_task(self) -> None:
        """OpenEQA question should produce a valid QA task spec."""
        sample = self._create_mock_sample()

        # Simulate Stage 1
        stage1_result = mock_stage1_retrieval(
            query=sample.question,
            scene_id="scene0001",
            target_objects=[
                SceneObject(
                    obj_id=1,
                    category="pillow",
                    object_tag="throw_pillow",
                    summary="A blue pillow resting on the sofa.",
                )
            ],
            anchor_objects=[
                SceneObject(
                    obj_id=2,
                    category="sofa",
                    object_tag="sofa",
                    summary="A large fabric sofa.",
                )
            ],
        )

        # Build evidence bundle
        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id="scene0001",
            scene_summary="Living room with sofa and pillows.",
        )

        # Create task spec
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query=sample.question,
            plan_mode=Stage2PlanMode.BRIEF,
        )

        self.assertEqual(task.task_type, Stage2TaskType.QA)
        self.assertIn("pillow", task.user_query.lower())
        self.assertIsNotNone(bundle.hypothesis)
        self.assertEqual(bundle.hypothesis.target_categories, ["pillow"])

    def test_openeqa_runs_through_stage2_agent(self) -> None:
        """OpenEQA sample can be processed by Stage 2 agent."""
        sample = self._create_mock_sample()

        stage1_result = mock_stage1_retrieval(
            query=sample.question,
            scene_id="scene0001",
        )
        bundle = build_stage2_evidence_bundle(stage1_result, scene_id="scene0001")

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query=sample.question,
        )

        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=False)
        )
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "The pillow is blue.",
                    "confidence": 0.85,
                    "uncertainties": [],
                    "cited_frame_indices": [0],
                    "evidence_items": [
                        {
                            "claim": "Blue pillow visible on sofa",
                            "frame_indices": [0],
                            "object_terms": ["pillow"],
                        }
                    ],
                    "plan": [],
                    "payload": {
                        "answer": "blue",
                        "supporting_claims": ["Visible in frame 0"],
                    },
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertEqual(result.result.payload["answer"], "blue")


# ---------------------------------------------------------------------------
# SQA3D Integration Tests
# ---------------------------------------------------------------------------


class TestSQA3DIntegration(unittest.TestCase):
    """Test SQA3D benchmark → Stage2 agent integration."""

    def _create_mock_sample(self) -> SQA3DSample:
        """Create a representative SQA3D sample with situation context."""
        return SQA3DSample(
            question_id="sqa3d_q1",
            question="What is to my left?",
            answers=["table", "desk"],
            situation=SQA3DSituation(
                position=[1.5, 0.0, 2.0],
                orientation=[0.0, 0.0, 1.0],
                room_description="Standing in the center of the room, facing the window.",
            ),
            scene_id="scene0011_00",
            question_type="what",
            answer_type="single_word",
        )

    def test_sqa3d_sample_includes_situation_context(self) -> None:
        """SQA3D samples should include egocentric situation context."""
        sample = self._create_mock_sample()

        # Verify situation context is accessible
        self.assertEqual(sample.situation.position, [1.5, 0.0, 2.0])
        self.assertIn("facing the window", sample.situation.room_description)
        self.assertEqual(sample.scene_id, "scene0011_00")

    def test_sqa3d_egocentric_query_enrichment(self) -> None:
        """SQA3D situation context should enrich the query for Stage 1."""
        sample = self._create_mock_sample()

        # Enrich query with situation context
        enriched_query = (
            f"{sample.question} "
            f"(Situation: {sample.situation.room_description})"
        )

        stage1_result = mock_stage1_retrieval(
            query=enriched_query,
            scene_id=sample.scene_id,
            target_objects=[
                SceneObject(
                    obj_id=3,
                    category="table",
                    object_tag="table",
                    summary="A wooden table to the left of the agent's position.",
                )
            ],
        )

        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id=sample.scene_id,
            scene_summary=sample.situation.room_description,
        )

        # The bundle should preserve the egocentric context
        self.assertEqual(bundle.scene_summary, sample.situation.room_description)
        self.assertIn("table", bundle.object_context)

    def test_sqa3d_runs_through_stage2_agent(self) -> None:
        """SQA3D sample can be processed by Stage 2 agent."""
        sample = self._create_mock_sample()

        enriched_query = (
            f"{sample.question} "
            f"(Situation: {sample.situation.room_description})"
        )
        stage1_result = mock_stage1_retrieval(
            query=enriched_query,
            scene_id=sample.scene_id,
        )
        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id=sample.scene_id,
            scene_summary=sample.situation.room_description,
        )

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query=enriched_query,
        )

        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=False)
        )
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",
                    "summary": "A table is to your left.",
                    "confidence": 0.78,
                    "uncertainties": ["Exact object category uncertain"],
                    "cited_frame_indices": [0, 1],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {
                        "answer": "table",
                        "supporting_claims": [
                            "Table visible on left side of frame 0",
                        ],
                    },
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertIn("table", result.result.payload["answer"])


# ---------------------------------------------------------------------------
# ScanRefer Integration Tests
# ---------------------------------------------------------------------------


class TestScanReferIntegration(unittest.TestCase):
    """Test ScanRefer benchmark → Stage2 agent integration."""

    def _create_mock_sample(self) -> ScanReferSample:
        """Create a representative ScanRefer sample."""
        return ScanReferSample(
            sample_id="scanrefer_001",
            scene_id="scene0707_00",
            object_id="42",
            object_name="chair",
            description="The brown wooden chair next to the desk.",
            target_bbox=BoundingBox3D(
                center=[1.2, 0.5, 0.8],
                size=[0.6, 0.6, 0.9],
            ),
        )

    def test_scanrefer_sample_has_3d_bbox(self) -> None:
        """ScanRefer samples should include 3D bounding box ground truth."""
        sample = self._create_mock_sample()

        self.assertEqual(sample.target_bbox.center, [1.2, 0.5, 0.8])
        self.assertEqual(sample.target_bbox.size, [0.6, 0.6, 0.9])
        self.assertGreater(sample.target_bbox.volume(), 0)

    def test_scanrefer_converts_to_visual_grounding_task(self) -> None:
        """ScanRefer description should produce a visual grounding task."""
        sample = self._create_mock_sample()

        stage1_result = mock_stage1_retrieval(
            query=sample.description,
            scene_id=sample.scene_id,
            target_objects=[
                SceneObject(
                    obj_id=int(sample.object_id),
                    category=sample.object_name,
                    object_tag=f"{sample.object_name}_{sample.object_id}",
                    summary=sample.description,
                )
            ],
            anchor_objects=[
                SceneObject(
                    obj_id=99,
                    category="desk",
                    object_tag="desk",
                    summary="A wooden desk.",
                )
            ],
        )

        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id=sample.scene_id,
        )

        # For ScanRefer we use VISUAL_GROUNDING task type
        task = Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query=sample.description,
        )

        self.assertEqual(task.task_type, Stage2TaskType.VISUAL_GROUNDING)
        self.assertEqual(bundle.hypothesis.target_categories, ["chair"])

    def test_scanrefer_runs_through_stage2_agent(self) -> None:
        """ScanRefer sample can be processed by Stage 2 agent."""
        sample = self._create_mock_sample()

        stage1_result = mock_stage1_retrieval(
            query=sample.description,
            scene_id=sample.scene_id,
        )
        bundle = build_stage2_evidence_bundle(
            stage1_result,
            scene_id=sample.scene_id,
        )

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query=sample.description,
        )

        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=False)
        )
        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "visual_grounding",
                    "status": "completed",
                    "summary": "Brown chair located next to desk in frame 0.",
                    "confidence": 0.91,
                    "uncertainties": [],
                    "cited_frame_indices": [0],
                    "evidence_items": [
                        {
                            "claim": "Brown wooden chair visible next to desk",
                            "frame_indices": [0],
                            "object_terms": ["chair", "desk"],
                        }
                    ],
                    "plan": [],
                    "payload": {
                        "best_frames": [0],
                        "target_description": "Brown wooden chair next to desk",
                        "grounding_rationale": "Chair clearly visible with matching description",
                    },
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        self.assertEqual(result.result.status, Stage2Status.COMPLETED)
        self.assertEqual(result.result.payload["best_frames"], [0])


# ---------------------------------------------------------------------------
# Cross-Benchmark Pipeline Tests
# ---------------------------------------------------------------------------


class TestCrossBenchmarkPipeline(unittest.TestCase):
    """Test that all benchmark types flow through the unified pipeline."""

    def test_all_benchmark_types_produce_valid_tasks(self) -> None:
        """Each benchmark type should map to a valid Stage2 task type."""
        benchmark_task_mapping = {
            "openeqa": Stage2TaskType.QA,
            "sqa3d": Stage2TaskType.QA,  # SQA3D is QA with situation
            "scanrefer": Stage2TaskType.VISUAL_GROUNDING,  # 3D grounding
        }

        for benchmark, expected_task_type in benchmark_task_mapping.items():
            with self.subTest(benchmark=benchmark):
                task = Stage2TaskSpec(
                    task_type=expected_task_type,
                    user_query=f"Sample query for {benchmark}",
                )
                self.assertEqual(task.task_type, expected_task_type)

    def test_stage2_handles_insufficient_evidence_gracefully(self) -> None:
        """Stage 2 should report insufficient evidence rather than hallucinate."""
        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(
                enable_uncertainty_stopping=True,
                confidence_threshold=0.7,
            )
        )

        task = Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query="What color is the carpet?",
        )

        # Empty bundle simulates failed Stage 1 retrieval
        bundle = Stage2EvidenceBundle(scene_id="unknown_scene", keyframes=[])

        fake_graph = _FakeGraph(
            {
                "structured_response": {
                    "task_type": "qa",
                    "status": "completed",  # Agent claims completion
                    "summary": "Cannot determine carpet color.",
                    "confidence": 0.35,  # Low confidence
                    "uncertainties": ["No carpet visible in keyframes"],
                    "cited_frame_indices": [],
                    "evidence_items": [],
                    "plan": [],
                    "payload": {"answer": "unknown", "supporting_claims": []},
                }
            }
        )

        with patch.object(
            agent, "build_agent", return_value=(fake_graph, _Stage2RuntimeState(bundle))
        ):
            result = agent.run(task, bundle)

        # Uncertainty stopping should downgrade to INSUFFICIENT_EVIDENCE
        self.assertEqual(result.result.status, Stage2Status.INSUFFICIENT_EVIDENCE)

    def test_tool_callbacks_enable_evidence_refinement(self) -> None:
        """Tool callbacks allow Stage 2 to request additional evidence."""
        more_views_invoked = [False]

        def mock_more_views_callback(bundle, request):
            more_views_invoked[0] = True
            # Return updated bundle with additional keyframe
            new_bundle = bundle.model_copy(deep=True)
            new_bundle.keyframes.append(
                KeyframeEvidence(
                    keyframe_idx=99,
                    image_path="/mock/new_view.jpg",
                    note="Additional view from callback",
                )
            )
            return {"response": "Added new view", "updated_bundle": new_bundle}

        agent = Stage2DeepResearchAgent(
            config=Stage2DeepAgentConfig(enable_uncertainty_stopping=False),
            more_views_callback=mock_more_views_callback,
        )

        bundle = Stage2EvidenceBundle(
            scene_id="room0",
            keyframes=[
                KeyframeEvidence(keyframe_idx=0, image_path="/mock/frame0.jpg")
            ],
        )

        # Build tools and invoke request_more_views
        runtime = _Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        tools = {tool.name: tool for tool in agent._build_runtime_tools(runtime)}

        response = tools["request_more_views"].invoke(
            {
                "request_text": "Need wider view of the room",
                "frame_indices": [0],
                "object_terms": [],
            }
        )

        self.assertTrue(more_views_invoked[0])
        self.assertEqual(len(runtime.bundle.keyframes), 2)
        self.assertTrue(runtime.evidence_updated)


if __name__ == "__main__":
    unittest.main()
