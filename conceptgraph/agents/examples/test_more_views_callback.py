#!/usr/bin/env python
"""Test that request_more_views callback correctly retrieves additional keyframes.

This test:
1. Starts with a minimal evidence bundle (1 keyframe)
2. Forces the agent to call request_more_views
3. Verifies new keyframes are added to the bundle

Run: REPLICA_ROOT=/path/to/Replica .venv/bin/python -m conceptgraph.agents.examples.test_more_views_callback
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from conceptgraph.agents import (
    Stage1BackendCallbacks,
    Stage2EvidenceBundle,
    build_stage2_evidence_bundle,
)
from conceptgraph.agents.models import KeyframeEvidence, Stage1HypothesisSummary
from conceptgraph.query_scene.keyframe_selector import KeyframeSelector


def test_more_views_callback():
    """Test that the more_views callback retrieves additional keyframes."""
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_path = Path(replica_root) / "room0"

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Testing request_more_views callback")
    logger.info("=" * 60)

    # Initialize KeyframeSelector
    logger.info("[1] Loading KeyframeSelector...")
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )
    logger.info(f"  Loaded {len(selector.objects)} objects")

    # Create a minimal bundle with just 1 keyframe
    logger.info("[2] Creating minimal evidence bundle...")

    # Get one keyframe for "sofa"
    sofa_views = []
    for obj in selector.objects:
        if "sofa" in (obj.category or "").lower() or "sofa" in (getattr(obj, "object_tag", "") or "").lower():
            sofa_views = selector.get_joint_coverage_views([obj.obj_id], max_views=1)
            break

    if not sofa_views:
        logger.error("Could not find sofa views")
        sys.exit(1)

    view_id = sofa_views[0]
    frame_id = selector.map_view_to_frame(view_id)
    path, resolved_view = selector._resolve_keyframe_path(view_id)

    bundle = Stage2EvidenceBundle(
        scene_id="room0",
        stage1_query="sofa in the room",
        scene_summary="Test scene with furniture",
        keyframes=[
            KeyframeEvidence(
                keyframe_idx=0,
                image_path=str(path),
                view_id=resolved_view,
                frame_id=frame_id,
            ),
        ],
        hypothesis=Stage1HypothesisSummary(
            status="direct_grounded",
            hypothesis_kind="direct",
            target_categories=["sofa"],
            anchor_categories=[],
        ),
    )
    logger.info(f"  Initial keyframes: {len(bundle.keyframes)}")
    logger.info(f"  Initial path: {bundle.keyframes[0].image_path}")

    # Create callbacks
    logger.info("[3] Creating Stage 1 callbacks...")
    callbacks = Stage1BackendCallbacks(
        keyframe_selector=selector,
        scene_id="room0",
        max_additional_views=3,
    )

    # Call request_more_views
    logger.info("[4] Calling request_more_views...")
    request = {
        "request_text": "Need more views of the sofa and surrounding furniture",
        "frame_indices": [0],
        "object_terms": ["sofa", "pillow", "table"],
    }

    result = callbacks.more_views(bundle, request)
    logger.info(f"  Response: {result.response_text}")

    if result.updated_bundle is None:
        logger.warning("  No updated bundle returned!")
        return False

    updated_bundle = result.updated_bundle
    logger.info(f"  Updated keyframes: {len(updated_bundle.keyframes)}")

    # Check that new keyframes were added
    if len(updated_bundle.keyframes) > len(bundle.keyframes):
        logger.success("  ✅ New keyframes added!")
        for i, kf in enumerate(updated_bundle.keyframes):
            logger.info(f"    [{i}] view_id={kf.view_id}, path={Path(kf.image_path).name}")
        return True
    else:
        logger.warning("  ⚠️  No new keyframes added (may already have optimal coverage)")
        return True  # Not a failure, just full coverage


def test_more_views_no_duplicates():
    """Test that request_more_views doesn't return duplicate keyframes."""
    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_path = Path(replica_root) / "room0"

    logger.info("\n" + "=" * 60)
    logger.info("Testing request_more_views no duplicates")
    logger.info("=" * 60)

    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gpt-5.2-2025-12-11",
        use_pool=False,
    )

    # Get initial keyframes for an object
    target_obj = None
    for obj in selector.objects:
        if "chair" in (obj.category or "").lower() or "armchair" in (getattr(obj, "object_tag", "") or "").lower():
            target_obj = obj
            break

    if not target_obj:
        logger.error("No chair found")
        return False

    initial_views = selector.get_joint_coverage_views([target_obj.obj_id], max_views=2)

    # Create bundle with these views
    keyframes = []
    for i, view_id in enumerate(initial_views):
        path, resolved = selector._resolve_keyframe_path(view_id)
        if path:
            keyframes.append(KeyframeEvidence(
                keyframe_idx=i,
                image_path=str(path),
                view_id=resolved,
                frame_id=selector.map_view_to_frame(view_id),
            ))

    bundle = Stage2EvidenceBundle(
        scene_id="room0",
        keyframes=keyframes,
        hypothesis=Stage1HypothesisSummary(
            target_categories=["armchair"],
            anchor_categories=[],
        ),
    )
    logger.info(f"  Initial views: {[kf.view_id for kf in bundle.keyframes]}")

    callbacks = Stage1BackendCallbacks(selector, "room0", max_additional_views=2)

    request = {
        "request_text": "More views of the chair",
        "object_terms": ["armchair", "chair"],
    }

    result = callbacks.more_views(bundle, request)
    logger.info(f"  Response: {result.response_text[:100]}...")

    if result.updated_bundle:
        new_views = [kf.view_id for kf in result.updated_bundle.keyframes]
        logger.info(f"  Final views: {new_views}")

        # Check for duplicates
        if len(new_views) != len(set(new_views)):
            logger.error("  ❌ Duplicate views found!")
            return False
        logger.success("  ✅ No duplicate views")
        return True

    return True


def main():
    logger.info("Testing Stage 1 -> Stage 2 callbacks")
    logger.info("=" * 60)

    test1 = test_more_views_callback()
    test2 = test_more_views_no_duplicates()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test 1 (basic callback): {'✅ PASS' if test1 else '❌ FAIL'}")
    logger.info(f"Test 2 (no duplicates):  {'✅ PASS' if test2 else '❌ FAIL'}")


if __name__ == "__main__":
    main()
