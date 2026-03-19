"""Stage 1 backend callbacks for Stage 2 agent tools.

This module provides real implementations for:
- request_more_views: retrieve additional keyframes from KeyframeSelector
- request_crops: generate object crops from keyframes (future)
- switch_or_expand_hypothesis: re-run Stage 1 with different hypothesis (future)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from loguru import logger

from .adapters import build_stage2_evidence_bundle
from .models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2ToolResult,
)


def create_more_views_callback(
    keyframe_selector: Any,
    scene_id: str = "",
    max_additional_views: int = 3,
) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
    """Create a callback that retrieves more views from Stage 1 KeyframeSelector.

    The callback:
    1. Extracts object IDs of interest from the request
    2. Uses KeyframeSelector.get_joint_coverage_views() to find additional views
    3. Resolves view IDs to actual image paths
    4. Returns an updated bundle with new keyframes appended

    Args:
        keyframe_selector: KeyframeSelector instance with visibility index
        scene_id: Scene identifier for the bundle
        max_additional_views: Maximum number of new views to add per request

    Returns:
        A callback compatible with Stage2DeepResearchAgent.more_views_callback
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: Dict[str, Any],
    ) -> Stage2ToolResult:
        request_text = request.get("request_text", "")
        requested_frames = request.get("frame_indices", [])
        object_terms = request.get("object_terms", [])

        logger.info(
            "[Stage1Callback] request_more_views: text='{}', frames={}, objects={}",
            request_text[:50],
            requested_frames,
            object_terms,
        )

        # Collect object IDs to consider
        # 1. From existing keyframes (if specified by frame_indices)
        # 2. From object_terms matched against scene objects
        candidate_object_ids: List[int] = []

        # Extract target/anchor object IDs from hypothesis
        if bundle.hypothesis:
            target_cats = bundle.hypothesis.target_categories or []
            anchor_cats = bundle.hypothesis.anchor_categories or []

            # Search for matching objects in selector
            for obj in keyframe_selector.objects:
                obj_cat = getattr(obj, "category", "") or ""
                obj_tag = getattr(obj, "object_tag", "") or ""

                # Match against target categories
                for cat in target_cats:
                    if cat.lower() in obj_cat.lower() or cat.lower() in obj_tag.lower():
                        candidate_object_ids.append(obj.obj_id)
                        break

                # Match against anchor categories
                for cat in anchor_cats:
                    if cat.lower() in obj_cat.lower() or cat.lower() in obj_tag.lower():
                        candidate_object_ids.append(obj.obj_id)
                        break

        # Also search based on user-provided object_terms
        for term in object_terms:
            term_lower = term.lower()
            for obj in keyframe_selector.objects:
                obj_cat = getattr(obj, "category", "") or ""
                obj_tag = getattr(obj, "object_tag", "") or ""
                if term_lower in obj_cat.lower() or term_lower in obj_tag.lower():
                    if obj.obj_id not in candidate_object_ids:
                        candidate_object_ids.append(obj.obj_id)

        if not candidate_object_ids:
            return Stage2ToolResult(
                response_text="No matching objects found for additional view retrieval.",
            )

        # Get existing view IDs to exclude
        existing_view_ids = set()
        for keyframe in bundle.keyframes:
            if keyframe.view_id is not None:
                existing_view_ids.add(keyframe.view_id)

        # Get joint coverage views
        all_views = keyframe_selector.get_joint_coverage_views(
            candidate_object_ids,
            max_views=len(existing_view_ids) + max_additional_views,
        )

        # Filter to only new views
        new_view_ids = [v for v in all_views if v not in existing_view_ids]
        new_view_ids = new_view_ids[:max_additional_views]

        if not new_view_ids:
            return Stage2ToolResult(
                response_text="No additional views available. Current keyframes already provide best coverage.",
            )

        # Resolve view IDs to paths
        new_keyframes: List[KeyframeEvidence] = list(bundle.keyframes)
        added_paths: List[str] = []

        for view_id in new_view_ids:
            frame_id = keyframe_selector.map_view_to_frame(view_id)
            path, resolved_view_id = keyframe_selector._resolve_keyframe_path(view_id)

            if path is not None and path.exists():
                new_keyframe = KeyframeEvidence(
                    keyframe_idx=len(new_keyframes),
                    image_path=str(path),
                    view_id=resolved_view_id,
                    frame_id=frame_id,
                    note=f"Additional view from request: {request_text[:30]}",
                )
                new_keyframes.append(new_keyframe)
                added_paths.append(str(path))
                logger.info(
                    "[Stage1Callback] Added keyframe: view_id={}, frame_id={}, path={}",
                    resolved_view_id,
                    frame_id,
                    path.name,
                )
            else:
                logger.warning(
                    "[Stage1Callback] Could not resolve path for view_id={}",
                    view_id,
                )

        if not added_paths:
            return Stage2ToolResult(
                response_text="Could not resolve any additional keyframe paths.",
            )

        # Create updated bundle
        updated_bundle = bundle.model_copy(
            update={"keyframes": new_keyframes},
            deep=True,
        )

        response = (
            f"Added {len(added_paths)} new keyframe(s) for objects: "
            f"{candidate_object_ids[:5]}. "
            f"New view IDs: {new_view_ids}. "
            f"Total keyframes now: {len(new_keyframes)}."
        )

        return Stage2ToolResult(
            response_text=response,
            updated_bundle=updated_bundle,
        )

    return callback


def create_crop_callback(
    keyframe_selector: Any,
    scene_id: str = "",
    crop_padding: float = 0.1,
) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
    """Create a callback that generates object-centric crops from keyframes.

    Future implementation - currently returns a stub response.

    Args:
        keyframe_selector: KeyframeSelector instance
        scene_id: Scene identifier
        crop_padding: Padding ratio around bounding boxes

    Returns:
        A callback compatible with Stage2DeepResearchAgent.crop_callback
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: Dict[str, Any],
    ) -> Stage2ToolResult:
        # TODO: Implement actual crop generation
        # 1. Get object bounding boxes from visibility index
        # 2. Crop regions from keyframe images
        # 3. Save crops and add as additional keyframes
        return Stage2ToolResult(
            response_text="Crop generation not yet implemented. Please use existing keyframes.",
        )

    return callback


def create_hypothesis_callback(
    keyframe_selector: Any,
    scene_id: str = "",
) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
    """Create a callback that re-runs Stage 1 with alternative hypothesis.

    Future implementation - currently returns a stub response.

    Args:
        keyframe_selector: KeyframeSelector instance
        scene_id: Scene identifier

    Returns:
        A callback compatible with Stage2DeepResearchAgent.hypothesis_callback
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: Dict[str, Any],
    ) -> Stage2ToolResult:
        # TODO: Implement hypothesis switching
        # 1. Parse preferred_kind from request
        # 2. Re-execute Stage 1 with different hypothesis selection
        # 3. Build new evidence bundle
        return Stage2ToolResult(
            response_text="Hypothesis switching not yet implemented. Please work with current evidence.",
        )

    return callback


class Stage1BackendCallbacks:
    """Convenience class to hold all Stage 1 callbacks together."""

    def __init__(
        self,
        keyframe_selector: Any,
        scene_id: str = "",
        max_additional_views: int = 3,
    ) -> None:
        self.keyframe_selector = keyframe_selector
        self.scene_id = scene_id

        self.more_views = create_more_views_callback(
            keyframe_selector,
            scene_id=scene_id,
            max_additional_views=max_additional_views,
        )
        self.crops = create_crop_callback(
            keyframe_selector,
            scene_id=scene_id,
        )
        self.hypothesis = create_hypothesis_callback(
            keyframe_selector,
            scene_id=scene_id,
        )
