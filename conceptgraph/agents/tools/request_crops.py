"""Backend implementation for the request_crops Stage-2 tool.

This module provides the real crop extraction capability for the Stage-2 VLM agent.
It supports:
1. Cropping objects from keyframes using 2D bounding boxes
2. Multiple crop requests in a single call
3. Returning cropped images as new KeyframeEvidence entries in the bundle

Design rationale (Academic alignment):
- Supports "adaptive evidence acquisition" by letting the agent request focused
  object crops when full-frame evidence is insufficient
- Enables "symbolic-to-visual repair" by cropping specific objects for detailed
  inspection when Stage-1 scene graph hypotheses need verification
- Works across all task types (QA, grounding, nav, manipulation)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from ..models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2ToolResult,
)


@dataclass
class BBox2D:
    """2D bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_valid(self, image_size: Tuple[int, int]) -> bool:
        """Check if bbox is valid within image bounds."""
        w, h = image_size
        return 0 <= self.x1 < self.x2 <= w and 0 <= self.y1 < self.y2 <= h

    def clamp(self, image_size: Tuple[int, int]) -> "BBox2D":
        """Clamp bbox to image bounds."""
        w, h = image_size
        return BBox2D(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(1, min(self.x2, w)),
            y2=max(1, min(self.y2, h)),
        )

    def with_padding(
        self,
        padding: float,
        image_size: Tuple[int, int],
    ) -> "BBox2D":
        """Return bbox expanded by padding ratio, clamped to image bounds."""
        pad_x = int(self.width * padding)
        pad_y = int(self.height * padding)
        return BBox2D(
            x1=self.x1 - pad_x,
            y1=self.y1 - pad_y,
            x2=self.x2 + pad_x,
            y2=self.y2 + pad_y,
        ).clamp(image_size)

    @classmethod
    def from_xyxy(cls, coords: Tuple[int, int, int, int]) -> "BBox2D":
        """Create from (x1, y1, x2, y2) tuple."""
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class CropRequest:
    """A single crop request for one frame-bbox pair."""

    frame_idx: int  # Index in bundle.keyframes
    bbox: Optional[BBox2D] = None  # If None, use object_term to find bbox
    object_term: Optional[str] = None  # Object category to find
    padding: float = 0.15  # Padding ratio around bbox
    min_size: int = 32  # Minimum crop dimension
    note: str = ""  # Human-readable note for the crop


@dataclass
class CropResult:
    """Result of a single crop operation."""

    success: bool
    crop_path: Optional[str] = None
    original_frame_idx: int = -1
    bbox: Optional[BBox2D] = None
    width: int = 0
    height: int = 0
    error: str = ""
    note: str = ""


@dataclass
class CropBackendConfig:
    """Configuration for the crop backend."""

    output_dir: Optional[str] = None  # Directory for saving crops; None = temp dir
    padding: float = 0.15  # Default padding ratio
    min_size: int = 32  # Minimum crop size
    max_crops: int = 8  # Maximum crops per request
    jpeg_quality: int = 90  # JPEG quality for saved crops


class CropBackend:
    """Backend for extracting object crops from keyframes.

    This backend implements the real crop extraction capability:
    1. Reads keyframe images from paths in the evidence bundle
    2. Crops regions based on provided bounding boxes or object terms
    3. Saves crops to disk and returns updated evidence bundle

    Usage:
        backend = CropBackend()
        callback = backend.create_callback()
        agent = Stage2DeepResearchAgent(crop_callback=callback)
    """

    def __init__(
        self,
        config: Optional[CropBackendConfig] = None,
        bbox_resolver: Optional[Callable[[str, int, str], Optional[BBox2D]]] = None,
    ) -> None:
        """Initialize crop backend.

        Args:
            config: Backend configuration
            bbox_resolver: Optional callback to resolve object_term to bbox.
                          Signature: (scene_id, frame_idx, object_term) -> Optional[BBox2D]
        """
        self.config = config or CropBackendConfig()
        self.bbox_resolver = bbox_resolver
        self._output_dir: Optional[Path] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def _get_output_dir(self) -> Path:
        """Get or create output directory for crops."""
        if self.config.output_dir:
            out = Path(self.config.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out

        # Use temp directory
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="stage2_crops_")
        return Path(self._temp_dir.name)

    def _generate_crop_filename(
        self,
        image_path: str,
        bbox: BBox2D,
    ) -> str:
        """Generate a unique filename for a crop."""
        # Hash of original path + bbox to ensure uniqueness
        hash_input = f"{image_path}_{bbox.to_tuple()}"
        hash_str = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        stem = Path(image_path).stem
        return f"crop_{stem}_{hash_str}.jpg"

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path as RGB array."""
        if not Path(image_path).exists():
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save_crop(self, crop: np.ndarray, output_path: Path) -> bool:
        """Save crop image to disk."""
        try:
            # Convert RGB to BGR for OpenCV
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            cv2.imwrite(str(output_path), crop_bgr, params)
            return True
        except Exception as e:
            logger.error(f"Failed to save crop: {e}")
            return False

    def crop_from_bbox(
        self,
        image: np.ndarray,
        bbox: BBox2D,
        padding: float = 0.15,
        min_size: int = 32,
    ) -> Optional[np.ndarray]:
        """Crop region from image with padding.

        Args:
            image: RGB image (H, W, 3)
            bbox: 2D bounding box
            padding: Padding ratio around bbox
            min_size: Minimum crop dimension

        Returns:
            Cropped image region or None if invalid
        """
        h, w = image.shape[:2]

        # Apply padding and clamp
        padded_bbox = bbox.with_padding(padding, (w, h))

        # Check minimum size
        if padded_bbox.width < min_size or padded_bbox.height < min_size:
            return None

        # Extract crop
        return image[
            padded_bbox.y1 : padded_bbox.y2, padded_bbox.x1 : padded_bbox.x2
        ].copy()

    def process_crop_request(
        self,
        request: CropRequest,
        bundle: Stage2EvidenceBundle,
    ) -> CropResult:
        """Process a single crop request.

        Args:
            request: The crop request to process
            bundle: Current evidence bundle

        Returns:
            CropResult with success/failure status
        """
        # Validate frame index
        if request.frame_idx < 0 or request.frame_idx >= len(bundle.keyframes):
            return CropResult(
                success=False,
                original_frame_idx=request.frame_idx,
                error=f"Invalid frame index: {request.frame_idx}",
            )

        keyframe = bundle.keyframes[request.frame_idx]
        image_path = keyframe.image_path

        # Load image
        image = self._load_image(image_path)
        if image is None:
            return CropResult(
                success=False,
                original_frame_idx=request.frame_idx,
                error=f"Failed to load image: {image_path}",
            )

        h, w = image.shape[:2]

        # Resolve bbox
        bbox = request.bbox
        if bbox is None and request.object_term:
            # Try to resolve via bbox_resolver callback
            if self.bbox_resolver is not None:
                bbox = self.bbox_resolver(
                    bundle.scene_id,
                    request.frame_idx,
                    request.object_term,
                )

            if bbox is None:
                return CropResult(
                    success=False,
                    original_frame_idx=request.frame_idx,
                    error=f"Could not resolve bbox for object: {request.object_term}",
                )

        if bbox is None:
            return CropResult(
                success=False,
                original_frame_idx=request.frame_idx,
                error="No bbox provided and no object_term specified",
            )

        # Extract crop
        padding = request.padding if request.padding > 0 else self.config.padding
        min_size = request.min_size if request.min_size > 0 else self.config.min_size

        crop = self.crop_from_bbox(image, bbox, padding, min_size)
        if crop is None:
            return CropResult(
                success=False,
                original_frame_idx=request.frame_idx,
                bbox=bbox,
                error=f"Crop too small (min_size={min_size})",
            )

        # Save crop
        output_dir = self._get_output_dir()
        filename = self._generate_crop_filename(image_path, bbox)
        output_path = output_dir / filename

        if not self._save_crop(crop, output_path):
            return CropResult(
                success=False,
                original_frame_idx=request.frame_idx,
                bbox=bbox,
                error="Failed to save crop image",
            )

        crop_h, crop_w = crop.shape[:2]
        return CropResult(
            success=True,
            crop_path=str(output_path),
            original_frame_idx=request.frame_idx,
            bbox=bbox,
            width=crop_w,
            height=crop_h,
            note=request.note or f"Crop from frame {request.frame_idx}",
        )

    def process_requests(
        self,
        requests: List[CropRequest],
        bundle: Stage2EvidenceBundle,
    ) -> Tuple[List[CropResult], Stage2EvidenceBundle]:
        """Process multiple crop requests and return updated bundle.

        Args:
            requests: List of crop requests
            bundle: Current evidence bundle

        Returns:
            Tuple of (crop_results, updated_bundle)
        """
        # Limit number of crops
        actual_requests = requests[: self.config.max_crops]
        if len(requests) > self.config.max_crops:
            logger.warning(
                f"Limiting crop requests from {len(requests)} to {self.config.max_crops}"
            )

        results: List[CropResult] = []
        new_keyframes: List[KeyframeEvidence] = []

        for request in actual_requests:
            result = self.process_crop_request(request, bundle)
            results.append(result)

            if result.success and result.crop_path:
                # Create new KeyframeEvidence for the crop
                base_keyframe = bundle.keyframes[result.original_frame_idx]
                new_idx = len(bundle.keyframes) + len(new_keyframes)

                new_keyframes.append(
                    KeyframeEvidence(
                        keyframe_idx=new_idx,
                        image_path=result.crop_path,
                        view_id=base_keyframe.view_id,
                        frame_id=base_keyframe.frame_id,
                        score=base_keyframe.score,
                        note=f"crop:{result.note}",
                    )
                )

        # Create updated bundle with new crops
        updated_bundle = bundle.model_copy(deep=True)
        updated_bundle.keyframes.extend(new_keyframes)

        return results, updated_bundle

    def handle_tool_request(
        self,
        bundle: Stage2EvidenceBundle,
        request_dict: Dict[str, Any],
    ) -> Stage2ToolResult:
        """Handle a tool request from the Stage-2 agent.

        This is the main entry point called by the agent's request_crops tool.

        Args:
            bundle: Current evidence bundle
            request_dict: Raw request dictionary from the agent tool call

        Returns:
            Stage2ToolResult with response text and optional updated bundle
        """
        request_text = request_dict.get("request_text", "")
        frame_indices = request_dict.get("frame_indices", [])
        object_terms = request_dict.get("object_terms", [])

        # Parse requests from agent input
        crop_requests = self._parse_agent_request(
            request_text=request_text,
            frame_indices=frame_indices,
            object_terms=object_terms,
            bundle=bundle,
        )

        if not crop_requests:
            return Stage2ToolResult(
                response_text="No valid crop requests could be parsed from input.",
            )

        # Process crops
        results, updated_bundle = self.process_requests(crop_requests, bundle)

        # Format response
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        response_lines = [
            f"Processed {len(results)} crop requests: "
            f"{len(successful)} successful, {len(failed)} failed."
        ]

        if successful:
            response_lines.append("\nSuccessful crops:")
            for r in successful:
                response_lines.append(
                    f"  - Frame {r.original_frame_idx}: {r.width}x{r.height}px "
                    f"from bbox {r.bbox.to_tuple() if r.bbox else 'N/A'}"
                )

        if failed:
            response_lines.append("\nFailed crops:")
            for r in failed:
                response_lines.append(f"  - Frame {r.original_frame_idx}: {r.error}")

        response_lines.append(
            f"\nTotal keyframes in updated bundle: {len(updated_bundle.keyframes)}"
        )

        return Stage2ToolResult(
            response_text="\n".join(response_lines),
            updated_bundle=updated_bundle if successful else None,
        )

    def _parse_agent_request(
        self,
        request_text: str,
        frame_indices: List[int],
        object_terms: List[str],
        bundle: Stage2EvidenceBundle,
    ) -> List[CropRequest]:
        """Parse agent request into structured CropRequests.

        The agent might request crops in several ways:
        1. Specific frame_indices with object_terms -> crop those objects in those frames
        2. Only frame_indices -> need bboxes from elsewhere or use whole frame
        3. Only object_terms -> find objects across all keyframes
        """
        requests: List[CropRequest] = []

        # Normalize frame indices
        if not frame_indices:
            # Default to all keyframes if none specified
            frame_indices = list(range(len(bundle.keyframes)))
        else:
            # Filter to valid indices
            frame_indices = [i for i in frame_indices if 0 <= i < len(bundle.keyframes)]

        # Case 1: Frame indices + object terms -> match pairs
        if frame_indices and object_terms:
            for frame_idx in frame_indices:
                for obj_term in object_terms:
                    requests.append(
                        CropRequest(
                            frame_idx=frame_idx,
                            object_term=obj_term,
                            note=f"{obj_term} in frame {frame_idx}",
                        )
                    )

        # Case 2: Only frame indices + no object terms
        # In this case, we need more info to create bboxes
        # For now, we'll rely on the bbox_resolver if available
        elif frame_indices and not object_terms:
            # Try to extract object terms from request_text
            # Simple heuristic: look for quoted terms or common object words
            parsed_terms = self._extract_object_terms_from_text(request_text)
            if parsed_terms:
                for frame_idx in frame_indices:
                    for obj_term in parsed_terms:
                        requests.append(
                            CropRequest(
                                frame_idx=frame_idx,
                                object_term=obj_term,
                                note=f"{obj_term} in frame {frame_idx}",
                            )
                        )
            else:
                # Can't create crops without bbox info
                logger.warning(
                    "Crop request with frame_indices but no object_terms or bboxes"
                )

        # Case 3: Only object terms -> search across all frames
        elif object_terms and not frame_indices:
            for frame_idx in range(len(bundle.keyframes)):
                for obj_term in object_terms:
                    requests.append(
                        CropRequest(
                            frame_idx=frame_idx,
                            object_term=obj_term,
                            note=f"{obj_term} in frame {frame_idx}",
                        )
                    )

        return requests

    def _extract_object_terms_from_text(self, text: str) -> List[str]:
        """Extract potential object terms from free-form request text.

        Simple heuristic extraction - looks for quoted strings or common patterns.
        """
        import re

        terms = []

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        terms.extend(quoted)

        quoted_single = re.findall(r"'([^']+)'", text)
        terms.extend(quoted_single)

        # Deduplicate while preserving order
        seen = set()
        result = []
        for term in terms:
            term = term.strip().lower()
            if term and term not in seen:
                seen.add(term)
                result.append(term)

        return result

    def create_callback(
        self,
    ) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
        """Create a callback function for the Stage-2 agent.

        Returns:
            Callback compatible with Stage2DeepResearchAgent.crop_callback
        """

        def callback(
            bundle: Stage2EvidenceBundle,
            request: Dict[str, Any],
        ) -> Stage2ToolResult:
            return self.handle_tool_request(bundle, request)

        return callback


def create_crop_callback(
    config: Optional[CropBackendConfig] = None,
    bbox_resolver: Optional[Callable[[str, int, str], Optional[BBox2D]]] = None,
) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
    """Convenience function to create a crop callback.

    Args:
        config: Optional backend configuration
        bbox_resolver: Optional callback to resolve object_term to bbox

    Returns:
        Callback function for Stage2DeepResearchAgent.crop_callback

    Example:
        callback = create_crop_callback()
        agent = Stage2DeepResearchAgent(crop_callback=callback)
    """
    backend = CropBackend(config=config, bbox_resolver=bbox_resolver)
    return backend.create_callback()
