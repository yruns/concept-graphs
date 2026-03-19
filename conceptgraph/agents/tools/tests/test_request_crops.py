"""Unit tests for the request_crops tool backend.

Tests cover:
1. BBox2D data class operations (validation, clamping, padding)
2. CropBackend.crop_from_bbox() functionality
3. CropBackend.process_crop_request() with various inputs
4. CropBackend.handle_tool_request() integration
5. Callback creation and usage pattern
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from conceptgraph.agents.models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2ToolResult,
)
from conceptgraph.agents.tools.request_crops import (
    BBox2D,
    CropBackend,
    CropBackendConfig,
    CropRequest,
    CropResult,
    create_crop_callback,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for crop outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_image_path(temp_output_dir):
    """Create a test image file and return its path."""
    img_path = Path(temp_output_dir) / "test_frame.jpg"
    # Create a simple 200x200 RGB image with distinct colors
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [255, 0, 0]  # Red square in center
    img[0:50, 0:50] = [0, 255, 0]  # Green corner
    # Save as BGR (OpenCV format)
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(img_path)


@pytest.fixture
def sample_bundle(test_image_path):
    """Create a sample evidence bundle with one keyframe."""
    return Stage2EvidenceBundle(
        scene_id="test_scene",
        keyframes=[
            KeyframeEvidence(
                keyframe_idx=0,
                image_path=test_image_path,
                view_id=1,
                frame_id=100,
                score=0.9,
                note="test keyframe",
            )
        ],
        object_context={"summary": "test objects"},
    )


@pytest.fixture
def backend(temp_output_dir):
    """Create a CropBackend with test configuration."""
    config = CropBackendConfig(output_dir=temp_output_dir)
    return CropBackend(config=config)


# ============================================================================
# BBox2D Tests
# ============================================================================


class TestBBox2D:
    """Tests for BBox2D data class."""

    def test_basic_properties(self):
        """Test width, height, area calculations."""
        bbox = BBox2D(x1=10, y1=20, x2=110, y2=70)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000

    def test_is_valid_positive(self):
        """Test valid bbox within image bounds."""
        bbox = BBox2D(x1=10, y1=20, x2=110, y2=70)
        assert bbox.is_valid((200, 100)) is True

    def test_is_valid_negative_out_of_bounds(self):
        """Test bbox exceeding image bounds."""
        bbox = BBox2D(x1=10, y1=20, x2=300, y2=70)
        assert bbox.is_valid((200, 100)) is False

    def test_is_valid_negative_inverted(self):
        """Test inverted bbox (x1 > x2)."""
        bbox = BBox2D(x1=100, y1=20, x2=10, y2=70)
        assert bbox.is_valid((200, 100)) is False

    def test_clamp(self):
        """Test clamping bbox to image bounds."""
        bbox = BBox2D(x1=-10, y1=-5, x2=250, y2=120)
        clamped = bbox.clamp((200, 100))
        assert clamped.x1 == 0
        assert clamped.y1 == 0
        assert clamped.x2 == 200
        assert clamped.y2 == 100

    def test_with_padding(self):
        """Test padding expansion and clamping."""
        bbox = BBox2D(x1=50, y1=50, x2=150, y2=100)
        # padding=0.1 means 10% of each dimension
        # width=100, height=50 -> pad_x=10, pad_y=5
        padded = bbox.with_padding(0.1, (200, 200))
        assert padded.x1 == 40
        assert padded.y1 == 45
        assert padded.x2 == 160
        assert padded.y2 == 105

    def test_with_padding_clamps_to_bounds(self):
        """Test padding respects image bounds."""
        bbox = BBox2D(x1=0, y1=0, x2=50, y2=50)
        padded = bbox.with_padding(0.5, (100, 100))
        assert padded.x1 == 0  # Can't go negative
        assert padded.y1 == 0

    def test_from_xyxy(self):
        """Test creation from tuple."""
        bbox = BBox2D.from_xyxy((10, 20, 30, 40))
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 30
        assert bbox.y2 == 40

    def test_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BBox2D(x1=10, y1=20, x2=30, y2=40)
        assert bbox.to_tuple() == (10, 20, 30, 40)


# ============================================================================
# CropBackend.crop_from_bbox Tests
# ============================================================================


class TestCropFromBbox:
    """Tests for CropBackend.crop_from_bbox()."""

    def test_basic_crop(self, backend):
        """Test basic crop extraction."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150] = [255, 0, 0]  # Red square

        bbox = BBox2D(x1=50, y1=50, x2=150, y2=150)
        crop = backend.crop_from_bbox(image, bbox, padding=0.0, min_size=10)

        assert crop is not None
        assert crop.shape == (100, 100, 3)
        assert np.all(crop == [255, 0, 0])

    def test_crop_with_padding(self, backend):
        """Test crop includes padding."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150] = [255, 0, 0]

        bbox = BBox2D(x1=60, y1=60, x2=140, y2=140)  # 80x80
        crop = backend.crop_from_bbox(image, bbox, padding=0.25, min_size=10)

        # padding=0.25 -> expand by 20px each side -> 120x120
        assert crop is not None
        assert crop.shape[0] == 120
        assert crop.shape[1] == 120

    def test_crop_min_size_rejection(self, backend):
        """Test crop is rejected if too small."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = BBox2D(x1=50, y1=50, x2=60, y2=60)  # 10x10

        crop = backend.crop_from_bbox(image, bbox, padding=0.0, min_size=32)
        assert crop is None

    def test_crop_clamped_to_bounds(self, backend):
        """Test crop bbox clamped to image bounds."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BBox2D(x1=80, y1=80, x2=120, y2=120)

        crop = backend.crop_from_bbox(image, bbox, padding=0.0, min_size=10)
        # Clamped to (80,80) - (100,100) = 20x20
        assert crop is not None
        assert crop.shape == (20, 20, 3)


# ============================================================================
# CropBackend.process_crop_request Tests
# ============================================================================


class TestProcessCropRequest:
    """Tests for CropBackend.process_crop_request()."""

    def test_successful_crop(self, backend, sample_bundle):
        """Test successful crop with valid bbox."""
        request = CropRequest(
            frame_idx=0,
            bbox=BBox2D(x1=50, y1=50, x2=150, y2=150),
            note="test crop",
        )
        result = backend.process_crop_request(request, sample_bundle)

        assert result.success is True
        assert result.crop_path is not None
        assert Path(result.crop_path).exists()
        assert result.width > 0
        assert result.height > 0
        assert result.original_frame_idx == 0

    def test_invalid_frame_index(self, backend, sample_bundle):
        """Test error on invalid frame index."""
        request = CropRequest(
            frame_idx=99,
            bbox=BBox2D(x1=50, y1=50, x2=150, y2=150),
        )
        result = backend.process_crop_request(request, sample_bundle)

        assert result.success is False
        assert "Invalid frame index" in result.error

    def test_no_bbox_no_object_term(self, backend, sample_bundle):
        """Test error when neither bbox nor object_term provided."""
        request = CropRequest(frame_idx=0)  # No bbox, no object_term
        result = backend.process_crop_request(request, sample_bundle)

        assert result.success is False
        assert "No bbox provided" in result.error

    def test_object_term_without_resolver(self, backend, sample_bundle):
        """Test error when object_term used but no resolver configured."""
        request = CropRequest(frame_idx=0, object_term="chair")
        result = backend.process_crop_request(request, sample_bundle)

        assert result.success is False
        assert "Could not resolve bbox" in result.error

    def test_object_term_with_resolver(self, sample_bundle, temp_output_dir):
        """Test successful crop when bbox_resolver provides bbox."""

        def mock_resolver(
            scene_id: str, frame_idx: int, obj_term: str
        ) -> Optional[BBox2D]:
            if obj_term == "red_square":
                return BBox2D(x1=50, y1=50, x2=150, y2=150)
            return None

        config = CropBackendConfig(output_dir=temp_output_dir)
        backend = CropBackend(config=config, bbox_resolver=mock_resolver)

        request = CropRequest(frame_idx=0, object_term="red_square")
        result = backend.process_crop_request(request, sample_bundle)

        assert result.success is True
        assert result.crop_path is not None


# ============================================================================
# CropBackend.process_requests Tests
# ============================================================================


class TestProcessRequests:
    """Tests for CropBackend.process_requests()."""

    def test_multiple_crops(self, backend, sample_bundle):
        """Test processing multiple crop requests."""
        requests = [
            CropRequest(
                frame_idx=0,
                bbox=BBox2D(x1=50, y1=50, x2=100, y2=100),
                note="crop1",
            ),
            CropRequest(
                frame_idx=0,
                bbox=BBox2D(x1=100, y1=100, x2=150, y2=150),
                note="crop2",
            ),
        ]

        results, updated_bundle = backend.process_requests(requests, sample_bundle)

        assert len(results) == 2
        assert all(r.success for r in results)
        # Original 1 keyframe + 2 new crops
        assert len(updated_bundle.keyframes) == 3

    def test_max_crops_limit(self, sample_bundle, temp_output_dir):
        """Test that max_crops config is respected."""
        config = CropBackendConfig(output_dir=temp_output_dir, max_crops=2)
        backend = CropBackend(config=config)

        # Request 5 crops, only 2 should be processed
        requests = [
            CropRequest(
                frame_idx=0,
                bbox=BBox2D(x1=i * 30, y1=i * 30, x2=i * 30 + 40, y2=i * 30 + 40),
            )
            for i in range(5)
        ]

        results, _ = backend.process_requests(requests, sample_bundle)
        assert len(results) == 2

    def test_bundle_updated_with_crop_metadata(self, backend, sample_bundle):
        """Test that new crops get proper metadata in bundle."""
        request = CropRequest(
            frame_idx=0,
            bbox=BBox2D(x1=50, y1=50, x2=150, y2=150),
            note="object detail",
        )
        _, updated_bundle = backend.process_requests([request], sample_bundle)

        new_keyframe = updated_bundle.keyframes[-1]
        assert new_keyframe.keyframe_idx == 1
        assert "crop:" in new_keyframe.note
        # Inherits view_id and frame_id from original
        assert new_keyframe.view_id == sample_bundle.keyframes[0].view_id

    def test_mixed_success_failure(self, backend, sample_bundle):
        """Test bundle only updated for successful crops."""
        requests = [
            CropRequest(
                frame_idx=0,
                bbox=BBox2D(x1=50, y1=50, x2=150, y2=150),
            ),
            CropRequest(
                frame_idx=99,  # Invalid
                bbox=BBox2D(x1=50, y1=50, x2=150, y2=150),
            ),
        ]

        results, updated_bundle = backend.process_requests(requests, sample_bundle)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 1
        assert len(failed) == 1
        # Only 1 new crop added
        assert len(updated_bundle.keyframes) == 2


# ============================================================================
# CropBackend.handle_tool_request Tests
# ============================================================================


class TestHandleToolRequest:
    """Tests for CropBackend.handle_tool_request() integration."""

    def test_full_request_flow(self, backend, sample_bundle):
        """Test complete request flow from agent input to response."""
        request_dict = {
            "request_text": "Crop the object at specified bbox",
            "frame_indices": [0],
            "object_terms": [],
        }

        # We need to provide explicit bboxes for this to work without resolver
        # For this test, use the _parse_agent_request limitations
        # Actually, frame_indices without object_terms won't create requests
        # Let's test with object_terms

        # Reconfigure with a resolver
        def resolver(scene_id, frame_idx, obj_term):
            return BBox2D(x1=50, y1=50, x2=150, y2=150)

        backend.bbox_resolver = resolver
        request_dict["object_terms"] = ["test_object"]

        result = backend.handle_tool_request(sample_bundle, request_dict)

        assert isinstance(result, Stage2ToolResult)
        assert "Processed 1 crop requests" in result.response_text
        assert "1 successful" in result.response_text
        assert result.updated_bundle is not None

    def test_no_valid_requests(self, backend, sample_bundle):
        """Test response when no valid requests can be parsed."""
        request_dict = {
            "request_text": "",
            "frame_indices": [],
            "object_terms": [],
        }

        result = backend.handle_tool_request(sample_bundle, request_dict)

        assert result.updated_bundle is None
        assert "No valid crop requests" in result.response_text

    def test_response_format_on_failures(self, backend, sample_bundle):
        """Test response includes failure details."""
        request_dict = {
            "request_text": "",
            "frame_indices": [0, 99],  # One valid, one invalid
            "object_terms": ["nonexistent"],  # No resolver
        }

        result = backend.handle_tool_request(sample_bundle, request_dict)

        # Both will fail - valid frame but no bbox resolver
        assert "Failed crops" in result.response_text


# ============================================================================
# Callback Creation Tests
# ============================================================================


class TestCreateCallback:
    """Tests for callback creation and usage pattern."""

    def test_create_callback_function(self, temp_output_dir):
        """Test create_crop_callback returns callable."""
        callback = create_crop_callback(
            config=CropBackendConfig(output_dir=temp_output_dir)
        )
        assert callable(callback)

    def test_callback_signature(self, temp_output_dir, sample_bundle):
        """Test callback has correct signature for agent integration."""

        def resolver(scene_id, frame_idx, obj_term):
            return BBox2D(x1=50, y1=50, x2=150, y2=150)

        callback = create_crop_callback(
            config=CropBackendConfig(output_dir=temp_output_dir),
            bbox_resolver=resolver,
        )

        request = {
            "request_text": "Crop object",
            "frame_indices": [0],
            "object_terms": ["chair"],
        }

        result = callback(sample_bundle, request)

        assert isinstance(result, Stage2ToolResult)
        assert result.response_text

    def test_callback_returns_updated_bundle(self, temp_output_dir, sample_bundle):
        """Test callback returns updated bundle on success."""

        def resolver(scene_id, frame_idx, obj_term):
            return BBox2D(x1=50, y1=50, x2=150, y2=150)

        callback = create_crop_callback(
            config=CropBackendConfig(output_dir=temp_output_dir),
            bbox_resolver=resolver,
        )

        request = {
            "request_text": "",
            "frame_indices": [0],
            "object_terms": ["object"],
        }

        result = callback(sample_bundle, request)

        assert result.updated_bundle is not None
        assert len(result.updated_bundle.keyframes) == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_nonexistent_image_path(self, backend):
        """Test handling of missing image file."""
        bundle = Stage2EvidenceBundle(
            scene_id="test",
            keyframes=[
                KeyframeEvidence(
                    keyframe_idx=0,
                    image_path="/nonexistent/path/image.jpg",
                )
            ],
            object_context={},
        )

        request = CropRequest(
            frame_idx=0,
            bbox=BBox2D(x1=0, y1=0, x2=100, y2=100),
        )
        result = backend.process_crop_request(request, bundle)

        assert result.success is False
        assert "Failed to load image" in result.error

    def test_empty_keyframes(self, backend):
        """Test handling of bundle with no keyframes."""
        bundle = Stage2EvidenceBundle(
            scene_id="test",
            keyframes=[],
            object_context={},
        )

        request_dict = {
            "request_text": "",
            "frame_indices": [0],
            "object_terms": ["chair"],
        }

        result = backend.handle_tool_request(bundle, request_dict)
        # Should not crash, should report no valid requests
        assert "No valid crop requests" in result.response_text

    def test_extract_object_terms_from_text(self, backend):
        """Test object term extraction from request text."""
        text = "Please crop the \"red chair\" and the 'blue table'"
        terms = backend._extract_object_terms_from_text(text)

        assert "red chair" in terms
        assert "blue table" in terms

    def test_unique_crop_filenames(self, backend, sample_bundle):
        """Test that different bboxes produce different filenames."""
        bbox1 = BBox2D(x1=0, y1=0, x2=50, y2=50)
        bbox2 = BBox2D(x1=50, y1=50, x2=100, y2=100)

        filename1 = backend._generate_crop_filename("test.jpg", bbox1)
        filename2 = backend._generate_crop_filename("test.jpg", bbox2)

        assert filename1 != filename2
