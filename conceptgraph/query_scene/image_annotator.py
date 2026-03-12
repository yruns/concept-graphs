"""Image annotation utilities for Query Case Generator.

Draws red bounding boxes with anonymous markers (A, B, C...) on images
to highlight target objects for Gemini query generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Anonymous markers: A, B, C... (no obj_id or category leakage)
MARKERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def annotate_image_with_targets(
    image_path: Path,
    objects: List,  # List[SceneObject]
    view_id: int,
    box_color: str = "red",
    box_width: int = 4,
    min_bbox_area: int = 500,
    min_conf: float = 0.3,
    relative_min_area: float = 0.001,  # Min area as fraction of image area
) -> Tuple[Image.Image, Dict[str, int]]:
    """Annotate image with red bounding boxes using anonymous markers.

    Args:
        image_path: Path to the original image
        objects: Target objects to annotate
        view_id: View ID for bbox lookup (not frame_idx)
        box_color: Color for bounding box outlines
        box_width: Width of bounding box lines
        min_bbox_area: Minimum absolute bbox area in pixels
        min_conf: Minimum detection confidence
        relative_min_area: Minimum area as fraction of image area

    Returns:
        Tuple of (annotated PIL Image, marker-to-obj_id mapping)

    Raises:
        ValueError: If any target object has no valid bbox in the view
        FileNotFoundError: If image_path does not exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use context manager for proper resource cleanup
    with Image.open(image_path) as img:
        img = img.copy()  # Copy to use after context exits
        img_width, img_height = img.size
        img_area = img_width * img_height

        # Resolution-normalized minimum area threshold
        effective_min_area = max(min_bbox_area, int(img_area * relative_min_area))

        draw = ImageDraw.Draw(img)
        marker_to_obj_id = {}

        for i, obj in enumerate(objects):
            marker = MARKERS[i % len(MARKERS)]

            # Get best valid bbox for this object in the view
            bbox = get_best_bbox_in_view(
                obj,
                view_id,
                min_area=effective_min_area,
                min_conf=min_conf,
                img_bounds=(img_width, img_height),
            )
            if bbox is None:
                raise ValueError(
                    f"Object {obj.obj_id} has no valid bbox in view {view_id}"
                )

            x1, y1, x2, y2 = bbox

            # Draw red bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            # Draw anonymous marker label
            _draw_marker_label(
                draw, marker, x1, y1, img_width, img_height, box_color
            )

            marker_to_obj_id[marker] = obj.obj_id

    return img, marker_to_obj_id


def _draw_marker_label(
    draw: ImageDraw.ImageDraw,
    marker: str,
    x1: float,
    y1: float,
    img_width: int,
    img_height: int,
    box_color: str,
    label_height: int = 35,
    label_width: int = 35,
) -> None:
    """Draw marker label above or inside the bounding box.

    Label placement strategy:
    - Prefer above bbox if space available
    - Fall back to inside bbox top if at image edge
    - Clamp to image boundaries
    """
    # Determine vertical position
    if y1 >= label_height:
        label_y_start = y1 - label_height
    else:
        label_y_start = y1 + 2  # Inside bbox

    # Clamp to image boundaries
    label_x_start = max(0, min(x1, img_width - label_width))
    label_y_start = max(0, min(label_y_start, img_height - label_height))

    label_bg = [
        label_x_start,
        label_y_start,
        label_x_start + label_width,
        label_y_start + label_height,
    ]

    draw.rectangle(label_bg, fill="white", outline=box_color, width=2)
    draw.text((label_x_start + 10, label_y_start + 5), marker, fill=box_color)


def get_best_bbox_in_view(
    obj,  # SceneObject
    view_id: int,
    min_area: int = 500,
    min_conf: float = 0.0,  # Default to 0.0 - conf may not be available
    img_bounds: Optional[Tuple[int, int]] = None,
) -> Optional[List[float]]:
    """Get the best valid 2D bbox for an object in a specific view.

    Selects the highest-confidence detection that passes geometric validation.

    Args:
        obj: Scene object with image_idx, xyxy, and optional conf attributes
        view_id: View ID to look up
        min_area: Minimum bbox area threshold
        min_conf: Minimum confidence threshold (0.0 if conf not available)
        img_bounds: Optional (width, height) to clip bbox to image bounds

    Returns:
        Valid bbox [x1, y1, x2, y2] or None if no valid detection
    """
    # Check if object is detected in this view
    if not hasattr(obj, "image_idx") or view_id not in obj.image_idx:
        return None

    # Find all detection indices for this view
    indices = [i for i, v in enumerate(obj.image_idx) if v == view_id]

    if not indices:
        return None

    # Get conf array (may be None or empty)
    conf_arr = getattr(obj, "conf", None)
    has_conf = conf_arr is not None and len(conf_arr) > 0

    # Filter and validate detections
    valid_detections = []
    for idx in indices:
        if idx >= len(getattr(obj, "xyxy", [])):
            continue

        bbox = obj.xyxy[idx]

        # Get confidence (default to 1.0 if not available)
        if has_conf and idx < len(conf_arr):
            conf = float(conf_arr[idx]) if conf_arr[idx] is not None else 1.0
        else:
            conf = 1.0  # Default confidence when not available

        # Validate bbox format
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        # Geometric validity check
        if x2 <= x1 or y2 <= y1:
            continue

        # Clip to image bounds if provided
        if img_bounds:
            img_w, img_h = img_bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # Re-check validity after clipping
            if x2 <= x1 or y2 <= y1:
                continue

        # Area check
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        # Confidence check
        if conf < min_conf:
            continue

        valid_detections.append((idx, conf, [x1, y1, x2, y2]))

    if not valid_detections:
        return None

    # Select highest confidence detection
    best = max(valid_detections, key=lambda x: x[1])
    return best[2]


def build_view_score_dict(
    obj_id: int,
    object_to_views: Dict[int, List[Tuple[int, float]]],
) -> Dict[int, float]:
    """Convert List[(view_id, score)] format to {view_id: score} dict.

    Handles the actual visibility_index format used in the codebase.
    """
    views_list = object_to_views.get(obj_id, [])
    return {view_id: score for view_id, score in views_list}


def find_best_view_for_objects(
    objects: List,  # List[SceneObject]
    object_to_views: Dict[int, List[Tuple[int, float]]],
    min_visibility_score: float = 0.1,  # Lowered from 0.3 due to actual score ranges
) -> Optional[int]:
    """Find the best view where all target objects are visible.

    Strategy:
    1. Get intersection of high-score views for all targets
    2. Rank by average visibility score
    3. Return the best view ID

    Args:
        objects: List of target objects
        object_to_views: Visibility index mapping obj_id -> [(view_id, score)]
        min_visibility_score: Minimum visibility score threshold

    Returns:
        Best view ID or None if no common view found
    """
    if not objects:
        return None

    # Build score dicts and high-score view sets
    view_score_dicts = []
    view_sets = []
    for obj in objects:
        scores = build_view_score_dict(obj.obj_id, object_to_views)
        view_score_dicts.append(scores)
        high_score_views = {v for v, s in scores.items() if s >= min_visibility_score}
        view_sets.append(high_score_views)

    if not view_sets:
        return None

    # Find intersection of all view sets
    common_views = set.intersection(*view_sets)

    if not common_views:
        return None

    # Rank by average score
    def avg_score(view_id: int) -> float:
        scores = [d.get(view_id, 0) for d in view_score_dicts]
        return sum(scores) / len(scores)

    return max(common_views, key=avg_score)
