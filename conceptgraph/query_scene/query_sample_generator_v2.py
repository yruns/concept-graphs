#!/usr/bin/env python3
"""
Query Sample Generator V2 - Advanced Multi-Modal Query Generation.

Generates diverse, realistic queries for 3D scene understanding using:
- Annotated frame images with bounding boxes
- Rich scene context (objects, spatial relations, affordances)
- Multiple task types: QA, navigation, manipulation, etc.
- Difficulty matrix with missing detection and synonym cases

Usage:
    generator = QuerySampleGeneratorV2.from_scene_path("/path/to/room0")
    samples = generator.generate_samples(n=50, frames_per_batch=3)
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ObjectInfo:
    """Complete object information for query generation."""
    obj_id: int
    object_tag: str
    class_name: str
    category: str
    summary: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]  # width, depth, height
    volume: float
    co_objects: List[str]
    typical_location: str
    primary_functions: List[str]
    best_views: List[int]
    num_detections: int
    bbox_2d_samples: Dict[int, Tuple[int, int, int, int]]  # view_id -> xyxy


@dataclass
class FrameInfo:
    """Frame information with visible objects."""
    frame_id: int
    image_path: Path
    visible_objects: List[int]  # object IDs visible in this frame
    detections: Dict[int, Tuple[int, int, int, int]]  # obj_id -> bbox xyxy


@dataclass
class SceneContextV2:
    """Complete scene context for advanced query generation."""
    scene_name: str
    objects: List[ObjectInfo]
    object_id_to_idx: Dict[int, int]
    categories: List[str]
    object_tags: List[str]
    frames: List[FrameInfo]
    tag_to_ids: Dict[str, List[int]]  # object_tag -> list of obj_ids
    total_views: int


@dataclass
class GeneratedQueryV2:
    """A generated query sample with rich metadata."""
    query: str
    query_type: str  # qa, navigation, manipulation, grounding, counting, comparison
    difficulty: str  # easy, medium, hard, expert
    complexity_type: str  # direct, spatial, nested, superlative, multi_anchor, etc.
    target_objects: List[str]
    anchor_objects: List[str]
    spatial_relations: List[str]
    special_case: str  # none, missing_detection, synonym, ambiguous, multi_instance
    rationale: str
    source_frames: List[int]  # frame IDs used to generate this query


# =============================================================================
# System Prompt Design
# =============================================================================

SYSTEM_PROMPT = """You are an expert query generator for 3D scene understanding and embodied AI tasks.

Your goal is to generate diverse, realistic queries that users might ask about objects in a 3D indoor scene. These queries should be:
1. **Grounded in reality** - Based on actual objects visible in the provided frames
2. **Diverse in task type** - Cover various 3D AI tasks (QA, navigation, manipulation, etc.)
3. **Varied in difficulty** - Range from simple to expert-level queries
4. **Natural language** - Use varied phrasings, synonyms, and colloquial expressions

## Query Types (you MUST cover all types):

### 1. Visual QA (qa)
Questions about object properties, relationships, or scene understanding.
Examples:
- "What color is the throw pillow on the gray sofa?"
- "How many chairs are around the dining table?"
- "Is there a lamp near the window?"

### 2. Navigation/Patrol Planning (navigation)
Instructions for robot navigation or path planning in the scene.
Examples:
- "Navigate to the coffee table, then turn left and go to the bookshelf"
- "Find a path from the door to the sofa that avoids the coffee table"
- "Patrol the room starting from the entrance, visiting all seating areas"

### 3. Object Manipulation (manipulation)
Instructions for picking up, moving, or interacting with objects.
Examples:
- "Pick up the book on the coffee table and place it on the shelf"
- "Move all the throw pillows from the sofa to the armchair"
- "Open the drawer of the nightstand next to the bed"

### 4. Visual Grounding (grounding)
Queries to identify specific objects using spatial and attribute constraints.
Examples:
- "The red cushion on the larger sofa nearest the window"
- "Find the lamp that is between the sofa and the bookshelf"
- "Locate the smallest pillow on the corner of the sectional sofa"

### 5. Counting & Comparison (counting)
Queries involving counting objects or comparing properties.
Examples:
- "How many throw pillows are there in total?"
- "Which sofa is larger, the one near the window or the one facing the TV?"
- "Count all the light sources in the living room"

### 6. Scene Reasoning (reasoning)
Complex queries requiring inference about scene layout or object functions.
Examples:
- "Where would be the best place to sit and read in this room?"
- "If I want to watch TV, which seating option gives the best view?"
- "What objects suggest this is a living room rather than a bedroom?"

## Difficulty Levels:

### Easy
- Single object reference: "the sofa", "a lamp", "the coffee table"
- Simple attribute: "the red pillow", "the large sofa"
- Direct category: "find a chair"

### Medium
- Single spatial relation: "pillow on the sofa", "lamp near the door"
- Simple superlative: "the largest sofa", "the nearest chair"
- Basic counting: "how many pillows"

### Hard
- Nested spatial relations: "pillow on the sofa near the coffee table"
- Multiple constraints: "the red pillow on the gray sofa"
- Ordinals: "the second largest chair", "the third lamp from the left"
- Between relations: "the table between the two sofas"

### Expert
- Multi-step navigation with constraints
- Manipulation sequences with spatial reasoning
- Queries requiring scene-level understanding
- Ambiguous references requiring disambiguation

## Special Cases (important for dataset diversity):

### Missing Detection (missing_detection)
Objects visible in frames but NOT in the scene graph. Generate queries for these to test robustness.
Look at the frames carefully - if you see objects like remote controls, magazines, coasters, etc. that aren't in the object list, create queries for them.

### Synonym/Near-Synonym (synonym)
Use alternative names: couch/sofa, cushion/pillow, lamp/light, ottoman/footrest, rug/carpet
Example: If scene has "sofa", query might say "the couch near the window"

### Ambiguous Reference (ambiguous)
Queries that could match multiple objects - tests disambiguation ability.
Example: "the pillow" when there are 5 pillows

### Multi-Instance (multi_instance)
Queries targeting all instances of a category.
Example: "all the throw pillows on the sofas"

## Output Format

Return a JSON array where each item has:
```json
{
  "query": "Navigate to the large gray sofa, then pick up the red pillow nearest the armrest",
  "query_type": "manipulation",
  "difficulty": "expert",
  "complexity_type": "multi_step",
  "target_objects": ["throw_pillow"],
  "anchor_objects": ["sofa", "armrest"],
  "spatial_relations": ["on", "nearest"],
  "special_case": "none",
  "rationale": "Valid because there is a gray sofa with multiple pillows, requires navigation + manipulation"
}
```

IMPORTANT:
- Generate queries that are ACTUALLY answerable from the scene
- For missing_detection cases, clearly mark them and explain what object you saw in the frame
- Be creative with phrasings - don't use the same structure repeatedly
- Cover ALL query types and difficulty levels in each batch
"""


# =============================================================================
# Image Annotation
# =============================================================================

def annotate_frame_with_boxes(
    image_path: Path,
    objects: List[ObjectInfo],
    visible_obj_ids: List[int],
    frame_id: int,
    detections: Dict[int, Tuple[int, int, int, int]],
    max_size: int = 1024,
) -> bytes:
    """
    Load frame image and annotate with bounding boxes and labels.
    Returns base64-encoded JPEG.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        scale = ratio
    else:
        scale = 1.0

    draw = ImageDraw.Draw(img)

    # Try to load a larger font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=18)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

    # Color palette for different objects
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    ]

    obj_id_to_info = {obj.obj_id: obj for obj in objects}

    for idx, obj_id in enumerate(visible_obj_ids):
        if obj_id not in detections:
            continue

        bbox = detections[obj_id]
        x1, y1, x2, y2 = [int(c * scale) for c in bbox]

        color = colors[idx % len(colors)]
        obj_info = obj_id_to_info.get(obj_id)
        label = obj_info.object_tag if obj_info else f"obj_{obj_id}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 22), label, fill="black", font=font)

        # Add object ID
        id_text = f"[{obj_id}]"
        draw.text((x1, y2 + 2), id_text, fill=color, font=font_small)

    # Add frame label
    frame_label = f"Frame {frame_id}"
    draw.text((10, 10), frame_label, fill="white", font=font)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# =============================================================================
# Main Generator Class
# =============================================================================

class QuerySampleGeneratorV2:
    """Advanced query generator with multi-modal input and diverse task types."""

    def __init__(self, scene_context: SceneContextV2):
        self.scene = scene_context
        self._pool = None

    @classmethod
    def from_scene_path(cls, scene_path: str, max_frames: int = 100) -> "QuerySampleGeneratorV2":
        """Load scene and create generator with full context."""
        scene_path = Path(scene_path)
        scene_name = scene_path.name

        logger.info(f"Loading scene from {scene_path}")

        # Load PCD data
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))
        if not pcd_files:
            raise FileNotFoundError(f"No pkl.gz files found in {pcd_dir}")

        with gzip.open(pcd_files[0], "rb") as f:
            pcd_data = pickle.load(f)

        # Load affordances
        aff_file = scene_path / "sg_cache_detect" / "object_affordances.json"
        if not aff_file.exists():
            aff_file = scene_path / "sg_cache" / "object_affordances.json"

        affordances = {}
        if aff_file.exists():
            with open(aff_file) as f:
                aff_list = json.load(f)
                affordances = {a["id"]: a for a in aff_list}

        # Load visibility index
        vis_index = {}
        vis_file = scene_path / "indices" / "visibility_index.pkl"
        if vis_file.exists():
            with open(vis_file, "rb") as f:
                vis_index = pickle.load(f)

        # Load frame detection files for 2D bboxes
        det_dir = scene_path / "gsa_detections_ram_withbg_allclasses"
        frame_dir = scene_path / "results"

        # Build object info list
        objects_raw = pcd_data.get("objects", pcd_data)
        objects: List[ObjectInfo] = []
        object_tags: List[str] = []
        categories = set()
        tag_to_ids: Dict[str, List[int]] = {}
        object_id_to_idx: Dict[int, int] = {}

        for i, obj in enumerate(objects_raw):
            if hasattr(obj, "__dict__"):
                obj = obj.__dict__

            aff = affordances.get(i, {})
            aff_data = aff.get("affordances", {})

            tag = aff.get("object_tag", obj.get("class_name", "unknown"))
            if isinstance(tag, list):
                tag = tag[0] if tag else "unknown"

            bbox = obj.get("bbox_np")
            if bbox is not None:
                center = tuple(float(x) for x in bbox.mean(axis=0))
                size = tuple(float(x) for x in (bbox.max(axis=0) - bbox.min(axis=0)))
                volume = float(np.prod(size))
            else:
                center = (0.0, 0.0, 0.0)
                size = (0.0, 0.0, 0.0)
                volume = 0.0

            best_views = [v[0] for v in vis_index.get("object_to_views", {}).get(i, [])[:10]]

            # Get 2D bbox samples from detections
            bbox_2d_samples = {}
            for view_id in best_views[:5]:
                xyxy = obj.get("xyxy")
                if xyxy is not None and len(xyxy) > 0:
                    # Use stored xyxy if available
                    bbox_2d_samples[view_id] = tuple(int(x) for x in xyxy[-1])

            # Handle class_name which may be a list
            cls_name = obj.get("class_name", "unknown")
            if isinstance(cls_name, list):
                cls_name = cls_name[0] if cls_name else "unknown"

            info = ObjectInfo(
                obj_id=i,
                object_tag=tag,
                class_name=str(cls_name),
                category=aff.get("category", "unknown"),
                summary=aff.get("summary", ""),
                center=center,
                size=size,
                volume=volume,
                co_objects=aff_data.get("co_objects", []),
                typical_location=aff_data.get("typical_location", ""),
                primary_functions=aff_data.get("primary_functions", []),
                best_views=best_views,
                num_detections=obj.get("num_detections", 0),
                bbox_2d_samples=bbox_2d_samples,
            )
            objects.append(info)
            object_tags.append(tag)
            categories.add(info.category)
            object_id_to_idx[i] = len(objects) - 1

            if tag not in tag_to_ids:
                tag_to_ids[tag] = []
            tag_to_ids[tag].append(i)

        # Build frame info - prioritize frames with detection files
        frames: List[FrameInfo] = []
        view_to_objects = vis_index.get("view_to_objects", {})

        # Get available detection files (they exist every N frames)
        det_files = sorted(det_dir.glob("frame*.pkl.gz"))
        det_frame_ids = set()
        for det_file in det_files:
            try:
                fid = int(det_file.stem.replace("frame", "").replace(".pkl", ""))
                det_frame_ids.add(fid)
            except ValueError:
                continue

        logger.debug(f"Found {len(det_frame_ids)} frames with detection files")

        # Process frames with detections first
        frame_files = sorted(frame_dir.glob("frame*.jpg"))
        frames_with_det = []
        frames_without_det = []

        for frame_file in frame_files:
            frame_id = int(frame_file.stem.replace("frame", ""))
            visible_objs = [obj_id for obj_id, _ in view_to_objects.get(frame_id, [])]

            if not visible_objs:
                continue

            # Load detections for this frame
            det_file = det_dir / f"frame{frame_id:06d}.pkl.gz"
            detections: Dict[int, Tuple[int, int, int, int]] = {}

            if det_file.exists():
                try:
                    with gzip.open(det_file, "rb") as f:
                        det_data = pickle.load(f)
                    xyxy = det_data.get("xyxy", [])
                    classes = det_data.get("classes", [])

                    # Map detections to objects by class similarity
                    for det_idx, bbox in enumerate(xyxy):
                        if det_idx < len(det_data.get("class_id", [])):
                            class_id = det_data["class_id"][det_idx]
                            if class_id < len(classes):
                                det_class = classes[int(class_id)].lower()
                                # Find matching object from visibility list
                                for obj_id in visible_objs:
                                    if obj_id in object_id_to_idx and obj_id not in detections:
                                        obj = objects[object_id_to_idx[obj_id]]
                                        # Get class_name (handle list type)
                                        cls_name = obj.class_name
                                        if isinstance(cls_name, list):
                                            cls_name = cls_name[0] if cls_name else ""
                                        cls_name = str(cls_name).lower()
                                        obj_tag = obj.object_tag.lower().replace("_", " ")
                                        # Match by tag or class
                                        if det_class in obj_tag or det_class in cls_name or \
                                           obj_tag in det_class or cls_name in det_class:
                                            detections[obj_id] = tuple(int(x) for x in bbox)
                                            break
                except Exception as e:
                    logger.warning(f"Failed to load detections for frame {frame_id}: {e}")

            frame_info = FrameInfo(
                frame_id=frame_id,
                image_path=frame_file,
                visible_objects=visible_objs[:20],
                detections=detections,
            )

            if detections:
                frames_with_det.append(frame_info)
            else:
                frames_without_det.append(frame_info)

        # Prioritize frames with detections
        frames = frames_with_det[:max_frames]
        if len(frames) < max_frames:
            frames.extend(frames_without_det[:max_frames - len(frames)])

        logger.info(f"Selected {len(frames)} frames ({len(frames_with_det)} with bboxes)")

        context = SceneContextV2(
            scene_name=scene_name,
            objects=objects,
            object_id_to_idx=object_id_to_idx,
            categories=sorted(categories),
            object_tags=object_tags,
            frames=frames,
            tag_to_ids=tag_to_ids,
            total_views=len(view_to_objects),
        )

        logger.info(
            f"Loaded scene '{scene_name}': {len(objects)} objects, "
            f"{len(categories)} categories, {len(frames)} frames"
        )
        return cls(context)

    def _get_pool(self):
        """Get Gemini client pool."""
        if self._pool is None:
            from conceptgraph.utils.llm_client import GeminiClientPool
            self._pool = GeminiClientPool.get_instance()
        return self._pool

    def _build_scene_text_context(self) -> str:
        """Build rich text description of scene."""
        lines = [
            f"# Scene: {self.scene.scene_name}",
            f"Total objects in scene graph: {len(self.scene.objects)}",
            f"Total camera frames: {self.scene.total_views}",
            "",
            "## Object Categories:",
            ", ".join(self.scene.categories),
            "",
            "## Detailed Object List (objects in scene graph):",
        ]

        # Group objects by category
        by_category: Dict[str, List[ObjectInfo]] = {}
        for obj in self.scene.objects:
            cat = obj.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(obj)

        for cat, objs in sorted(by_category.items()):
            lines.append(f"\n### {cat.upper()}")
            for obj in objs:
                size_str = f"{obj.size[0]:.2f}×{obj.size[1]:.2f}×{obj.size[2]:.2f}m"
                pos_str = f"({obj.center[0]:.2f}, {obj.center[1]:.2f}, {obj.center[2]:.2f})"
                lines.append(f"- [{obj.obj_id}] **{obj.object_tag}** (class: {obj.class_name})")
                lines.append(f"    Size: {size_str}, Position: {pos_str}, Volume: {obj.volume:.3f}m³")
                if obj.summary:
                    lines.append(f"    Description: {obj.summary[:150]}")
                if obj.co_objects:
                    lines.append(f"    Often found with: {', '.join(obj.co_objects[:5])}")
                if obj.primary_functions:
                    lines.append(f"    Functions: {', '.join(obj.primary_functions[:3])}")

        # Multi-instance objects
        lines.append("\n## Multi-Instance Objects (same category, multiple instances):")
        for tag, ids in self.scene.tag_to_ids.items():
            if len(ids) > 1:
                objs = [self.scene.objects[self.scene.object_id_to_idx[i]] for i in ids]
                sizes = [o.volume for o in objs]
                lines.append(f"- {tag}: {len(ids)} instances (IDs: {ids}, volumes: {[f'{s:.3f}' for s in sizes]})")

        # Hint: Model should infer spatial relations from images + 3D coordinates
        lines.append("\n## Spatial Reasoning Instructions:")
        lines.append("- Use the provided 3D coordinates (x, y, z) to reason about spatial relations")
        lines.append("- Look at the annotated frames to understand visual layout")
        lines.append("- Generate diverse spatial relations: on, under, near, beside, between, above, below, left_of, right_of, in_front_of, behind, etc.")
        lines.append("- Be creative with spatial descriptions - don't just use simple proximity")

        # Instruction for natural language variation
        lines.append("\n## Language Variation Instructions:")
        lines.append("- Use natural synonyms and alternative names freely (e.g., couch/sofa, cushion/pillow, footrest/ottoman)")
        lines.append("- Vary sentence structures and phrasings")
        lines.append("- Use colloquial expressions when appropriate")
        lines.append("- Don't repeat the exact object tags - paraphrase them naturally")

        return "\n".join(lines)

    def _select_diverse_frames(self, n: int = 5) -> List[FrameInfo]:
        """Select diverse frames covering different objects."""
        if len(self.scene.frames) <= n:
            return self.scene.frames

        # Greedy selection for coverage
        selected = []
        covered_objects = set()

        # Sort frames by number of visible objects (descending)
        frames_sorted = sorted(self.scene.frames, key=lambda f: len(f.visible_objects), reverse=True)

        for frame in frames_sorted:
            if len(selected) >= n:
                break
            new_objs = set(frame.visible_objects) - covered_objects
            if new_objs or len(selected) < 2:  # Always get at least 2 frames
                selected.append(frame)
                covered_objects.update(frame.visible_objects)

        return selected

    def _prepare_annotated_frames(self, frames: List[FrameInfo]) -> List[Dict[str, Any]]:
        """Prepare annotated frame images for LLM."""
        frame_data = []

        for frame in frames:
            try:
                img_bytes = annotate_frame_with_boxes(
                    frame.image_path,
                    self.scene.objects,
                    frame.visible_objects,
                    frame.frame_id,
                    frame.detections,
                )
                b64_img = base64.b64encode(img_bytes).decode("utf-8")

                visible_obj_names = [
                    self.scene.objects[self.scene.object_id_to_idx[oid]].object_tag
                    for oid in frame.visible_objects
                    if oid in self.scene.object_id_to_idx
                ]

                frame_data.append({
                    "frame_id": frame.frame_id,
                    "image_b64": b64_img,
                    "visible_objects": visible_obj_names,
                    "num_objects": len(frame.visible_objects),
                })
            except Exception as e:
                logger.warning(f"Failed to annotate frame {frame.frame_id}: {e}")

        return frame_data

    def _build_user_prompt(
        self,
        batch_size: int,
        difficulty_distribution: Dict[str, int],
        query_type_distribution: Dict[str, int],
        frame_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build user prompt with text and images."""
        scene_context = self._build_scene_text_context()

        text_prompt = f"""Based on the scene information and annotated frames provided, generate exactly {batch_size} diverse queries.

## Scene Context
{scene_context}

## Provided Frames
I'm showing you {len(frame_data)} annotated frames from this scene. Each frame has bounding boxes and labels for detected objects.

**IMPORTANT**: Look carefully at the frames! You may see objects that are NOT in the scene graph above. These are "missing detections" - generate some queries for these to test robustness.

Frame summaries:
"""
        for fd in frame_data:
            text_prompt += f"- Frame {fd['frame_id']}: {fd['num_objects']} objects visible ({', '.join(fd['visible_objects'][:8])}...)\n"

        text_prompt += f"""

## Generation Requirements

### Difficulty Distribution (total {batch_size}):
{json.dumps(difficulty_distribution, indent=2)}

### Query Type Distribution:
{json.dumps(query_type_distribution, indent=2)}

### Special Case Requirements:
- At least 2 queries should be **missing_detection** (objects you see in frames but not in scene graph)
- At least 3 queries should use **synonyms** (couch instead of sofa, etc.)
- At least 2 queries should be **ambiguous** (could match multiple objects)
- At least 2 queries should be **multi_instance** (target all instances)

### Query Length and Complexity:
- Easy: 5-15 words
- Medium: 10-25 words
- Hard: 15-35 words
- Expert: 25-50+ words with multiple steps or complex reasoning

Generate {batch_size} queries now. Return ONLY a JSON array, no other text.
"""

        # Build multimodal message
        content = [{"type": "text", "text": text_prompt}]

        for fd in frame_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{fd['image_b64']}",
                    "detail": "high",
                },
            })

        return content

    def _parse_response(self, response_text: str, source_frames: List[int]) -> List[GeneratedQueryV2]:
        """Parse LLM response to GeneratedQueryV2 objects."""
        import re

        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            logger.warning("No JSON array found in response")
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

        queries = []
        for item in data:
            try:
                query = GeneratedQueryV2(
                    query=item.get("query", ""),
                    query_type=item.get("query_type", "grounding"),
                    difficulty=item.get("difficulty", "medium"),
                    complexity_type=item.get("complexity_type", "direct"),
                    target_objects=item.get("target_objects", []),
                    anchor_objects=item.get("anchor_objects", []),
                    spatial_relations=item.get("spatial_relations", []),
                    special_case=item.get("special_case", "none"),
                    rationale=item.get("rationale", ""),
                    source_frames=source_frames,
                )
                if query.query:
                    queries.append(query)
            except Exception as e:
                logger.warning(f"Failed to parse query item: {e}")

        return queries

    def generate_samples(
        self,
        n: int = 50,
        frames_per_batch: int = 4,
        max_workers: int = 3,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        query_type_distribution: Optional[Dict[str, float]] = None,
    ) -> List[GeneratedQueryV2]:
        """
        Generate n query samples using parallel Gemini calls with image input.

        Args:
            n: Number of samples to generate
            frames_per_batch: Number of annotated frames per LLM call
            max_workers: Number of parallel LLM calls
            difficulty_distribution: Ratio for each difficulty level
            query_type_distribution: Ratio for each query type
        """
        if difficulty_distribution is None:
            difficulty_distribution = {
                "easy": 0.20,
                "medium": 0.30,
                "hard": 0.30,
                "expert": 0.20,
            }

        if query_type_distribution is None:
            query_type_distribution = {
                "grounding": 0.25,
                "qa": 0.20,
                "navigation": 0.15,
                "manipulation": 0.15,
                "counting": 0.10,
                "reasoning": 0.15,
            }

        # Calculate counts
        diff_counts = {k: max(1, int(n * v)) for k, v in difficulty_distribution.items()}
        type_counts = {k: max(1, int(n * v)) for k, v in query_type_distribution.items()}

        pool = self._get_pool()
        batch_size = max(8, n // max_workers)
        num_batches = (n + batch_size - 1) // batch_size

        logger.info(f"Generating {n} samples in {num_batches} batches of ~{batch_size}")

        all_queries: List[GeneratedQueryV2] = []

        def generate_batch(batch_idx: int) -> List[GeneratedQueryV2]:
            """Generate one batch of queries with images."""
            # Select diverse frames for this batch
            frames = self._select_diverse_frames(frames_per_batch)
            frame_data = self._prepare_annotated_frames(frames)
            source_frames = [f.frame_id for f in frames]

            if not frame_data:
                logger.warning(f"Batch {batch_idx}: No frame data available")
                return []

            # Distribute difficulty and types across batches
            batch_diff = {k: max(1, v // num_batches) for k, v in diff_counts.items()}
            batch_types = {k: max(1, v // num_batches) for k, v in type_counts.items()}
            actual_batch_size = sum(batch_diff.values())

            user_content = self._build_user_prompt(
                actual_batch_size, batch_diff, batch_types, frame_data
            )

            # Try with pool
            tried = set()
            while len(tried) < pool.pool_size:
                client, config_idx = pool.get_next_client(temperature=0.8, max_tokens=8000)
                if config_idx in tried:
                    continue
                tried.add(config_idx)

                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=user_content),
                    ]

                    response = client.invoke(messages)
                    content = response.content if hasattr(response, "content") else str(response)
                    pool.record_request(config_idx, rate_limited=False)

                    queries = self._parse_response(content, source_frames)
                    logger.info(f"Batch {batch_idx}: generated {len(queries)} queries from {len(frame_data)} frames")
                    return queries

                except Exception as e:
                    from conceptgraph.utils.llm_client import _is_rate_limit_error
                    is_rate_limited = _is_rate_limit_error(e)
                    pool.record_request(config_idx, rate_limited=is_rate_limited)

                    if is_rate_limited:
                        logger.warning(f"Batch {batch_idx}: rate limited, trying next key...")
                    else:
                        logger.error(f"Batch {batch_idx} error: {e}")
                    continue

            logger.error(f"Batch {batch_idx}: all keys exhausted")
            return []

        # Run batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_batch, i) for i in range(num_batches)]
            for future in as_completed(futures):
                queries = future.result()
                all_queries.extend(queries)

        logger.success(f"Generated {len(all_queries)} total queries")

        # Shuffle and trim
        random.shuffle(all_queries)
        return all_queries[:n]

    def save_samples(self, queries: List[GeneratedQueryV2], output_path: str):
        """Save generated samples to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "query": q.query,
                "query_type": q.query_type,
                "difficulty": q.difficulty,
                "complexity_type": q.complexity_type,
                "target_objects": q.target_objects,
                "anchor_objects": q.anchor_objects,
                "spatial_relations": q.spatial_relations,
                "special_case": q.special_case,
                "rationale": q.rationale,
                "source_frames": q.source_frames,
            }
            for q in queries
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(queries)} queries to {output_path}")

        # Print summary
        self._print_summary(queries)

    def _print_summary(self, queries: List[GeneratedQueryV2]):
        """Print distribution summary."""
        from collections import Counter

        difficulty_counts = Counter(q.difficulty for q in queries)
        type_counts = Counter(q.query_type for q in queries)
        special_counts = Counter(q.special_case for q in queries)
        complexity_counts = Counter(q.complexity_type for q in queries)

        logger.info("\n" + "=" * 60)
        logger.info("Generation Summary")
        logger.info("=" * 60)
        logger.info(f"Difficulty: {dict(difficulty_counts)}")
        logger.info(f"Query Types: {dict(type_counts)}")
        logger.info(f"Special Cases: {dict(special_counts)}")
        logger.info(f"Complexity: {dict(complexity_counts)}")

        # Average query length
        avg_len = sum(len(q.query.split()) for q in queries) / len(queries) if queries else 0
        logger.info(f"Average query length: {avg_len:.1f} words")


def main():
    """Generate samples for room0."""
    import os
    import sys

    replica_root = os.environ.get("REPLICA_ROOT", "/Users/bytedance/Replica")
    scene_path = Path(replica_root) / "room0"

    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        sys.exit(1)

    # Create generator
    generator = QuerySampleGeneratorV2.from_scene_path(str(scene_path), max_frames=50)

    # Generate samples
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    samples = generator.generate_samples(
        n=n,
        frames_per_batch=4,
        max_workers=3,
    )

    # Save to file
    output_path = scene_path / "generated_queries_v2.json"
    generator.save_samples(samples, str(output_path))

    # Print examples
    print("\n" + "=" * 70)
    print("Sample Generated Queries")
    print("=" * 70)

    # Group by type
    by_type: Dict[str, List[GeneratedQueryV2]] = {}
    for q in samples:
        if q.query_type not in by_type:
            by_type[q.query_type] = []
        by_type[q.query_type].append(q)

    for qtype, qs in by_type.items():
        print(f"\n### {qtype.upper()} ###")
        for q in qs[:3]:
            print(f"\n[{q.difficulty}/{q.complexity_type}] {q.query}")
            if q.special_case != "none":
                print(f"  Special: {q.special_case}")
            print(f"  Target: {q.target_objects}, Anchor: {q.anchor_objects}")


if __name__ == "__main__":
    main()
