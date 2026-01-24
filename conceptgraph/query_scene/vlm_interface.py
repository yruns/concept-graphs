"""VLM input construction and output parsing with query-adaptive strategies."""
from __future__ import annotations
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import requests
from .data_structures import GroundingResult, ObjectNode, QueryInfo, QueryType, ObjectDescriptions


@dataclass
class VLMInput:
    """Structured input for VLM inference."""
    images: List[np.ndarray]
    prompt: str
    object_annotations: Dict[int, str] = field(default_factory=dict)
    bev_image: Optional[np.ndarray] = None
    view_ids: List[int] = field(default_factory=list)


# =============================================================================
# Query-Adaptive Strategy Map (from plan)
# =============================================================================

STRATEGY_MAP = {
    QueryType.SIMPLE_OBJECT: {
        "views": "best_single",      # Most clear single view
        "descriptions": "target_only",
        "max_views": 1,
        "prompt_style": "simple",
    },
    QueryType.SPATIAL_RELATION: {
        "views": "joint_coverage",   # Cover both target and anchor
        "descriptions": "all_candidates",
        "max_views": 3,
        "prompt_style": "spatial",
    },
    QueryType.COUNTING: {
        "views": "multi_angle",      # Multiple angles to avoid occlusion
        "descriptions": "enumeration",
        "max_views": 4,
        "prompt_style": "counting",
    },
    QueryType.ATTRIBUTE: {
        "views": "close_up",         # Close-up for details
        "descriptions": "detailed",
        "max_views": 2,
        "prompt_style": "attribute",
    },
    QueryType.FUNCTIONAL_REGION: {
        "views": "region_overview",  # Wide view of region
        "descriptions": "region_summary",
        "max_views": 2,
        "prompt_style": "region",
    },
    QueryType.COMPARISON: {
        "views": "same_frame",       # Try to get objects in same view
        "descriptions": "comparative",
        "max_views": 2,
        "prompt_style": "comparison",
    },
}


class VLMInputConstructor:
    """Query-adaptive VLM input constructor."""
    
    def __init__(self, default_max_images: int = 4):
        self.default_max_images = default_max_images
    
    def construct(
        self,
        query_info: QueryInfo,
        candidates: List[ObjectNode],
        view_images: Dict[int, np.ndarray],
        descriptions: Dict[int, ObjectDescriptions] = None,
        bev_image: Optional[np.ndarray] = None,
        visibility_index: Any = None,
    ) -> VLMInput:
        """Construct optimal VLM input based on query type.
        
        Args:
            query_info: Parsed query information
            candidates: Candidate objects
            view_images: Dictionary of view_id -> RGB image
            descriptions: Object descriptions (id -> ObjectDescriptions)
            bev_image: Bird's eye view image
            visibility_index: Visibility index for view selection
        
        Returns:
            VLMInput with optimally selected images and prompt
        """
        strategy = STRATEGY_MAP.get(query_info.query_type, STRATEGY_MAP[QueryType.SIMPLE_OBJECT])
        
        # 1. Select views based on strategy
        view_ids = self._select_views(
            candidates,
            view_images,
            strategy["views"],
            strategy["max_views"],
            visibility_index,
            query_info,
        )
        images = [view_images[v] for v in view_ids if v in view_images]
        
        # 2. Build prompt based on strategy
        prompt = self._build_prompt(
            query_info,
            candidates,
            descriptions or {},
            strategy["prompt_style"],
        )
        
        # 3. Create annotations
        annotations = {obj.obj_id: f"#{obj.obj_id}: {obj.category}" for obj in candidates}
        
        # 4. Include BEV if requested
        bev = bev_image if query_info.use_bev else None
        
        return VLMInput(
            images=images,
            prompt=prompt,
            object_annotations=annotations,
            bev_image=bev,
            view_ids=view_ids,
        )
    
    def _select_views(
        self,
        candidates: List[ObjectNode],
        view_images: Dict[int, np.ndarray],
        strategy: str,
        max_views: int,
        visibility_index: Any,
        query_info: QueryInfo,
    ) -> List[int]:
        """Select views based on strategy."""
        
        if strategy == "best_single":
            # Get best view for first candidate
            if candidates and candidates[0].best_view_ids:
                for v in candidates[0].best_view_ids:
                    if v in view_images:
                        return [v]
            return list(view_images.keys())[:1]
        
        elif strategy == "joint_coverage":
            # Greedy selection to maximize joint coverage
            return self._greedy_joint_coverage(candidates, view_images, max_views, visibility_index)
        
        elif strategy == "multi_angle":
            # Select diverse viewpoints
            all_views = set()
            for obj in candidates:
                all_views.update(obj.best_view_ids[:2])
            return list(all_views)[:max_views]
        
        elif strategy == "close_up":
            # Select views with highest resolution
            best = []
            for obj in candidates[:1]:  # Focus on first candidate
                for v in obj.best_view_ids[:max_views]:
                    if v in view_images:
                        best.append(v)
            return best[:max_views] if best else list(view_images.keys())[:max_views]
        
        elif strategy == "same_frame":
            # Find views that contain multiple candidates
            scores = {}
            for v in view_images:
                count = sum(1 for obj in candidates if v in obj.best_view_ids)
                if count > 0:
                    scores[v] = count
            sorted_views = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return sorted_views[:max_views]
        
        else:  # region_overview, default
            # Simple: use available views from candidates
            views = set()
            for obj in candidates:
                views.update(obj.best_view_ids[:2])
            return list(views)[:max_views] if views else list(view_images.keys())[:max_views]
    
    def _greedy_joint_coverage(
        self,
        candidates: List[ObjectNode],
        view_images: Dict[int, np.ndarray],
        k: int,
        visibility_index: Any,
    ) -> List[int]:
        """Greedy algorithm to select k views that maximize joint coverage."""
        # Collect all candidate views
        candidate_views = set()
        for obj in candidates:
            candidate_views.update(v for v in obj.best_view_ids if v in view_images)
        
        if not candidate_views:
            return list(view_images.keys())[:k]
        
        # Greedy selection
        selected = []
        covered = set()
        
        for _ in range(k):
            if not candidate_views - set(selected):
                break
            
            best_view, best_gain = None, 0
            for v in candidate_views - set(selected):
                # Count newly covered objects
                gain = sum(1 for obj in candidates 
                          if obj.obj_id not in covered and v in obj.best_view_ids)
                if gain > best_gain:
                    best_gain, best_view = gain, v
            
            if best_view is None:
                break
            
            selected.append(best_view)
            covered.update(obj.obj_id for obj in candidates if best_view in obj.best_view_ids)
        
        return selected
    
    def _build_prompt(
        self,
        query_info: QueryInfo,
        candidates: List[ObjectNode],
        descriptions: Dict[int, ObjectDescriptions],
        style: str,
    ) -> str:
        """Build prompt based on style."""
        
        # Build candidate list
        candidate_lines = []
        for obj in candidates:
            desc = descriptions.get(obj.obj_id)
            if desc and desc.summary:
                candidate_lines.append(f"  #{obj.obj_id}: {obj.category} - {desc.summary}")
            else:
                candidate_lines.append(f"  #{obj.obj_id}: {obj.category}")
        candidate_text = "\n".join(candidate_lines)
        
        if style == "simple":
            return f'''Task: Find "{query_info.target}"

Candidates:
{candidate_text}

Return JSON: {{"object_id": <ID>, "confidence": 0-1}}'''
        
        elif style == "spatial":
            return f'''Task: {query_info.original_query}

Find the {query_info.target} that is {query_info.relation or "near"} the {query_info.anchor or "reference"}.

Candidates:
{candidate_text}

Analyze the spatial relationship and return JSON:
{{"object_id": <ID>, "confidence": 0-1, "reasoning": "explain spatial relation"}}'''
        
        elif style == "counting":
            return f'''Task: Count {query_info.target} in the scene.

Visible candidates:
{candidate_text}

Return JSON: {{"count": <number>, "object_ids": [<list of IDs>]}}'''
        
        elif style == "attribute":
            return f'''Task: {query_info.original_query}

Examine the candidate objects:
{candidate_text}

Return JSON: {{"object_id": <ID>, "attribute": "<value>", "confidence": 0-1}}'''
        
        elif style == "comparison":
            return f'''Task: {query_info.original_query}

Compare these candidates:
{candidate_text}

Return JSON: {{"object_id": <best match ID>, "reasoning": "comparison result"}}'''
        
        else:  # region
            return f'''Task: {query_info.original_query}

Objects in the area:
{candidate_text}

Return JSON: {{"object_id": <ID>, "confidence": 0-1}}'''


class VLMOutputParser:
    """Parse VLM text output to grounding result using multi-level strategy."""
    
    def __init__(self, objects: List[ObjectNode]):
        self.objects = {obj.obj_id: obj for obj in objects}
    
    def parse(self, vlm_output: str, query_info: QueryInfo) -> GroundingResult:
        """Parse VLM output using 3-level strategy."""
        # Level 1: JSON parsing (highest confidence)
        result = self._try_json_parse(vlm_output)
        if result:
            return result
        
        # Level 2: Regex ID extraction (medium confidence)
        result = self._try_regex_parse(vlm_output)
        if result:
            return result
        
        # Level 3: Description matching (lower confidence)
        result = self._try_description_match(vlm_output)
        if result:
            return result
        
        return GroundingResult.failure("Could not parse VLM output")
    
    def _try_json_parse(self, output: str) -> Optional[GroundingResult]:
        """Level 1: Parse JSON response."""
        try:
            # Find JSON in output
            match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
            if match:
                data = json.loads(match.group())
                
                # Handle counting queries
                if "count" in data:
                    obj_ids = data.get("object_ids", [])
                    if obj_ids and obj_ids[0] in self.objects:
                        return GroundingResult.from_object(
                            self.objects[obj_ids[0]],
                            confidence=0.9,
                            reasoning=f"Count: {data['count']}",
                        )
                
                # Handle standard object selection
                obj_id = data.get("object_id")
                if obj_id is not None and obj_id in self.objects:
                    return GroundingResult.from_object(
                        self.objects[obj_id],
                        confidence=data.get("confidence", 0.85),
                        reasoning=data.get("reasoning", ""),
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None
    
    def _try_regex_parse(self, output: str) -> Optional[GroundingResult]:
        """Level 2: Extract object ID using regex patterns."""
        patterns = [
            r'#(\d+)',                    # #12
            r'ID\s*[:=]?\s*(\d+)',         # ID: 12, ID=12
            r'object_id["\s:]+(\d+)',     # object_id: 12
            r'物体\s*(\d+)',               # 物体 12
            r'候选\s*(\d+)',               # 候选 1 (1-indexed)
        ]
        
        candidate_ids = set(self.objects.keys())
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                obj_id = int(match.group(1))
                if obj_id in candidate_ids:
                    return GroundingResult.from_object(
                        self.objects[obj_id],
                        confidence=0.65,
                        reasoning="Extracted from text",
                    )
        
        return None
    
    def _try_description_match(self, output: str) -> Optional[GroundingResult]:
        """Level 3: Match output text to object descriptions."""
        output_lower = output.lower()
        
        # Try to match category names
        best_match = None
        best_score = 0
        
        for obj_id, obj in self.objects.items():
            category = obj.category.lower()
            if category in output_lower:
                # Count occurrences as score
                score = output_lower.count(category)
                if score > best_score:
                    best_score = score
                    best_match = obj
        
        if best_match:
            return GroundingResult.from_object(
                best_match,
                confidence=0.4,
                reasoning="Matched by description",
            )
        
        return None


class VLMClient:
    """Client for VLM inference."""
    
    def __init__(self, url: str = "http://localhost:11434", model: str = "llava:7b"):
        self.url = url
        self.model = model
    
    def infer(self, vlm_input: VLMInput) -> str:
        """Run VLM inference and return response text."""
        # Encode images
        images_b64 = []
        for img in vlm_input.images:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            images_b64.append(base64.b64encode(buffer).decode())
        
        # Add BEV if present
        if vlm_input.bev_image is not None:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(vlm_input.bev_image, cv2.COLOR_RGB2BGR))
            images_b64.append(base64.b64encode(buffer).decode())
        
        # Try OpenAI-compatible API first
        try:
            content = [{"type": "text", "text": vlm_input.prompt}]
            for img_b64 in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 500,
                },
                timeout=60,
            )
            if response.ok:
                return response.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        
        # Fallback to Ollama API
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": vlm_input.prompt,
                    "images": images_b64,
                    "stream": False,
                },
                timeout=60,
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception:
            pass
        
        raise ConnectionError(f"VLM unavailable at {self.url}")
