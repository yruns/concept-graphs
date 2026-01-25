"""
Multi-level description generation using VLM.

This module generates rich descriptions for objects, enabling
the VLM to reason with natural language context rather than
just visual features.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from typing import Any, Dict, List, Optional
import numpy as np
import requests
from .data_structures import ObjectDescriptions, ObjectNode


class DescriptionGenerator:
    """Generate multi-level descriptions for objects using VLM.
    
    Generates 5 levels of description:
    - appearance: Visual attributes (color, shape, material)
    - function: What the object is used for
    - spatial: Location relative to other objects
    - context: Relationship to functional zones
    - summary: Brief one-line summary
    """
    
    def __init__(
        self,
        vlm_url: str = "http://localhost:11434",
        vlm_model: str = "llava:7b",
        llm_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
    ):
        self.vlm_url = vlm_url
        self.vlm_model = vlm_model
        self.llm_url = llm_url
        self.llm_model = llm_model
    
    def generate_descriptions(
        self,
        objects: List[ObjectNode],
        rgb_images: List[np.ndarray],
        scene_repr: Any = None,
    ) -> Dict[int, ObjectDescriptions]:
        """Generate descriptions for all objects.
        
        Args:
            objects: List of ObjectNode instances
            rgb_images: List of RGB images indexed by view_id
            scene_repr: Scene representation for spatial context
        
        Returns:
            Dictionary mapping object_id to ObjectDescriptions
        """
        results = {}
        
        for i, obj in enumerate(objects):
            if i % 10 == 0:
                logger.debug(f"Generating descriptions: {i}/{len(objects)}")
            
            try:
                desc = self._generate_object_description(obj, rgb_images, objects, scene_repr)
                results[obj.obj_id] = desc
            except Exception as e:
                logger.warning(f"Failed to generate description for object {obj.obj_id}: {e}")
                results[obj.obj_id] = self._default_description(obj)
        
        return results
    
    def _generate_object_description(
        self,
        obj: ObjectNode,
        rgb_images: List[np.ndarray],
        all_objects: List[ObjectNode],
        scene_repr: Any = None,
    ) -> ObjectDescriptions:
        """Generate description for a single object."""
        
        # Get best view crop (if available)
        crop_image = None
        if obj.best_view_ids and rgb_images:
            from .utils import project_3d_bbox_to_2d, crop_object_from_image
            
            for view_id in obj.best_view_ids[:3]:
                if view_id < len(rgb_images):
                    rgb = rgb_images[view_id]
                    h, w = rgb.shape[:2]
                    K = np.array([[600, 0, w/2], [0, 600, h/2], [0, 0, 1]], dtype=np.float32)
                    
                    # Simple projection using centroid
                    if obj.centroid is not None:
                        # Use center of image as approximation if no pose available
                        cx, cy = int(w/2), int(h/2)
                        size = min(w, h) // 4
                        x1 = max(0, cx - size)
                        y1 = max(0, cy - size)
                        x2 = min(w, cx + size)
                        y2 = min(h, cy + size)
                        crop_image = rgb[y1:y2, x1:x2]
                        break
        
        # Generate appearance description using VLM (if crop available)
        appearance = self._generate_appearance(obj, crop_image)
        
        # Generate function description based on category
        function = self._generate_function(obj)
        
        # Generate spatial description based on geometry
        spatial = self._generate_spatial(obj, all_objects)
        
        # Generate context description
        context = self._generate_context(obj, all_objects)
        
        # Generate summary
        summary = self._generate_summary(obj, appearance, spatial)
        
        return ObjectDescriptions(
            appearance=appearance,
            function=function,
            spatial=spatial,
            context=context,
            summary=summary,
        )
    
    def _generate_appearance(self, obj: ObjectNode, crop_image: Optional[np.ndarray]) -> str:
        """Generate appearance description using VLM."""
        if crop_image is None:
            return f"A {obj.category}"
        
        try:
            import base64
            import cv2
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            img_b64 = base64.b64encode(buffer).decode()
            
            prompt = f"""Describe this {obj.category} briefly in one sentence.
Focus on: color, material, shape, size.
Format: "[color] [material] [shape/style] {obj.category}"
Example: "white metal rectangular table" or "gray fabric L-shaped sofa"
"""
            
            # Try OpenAI-compatible API
            try:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
                resp = requests.post(
                    f"{self.vlm_url}/v1/chat/completions",
                    json={"model": self.vlm_model, "messages": [{"role": "user", "content": content}], "max_tokens": 100},
                    timeout=30
                )
                if resp.ok:
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except:
                pass
            
            # Try Ollama API
            try:
                resp = requests.post(
                    f"{self.vlm_url}/api/generate",
                    json={"model": self.vlm_model, "prompt": prompt, "images": [img_b64], "stream": False},
                    timeout=30
                )
                if resp.ok:
                    return resp.json().get("response", "").strip()
            except:
                pass
            
        except Exception as e:
            pass
        
        return f"A {obj.category}"
    
    def _generate_function(self, obj: ObjectNode) -> str:
        """Generate function description based on category."""
        # Common furniture functions
        functions = {
            "chair": "A seating furniture for one person",
            "table": "A surface for placing items or eating",
            "sofa": "A comfortable seating for relaxing or conversation",
            "lamp": "Provides lighting for the area",
            "bed": "A sleeping surface",
            "cabinet": "Storage furniture with doors or drawers",
            "desk": "A work surface for reading or writing",
            "shelf": "Storage for displaying or organizing items",
            "tv": "A display screen for entertainment",
            "plant": "Decorative greenery",
            "window": "Allows natural light and ventilation",
            "door": "An entrance or exit point",
            "pillow": "A cushion for comfort or decoration",
            "blanket": "A textile for warmth or decoration",
        }
        
        category_lower = obj.category.lower()
        for key, desc in functions.items():
            if key in category_lower:
                return desc
        
        return f"A {obj.category} for indoor use"
    
    def _generate_spatial(self, obj: ObjectNode, all_objects: List[ObjectNode]) -> str:
        """Generate spatial description based on geometry."""
        if obj.centroid is None:
            return "Location unknown"
        
        # Find nearby objects
        nearby = []
        for other in all_objects:
            if other.obj_id == obj.obj_id or other.centroid is None:
                continue
            dist = np.linalg.norm(obj.centroid - other.centroid)
            if dist < 2.0:  # Within 2 meters
                nearby.append((other, dist))
        
        nearby.sort(key=lambda x: x[1])
        
        if not nearby:
            return f"Standalone {obj.category}"
        
        # Describe relation to nearest object
        nearest, dist = nearby[0]
        
        # Determine relative position
        dx = obj.centroid[0] - nearest.centroid[0]
        dy = obj.centroid[1] - nearest.centroid[1]
        
        if abs(dx) > abs(dy):
            direction = "to the right of" if dx > 0 else "to the left of"
        else:
            direction = "in front of" if dy > 0 else "behind"
        
        return f"{direction.capitalize()} the {nearest.category}, about {dist:.1f}m away"
    
    def _generate_context(self, obj: ObjectNode, all_objects: List[ObjectNode]) -> str:
        """Generate context description based on surrounding objects."""
        if obj.centroid is None:
            return ""
        
        # Find objects within 3 meters
        nearby_categories = []
        for other in all_objects:
            if other.obj_id == obj.obj_id or other.centroid is None:
                continue
            dist = np.linalg.norm(obj.centroid - other.centroid)
            if dist < 3.0:
                nearby_categories.append(other.category)
        
        if not nearby_categories:
            return ""
        
        # Infer functional zone
        zone_indicators = {
            "living_area": ["sofa", "tv", "coffee table", "armchair"],
            "dining_area": ["dining table", "chair"],
            "bedroom": ["bed", "nightstand", "wardrobe"],
            "kitchen": ["stove", "refrigerator", "sink", "cabinet"],
            "office": ["desk", "office chair", "computer"],
        }
        
        for zone, indicators in zone_indicators.items():
            if any(ind in cat.lower() for cat in nearby_categories for ind in indicators):
                zone_name = zone.replace("_", " ")
                return f"Part of the {zone_name}"
        
        return f"Near {', '.join(set(nearby_categories[:3]))}"
    
    def _generate_summary(self, obj: ObjectNode, appearance: str, spatial: str) -> str:
        """Generate a brief summary combining key attributes."""
        if "unknown" in spatial.lower():
            return appearance
        
        # Extract key words
        if spatial and appearance != f"A {obj.category}":
            # Combine spatial info with brief appearance
            return f"{appearance.split('.')[0]} {spatial.lower().split(',')[0]}"
        
        return appearance
    
    def _default_description(self, obj: ObjectNode) -> ObjectDescriptions:
        """Return default description when generation fails."""
        return ObjectDescriptions(
            appearance=f"A {obj.category}",
            function=f"A {obj.category} for indoor use",
            spatial="Location in the scene",
            context="",
            summary=f"A {obj.category}",
        )


def generate_descriptions(
    objects: List[ObjectNode],
    rgb_images: List[np.ndarray] = None,
    vlm_url: str = "http://localhost:11434",
    vlm_model: str = "llava:7b",
) -> Dict[int, ObjectDescriptions]:
    """Convenience function to generate descriptions."""
    generator = DescriptionGenerator(vlm_url=vlm_url, vlm_model=vlm_model)
    return generator.generate_descriptions(objects, rgb_images or [])
