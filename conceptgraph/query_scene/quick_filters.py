"""
Quick Filters for Fast Spatial Filtering.

This module implements lightweight, fast filters that can quickly eliminate
irrelevant objects using simple coordinate comparisons or attribute matching.

These filters are designed to:
1. Run in O(1) or O(n) time with simple comparisons
2. Use loose thresholds to avoid false negatives
3. Be applied only when the query contains relevant constraints

The quick filters are a pre-filtering step before the full spatial relation
checking, which is more accurate but slower.

Usage:
    filters = QuickFilters()
    
    # Check if a relation has a quick filter
    if filters.has_filter("on"):
        candidates = filters.apply("on", candidates, anchor_objects)
    
    # Or use the all-in-one method
    candidates = filters.filter_candidates(candidates, anchor_objects, relation="on")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .keyframe_selector import SceneObject


class FilterType(str, Enum):
    """Type of quick filter."""
    VERTICAL = "vertical"      # Z-axis comparison
    HORIZONTAL = "horizontal"  # X/Y-axis comparison
    DISTANCE = "distance"      # Euclidean distance
    ATTRIBUTE = "attribute"    # Color, size, etc.


@dataclass
class FilterConfig:
    """Configuration for a quick filter."""
    filter_type: FilterType
    comparator: str  # "gt", "lt", "eq", "range", "min", "max"
    axis: Optional[str] = None  # "x", "y", "z", "xy", "xyz"
    threshold: Optional[float] = None
    loose_factor: float = 1.5  # Multiplier for loose threshold


# Pre-defined quick filters for common spatial relations
QUICK_FILTER_CONFIGS: Dict[str, FilterConfig] = {
    # Vertical relations (Z-axis)
    "on_top_of": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="gt",
        axis="z",
        threshold=0.0,  # target.z > anchor.z
        loose_factor=1.0,
    ),
    "above": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="gt",
        axis="z",
        threshold=0.1,
        loose_factor=1.0,
    ),
    "below": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="lt",
        axis="z",
        threshold=-0.1,
        loose_factor=1.0,
    ),
    "under": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="lt",
        axis="z",
        threshold=-0.1,
        loose_factor=1.0,
    ),
    "higher_than": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="gt",
        axis="z",
        threshold=0.05,
        loose_factor=1.0,
    ),
    "lower_than": FilterConfig(
        filter_type=FilterType.VERTICAL,
        comparator="lt",
        axis="z",
        threshold=-0.05,
        loose_factor=1.0,
    ),
    
    # Horizontal relations (X-axis, assuming +X is right)
    "left_of": FilterConfig(
        filter_type=FilterType.HORIZONTAL,
        comparator="lt",
        axis="x",
        threshold=-0.1,
        loose_factor=1.0,
    ),
    "right_of": FilterConfig(
        filter_type=FilterType.HORIZONTAL,
        comparator="gt",
        axis="x",
        threshold=0.1,
        loose_factor=1.0,
    ),
    
    # Horizontal relations (Y-axis, assuming +Y is forward)
    "in_front_of": FilterConfig(
        filter_type=FilterType.HORIZONTAL,
        comparator="gt",
        axis="y",
        threshold=0.1,
        loose_factor=1.0,
    ),
    "behind": FilterConfig(
        filter_type=FilterType.HORIZONTAL,
        comparator="lt",
        axis="y",
        threshold=-0.1,
        loose_factor=1.0,
    ),
    
    # Distance relations
    "near": FilterConfig(
        filter_type=FilterType.DISTANCE,
        comparator="lt",
        axis="xyz",
        threshold=3.0,  # 3 meters
        loose_factor=1.5,
    ),
    "next_to": FilterConfig(
        filter_type=FilterType.DISTANCE,
        comparator="lt",
        axis="xyz",
        threshold=1.5,  # 1.5 meters
        loose_factor=1.5,
    ),
    "beside": FilterConfig(
        filter_type=FilterType.DISTANCE,
        comparator="lt",
        axis="xyz",
        threshold=1.5,
        loose_factor=1.5,
    ),
    "close_to": FilterConfig(
        filter_type=FilterType.DISTANCE,
        comparator="lt",
        axis="xyz",
        threshold=2.0,
        loose_factor=1.5,
    ),
    "far_from": FilterConfig(
        filter_type=FilterType.DISTANCE,
        comparator="gt",
        axis="xyz",
        threshold=3.0,
        loose_factor=0.7,  # Tighter for "far"
    ),
}

# Relation aliases to canonical names
FILTER_ALIASES: Dict[str, str] = {
    "on": "on_top_of",
    "upon": "on_top_of",
    "atop": "on_top_of",
    "over": "above",
    "beneath": "below",
    "underneath": "below",
    "left": "left_of",
    "right": "right_of",
    "front": "in_front_of",
    "back": "behind",
    "back_of": "behind",
    "nearby": "near",
    "adjacent": "next_to",
    "adjacent_to": "next_to",
}

# Common color keywords (for attribute filtering)
COLOR_KEYWORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "beige", "tan",
    "gold", "silver", "bronze", "copper",
}

# Common size keywords
SIZE_KEYWORDS = {
    "large": "max",
    "big": "max",
    "biggest": "max",
    "largest": "max",
    "huge": "max",
    "small": "min",
    "little": "min",
    "smallest": "min",
    "tiny": "min",
}


class QuickFilters:
    """
    Fast spatial filters using simple coordinate comparisons.
    
    These filters are designed to quickly eliminate obviously irrelevant
    objects before the full spatial relation checking.
    
    Example:
        filters = QuickFilters()
        
        # Filter objects that are "on" the anchor
        filtered = filters.filter_candidates(
            candidates, 
            anchor_objects, 
            relation="on"
        )
    """
    
    def __init__(
        self,
        configs: Optional[Dict[str, FilterConfig]] = None,
        use_loose_threshold: bool = True,
    ):
        """
        Initialize quick filters.
        
        Args:
            configs: Custom filter configurations (uses defaults if None)
            use_loose_threshold: Whether to use loose thresholds
        """
        self.configs = configs or QUICK_FILTER_CONFIGS
        self.use_loose_threshold = use_loose_threshold
    
    def _get_canonical_relation(self, relation: str) -> str:
        """Get canonical relation name."""
        relation_lower = relation.lower().replace(" ", "_")
        return FILTER_ALIASES.get(relation_lower, relation_lower)
    
    def has_filter(self, relation: str) -> bool:
        """Check if a quick filter exists for this relation."""
        canonical = self._get_canonical_relation(relation)
        return canonical in self.configs
    
    def _get_centroid(self, obj: "SceneObject") -> np.ndarray:
        """Get object centroid."""
        if hasattr(obj, 'centroid') and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def _compute_anchor_center(self, anchors: List["SceneObject"]) -> np.ndarray:
        """Compute center point of anchor objects."""
        if not anchors:
            return np.zeros(3, dtype=np.float32)
        
        centroids = [self._get_centroid(a) for a in anchors]
        return np.mean(centroids, axis=0)
    
    def filter_candidates(
        self,
        candidates: List["SceneObject"],
        anchors: List["SceneObject"],
        relation: str,
        return_scores: bool = False,
    ) -> Union[List["SceneObject"], Tuple[List["SceneObject"], Dict[int, float]]]:
        """
        Filter candidates using quick spatial filters.
        
        Args:
            candidates: List of candidate objects
            anchors: List of anchor/reference objects
            relation: Spatial relation string
            return_scores: Whether to return confidence scores
            
        Returns:
            Filtered list of candidates (and optionally scores)
        """
        canonical = self._get_canonical_relation(relation)
        
        if canonical not in self.configs:
            logger.debug(f"[QuickFilter] No filter for '{relation}', returning all candidates")
            if return_scores:
                return candidates, {obj.obj_id: 1.0 for obj in candidates}
            return candidates
        
        config = self.configs[canonical]
        
        if config.filter_type == FilterType.VERTICAL:
            result = self._filter_vertical(candidates, anchors, config)
        elif config.filter_type == FilterType.HORIZONTAL:
            result = self._filter_horizontal(candidates, anchors, config)
        elif config.filter_type == FilterType.DISTANCE:
            result = self._filter_distance(candidates, anchors, config)
        else:
            result = [(c, 1.0) for c in candidates]
        
        logger.debug(
            f"[QuickFilter] '{relation}' ({canonical}): "
            f"{len(candidates)} -> {len(result)} candidates"
        )
        
        if return_scores:
            return [r[0] for r in result], {r[0].obj_id: r[1] for r in result}
        return [r[0] for r in result]
    
    def _filter_vertical(
        self,
        candidates: List["SceneObject"],
        anchors: List["SceneObject"],
        config: FilterConfig,
    ) -> List[Tuple["SceneObject", float]]:
        """Filter by vertical (Z-axis) position."""
        anchor_center = self._compute_anchor_center(anchors)
        anchor_z = anchor_center[2]
        
        threshold = config.threshold or 0.0
        if self.use_loose_threshold:
            # For "gt", we want to include more objects below the strict threshold
            if config.comparator == "gt":
                threshold -= abs(threshold) * (config.loose_factor - 1)
            else:  # "lt"
                threshold += abs(threshold) * (config.loose_factor - 1)
        
        results = []
        for obj in candidates:
            obj_z = self._get_centroid(obj)[2]
            diff = obj_z - anchor_z
            
            if config.comparator == "gt":
                if diff > threshold:
                    # Score: higher is better for "above"
                    score = min(1.0, diff / 1.0)  # Normalize by 1 meter
                    results.append((obj, score))
            elif config.comparator == "lt":
                if diff < threshold:
                    score = min(1.0, abs(diff) / 1.0)
                    results.append((obj, score))
        
        return results
    
    def _filter_horizontal(
        self,
        candidates: List["SceneObject"],
        anchors: List["SceneObject"],
        config: FilterConfig,
    ) -> List[Tuple["SceneObject", float]]:
        """Filter by horizontal (X or Y axis) position."""
        anchor_center = self._compute_anchor_center(anchors)
        
        axis_idx = {"x": 0, "y": 1}.get(config.axis, 0)
        anchor_val = anchor_center[axis_idx]
        
        threshold = config.threshold or 0.0
        if self.use_loose_threshold:
            if config.comparator == "gt":
                threshold -= abs(threshold) * (config.loose_factor - 1)
            else:
                threshold += abs(threshold) * (config.loose_factor - 1)
        
        results = []
        for obj in candidates:
            obj_val = self._get_centroid(obj)[axis_idx]
            diff = obj_val - anchor_val
            
            if config.comparator == "gt":
                if diff > threshold:
                    score = min(1.0, diff / 2.0)
                    results.append((obj, score))
            elif config.comparator == "lt":
                if diff < threshold:
                    score = min(1.0, abs(diff) / 2.0)
                    results.append((obj, score))
        
        return results
    
    def _filter_distance(
        self,
        candidates: List["SceneObject"],
        anchors: List["SceneObject"],
        config: FilterConfig,
    ) -> List[Tuple["SceneObject", float]]:
        """Filter by Euclidean distance."""
        anchor_center = self._compute_anchor_center(anchors)
        
        threshold = config.threshold or 2.0
        if self.use_loose_threshold:
            threshold *= config.loose_factor
        
        results = []
        for obj in candidates:
            obj_pos = self._get_centroid(obj)
            
            # Compute distance based on axis config
            if config.axis == "xy":
                dist = float(np.linalg.norm(obj_pos[:2] - anchor_center[:2]))
            elif config.axis == "z":
                dist = abs(obj_pos[2] - anchor_center[2])
            else:  # xyz
                dist = float(np.linalg.norm(obj_pos - anchor_center))
            
            if config.comparator == "lt":
                if dist < threshold:
                    # Score: closer is better for "near"
                    score = max(0, 1 - dist / threshold)
                    results.append((obj, score))
            elif config.comparator == "gt":
                if dist > threshold:
                    # Score: farther is better for "far"
                    score = min(1.0, dist / (threshold * 2))
                    results.append((obj, score))
        
        return results
    
    def filter_by_attribute(
        self,
        candidates: List["SceneObject"],
        attribute: str,
    ) -> List["SceneObject"]:
        """
        Filter candidates by attribute (color, size, etc.).
        
        Args:
            candidates: List of candidate objects
            attribute: Attribute string (e.g., "red", "large")
            
        Returns:
            Filtered list of candidates
        """
        attr_lower = attribute.lower()
        
        # Color filtering
        if attr_lower in COLOR_KEYWORDS:
            return self._filter_by_color(candidates, attr_lower)
        
        # Size filtering
        if attr_lower in SIZE_KEYWORDS:
            order = SIZE_KEYWORDS[attr_lower]
            return self._filter_by_size(candidates, order)
        
        # No matching filter, return all
        return candidates
    
    def _filter_by_color(
        self,
        candidates: List["SceneObject"],
        color: str,
    ) -> List["SceneObject"]:
        """Filter by color attribute."""
        results = []
        for obj in candidates:
            # Check if color is mentioned in object's summary or tags
            obj_text = ""
            if hasattr(obj, 'summary') and obj.summary:
                obj_text += obj.summary.lower()
            if hasattr(obj, 'object_tag') and obj.object_tag:
                obj_text += " " + obj.object_tag.lower()
            if hasattr(obj, 'category') and obj.category:
                obj_text += " " + obj.category.lower()
            
            if color in obj_text:
                results.append(obj)
        
        # If no color match found, return all (don't filter out everything)
        return results if results else candidates
    
    def _filter_by_size(
        self,
        candidates: List["SceneObject"],
        order: str,  # "min" or "max"
    ) -> List["SceneObject"]:
        """Filter by size (returns the smallest or largest)."""
        if not candidates:
            return []
        
        # Compute size for each candidate
        sizes = []
        for obj in candidates:
            if hasattr(obj, 'bbox_3d') and obj.bbox_3d is not None:
                size = np.prod(obj.bbox_3d.size)
            elif hasattr(obj, 'n_points') and obj.n_points:
                size = obj.n_points
            else:
                size = 1.0
            sizes.append((obj, size))
        
        # Sort by size
        sizes.sort(key=lambda x: x[1], reverse=(order == "max"))
        
        # Return top candidates (for superlative, usually just the first)
        return [sizes[0][0]]


class AttributeFilter:
    """
    Filter candidates by object attributes.
    
    Supports:
    - Color: red, blue, green, etc.
    - Size: large, small, biggest, smallest
    - Material: wooden, metal, plastic, etc. (if available)
    """
    
    # Extended color vocabulary with synonyms
    COLOR_SYNONYMS = {
        "red": ["red", "crimson", "scarlet", "maroon", "ruby"],
        "blue": ["blue", "navy", "azure", "cobalt", "cyan", "teal"],
        "green": ["green", "olive", "lime", "emerald", "forest"],
        "yellow": ["yellow", "gold", "golden", "amber"],
        "orange": ["orange", "tangerine", "coral"],
        "purple": ["purple", "violet", "lavender", "magenta", "plum"],
        "pink": ["pink", "rose", "salmon", "fuchsia"],
        "brown": ["brown", "chocolate", "tan", "beige", "khaki", "coffee"],
        "black": ["black", "dark", "ebony", "charcoal"],
        "white": ["white", "ivory", "cream", "pearl"],
        "gray": ["gray", "grey", "silver", "slate"],
    }
    
    def __init__(self):
        # Build reverse lookup
        self._color_lookup = {}
        for canonical, variants in self.COLOR_SYNONYMS.items():
            for v in variants:
                self._color_lookup[v] = canonical
    
    def filter_by_color(
        self,
        candidates: List["SceneObject"],
        color: str,
    ) -> List["SceneObject"]:
        """Filter by color, using synonyms."""
        # Normalize color
        color_lower = color.lower()
        canonical_color = self._color_lookup.get(color_lower, color_lower)
        
        # Get all color variants to search for
        variants = self.COLOR_SYNONYMS.get(canonical_color, [canonical_color])
        
        results = []
        for obj in candidates:
            obj_text = self._get_searchable_text(obj)
            if any(v in obj_text for v in variants):
                results.append(obj)
        
        return results if results else candidates
    
    def filter_by_size_rank(
        self,
        candidates: List["SceneObject"],
        order: str,  # "max" for largest, "min" for smallest
        top_k: int = 1,
    ) -> List["SceneObject"]:
        """Filter by size ranking."""
        if not candidates:
            return []
        
        # Compute sizes
        with_sizes = []
        for obj in candidates:
            size = self._compute_size(obj)
            with_sizes.append((obj, size))
        
        # Sort
        with_sizes.sort(key=lambda x: x[1], reverse=(order == "max"))
        
        return [x[0] for x in with_sizes[:top_k]]
    
    def _get_searchable_text(self, obj: "SceneObject") -> str:
        """Get all searchable text from an object."""
        parts = []
        if hasattr(obj, 'summary') and obj.summary:
            parts.append(obj.summary.lower())
        if hasattr(obj, 'object_tag') and obj.object_tag:
            parts.append(obj.object_tag.lower())
        if hasattr(obj, 'category') and obj.category:
            parts.append(obj.category.lower())
        return " ".join(parts)
    
    def _compute_size(self, obj: "SceneObject") -> float:
        """Compute size metric for an object."""
        if hasattr(obj, 'bbox_3d') and obj.bbox_3d is not None:
            return float(np.prod(obj.bbox_3d.size))
        if hasattr(obj, 'n_points') and obj.n_points:
            return float(obj.n_points)
        return 1.0


# Convenience functions
def quick_filter(
    candidates: List["SceneObject"],
    anchors: List["SceneObject"],
    relation: str,
) -> List["SceneObject"]:
    """Convenience function for quick filtering."""
    filters = QuickFilters()
    return filters.filter_candidates(candidates, anchors, relation)


def has_quick_filter(relation: str) -> bool:
    """Check if a quick filter exists for a relation."""
    filters = QuickFilters()
    return filters.has_filter(relation)


def get_supported_quick_filters() -> List[str]:
    """Get list of relations with quick filters."""
    return list(QUICK_FILTER_CONFIGS.keys())
