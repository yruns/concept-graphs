"""
Spatial Relation Checker.

This module implements geometric checks for common spatial relations
between 3D objects in a scene. It supports approximately 15 core relations
that cover the vast majority of natural language spatial queries.

Supported Relations:
- Vertical: on_top_of, above, below
- Horizontal: next_to, near, beside
- Directional: in_front_of, behind, left_of, right_of
- Containment: inside
- Multi-object: between

Usage:
    checker = SpatialRelationChecker()
    satisfies, score = checker.check(target_obj, anchor_obj, "on")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .keyframe_selector import SceneObject


# Relation aliases mapping natural language variants to canonical names
RELATION_ALIASES: Dict[str, str] = {
    # on_top_of variants
    "on": "on_top_of",
    "upon": "on_top_of",
    "atop": "on_top_of",
    "on_top": "on_top_of",
    "on_top_of": "on_top_of",
    "resting_on": "on_top_of",
    
    # above variants
    "above": "above",
    "over": "above",
    "higher_than": "above",
    
    # below variants
    "below": "below",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "lower_than": "below",
    
    # next_to / near variants
    "next_to": "next_to",
    "beside": "next_to",
    "near": "near",
    "by": "near",
    "close_to": "near",
    "adjacent_to": "next_to",
    "next": "next_to",
    
    # in_front_of variants
    "in_front_of": "in_front_of",
    "front": "in_front_of",
    "facing": "in_front_of",
    "before": "in_front_of",
    
    # behind variants
    "behind": "behind",
    "back": "behind",
    "back_of": "behind",
    "in_back_of": "behind",
    
    # left_of variants
    "left_of": "left_of",
    "left": "left_of",
    "to_the_left_of": "left_of",
    "on_the_left_of": "left_of",
    
    # right_of variants
    "right_of": "right_of",
    "right": "right_of",
    "to_the_right_of": "right_of",
    "on_the_right_of": "right_of",
    
    # inside variants
    "inside": "inside",
    "in": "inside",
    "within": "inside",
    "contained_in": "inside",
    
    # between
    "between": "between",
    "in_between": "between",
    
    # leaning/against
    "against": "against",
    "leaning_on": "against",
    "leaning_against": "against",
    
    # around
    "around": "around",
    "surrounding": "around",
}


@dataclass
class RelationResult:
    """Result of a spatial relation check."""
    
    satisfies: bool
    score: float  # 0.0 to 1.0, higher means stronger relation
    details: Dict[str, float] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SpatialRelationChecker:
    """
    Checks spatial relations between 3D objects.
    
    Supports approximately 15 core spatial relations that cover
    the vast majority of natural language spatial queries.
    
    Attributes:
        default_thresholds: Default distance/angle thresholds for each relation
    """
    
    # Default thresholds (in meters)
    DEFAULT_THRESHOLDS = {
        "on_top_of": {"max_horizontal": 0.5, "min_vertical": 0.0, "max_vertical": 1.0},
        "above": {"max_horizontal": 1.0, "min_vertical": 0.1},
        "below": {"max_horizontal": 1.0, "max_vertical": -0.1},
        "next_to": {"max_distance": 1.0},
        "near": {"max_distance": 2.0},
        "inside": {"margin": 0.1},
        "between": {"max_distance_ratio": 0.3},  # max distance from line / line length
    }
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize the spatial relation checker.
        
        Args:
            thresholds: Optional custom thresholds to override defaults
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key].update(value)
                else:
                    self.thresholds[key] = value
    
    def check(
        self,
        target: "SceneObject",
        anchor: Union["SceneObject", List["SceneObject"]],
        relation: str,
    ) -> RelationResult:
        """
        Check if target satisfies the spatial relation with anchor.
        
        Args:
            target: The target object
            anchor: The reference object(s). List for "between" relation.
            relation: Spatial relation (e.g., "on", "near", "between")
            
        Returns:
            RelationResult with satisfies (bool) and score (float)
        """
        # Normalize relation name
        canonical = RELATION_ALIASES.get(relation.lower().replace(" ", "_"), relation.lower())
        
        # Get the appropriate check function
        check_func = getattr(self, f"is_{canonical}", None)
        
        if check_func is None:
            # Unknown relation - fall back to proximity check
            return self.is_near(target, anchor)
        
        # Handle multi-anchor relations
        if canonical == "between":
            if isinstance(anchor, list) and len(anchor) >= 2:
                return check_func(target, anchor[0], anchor[1])
            else:
                return RelationResult(satisfies=False, score=0.0)
        
        # Single anchor relations
        if isinstance(anchor, list):
            anchor = anchor[0] if anchor else None
        
        if anchor is None:
            return RelationResult(satisfies=False, score=0.0)
        
        return check_func(target, anchor)
    
    def _get_centroid(self, obj: "SceneObject") -> np.ndarray:
        """Get object centroid as numpy array."""
        if hasattr(obj, 'centroid') and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def _get_bbox(self, obj: "SceneObject") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get object bounding box as (min_point, max_point)."""
        if hasattr(obj, 'bbox_3d') and obj.bbox_3d is not None:
            return (obj.bbox_3d.min_point, obj.bbox_3d.max_point)
        return None
    
    # ========== Vertical Relations ==========
    
    def is_on_top_of(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is on top of anchor.
        
        Criteria:
        - Target is above anchor (positive z difference)
        - Horizontal distance is small (within anchor's footprint)
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        thres = self.thresholds["on_top_of"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]
        
        # Must be above
        if vertical_diff < thres["min_vertical"]:
            return RelationResult(satisfies=False, score=0.0)
        
        # Must be within horizontal threshold
        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)
        
        # Score: higher when closer horizontally and moderately above
        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, vertical_diff / 0.3)  # Peak at 0.3m above
        score = 0.6 * h_score + 0.4 * v_score
        
        return RelationResult(
            satisfies=True,
            score=float(score),
            details={"horizontal_dist": horizontal_dist, "vertical_diff": vertical_diff}
        )
    
    def is_above(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is above anchor (not necessarily touching).
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        thres = self.thresholds["above"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]
        
        if vertical_diff < thres["min_vertical"]:
            return RelationResult(satisfies=False, score=0.0)
        
        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)
        
        # Score based on how directly above
        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, vertical_diff / 1.0)
        score = 0.5 * h_score + 0.5 * v_score
        
        return RelationResult(satisfies=True, score=float(score))
    
    def is_below(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is below anchor.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        thres = self.thresholds["below"]
        horizontal_dist = np.linalg.norm(diff[:2])
        vertical_diff = diff[2]
        
        if vertical_diff > thres["max_vertical"]:
            return RelationResult(satisfies=False, score=0.0)
        
        if horizontal_dist > thres["max_horizontal"]:
            return RelationResult(satisfies=False, score=0.0)
        
        h_score = max(0, 1 - horizontal_dist / thres["max_horizontal"])
        v_score = min(1.0, abs(vertical_diff) / 1.0)
        score = 0.5 * h_score + 0.5 * v_score
        
        return RelationResult(satisfies=True, score=float(score))
    
    # ========== Horizontal Distance Relations ==========
    
    def is_next_to(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is next to anchor (close proximity).
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        
        thres = self.thresholds["next_to"]
        distance = float(np.linalg.norm(t_pos - a_pos))
        
        if distance > thres["max_distance"]:
            return RelationResult(satisfies=False, score=0.0)
        
        score = max(0, 1 - distance / thres["max_distance"])
        return RelationResult(satisfies=True, score=score, details={"distance": distance})
    
    def is_near(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is near anchor (looser than next_to).
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        
        thres = self.thresholds["near"]
        distance = float(np.linalg.norm(t_pos - a_pos))
        
        if distance > thres["max_distance"]:
            return RelationResult(satisfies=False, score=0.0)
        
        score = max(0, 1 - distance / thres["max_distance"])
        return RelationResult(satisfies=True, score=score, details={"distance": distance})
    
    # ========== Directional Relations ==========
    # Note: These require a reference direction. We use the global coordinate system.
    # Typically: +X = right, +Y = forward, +Z = up (may need adjustment per scene)
    
    def is_in_front_of(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is in front of anchor.
        
        Uses +Y as the forward direction (scene-dependent).
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        # Target should be in positive Y direction from anchor
        if diff[1] <= 0:
            return RelationResult(satisfies=False, score=0.0)
        
        # Also check horizontal distance isn't too far
        lateral_dist = abs(diff[0])
        forward_dist = diff[1]
        
        if lateral_dist > forward_dist:
            # More to the side than in front
            return RelationResult(satisfies=False, score=0.0)
        
        score = min(1.0, forward_dist / 2.0) * max(0, 1 - lateral_dist / forward_dist)
        return RelationResult(satisfies=True, score=float(score))
    
    def is_behind(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is behind anchor.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        if diff[1] >= 0:
            return RelationResult(satisfies=False, score=0.0)
        
        lateral_dist = abs(diff[0])
        backward_dist = abs(diff[1])
        
        if lateral_dist > backward_dist:
            return RelationResult(satisfies=False, score=0.0)
        
        score = min(1.0, backward_dist / 2.0) * max(0, 1 - lateral_dist / backward_dist)
        return RelationResult(satisfies=True, score=float(score))
    
    def is_left_of(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is to the left of anchor.
        
        Uses -X as the left direction.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        if diff[0] >= 0:
            return RelationResult(satisfies=False, score=0.0)
        
        lateral_dist = abs(diff[0])
        forward_dist = abs(diff[1])
        
        if forward_dist > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)
        
        score = min(1.0, lateral_dist / 2.0)
        return RelationResult(satisfies=True, score=float(score))
    
    def is_right_of(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is to the right of anchor.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        diff = t_pos - a_pos
        
        if diff[0] <= 0:
            return RelationResult(satisfies=False, score=0.0)
        
        lateral_dist = abs(diff[0])
        forward_dist = abs(diff[1])
        
        if forward_dist > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)
        
        score = min(1.0, lateral_dist / 2.0)
        return RelationResult(satisfies=True, score=float(score))
    
    # ========== Containment Relations ==========
    
    def is_inside(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is inside anchor.
        
        Requires bounding box information for accurate check.
        Falls back to proximity if no bbox available.
        """
        t_pos = self._get_centroid(target)
        a_bbox = self._get_bbox(anchor)
        
        if a_bbox is None:
            # Fall back to proximity check
            return self.is_near(target, anchor)
        
        a_min, a_max = a_bbox
        margin = self.thresholds["inside"]["margin"]
        
        # Check if target centroid is within anchor bbox (with margin)
        inside = np.all(t_pos >= a_min - margin) and np.all(t_pos <= a_max + margin)
        
        if not inside:
            return RelationResult(satisfies=False, score=0.0)
        
        # Score based on how centered within the bbox
        a_center = (a_min + a_max) / 2
        a_size = a_max - a_min
        normalized_dist = np.abs(t_pos - a_center) / (a_size / 2 + 1e-6)
        score = float(1 - np.mean(normalized_dist))
        
        return RelationResult(satisfies=True, score=max(0, score))
    
    # ========== Multi-Object Relations ==========
    
    def is_between(
        self,
        target: "SceneObject",
        anchor1: "SceneObject",
        anchor2: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is between anchor1 and anchor2.
        """
        t_pos = self._get_centroid(target)
        a1_pos = self._get_centroid(anchor1)
        a2_pos = self._get_centroid(anchor2)
        
        # Vector from anchor1 to anchor2
        line_vec = a2_pos - a1_pos
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return RelationResult(satisfies=False, score=0.0)
        
        line_dir = line_vec / line_len
        
        # Project target onto the line
        t_vec = t_pos - a1_pos
        proj_len = np.dot(t_vec, line_dir)
        
        # Check if projection is between the two anchors
        if proj_len < 0 or proj_len > line_len:
            return RelationResult(satisfies=False, score=0.0)
        
        # Distance from target to the line
        proj_point = a1_pos + proj_len * line_dir
        dist_to_line = np.linalg.norm(t_pos - proj_point)
        
        # Check if close enough to the line
        thres = self.thresholds["between"]
        max_dist = line_len * thres["max_distance_ratio"]
        
        if dist_to_line > max_dist:
            return RelationResult(satisfies=False, score=0.0)
        
        # Score: higher when more centered and closer to line
        center_score = 1 - abs(proj_len - line_len / 2) / (line_len / 2)
        line_score = 1 - dist_to_line / max_dist if max_dist > 0 else 1.0
        score = 0.5 * center_score + 0.5 * line_score
        
        return RelationResult(
            satisfies=True,
            score=float(score),
            details={"proj_len": proj_len, "line_len": line_len, "dist_to_line": dist_to_line}
        )
    
    # ========== Contact Relations ==========
    
    def is_against(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is against (leaning on) anchor.
        
        Similar to next_to but with stricter distance threshold.
        """
        t_pos = self._get_centroid(target)
        a_pos = self._get_centroid(anchor)
        
        distance = float(np.linalg.norm(t_pos - a_pos))
        
        # Very close proximity for "against"
        max_dist = 0.5
        if distance > max_dist:
            return RelationResult(satisfies=False, score=0.0)
        
        score = max(0, 1 - distance / max_dist)
        return RelationResult(satisfies=True, score=score)
    
    def is_around(
        self,
        target: "SceneObject",
        anchor: "SceneObject",
    ) -> RelationResult:
        """
        Check if target is around anchor.
        
        This is essentially the same as "near" but can be used for
        collections of objects surrounding something.
        """
        return self.is_near(target, anchor)


# Convenience function
def check_relation(
    target: "SceneObject",
    anchor: Union["SceneObject", List["SceneObject"]],
    relation: str,
    checker: Optional[SpatialRelationChecker] = None,
) -> Tuple[bool, float]:
    """
    Check if target satisfies the spatial relation with anchor.
    
    Args:
        target: Target object
        anchor: Reference object(s)
        relation: Spatial relation string
        checker: Optional pre-initialized checker
        
    Returns:
        Tuple of (satisfies, score)
    """
    if checker is None:
        checker = SpatialRelationChecker()
    
    result = checker.check(target, anchor, relation)
    return result.satisfies, result.score


def get_canonical_relation(relation: str) -> str:
    """Get the canonical relation name for a given relation string."""
    return RELATION_ALIASES.get(relation.lower().replace(" ", "_"), relation.lower())
