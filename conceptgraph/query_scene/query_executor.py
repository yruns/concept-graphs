"""
Query Executor for Nested Spatial Queries.

This module implements a recursive query executor that evaluates
GroundingQuery structures against a set of scene objects.

The execution follows a bottom-up approach:
1. Start from the innermost (leaf) nodes
2. Evaluate spatial constraints to filter candidates
3. Apply select constraints to choose final results
4. Propagate results up to parent nodes

Usage:
    executor = QueryExecutor(objects, relation_checker)
    results = executor.execute(grounding_query)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from loguru import logger

from .query_structures import (
    GroundingQuery,
    QueryNode,
    SpatialConstraint,
    SelectConstraint,
    ConstraintType,
)
from .spatial_relations import SpatialRelationChecker, RelationResult
from .quick_filters import QuickFilters, AttributeFilter, has_quick_filter

if TYPE_CHECKING:
    from .keyframe_selector import SceneObject


@dataclass
class ExecutionResult:
    """Result of executing a query node."""
    
    node_id: str
    matched_objects: List["SceneObject"]
    scores: Dict[int, float] = field(default_factory=dict)  # obj_id -> score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_empty(self) -> bool:
        return len(self.matched_objects) == 0
    
    @property
    def best_object(self) -> Optional["SceneObject"]:
        if not self.matched_objects:
            return None
        if self.scores:
            best_id = max(self.scores.keys(), key=lambda x: self.scores[x])
            for obj in self.matched_objects:
                if obj.obj_id == best_id:
                    return obj
        return self.matched_objects[0]


class QueryExecutor:
    """
    Recursive executor for nested spatial queries.
    
    Evaluates GroundingQuery structures by:
    1. Finding objects matching each node's category
    2. Applying spatial constraints as filters
    3. Applying select constraints for final selection
    
    Attributes:
        objects: List of scene objects to search
        relation_checker: Spatial relation checker instance
        category_index: Index mapping categories to objects
    """
    
    def __init__(
        self,
        objects: List["SceneObject"],
        relation_checker: Optional[SpatialRelationChecker] = None,
        clip_features: Optional[np.ndarray] = None,
        clip_encoder: Optional[Any] = None,
        use_quick_filters: bool = True,
    ):
        """
        Initialize the query executor.
        
        Args:
            objects: List of scene objects
            relation_checker: Optional pre-configured relation checker
            clip_features: Optional pre-computed CLIP features for objects
            clip_encoder: Optional CLIP text encoder for semantic matching
            use_quick_filters: Whether to use quick filters for pre-filtering
        """
        self.objects = objects
        self.relation_checker = relation_checker or SpatialRelationChecker()
        self.clip_features = clip_features
        self.clip_encoder = clip_encoder
        self.use_quick_filters = use_quick_filters
        
        # Quick filters for fast pre-filtering
        self._quick_filters = QuickFilters() if use_quick_filters else None
        self._attribute_filter = AttributeFilter() if use_quick_filters else None
        
        # Build category index
        self._category_index: Dict[str, List["SceneObject"]] = {}
        for obj in objects:
            category = self._get_category(obj).lower()
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(obj)
        
        # Execution cache for memoization
        self._cache: Dict[str, ExecutionResult] = {}
    
    def _get_category(self, obj: "SceneObject") -> str:
        """Get the category string for an object."""
        if hasattr(obj, 'object_tag') and obj.object_tag:
            return obj.object_tag
        return obj.category
    
    def _get_centroid(self, obj: "SceneObject") -> np.ndarray:
        """Get object centroid."""
        if hasattr(obj, 'centroid') and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def execute(self, query: GroundingQuery) -> ExecutionResult:
        """
        Execute a grounding query.
        
        Args:
            query: GroundingQuery to execute
            
        Returns:
            ExecutionResult with matched objects
        """
        logger.info(f"[QueryExecutor] Executing query: '{query.raw_query}'")
        
        # Clear cache for new query
        self._cache.clear()
        
        # Execute from root
        result = self._execute_node(query.root)
        
        # If expect_unique, keep only the best result
        if query.expect_unique and len(result.matched_objects) > 1:
            best = result.best_object
            if best:
                result = ExecutionResult(
                    node_id=result.node_id,
                    matched_objects=[best],
                    scores={best.obj_id: result.scores.get(best.obj_id, 1.0)},
                    metadata=result.metadata
                )
        
        logger.info(f"[QueryExecutor] Found {len(result.matched_objects)} objects")
        return result
    
    def _execute_node(self, node: QueryNode) -> ExecutionResult:
        """
        Recursively execute a query node.
        
        Execution order:
        1. Find candidates by category
        2. Filter by attributes
        3. Apply spatial constraints (filter phase)
        4. Apply select constraint (selection phase)
        """
        # Check cache
        if node.node_id and node.node_id in self._cache:
            return self._cache[node.node_id]
        
        logger.debug(f"[QueryExecutor] Executing node: category='{node.category}'")
        
        # Step 1: Find candidates by category
        candidates = self._find_by_category(node.category)
        logger.debug(f"[QueryExecutor] Found {len(candidates)} candidates for '{node.category}'")
        
        if not candidates:
            result = ExecutionResult(node_id=node.node_id, matched_objects=[])
            self._cache[node.node_id] = result
            return result
        
        # Step 2: Filter by attributes
        if node.attributes:
            candidates = self._filter_by_attributes(candidates, node.attributes)
            logger.debug(f"[QueryExecutor] After attribute filter: {len(candidates)} candidates")
        
        # Step 3: Apply spatial constraints (AND logic)
        scores = {obj.obj_id: 1.0 for obj in candidates}
        
        for constraint in node.spatial_constraints:
            candidates, constraint_scores = self._apply_spatial_constraint(
                candidates, constraint
            )
            # Combine scores
            for obj_id, score in constraint_scores.items():
                if obj_id in scores:
                    scores[obj_id] *= score
            
            logger.debug(
                f"[QueryExecutor] After '{constraint.relation}' constraint: "
                f"{len(candidates)} candidates"
            )
            
            if not candidates:
                break
        
        # Step 4: Apply select constraint
        if candidates and node.select_constraint:
            candidates, scores = self._apply_select_constraint(
                candidates, scores, node.select_constraint
            )
            logger.debug(
                f"[QueryExecutor] After select constraint: {len(candidates)} candidates"
            )
        
        result = ExecutionResult(
            node_id=node.node_id,
            matched_objects=candidates,
            scores=scores,
            metadata={"category": node.category}
        )
        
        if node.node_id:
            self._cache[node.node_id] = result
        
        return result
    
    def _find_by_category(self, category: str) -> List["SceneObject"]:
        """
        Find objects matching a category.
        
        Uses exact match first, then substring match, then CLIP similarity.
        """
        category_lower = category.lower()
        
        # Exact match
        if category_lower in self._category_index:
            return list(self._category_index[category_lower])
        
        # Substring match
        matches = []
        for cat, objs in self._category_index.items():
            if category_lower in cat or cat in category_lower:
                matches.extend(objs)
        
        if matches:
            return matches
        
        # Try common synonyms
        synonyms = {
            "pillow": ["throw_pillow", "cushion"],
            "couch": ["sofa"],
            "sofa": ["couch"],
            "lamp": ["table_lamp", "floor_lamp"],
            "table": ["side_table", "coffee_table", "dining_table"],
            "chair": ["armchair", "dining_chair"],
            "tv": ["television"],
            "television": ["tv"],
        }
        
        if category_lower in synonyms:
            for syn in synonyms[category_lower]:
                if syn in self._category_index:
                    matches.extend(self._category_index[syn])
        
        if matches:
            return matches
        
        # CLIP similarity fallback (if available)
        if self.clip_features is not None and self.clip_encoder is not None:
            return self._find_by_clip_similarity(category)
        
        # Last resort: return all objects
        logger.warning(f"[QueryExecutor] No exact match for '{category}', returning all objects")
        return []
    
    def _find_by_clip_similarity(
        self, category: str, top_k: int = 10, min_similarity: float = 0.2
    ) -> List["SceneObject"]:
        """Find objects by CLIP text-image similarity."""
        try:
            # Encode text
            text_feature = self.clip_encoder(category)
            if text_feature is None:
                return []
            
            # Compute similarities
            similarities = self.clip_features @ text_feature
            top_indices = np.argsort(-similarities)[:top_k]
            
            matches = [
                self.objects[i]
                for i in top_indices
                if similarities[i] > min_similarity
            ]
            
            if matches:
                logger.info(
                    f"[QueryExecutor] CLIP matched '{category}' -> "
                    f"{[(self._get_category(m), f'{similarities[self.objects.index(m)]:.2f}') for m in matches[:3]]}"
                )
            
            return matches
        except Exception as e:
            logger.warning(f"[QueryExecutor] CLIP matching failed: {e}")
            return []
    
    def _filter_by_attributes(
        self,
        candidates: List["SceneObject"],
        attributes: List[str],
    ) -> List["SceneObject"]:
        """Filter candidates by attributes (color, size, etc.).
        
        Uses AttributeFilter for color and size filtering.
        """
        if not attributes or not self._attribute_filter:
            return candidates
        
        filtered = candidates
        for attr in attributes:
            attr_lower = attr.lower()
            
            # Try color filtering
            if self._attribute_filter._color_lookup.get(attr_lower):
                filtered = self._attribute_filter.filter_by_color(filtered, attr_lower)
                logger.debug(f"[QueryExecutor] After color filter '{attr}': {len(filtered)} candidates")
            
            # Note: size filtering (largest, smallest) is handled by select_constraint
        
        return filtered
    
    def _apply_spatial_constraint(
        self,
        candidates: List["SceneObject"],
        constraint: SpatialConstraint,
    ) -> Tuple[List["SceneObject"], Dict[int, float]]:
        """
        Apply a spatial constraint to filter candidates.
        
        Uses a two-phase approach:
        1. Quick filter: Fast pre-filtering using simple coordinate comparisons
        2. Full check: Accurate spatial relation checking for remaining candidates
        
        Args:
            candidates: Current candidate objects
            constraint: Spatial constraint to apply
            
        Returns:
            Tuple of (filtered candidates, scores dict)
        """
        # Execute anchor nodes to get reference objects
        anchor_objects = []
        for anchor_node in constraint.anchors:
            anchor_result = self._execute_node(anchor_node)
            anchor_objects.extend(anchor_result.matched_objects)
        
        if not anchor_objects:
            logger.warning(
                f"[QueryExecutor] No anchor objects found for relation '{constraint.relation}'"
            )
            return candidates, {obj.obj_id: 1.0 for obj in candidates}
        
        # Phase 1: Quick filter (if available)
        pre_filtered = candidates
        if self._quick_filters and self._quick_filters.has_filter(constraint.relation):
            pre_filtered = self._quick_filters.filter_candidates(
                candidates, anchor_objects, constraint.relation
            )
            logger.debug(
                f"[QueryExecutor] Quick filter '{constraint.relation}': "
                f"{len(candidates)} -> {len(pre_filtered)} candidates"
            )
            
            # If quick filter eliminated all candidates, fall back to full list
            if not pre_filtered:
                logger.warning(
                    f"[QueryExecutor] Quick filter eliminated all candidates, using full list"
                )
                pre_filtered = candidates
        
        # Phase 2: Full spatial relation check
        filtered = []
        scores = {}
        
        for cand in pre_filtered:
            best_score = 0.0
            satisfies_any = False
            
            # For "between", we need to pass both anchors
            if constraint.relation.lower() == "between" and len(anchor_objects) >= 2:
                result = self.relation_checker.check(
                    cand, anchor_objects[:2], constraint.relation
                )
                if result.satisfies:
                    satisfies_any = True
                    best_score = result.score
            else:
                # For other relations, check against each anchor
                for anchor in anchor_objects:
                    result = self.relation_checker.check(
                        cand, anchor, constraint.relation
                    )
                    if result.satisfies:
                        satisfies_any = True
                        best_score = max(best_score, result.score)
            
            if satisfies_any:
                filtered.append(cand)
                scores[cand.obj_id] = best_score
        
        return filtered, scores
    
    def _apply_select_constraint(
        self,
        candidates: List["SceneObject"],
        scores: Dict[int, float],
        constraint: SelectConstraint,
    ) -> Tuple[List["SceneObject"], Dict[int, float]]:
        """
        Apply a select constraint (superlative/ordinal).
        
        Args:
            candidates: Current candidate objects
            scores: Current scores
            constraint: Select constraint to apply
            
        Returns:
            Tuple of (selected candidates, updated scores)
        """
        if not candidates:
            return [], {}
        
        if constraint.constraint_type == ConstraintType.SUPERLATIVE:
            return self._apply_superlative(candidates, scores, constraint)
        elif constraint.constraint_type == ConstraintType.ORDINAL:
            return self._apply_ordinal(candidates, scores, constraint)
        elif constraint.constraint_type == ConstraintType.COMPARATIVE:
            return self._apply_comparative(candidates, scores, constraint)
        else:
            return candidates, scores
    
    def _apply_superlative(
        self,
        candidates: List["SceneObject"],
        scores: Dict[int, float],
        constraint: SelectConstraint,
    ) -> Tuple[List["SceneObject"], Dict[int, float]]:
        """Apply superlative constraint (nearest, largest, etc.)."""
        metric = constraint.metric.lower()
        order = constraint.order.lower()
        
        # Get reference objects if needed
        ref_objects = []
        if constraint.reference:
            ref_result = self._execute_node(constraint.reference)
            ref_objects = ref_result.matched_objects
        
        # Compute metric values for each candidate
        values = []
        for cand in candidates:
            if metric == "distance" and ref_objects:
                # Distance to nearest reference object
                cand_pos = self._get_centroid(cand)
                min_dist = float('inf')
                for ref in ref_objects:
                    ref_pos = self._get_centroid(ref)
                    dist = float(np.linalg.norm(cand_pos - ref_pos))
                    min_dist = min(min_dist, dist)
                values.append(min_dist)
            
            elif metric == "size":
                # Use bounding box volume (not point count)
                if hasattr(cand, 'bbox_3d') and cand.bbox_3d is not None:
                    size = np.prod(cand.bbox_3d.size)
                elif hasattr(cand, 'pcd_np') and cand.pcd_np is not None and len(cand.pcd_np) > 0:
                    # Compute volume from point cloud bounding box
                    pts = np.asarray(cand.pcd_np)
                    bbox_size = pts.max(axis=0) - pts.min(axis=0)
                    size = np.prod(bbox_size)
                elif hasattr(cand, 'point_cloud') and cand.point_cloud is not None:
                    pts = np.asarray(cand.point_cloud)
                    if len(pts) > 0:
                        bbox_size = pts.max(axis=0) - pts.min(axis=0)
                        size = np.prod(bbox_size)
                    else:
                        size = 0.0
                else:
                    size = 0.0
                values.append(size)
            
            elif metric == "height":
                # Z coordinate
                pos = self._get_centroid(cand)
                values.append(pos[2])
            
            elif metric in ["x_position", "x"]:
                pos = self._get_centroid(cand)
                values.append(pos[0])
            
            elif metric in ["y_position", "y"]:
                pos = self._get_centroid(cand)
                values.append(pos[1])
            
            else:
                # Default: use existing score
                values.append(scores.get(cand.obj_id, 0.0))
        
        # Sort and select
        indexed = list(zip(candidates, values))
        
        if order == "min":
            indexed.sort(key=lambda x: x[1])
        else:  # max
            indexed.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the best
        best_cand, best_value = indexed[0]
        new_scores = {best_cand.obj_id: 1.0}
        
        logger.debug(
            f"[QueryExecutor] Superlative '{order} {metric}': "
            f"selected {self._get_category(best_cand)} with value {best_value:.3f}"
        )
        
        return [best_cand], new_scores
    
    def _apply_ordinal(
        self,
        candidates: List["SceneObject"],
        scores: Dict[int, float],
        constraint: SelectConstraint,
    ) -> Tuple[List["SceneObject"], Dict[int, float]]:
        """Apply ordinal constraint (first, second, etc.)."""
        if constraint.position is None:
            return candidates, scores
        
        position = constraint.position  # 1-indexed
        metric = constraint.metric.lower()
        order = constraint.order.lower()
        
        # Sort candidates by metric
        def get_value(cand):
            pos = self._get_centroid(cand)
            if metric in ["x_position", "x"]:
                return pos[0]
            elif metric in ["y_position", "y"]:
                return pos[1]
            elif metric == "height":
                return pos[2]
            else:
                return scores.get(cand.obj_id, 0.0)
        
        sorted_candidates = sorted(candidates, key=get_value, reverse=(order == "desc"))
        
        if position <= 0 or position > len(sorted_candidates):
            logger.warning(
                f"[QueryExecutor] Ordinal position {position} out of range "
                f"(have {len(sorted_candidates)} candidates)"
            )
            return [], {}
        
        selected = sorted_candidates[position - 1]
        return [selected], {selected.obj_id: 1.0}
    
    def _apply_comparative(
        self,
        candidates: List["SceneObject"],
        scores: Dict[int, float],
        constraint: SelectConstraint,
    ) -> Tuple[List["SceneObject"], Dict[int, float]]:
        """Apply comparative constraint (closer than, larger than)."""
        # Comparative requires a reference object to compare against
        # For now, this is similar to superlative but keeps multiple results
        
        # Get reference objects
        ref_objects = []
        if constraint.reference:
            ref_result = self._execute_node(constraint.reference)
            ref_objects = ref_result.matched_objects
        
        if not ref_objects:
            return candidates, scores
        
        metric = constraint.metric.lower()
        order = constraint.order.lower()
        
        # Filter candidates that satisfy the comparison
        # This would require knowing what to compare against
        # For now, return all candidates
        
        return candidates, scores


# Convenience function
def execute_query(
    query: GroundingQuery,
    objects: List["SceneObject"],
    relation_checker: Optional[SpatialRelationChecker] = None,
) -> ExecutionResult:
    """
    Execute a grounding query against scene objects.
    
    Args:
        query: GroundingQuery to execute
        objects: List of scene objects
        relation_checker: Optional spatial relation checker
        
    Returns:
        ExecutionResult with matched objects
    """
    executor = QueryExecutor(objects, relation_checker)
    return executor.execute(query)
