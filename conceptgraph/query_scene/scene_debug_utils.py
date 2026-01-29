"""
Scene Debug Utilities for inspecting spatial relations between objects.
"""

from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .spatial_relations import SpatialRelationChecker, RelationResult

if TYPE_CHECKING:
    from .keyframe_selector import SceneObject


class SceneDebugger:
    """Debug utilities for inspecting scene objects and their spatial relations."""
    
    def __init__(self, objects, relation_checker=None):
        self.objects = objects
        self.checker = relation_checker or SpatialRelationChecker()
        self._category_index = {}
        for obj in objects:
            cat = self._get_category(obj).lower()
            if cat not in self._category_index:
                self._category_index[cat] = []
            self._category_index[cat].append(obj)
    
    def _get_category(self, obj):
        if hasattr(obj, "object_tag") and obj.object_tag:
            return obj.object_tag
        return getattr(obj, "category", "unknown")
    
    def _get_centroid(self, obj):
        if hasattr(obj, "centroid") and obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def _compute_distance(self, obj1, obj2):
        c1 = self._get_centroid(obj1)
        c2 = self._get_centroid(obj2)
        return float(np.linalg.norm(c1 - c2))
    
    def print_category_summary(self):
        """Print summary of object categories in the scene."""
        print("Scene Category Summary")
        print("=" * 50)
        counts = Counter()
        for cat, objs in self._category_index.items():
            counts[cat] = len(objs)
        for cat, count in counts.most_common():
            obj_ids = [o.obj_id for o in self._category_index[cat]]
            print(f"  {cat}: {count} objects (ids: {obj_ids})")
        print(f"Total: {len(self.objects)} objects")
    
    def check_relation(self, category1, category2, relation="near", verbose=True):
        """Check spatial relation between all pairs of two categories."""
        objs1 = self._category_index.get(category1.lower(), [])
        objs2 = self._category_index.get(category2.lower(), [])
        if not objs1 or not objs2:
            print(f"Warning: Missing category")
            return []
        if verbose:
            print(f"Checking: {category1} {relation} {category2}")
        satisfying_pairs = []
        for obj1 in objs1:
            for obj2 in objs2:
                if obj1.obj_id == obj2.obj_id:
                    continue
                result = self.checker.check(obj1, obj2, relation)
                distance = self._compute_distance(obj1, obj2)
                status = "Y" if result.satisfies else "N"
                if verbose:
                    print(f"  {obj1.obj_id} <-> {obj2.obj_id}: {status} dist={distance:.2f}m")
                if result.satisfies:
                    satisfying_pairs.append((obj1, obj2, result))
        if verbose:
            print(f"Found {len(satisfying_pairs)} satisfying pairs")
        return satisfying_pairs
    
    def find_pairs(self, relation="near", max_pairs=20, verbose=True):
        """Find all pairs of objects satisfying a relation."""
        pairs = []
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                result = self.checker.check(obj1, obj2, relation)
                if result.satisfies:
                    distance = self._compute_distance(obj1, obj2)
                    pairs.append((obj1, obj2, distance))
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:max_pairs]
        if verbose:
            print(f"Pairs satisfying {relation}:")
            for obj1, obj2, dist in pairs:
                cat1 = self._get_category(obj1)
                cat2 = self._get_category(obj2)
                print(f"  {cat1}({obj1.obj_id}) <-> {cat2}({obj2.obj_id}): {dist:.2f}m")
        return pairs
    
    def get_object_by_id(self, obj_id):
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None
    
    def get_objects_by_category(self, category):
        return self._category_index.get(category.lower(), [])


def debug_spatial_relations(objects, category1, category2, relation="near"):
    """Convenience function to check spatial relations between categories."""
    debugger = SceneDebugger(objects)
    debugger.check_relation(category1, category2, relation)
