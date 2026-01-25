#!/usr/bin/env python3
"""
Focused debug test for: "the lamp on the table near the sofa".

This script replays the parsed structure and prints step-by-step execution
details (candidates, anchors, spatial checks) to explain why no results occur.
"""

import sys
from pathlib import Path
from typing import List

from loguru import logger

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from conceptgraph.query_scene.examples.e2e_query_test import load_scene_objects, apply_affordances
from conceptgraph.query_scene.query_structures import GroundingQuery, QueryNode, SpatialConstraint
from conceptgraph.query_scene.query_executor import QueryExecutor
from conceptgraph.query_scene.spatial_relations import SpatialRelationChecker


def _log_candidates(prefix: str, candidates: List, limit: int = 5) -> None:
    sample = ", ".join(f"{c.obj_id}:{getattr(c, 'object_tag', '')}" for c in candidates[:limit])
    logger.info(f"{prefix} count={len(candidates)} sample=[{sample}]")


def analyze_node(executor: QueryExecutor, node: QueryNode, label: str, depth: int = 0) -> List:
    indent = "  " * depth
    candidates = executor._find_by_category(node.category)
    logger.info(f"{indent}{label} category='{node.category}'")
    _log_candidates(f"{indent}{label} candidates", candidates)

    if node.attributes:
        candidates = executor._filter_by_attributes(candidates, node.attributes)
        _log_candidates(f"{indent}{label} after attributes {node.attributes}", candidates)

    # Apply spatial constraints
    for i, constraint in enumerate(node.spatial_constraints):
        logger.info(f"{indent}{label} constraint[{i}] relation='{constraint.relation}'")
        anchor_objects = []
        for j, anchor_node in enumerate(constraint.anchors):
            anchor_label = f"{label}.anchor[{i}][{j}]"
            anchor_candidates = analyze_node(executor, anchor_node, anchor_label, depth + 1)
            anchor_objects.extend(anchor_candidates)

        if not anchor_objects:
            logger.warning(f"{indent}{label} constraint[{i}] has NO anchors (will skip)")
        else:
            # Show relation check details for a small subset
            for cand in candidates[:5]:
                for anchor in anchor_objects[:5]:
                    rel = executor.relation_checker.check(cand, anchor, constraint.relation)
                    logger.info(
                        f"{indent}  check cand={cand.obj_id} anchor={anchor.obj_id} "
                        f"-> satisfies={rel.satisfies}, score={rel.score:.3f}, details={rel.details}"
                    )

        filtered, _ = executor._apply_spatial_constraint(candidates, constraint)
        candidates = filtered
        _log_candidates(f"{indent}{label} after '{constraint.relation}'", candidates)

    if node.select_constraint and candidates:
        candidates, _ = executor._apply_select_constraint(candidates, {c.obj_id: 1.0 for c in candidates}, node.select_constraint)
        _log_candidates(f"{indent}{label} after select {node.select_constraint.metric}", candidates)

    return candidates


def build_query(root_category: str, table_category: str, sofa_category: str) -> GroundingQuery:
    return GroundingQuery(
        raw_query="the lamp on the table near the sofa",
        root=QueryNode(
            category=root_category,
            spatial_constraints=[
                SpatialConstraint(
                    relation="on",
                    anchors=[
                        QueryNode(
                            category=table_category,
                            spatial_constraints=[
                                SpatialConstraint(
                                    relation="near",
                                    anchors=[QueryNode(category=sofa_category)],
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
        expect_unique=True,
    )


def main():
    scene_path = project_root / "room0"
    objects, _ = load_scene_objects(str(scene_path))
    apply_affordances(objects, scene_path)

    executor = QueryExecutor(
        objects=objects,
        relation_checker=SpatialRelationChecker(),
        use_quick_filters=True,
    )

    # Match the parsed structure seen in the log
    query = build_query(root_category="lamp", table_category="side_table", sofa_category="sofa")

    logger.info("=" * 70)
    logger.info("Debugging case: lamp on the table near the sofa")
    logger.info("=" * 70)
    analyze_node(executor, query.root, "root", depth=0)

    result = executor.execute(query)
    logger.info("=" * 70)
    logger.info(f"Final matched objects: {len(result.matched_objects)}")
    for obj in result.matched_objects:
        logger.info(f"  - {obj.object_tag} (id={obj.obj_id})")


if __name__ == "__main__":
    main()
