#!/usr/bin/env python3
"""
End-to-End Query Test with Step-by-Step Visualization.

Outputs for each query:
- query_name/
  ├── 00_all_candidates.ply          (所有候选物体 - 白色)
  ├── 01_category_filtered.ply       (类别过滤后 - 蓝色)
  ├── 02_spatial_filtered.ply        (空间过滤后 - 黄色)
  ├── 03_final_result.ply            (最终结果 - 红色高亮)
  ├── filtering_process.ply          (全过程合并 - 多色)
  └── keyframes/
      └── *.jpg
"""

import sys
import json
import gzip
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from conceptgraph.query_scene.keyframe_selector import SceneObject

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


# Color definitions for visualization
COLORS = {
    'white': (200, 200, 200),      # All objects (dimmed)
    'gray': (80, 80, 80),          # Filtered out objects
    'blue': (50, 100, 255),        # Category candidates
    'yellow': (255, 200, 50),      # After spatial filter
    'orange': (255, 150, 50),      # After quick filter
    'green': (50, 255, 100),       # After select constraint
    'red': (255, 50, 50),          # Final result
}


@dataclass
class FilteringStep:
    """Record of a filtering step."""
    step_name: str
    description: str
    object_ids: Set[int]
    color: Tuple[int, int, int]


@dataclass
class QueryVisualization:
    """Visualization data for a query."""
    query: str
    steps: List[FilteringStep] = field(default_factory=list)
    final_ids: Set[int] = field(default_factory=set)


def load_scene_objects(scene_path: str) -> Tuple[List[SceneObject], Dict]:
    """Load scene objects from pkl.gz file.
    
    Uses SceneObject.from_dict() to create objects with all attributes
    from the pkl.gz file (output of 2b_build_3d_object_map_detect.sh).
    """
    pcd_dir = Path(scene_path) / "pcd_saves"
    
    pkl_files = list(pcd_dir.glob("*ram_withbg*_post.pkl.gz"))
    if not pkl_files:
        pkl_files = list(pcd_dir.glob("*_post.pkl.gz"))
    if not pkl_files:
        pkl_files = list(pcd_dir.glob("*.pkl.gz"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No pkl.gz files found in {pcd_dir}")
    
    pkl_file = pkl_files[0]
    logger.info(f"Loading scene from: {pkl_file.name}")
    
    with gzip.open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    objects = []
    obj_list = data.get('objects', []) if isinstance(data, dict) else data
    
    for i, obj_dict in enumerate(obj_list):
        if hasattr(obj_dict, '__dict__'):
            obj_dict = obj_dict.__dict__
        if not isinstance(obj_dict, dict):
            continue
        
        try:
            obj = SceneObject.from_dict(obj_id=i, data=obj_dict)
            objects.append(obj)
        except Exception as e:
            logger.warning(f"Failed to load object {i}: {e}")
    
    logger.info(f"Loaded {len(objects)} objects")
    return objects, data


def save_ply_with_colors(
    objects: List[SceneObject],
    color_map: Dict[int, Tuple[int, int, int]],
    output_path: Path,
    default_color: Tuple[int, int, int] = (50, 50, 50),
):
    """Save PLY file with specified colors for each object."""
    all_points = []
    all_colors = []
    
    for obj in objects:
        if obj.pcd_np is None or len(obj.pcd_np) == 0:
            continue
        
        points = obj.pcd_np
        color = color_map.get(obj.obj_id, default_color)
        colors = np.array([color] * len(points), dtype=np.uint8)
        
        all_points.append(points)
        all_colors.append(colors)
    
    if not all_points:
        return
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(all_points)):
            f.write(f"{all_points[i, 0]:.6f} {all_points[i, 1]:.6f} {all_points[i, 2]:.6f} ")
            f.write(f"{all_colors[i, 0]} {all_colors[i, 1]} {all_colors[i, 2]}\n")
    
    logger.info(f"Saved: {output_path.name}")


def save_filtering_steps(
    objects: List[SceneObject],
    vis: QueryVisualization,
    output_dir: Path,
):
    """Save PLY files showing the filtering process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_ids = set(obj.obj_id for obj in objects)
    
    # Save each step
    for i, step in enumerate(vis.steps):
        color_map = {}
        for obj_id in all_ids:
            if obj_id in step.object_ids:
                color_map[obj_id] = step.color
            else:
                color_map[obj_id] = COLORS['gray']
        
        filename = f"{i:02d}_{step.step_name}.ply"
        save_ply_with_colors(objects, color_map, output_dir / filename)
    
    # Save combined visualization showing all steps
    # Objects colored by their final state in the pipeline
    color_map = {}
    for obj_id in all_ids:
        color_map[obj_id] = COLORS['gray']  # Default: filtered out
    
    # Color by the latest step they survived
    for step in vis.steps:
        for obj_id in step.object_ids:
            color_map[obj_id] = step.color
    
    # Final results in red
    for obj_id in vis.final_ids:
        color_map[obj_id] = COLORS['red']
    
    save_ply_with_colors(objects, color_map, output_dir / "final_combined.ply")
    
    # Save legend
    legend_path = output_dir / "color_legend.txt"
    with open(legend_path, 'w') as f:
        f.write(f"Query: {vis.query}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Color Legend:\n")
        f.write("-" * 30 + "\n")
        for i, step in enumerate(vis.steps):
            color_name = [k for k, v in COLORS.items() if v == step.color][0]
            count = len(step.object_ids)
            f.write(f"{i:02d}. {step.step_name}: {color_name} ({count} objects)\n")
            f.write(f"    {step.description}\n")
        f.write(f"\nFinal Result: red ({len(vis.final_ids)} objects)\n")
    
    logger.info(f"Saved legend: {legend_path.name}")


def save_keyframes(
    objects: List[SceneObject],
    matched_ids: Set[int],
    scene_path: Path,
    output_dir: Path,
    max_keyframes: int = 3,
    stride: int = 5,
):
    """Save keyframe images for matched objects.
    
    Note: image_idx in objects stores the VIEW index (detection frame index).
    The actual frame file index = view_idx * stride.
    """
    import shutil
    
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = scene_path / "results"
    if not results_dir.exists():
        return
    
    for obj_id in matched_ids:
        obj = next((o for o in objects if o.obj_id == obj_id), None)
        if obj is None or not obj.image_idx:
            continue
        
        # image_idx stores view indices, need to convert to actual frame indices
        frame_counts = Counter(obj.image_idx)
        top_view_ids = [idx for idx, _ in frame_counts.most_common(max_keyframes)]
        
        for i, view_idx in enumerate(top_view_ids):
            # Convert view index to actual frame index
            actual_frame_idx = view_idx * stride
            
            for ext in ['.jpg', '.png']:
                frame_file = results_dir / f"frame{actual_frame_idx:06d}{ext}"
                if frame_file.exists():
                    dst = keyframe_dir / f"obj{obj_id}_{obj.object_tag}_view{view_idx}_frame{actual_frame_idx:06d}{ext}"
                    shutil.copy(frame_file, dst)
                    logger.info(f"Saved keyframe: view {view_idx} -> frame {actual_frame_idx}")
                    break


def execute_with_tracking(
    query_result,
    objects: List[SceneObject],
) -> Tuple[Any, QueryVisualization]:
    """Execute query and track filtering steps."""
    from conceptgraph.query_scene.query_executor import QueryExecutor
    from conceptgraph.query_scene.spatial_relations import SpatialRelationChecker
    
    vis = QueryVisualization(query=query_result.raw_query)
    
    # Create executor
    executor = QueryExecutor(
        objects=objects,
        relation_checker=SpatialRelationChecker(),
        use_quick_filters=True
    )
    
    # Manual execution to track steps
    root = query_result.root
    
    # Step 1: Category filter
    category_matches = executor._find_by_category(root.category)
    category_ids = set(obj.obj_id for obj in category_matches)
    vis.steps.append(FilteringStep(
        step_name="category_filter",
        description=f"Objects matching category '{root.category}'",
        object_ids=category_ids,
        color=COLORS['blue']
    ))
    
    candidates = category_matches
    current_ids = category_ids
    
    # Step 2+: Spatial constraints
    for i, constraint in enumerate(root.spatial_constraints):
        # Find anchors
        anchor_objects = []
        for anchor_node in constraint.anchors:
            anchor_result = executor._execute_node(anchor_node)
            anchor_objects.extend(anchor_result.matched_objects)
        
        anchor_ids = set(obj.obj_id for obj in anchor_objects)
        vis.steps.append(FilteringStep(
            step_name=f"anchor_{constraint.relation}",
            description=f"Anchor objects for '{constraint.relation}': {[a.category for a in constraint.anchors]}",
            object_ids=anchor_ids,
            color=COLORS['green']
        ))
        
        # Apply spatial filter
        filtered, scores = executor._apply_spatial_constraint(candidates, constraint)
        filtered_ids = set(obj.obj_id for obj in filtered)
        
        vis.steps.append(FilteringStep(
            step_name=f"spatial_{constraint.relation}",
            description=f"After spatial filter '{constraint.relation}': {len(filtered)} objects",
            object_ids=filtered_ids,
            color=COLORS['yellow']
        ))
        
        candidates = filtered
        current_ids = filtered_ids
    
    # Step 3: Select constraint
    if root.select_constraint and candidates:
        scores = {obj.obj_id: 1.0 for obj in candidates}
        selected, new_scores = executor._apply_select_constraint(
            candidates, scores, root.select_constraint
        )
        selected_ids = set(obj.obj_id for obj in selected)
        
        sc = root.select_constraint
        vis.steps.append(FilteringStep(
            step_name="select_constraint",
            description=f"After {sc.constraint_type.value} selection ({sc.metric}, {sc.order})",
            object_ids=selected_ids,
            color=COLORS['orange']
        ))
        
        candidates = selected
        current_ids = selected_ids
    
    # Final result
    vis.final_ids = set(obj.obj_id for obj in candidates)
    
    # Create proper result
    from conceptgraph.query_scene.query_executor import ExecutionResult
    result = ExecutionResult(
        node_id=root.node_id,
        matched_objects=candidates,
        scores={obj.obj_id: 1.0 for obj in candidates}
    )
    
    return result, vis


def run_e2e_test(
    query: str,
    objects: List[SceneObject],
    scene_categories: List[str],
    scene_path: Path,
    output_base_dir: Path,
    test_name: str
) -> Dict[str, Any]:
    """Run end-to-end test with step-by-step visualization."""
    from conceptgraph.query_scene.query_parser import QueryParser
    
    logger.info("=" * 70)
    logger.info(f"Test: {test_name}")
    logger.info(f"Query: \"{query}\"")
    logger.info("=" * 70)
    
    # Create query-specific output directory
    safe_name = query.replace(" ", "_").replace("\"", "").replace("'", "")[:50]
    output_dir = output_base_dir / safe_name
    
    result = {
        "query": query,
        "test_name": test_name,
        "output_dir": str(output_dir),
        "parse_success": False,
        "execute_success": False,
        "matched_objects": [],
        "steps": [],
    }
    
    # Parse query
    logger.info("[Step 1] Parsing query...")
    try:
        parser = QueryParser(
            llm_model="gpt-5.2-2025-12-11",
            scene_categories=scene_categories
        )
        parsed = parser.parse(query)
        result["parse_success"] = True
        
        logger.success(f"Root: {parsed.root.category}")
        if parsed.root.spatial_constraints:
            for sc in parsed.root.spatial_constraints:
                logger.info(f"  Spatial: {sc.relation} → {[a.category for a in sc.anchors]}")
        if parsed.root.select_constraint:
            sc = parsed.root.select_constraint
            logger.info(f"  Select: {sc.constraint_type.value} ({sc.metric})")
        
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return result
    
    # Execute with tracking
    logger.info("[Step 2] Executing query (with step tracking)...")
    try:
        exec_result, vis = execute_with_tracking(parsed, objects)
        result["execute_success"] = True
        
        # Record steps
        for step in vis.steps:
            result["steps"].append({
                "name": step.step_name,
                "description": step.description,
                "count": len(step.object_ids),
            })
            logger.info(f"  {step.step_name}: {len(step.object_ids)} objects")
        
        if exec_result.matched_objects:
            logger.success(f"Final: {len(exec_result.matched_objects)} object(s)")
            for obj in exec_result.matched_objects:
                logger.info(f"  - {obj.object_tag} (id={obj.obj_id})")
            
            result["matched_objects"] = [
                {"id": obj.obj_id, "tag": obj.object_tag}
                for obj in exec_result.matched_objects
            ]
        else:
            logger.warning("No objects matched")
            
    except Exception as e:
        logger.exception(f"Execute failed: {e}")
        return result
    
    # Generate visualizations
    logger.info("[Step 3] Generating visualizations...")
    try:
        save_filtering_steps(objects, vis, output_dir)
        # Note: stride=5 is the default used during mapping
        save_keyframes(objects, vis.final_ids, scene_path, output_dir, stride=5)
        logger.success(f"Saved to: {output_dir.name}/")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    return result


def main():
    """Main test function."""
    scene_path = project_root / "room0"
    output_dir = scene_path / "query_visualizations"
    
    if not scene_path.exists():
        logger.error(f"Scene not found: {scene_path}")
        return
    
    logger.info("=" * 70)
    logger.info("Loading Scene Objects")
    logger.info("=" * 70)
    
    objects, _ = load_scene_objects(str(scene_path))
    if not objects:
        logger.error("No objects loaded")
        return
    
    categories = Counter(obj.object_tag for obj in objects)
    logger.info(f"Loaded {len(objects)} objects")
    scene_categories = list(categories.keys())
    
    # Test queries - designed to cover various complexity levels
    test_queries = [
        # Basic queries
        ("the pillow on the armchair", "01. Simple spatial (ON)"),
        ("the largest pillow", "02. Superlative (SIZE)"),
        
        # Multi-level nesting
        ("the lamp on the table near the sofa", "03. Two-level nesting (ON + NEAR)"),
        ("the pillow on the armchair nearest the door", "04. Anchor superlative (ON + NEAREST)"),
        
        # Complex superlatives
        ("the smallest pillow on the largest armchair", "05. Dual superlative (target + anchor)"),
        ("the second stool from the left", "06. Ordinal selection"),
        
        # Multiple spatial constraints on single target
        ("the lamp near the sofa and near the window", "07. Multi-anchor (AND logic)"),
        
        # Between relation (two anchors)
        ("the pillow between the sofa and the armchair", "08. Between relation"),
        
        # Three-level nesting
        ("the lamp on the table near the sofa closest to the door", "09. Three-level nesting"),
        
        # Complex combinations
        ("all pillows on armchairs", "10. Multi-target (all)"),
        ("the red pillow on the sofa", "11. Attribute + spatial"),
        ("the second largest lamp on a table", "12. Ordinal superlative + spatial"),
    ]
    
    all_results = []
    for query, test_name in test_queries:
        result = run_e2e_test(query, objects, scene_categories, scene_path, output_dir, test_name)
        all_results.append(result)
    
    # Summary
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    passed = 0
    for r in all_results:
        obj_count = len(r["matched_objects"])
        if r["matched_objects"]:
            logger.success(f"{r['test_name']:40} -> {obj_count} objects")
            passed += 1
        else:
            logger.warning(f"{r['test_name']:40} -> {obj_count} objects")
        logger.info(f"    Output: {Path(r['output_dir']).name}/")
    
    logger.info(f"Total: {passed}/{len(all_results)} tests passed")
    logger.info(f"All visualizations: {output_dir}")
    
    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
