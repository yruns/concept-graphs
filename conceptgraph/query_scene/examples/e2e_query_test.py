#!/usr/bin/env python3
"""
End-to-End Query Test with Step-by-Step Visualization.

Outputs for each query:
- query_name/
  ├── 00_initial_candidates.ply      (初始候选 - 蓝色)
  ├── 01_final_candidates.ply        (最终结果 - 红色)
  ├── final_combined.ply             (合并展示 - 多色)
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
    'blue': (50, 100, 255),        # Initial candidates
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


def apply_affordances(objects: List[SceneObject], scene_path: Path) -> None:
    """Merge affordance outputs into scene objects."""
    affordance_file = scene_path / "sg_cache_detect" / "object_affordances.json"
    if not affordance_file.exists():
        affordance_file = scene_path / "sg_cache" / "object_affordances.json"
    if not affordance_file.exists():
        logger.warning("No affordance file found; using raw categories")
        return

    try:
        with open(affordance_file) as f:
            affordances = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load affordances: {e}")
        return

    aff_by_id = {a.get("id"): a for a in affordances if "id" in a}
    updated = 0
    for obj in objects:
        aff = aff_by_id.get(obj.obj_id)
        if not aff:
            continue
        obj.object_tag = aff.get("object_tag", obj.object_tag)
        if obj.object_tag:
            obj.category = obj.object_tag
        obj.summary = aff.get("summary", obj.summary)
        obj.affordance_category = aff.get("category", obj.affordance_category)
        affs = aff.get("affordances", {})
        if isinstance(affs, dict):
            obj.affordances = affs
            obj.co_objects = affs.get("co_objects", obj.co_objects)
        updated += 1

    logger.info(f"Applied affordances to {updated} objects")


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
        has_final_step = any(step.step_name == "final_candidates" for step in vis.steps)
        if not has_final_step:
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
    """Execute query and track initial/final candidates."""
    from conceptgraph.query_scene.query_executor import QueryExecutor
    from conceptgraph.query_scene.spatial_relations import SpatialRelationChecker
    
    # Create executor
    executor = QueryExecutor(
        objects=objects,
        relation_checker=SpatialRelationChecker(),
        use_quick_filters=True
    )
    
    vis = QueryVisualization(query=query_result.raw_query)

    # Initial candidates: category match before full execution
    root = query_result.root
    initial_candidates = executor._find_by_category(root.category)
    initial_ids = set(obj.obj_id for obj in initial_candidates)
    vis.steps.append(FilteringStep(
        step_name="initial_candidates",
        description=f"Initial candidates for category '{root.category}'",
        object_ids=initial_ids,
        color=COLORS['blue']
    ))

    # Full execution path
    result = executor.execute(query_result)

    # Final candidates
    vis.final_ids = set(obj.obj_id for obj in result.matched_objects)
    vis.steps.append(FilteringStep(
        step_name="final_candidates",
        description="Final matched objects after full execution",
        object_ids=vis.final_ids,
        color=COLORS['red']
    ))

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
    
    # Execute with QueryExecutor.execute
    logger.info("[Step 2] Executing query (QueryExecutor.execute)...")
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
    apply_affordances(objects, scene_path)
    if not objects:
        logger.error("No objects loaded")
        return
    
    categories = Counter(obj.category for obj in objects)
    logger.info(f"Loaded {len(objects)} objects")
    logger.info(f"Categories: {categories}")
    scene_categories = list(categories.keys())
    
    # Test queries - from simple to complex (Level 0 to Level 4)
    # Actual categories (from affordances): 
    #   throw_pillow(7), armchair(3), sofa(3), ottoman(3), stool(1), 
    #   side_table(4), coffee_table(1), floor_lamp(1), door(1), window_blinds(3)
    test_queries = [
        # ============== Level 0: Single object (no constraints) ==============
        ("a throw_pillow", "L0-01. Single object"),
        ("the sofa", "L0-02. Single object (definite)"),
        
        # ============== Level 1: Single constraint ==============
        # Superlative
        ("the largest throw_pillow", "L1-01. Superlative (max size)"),
        ("the smallest ottoman", "L1-02. Superlative (min size)"),
        
        # Ordinal
        ("the first ottoman from the left", "L1-03. Ordinal (position)"),
        ("the second largest side_table", "L1-04. Ordinal (size)"),
        
        # Single spatial relation
        ("the throw_pillow near the sofa", "L1-05. Spatial (NEAR)"),
        ("the ottoman near the coffee_table", "L1-06. Spatial (NEAR)"),
        
        # Multi-target
        ("all throw_pillows", "L1-07. Multi-target (all)"),
        ("all ottomans", "L1-08. Multi-target (all)"),
        
        # ============== Level 2: Two constraints / 2-level nesting ==============
        # Spatial + Superlative
        ("the smallest ottoman near the sofa", "L2-01. Spatial + Superlative"),
        ("the largest throw_pillow near the armchair", "L2-02. Spatial + Superlative"),
        
        # Anchor superlative (target near anchor[superlative])
        ("the armchair nearest the door", "L2-03. Anchor superlative (nearest)"),
        ("the sofa nearest the coffee_table", "L2-04. Anchor superlative (nearest)"),
        
        # 2-level spatial nesting (A near B near C)
        ("the ottoman near the sofa near the window_blinds", "L2-05. 2-level nesting (NEAR+NEAR)"),
        ("the throw_pillow near the armchair near the door", "L2-06. 2-level nesting (NEAR+NEAR)"),
        
        # Multi-anchor (AND logic)
        ("the ottoman near the sofa and near the coffee_table", "L2-07. Multi-anchor (AND)"),
        
        # Multi-target + spatial
        ("all throw_pillows near the sofa", "L2-08. Multi-target + Spatial"),
        
        # ============== Level 3: Three constraints / 3-level nesting ==============
        # 3-level spatial nesting
        ("the throw_pillow near the ottoman near the sofa near the window_blinds", "L3-01. 3-level nesting"),
        
        # Spatial + Anchor superlative + constraint
        ("the throw_pillow near the armchair nearest the door", "L3-02. Spatial + Anchor superlative"),
        
        # Superlative + 2-level spatial
        ("the largest throw_pillow near the sofa near the window_blinds", "L3-03. Superlative + 2-level spatial"),
        
        # Multi-anchor + superlative
        ("the smallest ottoman near the sofa and near the armchair", "L3-04. Multi-anchor + Superlative"),
        
        # ============== Level 4: Four+ constraints / 4-level nesting ==============
        # 4-level spatial nesting
        ("the throw_pillow near the ottoman near the sofa near the armchair near the door", "L4-01. 4-level nesting"),
        
        # Complex combination
        ("the smallest throw_pillow near the largest sofa nearest the door", "L4-02. Multi-superlative + spatial"),
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
