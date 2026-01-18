#!/usr/bin/env python3
"""
Hierarchical Scene Segmentation Evaluation
==========================================

Evaluates the quality of scene segmentation from multiple perspectives:
1. Semantic Coherence: Do objects in the same zone share functional relevance?
2. Spatial Compactness: Are objects in a zone spatially clustered?
3. Coverage: Are all objects reasonably assigned?
4. Balance: Is the distribution of objects across zones reasonable?
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
from dataclasses import dataclass
import argparse


@dataclass
class EvaluationResult:
    """Evaluation result container"""
    semantic_coherence: float  # 0-1, higher is better
    spatial_compactness: float  # 0-1, higher is better (normalized inverse of avg distance)
    coverage_ratio: float  # Ratio of assigned objects
    balance_score: float  # 0-1, distribution balance (entropy-based)
    zone_metrics: Dict[str, Dict]  # Per-zone metrics
    overall_score: float  # Weighted average
    
    def to_dict(self) -> Dict:
        return {
            "semantic_coherence": self.semantic_coherence,
            "spatial_compactness": self.spatial_compactness,
            "coverage_ratio": self.coverage_ratio,
            "balance_score": self.balance_score,
            "zone_metrics": self.zone_metrics,
            "overall_score": self.overall_score
        }


# Semantic category mappings for coherence evaluation
FUNCTIONAL_CATEGORIES = {
    "seating": ["sofa", "couch", "chair", "armchair", "ottoman", "stool", "bench", "seat"],
    "lighting": ["lamp", "light", "chandelier", "sconce", "bulb"],
    "storage": ["shelf", "cabinet", "drawer", "dresser", "closet", "wardrobe", "box", "basket", "container"],
    "surface": ["table", "desk", "counter", "sideboard", "console", "nightstand"],
    "decor": ["plant", "vase", "painting", "art", "frame", "sculpture", "mirror", "clock"],
    "textile": ["pillow", "cushion", "rug", "carpet", "curtain", "blanket", "throw"],
    "electronics": ["tv", "television", "monitor", "speaker", "computer", "phone"],
    "entry": ["door", "doorway", "entrance", "mat"],
    "climate": ["fan", "heater", "vent", "radiator", "air_conditioner"],
}


def get_object_category(tag: str) -> str:
    """Map object tag to functional category"""
    tag_lower = tag.lower().replace("_", " ")
    for category, keywords in FUNCTIONAL_CATEGORIES.items():
        for kw in keywords:
            if kw in tag_lower:
                return category
    return "other"


def compute_semantic_coherence(zones: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Compute semantic coherence: ratio of objects sharing functional category with zone majority.
    Higher score = objects in same zone have similar functions.
    """
    zone_scores = {}
    total_coherent = 0
    total_objects = 0
    
    for zone in zones:
        objects = zone.get("objects", [])
        if not objects:
            continue
        
        # Get functional categories of all objects
        categories = [get_object_category(obj["object_tag"]) for obj in objects]
        
        # Find dominant category
        cat_counts = Counter(categories)
        if not cat_counts:
            continue
        
        dominant_cat, dominant_count = cat_counts.most_common(1)[0]
        
        # Coherence = ratio of objects in dominant category
        coherence = dominant_count / len(objects) if objects else 0
        zone_scores[zone["zone_name"]] = coherence
        
        total_coherent += dominant_count
        total_objects += len(objects)
    
    avg_coherence = total_coherent / total_objects if total_objects > 0 else 0
    return avg_coherence, zone_scores


def compute_spatial_compactness(zones: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Compute spatial compactness: inverse of average intra-zone distance (normalized).
    Higher score = objects in same zone are closer together.
    """
    zone_scores = {}
    all_avg_distances = []
    
    for zone in zones:
        objects = zone.get("objects", [])
        positions = [obj["position"] for obj in objects if obj.get("position")]
        
        if len(positions) < 2:
            zone_scores[zone["zone_name"]] = 1.0  # Single object = perfectly compact
            continue
        
        positions = np.array(positions)
        
        # Compute pairwise distances
        n = len(positions)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances) if distances else 0
        all_avg_distances.append(avg_dist)
        zone_scores[zone["zone_name"]] = avg_dist
    
    # Normalize: convert distance to compactness score (smaller distance = higher score)
    if all_avg_distances:
        max_dist = max(all_avg_distances) if max(all_avg_distances) > 0 else 1
        for zone_name in zone_scores:
            zone_scores[zone_name] = 1 - (zone_scores[zone_name] / max_dist)
        avg_compactness = np.mean(list(zone_scores.values()))
    else:
        avg_compactness = 1.0
    
    return avg_compactness, zone_scores


def compute_coverage(zones: List[Dict], total_objects: int) -> float:
    """Compute ratio of objects assigned to zones"""
    assigned = sum(len(zone.get("objects", [])) for zone in zones)
    return assigned / total_objects if total_objects > 0 else 0


def compute_balance(zones: List[Dict]) -> float:
    """
    Compute balance score using entropy.
    Higher score = more balanced distribution across zones.
    """
    if not zones:
        return 0
    
    counts = [len(zone.get("objects", [])) for zone in zones]
    total = sum(counts)
    
    if total == 0:
        return 0
    
    # Compute normalized entropy
    probs = [c / total for c in counts if c > 0]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log2(len(zones)) if len(zones) > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def compute_defining_object_quality(zones: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate quality of defining objects:
    - Are defining objects actually zone-specific?
    - Do they make semantic sense for the zone?
    """
    results = {}
    
    for zone in zones:
        objects = zone.get("objects", [])
        defining = [o for o in objects if o.get("relation_type") == "defining"]
        
        results[zone["zone_name"]] = {
            "n_defining": len(defining),
            "n_total": len(objects),
            "defining_ratio": len(defining) / len(objects) if objects else 0,
            "defining_objects": [o["object_tag"] for o in defining]
        }
    
    return results


def evaluate_scene_graph(scene_graph_path: str, verbose: bool = True) -> EvaluationResult:
    """
    Main evaluation function.
    
    Args:
        scene_graph_path: Path to hierarchical_scene_graph.json
        verbose: Print detailed results
    
    Returns:
        EvaluationResult with all metrics
    """
    with open(scene_graph_path) as f:
        sg = json.load(f)
    
    zones = sg.get("functional_zones", [])
    total_objects = sg.get("metadata", {}).get("n_objects", 0)
    
    if not zones:
        print("Warning: No zones found in scene graph")
        return EvaluationResult(0, 0, 0, 0, {}, 0)
    
    # Compute metrics
    semantic_coherence, sem_zone_scores = compute_semantic_coherence(zones)
    spatial_compactness, spatial_zone_scores = compute_spatial_compactness(zones)
    coverage = compute_coverage(zones, total_objects)
    balance = compute_balance(zones)
    defining_quality = compute_defining_object_quality(zones)
    
    # Combine zone metrics
    zone_metrics = {}
    for zone in zones:
        name = zone["zone_name"]
        zone_metrics[name] = {
            "n_objects": len(zone.get("objects", [])),
            "semantic_coherence": sem_zone_scores.get(name, 0),
            "spatial_compactness": spatial_zone_scores.get(name, 0),
            "defining_info": defining_quality.get(name, {})
        }
    
    # Compute overall score (weighted average)
    weights = {
        "semantic": 0.3,
        "spatial": 0.25,
        "coverage": 0.25,
        "balance": 0.2
    }
    overall = (
        weights["semantic"] * semantic_coherence +
        weights["spatial"] * spatial_compactness +
        weights["coverage"] * coverage +
        weights["balance"] * balance
    )
    
    result = EvaluationResult(
        semantic_coherence=semantic_coherence,
        spatial_compactness=spatial_compactness,
        coverage_ratio=coverage,
        balance_score=balance,
        zone_metrics=zone_metrics,
        overall_score=overall
    )
    
    if verbose:
        print_evaluation_report(result, zones)
    
    return result


def print_evaluation_report(result: EvaluationResult, zones: List[Dict]):
    """Print formatted evaluation report"""
    print("\n" + "=" * 60)
    print("SCENE SEGMENTATION EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Score':<10} {'Rating'}")
    print("-" * 50)
    
    def rating(score):
        if score >= 0.8: return "Excellent"
        if score >= 0.6: return "Good"
        if score >= 0.4: return "Fair"
        return "Poor"
    
    print(f"{'Semantic Coherence':<25} {result.semantic_coherence:.3f}      {rating(result.semantic_coherence)}")
    print(f"{'Spatial Compactness':<25} {result.spatial_compactness:.3f}      {rating(result.spatial_compactness)}")
    print(f"{'Coverage Ratio':<25} {result.coverage_ratio:.3f}      {rating(result.coverage_ratio)}")
    print(f"{'Balance Score':<25} {result.balance_score:.3f}      {rating(result.balance_score)}")
    print("-" * 50)
    print(f"{'OVERALL SCORE':<25} {result.overall_score:.3f}      {rating(result.overall_score)}")
    
    print(f"\n\nPER-ZONE ANALYSIS")
    print("-" * 60)
    
    for zone in zones:
        name = zone["zone_name"]
        metrics = result.zone_metrics.get(name, {})
        
        print(f"\n[{name}]")
        print(f"  Objects: {metrics.get('n_objects', 0)}")
        print(f"  Semantic coherence: {metrics.get('semantic_coherence', 0):.3f}")
        print(f"  Spatial compactness: {metrics.get('spatial_compactness', 0):.3f}")
        
        defining_info = metrics.get('defining_info', {})
        if defining_info.get('defining_objects'):
            print(f"  Defining objects ({defining_info['n_defining']}): {', '.join(defining_info['defining_objects'][:5])}")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if result.semantic_coherence < 0.5:
        print("  - Consider refining zone definitions based on object functionality")
    if result.spatial_compactness < 0.5:
        print("  - Some zones may contain spatially scattered objects")
    if result.balance_score < 0.5:
        print("  - Object distribution is unbalanced; consider splitting large zones")
    if result.overall_score >= 0.7:
        print("  - Overall segmentation quality is good!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical scene segmentation")
    parser.add_argument("--scene_graph", type=str, required=True,
                       help="Path to hierarchical_scene_graph.json")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for evaluation results JSON")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    args = parser.parse_args()
    
    result = evaluate_scene_graph(args.scene_graph, verbose=not args.quiet)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
