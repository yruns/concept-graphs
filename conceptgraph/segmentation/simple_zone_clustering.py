#!/usr/bin/env python3
"""
Simple Zone Clustering using DBSCAN + CLIP
==========================================

Fast prototype for functional zone segmentation:
1. Load ConceptGraphs object-level scene graph
2. DBSCAN clustering on object positions for spatial grouping
3. CLIP similarity for zone naming

Usage:
    python -m conceptgraph.segmentation.simple_zone_clustering \
        --pcd_file /path/to/pcd.pkl.gz \
        --output /path/to/output.json
"""
import os
import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage


# Zone name candidates for CLIP matching
ZONE_CANDIDATES = [
    "living_room", "lounge_area", "seating_area",
    "dining_area", "kitchen", "cooking_area",
    "bedroom", "sleeping_area",
    "office", "work_area", "study_area",
    "storage_area", "display_area",
    "entrance", "hallway", "corridor",
    "bathroom", "utility_area"
]

# Object category to zone affinity mapping (fallback without CLIP)
OBJECT_ZONE_AFFINITY = {
    # Living/Lounge
    "sofa": "lounge_area", "couch": "lounge_area", "armchair": "lounge_area",
    "coffee_table": "lounge_area", "ottoman": "lounge_area", "rug": "lounge_area",
    "tv": "lounge_area", "television": "lounge_area",
    
    # Dining
    "dining_table": "dining_area", "dining_chair": "dining_area",
    
    # Kitchen
    "stove": "kitchen", "refrigerator": "kitchen", "sink": "kitchen",
    "microwave": "kitchen", "oven": "kitchen", "cabinet": "kitchen",
    
    # Bedroom
    "bed": "bedroom", "nightstand": "bedroom", "dresser": "bedroom",
    "wardrobe": "bedroom", "closet": "bedroom",
    
    # Office
    "desk": "work_area", "office_chair": "work_area", "computer": "work_area",
    "monitor": "work_area", "bookshelf": "work_area",
    
    # Storage/Display
    "shelf": "storage_area", "sideboard": "storage_area", "cabinet": "storage_area",
    "vase": "display_area", "plant": "display_area", "art": "display_area",
    
    # Entry/Utility
    "door": "entrance", "doormat": "entrance",
    "thermostat": "utility_area", "vent": "utility_area",
}


@dataclass
class SimpleZone:
    """Simple zone representation"""
    zone_id: str
    zone_name: str
    objects: List[Dict]  # [{id, tag, position}]
    center: List[float]
    bbox: Dict[str, List[float]]  # {min: [...], max: [...]}
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "n_objects": len(self.objects),
            "objects": self.objects,
            "center": self.center,
            "bbox": self.bbox,
            "confidence": self.confidence
        }


class SimpleZoneClustering:
    """Simple zone clustering using DBSCAN + heuristic naming"""
    
    def __init__(self, eps: float = 1.5, min_samples: int = 2, use_clip: bool = False):
        """
        Args:
            eps: DBSCAN epsilon (max distance between points in cluster)
            min_samples: DBSCAN min samples per cluster
            use_clip: Whether to use CLIP for zone naming (requires clip model)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.use_clip = use_clip
        self.clip_model = None
        self.clip_preprocess = None
        
        if use_clip:
            self._load_clip()
    
    def _load_clip(self):
        """Load CLIP model for zone naming"""
        try:
            import clip
            import torch
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            self.clip_model.eval()
            print("CLIP model loaded")
        except ImportError:
            print("Warning: CLIP not available, using heuristic naming")
            self.use_clip = False
    
    def _hierarchical_cluster(self, positions: np.ndarray, distance_threshold: float, 
                               min_samples: int) -> np.ndarray:
        """
        Simple hierarchical clustering with distance threshold.
        Returns labels array where -1 indicates noise.
        """
        n = len(positions)
        if n < 2:
            return np.zeros(n, dtype=int)
        
        # Compute pairwise distances
        distances = pdist(positions)
        
        # Hierarchical clustering
        Z = linkage(distances, method='average')
        
        # Cut tree at distance threshold
        raw_labels = fcluster(Z, t=distance_threshold, criterion='distance')
        
        # Convert to 0-indexed, mark small clusters as noise (-1)
        from collections import Counter
        label_counts = Counter(raw_labels)
        
        labels = np.zeros(n, dtype=int)
        cluster_id = 0
        label_map = {}
        
        for label, count in label_counts.items():
            if count >= min_samples:
                label_map[label] = cluster_id
                cluster_id += 1
            else:
                label_map[label] = -1  # noise
        
        for i in range(n):
            labels[i] = label_map[raw_labels[i]]
        
        return labels
    
    def cluster_objects(self, pcd_file: str) -> List[SimpleZone]:
        """
        Main clustering pipeline.
        
        Args:
            pcd_file: Path to ConceptGraphs pcd pickle file
            
        Returns:
            List of SimpleZone objects
        """
        # Load objects
        objects = self._load_objects(pcd_file)
        print(f"Loaded {len(objects)} objects")
        
        if len(objects) < 2:
            print("Warning: Not enough objects for clustering")
            return []
        
        # Extract positions
        positions = []
        valid_objects = []
        for obj in objects:
            pos = obj.get('position')
            if pos is not None:
                positions.append(pos)
                valid_objects.append(obj)
        
        if len(positions) < 2:
            print("Warning: Not enough objects with positions")
            return []
        
        positions = np.array(positions)
        print(f"Objects with valid positions: {len(positions)}")
        
        # Run hierarchical clustering with distance threshold
        print(f"Running hierarchical clustering (distance_threshold={self.eps})...")
        labels = self._hierarchical_cluster(positions, self.eps, self.min_samples)
        
        # Count clusters (excluding noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        # Build zones from clusters
        zones = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_objects = [valid_objects[i] for i in range(len(valid_objects)) if cluster_mask[i]]
            cluster_positions = positions[cluster_mask]
            
            # Compute zone properties
            center = cluster_positions.mean(axis=0).tolist()
            bbox = {
                "min": cluster_positions.min(axis=0).tolist(),
                "max": cluster_positions.max(axis=0).tolist()
            }
            
            # Name the zone
            zone_name = self._name_zone(cluster_objects)
            
            zone = SimpleZone(
                zone_id=f"zone_{cluster_id}",
                zone_name=zone_name,
                objects=[{"id": o["id"], "tag": o["tag"], "position": o["position"]} 
                        for o in cluster_objects],
                center=center,
                bbox=bbox
            )
            zones.append(zone)
        
        # Handle noise points (assign to nearest zone or create misc zone)
        if n_noise > 0:
            noise_objects = [valid_objects[i] for i in range(len(valid_objects)) if labels[i] == -1]
            zones = self._handle_noise_points(zones, noise_objects, positions, labels)
        
        # Sort zones by number of objects (largest first)
        zones.sort(key=lambda z: len(z.objects), reverse=True)
        
        # Rename zone_ids to be sequential
        for i, zone in enumerate(zones):
            zone.zone_id = f"zone_{i}"
        
        return zones
    
    def _load_objects(self, pcd_file: str) -> List[Dict]:
        """Load objects from ConceptGraphs pcd file"""
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        objects = []
        for i, obj in enumerate(data.get('objects', [])):
            # Get object tag
            tag = "unknown"
            class_names = obj.get('class_name', [])
            if class_names:
                # Get most common non-'item' class
                valid_names = [n for n in class_names if n.lower() != 'item']
                if valid_names:
                    tag = Counter(valid_names).most_common(1)[0][0]
                elif class_names:
                    tag = class_names[0]
            
            # Get position
            position = None
            if 'pcd_np' in obj and len(obj['pcd_np']) > 0:
                position = obj['pcd_np'].mean(axis=0).tolist()
            elif 'bbox_np' in obj and len(obj['bbox_np']) > 0:
                position = obj['bbox_np'].mean(axis=0).tolist()
            
            objects.append({
                "id": i,
                "tag": tag,
                "position": position,
                "n_points": len(obj.get('pcd_np', [])),
            })
        
        return objects
    
    def _name_zone(self, objects: List[Dict]) -> str:
        """Name a zone based on its objects"""
        if self.use_clip and self.clip_model is not None:
            return self._name_zone_with_clip(objects)
        else:
            return self._name_zone_heuristic(objects)
    
    def _name_zone_heuristic(self, objects: List[Dict]) -> str:
        """Name zone using object-zone affinity mapping"""
        zone_votes = Counter()
        
        for obj in objects:
            tag_lower = obj["tag"].lower().replace("_", " ")
            
            # Check direct mapping
            for keyword, zone in OBJECT_ZONE_AFFINITY.items():
                if keyword in tag_lower:
                    zone_votes[zone] += 1
                    break
        
        if zone_votes:
            return zone_votes.most_common(1)[0][0]
        
        # Fallback: use most common object type
        tags = [obj["tag"] for obj in objects]
        most_common_tag = Counter(tags).most_common(1)[0][0]
        return f"{most_common_tag}_area"
    
    def _name_zone_with_clip(self, objects: List[Dict]) -> str:
        """Name zone using CLIP text similarity"""
        import torch
        import clip
        
        # Create description from objects
        tags = [obj["tag"] for obj in objects]
        tag_counts = Counter(tags)
        description = ", ".join([f"{count} {tag}" for tag, count in tag_counts.most_common(5)])
        
        # Encode zone candidates
        zone_texts = [f"a {zone.replace('_', ' ')}" for zone in ZONE_CANDIDATES]
        text_tokens = clip.tokenize(zone_texts)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Encode object description
            desc_tokens = clip.tokenize([f"an area with {description}"])
            desc_features = self.clip_model.encode_text(desc_tokens)
            desc_features /= desc_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (desc_features @ text_features.T).squeeze()
            best_idx = similarities.argmax().item()
        
        return ZONE_CANDIDATES[best_idx]
    
    def _handle_noise_points(self, zones: List[SimpleZone], noise_objects: List[Dict],
                             all_positions: np.ndarray, labels: np.ndarray) -> List[SimpleZone]:
        """Assign noise points to nearest zone or create misc zone"""
        if not zones:
            # No zones exist, create one misc zone
            center = np.array([o["position"] for o in noise_objects]).mean(axis=0).tolist()
            misc_zone = SimpleZone(
                zone_id="zone_misc",
                zone_name="miscellaneous_area",
                objects=[{"id": o["id"], "tag": o["tag"], "position": o["position"]} 
                        for o in noise_objects],
                center=center,
                bbox={"min": center, "max": center},
                confidence=0.5
            )
            return [misc_zone]
        
        # Compute zone centers
        zone_centers = np.array([z.center for z in zones])
        
        # Assign each noise point to nearest zone
        for obj in noise_objects:
            pos = np.array(obj["position"])
            distances = np.linalg.norm(zone_centers - pos, axis=1)
            nearest_zone_idx = distances.argmin()
            
            # Add to nearest zone
            zones[nearest_zone_idx].objects.append({
                "id": obj["id"],
                "tag": obj["tag"],
                "position": obj["position"]
            })
        
        # Recompute zone properties after adding noise points
        for zone in zones:
            positions = np.array([o["position"] for o in zone.objects])
            zone.center = positions.mean(axis=0).tolist()
            zone.bbox = {
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist()
            }
        
        return zones
    
    def save_result(self, zones: List[SimpleZone], output_path: str):
        """Save clustering result to JSON"""
        result = {
            "method": "dbscan_clustering",
            "params": {"eps": self.eps, "min_samples": self.min_samples},
            "n_zones": len(zones),
            "zones": [z.to_dict() for z in zones]
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Result saved to: {output_path}")
    
    def print_summary(self, zones: List[SimpleZone]):
        """Print clustering summary"""
        print("\n" + "=" * 50)
        print("ZONE CLUSTERING SUMMARY")
        print("=" * 50)
        
        total_objects = sum(len(z.objects) for z in zones)
        print(f"Total zones: {len(zones)}")
        print(f"Total objects: {total_objects}")
        
        for zone in zones:
            print(f"\n[{zone.zone_id}] {zone.zone_name}")
            print(f"  Objects ({len(zone.objects)}): ", end="")
            tags = [o["tag"] for o in zone.objects[:8]]
            if len(zone.objects) > 8:
                tags.append(f"+{len(zone.objects)-8} more")
            print(", ".join(tags))
            print(f"  Center: ({zone.center[0]:.2f}, {zone.center[1]:.2f}, {zone.center[2]:.2f})")
        
        print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Simple zone clustering using DBSCAN")
    parser.add_argument("--pcd_file", type=str, required=True, help="Path to pcd pickle file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--eps", type=float, default=1.5, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min samples")
    parser.add_argument("--use_clip", action="store_true", help="Use CLIP for zone naming")
    args = parser.parse_args()
    
    clusterer = SimpleZoneClustering(eps=args.eps, min_samples=args.min_samples, use_clip=args.use_clip)
    zones = clusterer.cluster_objects(args.pcd_file)
    clusterer.print_summary(zones)
    clusterer.save_result(zones, args.output)


if __name__ == "__main__":
    main()
