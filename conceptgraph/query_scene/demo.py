#!/usr/bin/env python3
"""Demo script for QueryScene on Replica room0."""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conceptgraph.query_scene.query_pipeline import QueryScenePipeline
from conceptgraph.query_scene.utils import format_result


def main():
    """Run demo queries on Replica room0."""
    import os
    
    # Get scene path from environment or default
    replica_root = os.environ.get("REPLICA_ROOT", "/home/shyue/Datasets/Replica/Replica")
    scene_path = Path(replica_root) / "room0"
    
    if not scene_path.exists():
        print(f"Scene path not found: {scene_path}")
        print("Please set REPLICA_ROOT environment variable or ensure the path exists.")
        return
    
    print("=" * 60)
    print("QueryScene Demo - Replica room0")
    print("=" * 60)
    
    # Find pcd file
    pcd_dir = scene_path / "pcd_saves"
    pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
    if not pcd_files:
        pcd_files = list(pcd_dir.glob("*.pkl.gz"))
    
    if not pcd_files:
        print(f"No pcd file found in {pcd_dir}")
        print("Please run the ConceptGraphs pipeline first:")
        print("  bash bashes/1_extract_2d_segmentation.sh")
        print("  bash bashes/2_build_3d_object_map.sh")
        return
    
    pcd_file = pcd_files[0]
    print(f"\nUsing pcd file: {pcd_file.name}")
    
    # Create pipeline
    print("\nLoading scene and building indices...")
    try:
        pipeline = QueryScenePipeline.from_scene(
            str(scene_path),
            str(pcd_file),
            stride=5,
            llm_url=os.environ.get("LLM_BASE_URL", "http://10.21.231.7:8006"),
            llm_model=os.environ.get("LLM_MODEL", "gemini-2.0-flash"),
        )
    except Exception as e:
        print(f"Error loading scene: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print scene summary
    summary = pipeline.summary()
    print(f"\nScene loaded:")
    print(f"  Objects: {summary['scene']['n_objects']}")
    print(f"  Views: {summary['scene']['n_views']}")
    print(f"  Categories: {list(summary['scene']['categories'].keys())[:10]}...")
    
    # Demo queries
    queries = [
        "椅子",
        "桌子",
        "沙发旁边的台灯",
        "窗户附近的物体",
    ]
    
    print("\n" + "=" * 60)
    print("Running queries...")
    print("=" * 60)
    
    for query in queries:
        print(f"\n>>> Query: \"{query}\"")
        try:
            result = pipeline.query(query)
            print(format_result(result))
        except Exception as e:
            print(f"Error: {e}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            result = pipeline.query(query)
            print(format_result(result))
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
