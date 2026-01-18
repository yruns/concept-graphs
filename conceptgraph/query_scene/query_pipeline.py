"""Complete query processing pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from .data_structures import GroundingResult, ObjectNode, QueryInfo
from .scene_representation import QuerySceneRepresentation
from .index_builder import CLIPIndex, SpatialIndex, VisibilityIndex, SceneIndices
from .query_parser import QueryParser
from .vlm_interface import VLMClient, VLMInputConstructor, VLMOutputParser


class QueryScenePipeline:
    """Complete pipeline for query-driven scene understanding."""
    
    def __init__(
        self,
        scene: QuerySceneRepresentation,
        indices: Optional[SceneIndices] = None,
        llm_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
        vlm_url: str = "http://localhost:11434",
        vlm_model: str = "llava:7b",
    ):
        self.scene = scene
        self.indices = indices
        self.llm_url = llm_url
        self.llm_model = llm_model
        
        self.parser = QueryParser(llm_url, llm_model)
        self.vlm_client = VLMClient(vlm_url, vlm_model)
        self.vlm_constructor = VLMInputConstructor()
        
        if self.indices is None:
            self._build_indices()
    
    def _build_indices(self):
        """Build scene indices."""
        print("Building scene indices...")
        clip_index = CLIPIndex(self.scene.feature_dim)
        clip_index.build_from_objects(self.scene.objects)
        
        visibility_index = VisibilityIndex()
        visibility_index.build(self.scene.objects, self.scene.camera_poses, self.scene.depth_paths)
        
        spatial_index = SpatialIndex()
        spatial_index.build(self.scene.objects)
        
        self.indices = SceneIndices(clip_index, visibility_index, spatial_index)
        
        for obj in self.scene.objects:
            best_views = visibility_index.get_best_views(obj.obj_id, top_k=5)
            obj.best_view_ids = [v[0] for v in best_views]
    
    @classmethod
    def from_scene(cls, scene_path: str, pcd_file: Optional[str] = None, stride: int = 5, **kwargs):
        """Create pipeline from scene path."""
        scene_path = Path(scene_path)
        
        if pcd_file is None:
            pcd_dir = scene_path / "pcd_saves"
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
            if not pcd_files:
                pcd_files = list(pcd_dir.glob("*.pkl.gz"))
            if not pcd_files:
                raise FileNotFoundError(f"No pcd file found in {pcd_dir}")
            pcd_file = pcd_files[0]
        
        scene = QuerySceneRepresentation.from_pcd_file(pcd_file, scene_path, stride)
        return cls(scene, **kwargs)
    
    def query(self, query_text: str, top_k: int = 5) -> GroundingResult:
        """Process a query and return grounding result."""
        print(f"Processing query: {query_text}")
        
        query_info = self.parser.parse(query_text)
        print(f"  Parsed: target={query_info.target}, type={query_info.query_type.value}")
        
        candidates = self._semantic_retrieval(query_info, top_k * 2)
        print(f"  Retrieved {len(candidates)} candidates")
        
        if not candidates:
            return GroundingResult.failure("No matching objects found")
        
        if query_info.anchor:
            candidates = self._spatial_filter(candidates, query_info)
            print(f"  After spatial filter: {len(candidates)} candidates")
        
        if len(candidates) == 1:
            return GroundingResult.from_object(candidates[0], confidence=0.9, reasoning="Single match")
        
        return GroundingResult.from_object(candidates[0], confidence=0.7,
            reasoning=f"Best CLIP match from {len(candidates)} candidates")
    
    def _semantic_retrieval(self, query_info: QueryInfo, top_k: int) -> List[ObjectNode]:
        """Retrieve candidate objects using CLIP."""
        if self.indices is None or self.indices.clip_index.index is None:
            return self.scene.get_objects_by_category(query_info.target)[:top_k]
        
        try:
            import torch
            import open_clip
            
            model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-H-14")
            
            with torch.no_grad():
                tokens = tokenizer([query_info.target])
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                query_feat = feat.cpu().numpy().flatten()
            
            results = self.indices.clip_index.search(query_feat, top_k)
            return [self.scene.objects[r[0]] for r in results if r[0] < len(self.scene.objects)]
        except Exception as e:
            print(f"  CLIP encoding failed: {e}, using category match")
            return self.scene.get_objects_by_category(query_info.target)[:top_k]
    
    def _spatial_filter(self, candidates: List[ObjectNode], query_info: QueryInfo) -> List[ObjectNode]:
        """Filter candidates by spatial relation."""
        if not query_info.anchor:
            return candidates
        
        anchor_objs = self.scene.get_objects_by_category(query_info.anchor)
        if not anchor_objs:
            return candidates
        
        anchor = anchor_objs[0]
        if anchor.centroid is None:
            return candidates
        
        radius = 2.0
        filtered = [obj for obj in candidates if obj.centroid is not None 
                   and np.linalg.norm(obj.centroid - anchor.centroid) < radius]
        
        return filtered if filtered else candidates
    
    def summary(self) -> Dict[str, Any]:
        """Return pipeline summary."""
        return {"scene": self.scene.summary(), "indices_built": self.indices is not None}


def run_query(scene_path: str, query: str, **kwargs) -> GroundingResult:
    """Convenience function to run a query on a scene."""
    pipeline = QueryScenePipeline.from_scene(scene_path, **kwargs)
    return pipeline.query(query)


def visualize_result(
    pcd_file: str,
    result: GroundingResult,
    output_dir: str,
    query_info: Optional[QueryInfo] = None,
) -> Dict[str, str]:
    """Visualize query result and save to output directory.
    
    Args:
        pcd_file: Path to the pcd file.
        result: Grounding result to visualize.
        output_dir: Directory to save visualizations.
        query_info: Optional query info for anchor visualization.
    
    Returns:
        Dictionary of output file paths.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from implicit_scene.visualize import visualize_query_result, visualize_topdown
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    if not result.success:
        print(f"Cannot visualize failed result: {result.reason}")
        return outputs
    
    target_ids = [result.object_id] if result.object_id is not None else []
    reference_ids = []
    
    # Get labels
    labels = {}
    if result.object_node:
        labels[result.object_id] = result.object_node.category
    
    # 3D point cloud visualization
    ply_path = str(output_dir / "query_result.ply")
    try:
        visualize_query_result(
            pcd_file,
            target_ids=target_ids,
            reference_ids=reference_ids,
            output_path=ply_path,
            show_all=True,
        )
        outputs["ply"] = ply_path
    except Exception as e:
        print(f"PLY visualization failed: {e}")
    
    # 2D top-down visualization
    png_path = str(output_dir / "query_result_topdown.png")
    try:
        visualize_topdown(
            pcd_file,
            target_ids=target_ids,
            reference_ids=reference_ids,
            output_path=png_path,
            labels=labels,
        )
        outputs["png"] = png_path
    except Exception as e:
        print(f"Top-down visualization failed: {e}")
    
    return outputs
