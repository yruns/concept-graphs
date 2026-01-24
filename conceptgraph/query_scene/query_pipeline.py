"""Complete query processing pipeline with loguru logging."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger

from .data_structures import GroundingResult, ObjectNode, QueryInfo, ObjectDescriptions
from .scene_representation import QuerySceneRepresentation
from .index_builder import CLIPIndex, SpatialIndex, VisibilityIndex, RegionIndex, SceneIndices
from .query_parser import QueryParser
from .vlm_interface import VLMClient, VLMInputConstructor, VLMOutputParser
from .utils import (
    generate_bev, annotate_bev_with_distances, load_image,
    annotate_image_with_objects, project_3d_bbox_to_2d, crop_object_from_image
)


class QueryScenePipeline:
    """Complete pipeline for query-driven scene understanding with hierarchical retrieval."""
    
    def __init__(
        self,
        scene: QuerySceneRepresentation,
        indices: Optional[SceneIndices] = None,
        descriptions: Optional[Dict[int, ObjectDescriptions]] = None,
        llm_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
        vlm_url: str = "http://localhost:11434",
        vlm_model: str = "llava:7b",
        use_vlm: bool = True,
    ):
        logger.info("Initializing QueryScenePipeline")
        self.scene = scene
        self.indices = indices
        self.descriptions = descriptions or {}
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.use_vlm = use_vlm
        
        logger.debug(f"  use_vlm={use_vlm}, llm_url={llm_url}")
        
        self.parser = QueryParser(llm_url, llm_model)
        self.vlm_client = VLMClient(vlm_url, vlm_model)
        self.vlm_constructor = VLMInputConstructor()
        
        if self.indices is None:
            self._build_indices()
        
        logger.success("Pipeline initialized")
    
    def _build_indices(self):
        """Build scene indices including region index."""
        logger.info("Building scene indices...")
        
        # Use SceneIndices.build_all for hierarchical index
        self.indices = SceneIndices.build_all(self.scene, build_regions=True)
        
        # Update objects with best view IDs and nearby objects
        for obj in self.scene.objects:
            # Best views
            best_views = self.indices.visibility_index.get_best_views(obj.obj_id, top_k=5)
            obj.best_view_ids = [v[0] for v in best_views]
            
            # Nearby objects (within 1.5m)
            if obj.centroid is not None and self.indices.spatial_index:
                nearby = self.indices.spatial_index.find_nearby(obj.obj_id, radius=1.5)
                obj.nearby_object_ids = [oid for oid, _ in nearby if oid != obj.obj_id]
        
        logger.success(f"Indices built: {len(self.scene.objects)} objects indexed")
    
    @classmethod
    def from_scene(cls, scene_path: str, pcd_file: Optional[str] = None, stride: int = 5, **kwargs):
        """Create pipeline from scene path."""
        logger.info(f"Creating pipeline from scene: {scene_path}")
        scene_path = Path(scene_path)
        
        if pcd_file is None:
            pcd_dir = scene_path / "pcd_saves"
            # Prefer RAM-tagged files (have semantic labels)
            pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
            if not pcd_files:
                pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
            if not pcd_files:
                pcd_files = list(pcd_dir.glob("*.pkl.gz"))
            if not pcd_files:
                logger.error(f"No pcd file found in {pcd_dir}")
                raise FileNotFoundError(f"No pcd file found in {pcd_dir}")
            pcd_file = str(pcd_files[0])
        
        logger.debug(f"Using PCD file: {pcd_file}")
        scene = QuerySceneRepresentation.from_pcd_file(pcd_file, scene_path, stride)
        logger.info(f"Scene loaded: {len(scene.objects)} objects, {len(scene.camera_poses)} poses")
        
        return cls(scene, **kwargs)
    
    def query(self, query_text: str, top_k: int = 10) -> GroundingResult:
        """Process a query and return grounding result."""
        logger.info("=" * 50)
        logger.info(f"Query: {query_text}")
        
        # Step 1: Parse query
        logger.debug("Step 1: Parsing query...")
        query_info = self.parser.parse(query_text)
        logger.info(f"  Parsed: target={query_info.target}, anchor={query_info.anchor}, "
                   f"type={query_info.query_type.value}, use_bev={query_info.use_bev}")
        
        # Step 2: Hierarchical retrieval
        logger.debug("Step 2: Hierarchical retrieval...")
        candidates = self._hierarchical_retrieval(query_info, top_k * 2)
        logger.info(f"  Retrieved {len(candidates)} candidates")
        for c in candidates[:5]:
            logger.debug(f"    - {c.obj_id}: {c.category}")
        
        if not candidates:
            logger.warning("No matching objects found")
            return GroundingResult.failure("No matching objects found")
        
        # Step 3: Spatial filter
        anchor_obj = None
        if query_info.anchor:
            logger.debug(f"Step 3: Spatial filter (anchor={query_info.anchor})...")
            candidates, anchor_obj = self._spatial_filter(candidates, query_info)
            logger.info(f"  After spatial filter: {len(candidates)} candidates")
        
        if not candidates:
            logger.warning("No objects satisfy spatial constraint")
            return GroundingResult.failure("No objects satisfy spatial constraint")
        
        # Step 3.5: BEV filter (optional)
        if query_info.use_bev and len(candidates) > 2 and anchor_obj:
            logger.debug("Step 3.5: BEV spatial filter...")
            candidates = self._bev_filter(candidates, anchor_obj)
            logger.info(f"  After BEV filter: {len(candidates)} candidates")
        
        # Single match - return directly
        if len(candidates) == 1:
            logger.success(f"Single match found: {candidates[0].obj_id} ({candidates[0].category})")
            return GroundingResult.from_object(candidates[0], confidence=0.9, reasoning="Single match")
        
        # Step 4: VLM inference (if enabled)
        if self.use_vlm and len(candidates) > 1:
            logger.debug("Step 4: VLM inference...")
            vlm_result = self._vlm_inference(candidates, query_info, anchor_obj)
            if vlm_result and vlm_result.success:
                logger.success(f"VLM selected: {vlm_result.object_id}")
                return vlm_result
            logger.warning("VLM inference failed, using CLIP ranking")
        
        # Fallback: Best CLIP match
        result = GroundingResult.from_object(
            candidates[0], confidence=0.7,
            reasoning=f"Best match from {len(candidates)} candidates"
        )
        logger.success(f"Best match: {result.object_id} ({candidates[0].category})")
        return result
    
    def _hierarchical_retrieval(self, query_info: QueryInfo, top_k: int) -> List[ObjectNode]:
        """Hierarchical retrieval: region -> object."""
        # Try hierarchical search if region index available
        if self.indices and self.indices.region_index:
            logger.debug("  Using hierarchical retrieval (region -> object)")
            query_feat = self._encode_text(query_info.target)
            if query_feat is not None:
                results = self.indices.hierarchical_search(query_feat, top_k_regions=3, top_k_objects=top_k)
                obj_ids = [meta.get("obj_id") for _, _, meta in results]
                candidates = [self.scene.get_object_by_id(oid) for oid in obj_ids if oid]
                candidates = [c for c in candidates if c is not None]
                if candidates:
                    logger.debug(f"  Hierarchical search found {len(candidates)} candidates")
                    return candidates
        
        # Fallback to direct CLIP search
        return self._semantic_retrieval(query_info, top_k)
    
    def _semantic_retrieval(self, query_info: QueryInfo, top_k: int) -> List[ObjectNode]:
        """Direct CLIP-based retrieval."""
        logger.debug("  Using direct CLIP retrieval")
        
        if self.indices is None or self.indices.clip_index.index is None:
            logger.debug("  No CLIP index, using category match")
            return self.scene.get_objects_by_category(query_info.target)[:top_k]
        
        query_feat = self._encode_text(query_info.target)
        if query_feat is not None:
            results = self.indices.clip_index.search(query_feat, top_k)
            logger.debug(f"  CLIP search returned {len(results)} results")
            obj_ids = [meta.get("obj_id") for _, _, meta in results]
            return [self.scene.get_object_by_id(oid) for oid in obj_ids 
                   if self.scene.get_object_by_id(oid) is not None]
        
        logger.debug("  CLIP encoding failed, using category match")
        return self.scene.get_objects_by_category(query_info.target)[:top_k]
    
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to CLIP feature."""
        try:
            import torch
            import open_clip
            
            model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-H-14")
            
            with torch.no_grad():
                tokens = tokenizer([text])
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"CLIP encoding failed: {e}")
            return None
    
    def _spatial_filter(self, candidates: List[ObjectNode], query_info: QueryInfo):
        """Filter candidates by spatial relation."""
        if not query_info.anchor:
            return candidates, None
        
        anchor_objs = self.scene.get_objects_by_category(query_info.anchor)
        if not anchor_objs:
            logger.warning(f"Anchor '{query_info.anchor}' not found in scene")
            return candidates, None
        
        anchor = anchor_objs[0]
        logger.debug(f"  Found anchor: {anchor.obj_id} ({anchor.category})")
        
        if anchor.centroid is None:
            return candidates, anchor
        
        # Determine radius based on relation
        relation = (query_info.relation or "").lower()
        if "旁边" in relation or "beside" in relation:
            radius = 1.5
        elif "附近" in relation or "near" in relation:
            radius = 2.5
        else:
            radius = 2.0
        
        logger.debug(f"  Filtering with radius={radius}m from anchor")
        
        filtered = []
        for obj in candidates:
            if obj.centroid is not None:
                dist = np.linalg.norm(obj.centroid - anchor.centroid)
                if dist < radius:
                    filtered.append(obj)
                    logger.debug(f"    {obj.obj_id} ({obj.category}): dist={dist:.2f}m ✓")
                else:
                    logger.debug(f"    {obj.obj_id} ({obj.category}): dist={dist:.2f}m ✗")
        
        return (filtered if filtered else candidates), anchor
    
    def _bev_filter(self, candidates: List[ObjectNode], anchor: ObjectNode) -> List[ObjectNode]:
        """Filter using BEV spatial analysis."""
        if anchor.centroid is None:
            return candidates
        
        def dist_to_anchor(obj):
            if obj.centroid is None:
                return float('inf')
            return np.linalg.norm(obj.centroid - anchor.centroid)
        
        # Sort by distance first
        sorted_candidates = sorted(candidates, key=dist_to_anchor)
        logger.debug(f"  BEV sorted by distance to anchor:")
        for i, obj in enumerate(sorted_candidates[:5]):
            d = dist_to_anchor(obj)
            logger.debug(f"    {i+1}. {obj.obj_id} ({obj.category}): {d:.2f}m")
        
        return sorted_candidates[:5]
    
    def _vlm_inference(
        self,
        candidates: List[ObjectNode],
        query_info: QueryInfo,
        anchor: Optional[ObjectNode] = None,
    ) -> Optional[GroundingResult]:
        """VLM-based inference with annotated images.
        
        Steps:
        1. Select best views for candidates
        2. Load and annotate RGB images
        3. Optionally generate annotated BEV
        4. Construct VLM input
        5. Call VLM and parse output
        """
        if not candidates:
            return None
        
        # Step 1: Collect best views for all candidates
        view_ids = set()
        for obj in candidates[:5]:  # Limit to top 5 for efficiency
            if obj.best_view_ids:
                view_ids.update(obj.best_view_ids[:2])
        
        if not view_ids and self.indices and self.indices.visibility_index:
            # Fallback: get views from visibility index
            for obj in candidates[:5]:
                views = self.indices.visibility_index.get_best_views(obj.obj_id, top_k=2)
                view_ids.update([v[0] for v in views])
        
        view_ids = list(view_ids)[:3]  # Limit to 3 views
        logger.debug(f"  Selected views: {view_ids}")
        
        if not view_ids:
            logger.warning("  No valid views found for VLM inference")
            return None
        
        # Step 2: Load and annotate RGB images
        # Note: view_ids from visibility_index correspond to camera_poses indices
        # image_paths should have same length as camera_poses (both sampled by stride)
        annotated_images = []
        highlight_ids = [obj.obj_id for obj in candidates[:5]]
        n_poses = len(self.scene.camera_poses)
        n_images = len(self.scene.image_paths)
        
        logger.debug(f"  camera_poses: {n_poses}, image_paths: {n_images}")
        
        for view_id in view_ids:
            # Map view_id to image index (they should correspond if stride is consistent)
            if view_id >= n_poses:
                continue
            
            # Use same index for image if available
            img_idx = view_id if view_id < n_images else (view_id % n_images if n_images > 0 else -1)
            if img_idx < 0 or img_idx >= n_images:
                logger.debug(f"  View {view_id}: no image available")
                continue
                
            pose = self.scene.camera_poses[view_id]
            rgb_path = self.scene.image_paths[img_idx]
            if not rgb_path:
                continue
            
            rgb = load_image(rgb_path)
            if rgb is None:
                continue
            
            # Annotate image with candidate objects
            K = pose.intrinsics if pose.intrinsics is not None else np.array([
                [600, 0, 320], [0, 600, 240], [0, 0, 1]
            ])
            annotated = annotate_image_with_objects(
                rgb, candidates[:5], pose, K, highlight_ids=highlight_ids
            )
            if annotated is not None:
                annotated_images.append(annotated)
                logger.debug(f"  Annotated view {view_id}")
        
        if not annotated_images:
            logger.warning("  Failed to annotate any images")
            return None
        
        # Step 3: Optionally generate BEV
        bev_image = None
        if query_info.use_bev:
            try:
                bev = generate_bev(self.scene.objects, resolution=0.02, size=(400, 400))
                if bev is not None and anchor:
                    bev_image = annotate_bev_with_distances(
                        bev, anchor, candidates[:5], resolution=0.02
                    )
                    logger.debug("  Generated annotated BEV")
            except Exception as e:
                logger.warning(f"  BEV generation failed: {e}")
        
        # Step 4: Construct VLM input
        # Convert list to dict format expected by construct()
        view_images_dict = {view_ids[i]: img for i, img in enumerate(annotated_images)}
        
        vlm_input = self.vlm_constructor.construct(
            query_info=query_info,
            candidates=candidates[:5],
            view_images=view_images_dict,
            bev_image=bev_image,
            visibility_index=self.indices.visibility_index if self.indices else None,
        )
        logger.debug(f"  VLM input: {len(vlm_input.images)} images, prompt length={len(vlm_input.prompt)}")
        
        # Step 5: Call VLM
        try:
            vlm_response = self.vlm_client.infer(vlm_input)
            logger.debug(f"  VLM response: {vlm_response[:200] if vlm_response else 'None'}...")
            
            if vlm_response:
                # Step 6: Parse VLM output
                parser = VLMOutputParser(candidates[:5])
                result = parser.parse(vlm_response, query_info)
                if result.success:
                    return result
        except Exception as e:
            logger.error(f"  VLM inference error: {e}")
        
        return None
    
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
