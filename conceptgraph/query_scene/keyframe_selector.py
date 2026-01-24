"""
Keyframe Selector: Query-driven keyframe selection for VLM grounding.

This module implements the core functionality of selecting relevant keyframes
(RGB images) given a natural language query about a 3D scene.

Key Features:
1. Scene-aware query parsing: LLM maps query terms to scene object labels
2. CLIP-based semantic retrieval: Handles synonyms and semantic similarity
3. Visibility-based view selection: Finds views that best observe target objects
4. Joint coverage optimization: Selects views covering both target and anchor objects

Usage:
    selector = KeyframeSelector.from_scene_path("/path/to/scene")
    result = selector.select_keyframes("the pillow on the sofa", k=3)
    
    # result.keyframe_indices: [42, 67, 89]
    # result.keyframe_paths: [Path("frame000042.jpg"), ...]
    # result.target_objects: [ObjectNode(...), ...]
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from loguru import logger

# Optional imports
try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("open_clip not available, CLIP features will not work")


@dataclass
class SceneObject:
    """Lightweight object representation for keyframe selection."""
    
    obj_id: int
    category: str  # e.g., "table_lamp", "ottoman"
    centroid: np.ndarray  # 3D position [x, y, z]
    clip_feature: Optional[np.ndarray] = None  # CLIP visual feature
    
    # Optional rich information from affordance extraction
    object_tag: str = ""  # Refined tag e.g., "throw_pillow"
    summary: str = ""  # Description
    affordance_category: str = ""  # e.g., "lighting", "seating"
    co_objects: List[str] = field(default_factory=list)  # Related objects


@dataclass
class KeyframeResult:
    """Result of keyframe selection."""
    
    query: str
    target_term: str
    anchor_term: Optional[str]
    
    # Selected keyframes
    keyframe_indices: List[int]  # Frame indices (view_id)
    keyframe_paths: List[Path]  # Paths to RGB images
    
    # Matched objects
    target_objects: List[SceneObject]
    anchor_objects: List[SceneObject]
    
    # Scores and metadata
    selection_scores: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Query: {self.query}",
            f"Target: '{self.target_term}' -> {len(self.target_objects)} objects",
        ]
        if self.anchor_term:
            lines.append(f"Anchor: '{self.anchor_term}' -> {len(self.anchor_objects)} objects")
        lines.append(f"Selected {len(self.keyframe_indices)} keyframes: {self.keyframe_indices}")
        return "\n".join(lines)


class KeyframeSelector:
    """Query-driven keyframe selector for VLM grounding.
    
    This class encapsulates the complete pipeline from query to keyframe selection:
    1. Load scene data (objects, poses, images)
    2. Build object-view visibility index
    3. Parse query with scene context
    4. Select optimal keyframes
    """
    
    def __init__(
        self,
        scene_path: Path,
        pcd_file: Optional[Path] = None,
        affordance_file: Optional[Path] = None,
        stride: int = 5,
        llm_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
    ):
        """Initialize keyframe selector.
        
        Args:
            scene_path: Root path of the scene
            pcd_file: Path to .pkl.gz file with 3D objects
            affordance_file: Path to object_affordances.json (optional)
            stride: Frame stride used during mapping
            llm_url: LLM server URL for query parsing
            llm_model: LLM model name
        """
        self.scene_path = Path(scene_path)
        self.stride = stride
        self.llm_url = llm_url
        self.llm_model = llm_model
        
        # Data containers
        self.objects: List[SceneObject] = []
        self.object_features: Optional[np.ndarray] = None  # (N, D)
        self.camera_poses: List[np.ndarray] = []  # List of 4x4 matrices
        self.image_paths: List[Path] = []
        self.depth_paths: List[Path] = []
        self.intrinsics: np.ndarray = np.array([
            [600.0, 0, 599.5],
            [0, 600.0, 339.5],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # Category index for query parsing
        self.scene_categories: List[str] = []
        
        # Bidirectional visibility index
        # object_to_views: obj_id -> [(view_id, score), ...] sorted by score desc
        self.object_to_views: Dict[int, List[Tuple[int, float]]] = {}
        # view_to_objects: view_id -> [(obj_id, score), ...] sorted by score desc
        self.view_to_objects: Dict[int, List[Tuple[int, float]]] = {}
        
        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_tokenizer = None
        
        # Load data
        self._load_scene(pcd_file, affordance_file)
        self._load_or_build_visibility_index()
    
    @classmethod
    def from_scene_path(
        cls,
        scene_path: str,
        stride: int = 5,
        **kwargs,
    ) -> KeyframeSelector:
        """Create selector from scene path, auto-detecting files.
        
        Args:
            scene_path: Path to scene directory
            stride: Frame stride
            **kwargs: Additional arguments passed to __init__
        
        Returns:
            Initialized KeyframeSelector
        """
        scene_path = Path(scene_path)
        
        # Auto-detect PCD file
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))
        
        pcd_file = pcd_files[0] if pcd_files else None
        logger.info(f"Auto-detected PCD file: {pcd_file}")
        
        # Auto-detect affordance file
        affordance_file = scene_path / "sg_cache_detect" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = scene_path / "sg_cache" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = None
            logger.warning("No affordance file found, using basic object info")
        else:
            logger.info(f"Using affordance file: {affordance_file}")
        
        return cls(scene_path, pcd_file, affordance_file, stride, **kwargs)
    
    def _load_scene(self, pcd_file: Optional[Path], affordance_file: Optional[Path]) -> None:
        """Load scene data from files."""
        logger.info(f"Loading scene from: {self.scene_path}")
        
        # Load 3D objects from PCD file
        if pcd_file and pcd_file.exists():
            self._load_objects_from_pcd(pcd_file)
        
        # Enrich with affordance data if available
        if affordance_file and affordance_file.exists():
            self._load_affordances(affordance_file)
        
        # Load camera poses
        self._load_camera_poses()
        
        # Set image paths
        self._set_image_paths()
        
        # Build category index
        self.scene_categories = list(set(
            obj.object_tag if obj.object_tag else obj.category 
            for obj in self.objects
        ))
        
        logger.success(f"Loaded {len(self.objects)} objects, {len(self.camera_poses)} poses")
        logger.info(f"Scene categories: {self.scene_categories[:20]}...")
    
    def _load_objects_from_pcd(self, pcd_file: Path) -> None:
        """Load objects from ConceptGraphs PCD file."""
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        
        raw_objects = data.get('objects', [])
        features = []
        
        for i, obj in enumerate(raw_objects):
            # Get point cloud for centroid
            pcd_np = obj.get('pcd_np')
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            pcd_np = np.asarray(pcd_np, dtype=np.float32)
            centroid = pcd_np.mean(axis=0)
            
            # Get CLIP feature
            clip_ft = obj.get('clip_ft')
            if clip_ft is not None:
                if hasattr(clip_ft, 'cpu'):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.asarray(clip_ft, dtype=np.float32).flatten()
                features.append(clip_ft)
            else:
                features.append(None)
            
            # Get category from class_name list
            class_names = obj.get('class_name', [])
            if class_names:
                valid = [n for n in class_names if n and n.lower() not in ['item', 'object']]
                category = Counter(valid).most_common(1)[0][0] if valid else class_names[0]
            else:
                category = f"object_{i}"
            
            scene_obj = SceneObject(
                obj_id=len(self.objects),
                category=category,
                centroid=centroid,
                clip_feature=clip_ft,
            )
            self.objects.append(scene_obj)
        
        # Stack features for batch similarity computation
        valid_features = [f for f in features if f is not None]
        if valid_features:
            self.object_features = np.stack(valid_features)
            # L2 normalize
            norms = np.linalg.norm(self.object_features, axis=1, keepdims=True)
            self.object_features = self.object_features / (norms + 1e-8)
    
    def _load_affordances(self, affordance_file: Path) -> None:
        """Load and merge affordance data."""
        with open(affordance_file) as f:
            affordances = json.load(f)
        
        # Create mapping by ID
        aff_by_id = {a['id']: a for a in affordances}
        
        for obj in self.objects:
            if obj.obj_id in aff_by_id:
                aff = aff_by_id[obj.obj_id]
                obj.object_tag = aff.get('object_tag', '')
                obj.summary = aff.get('summary', '')
                obj.affordance_category = aff.get('category', '')
                
                affs = aff.get('affordances', {})
                obj.co_objects = affs.get('co_objects', [])
    
    def _load_camera_poses(self) -> None:
        """Load camera poses from trajectory file."""
        traj_file = self.scene_path / 'traj.txt'
        if not traj_file.exists():
            logger.warning(f"Trajectory file not found: {traj_file}")
            return
        
        with open(traj_file) as f:
            lines = f.readlines()
        
        # Check format: each line could be 16 numbers (4x4 matrix) or other formats
        first_line_nums = len(lines[0].split()) if lines else 0
        
        all_poses = []
        if first_line_nums == 16:
            # Each line is a full 4x4 matrix
            for line in lines:
                nums = [float(x) for x in line.split()]
                if len(nums) == 16:
                    pose = np.array(nums).reshape(4, 4)
                    all_poses.append(pose)
        else:
            # Traditional format: 4 lines per matrix
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    try:
                        pose = np.array([
                            [float(x) for x in lines[i].split()],
                            [float(x) for x in lines[i+1].split()],
                            [float(x) for x in lines[i+2].split()],
                            [float(x) for x in lines[i+3].split()],
                        ])
                        all_poses.append(pose)
                    except:
                        continue
        
        # Apply stride
        self.camera_poses = [all_poses[i] for i in range(0, len(all_poses), self.stride)
                           if i < len(all_poses)]
    
    def _set_image_paths(self) -> None:
        """Set paths to RGB and depth images."""
        results_dir = self.scene_path / 'results'
        
        # Find all RGB images
        all_images = sorted(results_dir.glob('frame*.jpg'))
        if not all_images:
            all_images = sorted(results_dir.glob('*.jpg'))
        
        # Apply stride
        self.image_paths = [all_images[i] for i in range(0, len(all_images), self.stride)
                          if i < len(all_images)]
        
        # Find depth images
        all_depths = sorted(results_dir.glob('depth*.png'))
        self.depth_paths = [all_depths[i] for i in range(0, len(all_depths), self.stride)
                          if i < len(all_depths)]
    
    def _load_or_build_visibility_index(self) -> None:
        """Load precomputed visibility index or build online.
        
        Prefers offline index from scene_path/indices/visibility_index.pkl
        Falls back to online computation if not available.
        """
        index_path = self.scene_path / "indices" / "visibility_index.pkl"
        
        if index_path.exists():
            try:
                self._load_visibility_index(index_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load visibility index: {e}")
        
        logger.warning(
            f"Visibility index not found at {index_path}. "
            "Building online (slower). Run 6b_build_visibility_index.sh for faster inference."
        )
        self._build_visibility_index_online()
    
    def _load_visibility_index(self, index_path: Path) -> None:
        """Load precomputed bidirectional visibility index from file."""
        logger.info(f"Loading visibility index from {index_path}")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        
        # Support both old format (visibility_index) and new format (object_to_views, view_to_objects)
        if 'object_to_views' in data:
            # New bidirectional format
            raw_obj_to_views = data.get('object_to_views', {})
            raw_view_to_objs = data.get('view_to_objects', {})
            
            self.object_to_views = {int(k): v for k, v in raw_obj_to_views.items()}
            self.view_to_objects = {int(k): v for k, v in raw_view_to_objs.items()}
        else:
            # Old format - only object_to_views, build reverse index
            raw_index = data.get('visibility_index', {})
            self.object_to_views = {int(k): v for k, v in raw_index.items()}
            
            # Build reverse index
            self.view_to_objects = {}
            for obj_id, views in self.object_to_views.items():
                for view_id, score in views:
                    if view_id not in self.view_to_objects:
                        self.view_to_objects[view_id] = []
                    self.view_to_objects[view_id].append((obj_id, score))
            
            # Sort by score
            for view_id in self.view_to_objects:
                self.view_to_objects[view_id].sort(key=lambda x: x[1], reverse=True)
        
        # Check stride consistency
        saved_stride = metadata.get('stride', self.stride)
        if saved_stride != self.stride:
            logger.warning(
                f"Stride mismatch: saved={saved_stride}, current={self.stride}. "
                "View indices may be incorrect."
            )
        
        logger.success(
            f"Loaded bidirectional visibility index: "
            f"{len(self.object_to_views)} objects, {len(self.view_to_objects)} views"
        )
    
    def _build_visibility_index_online(self) -> None:
        """Build bidirectional visibility index online (fallback)."""
        logger.info("Building visibility index online...")
        
        max_distance = 5.0
        self.view_to_objects = {i: [] for i in range(len(self.camera_poses))}
        
        for obj in self.objects:
            scores = []
            for view_id, pose in enumerate(self.camera_poses):
                cam_pos = pose[:3, 3]
                distance = np.linalg.norm(obj.centroid - cam_pos)
                if distance > max_distance:
                    continue
                dist_score = max(0, 1 - distance / max_distance)
                view_dir = (obj.centroid - cam_pos)
                view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                cam_forward = -pose[:3, 2]
                angle_score = max(0, np.dot(view_dir, cam_forward))
                combined = 0.6 * dist_score + 0.4 * angle_score
                if combined > 0.1:
                    scores.append((view_id, combined))
                    self.view_to_objects[view_id].append((obj.obj_id, combined))
            scores.sort(key=lambda x: x[1], reverse=True)
            self.object_to_views[obj.obj_id] = scores
        
        for vid in list(self.view_to_objects.keys()):
            if self.view_to_objects[vid]:
                self.view_to_objects[vid].sort(key=lambda x: x[1], reverse=True)
            else:
                del self.view_to_objects[vid]
        
        logger.success(f"Built index: {len(self.object_to_views)} objects, {len(self.view_to_objects)} views")
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for text encoding."""
        if self._clip_model is not None:
            return
        
        if not HAS_CLIP:
            logger.error("CLIP not available")
            return
        
        logger.info("Loading CLIP model...")
        
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            self._clip_model = model.eval()
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.cuda()
            
            logger.success("CLIP model loaded")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
    
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to CLIP feature."""
        self._load_clip_model()
        
        if self._clip_model is None:
            return None
        
        try:
            tokens = self._clip_tokenizer([text])
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            
            with torch.no_grad():
                feat = self._clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Text encoding failed: {e}")
            return None
    
    def parse_query(self, query: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse query to extract target, anchor, and relation.
        
        Uses LLM with scene category context to map query terms to scene labels.
        LLM is the only parsing method - no regex fallback.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (target, anchor, relation)
        """
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = self._parse_with_llm(query)
                # Validate result - target should not be empty
                if result[0] and len(result[0]) > 0:
                    return result
                else:
                    logger.warning(f"LLM returned empty target, retrying...")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM parsing attempt {attempt + 1} failed: {e}")
        
        # If all retries failed, return query as target with no anchor
        logger.error(f"LLM parsing failed after {max_retries} attempts: {last_error}")
        return (query.strip(), None, None)
    
    def _parse_with_llm(self, query: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse query using LLM with scene category context."""
        import requests
        
        category_list = ", ".join(sorted(set(self.scene_categories)))
        
        prompt = f'''Parse this query to find objects in a 3D scene.

**Scene contains these objects:** [{category_list}]

Query: "{query}"

RULES:
1. "target" = the MAIN object to find (the one being searched for)
2. "anchor" = a REFERENCE object used for spatial context (if any)
3. For "X on/near/beside Y": target=X, anchor=Y
4. Map synonyms to scene labels: pillow→throw_pillow, couch→sofa, lamp→table_lamp
5. If no spatial relation, anchor and relation are null

Examples:
- "the pillow on the sofa" → {{"target": "throw_pillow", "anchor": "sofa", "relation": "on"}}
- "lamp near the couch" → {{"target": "table_lamp", "anchor": "sofa", "relation": "near"}}
- "red chair" → {{"target": "chair", "anchor": null, "relation": null}}
- "ottoman" → {{"target": "ottoman", "anchor": null, "relation": null}}

Return ONLY JSON:
{{"target": "<main object to find>", "anchor": "<reference object or null>", "relation": "<on/near/beside/under or null>"}}
'''
        
        # Try Ollama/OpenAI compatible API
        try:
            r = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={"model": self.llm_model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            if r.ok:
                response = r.json()["choices"][0]["message"]["content"]
            else:
                raise Exception("API failed")
        except:
            # Try Ollama native API
            r = requests.post(
                f"{self.llm_url}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=30
            )
            if r.ok:
                response = r.json().get("response", "")
            else:
                raise Exception("LLM unavailable")
        
        # Parse JSON from response
        import re
        match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                target = data.get("target", "").strip()
                anchor = data.get("anchor")
                relation = data.get("relation")
                
                # Clean up null strings
                if anchor and anchor.lower() in ["null", "none", ""]:
                    anchor = None
                if relation and relation.lower() in ["null", "none", ""]:
                    relation = None
                
                if target:
                    return (target, anchor, relation)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
        
        # If JSON parsing failed, raise exception to trigger retry
        raise ValueError(f"Failed to parse LLM response: {response[:200]}")
    
    def find_objects(self, query_term: str, top_k: int = 10) -> List[SceneObject]:
        """Find objects matching a query term.
        
        Two-stage matching:
        1. Exact/fuzzy string match on object_tag or category
        2. CLIP semantic similarity fallback
        
        Args:
            query_term: Search term
            top_k: Maximum objects to return
            
        Returns:
            List of matching SceneObject
        """
        query_lower = query_term.lower()
        
        # Stage 1: String matching
        matches = []
        for obj in self.objects:
            tag = (obj.object_tag or obj.category).lower()
            # Exact substring match
            if query_lower in tag or tag in query_lower:
                matches.append((obj, 1.0))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in matches[:top_k]]
        
        # Stage 2: CLIP semantic matching
        if self.object_features is not None:
            query_feat = self._encode_text(query_term)
            if query_feat is not None:
                # Compute similarities
                similarities = self.object_features @ query_feat
                top_indices = np.argsort(-similarities)[:top_k]
                
                # Filter by minimum similarity
                min_sim = 0.2
                matches = [
                    (self.objects[i], similarities[i])
                    for i in top_indices if similarities[i] > min_sim
                ]
                
                if matches:
                    logger.info(f"CLIP matched '{query_term}' -> {[(m[0].object_tag or m[0].category, f'{m[1]:.2f}') for m in matches[:5]]}")
                    return [m[0] for m in matches]
        
        logger.warning(f"No objects found for '{query_term}'")
        return []
    
    def get_best_views_for_object(self, obj_id: int, top_k: int = 5) -> List[int]:
        """Get best view indices for a single object."""
        views = self.object_to_views.get(obj_id, [])
        return [v[0] for v in views[:top_k]]
    
    def get_joint_coverage_views(
        self,
        object_ids: List[int],
        max_views: int = 3,
    ) -> List[int]:
        """Greedy selection of views that maximize joint coverage of objects.
        
        Args:
            object_ids: Object IDs to cover
            max_views: Maximum views to select
            
        Returns:
            List of selected view indices
        """
        if not object_ids:
            return []
        
        # Collect all candidate views
        candidate_views: Set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)
        
        if not candidate_views:
            return []
        
        # Build view -> {obj_id: score} mapping
        view_scores: Dict[int, Dict[int, float]] = {}
        for obj_id in object_ids:
            for view_id, score in self.object_to_views.get(obj_id, []):
                if view_id not in view_scores:
                    view_scores[view_id] = {}
                view_scores[view_id][obj_id] = score
        
        # Greedy selection
        selected = []
        covered_quality = {obj_id: 0.0 for obj_id in object_ids}
        
        for _ in range(max_views):
            best_view, best_gain = None, 0.0
            
            for view_id in candidate_views - set(selected):
                # Compute marginal gain
                gain = 0.0
                for obj_id in object_ids:
                    obj_score = view_scores.get(view_id, {}).get(obj_id, 0)
                    if obj_score > covered_quality[obj_id]:
                        gain += obj_score - covered_quality[obj_id]
                
                if gain > best_gain:
                    best_gain, best_view = gain, view_id
            
            if best_view is None:
                break
            
            selected.append(best_view)
            
            # Update covered quality
            for obj_id in object_ids:
                obj_score = view_scores.get(best_view, {}).get(obj_id, 0)
                covered_quality[obj_id] = max(covered_quality[obj_id], obj_score)
        
        return selected
    
    def select_keyframes(
        self,
        query: str,
        k: int = 3,
        strategy: str = "joint_coverage",
    ) -> KeyframeResult:
        """Select k keyframes for a query.
        
        This is the main entry point for keyframe selection.
        
        Args:
            query: Natural language query
            k: Number of keyframes to select
            strategy: Selection strategy
                - "joint_coverage": Maximize coverage of all relevant objects
                - "best_per_object": Best view for each object (may exceed k)
                - "target_only": Only consider target objects
                
        Returns:
            KeyframeResult with selected keyframes and metadata
        """
        logger.info(f"Selecting {k} keyframes for: '{query}'")
        
        # Step 1: Parse query
        target_term, anchor_term, relation = self.parse_query(query)
        logger.info(f"Parsed: target='{target_term}', anchor='{anchor_term}', relation='{relation}'")
        
        # Step 2: Find matching objects
        target_objects = self.find_objects(target_term, top_k=10)
        anchor_objects = self.find_objects(anchor_term, top_k=5) if anchor_term else []
        
        if not target_objects:
            logger.warning(f"No target objects found for '{target_term}'")
            return KeyframeResult(
                query=query,
                target_term=target_term,
                anchor_term=anchor_term,
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={"error": "No matching objects"}
            )
        
        logger.info(f"Found {len(target_objects)} target(s): {[o.object_tag or o.category for o in target_objects[:5]]}")
        if anchor_objects:
            logger.info(f"Found {len(anchor_objects)} anchor(s): {[o.object_tag or o.category for o in anchor_objects[:3]]}")
        
        # Step 3: Spatial filtering (if anchor exists)
        if anchor_objects and relation:
            target_objects = self._spatial_filter(target_objects, anchor_objects[0], relation)
            logger.info(f"After spatial filter: {len(target_objects)} targets")
        
        # Step 4: Select keyframes
        all_object_ids = [obj.obj_id for obj in target_objects[:5]]
        if anchor_objects:
            all_object_ids.extend([obj.obj_id for obj in anchor_objects[:2]])
        
        if strategy == "joint_coverage":
            keyframe_indices = self.get_joint_coverage_views(all_object_ids, max_views=k)
        elif strategy == "target_only":
            keyframe_indices = self.get_joint_coverage_views(
                [obj.obj_id for obj in target_objects[:5]], max_views=k
            )
        else:  # best_per_object
            view_set = set()
            for obj in target_objects[:5] + anchor_objects[:2]:
                views = self.get_best_views_for_object(obj.obj_id, top_k=1)
                view_set.update(views)
            keyframe_indices = list(view_set)[:k]
        
        # Map to image paths
        keyframe_paths = []
        for idx in keyframe_indices:
            if idx < len(self.image_paths):
                keyframe_paths.append(self.image_paths[idx])
        
        logger.success(f"Selected keyframes: {keyframe_indices}")
        
        return KeyframeResult(
            query=query,
            target_term=target_term,
            anchor_term=anchor_term,
            keyframe_indices=keyframe_indices,
            keyframe_paths=keyframe_paths,
            target_objects=target_objects,
            anchor_objects=anchor_objects,
            metadata={
                "strategy": strategy,
                "all_object_ids": all_object_ids,
            }
        )
    
    def _spatial_filter(
        self,
        candidates: List[SceneObject],
        anchor: SceneObject,
        relation: str,
    ) -> List[SceneObject]:
        """Filter candidates by spatial relation to anchor."""
        relation_lower = relation.lower()
        
        # Determine distance threshold based on relation
        if "on" in relation_lower:
            # "on" implies close vertical relation
            max_horizontal = 0.5
            max_vertical = 1.0
        elif "beside" in relation_lower or "near" in relation_lower:
            max_horizontal = 1.5
            max_vertical = 1.5
        else:
            max_horizontal = 2.0
            max_vertical = 2.0
        
        filtered = []
        for obj in candidates:
            diff = obj.centroid - anchor.centroid
            horizontal_dist = np.linalg.norm(diff[:2])  # x, y
            vertical_dist = abs(diff[2])  # z
            
            if horizontal_dist < max_horizontal and vertical_dist < max_vertical:
                filtered.append(obj)
        
        return filtered if filtered else candidates
    
    def get_image(self, view_id: int) -> Optional[np.ndarray]:
        """Load RGB image for a view."""
        if view_id >= len(self.image_paths):
            return None
        
        import cv2
        img = cv2.imread(str(self.image_paths[view_id]))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None


# Convenience function
def select_keyframes(
    scene_path: str,
    query: str,
    k: int = 3,
    **kwargs,
) -> KeyframeResult:
    """Convenience function to select keyframes for a query.
    
    Args:
        scene_path: Path to scene directory
        query: Natural language query
        k: Number of keyframes
        **kwargs: Additional arguments for KeyframeSelector
        
    Returns:
        KeyframeResult
    """
    selector = KeyframeSelector.from_scene_path(scene_path, **kwargs)
    return selector.select_keyframes(query, k=k)
