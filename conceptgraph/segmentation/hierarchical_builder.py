#!/usr/bin/env python3
"""
层次化场景图构建器 - 整合所有模块构建三层场景图
"""
import os
import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .data_structures import (
    HierarchicalSceneGraph, SpatialUnit, FunctionalZone, ObjectInfo,
    ObjectRegionRelation, SpatialInfo, TrajectoryEvidence, ZoneRelation, TaskAffordances
)
from .enhanced_affordance import EnhancedAffordanceExtractor
from .visibility_keyframe import VisibilityBasedKeyframeSelector, build_visibility_matrix
from .trajectory_behavior import TrajectoryBehaviorAnalyzer, TrajectoryBehaviorAnalysis
from .vlm_functional_analyzer import VLMFunctionalAnalyzer, FrameAnalysis
from .llm_zone_inference import LLMZoneInference, ZoneInferenceResult, ObjectAssignmentResult


class HierarchicalSceneBuilder:
    """层次化场景图构建器"""
    
    def __init__(self, scene_path: str, stride: int = 5, n_keyframes: int = 15,
                 visibility_radius: float = 3.0, use_vlm: bool = True,
                 use_llm: bool = True, llm_base_url: str = None):
        self.scene_path = Path(scene_path)
        self.stride = stride
        self.n_keyframes = n_keyframes
        self.visibility_radius = visibility_radius
        self.use_vlm = use_vlm
        self.use_llm = use_llm
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://10.21.231.7:8005")
        
        self.poses = None
        self.objects = None
        self.visibility_matrix = None
        self.object_affordances = []
        self.keyframes = []
        self.trajectory_analysis = None
        self.vlm_analysis = []
        self.zones_result = None
        self.assignments_result = None
    
    def build(self) -> HierarchicalSceneGraph:
        """构建层次化场景图"""
        print("=" * 60)
        print("开始构建层次化场景图")
        print("=" * 60)
        
        print("\n[1/7] 加载场景数据...")
        self._load_data()
        print("\n[2/7] 提取物体Affordance...")
        self._extract_affordances()
        print("\n[3/7] 选取关键帧...")
        self._select_keyframes()
        print("\n[4/7] 分析轨迹行为...")
        self._analyze_trajectory()
        
        if self.use_vlm:
            print("\n[5/7] VLM分析关键帧...")
            self._analyze_keyframes_with_vlm()
        else:
            print("\n[5/7] 跳过VLM分析")
        
        print("\n[6/7] LLM推理功能区域...")
        self._infer_zones_and_assignments()
        print("\n[7/7] 构建层次化场景图...")
        scene_graph = self._build_scene_graph()
        
        print("\n" + "=" * 60)
        print("构建完成!")
        print(scene_graph.summary())
        return scene_graph
    
    def _load_data(self):
        pose_file = self.scene_path / 'traj.txt'
        if not pose_file.exists():
            pose_file = self.scene_path / 'traj_w_c.txt'
        
        all_poses = []
        with open(pose_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    all_poses.append([float(p) for p in parts[:7]])
        self.poses = np.array(all_poses[::self.stride])
        print(f"  位姿: {len(self.poses)} 帧")
        
        pcd_files = list((self.scene_path / 'pcd_saves').glob('*_post.pkl.gz'))
        if not pcd_files:
            pcd_files = list((self.scene_path / 'pcd_saves').glob('*.pkl.gz'))
        if pcd_files:
            with gzip.open(pcd_files[0], 'rb') as f:
                data = pickle.load(f)
            self.objects = data.get('objects', [])
            print(f"  3D物体: {len(self.objects)} 个")
        else:
            self.objects = []
        
        if self.objects and len(self.poses) > 0:
            self.visibility_matrix = build_visibility_matrix(
                self.objects, self.poses, self.visibility_radius)
    
    def _extract_affordances(self):
        captions = {}
        captions_file = self.scene_path / 'captions.json'
        if captions_file.exists():
            with open(captions_file) as f:
                for item in json.load(f):
                    captions[item.get('id', item.get('object_id'))] = item.get('caption', '')
        extractor = EnhancedAffordanceExtractor(use_llm=False)
        self.object_affordances = extractor.extract_affordances(self.objects, captions)
        print(f"  提取了 {len(self.object_affordances)} 个物体的Affordance")
    
    def _select_keyframes(self):
        if self.visibility_matrix is None or self.visibility_matrix.size == 0:
            n_frames = len(self.poses)
            indices = np.linspace(0, n_frames - 1, self.n_keyframes, dtype=int)
            self.keyframes = [{"frame_idx": int(i), "visible_objects": []} for i in indices]
        else:
            selector = VisibilityBasedKeyframeSelector(
                self.visibility_matrix, self.n_keyframes, self.stride)
            self.keyframes = [kf.to_dict() for kf in selector.select()]
        print(f"  选取了 {len(self.keyframes)} 个关键帧")
    
    def _analyze_trajectory(self):
        if len(self.poses) < 10:
            self.trajectory_analysis = TrajectoryBehaviorAnalysis()
            return
        analyzer = TrajectoryBehaviorAnalyzer(self.poses, 30.0 / self.stride, self.stride)
        self.trajectory_analysis = analyzer.analyze()
        print(f"  停留点: {len(self.trajectory_analysis.dwell_points)} 个")
    
    def _analyze_keyframes_with_vlm(self):
        rgb_dir = self.scene_path / 'rgb'
        if not rgb_dir.exists():
            rgb_dir = self.scene_path / 'results'
        if not rgb_dir.exists():
            return
        image_files = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
        if not image_files:
            return
        
        analyzer = VLMFunctionalAnalyzer(base_url=self.llm_base_url)
        for i, kf in enumerate(self.keyframes[:min(8, len(self.keyframes))]):
            frame_idx = kf["frame_idx"]
            original_idx = frame_idx * self.stride
            if original_idx < len(image_files):
                print(f"  分析关键帧 {i+1}: 帧 {frame_idx}")
                try:
                    analysis = analyzer.analyze_single_frame(str(image_files[original_idx]), frame_idx)
                    self.vlm_analysis.append(analysis)
                except Exception as e:
                    print(f"    失败: {e}")
        print(f"  完成 {len(self.vlm_analysis)} 帧VLM分析")
    
    def _infer_zones_and_assignments(self):
        object_positions = {o.object_id: o.position for o in self.object_affordances if o.position}
        if self.use_llm:
            inference = LLMZoneInference(base_url=self.llm_base_url)
            self.zones_result, self.assignments_result, _ = inference.run_inference(
                self.vlm_analysis, self.object_affordances, self.trajectory_analysis, object_positions)
        else:
            self.zones_result, self.assignments_result = self._fallback_inference()
    
    def _fallback_inference(self):
        zone_names = set()
        for obj in self.object_affordances:
            zone_names.update(obj.typical_zones)
        zones = [FunctionalZone(f"fz_{i}", name, "su_0", name.replace("_zone", ""), confidence=0.6) 
                 for i, name in enumerate(zone_names)]
        assignments = []
        zone_map = {z.zone_name: z.zone_id for z in zones}
        for obj in self.object_affordances:
            if obj.typical_zones:
                zone_id = zone_map.get(obj.typical_zones[0])
                if zone_id:
                    assignments.append({"object_id": obj.object_id, "object_tag": obj.object_tag,
                                       "assigned_zone": zone_id, "relation_type": "supporting"})
        return ZoneInferenceResult(zones=zones), ObjectAssignmentResult(assignments=assignments)
    
    def _build_scene_graph(self) -> HierarchicalSceneGraph:
        scene_id = self.scene_path.name
        scene_graph = HierarchicalSceneGraph(scene_id=scene_id, metadata={"n_objects": len(self.objects)})
        
        assignment_map = {}
        for a in self.assignments_result.assignments:
            zone_id = a["assigned_zone"]
            if zone_id not in assignment_map:
                assignment_map[zone_id] = []
            assignment_map[zone_id].append(a)
        
        for zone in self.zones_result.zones:
            zone_objects = []
            for a in assignment_map.get(zone.zone_id, []):
                obj_id = a["object_id"]
                if obj_id < len(self.object_affordances):
                    obj_info = self.object_affordances[obj_id]
                    obj_info.relation_type = ObjectRegionRelation(a.get("relation_type", "supporting"))
                    zone_objects.append(obj_info)
            zone.objects = zone_objects
            if zone_objects:
                positions = [o.position for o in zone_objects if o.position]
                if positions:
                    positions = np.array(positions)
                    zone.spatial = SpatialInfo(
                        center=positions.mean(axis=0).tolist(),
                        bounding_box={"min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()})
            scene_graph.functional_zones.append(zone)
        
        unit = SpatialUnit("su_0", f"{scene_id}_room", "room", 
                          functional_zones=[z.zone_id for z in scene_graph.functional_zones])
        scene_graph.spatial_units.append(unit)
        scene_graph.task_affordances = self._build_task_affordances(scene_graph)
        return scene_graph
    
    def _build_task_affordances(self, sg):
        task_aff = TaskAffordances()
        for zone in sg.functional_zones:
            if zone.spatial:
                task_aff.navigation_goals.append({"zone_id": zone.zone_id, "position": zone.spatial.center})
            for obj in zone.objects:
                if obj.object_tag not in task_aff.object_search_hints:
                    task_aff.object_search_hints[obj.object_tag] = []
                task_aff.object_search_hints[obj.object_tag].append(zone.zone_name)
        return task_aff


def build_hierarchical_scene_graph(scene_path, output_path=None, **kwargs):
    builder = HierarchicalSceneBuilder(scene_path, **kwargs)
    scene_graph = builder.build()
    if output_path:
        scene_graph.save(output_path)
        print(f"\n场景图保存到: {output_path}")
    return scene_graph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--no_vlm", action="store_true")
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--llm_url", type=str, default=None, help="LLM服务地址，默认从LLM_BASE_URL环境变量读取")
    args = parser.parse_args()
    build_hierarchical_scene_graph(args.scene_path, args.output, stride=args.stride,
                                   use_vlm=not args.no_vlm, use_llm=not args.no_llm, llm_base_url=args.llm_url)
