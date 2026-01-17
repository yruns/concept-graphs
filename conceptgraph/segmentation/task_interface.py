#!/usr/bin/env python3
"""任务对齐接口 - 为下游任务提供便捷查询"""
from typing import List, Dict, Any, Optional
from .data_structures import HierarchicalSceneGraph, ObjectRegionRelation

class TaskInterface:
    def __init__(self, scene_graph: HierarchicalSceneGraph):
        self.sg = scene_graph
        self.zone_index = {z.zone_id: z for z in self.sg.functional_zones}
        self.object_index = {}
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                if obj.object_tag not in self.object_index:
                    self.object_index[obj.object_tag] = []
                self.object_index[obj.object_tag].append((zone.zone_id, obj))
    
    def get_navigation_goal(self, zone_name: str) -> Optional[List[float]]:
        for zone in self.sg.functional_zones:
            if zone_name in zone.zone_name and zone.spatial:
                return zone.spatial.center
        return None
    
    def find_object(self, object_tag: str) -> List[Dict[str, Any]]:
        results = []
        for tag, locations in self.object_index.items():
            if object_tag.lower() in tag or tag in object_tag.lower():
                for zone_id, obj in locations:
                    results.append({"tag": obj.object_tag, "zone": zone_id, "position": obj.position})
        return results
    
    def get_zone_for_task(self, task: str) -> List[str]:
        return [z.zone_id for z in self.sg.functional_zones if task in z.primary_activity]
    
    def get_scene_summary(self) -> Dict[str, Any]:
        return {"scene_id": self.sg.scene_id, "n_zones": len(self.sg.functional_zones),
                "zones": [{"id": z.zone_id, "name": z.zone_name} for z in self.sg.functional_zones]}

def create_task_interface(path: str) -> TaskInterface:
    return TaskInterface(HierarchicalSceneGraph.load(path))
