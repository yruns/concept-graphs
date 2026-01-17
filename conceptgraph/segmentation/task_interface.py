#!/usr/bin/env python3
"""
任务对齐接口 - 为下游任务提供便捷的查询接口
"""
from typing import List, Dict, Any, Optional
from .data_structures import HierarchicalSceneGraph, FunctionalZone, ObjectInfo, ObjectRegionRelation


class TaskInterface:
    """任务对齐接口"""
    
    def __init__(self, scene_graph: HierarchicalSceneGraph):
        self.sg = scene_graph
        self._build_indices()
    
    def _build_indices(self):
        """构建索引加速查询"""
        self.object_index = {}
        self.zone_index = {z.zone_id: z for z in self.sg.functional_zones}
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                if obj.object_tag not in self.object_index:
                    self.object_index[obj.object_tag] = []
                self.object_index[obj.object_tag].append((zone.zone_id, obj))
    
    def get_navigation_goal(self, zone_name: str) -> Optional[List[float]]:
        """获取区域的导航目标点"""
        for zone in self.sg.functional_zones:
            if zone_name in zone.zone_name or zone.zone_name in zone_name:
                if zone.spatial:
                    return zone.spatial.center
        return None
    
    def get_zone_for_task(self, task: str) -> List[str]:
        """获取适合执行任务的区域"""
        if self.sg.task_affordances:
            return self.sg.task_affordances.task_zones.get(task, [])
        results = []
        for zone in self.sg.functional_zones:
            if task in zone.supported_activities or task in zone.primary_activity:
                results.append(zone.zone_id)
        return results
    
    def find_object(self, object_tag: str) -> List[Dict[str, Any]]:
        """查找物体位置"""
        results = []
        tag_lower = object_tag.lower()
        for tag, locations in self.object_index.items():
            if tag_lower in tag or tag in tag_lower:
                for zone_id, obj in locations:
                    zone = self.zone_index.get(zone_id)
                    results.append({
                        "object_tag": obj.object_tag,
                        "object_id": obj.object_id,
                        "zone_id": zone_id,
                        "zone_name": zone.zone_name if zone else "",
                        "position": obj.position,
                        "relation_type": obj.relation_type.value
                    })
        return results
    
    def get_search_zones(self, object_tag: str) -> List[str]:
        """获取可能找到物体的区域"""
        if self.sg.task_affordances:
            hints = self.sg.task_affordances.object_search_hints.get(object_tag, [])
            if hints:
                return hints
        locations = self.find_object(object_tag)
        return list(set(loc["zone_name"] for loc in locations))
    
    def get_zone_info(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """获取区域详细信息"""
        zone = self.zone_index.get(zone_id)
        if not zone:
            return None
        defining_objs = [o for o in zone.objects if o.relation_type == ObjectRegionRelation.DEFINING]
        return {
            "zone_id": zone.zone_id,
            "zone_name": zone.zone_name,
            "primary_activity": zone.primary_activity,
            "center": zone.spatial.center if zone.spatial else None,
            "n_objects": len(zone.objects),
            "defining_objects": [o.object_tag for o in defining_objs],
            "confidence": zone.confidence
        }
    
    def get_scene_summary(self) -> Dict[str, Any]:
        """获取场景摘要"""
        return {
            "scene_id": self.sg.scene_id,
            "n_spatial_units": len(self.sg.spatial_units),
            "n_functional_zones": len(self.sg.functional_zones),
            "zones": [{"id": z.zone_id, "name": z.zone_name, "activity": z.primary_activity} 
                     for z in self.sg.functional_zones],
            "total_objects": sum(len(z.objects) for z in self.sg.functional_zones)
        }
    
    def query(self, query_type: str, **kwargs) -> Any:
        """统一查询接口"""
        if query_type == "navigate":
            return self.get_navigation_goal(kwargs.get("zone", ""))
        elif query_type == "find_object":
            return self.find_object(kwargs.get("object", ""))
        elif query_type == "task_zone":
            return self.get_zone_for_task(kwargs.get("task", ""))
        elif query_type == "zone_info":
            return self.get_zone_info(kwargs.get("zone_id", ""))
        elif query_type == "summary":
            return self.get_scene_summary()
        return None


def create_task_interface(scene_graph_path: str) -> TaskInterface:
    """从文件创建任务接口"""
    sg = HierarchicalSceneGraph.load(scene_graph_path)
    return TaskInterface(sg)
