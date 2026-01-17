#!/usr/bin/env python3
"""
层次化场景图数据结构
====================

三层结构：
- Layer 1: SpatialUnit (空间单元) - 房间级别
- Layer 2: FunctionalZone (功能区域) - 活动区域级别
- Layer 3: ObjectCluster (物体群组) - 物体组级别
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import numpy as np


class ObjectRegionRelation(Enum):
    """物体-区域关系类型"""
    DEFINING = "defining"       # 定义性：物体定义了区域（如炉灶定义烹饪区）
    SUPPORTING = "supporting"   # 支持性：物体支持区域功能（如锅支持烹饪）
    SHARED = "shared"           # 共享性：物体被多个区域共享（如垃圾桶）
    BOUNDARY = "boundary"       # 边界性：物体位于区域边界（如吧台）


@dataclass
class EnhancedAffordance:
    """增强的Affordance结构"""
    action: str                          # 动作（如 cook, sit, store）
    context: str = ""                    # 使用场景（如 meal_preparation）
    duration: str = "short"              # 持续时间: short/medium/long
    co_objects: List[str] = field(default_factory=list)   # 配合物体
    posture: str = "standing"            # 姿态: standing/sitting/bending
    frequency: str = "occasional"        # 频率: frequent/occasional/rare
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "context": self.context,
            "duration": self.duration,
            "co_objects": self.co_objects,
            "posture": self.posture,
            "frequency": self.frequency
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "EnhancedAffordance":
        return cls(
            action=d.get("action", ""),
            context=d.get("context", ""),
            duration=d.get("duration", "short"),
            co_objects=d.get("co_objects", []),
            posture=d.get("posture", "standing"),
            frequency=d.get("frequency", "occasional")
        )


@dataclass
class ObjectInfo:
    """物体信息"""
    object_id: int
    object_tag: str
    relation_type: ObjectRegionRelation
    confidence: float = 1.0
    position: Optional[List[float]] = None           # 3D位置
    bbox_3d: Optional[Dict[str, List[float]]] = None # 3D边界框
    affordances: List[EnhancedAffordance] = field(default_factory=list)
    typical_zones: List[str] = field(default_factory=list)
    importance_score: float = 0.5
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "object_id": self.object_id,
            "object_tag": self.object_tag,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "position": self.position,
            "bbox_3d": self.bbox_3d,
            "affordances": [a.to_dict() for a in self.affordances],
            "typical_zones": self.typical_zones,
            "importance_score": self.importance_score,
            "reasoning": self.reasoning
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ObjectInfo":
        return cls(
            object_id=d["object_id"],
            object_tag=d["object_tag"],
            relation_type=ObjectRegionRelation(d.get("relation_type", "supporting")),
            confidence=d.get("confidence", 1.0),
            position=d.get("position"),
            bbox_3d=d.get("bbox_3d"),
            affordances=[EnhancedAffordance.from_dict(a) for a in d.get("affordances", [])],
            typical_zones=d.get("typical_zones", []),
            importance_score=d.get("importance_score", 0.5),
            reasoning=d.get("reasoning", "")
        )


@dataclass
class ObjectCluster:
    """Layer 3: 物体群组"""
    cluster_id: str
    cluster_name: str
    parent_zone: str                     # 所属功能区域ID
    cluster_affordance: str              # 群组功能（如 heat_food）
    objects: List[ObjectInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "parent_zone": self.parent_zone,
            "cluster_affordance": self.cluster_affordance,
            "objects": [o.to_dict() for o in self.objects]
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ObjectCluster":
        return cls(
            cluster_id=d["cluster_id"],
            cluster_name=d["cluster_name"],
            parent_zone=d["parent_zone"],
            cluster_affordance=d.get("cluster_affordance", ""),
            objects=[ObjectInfo.from_dict(o) for o in d.get("objects", [])]
        )


@dataclass
class TrajectoryEvidence:
    """轨迹行为证据"""
    dwell_time_seconds: float = 0.0      # 停留时间
    look_around_events: int = 0          # 环顾事件数
    traverse_count: int = 0              # 穿越次数
    importance_heatmap_value: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "dwell_time_seconds": self.dwell_time_seconds,
            "look_around_events": self.look_around_events,
            "traverse_count": self.traverse_count,
            "importance_heatmap_value": self.importance_heatmap_value
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TrajectoryEvidence":
        return cls(
            dwell_time_seconds=d.get("dwell_time_seconds", 0.0),
            look_around_events=d.get("look_around_events", 0),
            traverse_count=d.get("traverse_count", 0),
            importance_heatmap_value=d.get("importance_heatmap_value", 0.0)
        )


@dataclass
class SpatialInfo:
    """空间信息"""
    center: List[float]                  # 中心点 [x, y, z]
    bounding_box: Dict[str, List[float]] # {"min": [...], "max": [...]}
    area_m2: float = 0.0                 # 面积
    vertices: Optional[List[List[float]]] = None  # 2D多边形顶点
    
    def to_dict(self) -> Dict:
        return {
            "center": self.center,
            "bounding_box": self.bounding_box,
            "area_m2": self.area_m2,
            "vertices": self.vertices
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SpatialInfo":
        return cls(
            center=d["center"],
            bounding_box=d["bounding_box"],
            area_m2=d.get("area_m2", 0.0),
            vertices=d.get("vertices")
        )


@dataclass
class FunctionalZone:
    """Layer 2: 功能区域"""
    zone_id: str
    zone_name: str
    parent_unit: str                     # 所属空间单元ID
    primary_activity: str                # 主要活动（如 cooking）
    supported_activities: List[str] = field(default_factory=list)
    affordances: List[str] = field(default_factory=list)
    spatial: Optional[SpatialInfo] = None
    object_clusters: List[str] = field(default_factory=list)  # 物体群组ID列表
    objects: List[ObjectInfo] = field(default_factory=list)   # 直接包含的物体
    importance_score: float = 0.5
    trajectory_evidence: Optional[TrajectoryEvidence] = None
    defining_evidence: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "parent_unit": self.parent_unit,
            "primary_activity": self.primary_activity,
            "supported_activities": self.supported_activities,
            "affordances": self.affordances,
            "spatial": self.spatial.to_dict() if self.spatial else None,
            "object_clusters": self.object_clusters,
            "objects": [o.to_dict() for o in self.objects],
            "importance_score": self.importance_score,
            "trajectory_evidence": self.trajectory_evidence.to_dict() if self.trajectory_evidence else None,
            "defining_evidence": self.defining_evidence,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FunctionalZone":
        return cls(
            zone_id=d["zone_id"],
            zone_name=d["zone_name"],
            parent_unit=d.get("parent_unit", ""),
            primary_activity=d.get("primary_activity", ""),
            supported_activities=d.get("supported_activities", []),
            affordances=d.get("affordances", []),
            spatial=SpatialInfo.from_dict(d["spatial"]) if d.get("spatial") else None,
            object_clusters=d.get("object_clusters", []),
            objects=[ObjectInfo.from_dict(o) for o in d.get("objects", [])],
            importance_score=d.get("importance_score", 0.5),
            trajectory_evidence=TrajectoryEvidence.from_dict(d["trajectory_evidence"]) if d.get("trajectory_evidence") else None,
            defining_evidence=d.get("defining_evidence", {}),
            confidence=d.get("confidence", 0.8)
        )


@dataclass
class NavigationInfo:
    """导航信息"""
    entry_points: List[Dict[str, Any]] = field(default_factory=list)
    accessible_from: List[str] = field(default_factory=list)
    traversable: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "entry_points": self.entry_points,
            "accessible_from": self.accessible_from,
            "traversable": self.traversable
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "NavigationInfo":
        return cls(
            entry_points=d.get("entry_points", []),
            accessible_from=d.get("accessible_from", []),
            traversable=d.get("traversable", True)
        )


@dataclass
class SpatialUnit:
    """Layer 1: 空间单元（房间级别）"""
    unit_id: str
    unit_name: str
    unit_type: str = "room"              # room/corridor/open_space
    spatial: Optional[SpatialInfo] = None
    navigation: Optional[NavigationInfo] = None
    functional_zones: List[str] = field(default_factory=list)  # 功能区域ID列表
    
    def to_dict(self) -> Dict:
        return {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "unit_type": self.unit_type,
            "spatial": self.spatial.to_dict() if self.spatial else None,
            "navigation": self.navigation.to_dict() if self.navigation else None,
            "functional_zones": self.functional_zones
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SpatialUnit":
        return cls(
            unit_id=d["unit_id"],
            unit_name=d["unit_name"],
            unit_type=d.get("unit_type", "room"),
            spatial=SpatialInfo.from_dict(d["spatial"]) if d.get("spatial") else None,
            navigation=NavigationInfo.from_dict(d["navigation"]) if d.get("navigation") else None,
            functional_zones=d.get("functional_zones", [])
        )


@dataclass
class ZoneRelation:
    """区域间关系"""
    zone_from: str
    zone_to: str
    relation_type: str                   # adjacent/connected/overlapping
    boundary_indicator: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "zone_from": self.zone_from,
            "zone_to": self.zone_to,
            "relation_type": self.relation_type,
            "boundary_indicator": self.boundary_indicator
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ZoneRelation":
        return cls(
            zone_from=d["zone_from"],
            zone_to=d["zone_to"],
            relation_type=d["relation_type"],
            boundary_indicator=d.get("boundary_indicator", "")
        )


@dataclass
class TaskAffordances:
    """任务级别的Affordance信息"""
    navigation_goals: List[Dict[str, Any]] = field(default_factory=list)
    object_search_hints: Dict[str, List[str]] = field(default_factory=dict)
    task_zones: Dict[str, List[str]] = field(default_factory=dict)
    object_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "navigation_goals": self.navigation_goals,
            "object_search_hints": self.object_search_hints,
            "task_zones": self.task_zones,
            "object_distribution": self.object_distribution
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TaskAffordances":
        return cls(
            navigation_goals=d.get("navigation_goals", []),
            object_search_hints=d.get("object_search_hints", {}),
            task_zones=d.get("task_zones", {}),
            object_distribution=d.get("object_distribution", {})
        )


@dataclass
class HierarchicalSceneGraph:
    """层次化场景图"""
    scene_id: str
    spatial_units: List[SpatialUnit] = field(default_factory=list)
    functional_zones: List[FunctionalZone] = field(default_factory=list)
    object_clusters: List[ObjectCluster] = field(default_factory=list)
    zone_relations: List[ZoneRelation] = field(default_factory=list)
    task_affordances: Optional[TaskAffordances] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "spatial_units": [u.to_dict() for u in self.spatial_units],
            "functional_zones": [z.to_dict() for z in self.functional_zones],
            "object_clusters": [c.to_dict() for c in self.object_clusters],
            "zone_relations": [r.to_dict() for r in self.zone_relations],
            "task_affordances": self.task_affordances.to_dict() if self.task_affordances else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "HierarchicalSceneGraph":
        return cls(
            scene_id=d["scene_id"],
            spatial_units=[SpatialUnit.from_dict(u) for u in d.get("spatial_units", [])],
            functional_zones=[FunctionalZone.from_dict(z) for z in d.get("functional_zones", [])],
            object_clusters=[ObjectCluster.from_dict(c) for c in d.get("object_clusters", [])],
            zone_relations=[ZoneRelation.from_dict(r) for r in d.get("zone_relations", [])],
            task_affordances=TaskAffordances.from_dict(d["task_affordances"]) if d.get("task_affordances") else None,
            metadata=d.get("metadata", {})
        )
    
    def save(self, path: str):
        """保存到JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "HierarchicalSceneGraph":
        """从JSON文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def get_zone_by_id(self, zone_id: str) -> Optional[FunctionalZone]:
        """根据ID获取功能区域"""
        for zone in self.functional_zones:
            if zone.zone_id == zone_id:
                return zone
        return None
    
    def get_unit_by_id(self, unit_id: str) -> Optional[SpatialUnit]:
        """根据ID获取空间单元"""
        for unit in self.spatial_units:
            if unit.unit_id == unit_id:
                return unit
        return None
    
    def get_objects_in_zone(self, zone_id: str) -> List[ObjectInfo]:
        """获取区域中的所有物体"""
        zone = self.get_zone_by_id(zone_id)
        if zone:
            return zone.objects
        return []
    
    def get_defining_objects(self, zone_id: str) -> List[ObjectInfo]:
        """获取区域的定义性物体"""
        objects = self.get_objects_in_zone(zone_id)
        return [o for o in objects if o.relation_type == ObjectRegionRelation.DEFINING]
    
    def summary(self) -> str:
        """生成场景图摘要"""
        lines = [
            f"场景: {self.scene_id}",
            f"空间单元: {len(self.spatial_units)} 个",
            f"功能区域: {len(self.functional_zones)} 个",
            f"物体群组: {len(self.object_clusters)} 个",
            "",
            "层次结构:",
        ]
        
        for unit in self.spatial_units:
            lines.append(f"  📍 {unit.unit_name} ({unit.unit_type})")
            for zone_id in unit.functional_zones:
                zone = self.get_zone_by_id(zone_id)
                if zone:
                    n_objects = len(zone.objects)
                    n_defining = len([o for o in zone.objects if o.relation_type == ObjectRegionRelation.DEFINING])
                    lines.append(f"    └─ 🎯 {zone.zone_name} [{zone.primary_activity}] ({n_objects}物体, {n_defining}定义性)")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试数据结构
    scene = HierarchicalSceneGraph(scene_id="test_scene")
    
    # 创建一个功能区域
    zone = FunctionalZone(
        zone_id="fz_0",
        zone_name="cooking_zone",
        parent_unit="su_0",
        primary_activity="cooking",
        affordances=["cook", "fry", "boil"],
        spatial=SpatialInfo(
            center=[1.0, 0.5, 1.2],
            bounding_box={"min": [0, 0, 0.5], "max": [2, 1, 2]},
            area_m2=2.0
        )
    )
    
    # 添加物体
    zone.objects.append(ObjectInfo(
        object_id=0,
        object_tag="stove",
        relation_type=ObjectRegionRelation.DEFINING,
        affordances=[EnhancedAffordance(
            action="cook",
            context="meal_preparation",
            duration="medium",
            co_objects=["pot", "pan"]
        )],
        importance_score=0.9
    ))
    
    scene.functional_zones.append(zone)
    
    # 创建空间单元
    unit = SpatialUnit(
        unit_id="su_0",
        unit_name="Kitchen",
        unit_type="room",
        functional_zones=["fz_0"]
    )
    scene.spatial_units.append(unit)
    
    print(scene.summary())
    print("\nJSON:")
    print(json.dumps(scene.to_dict(), indent=2, ensure_ascii=False)[:1000] + "...")
