#!/usr/bin/env python3
"""
物体-区域关系分类器
==================

分类物体与功能区域的关系类型：
- defining: 定义性（物体定义了区域）
- supporting: 支持性（物体支持区域功能）
- shared: 共享性（物体被多个区域共享）
- boundary: 边界性（物体位于区域边界）
"""

from typing import List, Dict, Optional, Tuple
from .data_structures import ObjectRegionRelation, ObjectInfo, FunctionalZone


# 定义性物体映射：zone_type -> defining_objects
DEFINING_OBJECTS = {
    "cooking_zone": ["stove", "oven", "cooktop", "range", "burner"],
    "washing_zone": ["sink", "dishwasher", "basin"],
    "storage_zone": ["refrigerator", "fridge", "freezer", "pantry"],
    "preparation_zone": ["counter", "countertop", "cutting_board"],
    "dining_zone": ["dining_table", "dinner_table"],
    "relaxation_zone": ["sofa", "couch", "armchair"],
    "entertainment_zone": ["tv", "television", "gaming_console"],
    "sleeping_zone": ["bed"],
    "work_zone": ["desk", "computer", "workstation"],
    "bathing_zone": ["bathtub", "shower"],
    "bathroom_zone": ["toilet"],
    "grooming_zone": ["bathroom_sink", "vanity", "mirror"],
    "dressing_zone": ["wardrobe", "closet"],
    "reading_zone": ["bookshelf", "reading_lamp"],
    "entry_zone": ["door", "entrance"],
}

# 边界物体（通常出现在区域交界处）
BOUNDARY_OBJECTS = [
    "island", "kitchen_island", "bar", "counter", "partition",
    "door", "doorway", "arch", "divider", "screen"
]

# 共享物体（通常在多个区域使用）
SHARED_OBJECTS = [
    "trash_can", "garbage_bin", "recycle_bin",
    "light", "lamp", "clock", "plant",
    "chair", "stool"
]


class ObjectRegionClassifier:
    """物体-区域关系分类器"""
    
    def __init__(self):
        self.defining_map = DEFINING_OBJECTS.copy()
        self.boundary_objects = set(BOUNDARY_OBJECTS)
        self.shared_objects = set(SHARED_OBJECTS)
    
    def classify(
        self, 
        object_info: ObjectInfo, 
        zone: FunctionalZone,
        other_zones: List[FunctionalZone] = None
    ) -> Tuple[ObjectRegionRelation, float]:
        """
        分类物体与区域的关系
        
        Args:
            object_info: 物体信息
            zone: 目标功能区域
            other_zones: 其他功能区域（用于判断共享性）
            
        Returns:
            Tuple[ObjectRegionRelation, float]: (关系类型, 置信度)
        """
        tag = object_info.object_tag.lower()
        zone_name = zone.zone_name.lower()
        
        # 1. 检查是否为定义性物体
        if self._is_defining_object(tag, zone_name):
            return ObjectRegionRelation.DEFINING, 0.95
        
        # 2. 检查是否为边界物体
        if self._is_boundary_object(tag, object_info, zone, other_zones):
            return ObjectRegionRelation.BOUNDARY, 0.8
        
        # 3. 检查是否为共享物体
        if self._is_shared_object(tag, object_info, zone, other_zones):
            return ObjectRegionRelation.SHARED, 0.75
        
        # 4. 默认为支持性物体
        return ObjectRegionRelation.SUPPORTING, 0.7
    
    def _is_defining_object(self, tag: str, zone_name: str) -> bool:
        """检查物体是否为区域的定义性物体"""
        # 检查zone名称中是否包含关键词
        for zone_type, defining_objs in self.defining_map.items():
            if zone_type in zone_name or zone_name in zone_type:
                for def_obj in defining_objs:
                    if def_obj in tag or tag in def_obj:
                        return True
        
        # 检查物体的typical_zones
        # 如果物体只在一个区域出现，且重要性高，也认为是定义性
        return False
    
    def _is_boundary_object(
        self, 
        tag: str, 
        object_info: ObjectInfo,
        zone: FunctionalZone,
        other_zones: List[FunctionalZone]
    ) -> bool:
        """检查物体是否为边界物体"""
        # 基于物体类型
        for boundary_obj in self.boundary_objects:
            if boundary_obj in tag or tag in boundary_obj:
                return True
        
        # 基于空间位置（如果有）
        if object_info.position and zone.spatial and other_zones:
            pos = object_info.position
            zone_center = zone.spatial.center
            
            # 检查是否在多个区域的边界
            for other_zone in other_zones:
                if other_zone.zone_id == zone.zone_id:
                    continue
                if other_zone.spatial:
                    other_center = other_zone.spatial.center
                    # 简化的边界检测：物体到两个区域中心的距离接近
                    dist_to_zone = sum((a - b) ** 2 for a, b in zip(pos, zone_center)) ** 0.5
                    dist_to_other = sum((a - b) ** 2 for a, b in zip(pos, other_center)) ** 0.5
                    if abs(dist_to_zone - dist_to_other) < 0.5:  # 距离差小于0.5m
                        return True
        
        return False
    
    def _is_shared_object(
        self, 
        tag: str,
        object_info: ObjectInfo,
        zone: FunctionalZone,
        other_zones: List[FunctionalZone]
    ) -> bool:
        """检查物体是否为共享物体"""
        # 基于物体类型
        for shared_obj in self.shared_objects:
            if shared_obj in tag or tag in shared_obj:
                return True
        
        # 基于typical_zones（如果物体的典型区域包含多个）
        if len(object_info.typical_zones) > 1:
            return True
        
        return False
    
    def classify_all(
        self, 
        objects: List[ObjectInfo], 
        zones: List[FunctionalZone]
    ) -> Dict[int, Dict[str, Tuple[ObjectRegionRelation, float]]]:
        """
        批量分类所有物体与所有区域的关系
        
        Args:
            objects: 物体列表
            zones: 功能区域列表
            
        Returns:
            Dict[int, Dict[str, Tuple]]: {object_id: {zone_id: (relation, confidence)}}
        """
        results = {}
        
        for obj in objects:
            results[obj.object_id] = {}
            for zone in zones:
                relation, conf = self.classify(obj, zone, zones)
                results[obj.object_id][zone.zone_id] = (relation, conf)
        
        return results
    
    def update_object_relations(
        self, 
        objects: List[ObjectInfo],
        zone: FunctionalZone,
        all_zones: List[FunctionalZone]
    ) -> List[ObjectInfo]:
        """
        更新物体列表中的关系类型
        
        Args:
            objects: 物体列表
            zone: 目标区域
            all_zones: 所有区域
            
        Returns:
            更新后的物体列表
        """
        updated = []
        for obj in objects:
            relation, conf = self.classify(obj, zone, all_zones)
            obj.relation_type = relation
            obj.confidence = conf
            updated.append(obj)
        
        return updated


def infer_defining_objects(zone_name: str) -> List[str]:
    """
    推断区域的定义性物体类型
    
    Args:
        zone_name: 区域名称
        
    Returns:
        可能的定义性物体类型列表
    """
    zone_name = zone_name.lower()
    
    results = []
    for zone_type, defining_objs in DEFINING_OBJECTS.items():
        if zone_type in zone_name or any(word in zone_name for word in zone_type.split('_')):
            results.extend(defining_objs)
    
    return list(set(results))


def estimate_relation_importance(relation: ObjectRegionRelation) -> float:
    """
    估计关系类型的重要性权重
    
    Args:
        relation: 关系类型
        
    Returns:
        重要性权重 (0-1)
    """
    importance_map = {
        ObjectRegionRelation.DEFINING: 1.0,
        ObjectRegionRelation.SUPPORTING: 0.7,
        ObjectRegionRelation.SHARED: 0.5,
        ObjectRegionRelation.BOUNDARY: 0.6
    }
    return importance_map.get(relation, 0.5)


if __name__ == "__main__":
    # 测试
    from .data_structures import SpatialInfo, EnhancedAffordance
    
    # 创建测试数据
    zone = FunctionalZone(
        zone_id="fz_0",
        zone_name="cooking_zone",
        parent_unit="su_0",
        primary_activity="cooking",
        spatial=SpatialInfo(
            center=[1.0, 0.5, 1.0],
            bounding_box={"min": [0, 0, 0], "max": [2, 1, 2]}
        )
    )
    
    objects = [
        ObjectInfo(
            object_id=0,
            object_tag="stove",
            relation_type=ObjectRegionRelation.SUPPORTING,
            position=[1.0, 0.5, 1.0]
        ),
        ObjectInfo(
            object_id=1,
            object_tag="pot",
            relation_type=ObjectRegionRelation.SUPPORTING,
            position=[1.1, 0.6, 1.0]
        ),
        ObjectInfo(
            object_id=2,
            object_tag="trash_can",
            relation_type=ObjectRegionRelation.SUPPORTING,
            position=[0.5, 0.3, 1.0]
        ),
    ]
    
    classifier = ObjectRegionClassifier()
    
    print("物体-区域关系分类测试:")
    print("=" * 50)
    
    for obj in objects:
        relation, conf = classifier.classify(obj, zone, [])
        print(f"{obj.object_tag}: {relation.value} (置信度: {conf:.2f})")
    
    print("\n推断cooking_zone的定义性物体:")
    print(infer_defining_objects("cooking_zone"))
