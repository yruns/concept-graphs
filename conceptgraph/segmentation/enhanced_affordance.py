#!/usr/bin/env python3
"""
增强的Affordance提取器
====================

基于LLM从物体描述中提取增强的功能属性：
- 动作(action)
- 上下文(context)
- 持续时间(duration)
- 配合物体(co_objects)
- 姿态(posture)
- 频率(frequency)
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_structures import (
    EnhancedAffordance, 
    ObjectInfo, 
    ObjectRegionRelation
)


# 默认的物体-功能映射（作为fallback）
DEFAULT_AFFORDANCE_MAP = {
    # 厨房物体
    "stove": {"actions": ["cook", "heat", "fry"], "zones": ["cooking_zone"], "importance": 0.95},
    "oven": {"actions": ["bake", "roast", "heat"], "zones": ["cooking_zone"], "importance": 0.9},
    "microwave": {"actions": ["heat", "defrost"], "zones": ["cooking_zone", "storage_zone"], "importance": 0.7},
    "refrigerator": {"actions": ["store", "cool", "preserve"], "zones": ["storage_zone"], "importance": 0.9},
    "fridge": {"actions": ["store", "cool", "preserve"], "zones": ["storage_zone"], "importance": 0.9},
    "sink": {"actions": ["wash", "clean", "rinse"], "zones": ["washing_zone"], "importance": 0.95},
    "dishwasher": {"actions": ["wash", "clean"], "zones": ["washing_zone"], "importance": 0.8},
    "counter": {"actions": ["prepare", "place", "work"], "zones": ["preparation_zone"], "importance": 0.7},
    "countertop": {"actions": ["prepare", "place", "work"], "zones": ["preparation_zone"], "importance": 0.7},
    "cabinet": {"actions": ["store", "organize"], "zones": ["storage_zone"], "importance": 0.6},
    "drawer": {"actions": ["store", "organize"], "zones": ["storage_zone"], "importance": 0.5},
    "pot": {"actions": ["cook", "boil"], "zones": ["cooking_zone"], "importance": 0.6},
    "pan": {"actions": ["fry", "cook"], "zones": ["cooking_zone"], "importance": 0.6},
    "knife": {"actions": ["cut", "slice", "chop"], "zones": ["preparation_zone"], "importance": 0.6},
    "cutting_board": {"actions": ["cut", "prepare"], "zones": ["preparation_zone"], "importance": 0.6},
    
    # 餐厅物体
    "dining_table": {"actions": ["eat", "dine", "gather"], "zones": ["dining_zone"], "importance": 0.95},
    "table": {"actions": ["place", "work", "gather"], "zones": ["dining_zone", "work_zone"], "importance": 0.7},
    "chair": {"actions": ["sit", "rest"], "zones": ["dining_zone", "seating_zone"], "importance": 0.6},
    "plate": {"actions": ["eat", "serve"], "zones": ["dining_zone"], "importance": 0.5},
    "cup": {"actions": ["drink"], "zones": ["dining_zone", "cooking_zone"], "importance": 0.4},
    "glass": {"actions": ["drink"], "zones": ["dining_zone"], "importance": 0.4},
    
    # 客厅物体
    "sofa": {"actions": ["sit", "relax", "rest"], "zones": ["relaxation_zone"], "importance": 0.9},
    "couch": {"actions": ["sit", "relax", "rest"], "zones": ["relaxation_zone"], "importance": 0.9},
    "tv": {"actions": ["watch", "entertain"], "zones": ["entertainment_zone"], "importance": 0.85},
    "television": {"actions": ["watch", "entertain"], "zones": ["entertainment_zone"], "importance": 0.85},
    "coffee_table": {"actions": ["place", "rest"], "zones": ["relaxation_zone"], "importance": 0.6},
    "bookshelf": {"actions": ["store", "display", "read"], "zones": ["reading_zone", "storage_zone"], "importance": 0.7},
    
    # 卧室物体
    "bed": {"actions": ["sleep", "rest", "relax"], "zones": ["sleeping_zone"], "importance": 0.95},
    "wardrobe": {"actions": ["store", "dress"], "zones": ["dressing_zone"], "importance": 0.8},
    "closet": {"actions": ["store", "dress"], "zones": ["dressing_zone"], "importance": 0.8},
    "nightstand": {"actions": ["place", "store"], "zones": ["sleeping_zone"], "importance": 0.5},
    "dresser": {"actions": ["store", "dress"], "zones": ["dressing_zone"], "importance": 0.7},
    "lamp": {"actions": ["illuminate"], "zones": ["reading_zone", "sleeping_zone"], "importance": 0.4},
    
    # 浴室物体
    "toilet": {"actions": ["use"], "zones": ["bathroom_zone"], "importance": 0.95},
    "bathtub": {"actions": ["bathe", "wash"], "zones": ["bathing_zone"], "importance": 0.9},
    "shower": {"actions": ["wash", "bathe"], "zones": ["bathing_zone"], "importance": 0.9},
    "bathroom_sink": {"actions": ["wash", "clean"], "zones": ["bathroom_zone"], "importance": 0.8},
    "mirror": {"actions": ["groom", "look"], "zones": ["grooming_zone"], "importance": 0.6},
    
    # 办公物体
    "desk": {"actions": ["work", "write", "study"], "zones": ["work_zone"], "importance": 0.9},
    "computer": {"actions": ["work", "browse", "compute"], "zones": ["work_zone"], "importance": 0.85},
    "monitor": {"actions": ["view", "work"], "zones": ["work_zone"], "importance": 0.7},
    "keyboard": {"actions": ["type", "input"], "zones": ["work_zone"], "importance": 0.5},
    "office_chair": {"actions": ["sit", "work"], "zones": ["work_zone"], "importance": 0.6},
    
    # 通用物体
    "door": {"actions": ["enter", "exit", "access"], "zones": ["entry_zone", "transition_zone"], "importance": 0.7},
    "window": {"actions": ["view", "ventilate"], "zones": [], "importance": 0.3},
    "plant": {"actions": ["decorate"], "zones": [], "importance": 0.2},
    "trash_can": {"actions": ["dispose", "discard"], "zones": ["cooking_zone", "work_zone"], "importance": 0.4},
    "garbage": {"actions": ["dispose", "discard"], "zones": ["cooking_zone", "work_zone"], "importance": 0.4},
}


class EnhancedAffordanceExtractor:
    """增强的Affordance提取器"""
    
    def __init__(self, llm_client=None, use_llm: bool = True):
        """
        Args:
            llm_client: LLM客户端（用于chat_completions调用）
            use_llm: 是否使用LLM（False则只用规则）
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        self.affordance_map = DEFAULT_AFFORDANCE_MAP.copy()
    
    def extract_affordances(self, objects: List[Dict], captions: Dict[int, str] = None) -> List[ObjectInfo]:
        """
        提取所有物体的增强Affordance
        
        Args:
            objects: 物体列表，每个物体是一个dict，包含tag等信息
            captions: 物体描述字典 {object_id: caption}
            
        Returns:
            List[ObjectInfo]: 增强的物体信息列表
        """
        captions = captions or {}
        
        if self.use_llm:
            return self._extract_with_llm(objects, captions)
        else:
            return self._extract_with_rules(objects, captions)
    
    def _extract_with_rules(self, objects: List[Dict], captions: Dict[int, str]) -> List[ObjectInfo]:
        """基于规则提取Affordance"""
        results = []
        
        for i, obj in enumerate(objects):
            # 获取类别名称 - class_name是一个列表，取最常见的非'item'类别
            class_names = obj.get("class_name", [])
            if isinstance(class_names, list) and class_names:
                # 过滤掉 'item' 并取最常见的类别
                valid_names = [n for n in class_names if n and n.lower() != 'item']
                if valid_names:
                    from collections import Counter
                    tag = Counter(valid_names).most_common(1)[0][0].lower()
                else:
                    tag = class_names[0].lower() if class_names[0] else f'object_{i}'
            else:
                tag = f'object_{i}'
            
            # 清理tag
            tag = self._clean_tag(tag)

            # 查找映射
            affordance_info = self._find_affordance_info(tag)
            
            # 构建EnhancedAffordance列表
            affordances = []
            for action in affordance_info.get("actions", []):
                affordances.append(EnhancedAffordance(
                    action=action,
                    context=self._infer_context(action),
                    duration=self._infer_duration(action),
                    co_objects=self._infer_co_objects(tag, action),
                    posture=self._infer_posture(action),
                    frequency="occasional"
                ))
            
            # 获取位置
            position = None
            if 'pcd_np' in obj and len(obj['pcd_np']) > 0:
                position = obj['pcd_np'].mean(axis=0).tolist()
            elif 'bbox_np' in obj and len(obj['bbox_np']) > 0:
                position = obj['bbox_np'].mean(axis=0).tolist()
            
            # 创建ObjectInfo
            obj_info = ObjectInfo(
                object_id=i,
                object_tag=tag,
                relation_type=ObjectRegionRelation.SUPPORTING,  # 默认，后续会更新
                affordances=affordances,
                typical_zones=affordance_info.get("zones", []),
                importance_score=affordance_info.get("importance", 0.5),
                position=position
            )
            
            results.append(obj_info)
        
        return results
    
    def _extract_with_llm(self, objects: List[Dict], captions: Dict[int, str]) -> List[ObjectInfo]:
        """使用LLM提取增强Affordance"""
        # 准备批量请求
        object_list = []
        for i, obj in enumerate(objects):
            # 获取类别名称
            class_names = obj.get("class_name", [])
            if isinstance(class_names, list) and class_names:
                valid_names = [n for n in class_names if n and n.lower() != 'item']
                if valid_names:
                    from collections import Counter
                    tag = Counter(valid_names).most_common(1)[0][0].lower()
                else:
                    tag = class_names[0].lower() if class_names[0] else f'object_{i}'
            else:
                tag = obj.get('curr_obj_name', obj.get('tag', f'object_{i}')).lower()
            tag = self._clean_tag(tag)
            
            caption = captions.get(i, "")
            object_list.append({
                "id": i,
                "tag": tag,
                "description": caption[:200] if caption else ""
            })
        
        # 分批处理（每批最多10个物体）
        batch_size = 10
        all_results = []
        
        for batch_start in range(0, len(object_list), batch_size):
            batch = object_list[batch_start:batch_start + batch_size]
            batch_results = self._llm_batch_extract(batch, objects)
            all_results.extend(batch_results)
        
        return all_results
    
    def _llm_batch_extract(self, batch: List[Dict], original_objects: List[Dict]) -> List[ObjectInfo]:
        """LLM批量提取"""
        prompt = self._build_extraction_prompt(batch)
        
        try:
            from conceptgraph.llava.unified_client import chat_completions
            
            model_name = os.getenv("LLM_MODEL")
            if not model_name:
                raise ValueError("环境变量 LLM_MODEL 必须显式设置，例如: export LLM_MODEL=gemini-3-flash-preview")
            base_url = os.getenv("LLM_BASE_URL")
            if not base_url:
                raise ValueError("环境变量 LLM_BASE_URL 必须显式设置，例如: export LLM_BASE_URL=http://10.21.231.7:8006")
            response = chat_completions(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.3,
                max_tokens=2000,
                base_url=base_url,
                timeout=60.0
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._parse_llm_response(content, batch, original_objects)
            
        except Exception as e:
            print(f"LLM提取失败，使用规则: {e}")
            # Fallback to rules
            results = []
            for item in batch:
                obj_idx = item["id"]
                obj = original_objects[obj_idx]
                results.extend(self._extract_with_rules([obj], {}))
            return results
    
    def _build_extraction_prompt(self, batch: List[Dict]) -> str:
        """构建LLM提取prompt"""
        objects_text = "\n".join([
            f"- ID {item['id']}: {item['tag']}" + (f" - {item['description']}" if item['description'] else "")
            for item in batch
        ])
        
        return f"""请分析以下物体的功能属性(affordance)。

## 物体列表
{objects_text}

## 任务
为每个物体提取以下属性：
1. actions: 该物体支持的动作列表
2. typical_zones: 该物体通常出现的功能区域
3. importance_score: 对区域定义的重要性 (0-1)
4. functional_role: 功能角色说明

## 输出格式 (JSON)
```json
{{
  "objects": [
    {{
      "id": 0,
      "tag": "stove",
      "affordances": [
        {{"action": "cook", "context": "meal_preparation", "duration": "medium", "co_objects": ["pot", "pan"], "posture": "standing"}}
      ],
      "typical_zones": ["cooking_zone"],
      "importance_score": 0.95,
      "functional_role": "烹饪核心设备，定义烹饪区域"
    }}
  ]
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _parse_llm_response(self, content: str, batch: List[Dict], original_objects: List[Dict]) -> List[ObjectInfo]:
        """解析LLM响应"""
        results = []
        
        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        try:
            data = json.loads(content)
            llm_objects = {item["id"]: item for item in data.get("objects", [])}
        except json.JSONDecodeError:
            llm_objects = {}
        
        for item in batch:
            obj_idx = item["id"]
            obj = original_objects[obj_idx]
            llm_item = llm_objects.get(obj_idx, {})
            
            # 获取位置
            position = None
            if 'pcd_np' in obj and len(obj['pcd_np']) > 0:
                position = obj['pcd_np'].mean(axis=0).tolist()
            
            # 解析affordances
            affordances = []
            for aff in llm_item.get("affordances", []):
                affordances.append(EnhancedAffordance(
                    action=aff.get("action", "use"),
                    context=aff.get("context", ""),
                    duration=aff.get("duration", "short"),
                    co_objects=aff.get("co_objects", []),
                    posture=aff.get("posture", "standing"),
                    frequency=aff.get("frequency", "occasional")
                ))
            
            # 如果LLM没有返回，使用规则fallback
            if not affordances:
                tag_clean = item["tag"]
                aff_info = self._find_affordance_info(tag_clean)
                for action in aff_info.get("actions", ["use"]):
                    affordances.append(EnhancedAffordance(
                        action=action,
                        context=self._infer_context(action),
                        duration=self._infer_duration(action)
                    ))
            
            obj_info = ObjectInfo(
                object_id=obj_idx,
                object_tag=item["tag"],
                relation_type=ObjectRegionRelation.SUPPORTING,
                affordances=affordances,
                typical_zones=llm_item.get("typical_zones", self._find_affordance_info(item["tag"]).get("zones", [])),
                importance_score=llm_item.get("importance_score", self._find_affordance_info(item["tag"]).get("importance", 0.5)),
                position=position,
                reasoning=llm_item.get("functional_role", "")
            )
            
            results.append(obj_info)
        
        return results
    
    def _clean_tag(self, tag: str) -> str:
        """清理物体标签"""
        # 移除数字后缀
        tag = re.sub(r'_\d+$', '', tag)
        # 替换空格
        tag = tag.replace(' ', '_').lower()
        return tag
    
    def _find_affordance_info(self, tag: str) -> Dict:
        """查找物体的affordance信息"""
        # 精确匹配
        if tag in self.affordance_map:
            return self.affordance_map[tag]
        
        # 部分匹配
        for key, value in self.affordance_map.items():
            if key in tag or tag in key:
                return value
        
        # 默认
        return {"actions": ["use"], "zones": [], "importance": 0.3}
    
    def _infer_context(self, action: str) -> str:
        """推断动作上下文"""
        context_map = {
            "cook": "meal_preparation",
            "fry": "meal_preparation",
            "bake": "meal_preparation",
            "heat": "food_warming",
            "wash": "cleaning",
            "clean": "cleaning",
            "store": "organization",
            "sit": "resting",
            "sleep": "resting",
            "watch": "entertainment",
            "work": "productivity",
            "read": "leisure",
            "eat": "dining",
            "drink": "dining",
        }
        return context_map.get(action, "general_use")
    
    def _infer_duration(self, action: str) -> str:
        """推断动作持续时间"""
        long_actions = {"sleep", "work", "watch", "study", "bake"}
        medium_actions = {"cook", "eat", "read", "bathe", "clean"}
        
        if action in long_actions:
            return "long"
        elif action in medium_actions:
            return "medium"
        return "short"
    
    def _infer_posture(self, action: str) -> str:
        """推断动作姿态"""
        sitting_actions = {"sit", "work", "eat", "read", "watch", "type"}
        lying_actions = {"sleep", "rest", "relax"}
        bending_actions = {"clean", "wash", "pick"}
        
        if action in sitting_actions:
            return "sitting"
        elif action in lying_actions:
            return "lying"
        elif action in bending_actions:
            return "bending"
        return "standing"
    
    def _infer_co_objects(self, tag: str, action: str) -> List[str]:
        """推断配合物体"""
        co_object_map = {
            ("stove", "cook"): ["pot", "pan", "spatula"],
            ("oven", "bake"): ["baking_tray", "oven_mitt"],
            ("sink", "wash"): ["soap", "sponge"],
            ("desk", "work"): ["computer", "keyboard", "mouse"],
            ("bed", "sleep"): ["pillow", "blanket"],
            ("dining_table", "eat"): ["plate", "fork", "knife", "cup"],
        }
        return co_object_map.get((tag, action), [])


def extract_affordances_from_scene(
    scene_path: str,
    use_llm: bool = True,
    llm_client = None
) -> List[ObjectInfo]:
    """
    从场景数据中提取所有物体的增强Affordance
    
    Args:
        scene_path: 场景数据路径
        use_llm: 是否使用LLM
        llm_client: LLM客户端
        
    Returns:
        List[ObjectInfo]: 物体信息列表
    """
    import gzip
    import pickle
    
    scene_path = Path(scene_path)
    
    # 加载物体数据
    pcd_files = list((scene_path / 'pcd_saves').glob('*_post.pkl.gz'))
    if not pcd_files:
        pcd_files = list((scene_path / 'pcd_saves').glob('*.pkl.gz'))
    
    if not pcd_files:
        raise FileNotFoundError(f"No object map found in {scene_path}")
    
    with gzip.open(pcd_files[0], 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    
    # 加载物体描述
    captions = {}
    captions_file = scene_path / 'captions.json'
    if captions_file.exists():
        with open(captions_file) as f:
            captions_data = json.load(f)
            for item in captions_data:
                captions[item.get('id', item.get('object_id'))] = item.get('caption', '')
    
    # 提取Affordance
    extractor = EnhancedAffordanceExtractor(llm_client=llm_client, use_llm=use_llm)
    return extractor.extract_affordances(objects, captions)


if __name__ == "__main__":
    # 测试
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    objects = extract_affordances_from_scene(args.scene_path, use_llm=args.use_llm)
    
    print(f"提取了 {len(objects)} 个物体的Affordance")
    for obj in objects[:5]:
        print(f"\n{obj.object_tag}:")
        print(f"  典型区域: {obj.typical_zones}")
        print(f"  重要性: {obj.importance_score:.2f}")
        print(f"  Affordances: {[a.action for a in obj.affordances]}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump([o.to_dict() for o in objects], f, indent=2, ensure_ascii=False)
        print(f"\n保存到: {args.output}")
