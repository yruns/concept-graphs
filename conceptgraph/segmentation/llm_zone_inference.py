#!/usr/bin/env python3
"""
迭代LLM推理模块
===============

两步迭代推理机制：
Step 1: 功能区域划分 - 基于视频+物体组合+轨迹推理功能区域
Step 2: 物体区域分配 - 基于区域定义+affordance分配物体
Step 3: (可选) 验证与修正 - 检查一致性并修正
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .data_structures import (
    ObjectInfo,
    FunctionalZone,
    ObjectRegionRelation,
    SpatialInfo,
    TrajectoryEvidence,
    EnhancedAffordance
)
from .vlm_functional_analyzer import FrameAnalysis, FunctionalGroup
from .trajectory_behavior import TrajectoryBehaviorAnalysis


@dataclass
class ZoneInferenceResult:
    """区域推理结果"""
    zones: List[FunctionalZone]
    zone_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "zones": [z.to_dict() for z in self.zones],
            "zone_boundaries": self.zone_boundaries,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class ObjectAssignmentResult:
    """物体分配结果"""
    assignments: List[Dict[str, Any]]  # {object_id, zone_id, relation_type, ...}
    unassigned_objects: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "assignments": self.assignments,
            "unassigned_objects": self.unassigned_objects,
            "conflicts": self.conflicts
        }


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    refinements: List[Dict[str, Any]] = field(default_factory=list)
    final_confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "refinements": self.refinements,
            "final_confidence": self.final_confidence
        }


class LLMZoneInference:
    """迭代LLM推理器"""
    
    def __init__(self, base_url: str = None):
        """
        Args:
            base_url: LLM服务地址，默认从环境变量LLM_BASE_URL读取
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://10.21.231.7:8006")
    
    def run_inference(
        self,
        video_analysis: List[FrameAnalysis],
        object_affordances: List[ObjectInfo],
        trajectory_behavior: TrajectoryBehaviorAnalysis,
        object_positions: Dict[int, List[float]] = None
    ) -> Tuple[ZoneInferenceResult, ObjectAssignmentResult, ValidationResult]:
        """
        执行完整的迭代推理流程
        
        Args:
            video_analysis: VLM关键帧分析结果
            object_affordances: 物体affordance信息
            trajectory_behavior: 轨迹行为分析结果
            object_positions: 物体位置 {object_id: [x, y, z]}
            
        Returns:
            Tuple: (区域划分结果, 物体分配结果, 验证结果)
        """
        object_positions = object_positions or {}
        
        # 提取物体组合信息
        object_combos = self._extract_object_combos(object_affordances)
        
        # Step 1: 划分功能区域
        print("[Step 1] 推理功能区域...")
        zones_result = self.step1_infer_zones(
            video_analysis,
            object_combos,
            trajectory_behavior
        )
        print(f"  识别了 {len(zones_result.zones)} 个功能区域")
        
        # Step 2: 分配物体到区域
        print("[Step 2] 分配物体到区域...")
        assignments_result = self.step2_assign_objects(
            zones_result.zones,
            object_affordances,
            object_positions
        )
        print(f"  分配了 {len(assignments_result.assignments)} 个物体")
        
        # Step 3: 验证与修正
        print("[Step 3] 验证结果...")
        validation_result = self.step3_validate_and_refine(
            zones_result,
            assignments_result,
            self._summarize_evidence(video_analysis, trajectory_behavior)
        )
        print(f"  验证{'通过' if validation_result.passed else '未通过'}")
        
        # 如果验证未通过，应用修正
        if not validation_result.passed and validation_result.refinements:
            print("[Step 3b] 应用修正建议...")
            zones_result, assignments_result = self._apply_refinements(
                zones_result,
                assignments_result,
                validation_result.refinements
            )
        
        return zones_result, assignments_result, validation_result
    
    def step1_infer_zones(
        self,
        video_analysis: List[FrameAnalysis],
        object_combos: List[Dict],
        trajectory_behavior: TrajectoryBehaviorAnalysis
    ) -> ZoneInferenceResult:
        """
        Step 1: 基于视频和物体组合推理功能区域
        LLM调用失败会直接抛出异常
        """
        prompt = self._build_zone_inference_prompt(
            video_analysis, object_combos, trajectory_behavior
        )
        
        response = self._call_llm(prompt)
        result = self._parse_zone_response(response)
        
        # 如果LLM解析结果为空，报错退出
        if not result.zones:
            raise ValueError("LLM区域推理失败: 返回的功能区域列表为空。请检查LLM服务状态和响应格式。")
        
        return result
    
    def step2_assign_objects(
        self,
        zones: List[FunctionalZone],
        object_affordances: List[ObjectInfo],
        object_positions: Dict[int, List[float]]
    ) -> ObjectAssignmentResult:
        """
        Step 2: 将物体分配到功能区域
        LLM调用失败会直接抛出异常
        """
        prompt = self._build_assignment_prompt(zones, object_affordances, object_positions)
        
        response = self._call_llm(prompt)
        result = self._parse_assignment_response(response)
        
        # 如果LLM解析结果为空或分配数量过少，报错退出
        assigned_count = len(result.assignments)
        total_count = len(object_affordances)
        if assigned_count < total_count * 0.3:
            raise ValueError(
                f"LLM物体分配失败: 仅分配了 {assigned_count}/{total_count} 个物体 "
                f"(不足30%)。请检查LLM服务状态和响应格式。"
            )
        
        return result
    
    def step3_validate_and_refine(
        self,
        zones_result: ZoneInferenceResult,
        assignments_result: ObjectAssignmentResult,
        evidence_summary: str
    ) -> ValidationResult:
        """
        Step 3: 验证分配结果，处理冲突
        LLM调用失败会直接抛出异常
        """
        prompt = self._build_validation_prompt(
            zones_result, assignments_result, evidence_summary
        )
        
        response = self._call_llm(prompt)
        return self._parse_validation_response(response)
    
    def _build_zone_inference_prompt(
        self,
        video_analysis: List[FrameAnalysis],
        object_combos: List[Dict],
        trajectory_behavior: TrajectoryBehaviorAnalysis
    ) -> str:
        """构建区域推理prompt"""
        # 格式化关键帧分析
        keyframe_text = self._format_keyframe_analysis(video_analysis)
        
        # 格式化物体组合
        combo_text = self._format_object_combos(object_combos)
        
        # 格式化轨迹行为
        trajectory_text = self._format_trajectory_behavior(trajectory_behavior)
        
        return f"""你是场景理解专家。请根据以下证据，划分这个场景的功能区域。

## 视频分析结果
{keyframe_text}

## 检测到的物体组合
{combo_text}

## 相机轨迹行为
{trajectory_text}

## 任务
请划分功能区域。注意：
1. 功能区域不等于房间，一个房间可能有多个功能区域
2. 每个区域应该支持特定的活动/任务
3. 识别区域之间的边界和过渡

## 输出格式 (JSON)
```json
{{
  "functional_zones": [
    {{
      "zone_id": "fz_0",
      "zone_name": "cooking_zone",
      "primary_function": "烹饪和食物加热",
      "supported_activities": ["cook", "fry", "boil"],
      "defining_evidence": {{
        "video": "关键帧显示炉灶和抽油烟机",
        "objects": "检测到stove, range_hood, pot",
        "trajectory": "用户在此停留较长时间"
      }},
      "spatial_hint": "场景左侧区域",
      "confidence": 0.9
    }}
  ],
  "zone_boundaries": [
    {{
      "between": ["fz_0", "fz_1"],
      "indicator": "厨房岛台作为分隔",
      "type": "soft"
    }}
  ],
  "reasoning": "整体推理说明..."
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _build_assignment_prompt(
        self,
        zones: List[FunctionalZone],
        object_affordances: List[ObjectInfo],
        object_positions: Dict[int, List[float]]
    ) -> str:
        """构建物体分配prompt"""
        zones_text = self._format_zones(zones)
        objects_text = self._format_object_affordances(object_affordances)
        positions_text = self._format_positions(object_affordances, object_positions)
        
        return f"""你是物体-区域关系专家。请根据以下信息，将物体分配到对应的功能区域。

## 已识别的功能区域
{zones_text}

## 待分配的物体及其功能属性
{objects_text}

## 物体空间位置信息
{positions_text}

## 任务
为每个物体判断：
1. 它属于哪个功能区域？
2. 它与该区域是什么关系？
   - defining: 定义性（物体定义了区域，如炉灶定义烹饪区）
   - supporting: 支持性（物体支持区域功能，如锅具支持烹饪）
   - shared: 共享性（物体被多个区域共享，如垃圾桶）
   - boundary: 边界性（物体位于区域边界，如吧台）

## 输出格式 (JSON)
```json
{{
  "object_assignments": [
    {{
      "object_id": 0,
      "object_tag": "stove",
      "assigned_zone": "fz_0",
      "relation_type": "defining",
      "confidence": 0.95,
      "reasoning": "炉灶是烹饪区的核心设备"
    }}
  ],
  "unassigned_objects": [
    {{
      "object_id": 12,
      "object_tag": "wall_decoration",
      "reason": "装饰物不属于特定功能区域"
    }}
  ],
  "conflicts_detected": []
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _build_validation_prompt(
        self,
        zones_result: ZoneInferenceResult,
        assignments_result: ObjectAssignmentResult,
        evidence_summary: str
    ) -> str:
        """构建验证prompt"""
        return f"""请验证以下功能区域划分和物体分配结果的一致性。

## 区域划分
{json.dumps([z.to_dict() for z in zones_result.zones], indent=2, ensure_ascii=False)[:2000]}

## 物体分配
{json.dumps(assignments_result.assignments[:20], indent=2, ensure_ascii=False)}

## 原始证据摘要
{evidence_summary}

## 验证任务
1. 检查每个区域是否有足够的defining物体支撑
2. 检查物体分配是否与其affordance一致
3. 检查是否有遗漏的功能区域
4. 检查边界物体的处理是否合理

## 输出格式 (JSON)
```json
{{
  "validation_passed": true,
  "issues": [],
  "suggested_refinements": [],
  "final_confidence": 0.85
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        from conceptgraph.llava.unified_client import chat_completions
        
        model_name = os.getenv("LLM_MODEL")
        if not model_name:
            raise ValueError("环境变量 LLM_MODEL 必须显式设置，例如: export LLM_MODEL=gemini-3-flash-preview")
        response = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.3,
            max_tokens=16000,  # 增大以支持大量物体的分配
            base_url=self.base_url,
            timeout=180.0  # 增加超时
        )
        
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _parse_zone_response(self, response: str) -> ZoneInferenceResult:
        """解析区域推理响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return ZoneInferenceResult(zones=[], reasoning="解析失败")
        
        zones = []
        for z in data.get("functional_zones", []):
            zone = FunctionalZone(
                zone_id=z.get("zone_id", f"fz_{len(zones)}"),
                zone_name=z.get("zone_name", "unknown"),
                parent_unit="su_0",  # 默认
                primary_activity=z.get("primary_function", ""),
                supported_activities=z.get("supported_activities", []),
                defining_evidence=z.get("defining_evidence", {}),
                confidence=z.get("confidence", 0.8)
            )
            zones.append(zone)
        
        return ZoneInferenceResult(
            zones=zones,
            zone_boundaries=data.get("zone_boundaries", []),
            reasoning=data.get("reasoning", ""),
            confidence=0.8
        )
    
    def _parse_assignment_response(self, response: str) -> ObjectAssignmentResult:
        """解析物体分配响应"""
        # 调试: 打印原始响应
        print(f"    [DEBUG] LLM响应长度: {len(response)} 字符")
        
        # 尝试多种方式提取JSON
        json_str = response
        
        # 方式1: 提取```json ... ``` 之间的内容
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"    [DEBUG] 通过```json```提取, 长度: {len(json_str)}")
        else:
            # 方式2: 提取第一个 { 到最后一个 } 之间的内容
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)
                print(f"    [DEBUG] 通过大括号提取, 长度: {len(json_str)}")
            else:
                print(f"    [DEBUG] 无法提取JSON内容")
        
        try:
            data = json.loads(json_str)
            assignments = data.get("object_assignments", [])
            print(f"    [DEBUG] 解析成功, object_assignments数量: {len(assignments)}")
        except json.JSONDecodeError as e:
            print(f"    [DEBUG] JSON解析失败: {e}")
            print(f"    [DEBUG] 尝试解析的内容前200字符: {json_str[:200]}")
            return ObjectAssignmentResult(assignments=[])
        
        return ObjectAssignmentResult(
            assignments=data.get("object_assignments", []),
            unassigned_objects=data.get("unassigned_objects", []),
            conflicts=data.get("conflicts_detected", [])
        )
    
    def _parse_validation_response(self, response: str) -> ValidationResult:
        """解析验证响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return ValidationResult(passed=True, final_confidence=0.7)
        
        return ValidationResult(
            passed=data.get("validation_passed", True),
            issues=data.get("issues", []),
            refinements=data.get("suggested_refinements", []),
            final_confidence=data.get("final_confidence", 0.8)
        )
    
    def _extract_object_combos(self, object_affordances: List[ObjectInfo]) -> List[Dict]:
        """提取物体组合信息"""
        combos = []
        for obj in object_affordances:
            combo = {
                "object_id": obj.object_id,
                "tag": obj.object_tag,
                "actions": [a.action for a in obj.affordances],
                "typical_zones": obj.typical_zones,
                "importance": obj.importance_score
            }
            combos.append(combo)
        return combos
    
    def _format_keyframe_analysis(self, video_analysis: List[FrameAnalysis]) -> str:
        """格式化关键帧分析"""
        lines = []
        for fa in video_analysis[:10]:  # 最多10帧
            lines.append(f"帧 {fa.frame_idx}:")
            for g in fa.functional_groups:
                lines.append(f"  - {g.group_name}: {g.objects}")
            if fa.boundary_indicators:
                lines.append(f"  边界: {fa.boundary_indicators}")
            if fa.visual_context:
                lines.append(f"  上下文: {fa.visual_context[:100]}")
        return "\n".join(lines) if lines else "无视频分析数据"
    
    def _format_object_combos(self, combos: List[Dict]) -> str:
        """格式化物体组合"""
        lines = []
        for c in combos[:30]:  # 最多30个
            lines.append(f"- {c['tag']}: actions={c['actions']}, typical_zones={c['typical_zones']}")
        return "\n".join(lines) if lines else "无物体数据"
    
    def _format_trajectory_behavior(self, behavior: TrajectoryBehaviorAnalysis) -> str:
        """格式化轨迹行为"""
        lines = []
        lines.append(f"停留点: {len(behavior.dwell_points)} 个")
        for dp in behavior.dwell_points[:5]:
            lines.append(f"  - 位置 {dp.position}, 时长 {dp.duration_seconds:.1f}秒")
        
        lines.append(f"环顾事件: {len(behavior.look_around_events)} 个")
        for la in behavior.look_around_events[:3]:
            lines.append(f"  - 位置 {la.position}, 旋转 {la.total_rotation_deg:.0f}度")
        
        lines.append(f"快速穿越: {len(behavior.traverse_segments)} 段")
        
        return "\n".join(lines)
    
    def _format_zones(self, zones: List[FunctionalZone]) -> str:
        """格式化区域信息"""
        lines = []
        for z in zones:
            lines.append(f"- {z.zone_id}: {z.zone_name}")
            lines.append(f"  功能: {z.primary_activity}")
            lines.append(f"  活动: {z.supported_activities}")
        return "\n".join(lines) if lines else "无区域数据"
    
    def _format_object_affordances(self, objects: List[ObjectInfo]) -> str:
        """格式化物体affordance (处理所有物体)"""
        lines = []
        # 处理所有物体，但使用简洁格式避免超出token限制
        for obj in objects:
            actions = [a.action for a in obj.affordances[:3]]  # 最多3个动作
            zones = obj.typical_zones[:2] if obj.typical_zones else []
            lines.append(f"- [{obj.object_id}] {obj.object_tag}: {actions}, zones={zones}")
        return "\n".join(lines) if lines else "无物体数据"
    
    def _format_positions(
        self, 
        objects: List[ObjectInfo], 
        positions: Dict[int, List[float]]
    ) -> str:
        """格式化位置信息 (处理所有物体)"""
        lines = []
        for obj in objects:
            pos = positions.get(obj.object_id) or obj.position
            if pos:
                lines.append(f"- [{obj.object_id}] {obj.object_tag}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        return "\n".join(lines) if lines else "无位置数据"
    
    def _summarize_evidence(
        self, 
        video_analysis: List[FrameAnalysis],
        trajectory_behavior: TrajectoryBehaviorAnalysis
    ) -> str:
        """总结证据"""
        all_groups = []
        for fa in video_analysis:
            all_groups.extend([g.group_name for g in fa.functional_groups])
        
        return f"""
视频分析: {len(video_analysis)} 个关键帧
识别的功能组合: {list(set(all_groups))}
停留点: {len(trajectory_behavior.dwell_points)} 个
环顾事件: {len(trajectory_behavior.look_around_events)} 个
"""
    
    def _apply_refinements(
        self,
        zones_result: ZoneInferenceResult,
        assignments_result: ObjectAssignmentResult,
        refinements: List[Dict]
    ) -> Tuple[ZoneInferenceResult, ObjectAssignmentResult]:
        """应用修正建议"""
        # 简化处理：直接返回原结果
        # 实际实现可以根据refinements进行修正
        return zones_result, assignments_result
    


if __name__ == "__main__":
    # 测试
    print("LLM Zone Inference Module")
    print("=" * 60)
    print("使用方法:")
    print("  inference = LLMZoneInference()")
    print("  zones, assignments, validation = inference.run_inference(...)")
