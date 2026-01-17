#!/usr/bin/env python3
"""
VLM功能组合分析器
=================

使用视觉语言模型分析关键帧中的功能物体组合：
1. 单帧分析：描述帧中的功能物体组合
2. 多帧对比：识别功能区域的边界
3. 片段分析：总结连续帧的功能区域
"""

import os
import json
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FunctionalGroup:
    """功能物体组合"""
    group_name: str
    primary_function: str
    objects: List[str]
    supported_activities: List[str]
    spatial_description: str = ""
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            "group_name": self.group_name,
            "primary_function": self.primary_function,
            "objects": self.objects,
            "supported_activities": self.supported_activities,
            "spatial_description": self.spatial_description,
            "confidence": self.confidence
        }


@dataclass
class FrameAnalysis:
    """单帧分析结果"""
    frame_idx: int
    functional_groups: List[FunctionalGroup] = field(default_factory=list)
    boundary_indicators: List[str] = field(default_factory=list)
    visual_context: str = ""
    raw_response: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "functional_groups": [g.to_dict() for g in self.functional_groups],
            "boundary_indicators": self.boundary_indicators,
            "visual_context": self.visual_context
        }


@dataclass
class SegmentAnalysis:
    """片段分析结果"""
    frame_start: int
    frame_end: int
    dominant_zones: List[str]
    zone_transitions: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "dominant_zones": self.dominant_zones,
            "zone_transitions": self.zone_transitions,
            "summary": self.summary
        }


class VLMFunctionalAnalyzer:
    """VLM功能组合分析器"""
    
    def __init__(self, vlm_client=None, base_url: str = None):
        """
        Args:
            vlm_client: VLM客户端（可选）
            base_url: VLM服务地址，默认从环境变量LLM_BASE_URL读取
        """
        self.vlm_client = vlm_client
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://10.21.231.7:8006")
    
    def analyze_single_frame(
        self,
        image_path: str,
        frame_idx: int,
        visible_objects: List[str] = None
    ) -> FrameAnalysis:
        """
        分析单帧图像
        
        Args:
            image_path: 图像路径
            frame_idx: 帧索引
            visible_objects: 该帧中可见的物体列表（可选）
            
        Returns:
            FrameAnalysis: 分析结果
        """
        prompt = self._build_single_frame_prompt(visible_objects)
        
        try:
            response = self._call_vlm(prompt, image_path)
            return self._parse_frame_response(response, frame_idx)
        except Exception as e:
            print(f"VLM分析帧 {frame_idx} 失败: {e}")
            return FrameAnalysis(frame_idx=frame_idx, visual_context=f"分析失败: {e}")
    
    def analyze_frame_pair(
        self,
        image_path1: str,
        image_path2: str,
        frame_idx1: int,
        frame_idx2: int
    ) -> Dict[str, Any]:
        """
        分析帧对，识别区域边界
        
        Args:
            image_path1, image_path2: 两帧图像路径
            frame_idx1, frame_idx2: 帧索引
            
        Returns:
            Dict: 包含边界分析结果
        """
        prompt = self._build_frame_pair_prompt()
        
        try:
            response = self._call_vlm_multi_image(prompt, [image_path1, image_path2])
            return self._parse_boundary_response(response, frame_idx1, frame_idx2)
        except Exception as e:
            print(f"VLM分析帧对 {frame_idx1}-{frame_idx2} 失败: {e}")
            return {
                "frame_indices": [frame_idx1, frame_idx2],
                "is_boundary": False,
                "error": str(e)
            }
    
    def analyze_segment(
        self,
        image_paths: List[str],
        frame_indices: List[int]
    ) -> SegmentAnalysis:
        """
        分析视频片段
        
        Args:
            image_paths: 片段中的关键帧路径
            frame_indices: 对应的帧索引
            
        Returns:
            SegmentAnalysis: 分析结果
        """
        # 使用首尾帧分析
        if len(image_paths) >= 2:
            prompt = self._build_segment_prompt(len(image_paths))
            try:
                response = self._call_vlm_multi_image(prompt, image_paths[:4])  # 最多4帧
                return self._parse_segment_response(response, frame_indices)
            except Exception as e:
                print(f"VLM分析片段失败: {e}")
        
        return SegmentAnalysis(
            frame_start=frame_indices[0] if frame_indices else 0,
            frame_end=frame_indices[-1] if frame_indices else 0,
            dominant_zones=[],
            summary="分析失败"
        )
    
    def _build_single_frame_prompt(self, visible_objects: List[str] = None) -> str:
        """构建单帧分析prompt"""
        objects_hint = ""
        if visible_objects:
            objects_hint = f"\n\n已检测到的物体: {', '.join(visible_objects[:15])}"
        
        return f"""请分析这张室内场景图像中的功能物体组合。{objects_hint}

## 任务
识别图像中的功能物体组合，并推断它们支持的活动。
注意：不要简单地说"这是厨房"或"这是客厅"，而是要描述具体的功能组合。

## 输出格式 (JSON)
```json
{{
  "functional_groups": [
    {{
      "group_name": "cooking_equipment",
      "primary_function": "烹饪和加热食物",
      "objects": ["stove", "range_hood", "pot"],
      "supported_activities": ["cook", "fry", "boil"],
      "spatial_description": "位于图像左侧",
      "confidence": 0.9
    }}
  ],
  "boundary_indicators": ["吧台可能分隔烹饪区和用餐区"],
  "visual_context": "开放式厨房场景，可见烹饪设备和储物空间"
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _build_frame_pair_prompt(self) -> str:
        """构建帧对比较prompt"""
        return """请比较这两张图像，判断它们是否展示了不同的功能区域。

## 任务
1. 分析两张图像各自的功能物体组合
2. 判断是否存在功能区域的边界/过渡
3. 描述边界的指示物（如门、墙、家具分隔等）

## 输出格式 (JSON)
```json
{
  "image1_zone": "cooking_zone",
  "image2_zone": "dining_zone",
  "is_boundary": true,
  "boundary_type": "soft",
  "boundary_indicator": "厨房岛台分隔两个区域",
  "confidence": 0.85,
  "reasoning": "第一张图显示炉灶和烹饪设备，第二张图显示餐桌和椅子"
}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _build_segment_prompt(self, n_frames: int) -> str:
        """构建片段分析prompt"""
        return f"""请分析这{n_frames}张连续图像，总结该视频片段覆盖的功能区域。

## 任务
1. 识别片段中出现的主要功能区域
2. 检测是否有区域过渡/变化
3. 总结片段的空间内容

## 输出格式 (JSON)
```json
{{
  "dominant_zones": ["cooking_zone", "storage_zone"],
  "zone_transitions": [
    {{"from": "cooking_zone", "to": "storage_zone", "frame_position": "middle"}}
  ],
  "summary": "片段主要展示厨房的烹饪区和储物区，从炉灶区移动到冰箱区"
}}
```

请直接输出JSON，不要添加其他说明。"""
    
    def _call_vlm(self, prompt: str, image_path: str) -> str:
        """调用VLM（单图）"""
        from conceptgraph.llava.unified_client import chat_completions
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_path", "image_path": image_path}
            ]
        }]
        
        model_name = os.getenv("LLM_MODEL")
        if not model_name:
            raise ValueError("环境变量 LLM_MODEL 必须显式设置，例如: export LLM_MODEL=gemini-3-flash-preview")
        response = chat_completions(
            messages=messages,
            model=model_name,
            temperature=0.3,
            max_tokens=1500,
            base_url=self.base_url,
            timeout=60.0
        )
        
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _call_vlm_multi_image(self, prompt: str, image_paths: List[str]) -> str:
        """调用VLM（多图）"""
        from conceptgraph.llava.unified_client import chat_completions
        
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_path", "image_path": path})
        
        messages = [{"role": "user", "content": content}]
        
        model_name = os.getenv("LLM_MODEL")
        if not model_name:
            raise ValueError("环境变量 LLM_MODEL 必须显式设置，例如: export LLM_MODEL=gemini-3-flash-preview")
        response = chat_completions(
            messages=messages,
            model=model_name,
            temperature=0.3,
            max_tokens=2000,
            base_url=self.base_url,
            timeout=90.0
        )
        
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _parse_frame_response(self, response: str, frame_idx: int) -> FrameAnalysis:
        """解析单帧分析响应"""
        result = FrameAnalysis(frame_idx=frame_idx, raw_response=response)
        
        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            data = json.loads(response)
            
            for group in data.get("functional_groups", []):
                result.functional_groups.append(FunctionalGroup(
                    group_name=group.get("group_name", "unknown"),
                    primary_function=group.get("primary_function", ""),
                    objects=group.get("objects", []),
                    supported_activities=group.get("supported_activities", []),
                    spatial_description=group.get("spatial_description", ""),
                    confidence=group.get("confidence", 0.8)
                ))
            
            result.boundary_indicators = data.get("boundary_indicators", [])
            result.visual_context = data.get("visual_context", "")
            
        except json.JSONDecodeError:
            result.visual_context = response[:500]  # 保存原始响应
        
        return result
    
    def _parse_boundary_response(
        self, 
        response: str, 
        frame_idx1: int, 
        frame_idx2: int
    ) -> Dict[str, Any]:
        """解析边界分析响应"""
        result = {
            "frame_indices": [frame_idx1, frame_idx2],
            "is_boundary": False,
            "raw_response": response
        }
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            data = json.loads(response)
            result.update({
                "image1_zone": data.get("image1_zone", ""),
                "image2_zone": data.get("image2_zone", ""),
                "is_boundary": data.get("is_boundary", False),
                "boundary_type": data.get("boundary_type", ""),
                "boundary_indicator": data.get("boundary_indicator", ""),
                "confidence": data.get("confidence", 0.5),
                "reasoning": data.get("reasoning", "")
            })
        except json.JSONDecodeError:
            pass
        
        return result
    
    def _parse_segment_response(
        self, 
        response: str, 
        frame_indices: List[int]
    ) -> SegmentAnalysis:
        """解析片段分析响应"""
        result = SegmentAnalysis(
            frame_start=frame_indices[0] if frame_indices else 0,
            frame_end=frame_indices[-1] if frame_indices else 0,
            dominant_zones=[],
            summary=""
        )
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            data = json.loads(response)
            result.dominant_zones = data.get("dominant_zones", [])
            result.zone_transitions = data.get("zone_transitions", [])
            result.summary = data.get("summary", "")
        except json.JSONDecodeError:
            result.summary = response[:500]
        
        return result


def analyze_keyframes_with_vlm(
    scene_path: str,
    keyframe_indices: List[int],
    stride: int = 5
) -> List[FrameAnalysis]:
    """
    使用VLM分析关键帧
    
    Args:
        scene_path: 场景路径
        keyframe_indices: 关键帧索引（采样后）
        stride: 采样步长
        
    Returns:
        List[FrameAnalysis]: 分析结果列表
    """
    scene_path = Path(scene_path)
    
    # 找到RGB图像目录
    rgb_dir = scene_path / 'rgb'
    if not rgb_dir.exists():
        rgb_dir = scene_path / 'results'
    
    if not rgb_dir.exists():
        print(f"找不到RGB图像目录: {scene_path}")
        return []
    
    # 获取图像文件列表
    image_files = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
    if not image_files:
        print(f"目录中没有图像文件: {rgb_dir}")
        return []
    
    analyzer = VLMFunctionalAnalyzer()
    results = []
    
    for kf_idx in keyframe_indices:
        # 转换为原始帧索引
        original_idx = kf_idx * stride
        
        # 找到对应的图像文件
        if original_idx < len(image_files):
            image_path = str(image_files[original_idx])
            print(f"分析关键帧 {kf_idx} (原始: {original_idx})")
            
            analysis = analyzer.analyze_single_frame(
                image_path=image_path,
                frame_idx=kf_idx
            )
            results.append(analysis)
        else:
            print(f"关键帧 {kf_idx} 超出范围")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--keyframes", type=str, default=None, help="关键帧索引，逗号分隔")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.keyframes:
        keyframes = [int(x.strip()) for x in args.keyframes.split(',')]
    else:
        # 默认选择几个均匀分布的帧
        keyframes = [0, 50, 100, 150, 200]
    
    results = analyze_keyframes_with_vlm(
        args.scene_path,
        keyframes,
        stride=args.stride
    )
    
    print(f"\n分析了 {len(results)} 个关键帧:")
    print("=" * 60)
    
    for r in results:
        print(f"\n帧 {r.frame_idx}:")
        print(f"  功能组合: {len(r.functional_groups)} 个")
        for g in r.functional_groups:
            print(f"    - {g.group_name}: {g.objects}")
        print(f"  边界指示: {r.boundary_indicators}")
        print(f"  视觉上下文: {r.visual_context[:100]}...")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\n保存到: {args.output}")
