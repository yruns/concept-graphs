"""
SceneInterface: 下游任务统一接口
================================

支持的任务:
- ScanQA: 场景问答
- Visual Grounding: 物体定位
- Navigation Planning: 导航规划
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ..store.vector_store import SceneVectorStore, ObjectMeta
from ..query.query_engine import QueryEngine, QueryResult


@dataclass
class Waypoint:
    """导航路点"""
    position: np.ndarray
    description: str
    object_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "position": self.position.tolist(),
            "description": self.description,
            "object_id": self.object_id,
        }


class SceneInterface:
    """
    面向下游任务的统一接口
    
    所有任务都遵循: 检索 -> 推理 -> 输出
    """
    
    def __init__(self, store: SceneVectorStore, llm_url: str = None, llm_model: str = None):
        """
        Args:
            store: 场景向量存储
            llm_url: LLM服务URL
            llm_model: LLM模型名称
        """
        self.store = store
        self.engine = QueryEngine(store, llm_url=llm_url, llm_model=llm_model)
        self.llm_url = llm_url or "http://10.21.231.7:8006"
        self.llm_model = llm_model or "gemini-2.0-flash"
    
    # ============== ScanQA ==============
    
    def answer_question(self, question: str) -> Dict:
        """
        ScanQA: 回答关于场景的问题
        
        Examples:
            Q: "房间里有几把椅子？"
            Q: "沙发是什么颜色的？"
            Q: "桌子上有什么？"
        
        Args:
            question: 问题
            
        Returns:
            {"answer": str, "evidence": List[ObjectMeta]}
        """
        answer = self.engine.answer_question(question)
        
        # 获取相关物体作为证据
        results = self.store.semantic_search(question, top_k=10)
        evidence = [r.meta for r in results]
        
        return {
            "question": question,
            "answer": answer,
            "evidence": [e.to_dict() for e in evidence],
        }
    
    # ============== Visual Grounding ==============
    
    def ground(self, description: str, return_all: bool = False) -> Dict:
        """
        Visual Grounding: 根据描述定位物体
        
        Examples:
            "沙发旁边的台灯"
            "桌子上的红色杯子"
            "最大的椅子"
        
        Args:
            description: 自然语言描述
            return_all: 是否返回所有匹配
            
        Returns:
            {"objects": List[ObjectMeta], "confidence": float}
        """
        result = self.engine.query(description, top_k=5 if return_all else 1)
        
        return {
            "description": description,
            "objects": [o.to_dict() for o in result.target_objects],
            "context": [o.to_dict() for o in result.context_objects],
            "confidence": result.confidence,
            "parsed": result.parsed.to_dict(),
        }
    
    def ground_object(self, description: str) -> Optional[ObjectMeta]:
        """简化版grounding，返回单个物体"""
        result = self.engine.query(description, top_k=1)
        if result.target_objects:
            return result.target_objects[0]
        return None
    
    # ============== Navigation Planning ==============
    
    def plan_navigation(self, instruction: str) -> Dict:
        """
        Navigation Planning: 根据指令规划导航路径
        
        Examples:
            "去厨房拿杯子"
            "先到沙发，然后去窗户旁边"
            "找到所有的椅子"
        
        Args:
            instruction: 自然语言导航指令
            
        Returns:
            {"waypoints": List[Waypoint], "description": str}
        """
        # 提取指令中的地标
        landmarks = self._extract_landmarks(instruction)
        
        # 定位每个地标
        waypoints = []
        for landmark in landmarks:
            obj = self.ground_object(landmark)
            if obj:
                waypoints.append(Waypoint(
                    position=obj.position,
                    description=f"到达 {obj.tag}",
                    object_id=obj.id,
                ))
        
        return {
            "instruction": instruction,
            "landmarks": landmarks,
            "waypoints": [w.to_dict() for w in waypoints],
            "n_waypoints": len(waypoints),
        }
    
    def _extract_landmarks(self, instruction: str) -> List[str]:
        """从指令中提取地标"""
        import requests
        
        prompt = f'''从导航指令中提取需要经过的地标或物体名称。
只输出物体名称列表，用逗号分隔。

指令: "{instruction}"

地标列表:'''
        
        try:
            response = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # 解析逗号分隔的列表
                landmarks = [l.strip() for l in content.split(',') if l.strip()]
                return landmarks
        except Exception as e:
            print(f"LLM error: {e}")
        
        # fallback: 简单的关键词提取
        keywords = []
        common_objects = ["沙发", "椅子", "桌子", "床", "柜子", "门", "窗户", "台灯", "杯子"]
        for obj in common_objects:
            if obj in instruction:
                keywords.append(obj)
        
        return keywords if keywords else [instruction]
    
    # ============== 通用查询 ==============
    
    def query(self, query_text: str, top_k: int = 5) -> QueryResult:
        """通用查询接口"""
        return self.engine.query(query_text, top_k=top_k)
    
    def semantic_search(self, text: str, top_k: int = 10) -> List[Dict]:
        """语义搜索"""
        results = self.store.semantic_search(text, top_k=top_k)
        return [{"id": r.id, "score": r.score, "object": r.meta.to_dict()} for r in results]
    
    # ============== 场景信息 ==============
    
    def get_scene_summary(self) -> Dict:
        """获取场景摘要"""
        return self.store.summary()
    
    def list_objects(self, tag_filter: str = None) -> List[Dict]:
        """列出场景中的物体"""
        objects = self.store.objects
        if tag_filter:
            objects = [o for o in objects if tag_filter.lower() in o.tag.lower()]
        return [o.to_dict() for o in objects]
    
    def __repr__(self):
        return f"SceneInterface(store={self.store})"
