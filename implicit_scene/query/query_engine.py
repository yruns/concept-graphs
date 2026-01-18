"""
QueryEngine: 问题导向查询引擎
============================

双分支查询机制:
- 分支A: 视频回放 (视觉证据)
- 分支B: 特征检索 (语义匹配)
- LLM融合推理
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..store.vector_store import SceneVectorStore, ObjectMeta, SearchResult
from .query_parser import QueryParser, ParsedQuery


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    parsed: ParsedQuery
    target_objects: List[ObjectMeta]
    context_objects: List[ObjectMeta] = field(default_factory=list)
    visual_evidence: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "parsed": self.parsed.to_dict(),
            "target_objects": [obj.to_dict() for obj in self.target_objects],
            "context_objects": [obj.to_dict() for obj in self.context_objects],
            "visual_evidence": self.visual_evidence,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }
    
    def __repr__(self):
        targets = ", ".join(f"{o.tag}(id={o.id})" for o in self.target_objects[:3])
        return f"QueryResult(targets=[{targets}], confidence={self.confidence:.2f})"


class QueryEngine:
    """
    问题导向的查询引擎
    
    核心流程:
    1. LLM解析query
    2. 分支B: 语义检索候选物体
    3. 分支A: 视频回放 (如果需要视觉证据)
    4. 空间约束过滤
    5. LLM融合推理 (复杂query)
    """
    
    def __init__(self, store: SceneVectorStore, 
                 llm_url: str = None, llm_model: str = None):
        """
        Args:
            store: 场景向量存储
            llm_url: LLM服务URL
            llm_model: LLM模型名称
        """
        self.store = store
        self.llm_url = llm_url or "http://10.21.231.7:8006"
        self.llm_model = llm_model or "gemini-2.0-flash"
        self.parser = QueryParser(llm_url=llm_url, llm_model=llm_model)
    
    def query(self, query_text: str, top_k: int = 5) -> QueryResult:
        """
        执行查询
        
        Args:
            query_text: 自然语言查询
            top_k: 返回候选数量
            
        Returns:
            QueryResult
        """
        print(f"\n[Query] {query_text}")
        
        # 1. 解析query
        parsed = self.parser.parse(query_text)
        target_display = f"{parsed.target}" if parsed.target == parsed.target_cn else f"{parsed.target} ({parsed.target_cn})"
        refs_display = [f"{r.object}" for r in parsed.references]
        print(f"  Parsed: target='{target_display}', refs={refs_display}")
        print(f"  Constraint: {parsed.spatial_constraint}")
        
        # 2. 分支B: 语义检索目标物体
        target_results = self.store.semantic_search(parsed.target, top_k=top_k * 2)
        print(f"  Found {len(target_results)} candidates for '{parsed.target}'")
        
        # 3. 如果有参照物，检索参照物并应用空间约束
        if parsed.references:
            target_results = self._apply_spatial_constraints(
                target_results, parsed, top_k
            )
            print(f"  After spatial filter: {len(target_results)} candidates")
        
        # 4. 分支A: 视频回放 (如果需要)
        visual_evidence = []
        if parsed.needs_visual_evidence and self.store.frames:
            visual_evidence = self._search_visual_evidence(query_text)
            print(f"  Found {len(visual_evidence)} visual evidence frames")
        
        # 5. 构建结果
        target_objects = [r.meta for r in target_results[:top_k]]
        
        # 获取上下文物体
        context_objects = []
        if target_objects:
            for ref in parsed.references:
                ref_results = self.store.semantic_search(ref.object, top_k=3)
                context_objects.extend([r.meta for r in ref_results])
        
        # 计算置信度
        confidence = target_results[0].score if target_results else 0.0
        
        result = QueryResult(
            query=query_text,
            parsed=parsed,
            target_objects=target_objects,
            context_objects=context_objects,
            visual_evidence=visual_evidence,
            confidence=confidence,
        )
        
        print(f"  Result: {result}")
        
        return result
    
    def _apply_spatial_constraints(self, candidates: List[SearchResult], 
                                   parsed: ParsedQuery, top_k: int) -> List[SearchResult]:
        """
        应用空间约束
        
        根据parsed.spatial_constraint和references过滤候选
        """
        if not parsed.references:
            return candidates
        
        # 检索参照物
        reference_objects = []
        for ref in parsed.references:
            ref_results = self.store.semantic_search(ref.object, top_k=3)
            if ref_results:
                reference_objects.append((ref, ref_results[0].meta))
        
        if not reference_objects:
            return candidates
        
        # 解析空间约束
        constraint = parsed.spatial_constraint.lower()
        
        filtered = []
        for candidate in candidates:
            cand_pos = candidate.meta.position
            
            # 检查是否满足约束
            satisfies = True
            for ref, ref_obj in reference_objects:
                ref_pos = ref_obj.position
                
                # 根据约束类型判断
                if any(word in constraint for word in ["距离", "近", "旁边", "附近"]):
                    # 距离约束
                    dist = np.linalg.norm(cand_pos[:2] - ref_pos[:2])  # 水平距离
                    if dist > 2.0:  # 默认阈值2米
                        satisfies = False
                
                elif any(word in constraint for word in ["上方", "上面", "上"]):
                    # 上方约束
                    if cand_pos[2] <= ref_pos[2]:
                        satisfies = False
                    # 水平距离也要近
                    dist = np.linalg.norm(cand_pos[:2] - ref_pos[:2])
                    if dist > 1.0:
                        satisfies = False
                
                elif any(word in constraint for word in ["下方", "下面", "下"]):
                    # 下方约束
                    if cand_pos[2] >= ref_pos[2]:
                        satisfies = False
                
                elif any(word in constraint for word in ["之间", "中间"]):
                    # 之间约束 (需要两个参照物)
                    if len(reference_objects) >= 2:
                        ref1_pos = reference_objects[0][1].position
                        ref2_pos = reference_objects[1][1].position
                        mid_pos = (ref1_pos + ref2_pos) / 2
                        dist_to_mid = np.linalg.norm(cand_pos[:2] - mid_pos[:2])
                        if dist_to_mid > 1.5:
                            satisfies = False
            
            if satisfies:
                filtered.append(candidate)
        
        return filtered
    
    def _search_visual_evidence(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索视觉证据帧"""
        frame_results = self.store.semantic_search(query, top_k=top_k, search_type="frames")
        
        evidence = []
        for r in frame_results:
            evidence.append({
                "frame_idx": r.meta.idx,
                "image_path": r.meta.image_path,
                "score": r.score,
            })
        
        return evidence
    
    def ground(self, description: str) -> List[ObjectMeta]:
        """
        Visual Grounding: 根据描述定位物体
        
        Args:
            description: 自然语言描述
            
        Returns:
            匹配的物体列表
        """
        result = self.query(description, top_k=5)
        return result.target_objects
    
    def answer_question(self, question: str) -> str:
        """
        ScanQA: 回答关于场景的问题
        
        Args:
            question: 问题
            
        Returns:
            答案文本
        """
        # 检索相关物体
        results = self.store.semantic_search(question, top_k=15)
        
        # 构建上下文
        context_lines = []
        for r in results:
            obj = r.meta
            context_lines.append(
                f"- {obj.tag} (id={obj.id}, position={obj.position.tolist()})"
            )
        context = "\n".join(context_lines)
        
        # 构建prompt
        prompt = f"""基于以下场景中的物体信息回答问题：

场景物体:
{context}

问题: {question}

请简洁准确地回答："""
        
        # 调用LLM
        try:
            import requests
            response = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM error: {e}")
        
        return f"在场景中找到了 {len(results)} 个相关物体"
    
    def __repr__(self):
        return f"QueryEngine(store={self.store})"
