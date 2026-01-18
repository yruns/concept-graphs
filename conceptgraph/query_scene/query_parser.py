"""Query parsing and type classification."""

from __future__ import annotations
import json
import re
from typing import Optional
import requests
from .data_structures import QueryInfo, QueryType

SPATIAL_RELATIONS = {
    "旁边": "beside", "附近": "near", "左边": "left_of", "右边": "right_of",
    "前面": "in_front_of", "后面": "behind", "上面": "on", "下面": "under",
    "里面": "inside", "中间": "between", "边上": "beside", "对面": "across",
}

class QueryParser:
    """Parse natural language queries into structured QueryInfo."""
    
    def __init__(self, llm_url: str = "http://localhost:11434", 
                 llm_model: str = "llama3.1:8b", use_llm: bool = True):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.use_llm = use_llm
    
    def parse(self, query: str) -> QueryInfo:
        if self.use_llm:
            try:
                return self._parse_with_llm(query)
            except Exception as e:
                print(f"LLM parsing failed: {e}")
        return self._parse_with_regex(query)
    
    def _parse_with_llm(self, query: str) -> QueryInfo:
        prompt = f'''分析查询返回JSON: "{query}"
{{"target":"目标","anchor":"参照物或null","relation":"空间关系或null","query_type":"simple_object/spatial_relation/counting","use_bev":false}}'''
        
        response = self._call_llm(prompt)
        match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return QueryInfo(query, data.get("target", query), data.get("anchor"),
                           data.get("relation"), QueryType(data.get("query_type", "simple_object")),
                           data.get("use_bev", False))
        return self._parse_with_regex(query)
    
    def _call_llm(self, prompt: str) -> str:
        try:
            r = requests.post(f"{self.llm_url}/v1/chat/completions",
                json={"model": self.llm_model, "messages": [{"role": "user", "content": prompt}]}, timeout=30)
            if r.ok: return r.json()["choices"][0]["message"]["content"]
        except: pass
        try:
            r = requests.post(f"{self.llm_url}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False}, timeout=30)
            if r.ok: return r.json().get("response", "")
        except: pass
        raise ConnectionError("LLM unavailable")
    
    def _parse_with_regex(self, query: str) -> QueryInfo:
        target, anchor, relation = query, None, None
        query_type, use_bev = QueryType.SIMPLE_OBJECT, False
        
        for cn_rel, en_rel in SPATIAL_RELATIONS.items():
            if cn_rel in query:
                parts = query.split(cn_rel)
                if len(parts) >= 2:
                    anchor, target = parts[0].strip(), parts[-1].strip().lstrip("的")
                    relation, query_type = en_rel, QueryType.SPATIAL_RELATION
                    use_bev = en_rel in ["left_of", "right_of", "beside", "near"]
                break
        
        if re.search(r'几[个把张]|多少', query): query_type = QueryType.COUNTING
        if any(k in query for k in ['厨房', '客厅', '卧室']): query_type, use_bev = QueryType.FUNCTIONAL_REGION, True
        
        return QueryInfo(query, target, anchor, relation, query_type, use_bev)

def parse_query(query: str, llm_url: str = None, llm_model: str = None) -> QueryInfo:
    return QueryParser(llm_url or "http://localhost:11434", llm_model or "llama3.1:8b").parse(query)
