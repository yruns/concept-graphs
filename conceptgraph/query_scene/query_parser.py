"""Query parsing and type classification.

All outputs (target, anchor) must be in English for CLIP compatibility.
"""

from __future__ import annotations
import json
import re
from typing import Optional
from loguru import logger
import requests
from .data_structures import QueryInfo, QueryType


# English spatial relation patterns
SPATIAL_PATTERNS = [
    (r"(.+?)\s+(?:near|beside|next to|by)\s+(?:the\s+)?(.+)", "beside"),
    (r"(.+?)\s+(?:on|above|on top of)\s+(?:the\s+)?(.+)", "on"),
    (r"(.+?)\s+(?:under|below|beneath)\s+(?:the\s+)?(.+)", "under"),
    (r"(.+?)\s+(?:left of|to the left of)\s+(?:the\s+)?(.+)", "left_of"),
    (r"(.+?)\s+(?:right of|to the right of)\s+(?:the\s+)?(.+)", "right_of"),
    (r"(.+?)\s+(?:in front of|before)\s+(?:the\s+)?(.+)", "in_front_of"),
    (r"(.+?)\s+(?:behind|back of)\s+(?:the\s+)?(.+)", "behind"),
    (r"(.+?)\s+(?:between)\s+(?:the\s+)?(.+)", "between"),
]


class QueryParser:
    """Parse natural language queries into structured QueryInfo.
    
    All target and anchor outputs are in English for CLIP compatibility.
    """
    
    def __init__(self, llm_url: str = "http://localhost:11434", 
                 llm_model: str = "llama3.1:8b", use_llm: bool = True):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.use_llm = use_llm
    
    def parse(self, query: str) -> QueryInfo:
        """Parse query, always output English target/anchor."""
        if self.use_llm:
            try:
                return self._parse_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, using regex fallback")
        return self._parse_with_regex(query)
    
    def _parse_with_llm(self, query: str) -> QueryInfo:
        """Parse using LLM with explicit English output requirement."""
        prompt = f'''Parse this query and return JSON. ALL outputs must be in English.

Query: "{query}"

Return exactly this JSON format:
{{"target": "<object name in English>", "anchor": "<reference object in English or null>", "relation": "<spatial relation in English or null>", "query_type": "<simple_object|spatial_relation|counting|attribute>", "use_bev": <true for spatial/counting, false otherwise>}}

Examples:
- "lamp" -> {{"target": "lamp", "anchor": null, "relation": null, "query_type": "simple_object", "use_bev": false}}
- "lamp near the sofa" -> {{"target": "lamp", "anchor": "sofa", "relation": "beside", "query_type": "spatial_relation", "use_bev": true}}
- "chair next to table" -> {{"target": "chair", "anchor": "table", "relation": "beside", "query_type": "spatial_relation", "use_bev": true}}
- "red pillow" -> {{"target": "pillow", "anchor": null, "relation": null, "query_type": "attribute", "use_bev": false}}
- "how many chairs" -> {{"target": "chair", "anchor": null, "relation": null, "query_type": "counting", "use_bev": true}}

Output only the JSON, no explanation.'''
        
        response = self._call_llm(prompt)
        logger.debug(f"LLM response: {response[:200]}")
        
        match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            query_type_str = data.get("query_type", "simple_object")
            try:
                query_type = QueryType(query_type_str)
            except ValueError:
                query_type = QueryType.SIMPLE_OBJECT
            
            return QueryInfo(
                original_query=query,
                target=data.get("target", query),
                anchor=data.get("anchor"),
                relation=data.get("relation"),
                query_type=query_type,
                use_bev=data.get("use_bev", False),
            )
        
        return self._parse_with_regex(query)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        try:
            r = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={"model": self.llm_model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        
        try:
            r = requests.post(
                f"{self.llm_url}/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=30
            )
            if r.ok:
                return r.json().get("response", "")
        except Exception:
            pass
        
        raise ConnectionError("LLM unavailable")
    
    def _parse_with_regex(self, query: str) -> QueryInfo:
        """Fallback regex parsing for English queries."""
        target = query.strip()
        anchor = None
        relation = None
        query_type = QueryType.SIMPLE_OBJECT
        use_bev = False
        
        # Remove articles
        query_clean = re.sub(r'\b(the|a|an)\b', '', query.lower()).strip()
        
        # Check spatial patterns
        for pattern, rel in SPATIAL_PATTERNS:
            match = re.match(pattern, query_clean, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                anchor = match.group(2).strip()
                relation = rel
                query_type = QueryType.SPATIAL_RELATION
                use_bev = rel in ["beside", "left_of", "right_of", "between"]
                break
        
        # Check counting
        if re.search(r'how many|count|number of', query_clean, re.IGNORECASE):
            query_type = QueryType.COUNTING
            use_bev = True
            # Extract object after "how many"
            m = re.search(r'how many\s+(\w+)', query_clean, re.IGNORECASE)
            if m:
                target = m.group(1)
        
        # Check attribute (adjective + noun)
        if re.match(r'^(red|blue|green|white|black|big|small|tall|short)\s+\w+', query_clean):
            query_type = QueryType.ATTRIBUTE
        
        return QueryInfo(
            original_query=query,
            target=target,
            anchor=anchor,
            relation=relation,
            query_type=query_type,
            use_bev=use_bev,
        )

def parse_query(query: str, llm_url: str = None, llm_model: str = None) -> QueryInfo:
    return QueryParser(llm_url or "http://localhost:11434", llm_model or "llama3.1:8b").parse(query)
