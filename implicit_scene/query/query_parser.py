"""
QueryParser: LLM自由解析Query
=============================

将自然语言query解析为结构化形式，支持:
- 目标物体识别
- 参照物识别
- 空间约束提取 (自由形式，非预定义)
"""

import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Reference:
    """参照物"""
    object: str            # 英文名称 (用于CLIP检索)
    object_cn: str = ""    # 中文名称 (用于显示)
    spatial_hint: str = "" # 空间提示


@dataclass 
class ParsedQuery:
    """解析后的查询"""
    raw_query: str
    target: str                           # 目标物体 (英文, 用于CLIP检索)
    target_cn: str = ""                   # 目标物体 (中文, 用于显示)
    references: List[Reference] = field(default_factory=list)  # 参照物
    spatial_constraint: str = ""          # 自然语言空间约束
    attributes: Dict[str, str] = field(default_factory=dict)   # 属性 (颜色等)
    needs_visual_evidence: bool = False   # 是否需要视觉证据
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "raw_query": self.raw_query,
            "target": self.target,
            "target_cn": self.target_cn,
            "references": [{"object": r.object, "object_cn": getattr(r, 'object_cn', r.object), "spatial_hint": r.spatial_hint} for r in self.references],
            "spatial_constraint": self.spatial_constraint,
            "attributes": self.attributes,
            "needs_visual_evidence": self.needs_visual_evidence,
        }


class QueryParser:
    """
    LLM驱动的Query解析器
    
    将自然语言query解析为结构化形式，空间约束保持自然语言描述
    """
    
    PARSE_PROMPT = '''You are a scene understanding assistant. Parse the user's query and extract:

1. target: The target object (in English, for CLIP retrieval)
2. target_cn: The target object name in original language
3. references: List of reference objects with English names
4. spatial_constraint: Spatial constraint description (in English)
5. attributes: Object attributes like color, size
6. needs_visual_evidence: Whether visual confirmation is needed

Output in JSON format. Examples:

Query: "沙发旁边的台灯" (the lamp next to the sofa)
Output:
```json
{{
  "target": "lamp",
  "target_cn": "台灯",
  "references": [{{"object": "sofa", "object_cn": "沙发", "spatial_hint": "next to"}}],
  "spatial_constraint": "horizontal distance to sofa less than 1.5 meters",
  "attributes": {{}},
  "needs_visual_evidence": false
}}
```

Query: "红色的杯子" (the red cup)
Output:
```json
{{
  "target": "cup",
  "target_cn": "杯子",
  "references": [],
  "spatial_constraint": "",
  "attributes": {{"color": "red"}},
  "needs_visual_evidence": true
}}
```

Query: "桌子上的杯子" (the cup on the table)
Output:
```json
{{
  "target": "cup",
  "target_cn": "杯子",
  "references": [{{"object": "table", "object_cn": "桌子", "spatial_hint": "on"}}],
  "spatial_constraint": "target is above the table surface (higher z-coordinate)",
  "attributes": {{}},
  "needs_visual_evidence": false
}}
```

Query: "the pillow on the sofa closest to the door"
Output:
```json
{{
  "target": "pillow",
  "target_cn": "pillow",
  "references": [{{"object": "sofa", "object_cn": "sofa", "spatial_hint": "on"}}, {{"object": "door", "object_cn": "door", "spatial_hint": "closest to"}}],
  "spatial_constraint": "pillow is on top of sofa, and that sofa is the one nearest to the door",
  "attributes": {{}},
  "needs_visual_evidence": false
}}
```

Now parse this query:
Query: "{query}"
'''

    def __init__(self, llm_client=None, llm_url: str = None, llm_model: str = None):
        """
        Args:
            llm_client: LLM客户端实例
            llm_url: LLM服务URL (如果没有client)
            llm_model: LLM模型名称
        """
        self.llm_client = llm_client
        self.llm_url = llm_url or "http://10.21.231.7:8006"
        self.llm_model = llm_model or "gemini-2.0-flash"
    
    def parse(self, query: str) -> ParsedQuery:
        """
        解析查询
        
        Args:
            query: 自然语言查询
            
        Returns:
            ParsedQuery对象
        """
        # 尝试LLM解析
        try:
            return self._llm_parse(query)
        except Exception as e:
            print(f"LLM parse failed: {e}, falling back to rule-based")
            return self._rule_based_parse(query)
    
    def _llm_parse(self, query: str) -> ParsedQuery:
        """使用LLM解析"""
        import requests
        
        prompt = self.PARSE_PROMPT.format(query=query)
        
        response = requests.post(
            f"{self.llm_url}/v1/chat/completions",
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM request failed: {response.status_code}")
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # 提取JSON (多种格式尝试)
        json_str = None
        
        # 尝试1: ```json ... ```
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        
        # 尝试2: ``` ... ```
        if not json_str:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
        
        # 尝试3: 直接找 { ... }
        if not json_str:
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        # 尝试4: 找嵌套的 { ... }
        if not json_str:
            # 找第一个 { 和最后一个 }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end > start:
                json_str = content[start:end+1]
        
        if not json_str:
            raise Exception(f"No JSON found in response")
        
        parsed = json.loads(json_str)
        
        return ParsedQuery(
            raw_query=query,
            target=parsed.get("target", query),
            target_cn=parsed.get("target_cn", parsed.get("target", query)),
            references=[Reference(
                object=r["object"], 
                object_cn=r.get("object_cn", r["object"]),
                spatial_hint=r.get("spatial_hint", "")
            ) for r in parsed.get("references", [])],
            spatial_constraint=parsed.get("spatial_constraint", ""),
            attributes=parsed.get("attributes", {}),
            needs_visual_evidence=parsed.get("needs_visual_evidence", False),
        )
    
    # 常见物体中英对照
    CN_EN_DICT = {
        "沙发": "sofa", "台灯": "lamp", "灯": "lamp", "椅子": "chair", 
        "桌子": "table", "床": "bed", "柜子": "cabinet", "书架": "bookshelf",
        "门": "door", "窗户": "window", "窗": "window", "地毯": "carpet",
        "枕头": "pillow", "杯子": "cup", "花瓶": "vase", "植物": "plant",
        "电视": "TV", "电脑": "computer", "冰箱": "refrigerator",
        "画": "painting", "镜子": "mirror", "钟": "clock", "书": "book",
        "盆栽": "potted plant", "茶几": "coffee table", "餐桌": "dining table",
        "衣柜": "wardrobe", "床头柜": "nightstand", "抽屉": "drawer",
    }
    
    def _translate_to_english(self, text: str) -> str:
        """将中文物体名翻译为英文"""
        # 先检查是否已经是英文
        if text.isascii():
            return text
        
        # 查词典
        for cn, en in self.CN_EN_DICT.items():
            if cn in text:
                return en
        
        # 无法翻译，返回原文
        return text
    
    def _rule_based_parse(self, query: str) -> ParsedQuery:
        """基于规则的简单解析 (fallback)"""
        # 简单的中文模式匹配
        patterns = [
            # X旁边的Y
            (r'(.+?)(旁边|附近|边上)的(.+)', lambda m: (m.group(3), m.group(1), "next to", "near")),
            # X上的Y
            (r'(.+?)上的(.+)', lambda m: (m.group(2), m.group(1), "on", "above")),
            # X里的Y
            (r'(.+?)(里|内|中)的(.+)', lambda m: (m.group(3), m.group(1), "in", "inside")),
            # X和Y之间的Z
            (r'(.+?)和(.+?)之间的(.+)', lambda m: (m.group(3), m.group(1), "between", "between")),
        ]
        
        for pattern, extractor in patterns:
            match = re.match(pattern, query)
            if match:
                target_cn, ref_cn, hint, constraint = extractor(match)
                target_en = self._translate_to_english(target_cn.strip())
                ref_en = self._translate_to_english(ref_cn.strip())
                
                return ParsedQuery(
                    raw_query=query,
                    target=target_en,
                    target_cn=target_cn.strip(),
                    references=[Reference(ref_en, ref_cn.strip(), hint)],
                    spatial_constraint=constraint,
                )
        
        # 无法解析，整个query作为target
        target_en = self._translate_to_english(query)
        
        # 检查是否有颜色词
        color_words = ["红", "蓝", "绿", "黄", "白", "黑", "紫", "橙", "粉"]
        needs_visual = any(c in query for c in color_words)
        
        return ParsedQuery(
            raw_query=query,
            target=target_en,
            target_cn=query,
            needs_visual_evidence=needs_visual,
        )
    
    def __repr__(self):
        return f"QueryParser(llm_url={self.llm_url})"
