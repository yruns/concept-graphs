"""VLM input construction and output parsing."""
from __future__ import annotations
import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import requests
from .data_structures import GroundingResult, ObjectNode, QueryInfo, QueryType

@dataclass
class VLMInput:
    images: List[np.ndarray]
    prompt: str
    object_annotations: Dict[int, str] = None
    bev_image: Optional[np.ndarray] = None

class VLMInputConstructor:
    def __init__(self, max_images: int = 4):
        self.max_images = max_images
    
    def construct(self, query_info: QueryInfo, objects: List[ObjectNode],
                  views: Dict[int, np.ndarray], descs: Dict[int, str],
                  bev: Optional[np.ndarray] = None) -> VLMInput:
        images = self._select_images(objects, views)
        prompt = self._build_prompt(query_info, objects, descs)
        annotations = {o.obj_id: f"[{o.obj_id}]{o.category}" for o in objects}
        return VLMInput(images, prompt, annotations, bev if query_info.use_bev else None)
    
    def _select_images(self, objects, views):
        scores = {}
        for obj in objects:
            for v in obj.best_view_ids[:3]:
                if v in views: scores[v] = scores.get(v, 0) + 1
        sorted_v = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [views[v] for v in sorted_v[:self.max_images] if v in views]
    
    def _build_prompt(self, q, objects, descs):
        obj_list = "\n".join([f"- ID {o.obj_id}: {o.category}" for o in objects])
        return f'找到: {q.original_query}\n候选物体:\n{obj_list}\n返回JSON: {{"object_id": <ID>, "reasoning": "..."}}'

class VLMOutputParser:
    def __init__(self, objects: List[ObjectNode]):
        self.objects = {o.obj_id: o for o in objects}
    
    def parse(self, output: str, query_info: QueryInfo) -> GroundingResult:
        for fn in [self._try_json, self._try_regex, self._try_desc]:
            r = fn(output)
            if r: return r
        return GroundingResult.failure("Could not parse VLM output")
    
    def _try_json(self, output):
        try:
            m = re.search(r'\{[^{}]+\}', output, re.DOTALL)
            if m:
                d = json.loads(m.group())
                oid = d.get("object_id")
                if oid in self.objects:
                    return GroundingResult.from_object(self.objects[oid], d.get("confidence", 0.8), d.get("reasoning", ""))
        except: pass
        return None
    
    def _try_regex(self, output):
        for p in [r'ID\s*[:=]?\s*(\d+)', r'\[(\d+)\]', r'object\s*(\d+)']:
            m = re.search(p, output, re.I)
            if m:
                oid = int(m.group(1))
                if oid in self.objects: return GroundingResult.from_object(self.objects[oid], 0.6)
        return None
    
    def _try_desc(self, output):
        for oid, obj in self.objects.items():
            if obj.category.lower() in output.lower():
                return GroundingResult.from_object(obj, 0.4)
        return None

class VLMClient:
    def __init__(self, url: str = "http://localhost:11434", model: str = "llava:7b"):
        self.url, self.model = url, model
    
    def infer(self, inp: VLMInput) -> str:
        imgs = []
        for img in inp.images:
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            imgs.append(base64.b64encode(buf).decode())
        if inp.bev_image is not None:
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(inp.bev_image, cv2.COLOR_RGB2BGR))
            imgs.append(base64.b64encode(buf).decode())
        try:
            content = [{"type": "text", "text": inp.prompt}]
            for b in imgs: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
            r = requests.post(f"{self.url}/v1/chat/completions",
                json={"model": self.model, "messages": [{"role": "user", "content": content}]}, timeout=60)
            if r.ok: return r.json()["choices"][0]["message"]["content"]
        except: pass
        try:
            r = requests.post(f"{self.url}/api/generate",
                json={"model": self.model, "prompt": inp.prompt, "images": imgs, "stream": False}, timeout=60)
            if r.ok: return r.json().get("response", "")
        except: pass
        raise ConnectionError("VLM unavailable")
