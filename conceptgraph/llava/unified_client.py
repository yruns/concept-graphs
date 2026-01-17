"""
统一的LLM客户端,替换所有OpenAI GPT和Ollama调用
"""
import os
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Union, Iterable
import httpx

def _to_data_url(b64: str, mime: str) -> str:
    mime = mime or "image/png"
    return f"data:{mime};base64,{b64}"

def _read_file_to_b64(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _normalize_content(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """将content统一转换为数组格式 [{"type": "text", "text": "..."}]"""
    if isinstance(content, str):
        # 将字符串转换为数组格式，兼容要求数组格式的LLM服务
        return [{"type": "text", "text": content}]
    out: List[Dict[str, Any]] = []
    for item in content or []:
        t = item.get("type")
        if t == "text":
            out.append({"type": "text", "text": item.get("text", "")})
            continue
        if t == "image_url":
            out.append({"type": "image_url", "image_url": item.get("image_url")})
            continue
        if t == "image_path":
            p = item.get("image_path") or item.get("path")
            b64 = _read_file_to_b64(p)
            mime, _ = mimetypes.guess_type(p)
            out.append({"type": "image_url", "image_url": {"url": _to_data_url(b64, mime or "image/png")}})
            continue
        if t == "image_base64":
            b64 = item.get("image_base64") or item.get("base64")
            mime = item.get("mime_type") or item.get("mime") or "image/png"
            out.append({"type": "image_url", "image_url": {"url": _to_data_url(b64, mime)}})
            continue
        if "path" in item and not t:
            p = item["path"]
            b64 = _read_file_to_b64(p)
            mime, _ = mimetypes.guess_type(p)
            out.append({"type": "image_url", "image_url": {"url": _to_data_url(b64, mime or "image/png")}})
            continue
        out.append(item)
    return out

def chat_completions(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    base_url: str = "http://localhost:8000",
    logid: Optional[str] = None,
    timeout: float = 60.0,
) -> Union[Dict[str, Any], Iterable[bytes]]:
    assert model is not None, "model is required"

    norm = [{"role": m.get("role"), "content": _normalize_content(m.get("content"))} for m in messages]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": norm,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    headers = {"Content-Type": "application/json"}
    if logid:
        headers["X-TT-LOGID"] = logid
    url = base_url.rstrip("/") + "/v1/chat/completions"
    client = httpx.Client(timeout=httpx.Timeout(timeout))
    if stream:
        def _gen():
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                for chunk in resp.iter_bytes():
                    yield chunk
        return _gen()
    r = client.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

