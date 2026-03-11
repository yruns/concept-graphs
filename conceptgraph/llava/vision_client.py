"""
统一的视觉模型客户端,基于 llm_client 进行调用
"""
import os
import base64
from io import BytesIO

from PIL import Image
import numpy as np
from langchain_core.messages import HumanMessage

from conceptgraph.utils.llm_client import (
    DEFAULT_MODEL,
    get_gemini_pool,
    get_langchain_chat_model,
)


def _resolve_use_gemini_pool(model_name: str) -> bool:
    """
    Determine whether GeminiClientPool should be used.

    Env `USE_GEMINI_POOL` supports:
    - true values: 1/true/yes/on
    - false values: 0/false/no/off
    - auto/default: enabled only for gemini-2.5-pro
    """
    raw = os.getenv("USE_GEMINI_POOL", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return model_name.lower() == "gemini-2.5-pro"


class UnifiedVisionChat:
    """统一的视觉对话客户端"""
    
    def __init__(self, model_name=None, base_url=None):
        """
        初始化统一视觉客户端
        
        Args:
            model_name: 模型名称(如果不提供,从环境变量读取)
            base_url: 保留参数,当前未使用
        """
        self.model_name = model_name or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.use_pool = _resolve_use_gemini_pool(self.model_name)
        self._pool = get_gemini_pool() if self.use_pool else None
        self._llm = None if self.use_pool else get_langchain_chat_model(
            deployment_name=self.model_name,
            use_pool=False,
        )
        backend = "GeminiClientPool" if self.use_pool else "llm_client"
        print(f"统一视觉客户端初始化: {self.model_name} ({backend})")
    
    def reset(self):
        """重置对话状态(为了兼容LLaVA接口)"""
        # 无状态,不需要重置
        pass
    
    def _image_to_data_url(self, image, max_size=512):
        """
        将图像转换为data URL，并压缩以减小请求大小
        
        Args:
            image: PIL Image 或 numpy 数组
            max_size: 最大边长（像素），超过会等比例缩放
        """
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # 等比例缩放图像以减小请求大小
        w, h = pil_image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        # 使用 JPEG 格式和适度压缩，显著减小文件大小
        pil_image.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    
    def __call__(self, query, image_features=None, image=None):
        """
        主接口方法,兼容LLaVA
        
        Args:
            query: 文本查询/提示
            image_features: 忽略(为了LLaVA兼容性)
            image: PIL Image或numpy数组(实际图像数据)
            
        Returns:
            str: 生成的文本响应
        """
        if image is None:
            # 纯文本查询
            return self._generate_text(query)
        else:
            # 视觉+文本查询
            return self._generate_vision(query, image)
    
    def _generate_text(self, prompt):
        """生成纯文本响应"""
        try:
            if self.use_pool:
                # Pool mode: fetch a fresh client to improve load balancing.
                client, _config_idx = self._pool.get_next_client(temperature=0.0)
                response = client.invoke(prompt)
            else:
                response = self._llm.invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as e:
            print(f"文本生成错误: {e}")
            return f"错误: {str(e)}"
    
    def _generate_vision(self, prompt, image, max_retries=5):
        """生成带图像的响应（带重试机制，利用 pool 轮换 API key）"""
        import time
        import random

        data_url = self._image_to_data_url(image)
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "low"  # 使用低分辨率模式，加快处理速度
                }
            },
        ])

        last_error = None
        for attempt in range(max_retries):
            try:
                if self.use_pool:
                    # 每次尝试从 pool 获取下一个 client (轮换 API key)
                    llm, _config_idx = self._pool.get_next_client(temperature=0.0)
                    response = llm.invoke([message])
                else:
                    response = self._llm.invoke([message])
                return getattr(response, "content", str(response)).strip()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # 检查是否为可重试错误（限流、连接、超时等）
                retryable_patterns = [
                    "connection", "timeout", "rate", "429", "qpm", "limit",
                    "quota", "too many", "overloaded", "temporarily"
                ]
                should_retry = any(p in error_str for p in retryable_patterns)

                if should_retry and attempt < max_retries - 1:
                    # 指数退避 + 随机抖动: 1-2s, 2-4s, 4-8s, 8-16s
                    base_wait = 2 ** attempt
                    jitter = random.uniform(0, base_wait)
                    wait_time = base_wait + jitter
                    print(f"视觉生成重试 {attempt + 1}/{max_retries}，等待 {wait_time:.1f}s")
                    time.sleep(wait_time)
                elif not should_retry:
                    # 非重试类错误，直接退出
                    print(f"视觉生成错误: {e}")
                    return "small object"

        print(f"视觉生成失败（已重试 {max_retries} 次）: {last_error}")
        return "small object"  # 返回后备描述
    
    def load_image(self, image_file):
        """从文件路径加载图像(为了兼容LLaVA接口)"""
        if image_file.startswith("http") or image_file.startswith("https"):
            import requests
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def encode_image(self, image_tensor):
        """
        虚拟方法,为了LLaVA兼容性。
        返回输入,因为统一客户端不需要预编码的特征。
        """
        return image_tensor


# 工厂函数,便于集成
def create_vision_chat(model_name=None, base_url=None):
    """创建UnifiedVisionChat实例"""
    return UnifiedVisionChat(model_name=model_name, base_url=base_url)


if __name__ == "__main__":
    # 测试适配器
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args()
    
    # 初始化客户端
    chat = create_vision_chat(args.model_name, args.base_url)
    print(f"统一视觉客户端初始化完成")
    
    # 加载并测试图像
    image = chat.load_image(args.image_file)
    
    # 测试视觉查询
    query = "总结图片三点"
    response = chat(query=query, image=image)
    print(f"查询: {query}")
    print(f"响应: {response}")
