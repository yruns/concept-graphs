"""
统一的视觉模型客户端,替换LLaVA和Ollama视觉调用
"""
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from conceptgraph.llava.unified_client import chat_completions


class UnifiedVisionChat:
    """统一的视觉对话客户端"""
    
    def __init__(self, model_name=None, base_url=None):
        """
        初始化统一视觉客户端
        
        Args:
            model_name: 模型名称(如果不提供,从环境变量读取)
            base_url: 服务器URL(如果不提供,从环境变量读取)
        """
        self.model_name = model_name or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        if not self.model_name or not self.base_url:
            raise ValueError("必须设置环境变量 LLM_MODEL 和 LLM_BASE_URL，或通过参数传入")
        print(f"统一视觉客户端初始化: {self.model_name} @ {self.base_url}")
    
    def reset(self):
        """重置对话状态(为了兼容LLaVA接口)"""
        # 无状态,不需要重置
        pass
    
    def _image_to_temp_path(self, image):
        """将PIL Image保存为临时文件并返回路径"""
        import tempfile
        if isinstance(image, Image.Image):
            # 保存PIL图像到临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image.save(temp_file.name, format='PNG')
            return temp_file.name
        elif isinstance(image, np.ndarray):
            # 转换numpy数组为PIL然后保存
            pil_image = Image.fromarray(image.astype(np.uint8))
            return self._image_to_temp_path(pil_image)
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
    
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
            messages = [
                {"role": "user", "content": prompt}
            ]
            result = chat_completions(
                messages=messages,
                model=self.model_name,
                base_url=self.base_url,
                timeout=60.0
            )
            return result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        except Exception as e:
            print(f"文本生成错误: {e}")
            return f"错误: {str(e)}"
    
    def _generate_vision(self, prompt, image):
        """生成带图像的响应"""
        # 将图像保存为临时文件
        temp_path = self._image_to_temp_path(image)
        
        try:
            # 使用统一客户端的多模态功能
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_path", "image_path": temp_path}
                    ]
                }
            ]
            result = chat_completions(
                messages=messages,
                model=self.model_name,
                base_url=self.base_url,
                timeout=60.0
            )
            response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return response_text
        except Exception as e:
            print(f"视觉生成错误: {e}")
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass
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

