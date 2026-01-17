"""
Ollama adapter to replace OpenAI GPT-4 functionality in ConceptGraphs
"""

import json
import requests
import time


class OllamaGPTChat:
    """Adapter class to use Ollama language models as a drop-in replacement for OpenAI GPT-4"""
    
    def __init__(self, model_name="llama3.1:8b", base_url="http://localhost:11434"):
        """
        Initialize Ollama GPT chat
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b", "qwen2.5:7b", "mistral:latest")
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test if Ollama is running and model is available
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama server is running and model is loaded"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name not in model_names:
                    print(f"警告: 模型 {self.model_name} 未在可用模型中找到: {model_names}")
                    print(f"尝试使用该模型，Ollama会自动下载...")
                else:
                    print(f"Ollama连接成功。使用模型: {self.model_name}")
            else:
                raise ConnectionError(f"Ollama服务器响应状态码: {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"无法连接到Ollama服务器 {self.base_url}: {e}")
    
    def create_completion(self, messages, timeout=60):
        """
        Create a chat completion (compatible with OpenAI API format)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with OpenAI-compatible format:
            {
                "choices": [
                    {
                        "message": {
                            "content": "response text"
                        }
                    }
                ]
            }
        """
        # Convert messages to a single prompt string
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Request JSON output format
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent JSON output
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            # Convert Ollama response format to OpenAI format
            response_text = result.get('response', '').strip()
            
            return {
                "choices": [
                    {
                        "message": {
                            "content": response_text
                        }
                    }
                ]
            }
        except requests.exceptions.Timeout:
            print(f"请求超时 ({timeout}秒)")
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"object_tag": "invalid", "summary": "请求超时", "possible_tags": []}'
                        }
                    }
                ]
            }
        except Exception as e:
            print(f"Ollama文本生成错误: {e}")
            return {
                "choices": [
                    {
                        "message": {
                            "content": f'{{"object_tag": "invalid", "summary": "错误: {str(e)}", "possible_tags": []}}'
                        }
                    }
                ]
            }


class OllamaGPTWrapper:
    """Wrapper to make Ollama compatible with OpenAI's ChatCompletion interface"""
    
    def __init__(self):
        self.chat_instance = None
    
    def initialize(self, model_name="llama3.1:8b"):
        """Initialize the Ollama GPT chat instance"""
        self.chat_instance = OllamaGPTChat(model_name=model_name)
        return self.chat_instance
    
    def create(self, model, messages, timeout=60):
        """
        OpenAI-compatible create method
        
        Args:
            model: Model name (used to initialize if not already done)
            messages: List of message dictionaries
            timeout: Request timeout in seconds
            
        Returns:
            Response in OpenAI format
        """
        if self.chat_instance is None:
            self.initialize(model)
        
        return self.chat_instance.create_completion(messages, timeout)


# Create a singleton instance to mimic openai.ChatCompletion
ChatCompletion = OllamaGPTWrapper()


def test_ollama_gpt():
    """Test function to verify Ollama GPT adapter works correctly"""
    print("测试 Ollama GPT 适配器...")
    
    # Initialize
    chat = OllamaGPTChat(model_name="llama3.1:8b")
    
    # Test messages
    test_messages = [
        {
            "role": "system",
            "content": "你是一个乐于助人的助手，以JSON格式回复。"
        },
        {
            "role": "user",
            "content": '请用JSON格式描述一个红色的苹果。字段包括: "color", "object", "description"'
        }
    ]
    
    # Get response
    print("\n发送测试请求...")
    response = chat.create_completion(test_messages)
    print("\n响应:")
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    test_ollama_gpt()

