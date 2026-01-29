"""
LLM Model initialization based on environment configuration.
Supports both native OpenAI client and LangChain wrappers.
"""
import os
from typing import Optional, Dict, Any
from langchain_openai import AzureChatOpenAI


# Model configurations mapping
# Each model has its own endpoint and api_key
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-4o-2024-08-06": {
        "endpoint": "https://genai-va-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "api_key": "Qvylf4KKsq3GuvPsBilf7w4ynDicSfer",
        "api_version": "2024-02-15-preview",
    },
    "gpt-5.2-2025-12-11": {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "api_key": "Eyt11Oeoj77MfGcMweDRODBsbYnPkWUp",
        "api_version": "2024-03-01-preview",
    },
    "gemini-2.5-pro": {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "api_key": "K1Hn1GahMi3dpvLesYH67sS0S2Z1yFYE_GPT_AK",
        "api_version": "2024-02-15-preview",
    },
    "gemini-3-pro-preview-new": {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "api_key": "BaHKAkJz5tvH7EAerUgnmfUOVr3fEQ1s_GPT_AK",
        "api_version": "2024-02-15-preview",
    },
    "gemini-3-flash-preview": {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "api_key": "BaHKAkJz5tvH7EAerUgnmfUOVr3fEQ1s_GPT_AK",
        "api_version": "2024-02-15-preview",
    },
}

# Default model to use when none specified
DEFAULT_MODEL = "gpt-5.2-2025-12-11"


def get_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_langchain_chat_model(
    deployment_name: Optional[str] = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> AzureChatOpenAI:
    """
    Initialize and return LangChain Azure ChatOpenAI model.
    
    When deployment_name is provided, the function automatically fills in
    the corresponding endpoint and api_key from the MODEL_CONFIGS mapping.

    Args:
        deployment_name: Model deployment name. Supported values:
            - "gpt-5.2-2025-12-11" (default)
            - "gpt-4o-2024-08-06"
            - "gemini-2.5-pro"
            - "gemini-3-pro-preview-new"
            - "gemini-3-flash-preview"
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: None)
        **kwargs: Additional arguments passed to AzureChatOpenAI

    Returns:
        AzureChatOpenAI: Configured LangChain chat model

    Raises:
        ValueError: If deployment_name is not in MODEL_CONFIGS

    Example:
        >>> model = get_langchain_chat_model("gpt-5.2-2025-12-11")
        >>> response = model.invoke("Hello, how are you?")
        >>> print(response.content)
        
        >>> # Use gemini model
        >>> model = get_langchain_chat_model("gemini-2.5-pro")
        >>> response = model.invoke("Explain quantum computing")
    """
    # Use default model if none specified
    deployment = deployment_name or DEFAULT_MODEL
    
    # Validate deployment name
    if deployment not in MODEL_CONFIGS:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown deployment_name: '{deployment}'. "
            f"Available models: {available}"
        )
    
    # Get config for the specified model
    config = MODEL_CONFIGS[deployment]
    
    # 默认超时配置（视觉请求需要更长时间）
    default_timeout = kwargs.pop("timeout", 120)
    default_max_retries = kwargs.pop("max_retries", 3)
    
    model = AzureChatOpenAI(
        azure_deployment=deployment,
        model=deployment,
        api_key=config["api_key"],
        azure_endpoint=config["endpoint"],
        api_version=config["api_version"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=default_timeout,
        max_retries=default_max_retries,
        **kwargs
    )

    return model


# Convenience function aliases
def get_gpt52() -> AzureChatOpenAI:
    """Get GPT-5.2 model with default settings."""
    return get_langchain_chat_model("gpt-5.2-2025-12-11")


def get_gpt4o() -> AzureChatOpenAI:
    """Backward-compatible alias for GPT-5.2 default."""
    return get_gpt52()


def get_gemini_pro() -> AzureChatOpenAI:
    """Get Gemini 2.5 Pro model with default settings."""
    return get_langchain_chat_model("gemini-2.5-pro")


def get_gemini3_pro() -> AzureChatOpenAI:
    """Get Gemini 3 Pro Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-pro-preview-new")


def get_gemini3_flash() -> AzureChatOpenAI:
    """Get Gemini 3 Flash Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-flash-preview")


def test_vision_request(model_name: str = None, image_path: str = None) -> bool:
    """
    测试视觉请求是否正常工作。
    
    Args:
        model_name: 模型名称，默认使用 DEFAULT_MODEL
        image_path: 图像文件路径，默认使用内置测试图像
        
    Returns:
        bool: 测试是否通过
    """
    from langchain_core.messages import HumanMessage
    import base64
    from pathlib import Path
    
    model_name = model_name or DEFAULT_MODEL
    print(f"\n测试视觉请求: {model_name}")
    
    try:
        llm = get_langchain_chat_model(model_name, temperature=0.0)
        
        # 如果提供了图像路径，使用真实图像
        if image_path and Path(image_path).exists():
            print(f"  使用图像: {image_path}")
            with open(image_path, "rb") as f:
                image_data = f.read()
            # 检测图像格式
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
            data_url = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
            query = "请描述你在这张图像中看到了什么？用中文回答。"
        else:
            # 使用内置的测试图像
            print("  使用内置测试图像 (1x1 红色像素)")
            red_pixel_png = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIA"
                "C90WQgAAAABJRU5ErkJggg=="
            )
            data_url = f"data:image/png;base64,{red_pixel_png}"
            query = "What color is this pixel? Answer in one word."
        
        message = HumanMessage(content=[
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        
        import time
        start = time.time()
        response = llm.invoke([message])
        elapsed = time.time() - start
        
        content = getattr(response, "content", str(response))
        print(f"  ✓ 视觉响应 ({elapsed:.2f}s):")
        print(f"    {content}")
        return True
        
    except Exception as e:
        print(f"  ✗ 视觉测试失败: {e}")
        return False


if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import List
    import time
    import sys
    
    # Define test schema for structured output
    class TestObject(BaseModel):
        """A simple object for testing structured output."""
        name: str = Field(description="Name of the object")
        color: str = Field(description="Color of the object")
        size: str = Field(description="Size: small, medium, or large")
    
    class TestResponse(BaseModel):
        """Response containing a list of objects."""
        objects: List[TestObject] = Field(description="List of objects found")
        count: int = Field(description="Number of objects")
    
    # 检查是否只测试视觉
    # 用法: python llm_client.py --vision [model_name] [image_path]
    if len(sys.argv) > 1 and sys.argv[1] == "--vision":
        model = "gpt-4o-2024-08-06"
        image_path = None
        for arg in sys.argv[2:]:
            if arg.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_path = arg
            else:
                model = arg
        success = test_vision_request(model, image_path)
        sys.exit(0 if success else 1)
    
    print("=" * 60)
    print("LLM Client Test - All Models")
    print("=" * 60)
    print(f"Available models: {get_available_models()}")
    print()
    
    results = {}
    
    for model_name in get_available_models():
        print("-" * 60)
        print(f"Testing: {model_name}")
        print("-" * 60)
        
        try:
            llm = get_langchain_chat_model(model_name, temperature=0.0)
            
            # Test 1: Basic invoke
            print("  [1] Basic invoke...")
            start = time.time()
            response = llm.invoke("Say 'hello' in one word.")
            basic_time = time.time() - start
            print(f"      ✓ Response: {response.content[:50]}... ({basic_time:.2f}s)")
            
            # Test 2: Structured output
            print("  [2] Structured output...")
            start = time.time()
            structured_llm = llm.with_structured_output(TestResponse)
            response = structured_llm.invoke(
                "List 2 objects in a room: a red chair and a blue table."
            )
            struct_time = time.time() - start
            print(f"      ✓ Parsed {response.count} objects: {[o.name for o in response.objects]} ({struct_time:.2f}s)")
            
            results[model_name] = {"basic": True, "structured": True}
            print(f"  ✓ {model_name}: ALL PASSED")
            
        except Exception as e:
            results[model_name] = {"error": str(e)}
            print(f"  ✗ {model_name}: FAILED - {e}")
        
        print()
    
    # 额外测试默认模型的视觉请求
    print("-" * 60)
    print(f"Testing Vision: {DEFAULT_MODEL}")
    print("-" * 60)
    vision_ok = test_vision_request(DEFAULT_MODEL)
    results[f"{DEFAULT_MODEL}_vision"] = {"vision": vision_ok}
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for model, result in results.items():
        if "error" in result:
            print(f"  ✗ {model}: FAILED")
        elif "vision" in result:
            print(f"  {'✓' if result['vision'] else '✗'} {model}: {'OK' if result['vision'] else 'FAILED'}")
        else:
            print(f"  ✓ {model}: OK")
