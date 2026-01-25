"""
LLM Model initialization based on environment configuration.
Supports both native OpenAI client and LangChain wrappers.
"""
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
DEFAULT_MODEL = "gpt-4o-2024-08-06"


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
            - "gpt-4o-2024-08-06" (default)
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
        >>> model = get_langchain_chat_model("gpt-4o-2024-08-06")
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
    
    model = AzureChatOpenAI(
        azure_deployment=deployment,
        model=deployment,
        api_key=config["api_key"],
        azure_endpoint=config["endpoint"],
        api_version=config["api_version"],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

    return model


# Convenience function aliases
def get_gpt4o() -> AzureChatOpenAI:
    """Get GPT-4o model with default settings."""
    return get_langchain_chat_model("gpt-4o-2024-08-06")


def get_gemini_pro() -> AzureChatOpenAI:
    """Get Gemini 2.5 Pro model with default settings."""
    return get_langchain_chat_model("gemini-2.5-pro")


def get_gemini3_pro() -> AzureChatOpenAI:
    """Get Gemini 3 Pro Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-pro-preview-new")


def get_gemini3_flash() -> AzureChatOpenAI:
    """Get Gemini 3 Flash Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-flash-preview")


if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import List
    import time
    
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
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for model, result in results.items():
        if "error" in result:
            print(f"  ✗ {model}: FAILED")
        else:
            print(f"  ✓ {model}: OK")
