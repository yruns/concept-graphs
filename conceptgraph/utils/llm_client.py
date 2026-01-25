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
    # Test the model initialization
    print("Testing LLM Client Initialization...")
    print(f"Available models: {get_available_models()}")
    print()
    
    # Test with default model
    print("Testing default model (gpt-4o-2024-08-06)...")
    llm = get_langchain_chat_model()
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response.content}")
    print()
    
    # Test with specific model
    print("Testing gemini-3-flash-preview...")
    llm_gemini = get_langchain_chat_model("gemini-3-flash-preview")
    response = llm_gemini.invoke("What is 2+2?")
    print(f"Response: {response.content}")
