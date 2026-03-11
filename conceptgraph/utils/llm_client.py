"""
LLM Model initialization based on environment configuration.
Supports both native OpenAI client and LangChain wrappers.
Includes Gemini client pool for high-concurrency parallel requests.
"""
import os
import threading
import asyncio
from typing import Optional, Dict, Any, List, Callable, TypeVar, Tuple
from langchain_openai import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

T = TypeVar("T")


# ============================================================================
# Gemini Client Pool Configuration
# ============================================================================
# weight: relative weight for weighted round-robin (higher = more requests)
GEMINI_POOL_CONFIGS: List[Dict[str, Any]] = [
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro",
        "api_key": "UhwiVMPWPSy9Qk5aTiXeUgHiXOIABGKY_GPT_AK",
        "api_version": "2024-03-01-preview",
        "weight": 1,
    },
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro",
        "api_key": "rPbzfYhMWED5G6SBQRwGgrgsrSNA7ix5_GPT_AK",
        "api_version": "2024-03-01-preview",
        "weight": 1,
    },
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro",
        "api_key": "BaHKAkJz5tvH7EAerUgnmfUOVr3fEQ1s_GPT_AK",
        "api_version": "2024-03-01-preview",
        "weight": 1,
    },
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "model_name": "gemini-2.5-pro",
        "api_key": "cD6AGSVHrzftqONPxsFmgkVEuVlBynRb_GPT_AK",
        "api_version": "2024-03-01-preview",
        "weight": 3,  # 3x quota
    },
    {
        "endpoint": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "model_name": "gemini-2.5-pro",
        "api_key": "K1Hn1GahMi3dpvLesYH67sS0S2Z1yFYE_GPT_AK",
        "api_version": "2024-03-01-preview",
        "weight": 1,
    },
]


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit / QPS limit error."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "rate limit", "rate_limit", "ratelimit",
        "too many requests", "429",
        "quota exceeded", "quota_exceeded",
        "qps limit", "qps_limit",
        "resource exhausted",
    ])


class GeminiClientPool:
    """
    Weighted round-robin client pool for Gemini API with QPS limit handling.

    Features:
    - Weighted round-robin: Keys with higher weight get more requests
    - Auto-retry on rate limit: Automatically switches to next key on QPS limit
    - Rate limit monitoring: Warns when a key's rate limit ratio exceeds 50%
    - Thread-safe for concurrent access

    Usage:
        pool = GeminiClientPool.get_instance()

        # Simple usage (auto-retry on rate limit)
        result = pool.invoke_with_retry(prompt)

        # Get client for manual use
        client = pool.get_client()

        # Parallel execution with auto-retry
        results = pool.map_parallel(func, items, max_workers=5)
    """

    _instance: Optional["GeminiClientPool"] = None
    _lock = threading.Lock()

    def __init__(self, configs: List[Dict[str, Any]] = None):
        self._configs = configs or GEMINI_POOL_CONFIGS
        self._clients: List[AzureChatOpenAI] = []

        # Build weighted index list for round-robin
        # e.g., weights [1,1,1,3,1] -> indices [0,1,2,3,3,3,4]
        self._weighted_indices: List[int] = []
        for i, config in enumerate(self._configs):
            weight = config.get("weight", 1)
            self._weighted_indices.extend([i] * weight)

        self._index = 0
        self._index_lock = threading.Lock()
        self._initialized = False

        # Rate limit tracking per key (thread-safe counters)
        self._request_counts: Dict[int, int] = {}
        self._rate_limit_counts: Dict[int, int] = {}
        self._stats_lock = threading.Lock()

        # Warning threshold
        self._rate_limit_warn_threshold = 0.5  # 50%

    @classmethod
    def get_instance(cls) -> "GeminiClientPool":
        """Get singleton instance of the pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _ensure_initialized(self):
        """Lazy initialization of clients."""
        if self._initialized:
            return

        with self._index_lock:
            if self._initialized:
                return

            for i, config in enumerate(self._configs):
                client = AzureChatOpenAI(
                    azure_deployment=config["model_name"],
                    model=config["model_name"],
                    api_key=config["api_key"],
                    azure_endpoint=config["endpoint"],
                    api_version=config["api_version"],
                    temperature=0.0,
                    timeout=120,
                    max_retries=0,  # We handle retries ourselves
                )
                self._clients.append(client)
                self._request_counts[i] = 0
                self._rate_limit_counts[i] = 0

            self._initialized = True

    def _get_key_id(self, config_index: int) -> str:
        """Get short identifier for a key (for logging)."""
        api_key = self._configs[config_index]["api_key"]
        return api_key[:8] + "..."

    def _record_request(self, config_index: int, rate_limited: bool = False):
        """Record request statistics for a key."""
        with self._stats_lock:
            # Initialize counters if not present (for lazy init compatibility)
            if config_index not in self._request_counts:
                self._request_counts[config_index] = 0
                self._rate_limit_counts[config_index] = 0

            self._request_counts[config_index] += 1
            if rate_limited:
                self._rate_limit_counts[config_index] += 1

            # Check if rate limit ratio exceeds threshold
            total = self._request_counts[config_index]
            limited = self._rate_limit_counts[config_index]
            if total >= 10 and limited / total > self._rate_limit_warn_threshold:
                from loguru import logger
                key_id = self._get_key_id(config_index)
                ratio = limited / total * 100
                logger.warning(
                    f"[GeminiPool] Key {key_id} rate limit ratio: {ratio:.1f}% "
                    f"({limited}/{total} requests) - exceeds {self._rate_limit_warn_threshold*100:.0f}% threshold"
                )

    def get_next_config_index(self) -> int:
        """Get next config index using weighted round-robin."""
        with self._index_lock:
            weighted_idx = self._weighted_indices[self._index]
            self._index = (self._index + 1) % len(self._weighted_indices)
            return weighted_idx

    def get_client(self) -> AzureChatOpenAI:
        """Get next client using weighted round-robin selection."""
        self._ensure_initialized()
        config_idx = self.get_next_config_index()
        return self._clients[config_idx]

    def get_next_client(
        self,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Tuple[AzureChatOpenAI, int]:
        """
        Get next client with config index for rate limit tracking.

        Returns:
            Tuple of (client, config_idx) for use with record_request()
        """
        self._ensure_initialized()
        config_idx = self.get_next_config_index()
        config = self._configs[config_idx]

        client = AzureChatOpenAI(
            azure_deployment=config["model_name"],
            model=config["model_name"],
            api_key=config["api_key"],
            azure_endpoint=config["endpoint"],
            api_version=config["api_version"],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            max_retries=0,  # We handle retries ourselves
        )
        return client, config_idx

    def record_request(self, config_idx: int, rate_limited: bool = False):
        """Record request result for rate limit tracking (public API)."""
        self._record_request(config_idx, rate_limited)

    def get_config_index_for_client(self, client: AzureChatOpenAI) -> int:
        """Find config index for a given client."""
        for i, c in enumerate(self._clients):
            if c is client:
                return i
        # Fallback: match by api_key
        for i, config in enumerate(self._configs):
            if config["api_key"] == client.api_key:
                return i
        return 0

    def invoke_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Invoke LLM with automatic retry on rate limit errors.

        Tries all keys in the pool before giving up.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLM response

        Raises:
            Exception: If all keys are rate limited or other error occurs
        """
        self._ensure_initialized()
        from loguru import logger

        tried_indices = set()
        last_error = None

        while len(tried_indices) < len(self._configs):
            config_idx = self.get_next_config_index()

            # Skip if already tried
            if config_idx in tried_indices:
                continue

            tried_indices.add(config_idx)
            config = self._configs[config_idx]
            key_id = self._get_key_id(config_idx)

            try:
                client = AzureChatOpenAI(
                    azure_deployment=config["model_name"],
                    model=config["model_name"],
                    api_key=config["api_key"],
                    azure_endpoint=config["endpoint"],
                    api_version=config["api_version"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                    max_retries=0,
                )

                result = client.invoke(prompt)
                self._record_request(config_idx, rate_limited=False)
                return result

            except Exception as e:
                if _is_rate_limit_error(e):
                    self._record_request(config_idx, rate_limited=True)
                    logger.warning(f"[GeminiPool] Key {key_id} rate limited, trying next key...")
                    last_error = e
                    continue
                else:
                    # Non-rate-limit error, propagate immediately
                    self._record_request(config_idx, rate_limited=False)
                    raise

        # All keys exhausted
        logger.error(f"[GeminiPool] All {len(self._configs)} keys exhausted due to rate limits")
        raise last_error or RuntimeError("All API keys rate limited")

    @property
    def pool_size(self) -> int:
        """Number of unique keys in the pool."""
        return len(self._configs)

    @property
    def effective_pool_size(self) -> int:
        """Effective pool size considering weights."""
        return len(self._weighted_indices)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limit statistics for all keys."""
        with self._stats_lock:
            stats = {}
            for i in range(len(self._configs)):
                key_id = self._get_key_id(i)
                total = self._request_counts.get(i, 0)
                limited = self._rate_limit_counts.get(i, 0)
                ratio = limited / total if total > 0 else 0
                stats[key_id] = {
                    "total_requests": total,
                    "rate_limited": limited,
                    "rate_limit_ratio": ratio,
                    "weight": self._configs[i].get("weight", 1),
                }
            return stats

    def reset_stats(self):
        """Reset rate limit statistics."""
        with self._stats_lock:
            for i in range(len(self._configs)):
                self._request_counts[i] = 0
                self._rate_limit_counts[i] = 0

    def map_parallel(
        self,
        func: Callable[[AzureChatOpenAI, T], Any],
        items: List[T],
        max_workers: Optional[int] = None,
    ) -> List[Any]:
        """
        Execute func in parallel across pool clients with auto-retry on rate limit.

        Args:
            func: Function taking (client, item) and returning result
            items: List of items to process
            max_workers: Max concurrent threads (default: effective_pool_size)

        Returns:
            List of results in same order as items
        """
        self._ensure_initialized()
        from loguru import logger

        if max_workers is None:
            max_workers = min(len(items), self.effective_pool_size)

        results = [None] * len(items)
        errors = [None] * len(items)

        def worker(idx: int, item: T):
            tried_indices = set()
            last_error = None

            while len(tried_indices) < len(self._configs):
                config_idx = self.get_next_config_index()
                if config_idx in tried_indices:
                    continue

                tried_indices.add(config_idx)
                client = self._clients[config_idx]
                key_id = self._get_key_id(config_idx)

                try:
                    result = func(client, item)
                    self._record_request(config_idx, rate_limited=False)
                    return idx, result, None

                except Exception as e:
                    if _is_rate_limit_error(e):
                        self._record_request(config_idx, rate_limited=True)
                        logger.debug(f"[GeminiPool] Key {key_id} rate limited for item {idx}, trying next...")
                        last_error = e
                        continue
                    else:
                        self._record_request(config_idx, rate_limited=False)
                        return idx, None, e

            return idx, None, last_error

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, i, item) for i, item in enumerate(items)]
            for future in as_completed(futures):
                idx, result, error = future.result()
                results[idx] = result
                errors[idx] = error

        # Check for errors
        for i, err in enumerate(errors):
            if err is not None:
                logger.error(f"[GeminiPool] Item {i} failed: {err}")

        return results

    async def amap_parallel(
        self,
        func: Callable[[AzureChatOpenAI, T], Any],
        items: List[T],
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """
        Async version of map_parallel using asyncio.

        Args:
            func: Async or sync function taking (client, item)
            items: List of items to process
            max_concurrency: Max concurrent tasks (default: effective_pool_size)

        Returns:
            List of results in same order as items
        """
        self._ensure_initialized()

        if max_concurrency is None:
            max_concurrency = min(len(items), self.effective_pool_size)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_worker(idx: int, item: T):
            async with semaphore:
                client = self.get_client()
                if asyncio.iscoroutinefunction(func):
                    result = await func(client, item)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, client, item)
                return idx, result

        tasks = [bounded_worker(i, item) for i, item in enumerate(items)]
        completed = await asyncio.gather(*tasks)

        results = [None] * len(items)
        for idx, result in completed:
            results[idx] = result

        return results


# ============================================================================
# Legacy Model Configurations (for backward compatibility)
# ============================================================================
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
        "api_version": "2024-03-01-preview",
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

# New default: gemini-2.5-pro for consistency
DEFAULT_MODEL = "gemini-2.5-pro"


def get_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_langchain_chat_model(
    deployment_name: Optional[str] = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
    use_pool: bool = False,
    **kwargs
) -> AzureChatOpenAI:
    """
    Initialize and return LangChain Azure ChatOpenAI model.

    Args:
        deployment_name: Model deployment name. Supported values:
            - "gemini-2.5-pro" (default)
            - "gpt-5.2-2025-12-11"
            - "gpt-4o-2024-08-06"
            - "gemini-3-pro-preview-new"
            - "gemini-3-flash-preview"
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens in response (default: None)
        use_pool: If True and model is gemini-2.5-pro, use pool client
        **kwargs: Additional arguments passed to AzureChatOpenAI

    Returns:
        AzureChatOpenAI: Configured LangChain chat model
    """
    deployment = deployment_name or DEFAULT_MODEL

    # Use pool for gemini-2.5-pro if requested
    if use_pool and deployment == "gemini-2.5-pro":
        pool = GeminiClientPool.get_instance()
        return pool.get_client_with_config(
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens,
        )

    # Validate deployment name
    if deployment not in MODEL_CONFIGS:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown deployment_name: '{deployment}'. "
            f"Available models: {available}"
        )

    config = MODEL_CONFIGS[deployment]

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


# ============================================================================
# Convenience Functions
# ============================================================================
def get_gemini_pool() -> GeminiClientPool:
    """Get the Gemini client pool singleton."""
    return GeminiClientPool.get_instance()


def get_gemini_client() -> AzureChatOpenAI:
    """Get next Gemini client from pool (round-robin)."""
    return GeminiClientPool.get_instance().get_client()


def get_gpt52() -> AzureChatOpenAI:
    """Get GPT-5.2 model with default settings."""
    return get_langchain_chat_model("gpt-5.2-2025-12-11")


def get_gpt4o() -> AzureChatOpenAI:
    """Backward-compatible alias for GPT-5.2 default."""
    return get_gpt52()


def get_gemini_pro() -> AzureChatOpenAI:
    """Get Gemini 2.5 Pro model (from pool)."""
    return get_gemini_client()


def get_gemini3_pro() -> AzureChatOpenAI:
    """Get Gemini 3 Pro Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-pro-preview-new")


def get_gemini3_flash() -> AzureChatOpenAI:
    """Get Gemini 3 Flash Preview model with default settings."""
    return get_langchain_chat_model("gemini-3-flash-preview")


def test_vision_request(model_name: str = None, image_path: str = None) -> bool:
    """
    Test vision request functionality.
    """
    from langchain_core.messages import HumanMessage
    import base64
    from pathlib import Path

    model_name = model_name or DEFAULT_MODEL
    print(f"\nTesting vision: {model_name}")

    try:
        llm = get_langchain_chat_model(model_name, temperature=0.0)

        if image_path and Path(image_path).exists():
            print(f"  Using image: {image_path}")
            with open(image_path, "rb") as f:
                image_data = f.read()
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            data_url = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
            query = "Describe this image briefly."
        else:
            print("  Using test image (1x1 red pixel)")
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
        print(f"  OK ({elapsed:.2f}s): {content[:50]}")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import List
    import time
    import sys

    class TestObject(BaseModel):
        name: str = Field(description="Name of the object")
        color: str = Field(description="Color of the object")

    class TestResponse(BaseModel):
        objects: List[TestObject] = Field(description="List of objects found")
        count: int = Field(description="Number of objects")

    if len(sys.argv) > 1 and sys.argv[1] == "--pool":
        print("=" * 60)
        print("Gemini Pool Concurrency Test")
        print("=" * 60)

        pool = get_gemini_pool()
        print(f"Pool size: {pool.pool_size}")

        queries = [
            "Say 'A'", "Say 'B'", "Say 'C'",
            "Say 'D'", "Say 'E'", "Say 'F'",
            "Say 'G'", "Say 'H'", "Say 'I'", "Say 'J'",
        ]

        def invoke_query(client, query):
            response = client.invoke(query)
            return response.content.strip()

        print(f"\nRunning {len(queries)} queries in parallel...")
        start = time.time()
        results = pool.map_parallel(invoke_query, queries)
        elapsed = time.time() - start

        print(f"\nResults ({elapsed:.2f}s total):")
        for q, r in zip(queries, results):
            print(f"  {q} -> {r}")

        print(f"\nAvg time per query: {elapsed/len(queries):.2f}s")
        print(f"Speedup vs serial: ~{len(queries)}x")

    elif len(sys.argv) > 1 and sys.argv[1] == "--vision":
        model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
        image_path = sys.argv[3] if len(sys.argv) > 3 else None
        success = test_vision_request(model, image_path)
        sys.exit(0 if success else 1)

    else:
        print("=" * 60)
        print("LLM Client Test")
        print("=" * 60)
        print(f"Available models: {get_available_models()}")
        print(f"Default model: {DEFAULT_MODEL}")
        print(f"Pool size: {get_gemini_pool().pool_size}")
        print()

        # Quick test with pool
        print("Testing Gemini pool client...")
        client = get_gemini_client()
        start = time.time()
        response = client.invoke("Say 'hello' in one word.")
        print(f"  OK ({time.time() - start:.2f}s): {response.content}")

        print("\nUsage:")
        print("  python llm_client.py --pool    # Test parallel execution")
        print("  python llm_client.py --vision  # Test vision request")
