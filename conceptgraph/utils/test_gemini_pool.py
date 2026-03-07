#!/usr/bin/env python3
"""
Test Gemini API key connectivity for client pool.
"""
import asyncio
import time
from typing import Dict, Any, List
from openai import AzureOpenAI


LLM_CONFIGS = [
    {
        "base_url": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro",
        "api_key": "UhwiVMPWPSy9Qk5aTiXeUgHiXOIABGKY_GPT_AK",
        "api_version": "2024-03-01-preview",
    },
    {
        "base_url": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro",
        "api_key": "rPbzfYhMWED5G6SBQRwGgrgsrSNA7ix5_GPT_AK",
        "api_version": "2024-03-01-preview",
    },
    {
        "base_url": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "model_name": "gemini-2.5-pro-preview-03-25",
        "api_key": "BaHKAkJz5tvH7EAerUgnmfUOVr3fEQ1s_GPT_AK",
        "api_version": "2024-03-01-preview",
    },
    {
        "base_url": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "model_name": "gemini-2.5-pro",
        "api_key": "cD6AGSVHrzftqONPxsFmgkVEuVlBynRb_GPT_AK",
        "api_version": "2024-03-01-preview",
    },
    {
        "base_url": "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        "model_name": "gemini-2.5-pro",
        "api_key": "K1Hn1GahMi3dpvLesYH67sS0S2Z1yFYE_GPT_AK",
        "api_version": "2024-03-01-preview",
    },
]


def test_single_config(config: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Test a single API configuration."""
    result = {
        "index": index,
        "base_url": config["base_url"],
        "model_name": config["model_name"],
        "api_key": config["api_key"][:8] + "...",
        "success": False,
        "latency_ms": None,
        "error": None,
    }

    try:
        client = AzureOpenAI(
            azure_endpoint=config["base_url"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            timeout=30,
        )

        start = time.time()
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": "Say 'OK' in one word."}],
            max_tokens=10,
            temperature=0.0,
        )
        latency = (time.time() - start) * 1000

        content = response.choices[0].message.content.strip()
        result["success"] = True
        result["latency_ms"] = round(latency, 1)
        result["response"] = content[:50]

    except Exception as e:
        result["error"] = str(e)[:100]

    return result


def main():
    print("=" * 70)
    print("Gemini API Key Connectivity Test")
    print("=" * 70)
    print(f"Testing {len(LLM_CONFIGS)} configurations...\n")

    results: List[Dict[str, Any]] = []

    for i, config in enumerate(LLM_CONFIGS):
        print(f"[{i+1}/{len(LLM_CONFIGS)}] Testing {config['model_name']} @ {config['api_key'][:8]}...")
        result = test_single_config(config, i)
        results.append(result)

        if result["success"]:
            print(f"    ✓ OK ({result['latency_ms']}ms) - {result.get('response', '')}")
        else:
            print(f"    ✗ FAILED: {result['error']}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    success_count = sum(1 for r in results if r["success"])
    print(f"Success: {success_count}/{len(results)}")

    if success_count > 0:
        avg_latency = sum(r["latency_ms"] for r in results if r["success"]) / success_count
        print(f"Average latency: {avg_latency:.1f}ms")

    print("\nWorking configs:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} [{r['index']}] {r['model_name']} ({r['api_key']})")

    # Export valid configs
    valid_configs = [LLM_CONFIGS[r["index"]] for r in results if r["success"]]
    print(f"\n{len(valid_configs)} valid configs ready for pool.")

    return valid_configs


if __name__ == "__main__":
    main()
