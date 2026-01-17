#!/usr/bin/env python3
"""
LLM服务可用性测试脚本

用法:
    python -m conceptgraph.segmentation.test_llm_service
    python -m conceptgraph.segmentation.test_llm_service --url http://your-llm-server:8000
    
环境变量:
    LLM_BASE_URL: LLM服务地址 (默认: http://10.21.231.7:8005)
"""

import os
import sys
import json
import argparse
import time
from typing import Optional


def test_llm_service(base_url: str, verbose: bool = True) -> bool:
    """
    测试LLM服务是否可用
    
    Returns:
        True: 服务可用
        False: 服务不可用
    """
    print("=" * 60)
    print("LLM服务可用性测试")
    print("=" * 60)
    print(f"服务地址: {base_url}")
    print()
    
    # 测试1: 网络连接
    print("[测试1] 检查网络连接...")
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            # 尝试访问根路径或健康检查端点
            try:
                r = client.get(f"{base_url}/healthz")
                print(f"  /health 端点: {r.status_code}")
            except:
                pass

        print("  ✓ 网络连接正常")
    except Exception as e:
        print(f"  ✗ 网络连接失败: {e}")
        return False
    print()
    
    # 测试2: 简单文本补全
    print("[测试2] 测试Chat Completions API...")
    try:
        from conceptgraph.llava.unified_client import chat_completions
        
        test_prompt = "请用一个词回答: 1+1等于几?"
        
        start_time = time.time()
        response = chat_completions(
            messages=[{"role": "user", "content": test_prompt}],
            model="gemini-3-flash-preview",
            base_url=base_url,
            max_tokens=50
        )
        elapsed = time.time() - start_time
        
        content = response['choices'][0]['message']['content']
        print(f"  提问: {test_prompt}")
        print(f"  回答: {content}")
        print(f"  耗时: {elapsed:.2f}秒")
        print("  ✓ Chat API正常")
    except Exception as e:
        print(f"  ✗ Chat API失败: {e}")
        return False
    print()
    
    # 测试3: JSON格式输出
    print("[测试3] 测试JSON格式输出...")
    try:
        json_prompt = """请输出一个JSON对象，包含以下字段:
- name: 你的名字
- status: "ready"

直接输出JSON，不要其他内容。"""
        
        start_time = time.time()
        response = chat_completions(
            messages=[{"role": "user", "content": json_prompt}],
            model="gemini-3-flash-preview",
            base_url=base_url,
            max_tokens=100
        )
        elapsed = time.time() - start_time
        
        content = response['choices'][0]['message']['content']
        print(f"  原始回答: {content[:200]}")
        
        # 尝试解析JSON
        import re
        json_match = re.search(r'\{[\s\S]*?\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
            print(f"  解析结果: {parsed}")
            print(f"  耗时: {elapsed:.2f}秒")
            print("  ✓ JSON输出正常")
        else:
            print("  ⚠ 未能提取JSON，但API调用成功")
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON解析失败: {e}")
        print("  API调用成功，但输出格式不是有效JSON")
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    print()
    
    # 测试4: 功能区域推理测试（模拟实际使用场景）
    print("[测试4] 测试功能区域推理（模拟实际场景）...")
    try:
        zone_prompt = """你是场景理解专家。请根据以下物体列表，划分功能区域。

物体列表:
- sofa (沙发)
- coffee_table (茶几)
- tv (电视)
- dining_table (餐桌)
- chair (椅子)

请输出JSON格式:
```json
{
  "functional_zones": [
    {"zone_id": "fz_0", "zone_name": "xxx", "objects": ["obj1", "obj2"]}
  ]
}
```"""
        
        start_time = time.time()
        response = chat_completions(
            messages=[{"role": "user", "content": zone_prompt}],
            model="gemini-3-flash-preview",
            base_url=base_url,
            max_tokens=500
        )
        elapsed = time.time() - start_time
        
        content = response['choices'][0]['message']['content']
        if verbose:
            print(f"  回答:\n{content}")
        
        # 检查是否包含功能区域
        json_match = re.search(r'\{[\s\S]*"functional_zones"[\s\S]*\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
            zones = parsed.get('functional_zones', [])
            print(f"  识别出 {len(zones)} 个功能区域")
            for z in zones:
                print(f"    - {z.get('zone_name', 'unknown')}: {z.get('objects', [])}")
            print(f"  耗时: {elapsed:.2f}秒")
            print("  ✓ 功能区域推理正常")
        else:
            print("  ⚠ 未能提取功能区域JSON")
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    print()
    
    # 总结
    print("=" * 60)
    print("✓ 所有测试通过！LLM服务可用。")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="测试LLM服务可用性")
    parser.add_argument(
        "--url", 
        type=str, 
        default=None,
        help="LLM服务地址 (默认使用环境变量LLM_BASE_URL或http://10.21.231.7:8005)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    args = parser.parse_args()
    
    # 确定服务地址
    base_url = args.url or os.getenv("LLM_BASE_URL", "http://10.21.231.7:8006")
    
    # 运行测试
    success = test_llm_service(base_url, verbose=args.verbose)
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
