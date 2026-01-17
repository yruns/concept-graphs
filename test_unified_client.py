#!/usr/bin/env python3
"""
测试统一LLM客户端
"""
import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

def test_text_client():
    """测试纯文本客户端"""
    print("=" * 60)
    print("测试 1: 纯文本查询")
    print("=" * 60)
    
    from conceptgraph.llava.unified_client import chat_completions
    
    base_url = os.getenv("LLM_BASE_URL", "http://10.21.231.7:8005")
    model = os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")
    
    print(f"服务器: {base_url}")
    print(f"模型: {model}")
    print()
    
    try:
        response = chat_completions(
            messages=[
                {"role": "user", "content": "用一句话自我介绍"}
            ],
            model=model,
            base_url=base_url,
            timeout=30.0
        )
        
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        print("查询: 用一句话自我介绍")
        print(f"响应: {content}")
        print()
        print("✓ 纯文本测试通过")
        return True
    except Exception as e:
        print(f"✗ 纯文本测试失败: {e}")
        return False


def test_vision_client_with_image():
    """测试视觉客户端(带图像)"""
    print()
    print("=" * 60)
    print("测试 2: 视觉客户端(带图像)")
    print("=" * 60)
    
    from conceptgraph.llava.vision_client import create_vision_chat
    from PIL import Image
    import numpy as np
    
    # 创建一个简单的测试图像
    test_image = Image.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    )
    
    try:
        chat = create_vision_chat()
        print("统一视觉客户端初始化成功")
        print()
        
        # 测试图像查询
        response = chat(query="描述这个图像", image=test_image)
        print("查询: 描述这个图像")
        print(f"响应: {response}")
        print()
        print("✓ 视觉客户端(带图像)测试通过")
        return True
    except Exception as e:
        print(f"✗ 视觉客户端(带图像)测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_client_text_only():
    """测试视觉客户端(纯文本)"""
    print()
    print("=" * 60)
    print("测试 3: 视觉客户端(纯文本)")
    print("=" * 60)
    
    from conceptgraph.llava.vision_client import create_vision_chat
    
    try:
        chat = create_vision_chat()
        
        # 测试纯文本查询
        response = chat(query="1+1等于几?", image=None)
        print("查询: 1+1等于几?")
        print(f"响应: {response}")
        print()
        print("✓ 视觉客户端(纯文本)测试通过")
        return True
    except Exception as e:
        print(f"✗ 视觉客户端(纯文本)测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_with_path():
    """测试多模态(图像路径)"""
    print()
    print("=" * 60)
    print("测试 4: 多模态(图像路径)")
    print("=" * 60)
    
    from conceptgraph.llava.unified_client import chat_completions
    from PIL import Image
    import tempfile
    import numpy as np
    
    # 创建临时图像文件
    test_image = Image.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    )
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        test_image.save(temp_path)
    
    base_url = os.getenv("LLM_BASE_URL", "http://10.21.231.7:8005")
    model = os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")
    
    try:
        response = chat_completions(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是什么图像?"},
                        {"type": "image_path", "image_path": temp_path}
                    ]
                }
            ],
            model=model,
            base_url=base_url,
            timeout=30.0
        )
        
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        print("查询: 这是什么图像?")
        print(f"响应: {content}")
        print()
        print("✓ 多模态(图像路径)测试通过")
        
        # 清理临时文件
        os.unlink(temp_path)
        return True
    except Exception as e:
        print(f"✗ 多模态(图像路径)测试失败: {e}")
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("=" * 60)
    print("统一LLM客户端测试套件")
    print("=" * 60)
    print()
    
    # 检查环境变量
    base_url = os.getenv("LLM_BASE_URL", "http://10.21.231.7:8005")
    model = os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")
    
    print("配置:")
    print(f"  LLM_BASE_URL: {base_url}")
    print(f"  LLM_MODEL: {model}")
    print()
    
    # 运行测试
    results = []
    
    results.append(("纯文本客户端", test_text_client()))
    results.append(("视觉客户端(带图像)", test_vision_client_with_image()))
    results.append(("视觉客户端(纯文本)", test_vision_client_text_only()))
    results.append(("多模态(图像路径)", test_multimodal_with_path()))
    
    # 显示结果摘要
    print()
    print("=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print()
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print()
        print("✓ 所有测试通过!")
        return 0
    else:
        print()
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

