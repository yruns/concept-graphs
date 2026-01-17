#!/usr/bin/env python3
"""
步骤5b+: 精炼物体描述并提取Affordance（带图像）

将步骤5的caption精炼和4.5b的affordance提取合并，同时利用物体图像获得更准确的结果。

输入:
  - 3D物体地图 (pcd文件): 包含物体的color_path, xyxy, mask
  - 步骤4b的captions: sg_cache_detect/cfslam_llava_captions.json

输出:
  - sg_cache_detect/object_affordances.json
"""

import os
import io
import json
import gzip
import pickle
import base64
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conceptgraph.llava.unified_client import chat_completions


PROMPT_TEMPLATE = """你是一个场景理解专家。请分析这个物体的图像和多视角描述，生成精炼的物体信息和功能性属性。

## 多视角描述
{captions}

## 任务
基于图像和描述，输出以下JSON格式（不要输出其他内容）：

```json
{{
  "object_tag": "简洁的物体标签，如 lamp, chair, table",
  "summary": "一句话精炼描述，包含材质、颜色、形状等关键特征",
  "category": "家具/照明/装饰/电器/存储/其他",
  "affordances": {{
    "primary_functions": ["主要功能1", "主要功能2"],
    "interaction_type": "sit/lie/place/store/illuminate/decorate/other",
    "typical_location": "客厅/卧室/厨房/办公室/通用",
    "co_objects": ["经常一起出现的物体"],
    "usage_context": "使用场景描述"
  }}
}}
```

注意：
1. object_tag 应该是简洁的英文名词
2. 如果描述之间有冲突，以图像为准
3. affordances 要反映物体的实际功能用途
"""


def crop_image_with_mask(image: Image.Image, mask: np.ndarray, 
                         x1: int, y1: int, x2: int, y2: int, padding: int = 10) -> Optional[Image.Image]:
    """裁剪图像并应用mask（与步骤4相同的逻辑）"""
    image_np = np.array(image)
    
    # 边界检查
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(image_np.shape[1], int(x2) + padding)
    y2 = min(image_np.shape[0], int(y2) + padding)
    
    # 裁剪
    image_crop = image_np[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    
    # 检查尺寸
    if image_crop.shape[:2] != mask_crop.shape:
        return None
    
    # 可选：遮挡非物体区域（blackout）
    # black_image = np.zeros_like(image_crop)
    # black_image[mask_crop] = image_crop[mask_crop]
    # return Image.fromarray(black_image)
    
    return Image.fromarray(image_crop)


def get_best_object_image(obj: Dict, max_images: int = 1) -> List[Image.Image]:
    """从物体数据中获取最佳的裁剪图像"""
    images = []
    
    if "color_path" not in obj or "xyxy" not in obj:
        return images
    
    # 按置信度排序
    n_det = len(obj.get("color_path", []))
    if n_det == 0:
        return images
    
    conf = obj.get("conf", [1.0] * n_det)
    indices = np.argsort(conf)[::-1]  # 按置信度降序
    
    for idx in indices[:max_images]:
        try:
            image_path = obj["color_path"][idx]
            if not Path(image_path).exists():
                continue
            
            image = Image.open(image_path).convert("RGB")
            xyxy = obj["xyxy"][idx]
            mask = obj.get("mask", [None] * n_det)[idx]
            
            if mask is not None:
                cropped = crop_image_with_mask(image, mask, *xyxy)
            else:
                # 没有mask时直接裁剪
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cropped = image.crop((max(0, x1-10), max(0, y1-10), x2+10, y2+10))
            
            if cropped is not None:
                images.append(cropped)
        except Exception as e:
            continue
    
    return images


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_message_with_image(captions: List[str], images: List[Image.Image]) -> List[Dict]:
    """构建带图像的消息"""
    # 格式化captions
    captions_text = "\n".join([f"- 视角{i+1}: {c[:300]}..." if len(c) > 300 else f"- 视角{i+1}: {c}" 
                               for i, c in enumerate(captions[:5])])  # 最多5个视角
    
    prompt = PROMPT_TEMPLATE.format(captions=captions_text)
    
    content = [{"type": "text", "text": prompt}]
    
    # 添加图像
    for img in images[:2]:  # 最多2张图
        img_b64 = image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    
    return [{"role": "user", "content": content}]


def parse_response(response_text: str) -> Dict:
    """解析LLM响应"""
    import re
    
    # 尝试提取JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # 尝试直接解析
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {}


def process_object(obj_id: int, captions: List[str], images: List[Image.Image],
                   base_url: str, model: str) -> Dict:
    """处理单个物体"""
    messages = build_message_with_image(captions, images)
    
    try:
        response = chat_completions(
            messages=messages,
            base_url=base_url,
            model=model,
            max_tokens=1000,
            timeout=60.0
        )
        
        content = response["choices"][0]["message"]["content"]
        result = parse_response(content)
        
        if result:
            result["id"] = obj_id
            result["raw_response"] = content
            return result
    except Exception as e:
        print(f"  物体{obj_id}处理失败: {e}")
    
    # 失败时返回基本结构
    return {
        "id": obj_id,
        "object_tag": f"object_{obj_id}",
        "summary": captions[0][:200] if captions else "",
        "category": "其他",
        "affordances": {
            "primary_functions": [],
            "interaction_type": "other",
            "typical_location": "通用",
            "co_objects": [],
            "usage_context": ""
        },
        "error": True
    }


def main():
    parser = argparse.ArgumentParser(description="精炼物体描述并提取Affordance")
    parser.add_argument("--cache_dir", type=str, required=True, help="缓存目录路径")
    parser.add_argument("--pcd_file", type=str, default=None, help="pcd文件路径（用于获取物体图像）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    # 获取环境变量
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    
    if not base_url:
        raise ValueError("LLM_BASE_URL 环境变量未设置")
    if not model:
        raise ValueError("LLM_MODEL 环境变量未设置")
    
    print(f"LLM服务: {base_url}")
    print(f"模型: {model}")
    
    # 加载captions
    captions_file = cache_dir / "cfslam_llava_captions.json"
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions文件不存在: {captions_file}")
    
    with open(captions_file) as f:
        captions_data = json.load(f)
    
    print(f"物体数量: {len(captions_data)}")
    
    # 加载pcd文件获取物体图像数据
    objects = []
    if args.pcd_file and Path(args.pcd_file).exists():
        print(f"加载pcd文件: {args.pcd_file}")
        with gzip.open(args.pcd_file, 'rb') as f:
            pcd_data = pickle.load(f)
        objects = pcd_data.get('objects', [])
        print(f"从pcd加载了 {len(objects)} 个物体的图像数据")
    else:
        print("未提供pcd文件，将不使用图像")
    
    # 处理每个物体
    results = []
    for item in tqdm(captions_data, desc="处理物体"):
        obj_id = item["id"]
        captions = item.get("captions", [])
        
        # 从pcd数据获取物体图像
        images = []
        if obj_id < len(objects):
            images = get_best_object_image(objects[obj_id], max_images=1)
        
        result = process_object(obj_id, captions, images, base_url, model)
        results.append(result)
    
    # 保存结果
    output_path = Path(args.output) if args.output else cache_dir / "object_affordances.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存到: {output_path}")
    
    # 统计
    success_count = sum(1 for r in results if not r.get("error"))
    print(f"成功: {success_count}/{len(results)}")
    
    # 显示几个示例
    print("\n示例结果:")
    for r in results[:3]:
        print(f"  {r['id']}: {r.get('object_tag', 'N/A')} - {r.get('summary', 'N/A')[:50]}...")


if __name__ == "__main__":
    main()
