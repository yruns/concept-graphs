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
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conceptgraph.llava.unified_client import chat_completions


PROMPT_TEMPLATE = """You are an expert in indoor scene understanding and object affordance analysis. Analyze the target object marked with a red bounding box labeled "TARGET" in the provided images.

## Task
Examine the target object within its spatial context. Consider:
1. The object's visual attributes (shape, material, color, size)
2. Its spatial relationship with surrounding objects
3. The functional role it plays in this environment
4. How humans might interact with it

## Multi-view Descriptions (from previous visual analysis)
{captions}

## Output Format
Return ONLY a valid JSON object with the following structure:

```json
{{
  "object_tag": "<concise noun, e.g., lamp, armchair, coffee_table>",
  "summary": "<one-sentence description including material, color, style, and key features>",
  "category": "<one of: furniture, seating, lighting, storage, decoration, appliance, architectural, textile, other>",
  "affordances": {{
    "primary_functions": ["<main function 1>", "<main function 2>"],
    "interaction_type": "<primary interaction: sit, lie, place, store, illuminate, support, contain, decorate, divide, other>",
    "typical_location": "<room type: living_room, bedroom, kitchen, office, bathroom, dining_room, hallway, universal>",
    "co_objects": ["<nearby objects visible in the scene>"],
    "usage_context": "<brief description of how this object is used in this specific scene>"
  }}
}}
```

## Guidelines
- Prioritize visual evidence from images over textual descriptions when conflicts arise
- `object_tag` should be a single noun or compound noun (use underscore for multi-word tags)
- `co_objects` must list objects actually visible in the provided images
- `usage_context` should reflect the specific scene context, not generic usage
"""


def draw_bbox_on_image(image: Image.Image, x1: int, y1: int, x2: int, y2: int,
                       mask: Optional[np.ndarray] = None, 
                       color: tuple = (255, 0, 0), thickness: int = 4) -> Image.Image:
    """在完整图像上绘制标注框和可选的mask轮廓
    
    Args:
        image: 原始完整图像
        x1, y1, x2, y2: 边界框坐标
        mask: 可选的分割mask
        color: 标注颜色 (R, G, B)
        thickness: 线条粗细
    
    Returns:
        标注后的完整图像
    """
    image_np = np.array(image).copy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # 绘制边界框
    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, thickness)
    
    # 如果有mask，绘制轮廓
    # if mask is not None:
    #     try:
    #         contours, _ = cv2.findContours(
    #             mask.astype(np.uint8) * 255,
    #             cv2.RETR_EXTERNAL,
    #             cv2.CHAIN_APPROX_SIMPLE
    #         )
    #         cv2.drawContours(image_np, contours, -1, color, thickness // 2)
    #     except:
    #         pass
    
    # 添加标签背景
    label = "TARGET"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    label_thickness = 2
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
    
    # 标签位置（框上方）
    label_y = max(y1 - 10, label_h + 5)
    cv2.rectangle(image_np, (x1, label_y - label_h - 5), (x1 + label_w + 10, label_y + 5), color, -1)
    cv2.putText(image_np, label, (x1 + 5, label_y), font, font_scale, (255, 255, 255), label_thickness)
    
    return Image.fromarray(image_np)


def get_best_object_image(obj: Dict, max_images: int = 5) -> List[Image.Image]:
    """从物体数据中获取带标注框的完整图像
    
    不再裁剪，而是在完整图像上标注目标物体的位置，
    这样模型可以获得更多上下文信息和周围物体感知。
    """
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
            
            # 在完整图像上绘制标注框
            annotated = draw_bbox_on_image(image, *xyxy, mask=mask)
            images.append(annotated)
            
        except Exception as e:
            continue
    
    return images


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_message_with_image(captions: List[str], images: List[Image.Image], 
                             max_images: int = 3) -> List[Dict]:
    """Build message with images for LLM
    
    Args:
        captions: List of text descriptions
        images: List of annotated images
        max_images: Maximum number of images to include
    """
    # Format captions with view indices (match image count)
    captions_lines = []
    for i, c in enumerate(captions[:max_images]):
        caption_text = c[:400] + "..." if len(c) > 400 else c
        captions_lines.append(f"- View {i+1}: {caption_text}")
    captions_text = "\n".join(captions_lines) if captions_lines else "No previous descriptions available."
    
    prompt = PROMPT_TEMPLATE.format(captions=captions_text)
    
    content = [{"type": "text", "text": prompt}]
    
    # Add images (limited by max_images)
    for img in images[:max_images]:
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
                   base_url: str, model: str, max_images: int = 3) -> Dict:
    """Process a single object with LLM
    
    Args:
        obj_id: Object ID
        captions: Text descriptions from previous analysis
        images: Annotated images showing the object
        base_url: LLM service URL
        model: Model name
        max_images: Maximum number of images to use
    """
    messages = build_message_with_image(captions, images, max_images=max_images)
    
    try:
        response = chat_completions(
            messages=messages,
            base_url=base_url,
            model=model,
            max_tokens=1000,
            timeout=180
        )
        
        content = response["choices"][0]["message"]["content"]
        result = parse_response(content)
        
        if result:
            result["id"] = obj_id
            result["raw_response"] = content
            return result
    except Exception as e:
        print(f"  物体{obj_id}处理失败: {e}")
    
    # Fallback structure on failure
    return {
        "id": obj_id,
        "object_tag": f"object_{obj_id}",
        "summary": captions[0][:200] if captions else "",
        "category": "other",
        "affordances": {
            "primary_functions": [],
            "interaction_type": "other",
            "typical_location": "universal",
            "co_objects": [],
            "usage_context": ""
        },
        "error": True
    }


def main():
    parser = argparse.ArgumentParser(description="Refine object descriptions and extract affordances")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory path")
    parser.add_argument("--pcd_file", type=str, default=None, help="PCD file path (for object images)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--image_num", type=int, default=3, help="Number of images to use per object (default: 3)")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    image_num = args.image_num
    
    # Get environment variables
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    
    if not base_url:
        raise ValueError("LLM_BASE_URL environment variable not set")
    if not model:
        raise ValueError("LLM_MODEL environment variable not set")
    
    print(f"LLM Service: {base_url}")
    print(f"Model: {model}")
    print(f"Images per object: {image_num}")
    
    # Load captions
    captions_file = cache_dir / "cfslam_llava_captions.json"
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    with open(captions_file) as f:
        captions_data = json.load(f)
    
    print(f"Object count: {len(captions_data)}")
    
    # Load pcd file for object image data
    objects = []
    if args.pcd_file and Path(args.pcd_file).exists():
        print(f"Loading PCD file: {args.pcd_file}")
        with gzip.open(args.pcd_file, 'rb') as f:
            pcd_data = pickle.load(f)
        objects = pcd_data.get('objects', [])
        print(f"Loaded image data for {len(objects)} objects from PCD")
    else:
        print("No PCD file provided, running without images")
    
    # Process each object
    results = []
    for item in tqdm(captions_data, desc="Processing objects"):
        obj_id = item["id"]
        captions = item.get("captions", [])
        
        # Get object images from pcd data
        images = []
        if obj_id < len(objects):
            images = get_best_object_image(objects[obj_id], max_images=image_num)
        
        result = process_object(obj_id, captions, images, base_url, model, max_images=image_num)
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
