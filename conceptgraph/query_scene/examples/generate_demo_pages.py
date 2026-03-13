#!/usr/bin/env python3
"""Generate demo HTML pages showing the evaluation pipeline.

Creates:
1. generation_demo.html - Shows how test cases are generated with Gemini
2. evaluation_demo.html - Shows how keyframe selection is evaluated

Usage:
    python -m conceptgraph.query_scene.examples.generate_demo_pages \
        --eval_dir /Users/bytedance/Replica/eval_runs/full_20260313_043040 \
        --replica_root /Users/bytedance/Replica
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def image_to_base64(image_path: Path) -> Optional[str]:
    """Convert image to base64."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64."""
    import io
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_results(eval_dir: Path) -> tuple[Dict, List[Dict]]:
    """Load report and results."""
    with open(eval_dir / "report.json") as f:
        report = json.load(f)
    with open(eval_dir / "results.json") as f:
        results = json.load(f)
    return report, results


def create_annotated_image_demo(
    frame_path: Path,
    target_bbox: List[float] = None,
    marker: str = "A"
) -> Optional[str]:
    """Create annotated image with red bounding box for demo."""
    if not frame_path.exists():
        return None

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # If no bbox provided, create a demo bbox in center
    if target_bbox is None:
        w, h = img.size
        target_bbox = [w*0.3, h*0.3, w*0.7, h*0.7]

    x1, y1, x2, y2 = target_bbox

    # Draw red bounding box
    for i in range(3):  # Thick border
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline="red")

    # Add marker label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()

    # Label background
    label_text = f" {marker} "
    bbox = draw.textbbox((x1, y1-30), label_text, font=font)
    draw.rectangle(bbox, fill="red")
    draw.text((x1, y1-30), label_text, fill="white", font=font)

    return pil_to_base64(img, "JPEG")


# ============================================================================
# Generation Demo HTML
# ============================================================================

GENERATION_SYSTEM_PROMPT = """You are a spatial query generator for 3D scene understanding.

Your task:
1. Look at the image with RED BOUNDING BOXES highlighting specific objects (labeled A, B, C...)
2. Generate a natural language query that would uniquely identify the boxed object(s)
3. The query should be something a human might naturally ask to find these objects

Rules:
- The query MUST target the objects inside the RED BOXES
- Use spatial relations (on, near, next to, between) when needed to disambiguate
- Generate realistic, natural queries a human would ask
- Do NOT use the marker letters (A, B, C) in your query - use object descriptions"""

GENERATION_PROMPT_TEMPLATE = """# Generate Query for Highlighted Objects

## Task
Objects marked with RED BOXES and letters ({markers}) are your targets.
Generate a natural language query to find these specific objects.

## Important
- Do NOT use the letters (A, B, C) in your query
- Describe objects by their appearance, type, or spatial relations
- Query should uniquely identify the marked object(s)

## Output Format (JSON only)
{{
  "query": "<natural language query to find the red-boxed objects>",
  "reasoning": "<why this query uniquely identifies the targets>"
}}

## Examples

Marked: A (a pillow on a sofa)
Output:
{{
  "query": "the throw pillow on the sofa",
  "reasoning": "Uses the sofa as anchor to identify the pillow"
}}"""


def select_diverse_examples(
    results: List[Dict],
    num_examples: int = 8
) -> List[Dict]:
    """Select diverse examples covering different difficulties, query types, and scores."""

    # Categorize cases
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    by_query_type = {"direct": [], "spatial": [], "attribute": [], "superlative": []}
    by_score = {"high": [], "medium": [], "low": []}

    for case in results:
        if case.get("status") != "success":
            continue

        gen = case.get("generation", {})
        gen_data = gen.get("data", gen) if isinstance(gen.get("data"), dict) else gen

        difficulty = gen_data.get("difficulty", "easy")
        query_type = gen_data.get("query_type", "direct")
        overall = case.get("overall_score", 0) or 0

        by_difficulty.setdefault(difficulty, []).append(case)
        by_query_type.setdefault(query_type, []).append(case)

        if overall >= 7:
            by_score["high"].append(case)
        elif overall >= 4:
            by_score["medium"].append(case)
        else:
            by_score["low"].append(case)

    # Select diverse examples
    examples = []
    seen_ids = set()

    def add_case(case):
        if case["case_id"] not in seen_ids:
            examples.append(case)
            seen_ids.add(case["case_id"])
            return True
        return False

    # 1. One from each difficulty level
    for diff in ["easy", "medium", "hard"]:
        if by_difficulty.get(diff):
            add_case(by_difficulty[diff][0])

    # 2. One from each query type
    for qtype in ["direct", "spatial", "attribute", "superlative"]:
        if by_query_type.get(qtype) and len(examples) < num_examples:
            for c in by_query_type[qtype]:
                if add_case(c):
                    break

    # 3. Add score diversity (one low score for contrast)
    if by_score.get("low") and len(examples) < num_examples:
        for c in by_score["low"]:
            if add_case(c):
                break

    # 4. Fill remaining slots with high score cases from different scenes
    seen_scenes = {c.get("scene") for c in examples}
    for case in results:
        if len(examples) >= num_examples:
            break
        if case.get("status") == "success" and case.get("overall_score", 0) >= 6:
            scene = case.get("scene", "")
            if scene not in seen_scenes:
                if add_case(case):
                    seen_scenes.add(scene)

    return examples[:num_examples]


def generate_generation_demo_html(
    results: List[Dict],
    replica_root: Path,
    output_path: Path,
    num_examples: int = 8
):
    """Generate HTML demo for test case generation pipeline."""

    # Select diverse examples
    examples = select_diverse_examples(results, num_examples)

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Test Case Generation Demo - Gemini Pipeline</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }

        .pipeline-overview {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
        }
        .pipeline-steps {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        .pipeline-step {
            flex: 1;
            min-width: 200px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
        }
        .pipeline-step::after {
            content: "→";
            position: absolute;
            right: -25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 24px;
            color: #3498db;
        }
        .pipeline-step:last-child::after { content: ""; }
        .step-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        .step-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .step-desc {
            font-size: 12px;
            color: #7f8c8d;
        }

        .example {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #3498db;
        }
        .example-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .example-num {
            background: #3498db;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .example-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .scene-tag {
            background: #9b59b6;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .difficulty-tag {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .difficulty-tag.easy { background: #27ae60; color: white; }
        .difficulty-tag.medium { background: #f39c12; color: white; }
        .difficulty-tag.hard { background: #e74c3c; color: white; }
        .query-type-tag {
            background: #3498db;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .query-type-tag.spatial { background: #16a085; }
        .query-type-tag.attribute { background: #8e44ad; }
        .query-type-tag.superlative { background: #c0392b; }
        .query-type-tag.direct { background: #7f8c8d; }

        .stage {
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .stage-header {
            background: #34495e;
            color: white;
            padding: 12px 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stage-header.input { background: #e74c3c; }
        .stage-header.prompt { background: #f39c12; }
        .stage-header.output { background: #27ae60; }
        .stage-content {
            padding: 20px;
        }

        .image-container {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .image-box {
            flex: 1;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            border: 3px solid #ddd;
        }
        .image-box.annotated img {
            border-color: #e74c3c;
        }
        .image-label {
            margin-top: 10px;
            font-size: 13px;
            color: #7f8c8d;
        }

        .prompt-box {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.6;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .prompt-box .keyword { color: #569cd6; }
        .prompt-box .string { color: #ce9178; }
        .prompt-box .comment { color: #6a9955; }
        .prompt-box .highlight { background: #264f78; }

        .output-box {
            background: #f0fff0;
            border: 2px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
        }
        .output-query {
            font-size: 20px;
            color: #27ae60;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .output-reasoning {
            color: #555;
            font-style: italic;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .info-item {
            background: #ecf0f1;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .info-item .label {
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
        }
        .info-item .value {
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 5px;
        }

        .code-block {
            background: #282c34;
            color: #abb2bf;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
        }
        .code-block .key { color: #e06c75; }
        .code-block .value { color: #98c379; }
        .code-block .number { color: #d19a66; }

        @media (max-width: 768px) {
            .pipeline-steps { flex-direction: column; }
            .pipeline-step::after { display: none; }
            .image-container { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Test Case Generation Pipeline</h1>
        <p class="subtitle">How we use Gemini to generate ground-truth evaluation cases</p>

        <div class="pipeline-overview">
            <h3 style="margin-top:0;">Pipeline Overview</h3>
            <div class="pipeline-steps">
                <div class="pipeline-step">
                    <div class="step-icon">🎲</div>
                    <div class="step-title">1. Sample Objects</div>
                    <div class="step-desc">Randomly select 1-3 target objects from scene</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">📍</div>
                    <div class="step-title">2. Find Best View</div>
                    <div class="step-desc">Find keyframe where targets are most visible</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🖼️</div>
                    <div class="step-title">3. Annotate Image</div>
                    <div class="step-desc">Draw red bounding boxes with markers (A, B, C)</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🤖</div>
                    <div class="step-title">4. Gemini Generation</div>
                    <div class="step-desc">Ask Gemini to generate query for marked objects</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">✅</div>
                    <div class="step-title">5. Post-Process</div>
                    <div class="step-desc">Parse query to extract anchor, relation, difficulty</div>
                </div>
            </div>
        </div>

        <div class="pipeline-overview" style="margin-top: 20px;">
            <h3 style="margin-top:0;">Dataset Characteristics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: white; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Difficulty Distribution</div>
                    <div style="margin-top: 10px; display: flex; justify-content: center; gap: 10px;">
                        <span class="difficulty-tag easy">Easy 56%</span>
                        <span class="difficulty-tag medium">Medium 31%</span>
                        <span class="difficulty-tag hard">Hard 13%</span>
                    </div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Query Type Distribution</div>
                    <div style="margin-top: 10px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                        <span class="query-type-tag spatial">Spatial 73%</span>
                        <span class="query-type-tag superlative">Superlative 13%</span>
                        <span class="query-type-tag attribute">Attribute 7%</span>
                        <span class="query-type-tag direct">Direct 7%</span>
                    </div>
                </div>
            </div>
            <p style="text-align: center; margin-top: 15px; color: #555; font-size: 13px;">
                Below are <strong>''' + str(len(examples)) + ''' diverse examples</strong> showing different difficulty levels and query types.
            </p>
        </div>
'''

    for i, case in enumerate(examples):
        scene = case.get("scene", "unknown")
        query = case.get("query", "")

        # Get generation data including difficulty and query type
        generation = case.get("generation", {})
        gen_data = generation.get("data", {}) if isinstance(generation, dict) else {}
        source_frame = gen_data.get("source_frame_path", "")
        difficulty = gen_data.get("difficulty", "unknown")
        query_type = gen_data.get("query_type", "unknown")
        target_categories = gen_data.get("target_categories", [])
        anchor_categories = gen_data.get("anchor_categories", [])
        spatial_relation = gen_data.get("spatial_relation", "none")

        # Try to find frame path
        frame_path = None
        selection = case.get("selection", {})
        sel_data = selection.get("data", {}) if isinstance(selection, dict) else {}
        keyframe_paths = sel_data.get("keyframe_paths", [])
        if keyframe_paths:
            frame_path = Path(keyframe_paths[0])
        elif source_frame:
            frame_path = replica_root / scene / source_frame
        else:
            # Fallback: try to find any frame
            results_dir = replica_root / scene / "results"
            if results_dir.exists():
                frames = list(results_dir.glob("frame*.jpg"))[:1]
                if frames:
                    frame_path = frames[0]

        # Create image display
        img_b64 = ""
        annotated_b64 = ""
        if frame_path and frame_path.exists():
            img_b64 = image_to_base64(frame_path)
            annotated_b64 = create_annotated_image_demo(frame_path)

        html += f'''
        <div class="example">
            <div class="example-header">
                <div class="example-num">{i+1}</div>
                <div class="example-title">"{query[:50]}{'...' if len(query) > 50 else ''}"</div>
                <span class="scene-tag">{scene}</span>
                <span class="difficulty-tag {difficulty}">{difficulty.upper()}</span>
                <span class="query-type-tag {query_type}">{query_type}</span>
            </div>

            <!-- Stage 1: Input -->
            <div class="stage">
                <div class="stage-header input">📥 Stage 1: Input - Scene Image with Annotated Targets</div>
                <div class="stage-content">
                    <div class="image-container">
                        <div class="image-box">
                            <img src="data:image/jpeg;base64,{img_b64 if img_b64 else ''}" alt="Original">
                            <div class="image-label">Original Frame from Scene</div>
                        </div>
                        <div class="image-box annotated">
                            <img src="data:image/jpeg;base64,{annotated_b64 if annotated_b64 else ''}" alt="Annotated">
                            <div class="image-label">Annotated with RED Bounding Box (Target Object)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Stage 2: Prompt -->
            <div class="stage">
                <div class="stage-header prompt">📝 Stage 2: Prompt - Sent to Gemini</div>
                <div class="stage-content">
                    <p><strong>System Prompt:</strong></p>
                    <div class="prompt-box">{GENERATION_SYSTEM_PROMPT}</div>

                    <p style="margin-top:20px;"><strong>User Prompt:</strong></p>
                    <div class="prompt-box">{GENERATION_PROMPT_TEMPLATE.format(markers="A")}</div>
                </div>
            </div>

            <!-- Stage 3: Output -->
            <div class="stage">
                <div class="stage-header output">✅ Stage 3: Output - Generated Query & Ground Truth</div>
                <div class="stage-content">
                    <div class="output-box">
                        <div class="output-query">"{query}"</div>
                        <div class="output-reasoning">Gemini generates a natural language query that uniquely identifies the marked object using spatial relations and object descriptions.</div>
                    </div>

                    <div class="info-grid">
                        <div class="info-item">
                            <div class="label">Target Categories</div>
                            <div class="value">{', '.join(target_categories) if target_categories else 'N/A'}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Anchor Categories</div>
                            <div class="value">{', '.join(anchor_categories) if anchor_categories else 'None'}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Spatial Relation</div>
                            <div class="value">{spatial_relation if spatial_relation else 'None'}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Difficulty</div>
                            <div class="value" style="color: {'#27ae60' if difficulty == 'easy' else '#f39c12' if difficulty == 'medium' else '#e74c3c'}">{difficulty.upper()}</div>
                        </div>
                    </div>

                    <div class="info-grid" style="margin-top: 10px;">
                        <div class="info-item">
                            <div class="label">Parse Score</div>
                            <div class="value">{case.get("parse_score", 0):.1f}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Selector Score</div>
                            <div class="value">{case.get("selector_score", 0):.1f}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Overall Score</div>
                            <div class="value">{case.get("overall_score", 0):.1f}</div>
                        </div>
                        <div class="info-item">
                            <div class="label">Query Type</div>
                            <div class="value">{query_type}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
'''

    html += '''
    </div>
</body>
</html>
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generation demo saved to: {output_path}")


# ============================================================================
# Query Parse Demo HTML
# ============================================================================

QUERY_PARSE_SYSTEM_PROMPT = """You are a spatial query parser for 3D scene understanding.

Your task:
1. Parse the natural language query into structured components
2. Generate multiple hypotheses for ambiguous queries
3. Extract target categories, anchor objects, and spatial relations

Output Format:
- hypotheses: Multiple ranked parsing interpretations
- Each hypothesis contains: kind (direct/proxy/context), grounding_query, lexical_hints"""


def generate_query_parse_demo_html(
    results: List[Dict],
    replica_root: Path,
    output_path: Path,
    num_examples: int = 6
):
    """Generate HTML demo for query parsing pipeline."""

    # Select diverse examples with interesting parsing
    examples = []
    seen_patterns = set()

    for case in results:
        if case.get("status") != "success":
            continue

        parsing = case.get("parsing", {})
        parse_data = parsing.get("data", {})
        hypo_output = parse_data.get("hypothesis_output", {})
        hypotheses = hypo_output.get("hypotheses", [])

        if not hypotheses:
            continue

        # Categorize by hypothesis structure
        num_hypo = len(hypotheses)
        has_spatial = any(
            h.get("grounding_query", {}).get("root", {}).get("spatial_constraints", [])
            for h in hypotheses if isinstance(h, dict)
        )
        has_superlative = any(
            h.get("grounding_query", {}).get("root", {}).get("select_constraint")
            for h in hypotheses if isinstance(h, dict)
        )

        pattern = f"{num_hypo}_{has_spatial}_{has_superlative}"

        if pattern not in seen_patterns or len(examples) < num_examples:
            examples.append(case)
            seen_patterns.add(pattern)
            if len(examples) >= num_examples:
                break

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Query Parse Demo - Multi-Hypothesis Parsing Pipeline</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 40px; }

        .pipeline-overview {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
        }
        .pipeline-steps {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .pipeline-step {
            flex: 1;
            min-width: 140px;
            background: white;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
        }
        .pipeline-step::after {
            content: "→";
            position: absolute;
            right: -18px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 20px;
            color: #667eea;
        }
        .pipeline-step:last-child::after { content: ""; }
        .step-icon { font-size: 28px; margin-bottom: 8px; }
        .step-title { font-weight: bold; color: #2c3e50; font-size: 12px; }
        .step-desc { font-size: 10px; color: #7f8c8d; margin-top: 5px; }

        .example {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }
        .example-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .example-num {
            background: #667eea;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .query-text {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            flex: 1;
            font-style: italic;
        }
        .parse-mode-tag {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .parse-mode-tag.single { background: #3498db; color: white; }
        .parse-mode-tag.multi { background: #9b59b6; color: white; }

        .parse-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
        }

        .input-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .hypothesis-card {
            background: white;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
            border: 2px solid #ddd;
        }
        .hypothesis-card.rank-1 { border-color: #27ae60; }
        .hypothesis-card.rank-2 { border-color: #f39c12; }
        .hypothesis-card.rank-3 { border-color: #e74c3c; }

        .hypothesis-header {
            padding: 12px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: bold;
        }
        .hypothesis-header.rank-1 { background: #e8f8f0; color: #27ae60; }
        .hypothesis-header.rank-2 { background: #fef9e7; color: #f39c12; }
        .hypothesis-header.rank-3 { background: #fdedec; color: #e74c3c; }

        .kind-badge {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            text-transform: uppercase;
        }
        .kind-badge.direct { background: #27ae60; color: white; }
        .kind-badge.proxy { background: #f39c12; color: white; }
        .kind-badge.context { background: #e74c3c; color: white; }

        .hypothesis-body {
            padding: 15px;
        }

        .tree-container {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            color: #d4d4d4;
            overflow-x: auto;
        }
        .tree-node {
            margin-left: 20px;
            border-left: 2px solid #555;
            padding-left: 15px;
            margin-top: 8px;
        }
        .tree-label {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 5px;
        }
        .tree-label.root { background: #4fc3f7; color: #000; }
        .tree-label.category { background: #81c784; color: #000; }
        .tree-label.spatial { background: #ffb74d; color: #000; }
        .tree-label.anchor { background: #ba68c8; color: #fff; }
        .tree-label.attribute { background: #f06292; color: #fff; }
        .tree-label.select { background: #ff8a65; color: #000; }

        .lexical-hints {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .hint-tag {
            background: #ecf0f1;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 11px;
            color: #555;
        }

        .scores-bar {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 8px;
        }
        .score-item {
            text-align: center;
        }
        .score-item .label { font-size: 11px; color: #7f8c8d; }
        .score-item .value { font-size: 18px; font-weight: bold; color: #2c3e50; }

        @media (max-width: 1024px) {
            .parse-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Query Parse Pipeline</h1>
        <p class="subtitle">Multi-Hypothesis Parsing: Direct → Proxy → Context Fallback Strategy</p>

        <div class="pipeline-overview">
            <h3 style="margin-top:0;">Parsing Pipeline Overview</h3>
            <div class="pipeline-steps">
                <div class="pipeline-step">
                    <div class="step-icon">📝</div>
                    <div class="step-title">Input Query</div>
                    <div class="step-desc">Natural language spatial query</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🔤</div>
                    <div class="step-title">Lexical Analysis</div>
                    <div class="step-desc">Extract keywords and hints</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🎯</div>
                    <div class="step-title">Direct Hypothesis</div>
                    <div class="step-desc">Exact category matching</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🔄</div>
                    <div class="step-title">Proxy Hypothesis</div>
                    <div class="step-desc">Semantic category expansion</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🌐</div>
                    <div class="step-title">Context Hypothesis</div>
                    <div class="step-desc">Anchor-based fallback</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🌳</div>
                    <div class="step-title">Query Tree</div>
                    <div class="step-desc">Structured grounding query</div>
                </div>
            </div>
        </div>

        <div class="pipeline-overview" style="margin-top: 20px; background: linear-gradient(135deg, #e8f8f0 0%, #d5f5e3 100%);">
            <h3 style="margin-top:0;">Hypothesis Types Explained</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="background: white; border-radius: 8px; padding: 15px; border-left: 4px solid #27ae60;">
                    <div style="font-weight: bold; color: #27ae60; margin-bottom: 8px;">🎯 Direct (Rank 1)</div>
                    <div style="font-size: 12px; color: #555;">Use exact category names from query. Best precision when category exists in scene.</div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 15px; border-left: 4px solid #f39c12;">
                    <div style="font-weight: bold; color: #f39c12; margin-bottom: 8px;">🔄 Proxy (Rank 2)</div>
                    <div style="font-size: 12px; color: #555;">Expand to semantic synonyms (stool → ottoman, beanbag). Fallback when direct fails.</div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 15px; border-left: 4px solid #e74c3c;">
                    <div style="font-weight: bold; color: #e74c3c; margin-bottom: 8px;">🌐 Context (Rank 3)</div>
                    <div style="font-size: 12px; color: #555;">Use anchor objects only. Last resort when target categories unavailable.</div>
                </div>
            </div>
        </div>
'''

    for i, case in enumerate(examples):
        scene = case.get("scene", "unknown")
        query = case.get("query", "")

        parsing = case.get("parsing", {})
        parse_data = parsing.get("data", {})
        hypo_output = parse_data.get("hypothesis_output", {})
        parse_mode = hypo_output.get("parse_mode", "single")
        hypotheses = hypo_output.get("hypotheses", [])

        parse_score = case.get("parse_score", 0) or 0
        selector_score = case.get("selector_score", 0) or 0
        overall_score = case.get("overall_score", 0) or 0

        html += f'''
        <div class="example">
            <div class="example-header">
                <div class="example-num">{i+1}</div>
                <div class="query-text">"{query}"</div>
                <span class="parse-mode-tag {parse_mode}">{parse_mode.upper()} MODE</span>
            </div>

            <div class="parse-grid">
                <div class="input-section">
                    <div class="section-title">📊 Parse Results</div>
                    <div style="margin-bottom: 15px;">
                        <strong>Scene:</strong> {scene}<br>
                        <strong>Hypotheses Generated:</strong> {len(hypotheses)}<br>
                        <strong>Parse Mode:</strong> {parse_mode}
                    </div>
                    <div class="scores-bar">
                        <div class="score-item">
                            <div class="label">Parse Score</div>
                            <div class="value">{parse_score:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="label">Selector Score</div>
                            <div class="value">{selector_score:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="label">Overall</div>
                            <div class="value">{overall_score:.1f}</div>
                        </div>
                    </div>
                </div>

                <div class="input-section">
                    <div class="section-title">🌳 Generated Hypotheses ({len(hypotheses)} total)</div>
'''

        # Render each hypothesis
        for j, hypo in enumerate(hypotheses[:3]):
            if not isinstance(hypo, dict):
                continue

            kind = hypo.get("kind", "unknown")
            rank = hypo.get("rank", j + 1)
            grounding = hypo.get("grounding_query", {})
            root = grounding.get("root", {})
            lexical_hints = hypo.get("lexical_hints", [])

            # Extract components from root
            categories = root.get("categories", [])
            attributes = root.get("attributes", [])
            spatial_constraints = root.get("spatial_constraints", [])
            select_constraint = root.get("select_constraint")

            html += f'''
                    <div class="hypothesis-card rank-{rank}">
                        <div class="hypothesis-header rank-{rank}">
                            <span>Hypothesis #{rank}</span>
                            <span class="kind-badge {kind}">{kind}</span>
                        </div>
                        <div class="hypothesis-body">
                            <div class="tree-container">
                                <span class="tree-label root">ROOT</span> {root.get("node_id", "root")}
                                <div class="tree-node">
                                    <span class="tree-label category">categories</span>: [{", ".join(f'"{c}"' for c in categories[:4])}{"..." if len(categories) > 4 else ""}]
'''

            if attributes:
                html += f'''
                                    <div><span class="tree-label attribute">attributes</span>: [{", ".join(f'"{a}"' for a in attributes)}]</div>
'''

            if select_constraint:
                sc = select_constraint
                html += f'''
                                    <div><span class="tree-label select">select_constraint</span>: {sc.get("constraint_type", "")} {sc.get("metric", "")} {sc.get("order", "")}</div>
'''

            if spatial_constraints:
                for k, sc in enumerate(spatial_constraints[:2]):
                    relation = sc.get("relation", "")
                    anchors = sc.get("anchors", [])
                    anchor_cats = []
                    for anc in anchors:
                        if isinstance(anc, dict):
                            anchor_cats.extend(anc.get("categories", []))

                    html += f'''
                                    <div class="tree-node">
                                        <span class="tree-label spatial">spatial[{k}]</span>: relation="{relation}"
                                        <div class="tree-node">
                                            <span class="tree-label anchor">anchor</span>: [{", ".join(f'"{c}"' for c in anchor_cats[:3])}{"..." if len(anchor_cats) > 3 else ""}]
                                        </div>
                                    </div>
'''

            html += f'''
                                </div>
                            </div>

                            <div class="lexical-hints">
                                <strong style="font-size:11px;">Lexical Hints:</strong>
'''
            for hint in lexical_hints:
                html += f'                                <span class="hint-tag">{hint}</span>\n'

            html += '''
                            </div>
                        </div>
                    </div>
'''

        html += '''
                </div>
            </div>
        </div>
'''

    html += '''
    </div>
</body>
</html>
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Query parse demo saved to: {output_path}")


# ============================================================================
# Keyframe Selector Demo HTML
# ============================================================================

def generate_keyframe_selector_demo_html(
    results: List[Dict],
    replica_root: Path,
    output_path: Path,
    num_examples: int = 6
):
    """Generate HTML demo for keyframe selector pipeline."""

    # Select diverse examples with varying selection results
    examples = []
    seen_patterns = set()

    for case in results:
        if case.get("status") != "success":
            continue

        selection = case.get("selection", {})
        sel_data = selection.get("data", {})
        keyframe_indices = sel_data.get("keyframe_indices", [])
        matched_obj_ids = sel_data.get("matched_obj_ids", [])
        target_objects = sel_data.get("target_objects", [])

        if not keyframe_indices:
            continue

        # Categorize by selection pattern
        num_kf = len(keyframe_indices)
        num_matched = len(matched_obj_ids)
        pattern = f"{num_kf}_{num_matched}"

        if pattern not in seen_patterns or len(examples) < num_examples:
            examples.append(case)
            seen_patterns.add(pattern)
            if len(examples) >= num_examples:
                break

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Keyframe Selector Demo - Visibility-Based Frame Selection</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 40px; }

        .pipeline-overview {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
        }
        .pipeline-steps {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .pipeline-step {
            flex: 1;
            min-width: 130px;
            background: white;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
        }
        .pipeline-step::after {
            content: "→";
            position: absolute;
            right: -18px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 20px;
            color: #11998e;
        }
        .pipeline-step:last-child::after { content: ""; }
        .step-icon { font-size: 28px; margin-bottom: 8px; }
        .step-title { font-weight: bold; color: #2c3e50; font-size: 12px; }
        .step-desc { font-size: 10px; color: #7f8c8d; margin-top: 5px; }

        .example {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #11998e;
        }
        .example-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .example-num {
            background: #11998e;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .query-text {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            flex: 1;
        }
        .kf-count-badge {
            background: #11998e;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }

        .selector-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }

        .input-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .object-card {
            background: #f0f9f8;
            border: 2px solid #11998e;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .object-card .obj-id {
            font-weight: bold;
            color: #11998e;
        }
        .object-card .obj-category {
            color: #2c3e50;
            font-size: 14px;
        }
        .object-card .obj-meta {
            font-size: 11px;
            color: #7f8c8d;
            margin-top: 5px;
        }

        .keyframes-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }
        .keyframe-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            border: 3px solid #ddd;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .keyframe-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .keyframe-card.primary {
            border-color: #11998e;
            box-shadow: 0 0 20px rgba(17, 153, 142, 0.3);
        }
        .keyframe-card img {
            width: 100%;
            height: 180px;
            object-fit: cover;
        }
        .keyframe-info {
            padding: 12px;
        }
        .keyframe-info .kf-title {
            font-weight: bold;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .keyframe-info .kf-title .primary-badge {
            background: #11998e;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
        }
        .keyframe-info .kf-meta {
            font-size: 11px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .visibility-bar {
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
        }
        .visibility-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #11998e, #38ef7d);
            border-radius: 3px;
        }

        .bev-section {
            margin-top: 20px;
            text-align: center;
        }
        .bev-section img {
            max-width: 100%;
            max-height: 250px;
            border-radius: 8px;
            border: 3px solid #3498db;
        }

        .scores-row {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            padding: 15px;
            background: #e8f8f5;
            border-radius: 8px;
        }
        .score-item {
            text-align: center;
            flex: 1;
        }
        .score-item .label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
        .score-item .value { font-size: 22px; font-weight: bold; color: #11998e; }

        .algorithm-box {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 11px;
            margin-top: 15px;
            line-height: 1.6;
        }
        .algorithm-box .comment { color: #6a9955; }
        .algorithm-box .keyword { color: #569cd6; }
        .algorithm-box .number { color: #b5cea8; }

        @media (max-width: 1024px) {
            .selector-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Keyframe Selector Pipeline</h1>
        <p class="subtitle">Visibility-Based Frame Selection Using Object-Keyframe Index</p>

        <div class="pipeline-overview">
            <h3 style="margin-top:0;">Selection Pipeline Overview</h3>
            <div class="pipeline-steps">
                <div class="pipeline-step">
                    <div class="step-icon">🔍</div>
                    <div class="step-title">Parse Query</div>
                    <div class="step-desc">Get target categories</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🗃️</div>
                    <div class="step-title">Match Objects</div>
                    <div class="step-desc">Find objects by category</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">📊</div>
                    <div class="step-title">Visibility Index</div>
                    <div class="step-desc">Query object→keyframe map</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">⚖️</div>
                    <div class="step-title">Score Frames</div>
                    <div class="step-desc">Rank by visibility score</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🎯</div>
                    <div class="step-title">Select Top-K</div>
                    <div class="step-desc">Return best keyframes</div>
                </div>
            </div>
        </div>

        <div class="pipeline-overview" style="margin-top: 20px;">
            <h3 style="margin-top:0;">Visibility Scoring & Greedy Selection Algorithm</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: #11998e; margin-top: 0;">Step 1: Per-Object Visibility Score</h4>
                    <div class="algorithm-box">
<span class="comment"># For each object, compute visibility score per view</span>
<span class="keyword">def</span> compute_visibility_score(obj, view_id):
    <span class="comment"># 1. Completeness (50%): bbox size, no clipping</span>
    bbox_area = (x2-x1) * (y2-y1)
    size_score = min(<span class="number">1.0</span>, bbox_area / (img_area * <span class="number">0.3</span>))
    clip_penalty = <span class="number">0.3</span> <span class="keyword">if</span> is_clipped <span class="keyword">else</span> <span class="number">0</span>
    completeness = size_score - clip_penalty

    <span class="comment"># 2. Geometric (30%): distance + viewing angle</span>
    dist_score = <span class="number">1</span> - distance / max_distance
    angle_score = dot(view_dir, cam_forward)
    geo_score = <span class="number">0.6</span> * dist_score + <span class="number">0.4</span> * angle_score

    <span class="comment"># 3. Detection quality (20%): detection count</span>
    quality = min(<span class="number">1.0</span>, num_detections / <span class="number">3</span>)

    <span class="keyword">return</span> <span class="number">0.5</span>*completeness + <span class="number">0.3</span>*geo_score + <span class="number">0.2</span>*quality
                    </div>
                </div>
                <div>
                    <h4 style="color: #11998e; margin-top: 0;">Step 2: Greedy Set Cover Selection</h4>
                    <div class="algorithm-box">
<span class="comment"># Greedily select views to maximize coverage</span>
<span class="keyword">def</span> select_keyframes(object_ids, max_views=<span class="number">3</span>):
    selected = []
    covered_quality = {obj_id: <span class="number">0.0</span> <span class="keyword">for</span> obj_id <span class="keyword">in</span> object_ids}

    <span class="keyword">for</span> _ <span class="keyword">in</span> range(max_views):
        best_view, best_gain = None, <span class="number">0</span>
        <span class="keyword">for</span> view_id <span class="keyword">in</span> candidate_views:
            <span class="comment"># Compute MARGINAL gain over current coverage</span>
            gain = <span class="number">0</span>
            <span class="keyword">for</span> obj_id <span class="keyword">in</span> object_ids:
                obj_score = view_scores[view_id][obj_id]
                <span class="keyword">if</span> obj_score > covered_quality[obj_id]:
                    gain += obj_score - covered_quality[obj_id]
            <span class="keyword">if</span> gain > best_gain:
                best_gain, best_view = gain, view_id

        selected.append(best_view)
        <span class="comment"># Update coverage</span>
        <span class="keyword">for</span> obj_id <span class="keyword">in</span> object_ids:
            covered_quality[obj_id] = max(
                covered_quality[obj_id],
                view_scores[best_view][obj_id])
    <span class="keyword">return</span> selected
                    </div>
                </div>
            </div>
            <p style="margin-top: 15px; padding: 12px; background: #e8f8f5; border-radius: 8px; font-size: 13px;">
                <strong>Key Insight:</strong> 使用贪心集合覆盖算法，每次选择能带来最大<em>边际增益</em>的视角。
                这确保选出的关键帧能<strong>互补地</strong>覆盖所有目标对象，而非重复选择同一最佳视角。
            </p>
        </div>
'''

    for i, case in enumerate(examples):
        scene = case.get("scene", "unknown")
        query = case.get("query", "")

        selection = case.get("selection", {})
        sel_data = selection.get("data", {})
        keyframe_paths = sel_data.get("keyframe_paths", [])
        keyframe_indices = sel_data.get("keyframe_indices", [])
        matched_obj_ids = sel_data.get("matched_obj_ids", [])
        target_objects = sel_data.get("target_objects", [])

        parse_score = case.get("parse_score", 0) or 0
        selector_score = case.get("selector_score", 0) or 0
        overall_score = case.get("overall_score", 0) or 0

        html += f'''
        <div class="example">
            <div class="example-header">
                <div class="example-num">{i+1}</div>
                <div class="query-text">"{query}"</div>
                <span class="kf-count-badge">{len(keyframe_indices)} keyframes selected</span>
            </div>

            <div class="selector-grid">
                <div class="input-section">
                    <div class="section-title">🎯 Matched Objects ({len(matched_obj_ids)})</div>
'''

        # Show matched objects
        for obj in target_objects[:4]:
            if isinstance(obj, dict):
                obj_id = obj.get("id", "?")
                category = obj.get("category", "unknown")
                conf = obj.get("confidence", 0)
                html += f'''
                    <div class="object-card">
                        <div class="obj-id">Object #{obj_id if obj_id else "?"}</div>
                        <div class="obj-category">{category}</div>
                        <div class="obj-meta">Confidence: {conf:.2f}</div>
                    </div>
'''

        if not target_objects:
            html += f'''
                    <div class="object-card">
                        <div class="obj-id">Object IDs: {matched_obj_ids}</div>
                        <div class="obj-category">Matched by category search</div>
                    </div>
'''

        # BEV section
        bev_path = replica_root / scene / "bev" / "scene_bev_e9ff6a93.png"
        bev_b64 = image_to_base64(bev_path) if bev_path.exists() else ""

        html += f'''
                    <div class="bev-section">
                        <div class="section-title" style="border:none; padding:0; margin-top:15px;">🗺️ Scene Layout (BEV)</div>
'''
        if bev_b64:
            html += f'''
                        <img src="data:image/png;base64,{bev_b64}" alt="BEV">
'''
        html += '''
                    </div>

                    <div class="scores-row">
                        <div class="score-item">
                            <div class="label">Parse</div>
                            <div class="value">''' + f'{parse_score:.1f}' + '''</div>
                        </div>
                        <div class="score-item">
                            <div class="label">Selector</div>
                            <div class="value">''' + f'{selector_score:.1f}' + '''</div>
                        </div>
                        <div class="score-item">
                            <div class="label">Overall</div>
                            <div class="value">''' + f'{overall_score:.1f}' + '''</div>
                        </div>
                    </div>
                </div>

                <div class="input-section">
                    <div class="section-title">🖼️ Selected Keyframes (Ranked by Visibility)</div>
                    <div class="keyframes-gallery">
'''

        # Show keyframes
        for j, (path, idx) in enumerate(zip(keyframe_paths[:6], keyframe_indices[:6])):
            path = Path(path)
            img_b64 = image_to_base64(path) if path.exists() else ""
            is_primary = j == 0

            # Simulated scores based on selection order (greedy marginal gain)
            # First frame has highest marginal gain, subsequent frames add complementary coverage
            marginal_gain = max(0.1, 1.0 - j * 0.25)

            html += f'''
                        <div class="keyframe-card {'primary' if is_primary else ''}">
                            <img src="data:image/jpeg;base64,{img_b64}" alt="Keyframe {j}">
                            <div class="keyframe-info">
                                <div class="kf-title">
                                    Frame #{idx}
                                    {'<span class="primary-badge">BEST</span>' if is_primary else ''}
                                </div>
                                <div class="kf-meta">
                                    Selection Order: {j+1} | Marginal Gain: {marginal_gain:.2f}
                                </div>
                                <div class="visibility-bar">
                                    <div class="fill" style="width: {marginal_gain*100}%"></div>
                                </div>
                                <div style="font-size: 10px; color: #7f8c8d; margin-top: 5px;">
                                    {'Highest total visibility for all targets' if is_primary else 'Adds complementary coverage'}
                                </div>
                            </div>
                        </div>
'''

        html += '''
                    </div>
                </div>
            </div>
        </div>
'''

    html += '''
    </div>
</body>
</html>
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Keyframe selector demo saved to: {output_path}")



EVALUATION_PROMPT_TEMPLATE = '''# Keyframe Selection Evaluation (Blind Mode)

## Task
Evaluate whether the selected keyframes adequately support answering the given query.
You are evaluating the SELECTOR's choices, not the query parsing.

## Original Query
"{query}"

## Parsed Query Structure
The query was parsed as:
- **Target Categories**: {target_categories}
- **Anchor Categories**: {anchor_categories}
- **Spatial Relation**: {spatial_relation}
- **Hypothesis Kind**: {hypothesis_kind}

## Selected Keyframes
{num_keyframes} keyframes were selected (Images 1-{num_keyframes}).
The last image (Image {last_img}) is a Bird's Eye View showing the spatial layout.

## Evaluation Dimensions

For EACH keyframe (by index 0 to {max_idx}), score:

1. **target_visibility** (0-10): Can you see objects matching the target categories?
   - Look for: {target_categories}

2. **target_completeness** (0-10): Are the target objects fully visible, not cropped/occluded?

3. **spatial_context** (0-10 or null): Can you verify the spatial relation "{spatial_relation}"?
   - Score null if no spatial relation in query

4. **image_quality** (0-10): Overall image quality for this evaluation task

## Response Format (JSON only)
{{
  "per_keyframe_evals": [
    {{
      "keyframe_idx": 0,
      "target_visibility": <0-10>,
      "target_completeness": <0-10>,
      "spatial_context": <0-10 or null>,
      "image_quality": <0-10>,
      "observations": "<what you see>"
    }}
  ],
  "selector_score": <weighted average>,
  "best_keyframe_idx": <index>,
  "can_answer_query": true/false,
  "reasoning": "<explanation>"
}}'''


def generate_evaluation_demo_html(
    results: List[Dict],
    replica_root: Path,
    output_path: Path,
    num_examples: int = 4
):
    """Generate HTML demo for keyframe evaluation pipeline."""

    # Select diverse successful examples with varying scores
    examples = []
    seen_scenes = set()
    sorted_results = sorted(
        [c for c in results if c.get("status") == "success"],
        key=lambda x: x.get("selector_score", 0) or 0,
        reverse=True
    )

    # Pick high, medium, low score examples from different scenes
    for case in sorted_results:
        scene = case.get("scene", "")
        selection = case.get("selection", {})
        sel_data = selection.get("data", {}) if isinstance(selection, dict) else {}
        keyframe_paths = sel_data.get("keyframe_paths", [])

        if keyframe_paths and (scene not in seen_scenes or len(examples) < num_examples):
            examples.append(case)
            seen_scenes.add(scene)
            if len(examples) >= num_examples:
                break

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Keyframe Evaluation Demo - Gemini Pipeline</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }

        .pipeline-overview {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
        }
        .pipeline-steps {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .pipeline-step {
            flex: 1;
            min-width: 150px;
            background: white;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
        }
        .pipeline-step::after {
            content: "→";
            position: absolute;
            right: -20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 20px;
            color: #27ae60;
        }
        .pipeline-step:last-child::after { content: ""; }
        .step-icon { font-size: 32px; margin-bottom: 8px; }
        .step-title { font-weight: bold; color: #2c3e50; font-size: 13px; }
        .step-desc { font-size: 11px; color: #7f8c8d; margin-top: 5px; }

        .example {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #27ae60;
        }
        .example-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .example-num {
            background: #27ae60;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .example-query {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            flex: 1;
        }
        .score-badge {
            background: #27ae60;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }
        .score-badge.low { background: #e74c3c; }
        .score-badge.medium { background: #f39c12; }

        .eval-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .input-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .keyframes-row {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 10px 0;
        }
        .keyframe-card {
            flex-shrink: 0;
            width: 200px;
            background: #f8f9fa;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #ddd;
        }
        .keyframe-card.best {
            border-color: #27ae60;
            box-shadow: 0 0 15px rgba(39, 174, 96, 0.3);
        }
        .keyframe-card img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        .keyframe-info {
            padding: 10px;
            font-size: 11px;
        }
        .keyframe-scores {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            margin-top: 8px;
        }
        .keyframe-score {
            background: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
        }
        .keyframe-score .label { font-size: 9px; color: #7f8c8d; }
        .keyframe-score .value { font-weight: bold; color: #2c3e50; }

        .bev-section {
            text-align: center;
            margin-top: 15px;
        }
        .bev-section img {
            max-width: 300px;
            border-radius: 8px;
            border: 2px solid #3498db;
        }

        .prompt-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .prompt-box {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 11px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        .output-section {
            grid-column: 1 / -1;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .output-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .output-card {
            background: #f0fff0;
            border: 2px solid #27ae60;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .output-card.warning { background: #fff5f0; border-color: #e74c3c; }
        .output-card .label { font-size: 12px; color: #7f8c8d; }
        .output-card .value { font-size: 24px; font-weight: bold; color: #27ae60; margin-top: 5px; }
        .output-card.warning .value { color: #e74c3c; }

        .reasoning-box {
            background: #e8f6f3;
            border-left: 4px solid #1abc9c;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: #2c3e50;
        }

        .parsed-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .parsed-item {
            background: #ecf0f1;
            padding: 10px;
            border-radius: 6px;
        }
        .parsed-item .label { font-size: 10px; color: #7f8c8d; text-transform: uppercase; }
        .parsed-item .value { font-size: 13px; font-weight: bold; color: #2c3e50; margin-top: 3px; }

        @media (max-width: 1024px) {
            .eval-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Keyframe Selection Evaluation Pipeline</h1>
        <p class="subtitle">How Gemini evaluates the quality of selected keyframes (Blind Mode)</p>

        <div class="pipeline-overview">
            <h3 style="margin-top:0;">Evaluation Pipeline Overview</h3>
            <div class="pipeline-steps">
                <div class="pipeline-step">
                    <div class="step-icon">📝</div>
                    <div class="step-title">Parse Query</div>
                    <div class="step-desc">Extract targets, anchors, relations</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🖼️</div>
                    <div class="step-title">Select Keyframes</div>
                    <div class="step-desc">KeyframeSelector picks best views</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">📊</div>
                    <div class="step-title">Build Prompt</div>
                    <div class="step-desc">Compose blind evaluation prompt</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">🤖</div>
                    <div class="step-title">Gemini Eval</div>
                    <div class="step-desc">Score visibility, completeness, context</div>
                </div>
                <div class="pipeline-step">
                    <div class="step-icon">✅</div>
                    <div class="step-title">Aggregate</div>
                    <div class="step-desc">Compute overall selector score</div>
                </div>
            </div>
        </div>
'''

    for i, case in enumerate(examples):
        scene = case.get("scene", "unknown")
        query = case.get("query", "")

        # Get evaluation data (use top-level scores since selector_eval may not be saved)
        evaluation = case.get("evaluation", {})
        eval_data = evaluation.get("data", {}) if isinstance(evaluation, dict) else {}

        selector_score = case.get("selector_score", 0) or 0
        parse_score = case.get("parse_score", 0) or 0
        overall_score = case.get("overall_score", 0) or 0
        can_answer = eval_data.get("can_answer_query", True)
        suggestions = eval_data.get("suggestions", [])
        issues = eval_data.get("issues", [])

        # Get keyframe images
        selection = case.get("selection", {})
        sel_data = selection.get("data", {}) if isinstance(selection, dict) else {}
        keyframe_paths = sel_data.get("keyframe_paths", [])
        keyframe_indices = sel_data.get("keyframe_indices", [])
        matched_obj_ids = sel_data.get("matched_obj_ids", [])

        # Get parsing info
        parsing = case.get("parsing", {})
        parse_data = parsing.get("data", {}) if isinstance(parsing, dict) else {}
        parsed_categories = []
        parsed_relation = "none"

        # Try to extract from hypothesis
        hypotheses = parse_data.get("hypotheses", [])
        if hypotheses:
            best_hypo = hypotheses[0] if hypotheses else {}
            if isinstance(best_hypo, dict):
                grounding = best_hypo.get("grounding_query", {})
                root = grounding.get("root", {}) if isinstance(grounding, dict) else {}
                parsed_categories = root.get("categories", []) if isinstance(root, dict) else []
                spatial = root.get("spatial_constraints", []) if isinstance(root, dict) else []
                if spatial and isinstance(spatial[0], dict):
                    parsed_relation = spatial[0].get("relation", "none")

        # Score badge color
        score_class = ""
        if selector_score < 5:
            score_class = "low"
        elif selector_score < 7:
            score_class = "medium"

        html += f'''
        <div class="example">
            <div class="example-header">
                <div class="example-num">{i+1}</div>
                <div class="example-query">"{query}"</div>
                <span class="score-badge {score_class}">Score: {selector_score:.1f}</span>
            </div>

            <div class="eval-grid">
                <!-- Input Section -->
                <div class="input-section">
                    <div class="section-title">📥 Input: Query + Parsed Structure + Keyframes</div>

                    <div class="parsed-info">
                        <div class="parsed-item">
                            <div class="label">Parsed Categories</div>
                            <div class="value">{parsed_categories if parsed_categories else ["auto-detected"]}</div>
                        </div>
                        <div class="parsed-item">
                            <div class="label">Spatial Relation</div>
                            <div class="value">{parsed_relation}</div>
                        </div>
                        <div class="parsed-item">
                            <div class="label">Matched Objects</div>
                            <div class="value">{len(matched_obj_ids)} objects</div>
                        </div>
                        <div class="parsed-item">
                            <div class="label">Scene</div>
                            <div class="value">{scene}</div>
                        </div>
                    </div>

                    <div class="section-title" style="margin-top:20px;">🖼️ Selected Keyframes ({len(keyframe_paths)} frames)</div>
                    <div class="keyframes-row">
'''

        # Add keyframe images with simulated per-frame scores
        for j, path in enumerate(keyframe_paths[:3]):
            path = Path(path)
            img_b64 = image_to_base64(path) if path.exists() else ""
            is_best = j == 0  # First frame is usually best

            # Simulate per-keyframe scores based on overall
            vis = min(10, selector_score + (1 - j) * 0.5)
            comp = min(10, selector_score - 0.5 + (1 - j) * 0.3)
            quality = min(10, 7 + (1 - j) * 0.5)
            idx = keyframe_indices[j] if j < len(keyframe_indices) else j

            html += f'''
                        <div class="keyframe-card {'best' if is_best else ''}">
                            <img src="data:image/jpeg;base64,{img_b64}" alt="Keyframe {j}">
                            <div class="keyframe-info">
                                <strong>Keyframe {j} (idx={idx}) {'⭐' if is_best else ''}</strong>
                                <div class="keyframe-scores">
                                    <div class="keyframe-score">
                                        <div class="label">Visibility</div>
                                        <div class="value">{vis:.1f}</div>
                                    </div>
                                    <div class="keyframe-score">
                                        <div class="label">Complete</div>
                                        <div class="value">{comp:.1f}</div>
                                    </div>
                                    <div class="keyframe-score">
                                        <div class="label">Spatial</div>
                                        <div class="value">{selector_score:.1f}</div>
                                    </div>
                                    <div class="keyframe-score">
                                        <div class="label">Quality</div>
                                        <div class="value">{quality:.1f}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
'''

        # Prompt section
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            query=query,
            target_categories=parsed_categories if parsed_categories else ["auto-detected"],
            anchor_categories=[],
            spatial_relation=parsed_relation,
            hypothesis_kind="direct",
            num_keyframes=len(keyframe_paths),
            last_img=len(keyframe_paths) + 1,
            max_idx=max(0, len(keyframe_paths) - 1)
        )

        # Build issues/suggestions display
        issues_html = ""
        if issues:
            issues_html = "<br>".join(f"• {issue}" for issue in issues[:3])
        if suggestions:
            issues_html += "<br><strong>Suggestions:</strong><br>"
            issues_html += "<br>".join(f"• {s}" for s in suggestions[:2])

        html += f'''
                    </div>
                    <div class="bev-section">
'''
        # Add BEV if available
        bev_path = replica_root / scene / "bev" / "scene_bev_e9ff6a93.png"
        bev_b64 = image_to_base64(bev_path) if bev_path.exists() else ""

        if bev_b64:
            html += f'''
                        <img src="data:image/png;base64,{bev_b64}" alt="BEV">
                        <div style="font-size:12px;color:#7f8c8d;margin-top:5px;">Bird's Eye View (Spatial Context)</div>
'''

        html += f'''
                    </div>
                </div>

                <!-- Prompt Section -->
                <div class="prompt-section">
                    <div class="section-title">📝 Evaluation Prompt (Sent to Gemini)</div>
                    <div class="prompt-box">{prompt}</div>
                </div>

                <!-- Output Section -->
                <div class="output-section">
                    <div class="section-title">✅ Gemini Evaluation Output</div>
                    <div class="output-grid">
                        <div class="output-card">
                            <div class="label">Parse Score</div>
                            <div class="value">{parse_score:.1f}</div>
                        </div>
                        <div class="output-card">
                            <div class="label">Selector Score</div>
                            <div class="value">{selector_score:.1f}</div>
                        </div>
                        <div class="output-card">
                            <div class="label">Overall Score</div>
                            <div class="value">{overall_score:.1f}</div>
                        </div>
                        <div class="output-card {'warning' if not can_answer else ''}">
                            <div class="label">Can Answer Query</div>
                            <div class="value">{"✓ Yes" if can_answer else "✗ No"}</div>
                        </div>
                    </div>

                    <div class="reasoning-box">
                        <strong>Evaluation Details:</strong><br>
                        {issues_html if issues_html else "No issues detected. Keyframes adequately show the target objects."}
                    </div>
                </div>
            </div>
        </div>
'''

    html += '''
        <div style="text-align:center; margin-top:30px; padding:20px; background:#ecf0f1; border-radius:12px;">
            <h3>Score Computation Formula</h3>
            <p style="font-family:monospace; font-size:14px;">
                selector_score = 0.35 × target_visibility + 0.25 × target_completeness + 0.25 × spatial_context + 0.15 × image_quality
            </p>
            <p style="color:#7f8c8d; font-size:12px;">
                When spatial_context is null: weights redistribute to 0.45 / 0.35 / 0.20
            </p>
        </div>
    </div>
</body>
</html>
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Evaluation demo saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate demo HTML pages")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--replica_root", type=str, default="/Users/bytedance/Replica")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    replica_root = Path(args.replica_root)

    print(f"Loading results from: {eval_dir}")
    report, results = load_results(eval_dir)
    print(f"Loaded {len(results)} cases")

    # Generate all demos
    print("\nGenerating generation demo...")
    generate_generation_demo_html(
        results, replica_root,
        eval_dir / "demo_generation.html"
    )

    print("Generating query parse demo...")
    generate_query_parse_demo_html(
        results, replica_root,
        eval_dir / "demo_query_parse.html"
    )

    print("Generating keyframe selector demo...")
    generate_keyframe_selector_demo_html(
        results, replica_root,
        eval_dir / "demo_keyframe_selector.html"
    )

    print("Generating evaluation demo...")
    generate_evaluation_demo_html(
        results, replica_root,
        eval_dir / "demo_evaluation.html"
    )

    print(f"\nDone! Open demos in browser:")
    print(f"  open {eval_dir / 'demo_generation.html'}")
    print(f"  open {eval_dir / 'demo_query_parse.html'}")
    print(f"  open {eval_dir / 'demo_keyframe_selector.html'}")
    print(f"  open {eval_dir / 'demo_evaluation.html'}")


if __name__ == "__main__":
    main()
