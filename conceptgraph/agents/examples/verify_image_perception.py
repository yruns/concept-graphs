#!/usr/bin/env python
"""Minimal smoke test to verify Stage-2 agent actually sees images correctly.

This script:
1. Creates a simple test image with known visual content
2. Sends it directly to GPT 5.2 to verify multimodal perception
3. Then tests through the Stage2DeepResearchAgent to ensure the full pipeline works

Run: .venv/bin/python -m conceptgraph.agents.examples.verify_image_perception
"""

from __future__ import annotations

import base64
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from conceptgraph.agents import (
    Stage2DeepAgentConfig,
    Stage2DeepResearchAgent,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2TaskType,
)
from conceptgraph.agents.models import KeyframeEvidence


def create_test_image(text: str, color: tuple = (255, 0, 0), size: tuple = (256, 256)) -> Path:
    """Create a simple test image with text and colored background region."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    # Draw a colored rectangle
    rect_size = 80
    x0, y0 = (size[0] - rect_size) // 2, 20
    draw.rectangle([x0, y0, x0 + rect_size, y0 + rect_size], fill=color)

    # Draw the text below the rectangle
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (size[0] - text_width) // 2
    text_y = y0 + rect_size + 20
    draw.text((text_x, text_y), text, fill="black", font=font)

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    return Path(tmp.name)


def image_to_data_url(image_path: Path) -> str:
    """Convert image to base64 data URL."""
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def test_raw_multimodal_client():
    """Test 1: Direct AzureChatOpenAI multimodal call."""
    print("\n" + "=" * 60)
    print("TEST 1: Raw AzureChatOpenAI Multimodal Verification")
    print("=" * 60)

    config = Stage2DeepAgentConfig()

    # Create test image with red square and "HELLO"
    test_image = create_test_image("HELLO", color=(255, 0, 0))
    print(f"Created test image: {test_image}")

    # Build client directly
    client = AzureChatOpenAI(
        azure_deployment=config.model_name,
        model=config.model_name,
        api_key=config.api_key,
        azure_endpoint=config.base_url,
        api_version=config.api_version,
        temperature=0.1,
        max_tokens=500,
    )

    # Build multimodal message
    data_url = image_to_data_url(test_image)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe exactly what you see in this image. Be specific about colors, shapes, and any text visible.",
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]
    )

    print("Sending multimodal request to GPT 5.2...")
    response = client.invoke([message])
    print(f"\nModel response:\n{response.content}")

    # Verify key elements
    content_lower = response.content.lower()
    checks = {
        "sees red color": "red" in content_lower,
        "sees square/rectangle": "square" in content_lower or "rectangle" in content_lower,
        "sees text HELLO": "hello" in content_lower,
    }

    print("\nVerification checks:")
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

    # Cleanup
    test_image.unlink()

    return all(checks.values())


def test_stage2_agent_image_perception():
    """Test 2: Full Stage2DeepResearchAgent with image."""
    print("\n" + "=" * 60)
    print("TEST 2: Stage2DeepResearchAgent Image Perception")
    print("=" * 60)

    # Create test image with blue triangle and "TRIANGLE"
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)

    # Draw a blue triangle
    points = [(128, 30), (60, 150), (196, 150)]
    draw.polygon(points, fill=(0, 0, 255))

    # Draw text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((85, 180), "BLUE SHAPE", fill="black", font=font)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG", quality=95)
    test_image = Path(tmp.name)
    print(f"Created test image: {test_image}")

    # Build evidence bundle with the test image
    bundle = Stage2EvidenceBundle(
        scene_id="test_scene",
        scene_summary="Test scene with a geometric shape for perception verification.",
        keyframes=[
            KeyframeEvidence(
                keyframe_idx=0,
                image_path=str(test_image),
                view_id=0,
                frame_id=0,
                note="Test image with geometric shape",
            )
        ],
    )

    # Build task
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.QA,
        user_query="What shape and color do you see in the image? Also mention any text visible.",
        max_reasoning_turns=1,
    )

    # Run agent
    agent = Stage2DeepResearchAgent()
    print("Running Stage2DeepResearchAgent...")

    try:
        result = agent.run(task, bundle)

        print(f"\nAgent status: {result.result.status.value}")
        print(f"Agent summary: {result.result.summary}")
        print(f"Agent confidence: {result.result.confidence}")

        if result.result.payload:
            print(f"Agent payload: {result.result.payload}")

        # Check if agent saw the key elements
        summary_lower = result.result.summary.lower()
        payload_str = str(result.result.payload).lower()
        combined = summary_lower + " " + payload_str

        checks = {
            "sees blue color": "blue" in combined,
            "sees triangle/shape": "triangle" in combined or "shape" in combined,
            "sees text": "blue shape" in combined or "text" in combined,
        }

        print("\nVerification checks:")
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL (model may have different interpretation)"
            print(f"  {check}: {status}")

        success = all(checks.values())

    except Exception as e:
        print(f"\n❌ Agent failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Cleanup
    test_image.unlink()

    return success


def main():
    print("=" * 60)
    print("Stage-2 Agent Image Perception Verification")
    print("=" * 60)

    # Test 1: Raw client
    raw_ok = test_raw_multimodal_client()

    # Test 2: Full agent
    agent_ok = test_stage2_agent_image_perception()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Raw multimodal): {'✅ PASS' if raw_ok else '❌ FAIL'}")
    print(f"Test 2 (Stage2 agent):   {'✅ PASS' if agent_ok else '❌ FAIL'}")

    if raw_ok and agent_ok:
        print("\n✅ Image perception verified! The agent sees images correctly.")
    elif raw_ok and not agent_ok:
        print("\n⚠️  Raw client works but agent has issues. Check agent pipeline.")
    else:
        print("\n❌ Image perception failed. Check model connectivity and encoding.")


if __name__ == "__main__":
    main()
