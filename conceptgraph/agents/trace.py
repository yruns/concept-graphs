"""Execution trace recording and visualization for Stage-2 agent.

This module provides:
- TraceRecorder: captures multimodal execution events during agent runs
- HTMLTraceRenderer: generates interactive HTML reports with embedded images
"""

from __future__ import annotations

import base64
import html
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import (
    Stage2AgentResult,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2ToolObservation,
)


@dataclass
class TraceImageRef:
    """Reference to an image in the trace."""

    path: str
    thumbnail_b64: Optional[str] = None  # Small preview
    role: str = "keyframe"  # keyframe, bev, crop, etc.
    keyframe_idx: Optional[int] = None
    view_id: Optional[int] = None


@dataclass
class TraceToolCall:
    """Single tool invocation record."""

    tool_name: str
    tool_input: Dict[str, Any]
    response_text: str
    duration_ms: float = 0.0
    updated_bundle: bool = False  # Did this call update the evidence bundle?


@dataclass
class TraceTurn:
    """One reasoning turn in the agent execution."""

    turn_idx: int
    timestamp: float
    input_text: str
    input_images: List[TraceImageRef] = field(default_factory=list)
    tool_calls: List[TraceToolCall] = field(default_factory=list)
    llm_response: Optional[str] = None
    new_evidence_images: List[TraceImageRef] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExecutionTrace:
    """Complete execution trace for one Stage-2 agent run."""

    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    task: Optional[Stage2TaskSpec] = None
    initial_bundle: Optional[Stage2EvidenceBundle] = None
    turns: List[TraceTurn] = field(default_factory=list)
    final_result: Optional[Stage2AgentResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class TraceRecorder:
    """Records multimodal execution events during Stage-2 agent runs.

    Usage:
        recorder = TraceRecorder()
        recorder.start(task, bundle)
        # ... agent execution with recorder hooks ...
        recorder.finish(result)
        trace = recorder.get_trace()
    """

    def __init__(self, thumbnail_size: int = 150) -> None:
        self.thumbnail_size = thumbnail_size
        self._trace: Optional[ExecutionTrace] = None
        self._current_turn: Optional[TraceTurn] = None
        self._turn_counter = 0

    def start(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle) -> None:
        """Begin recording a new trace."""
        self._trace = ExecutionTrace(
            trace_id=f"trace_{int(time.time() * 1000)}",
            start_time=time.time(),
            task=task,
            initial_bundle=bundle.model_copy(deep=True),
        )
        self._turn_counter = 0

    def begin_turn(self, input_text: str, image_paths: List[str]) -> None:
        """Start recording a new reasoning turn."""
        self._turn_counter += 1
        self._current_turn = TraceTurn(
            turn_idx=self._turn_counter,
            timestamp=time.time(),
            input_text=input_text,
            input_images=[self._make_image_ref(p) for p in image_paths],
        )

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        response_text: str,
        duration_ms: float = 0.0,
        updated_bundle: bool = False,
    ) -> None:
        """Record a tool invocation within the current turn."""
        if self._current_turn is None:
            return
        self._current_turn.tool_calls.append(
            TraceToolCall(
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=response_text,
                duration_ms=duration_ms,
                updated_bundle=updated_bundle,
            )
        )

    def record_llm_response(self, response: str) -> None:
        """Record the LLM's text response for the current turn."""
        if self._current_turn is not None:
            self._current_turn.llm_response = response

    def record_evidence_update(self, new_image_paths: List[str]) -> None:
        """Record newly injected evidence images."""
        if self._current_turn is None:
            return
        self._current_turn.new_evidence_images = [
            self._make_image_ref(p, role="new_evidence") for p in new_image_paths
        ]

    def end_turn(self, notes: str = "") -> None:
        """Finish the current turn and add it to the trace."""
        if self._current_turn is not None:
            self._current_turn.notes = notes
            if self._trace is not None:
                self._trace.turns.append(self._current_turn)
            self._current_turn = None

    def finish(self, result: Stage2AgentResult) -> None:
        """Complete the trace with the final result."""
        if self._trace is not None:
            self._trace.end_time = time.time()
            self._trace.final_result = result

    def get_trace(self) -> Optional[ExecutionTrace]:
        """Return the recorded trace."""
        return self._trace

    def _make_image_ref(self, path: str, role: str = "keyframe") -> TraceImageRef:
        """Create an image reference with optional thumbnail."""
        ref = TraceImageRef(path=path, role=role)
        try:
            ref.thumbnail_b64 = self._generate_thumbnail(path)
        except Exception:
            pass
        return ref

    def _generate_thumbnail(self, image_path: str) -> Optional[str]:
        """Generate a small base64 thumbnail for preview."""
        try:
            from PIL import Image
        except ImportError:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        img = Image.open(path).convert("RGB")
        img.thumbnail((self.thumbnail_size, self.thumbnail_size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


class HTMLTraceRenderer:
    """Renders an ExecutionTrace as an interactive HTML report."""

    def __init__(self, full_image_size: int = 600) -> None:
        self.full_image_size = full_image_size

    def render(self, trace: ExecutionTrace, output_path: Optional[Path] = None) -> str:
        """Render the trace to HTML.

        Args:
            trace: The execution trace to render
            output_path: Optional path to write the HTML file

        Returns:
            The HTML content as a string
        """
        html_content = self._build_html(trace)
        if output_path is not None:
            output_path.write_text(html_content, encoding="utf-8")
        return html_content

    def _build_html(self, trace: ExecutionTrace) -> str:
        """Build the complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage-2 Agent Trace: {trace.trace_id}</title>
    {self._css()}
</head>
<body>
    <div class="container">
        <h1>🔍 Stage-2 Agent Execution Trace</h1>
        {self._render_header(trace)}
        {self._render_initial_evidence(trace)}
        {self._render_turns(trace)}
        {self._render_final_result(trace)}
    </div>
    {self._javascript()}
</body>
</html>"""

    def _css(self) -> str:
        return """<style>
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --text-primary: #eee;
        --text-secondary: #aaa;
        --accent-blue: #4a9eff;
        --accent-green: #4ade80;
        --accent-orange: #fb923c;
        --accent-purple: #a78bfa;
        --accent-red: #f87171;
        --border-color: #334155;
    }
    * { box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        margin: 0;
        padding: 20px;
        line-height: 1.6;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: var(--accent-blue); border-bottom: 2px solid var(--accent-blue); padding-bottom: 10px; }
    h2 { color: var(--accent-purple); margin-top: 30px; }
    h3 { color: var(--accent-green); }

    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    .header-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    .stat-box {
        background: var(--bg-card);
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }
    .stat-value { font-size: 24px; font-weight: bold; color: var(--accent-blue); }
    .stat-label { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; }

    .turn-card {
        background: var(--bg-secondary);
        border-left: 4px solid var(--accent-purple);
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
    }
    .turn-header {
        background: var(--bg-card);
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        border-radius: 0 8px 0 0;
    }
    .turn-header:hover { background: #1a4a7a; }
    .turn-content { padding: 20px; display: none; }
    .turn-content.expanded { display: block; }

    .tool-call {
        background: var(--bg-card);
        border-left: 3px solid var(--accent-orange);
        margin: 10px 0;
        padding: 15px;
        border-radius: 0 6px 6px 0;
    }
    .tool-name { color: var(--accent-orange); font-weight: bold; font-family: monospace; }
    .tool-updated { color: var(--accent-green); font-size: 12px; margin-left: 10px; }

    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
    .image-thumb {
        position: relative;
        cursor: pointer;
        border-radius: 6px;
        overflow: hidden;
        border: 2px solid transparent;
        transition: border-color 0.2s;
    }
    .image-thumb:hover { border-color: var(--accent-blue); }
    .image-thumb img { width: 100%; height: auto; display: block; }
    .image-label {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.7);
        color: white;
        font-size: 11px;
        padding: 4px 8px;
    }
    .new-evidence { border-color: var(--accent-green); }

    .json-block {
        background: #0d1117;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 15px;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 13px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .collapsible-header {
        cursor: pointer;
        user-select: none;
        padding: 8px 0;
        color: var(--accent-blue);
    }
    .collapsible-header::before { content: '▸ '; }
    .collapsible-header.expanded::before { content: '▾ '; }
    .collapsible-content { display: none; }
    .collapsible-content.expanded { display: block; }

    .result-card {
        border-left: 4px solid var(--accent-green);
    }
    .result-card.failed { border-left-color: var(--accent-red); }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-completed { background: var(--accent-green); color: black; }
    .status-failed { background: var(--accent-red); color: white; }
    .status-insufficient { background: var(--accent-orange); color: black; }

    .confidence-bar {
        height: 8px;
        background: var(--bg-card);
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-red), var(--accent-orange), var(--accent-green));
        border-radius: 4px;
        transition: width 0.3s;
    }

    /* Modal for full-size images */
    .modal {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.9);
        z-index: 1000;
        justify-content: center;
        align-items: center;
    }
    .modal.active { display: flex; }
    .modal img { max-width: 90%; max-height: 90%; object-fit: contain; }
    .modal-close {
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 40px;
        cursor: pointer;
    }
</style>"""

    def _javascript(self) -> str:
        return """<script>
    // Toggle turn expansion
    document.querySelectorAll('.turn-header').forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            content.classList.toggle('expanded');
        });
    });

    // Toggle collapsible sections
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            header.nextElementSibling.classList.toggle('expanded');
        });
    });

    // Image modal
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');

    document.querySelectorAll('.image-thumb').forEach(thumb => {
        thumb.addEventListener('click', () => {
            const fullPath = thumb.dataset.fullPath;
            const thumbImg = thumb.querySelector('img');
            // Use thumbnail as fallback, or load full image if path is provided
            modalImg.src = thumbImg.src;
            modal.classList.add('active');
        });
    });

    modal?.addEventListener('click', () => modal.classList.remove('active'));
</script>
<div id="imageModal" class="modal">
    <span class="modal-close">&times;</span>
    <img id="modalImage" src="">
</div>"""

    def _render_header(self, trace: ExecutionTrace) -> str:
        task = trace.task
        task_type = task.task_type.value if task else "unknown"
        query = html.escape(task.user_query if task else "N/A")
        duration = f"{trace.duration_ms:.0f}ms" if trace.end_time else "in progress"
        total_tools = sum(len(t.tool_calls) for t in trace.turns)

        return f"""
<div class="card">
    <div class="header-grid">
        <div class="stat-box">
            <div class="stat-value">{len(trace.turns)}</div>
            <div class="stat-label">Turns</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{total_tools}</div>
            <div class="stat-label">Tool Calls</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{duration}</div>
            <div class="stat-label">Duration</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{task_type}</div>
            <div class="stat-label">Task Type</div>
        </div>
    </div>
    <p style="margin-top: 15px;"><strong>Query:</strong> {query}</p>
    <p><strong>Trace ID:</strong> <code>{trace.trace_id}</code></p>
    <p><strong>Started:</strong> {datetime.fromtimestamp(trace.start_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>"""

    def _render_initial_evidence(self, trace: ExecutionTrace) -> str:
        bundle = trace.initial_bundle
        if bundle is None:
            return ""

        images_html = self._render_image_grid(
            [(kf.image_path, f"idx={kf.keyframe_idx} view={kf.view_id}")
             for kf in bundle.keyframes]
        )

        hypothesis_json = ""
        if bundle.hypothesis:
            hypothesis_json = html.escape(
                json.dumps(bundle.hypothesis.model_dump(), indent=2, ensure_ascii=False)
            )

        return f"""
<h2>📦 Initial Evidence Bundle</h2>
<div class="card">
    <p><strong>Scene:</strong> {bundle.scene_id or 'unknown'}</p>
    <p><strong>Stage-1 Query:</strong> {html.escape(bundle.stage1_query or 'N/A')}</p>
    <p><strong>Keyframes:</strong> {len(bundle.keyframes)}</p>

    <h3>Keyframe Images</h3>
    {images_html}

    <div class="collapsible-header">Stage-1 Hypothesis</div>
    <div class="collapsible-content">
        <pre class="json-block">{hypothesis_json or '{}'}</pre>
    </div>

    <div class="collapsible-header">Scene Summary</div>
    <div class="collapsible-content">
        <p>{html.escape(bundle.scene_summary or 'N/A')}</p>
    </div>
</div>"""

    def _render_turns(self, trace: ExecutionTrace) -> str:
        if not trace.turns:
            return "<p><em>No reasoning turns recorded.</em></p>"

        turns_html = []
        for turn in trace.turns:
            turns_html.append(self._render_turn(turn))

        return f"""
<h2>🔄 Reasoning Turns</h2>
{''.join(turns_html)}"""

    def _render_turn(self, turn: TraceTurn) -> str:
        tool_count = len(turn.tool_calls)
        new_images = len(turn.new_evidence_images)

        tools_html = ""
        for tc in turn.tool_calls:
            updated_badge = '<span class="tool-updated">✓ updated evidence</span>' if tc.updated_bundle else ""
            input_json = html.escape(json.dumps(tc.tool_input, indent=2, ensure_ascii=False))
            response = html.escape(tc.response_text[:500] + "..." if len(tc.response_text) > 500 else tc.response_text)
            tools_html += f"""
<div class="tool-call">
    <div><span class="tool-name">{tc.tool_name}</span>{updated_badge}</div>
    <div class="collapsible-header">Input</div>
    <div class="collapsible-content">
        <pre class="json-block">{input_json}</pre>
    </div>
    <div class="collapsible-header">Response</div>
    <div class="collapsible-content">
        <pre class="json-block">{response}</pre>
    </div>
</div>"""

        input_images_html = self._render_image_grid(
            [(img.path, f"{img.role} idx={img.keyframe_idx}") for img in turn.input_images],
            css_class=""
        ) if turn.input_images else ""

        new_images_html = ""
        if turn.new_evidence_images:
            new_images_html = f"""
<h4 style="color: var(--accent-green);">🆕 Newly Acquired Evidence</h4>
{self._render_image_grid(
    [(img.path, f"{img.role}") for img in turn.new_evidence_images],
    css_class="new-evidence"
)}"""

        input_text_preview = html.escape(turn.input_text[:200] + "..." if len(turn.input_text) > 200 else turn.input_text)

        return f"""
<div class="turn-card">
    <div class="turn-header">
        <span><strong>Turn {turn.turn_idx}</strong></span>
        <span>{tool_count} tool call(s) | {new_images} new image(s)</span>
    </div>
    <div class="turn-content">
        <h4>Input Message</h4>
        <p>{input_text_preview}</p>
        {input_images_html}

        {f'<h4>Tool Calls</h4>{tools_html}' if tools_html else '<p><em>No tool calls this turn.</em></p>'}

        {new_images_html}

        {f'<p><strong>Notes:</strong> {html.escape(turn.notes)}</p>' if turn.notes else ''}
    </div>
</div>"""

    def _render_final_result(self, trace: ExecutionTrace) -> str:
        result = trace.final_result
        if result is None:
            return "<p><em>No final result recorded.</em></p>"

        resp = result.result
        status_class = {
            "completed": "status-completed",
            "failed": "status-failed",
            "insufficient_evidence": "status-insufficient",
            "needs_more_evidence": "status-insufficient",
        }.get(resp.status.value, "")

        card_class = "failed" if resp.status.value == "failed" else ""
        confidence_pct = resp.confidence * 100

        payload_json = html.escape(json.dumps(resp.payload, indent=2, ensure_ascii=False))
        uncertainties = "".join(f"<li>{html.escape(u)}</li>" for u in resp.uncertainties) or "<li>None</li>"
        cited_frames = ", ".join(str(i) for i in resp.cited_frame_indices) or "None"

        return f"""
<h2>✅ Final Result</h2>
<div class="card result-card {card_class}">
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
        <span class="status-badge {status_class}">{resp.status.value}</span>
        <span>Confidence: <strong>{resp.confidence:.2f}</strong></span>
    </div>

    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
    </div>

    <h3>Summary</h3>
    <p>{html.escape(resp.summary)}</p>

    <h3>Cited Frames</h3>
    <p>{cited_frames}</p>

    <h3>Uncertainties</h3>
    <ul>{uncertainties}</ul>

    <div class="collapsible-header">Payload</div>
    <div class="collapsible-content">
        <pre class="json-block">{payload_json}</pre>
    </div>
</div>"""

    def _render_image_grid(
        self,
        images: List[tuple],  # [(path, label), ...]
        css_class: str = "",
    ) -> str:
        if not images:
            return "<p><em>No images</em></p>"

        items = []
        for path, label in images:
            # Try to generate thumbnail inline
            thumb_src = self._get_thumbnail_data_url(path)
            if thumb_src:
                items.append(f"""
<div class="image-thumb {css_class}" data-full-path="{html.escape(path)}">
    <img src="{thumb_src}" alt="{html.escape(label)}">
    <div class="image-label">{html.escape(label)}</div>
</div>""")
            else:
                items.append(f"""
<div class="image-thumb {css_class}" data-full-path="{html.escape(path)}">
    <div style="padding: 20px; text-align: center; background: var(--bg-card);">
        <div>📷</div>
        <div class="image-label">{html.escape(label)}</div>
    </div>
</div>""")

        return f'<div class="image-grid">{"".join(items)}</div>'

    def _get_thumbnail_data_url(self, path: str, size: int = 200) -> Optional[str]:
        """Generate a base64 data URL thumbnail for embedding in HTML."""
        try:
            from PIL import Image
        except ImportError:
            return None

        p = Path(path)
        if not p.exists():
            return None

        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((size, size))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None


def save_trace_report(
    result: Stage2AgentResult,
    task: Stage2TaskSpec,
    initial_bundle: Stage2EvidenceBundle,
    output_path: Union[str, Path],
) -> Path:
    """Convenience function to create a trace report from an agent result.

    This is useful when you don't have a TraceRecorder instance but want to
    generate a basic visualization from the final result.
    """
    # Create a minimal trace from the result
    trace = ExecutionTrace(
        trace_id=f"trace_{int(time.time() * 1000)}",
        start_time=time.time(),
        end_time=time.time(),
        task=task,
        initial_bundle=initial_bundle,
        final_result=result,
    )

    # Create a single turn from tool trace
    if result.tool_trace:
        turn = TraceTurn(
            turn_idx=1,
            timestamp=time.time(),
            input_text=task.user_query,
            input_images=[
                TraceImageRef(path=kf.image_path, keyframe_idx=kf.keyframe_idx, view_id=kf.view_id)
                for kf in initial_bundle.keyframes
            ],
            tool_calls=[
                TraceToolCall(
                    tool_name=tc.tool_name,
                    tool_input=tc.tool_input,
                    response_text=tc.response_text,
                    updated_bundle="updated_bundle" in tc.response_text.lower(),
                )
                for tc in result.tool_trace
            ],
        )
        trace.turns.append(turn)

    # Render
    renderer = HTMLTraceRenderer()
    output_path = Path(output_path)
    renderer.render(trace, output_path)
    return output_path
