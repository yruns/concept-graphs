"""SQLite-backed trace storage and web server for Stage-2 agent debugging.

This module provides:
- TraceDB: SQLite database for persisting execution traces
- TracingAgent: Wrapper that automatically records traces to DB
- TraceServer: Web server for browsing and querying traces

Usage:
    # Record traces
    db = TraceDB("traces.db")
    agent = TracingAgent(agent, db)
    result = agent.run(task, bundle)

    # Start server
    server = TraceServer(db)
    server.run(port=8765)
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

from loguru import logger

from .models import (
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
    Stage2ToolObservation,
)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TraceRecord:
    """One execution trace record stored in SQLite."""

    trace_id: str
    created_at: float
    finished_at: Optional[float] = None

    # Task info
    task_type: str = ""
    user_query: str = ""
    plan_mode: str = ""

    # Bundle info
    scene_id: str = ""
    initial_keyframe_count: int = 0
    final_keyframe_count: int = 0

    # Result info
    status: str = ""
    confidence: float = 0.0
    summary: str = ""

    # Metrics
    num_turns: int = 0
    num_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0

    # Serialized data
    tool_trace_json: str = "[]"
    initial_bundle_json: str = "{}"
    final_bundle_json: str = "{}"
    keyframe_paths_json: str = "[]"  # For image serving
    metadata_json: str = "{}"

    @property
    def tool_trace(self) -> List[Dict]:
        return json.loads(self.tool_trace_json)

    @property
    def keyframe_paths(self) -> List[str]:
        return json.loads(self.keyframe_paths_json)


@dataclass
class TurnRecord:
    """One reasoning turn within a trace."""

    turn_id: str
    trace_id: str
    turn_idx: int
    created_at: float

    input_text: str = ""
    input_image_count: int = 0
    tool_calls_json: str = "[]"
    new_evidence_count: int = 0
    llm_response: str = ""

    @property
    def tool_calls(self) -> List[Dict]:
        return json.loads(self.tool_calls_json)


# ============================================================================
# SQLite Database
# ============================================================================


class TraceDB:
    """SQLite database for persisting execution traces."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS traces (
        trace_id TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        finished_at REAL,
        task_type TEXT,
        user_query TEXT,
        plan_mode TEXT,
        scene_id TEXT,
        initial_keyframe_count INTEGER DEFAULT 0,
        final_keyframe_count INTEGER DEFAULT 0,
        status TEXT,
        confidence REAL DEFAULT 0.0,
        summary TEXT,
        num_turns INTEGER DEFAULT 0,
        num_tool_calls INTEGER DEFAULT 0,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        duration_ms REAL DEFAULT 0.0,
        tool_trace_json TEXT DEFAULT '[]',
        initial_bundle_json TEXT DEFAULT '{}',
        final_bundle_json TEXT DEFAULT '{}',
        keyframe_paths_json TEXT DEFAULT '[]',
        metadata_json TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS turns (
        turn_id TEXT PRIMARY KEY,
        trace_id TEXT NOT NULL,
        turn_idx INTEGER NOT NULL,
        created_at REAL NOT NULL,
        input_text TEXT,
        input_image_count INTEGER DEFAULT 0,
        tool_calls_json TEXT DEFAULT '[]',
        new_evidence_count INTEGER DEFAULT 0,
        llm_response TEXT,
        FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
    );

    CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_traces_task_type ON traces(task_type);
    CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
    CREATE INDEX IF NOT EXISTS idx_turns_trace ON turns(trace_id, turn_idx);
    """

    def __init__(self, db_path: Union[str, Path] = "traces.db") -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def insert_trace(self, trace: TraceRecord) -> None:
        """Insert a new trace record."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id, created_at, finished_at,
                    task_type, user_query, plan_mode,
                    scene_id, initial_keyframe_count, final_keyframe_count,
                    status, confidence, summary,
                    num_turns, num_tool_calls, input_tokens, output_tokens, duration_ms,
                    tool_trace_json, initial_bundle_json, final_bundle_json,
                    keyframe_paths_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id, trace.created_at, trace.finished_at,
                    trace.task_type, trace.user_query, trace.plan_mode,
                    trace.scene_id, trace.initial_keyframe_count, trace.final_keyframe_count,
                    trace.status, trace.confidence, trace.summary,
                    trace.num_turns, trace.num_tool_calls, trace.input_tokens, trace.output_tokens, trace.duration_ms,
                    trace.tool_trace_json, trace.initial_bundle_json, trace.final_bundle_json,
                    trace.keyframe_paths_json, trace.metadata_json,
                ),
            )

    def update_trace(self, trace_id: str, **updates) -> None:
        """Update a trace record."""
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [trace_id]
        with self.transaction() as conn:
            conn.execute(f"UPDATE traces SET {set_clause} WHERE trace_id = ?", values)

    def insert_turn(self, turn: TurnRecord) -> None:
        """Insert a turn record."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, trace_id, turn_idx, created_at,
                    input_text, input_image_count, tool_calls_json,
                    new_evidence_count, llm_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id, turn.trace_id, turn.turn_idx, turn.created_at,
                    turn.input_text, turn.input_image_count, turn.tool_calls_json,
                    turn.new_evidence_count, turn.llm_response,
                ),
            )

    def get_trace(self, trace_id: str) -> Optional[TraceRecord]:
        """Get a trace by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        return TraceRecord(**dict(row))

    def get_turns(self, trace_id: str) -> List[TurnRecord]:
        """Get all turns for a trace."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM turns WHERE trace_id = ? ORDER BY turn_idx", (trace_id,)
        ).fetchall()
        return [TurnRecord(**dict(row)) for row in rows]

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[TraceRecord]:
        """List traces with optional filtering."""
        conn = self._get_conn()
        query = "SELECT * FROM traces WHERE 1=1"
        params: List[Any] = []

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        if search:
            query += " AND (user_query LIKE ? OR summary LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [TraceRecord(**dict(row)) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        by_status = dict(conn.execute(
            "SELECT status, COUNT(*) FROM traces GROUP BY status"
        ).fetchall())
        by_task_type = dict(conn.execute(
            "SELECT task_type, COUNT(*) FROM traces GROUP BY task_type"
        ).fetchall())
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM traces WHERE status = 'completed'"
        ).fetchone()[0] or 0.0
        avg_duration = conn.execute(
            "SELECT AVG(duration_ms) FROM traces"
        ).fetchone()[0] or 0.0
        avg_tool_calls = conn.execute(
            "SELECT AVG(num_tool_calls) FROM traces"
        ).fetchone()[0] or 0.0

        return {
            "total_traces": total,
            "by_status": by_status,
            "by_task_type": by_task_type,
            "avg_confidence": avg_confidence,
            "avg_duration_ms": avg_duration,
            "avg_tool_calls": avg_tool_calls,
        }


# ============================================================================
# Tracing Agent Wrapper
# ============================================================================


class TracingAgent:
    """Wrapper that records execution traces to SQLite."""

    def __init__(
        self,
        agent: Any,  # Stage2DeepResearchAgent
        db: TraceDB,
    ) -> None:
        self.agent = agent
        self.db = db

    def run(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Stage2AgentResult:
        """Run the agent and record the trace."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # Collect keyframe paths for image serving
        keyframe_paths = [kf.image_path for kf in bundle.keyframes]
        if bundle.bev_image_path:
            keyframe_paths.append(bundle.bev_image_path)

        # Create initial trace record
        trace = TraceRecord(
            trace_id=trace_id,
            created_at=start_time,
            task_type=task.task_type.value,
            user_query=task.user_query,
            plan_mode=task.plan_mode.value,
            scene_id=bundle.scene_id,
            initial_keyframe_count=len(bundle.keyframes),
            initial_bundle_json=bundle.model_dump_json(),
            keyframe_paths_json=json.dumps(keyframe_paths),
            metadata_json=json.dumps(metadata or {}),
        )
        self.db.insert_trace(trace)

        logger.info(f"[TracingAgent] Started trace {trace_id}")

        try:
            # Run the actual agent
            result = self.agent.run(task, bundle)

            # Update trace with results
            end_time = time.time()

            # Collect all keyframe paths from final bundle
            final_paths = [kf.image_path for kf in result.final_bundle.keyframes]
            if result.final_bundle.bev_image_path:
                final_paths.append(result.final_bundle.bev_image_path)

            self.db.update_trace(
                trace_id,
                finished_at=end_time,
                final_keyframe_count=len(result.final_bundle.keyframes),
                status=result.result.status.value,
                confidence=result.result.confidence,
                summary=result.result.summary,
                num_tool_calls=len(result.tool_trace),
                duration_ms=(end_time - start_time) * 1000,
                tool_trace_json=json.dumps([t.model_dump() for t in result.tool_trace]),
                final_bundle_json=result.final_bundle.model_dump_json(),
                keyframe_paths_json=json.dumps(list(set(keyframe_paths + final_paths))),
            )

            logger.info(
                f"[TracingAgent] Completed trace {trace_id} in {(end_time - start_time)*1000:.0f}ms"
            )
            return result

        except Exception as e:
            self.db.update_trace(
                trace_id,
                finished_at=time.time(),
                status="failed",
                summary=str(e),
            )
            raise


# ============================================================================
# Web Server
# ============================================================================


def _generate_thumbnail(image_path: str, size: int = 200) -> Optional[str]:
    """Generate a base64 thumbnail."""
    try:
        from PIL import Image
    except ImportError:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


class TraceServerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for trace server."""

    db: TraceDB  # Set by TraceServer

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        """Send HTML response."""
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, image_data: bytes, content_type: str = "image/jpeg") -> None:
        """Send image response."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(image_data))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(image_data)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._serve_index()
            elif path == "/api/traces":
                self._api_list_traces(params)
            elif path == "/api/stats":
                self._api_stats()
            elif path.startswith("/api/traces/"):
                trace_id = path.split("/")[-1]
                self._api_get_trace(trace_id)
            elif path.startswith("/api/image/"):
                # /api/image/<trace_id>/<image_idx>
                parts = path.split("/")
                if len(parts) >= 5:
                    trace_id = parts[3]
                    image_idx = int(parts[4])
                    self._api_get_image(trace_id, image_idx)
                else:
                    self._send_json({"error": "Invalid image path"}, 400)
            elif path == "/trace":
                trace_id = params.get("id", [""])[0]
                self._serve_trace_detail(trace_id)
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as e:
            logger.exception(f"Error handling {path}")
            self._send_json({"error": str(e)}, 500)

    def _api_list_traces(self, params: Dict) -> None:
        """API: List traces."""
        limit = int(params.get("limit", [50])[0])
        offset = int(params.get("offset", [0])[0])
        task_type = params.get("task_type", [None])[0]
        status = params.get("status", [None])[0]
        search = params.get("search", [None])[0]

        traces = self.db.list_traces(limit, offset, task_type, status, search)
        self._send_json({
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "created_at": t.created_at,
                    "task_type": t.task_type,
                    "user_query": t.user_query[:100] + "..." if len(t.user_query) > 100 else t.user_query,
                    "status": t.status,
                    "confidence": t.confidence,
                    "num_tool_calls": t.num_tool_calls,
                    "duration_ms": t.duration_ms,
                }
                for t in traces
            ],
            "limit": limit,
            "offset": offset,
        })

    def _api_stats(self) -> None:
        """API: Get stats."""
        self._send_json(self.db.get_stats())

    def _api_get_trace(self, trace_id: str) -> None:
        """API: Get single trace with full details."""
        trace = self.db.get_trace(trace_id)
        if trace is None:
            self._send_json({"error": "Trace not found"}, 404)
            return

        turns = self.db.get_turns(trace_id)

        # Generate thumbnails for keyframe paths
        keyframe_paths = trace.keyframe_paths
        thumbnails = []
        for i, path in enumerate(keyframe_paths):
            thumb = _generate_thumbnail(path)
            thumbnails.append({
                "idx": i,
                "path": path,
                "thumbnail": thumb,
                "url": f"/api/image/{trace_id}/{i}",
            })

        self._send_json({
            "trace": {
                "trace_id": trace.trace_id,
                "created_at": trace.created_at,
                "finished_at": trace.finished_at,
                "task_type": trace.task_type,
                "user_query": trace.user_query,
                "plan_mode": trace.plan_mode,
                "scene_id": trace.scene_id,
                "initial_keyframe_count": trace.initial_keyframe_count,
                "final_keyframe_count": trace.final_keyframe_count,
                "status": trace.status,
                "confidence": trace.confidence,
                "summary": trace.summary,
                "num_turns": trace.num_turns,
                "num_tool_calls": trace.num_tool_calls,
                "duration_ms": trace.duration_ms,
                "tool_trace": trace.tool_trace,
                "metadata": json.loads(trace.metadata_json),
            },
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "created_at": t.created_at,
                    "input_text": t.input_text[:200] + "..." if len(t.input_text) > 200 else t.input_text,
                    "input_image_count": t.input_image_count,
                    "tool_calls": t.tool_calls,
                    "new_evidence_count": t.new_evidence_count,
                }
                for t in turns
            ],
            "keyframes": thumbnails,
        })

    def _api_get_image(self, trace_id: str, image_idx: int) -> None:
        """API: Serve a keyframe image."""
        trace = self.db.get_trace(trace_id)
        if trace is None:
            self._send_json({"error": "Trace not found"}, 404)
            return

        paths = trace.keyframe_paths
        if image_idx < 0 or image_idx >= len(paths):
            self._send_json({"error": "Image index out of range"}, 404)
            return

        path = Path(paths[image_idx])
        if not path.exists():
            self._send_json({"error": "Image file not found"}, 404)
            return

        # Read and resize image
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img.thumbnail((800, 800))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            self._send_image(buffer.getvalue())
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_index(self) -> None:
        """Serve the main index page."""
        html = self._build_index_html()
        self._send_html(html)

    def _serve_trace_detail(self, trace_id: str) -> None:
        """Serve trace detail page."""
        trace = self.db.get_trace(trace_id)
        if trace is None:
            self._send_html("<h1>Trace not found</h1>", 404)
            return
        html = self._build_trace_html(trace)
        self._send_html(html)

    def _build_index_html(self) -> str:
        """Build the index page HTML."""
        stats = self.db.get_stats()
        traces = self.db.list_traces(limit=20)

        traces_rows = ""
        for t in traces:
            status_class = "completed" if t.status == "completed" else "failed" if t.status == "failed" else "pending"
            ts = datetime.fromtimestamp(t.created_at).strftime("%m-%d %H:%M:%S")
            query_short = t.user_query[:60] + "..." if len(t.user_query) > 60 else t.user_query
            traces_rows += f"""
            <tr onclick="window.location='/trace?id={t.trace_id}'" style="cursor:pointer">
                <td>{ts}</td>
                <td>{t.task_type}</td>
                <td title="{t.user_query}">{query_short}</td>
                <td><span class="status {status_class}">{t.status}</span></td>
                <td>{t.confidence:.2f}</td>
                <td>{t.num_tool_calls}</td>
                <td>{t.duration_ms:.0f}ms</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html><head>
    <meta charset="UTF-8">
    <title>Stage-2 Trace Browser</title>
    {self._common_css()}
</head><body>
    <div class="container">
        <h1>🔍 Stage-2 Agent Trace Browser</h1>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{stats['total_traces']}</div>
                <div class="stat-label">Total Traces</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['avg_confidence']:.2f}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['avg_duration_ms']:.0f}ms</div>
                <div class="stat-label">Avg Duration</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['avg_tool_calls']:.1f}</div>
                <div class="stat-label">Avg Tool Calls</div>
            </div>
        </div>

        <h2>Recent Traces</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Type</th>
                    <th>Query</th>
                    <th>Status</th>
                    <th>Conf.</th>
                    <th>Tools</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>{traces_rows}</tbody>
        </table>
    </div>
</body></html>"""

    def _build_trace_html(self, trace: TraceRecord) -> str:
        """Build trace detail HTML."""
        turns = self.db.get_turns(trace.trace_id)
        keyframe_paths = trace.keyframe_paths

        # Keyframe thumbnails
        keyframes_html = ""
        for i, path in enumerate(keyframe_paths):
            thumb = _generate_thumbnail(path, 150)
            if thumb:
                keyframes_html += f"""
                <div class="image-thumb" onclick="showImage('/api/image/{trace.trace_id}/{i}')">
                    <img src="data:image/jpeg;base64,{thumb}" alt="keyframe {i}">
                    <div class="image-label">#{i}</div>
                </div>"""
            else:
                keyframes_html += f"""
                <div class="image-thumb">
                    <div class="placeholder">#{i}</div>
                </div>"""

        # Tool trace
        tools_html = ""
        for tc in trace.tool_trace:
            tools_html += f"""
            <div class="tool-call">
                <div class="tool-name">{tc['tool_name']}</div>
                <pre class="json">{json.dumps(tc['tool_input'], indent=2)}</pre>
                <div class="tool-response">{tc['response_text'][:300]}...</div>
            </div>"""

        ts = datetime.fromtimestamp(trace.created_at).strftime("%Y-%m-%d %H:%M:%S")
        status_class = "completed" if trace.status == "completed" else "failed"

        return f"""<!DOCTYPE html>
<html><head>
    <meta charset="UTF-8">
    <title>Trace: {trace.trace_id}</title>
    {self._common_css()}
    <style>
        .back {{ color: var(--accent-blue); text-decoration: none; }}
        .summary-box {{ background: var(--bg-card); padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head><body>
    <div class="container">
        <a href="/" class="back">← Back to list</a>
        <h1>Trace: {trace.trace_id}</h1>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{ts}</div>
                <div class="stat-label">Created</div>
            </div>
            <div class="stat-box">
                <div class="stat-value"><span class="status {status_class}">{trace.status}</span></div>
                <div class="stat-label">Status</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{trace.confidence:.2f}</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{trace.duration_ms:.0f}ms</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>

        <h2>Query</h2>
        <div class="summary-box">
            <strong>Task Type:</strong> {trace.task_type}<br>
            <strong>Plan Mode:</strong> {trace.plan_mode}<br>
            <strong>Query:</strong> {trace.user_query}
        </div>

        <h2>Summary</h2>
        <div class="summary-box">{trace.summary}</div>

        <h2>Keyframes ({len(keyframe_paths)})</h2>
        <div class="image-grid">{keyframes_html}</div>

        <h2>Tool Calls ({trace.num_tool_calls})</h2>
        {tools_html if tools_html else '<p>No tool calls</p>'}
    </div>

    <div id="modal" class="modal" onclick="this.style.display='none'">
        <img id="modal-img" src="">
    </div>

    <script>
    function showImage(url) {{
        document.getElementById('modal-img').src = url;
        document.getElementById('modal').style.display = 'flex';
    }}
    </script>
</body></html>"""

    def _common_css(self) -> str:
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
    h1 { color: var(--accent-blue); }
    h2 { color: var(--accent-orange); margin-top: 30px; }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    .stat-box {
        background: var(--bg-card);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stat-value { font-size: 20px; font-weight: bold; color: var(--accent-blue); }
    .stat-label { color: var(--text-secondary); font-size: 12px; }

    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
    th { background: var(--bg-card); }
    tr:hover { background: var(--bg-secondary); }

    .status {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .status.completed { background: var(--accent-green); color: black; }
    .status.failed { background: var(--accent-red); color: white; }
    .status.pending { background: var(--accent-orange); color: black; }

    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
    }
    .image-thumb {
        position: relative;
        cursor: pointer;
        border-radius: 6px;
        overflow: hidden;
        border: 2px solid transparent;
    }
    .image-thumb:hover { border-color: var(--accent-blue); }
    .image-thumb img { width: 100%; display: block; }
    .image-label {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.7);
        padding: 4px;
        font-size: 11px;
        text-align: center;
    }
    .placeholder {
        height: 100px;
        background: var(--bg-card);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .tool-call {
        background: var(--bg-secondary);
        border-left: 3px solid var(--accent-orange);
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 6px 6px 0;
    }
    .tool-name { color: var(--accent-orange); font-weight: bold; font-family: monospace; }
    .tool-response { color: var(--text-secondary); font-size: 13px; margin-top: 10px; }
    .json { background: #0d1117; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto; }

    .modal {
        display: none;
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.9);
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .modal img { max-width: 90%; max-height: 90%; }
</style>"""


class TraceServer:
    """Web server for browsing and querying traces."""

    def __init__(self, db: TraceDB, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.db = db
        self.host = host
        self.port = port

    def run(self) -> None:
        """Start the server."""
        # Inject db into handler class
        handler = type(
            "Handler",
            (TraceServerHandler,),
            {"db": self.db},
        )

        server = HTTPServer((self.host, self.port), handler)
        logger.info(f"Trace server running at http://{self.host}:{self.port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")
            server.shutdown()
