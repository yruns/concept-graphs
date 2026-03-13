#!/usr/bin/env python3
"""Visualize evaluation results with keyframe images.

Generates:
1. HTML report with all cases and keyframe images
2. Failed case analysis
3. Per-scene score distribution charts

Usage:
    python -m conceptgraph.query_scene.examples.visualize_eval_results \
        --eval_dir /Users/bytedance/Replica/eval_runs/full_20260313_043040
"""

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import matplotlib for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(eval_dir: Path) -> tuple[Dict, List[Dict]]:
    """Load report and results from evaluation directory."""
    report_path = eval_dir / "report.json"
    results_path = eval_dir / "results.json"

    with open(report_path) as f:
        report = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    return report, results


def get_frame_path(replica_root: Path, scene: str, view_id: int) -> Path:
    """Convert view_id to frame image path."""
    frame_idx = view_id * 5
    return replica_root / scene / "results" / f"frame{frame_idx:06d}.jpg"


def image_to_base64(image_path: Path) -> Optional[str]:
    """Convert image to base64 for embedding in HTML."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_keyframe_info(case: Dict) -> tuple[List[int], List[str]]:
    """Extract selected keyframe view_ids and paths from case.

    Returns:
        tuple of (view_ids, paths) - paths take precedence if available
    """
    selection = case.get("selection", {})
    if not selection:
        return [], []

    # Check different possible structures
    data = selection.get("data", selection)
    if not isinstance(data, dict):
        return [], []

    # Try different field names
    view_ids = data.get("selected_view_ids", []) or []
    paths = data.get("keyframe_paths", []) or []
    indices = data.get("keyframe_indices", []) or []

    # If we have paths, use them directly
    if paths:
        return indices, paths

    # Otherwise use view_ids
    return view_ids, []


def get_source_frame_info(case: Dict, replica_root: Path) -> tuple[Optional[int], Optional[Path]]:
    """Extract source frame info from generation stage.

    Returns:
        tuple of (source_view_id, source_frame_path)
    """
    generation = case.get("generation", {})
    if not generation:
        return None, None

    data = generation.get("data", generation)
    if not isinstance(data, dict):
        return None, None

    source_view_id = data.get("source_view_id")
    source_frame_path = data.get("source_frame_path")

    if source_frame_path:
        scene = case.get("scene", "")
        full_path = replica_root / scene / source_frame_path
        return source_view_id, full_path

    return source_view_id, None


def generate_case_html(case: Dict, case_idx: int, replica_root: Path) -> str:
    """Generate HTML for a single case."""
    scene = case.get("scene", "unknown")
    query = case.get("query", "")
    status = case.get("status", "unknown")

    parse_score = case.get("parse_score")
    selector_score = case.get("selector_score")
    overall_score = case.get("overall_score")

    # Get keyframe info (view_ids or paths) - from SELECTOR
    view_ids, paths = get_keyframe_info(case)

    # Get source frame info - from GENERATOR (GT reference)
    source_view_id, source_frame_path = get_source_frame_info(case, replica_root)

    # Status styling
    status_class = "success" if status == "success" else "failed"

    # Build HTML
    html = f'''
    <div class="case {status_class}" id="case-{case_idx}">
        <div class="case-header">
            <span class="case-id">#{case_idx}</span>
            <span class="scene-badge">{scene}</span>
            <span class="status-badge {status_class}">{status}</span>
        </div>
        <div class="query">"{query}"</div>
        <div class="scores">
            <div class="score">
                <span class="label">Parse</span>
                <span class="value">{format_score(parse_score)}</span>
            </div>
            <div class="score">
                <span class="label">Selector</span>
                <span class="value">{format_score(selector_score)}</span>
            </div>
            <div class="score">
                <span class="label">Overall</span>
                <span class="value">{format_score(overall_score)}</span>
            </div>
        </div>
    '''

    # Add source frame section (GT reference)
    if source_frame_path and source_frame_path.exists():
        source_b64 = image_to_base64(source_frame_path)
        html += f'''
        <div class="source-frame-section">
            <div class="section-label">📍 Ground Truth Source Frame (Generator)</div>
            <div class="source-frame">
                <img src="data:image/jpeg;base64,{source_b64}" alt="source_frame">
                <div class="keyframe-label">view_id={source_view_id} (used to create this case)</div>
            </div>
        </div>
        '''

    # Add selector keyframes section
    html += '''
        <div class="selector-frames-section">
            <div class="section-label">🎯 Selected Keyframes (Selector Output)</div>
            <div class="keyframes">
    '''

    # Check if source_view_id is in selected keyframes
    source_in_selected = source_view_id in view_ids if source_view_id and view_ids else False

    # Add keyframe images - prefer paths if available
    if paths:
        for i, frame_path in enumerate(paths[:5]):
            frame_path = Path(frame_path)
            vid = view_ids[i] if i < len(view_ids) else "?"
            is_source = (vid == source_view_id)
            b64 = image_to_base64(frame_path)
            if b64:
                html += f'''
            <div class="keyframe {'is-source' if is_source else ''}">
                <img src="data:image/jpeg;base64,{b64}" alt="keyframe_{i}">
                <div class="keyframe-label">idx={vid} {'✓ matches GT' if is_source else ''}</div>
            </div>
                '''
            else:
                html += f'''
            <div class="keyframe missing">
                <div class="placeholder">Image not found<br>{frame_path.name}</div>
                <div class="keyframe-label">idx={vid}</div>
            </div>
                '''
    elif view_ids:
        for i, vid in enumerate(view_ids[:5]):  # Max 5 keyframes
            frame_path = get_frame_path(replica_root, scene, vid)
            is_source = (vid == source_view_id)
            b64 = image_to_base64(frame_path)
            if b64:
                html += f'''
            <div class="keyframe {'is-source' if is_source else ''}">
                <img src="data:image/jpeg;base64,{b64}" alt="view_{vid}">
                <div class="keyframe-label">view_id={vid} {'✓ matches GT' if is_source else ''}</div>
            </div>
                '''
            else:
                html += f'''
            <div class="keyframe missing">
                <div class="placeholder">Image not found</div>
                <div class="keyframe-label">view_id={vid}</div>
            </div>
                '''
    else:
        html += '<div class="no-keyframes">No keyframes selected</div>'

    # Add match indicator - only show when matched
    if source_view_id is not None and source_in_selected:
        html += '<div class="match-indicator match">✓ Selector found the GT source frame</div>'

    html += '''
            </div>
        </div>
    '''

    # Add failure reason if failed
    if status != "success":
        failure_stage = case.get("failure_stage", "unknown")
        failure_reason = case.get("failure_reason", "unknown")
        html += f'''
        <div class="failure-info">
            <strong>Failure Stage:</strong> {failure_stage}<br>
            <strong>Reason:</strong> {failure_reason}
        </div>
        '''

    # Add evaluation details if available
    evaluation = case.get("evaluation", {})
    if evaluation and isinstance(evaluation, dict):
        eval_data = evaluation.get("data", {})
        if eval_data and isinstance(eval_data, dict):
            selector_eval = eval_data.get("selector_eval", {})
            if selector_eval:
                reasoning = selector_eval.get("reasoning", "")
                can_answer = selector_eval.get("can_answer_query", None)
                issues = selector_eval.get("issues", [])

                html += f'''
        <div class="eval-details">
            <div class="eval-item">
                <strong>Can Answer Query:</strong> {can_answer}
            </div>
            <div class="eval-item">
                <strong>Reasoning:</strong> {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}
            </div>
        '''
                if issues:
                    html += '<div class="eval-item"><strong>Issues:</strong><ul>'
                    for issue in issues[:3]:
                        html += f'<li>{issue}</li>'
                    html += '</ul></div>'
                html += '</div>'

    html += '''
        </div>
    </div>
    '''
    return html


def format_score(score) -> str:
    """Format score for display."""
    if score is None:
        return "-"
    return f"{score:.2f}"


def generate_summary_stats(report: Dict, results: List[Dict]) -> str:
    """Generate summary statistics HTML."""
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    by_scene = report.get("by_scene", {})

    html = '''
    <div class="summary-section">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card failed">
                <div class="stat-value">{}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.1%}</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <h3>Average Scores</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">Parse Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">Selector Score</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">Overall Score</div>
            </div>
        </div>

        <h3>By Scene</h3>
        <table class="scene-table">
            <thead>
                <tr>
                    <th>Scene</th>
                    <th>Total</th>
                    <th>Success</th>
                    <th>Rate</th>
                    <th>Avg Score</th>
                </tr>
            </thead>
            <tbody>
    '''.format(
        summary.get("total_cases", 0),
        summary.get("successful", 0),
        summary.get("failed", 0),
        summary.get("success_rate", 0),
        scores.get("avg_parse_score", 0),
        scores.get("avg_selector_score", 0),
        scores.get("avg_overall_score", 0),
    )

    for scene, data in sorted(by_scene.items()):
        total = data.get("total", 0)
        success = data.get("success", 0)
        rate = success / total if total > 0 else 0
        avg_score = data.get("avg_score", 0)
        score_class = "low-score" if avg_score < 5.5 else ""

        html += f'''
            <tr class="{score_class}">
                <td>{scene}</td>
                <td>{total}</td>
                <td>{success}</td>
                <td>{rate:.1%}</td>
                <td>{avg_score:.2f}</td>
            </tr>
        '''

    html += '''
            </tbody>
        </table>
    </div>
    '''
    return html


def generate_failed_cases_section(results: List[Dict], replica_root: Path) -> str:
    """Generate dedicated section for failed cases."""
    failed_cases = [(i, c) for i, c in enumerate(results) if c.get("status") != "success"]

    if not failed_cases:
        return '<div class="no-failures">No failed cases!</div>'

    html = f'''
    <div class="failed-section">
        <h2>Failed Cases Analysis ({len(failed_cases)} cases)</h2>
    '''

    for case_idx, case in failed_cases:
        html += generate_case_html(case, case_idx, replica_root)

    html += '</div>'
    return html


def generate_chart_image(report: Dict, output_path: Path) -> Optional[str]:
    """Generate score distribution chart and return base64."""
    if not HAS_MATPLOTLIB:
        return None

    by_scene = report.get("by_scene", {})
    if not by_scene:
        return None

    scenes = list(sorted(by_scene.keys()))
    scores = [by_scene[s].get("avg_score", 0) for s in scenes]
    success_rates = [by_scene[s].get("success", 0) / by_scene[s].get("total", 1) for s in scenes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score bar chart
    colors = ['#e74c3c' if s < 5.5 else '#3498db' for s in scores]
    bars = ax1.bar(scenes, scores, color=colors)
    ax1.axhline(y=5.5, color='orange', linestyle='--', label='Threshold (5.5)')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average Score by Scene')
    ax1.set_ylim(0, 10)
    ax1.legend()
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)

    # Success rate bar chart
    colors2 = ['#e74c3c' if r < 0.8 else '#2ecc71' for r in success_rates]
    bars2 = ax2.bar(scenes, [r * 100 for r in success_rates], color=colors2)
    ax2.axhline(y=80, color='orange', linestyle='--', label='Threshold (80%)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by Scene')
    ax2.set_ylim(0, 105)
    ax2.legend()
    for bar, rate in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return image_to_base64(output_path)


def generate_html_report(
    report: Dict,
    results: List[Dict],
    replica_root: Path,
    output_path: Path,
    chart_b64: Optional[str] = None
):
    """Generate complete HTML report."""

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluation Results Report</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }

        .summary-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card.success { background: #d5f4e6; }
        .stat-card.failed { background: #fadbd8; }
        .stat-card.highlight { background: #d4e6f1; }
        .stat-value { font-size: 32px; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }

        .scene-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .scene-table th, .scene-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .scene-table th { background: #34495e; color: white; }
        .scene-table tr:hover { background: #f5f5f5; }
        .scene-table .low-score { background: #fadbd8; }

        .chart-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        .chart-section img { max-width: 100%; }

        .filter-section {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-section select, .filter-section input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .filter-btn {
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .filter-btn:hover { background: #2980b9; }

        .cases-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }

        .case {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }
        .case.failed { border-left-color: #e74c3c; }
        .case.success { border-left-color: #2ecc71; }

        .case-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .case-id { font-weight: bold; color: #7f8c8d; }
        .scene-badge {
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .status-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .status-badge.success { background: #2ecc71; color: white; }
        .status-badge.failed { background: #e74c3c; color: white; }

        .query {
            font-size: 16px;
            color: #2c3e50;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            font-style: italic;
        }

        .scores {
            display: flex;
            gap: 20px;
            margin: 15px 0;
        }
        .score {
            text-align: center;
        }
        .score .label {
            font-size: 12px;
            color: #7f8c8d;
            display: block;
        }
        .score .value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }

        .keyframes {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 10px 0;
        }
        .keyframe {
            flex-shrink: 0;
            text-align: center;
        }
        .keyframe img {
            width: 200px;
            height: 150px;
            object-fit: cover;
            border-radius: 4px;
            border: 2px solid #ddd;
        }
        .keyframe.is-source img {
            border: 3px solid #27ae60;
            box-shadow: 0 0 10px rgba(39, 174, 96, 0.4);
        }
        .keyframe-label {
            font-size: 11px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .keyframe.is-source .keyframe-label {
            color: #27ae60;
            font-weight: bold;
        }
        .keyframe.missing .placeholder {
            width: 200px;
            height: 150px;
            background: #ecf0f1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            color: #95a5a6;
        }
        .no-keyframes {
            color: #95a5a6;
            font-style: italic;
            padding: 20px;
        }

        .source-frame-section {
            margin: 15px 0;
            padding: 15px;
            background: #fff9e6;
            border: 2px solid #f1c40f;
            border-radius: 8px;
        }
        .source-frame-section .section-label {
            font-weight: bold;
            color: #d68910;
            margin-bottom: 10px;
            font-size: 13px;
        }
        .source-frame {
            display: inline-block;
            text-align: center;
        }
        .source-frame img {
            width: 250px;
            height: 188px;
            object-fit: cover;
            border-radius: 4px;
            border: 3px solid #f1c40f;
        }

        .selector-frames-section {
            margin: 15px 0;
            padding: 15px;
            background: #e8f8f5;
            border: 2px solid #1abc9c;
            border-radius: 8px;
        }
        .selector-frames-section .section-label {
            font-weight: bold;
            color: #16a085;
            margin-bottom: 10px;
            font-size: 13px;
        }

        .match-indicator {
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .match-indicator.match {
            background: #d5f4e6;
            color: #27ae60;
        }
        .match-indicator.no-match {
            background: #fadbd8;
            color: #e74c3c;
        }

        .failure-info {
            background: #fadbd8;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 13px;
        }

        .eval-details {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 13px;
        }
        .eval-item { margin: 5px 0; }
        .eval-item ul { margin: 5px 0 5px 20px; padding: 0; }

        .failed-section {
            background: #fdf2f2;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        .nav-links {
            position: fixed;
            right: 20px;
            top: 100px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .nav-links a {
            display: block;
            padding: 5px 0;
            color: #3498db;
            text-decoration: none;
        }
        .nav-links a:hover { text-decoration: underline; }

        @media (max-width: 768px) {
            .cases-grid { grid-template-columns: 1fr; }
            .nav-links { display: none; }
        }
    </style>
</head>
<body>
    <h1>Evaluation Results Report</h1>
    <p>Generated from: ''' + str(report.get("config", {}).get("run_id", "unknown")) + '''</p>

    <nav class="nav-links">
        <a href="#summary">Summary</a>
        <a href="#charts">Charts</a>
        <a href="#failed">Failed Cases</a>
        <a href="#all-cases">All Cases</a>
    </nav>

    <section id="summary">
    ''' + generate_summary_stats(report, results) + '''
    </section>
    '''

    # Add chart if available
    if chart_b64:
        html += f'''
    <section id="charts" class="chart-section">
        <h2>Score Distribution</h2>
        <img src="data:image/png;base64,{chart_b64}" alt="Score Distribution Chart">
    </section>
        '''

    # Add failed cases section
    html += f'''
    <section id="failed">
    {generate_failed_cases_section(results, replica_root)}
    </section>
    '''

    # Filter section
    scenes = sorted(set(c.get("scene", "") for c in results))
    html += '''
    <section id="all-cases">
        <h2>All Cases</h2>
        <div class="filter-section">
            <label>Scene:</label>
            <select id="scene-filter" onchange="filterCases()">
                <option value="">All</option>
    '''
    for scene in scenes:
        html += f'<option value="{scene}">{scene}</option>'

    html += '''
            </select>
            <label>Status:</label>
            <select id="status-filter" onchange="filterCases()">
                <option value="">All</option>
                <option value="success">Success</option>
                <option value="failed">Failed</option>
            </select>
            <label>Min Score:</label>
            <input type="number" id="min-score" min="0" max="10" step="0.5" placeholder="0" onchange="filterCases()">
        </div>

        <div class="cases-grid" id="cases-container">
    '''

    # Add all cases
    for i, case in enumerate(results):
        html += generate_case_html(case, i, replica_root)

    html += '''
        </div>
    </section>

    <script>
        function filterCases() {
            const sceneFilter = document.getElementById('scene-filter').value;
            const statusFilter = document.getElementById('status-filter').value;
            const minScore = parseFloat(document.getElementById('min-score').value) || 0;

            document.querySelectorAll('.case').forEach(caseEl => {
                const scene = caseEl.querySelector('.scene-badge')?.textContent || '';
                const status = caseEl.classList.contains('success') ? 'success' : 'failed';
                const scoreEl = caseEl.querySelectorAll('.score .value')[2];
                const score = scoreEl ? parseFloat(scoreEl.textContent) || 0 : 0;

                let show = true;
                if (sceneFilter && scene !== sceneFilter) show = false;
                if (statusFilter && status !== statusFilter) show = false;
                if (score < minScore && scoreEl?.textContent !== '-') show = false;

                caseEl.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
    '''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--eval_dir", type=str, required=True, help="Evaluation output directory")
    parser.add_argument("--replica_root", type=str, default="/Users/bytedance/Replica", help="Replica dataset root")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: eval_dir/report.html)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    replica_root = Path(args.replica_root)
    output_path = Path(args.output) if args.output else eval_dir / "report.html"

    print(f"Loading results from: {eval_dir}")
    report, results = load_results(eval_dir)

    print(f"Loaded {len(results)} cases")

    # Generate chart
    chart_b64 = None
    if HAS_MATPLOTLIB:
        chart_path = eval_dir / "charts.png"
        print("Generating charts...")
        chart_b64 = generate_chart_image(report, chart_path)
    else:
        print("matplotlib not available, skipping charts")

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(report, results, replica_root, output_path, chart_b64)

    print(f"\nDone! Open the report in a browser:")
    print(f"  open {output_path}")


if __name__ == "__main__":
    main()
