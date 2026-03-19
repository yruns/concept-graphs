#!/bin/bash
# auto-claude.sh - Autonomous Research Loop with Worker + Reviewer Pattern
#
# Architecture:
#   1. Task Selector: Picks next pending task from TASKS.md
#   2. Worker Agent: Implements the task (code/research)
#   3. Reviewer Agent: Reviews changes, provides critique
#   4. Research Explorer: Periodically searches for new academic insights
#   5. Task Updater: Marks tasks complete, logs progress
#
# Usage:
#   chmod +x auto-claude.sh
#   ./auto-claude.sh              # Interactive mode
#   ./auto-claude.sh --daemon     # Background daemon mode
#   ./auto-claude.sh --dry-run    # Show what would happen
#
# Configuration via environment:
#   MAX_ITERATIONS=50              # Maximum iterations (default: 50)
#   RESEARCH_INTERVAL=5            # Run research explorer every N iterations
#   REVIEW_THRESHOLD=medium        # When to trigger review: always/medium/major
#   MODEL=claude-opus-4-5          # Model to use

set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
RESEARCH_INTERVAL="${RESEARCH_INTERVAL:-5}"
REVIEW_THRESHOLD="${REVIEW_THRESHOLD:-medium}"
MODEL="${MODEL:-claude-opus-4-5}"
SLEEP_BETWEEN=3
DRY_RUN=false
DAEMON_MODE=false

# File paths
TASKS_FILE="TASKS.md"
TODO_FILE="TODO.md"
LOG_DIR="results/auto-claude"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOG_DIR/session_$TIMESTAMP.log"
WORKER_LOG="$LOG_DIR/worker_$TIMESTAMP.log"
REVIEWER_LOG="$LOG_DIR/reviewer_$TIMESTAMP.log"
RESEARCH_LOG="$LOG_DIR/research_$TIMESTAMP.log"
STATE_FILE="$LOG_DIR/.state.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ════════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ════════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--daemon] [--dry-run] [--max-iterations N]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

mkdir -p "$LOG_DIR"

log() {
    local level="$1"
    local msg="$2"
    local color=""
    case $level in
        INFO)  color="$BLUE" ;;
        WARN)  color="$YELLOW" ;;
        ERROR) color="$RED" ;;
        OK)    color="$GREEN" ;;
        WORK)  color="$CYAN" ;;
        REV)   color="$MAGENTA" ;;
    esac
    echo -e "${color}[$level]${NC} $(date '+%H:%M:%S') $msg" | tee -a "$SESSION_LOG"
}

check_prerequisites() {
    if [ ! -f "$TASKS_FILE" ]; then
        log ERROR "TASKS.md not found. Please create it first."
        exit 1
    fi
    if ! command -v ttadk &> /dev/null; then
        log ERROR "ttadk command not found. Please install it."
        exit 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# TASK MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

# Get next pending task (returns TASK-XXX or empty)
get_next_task() {
    # Find first "- [ ] TASK-" line and extract task ID
    grep -m1 '^\- \[ \] TASK-[0-9]\+' "$TASKS_FILE" | sed -E 's/.*TASK-([0-9]+).*/TASK-\1/' || echo ""
}

# Get task details
get_task_details() {
    local task_id="$1"
    # Extract task block: from task line until next task, section header, or separator
    awk "
        /$task_id/ { found=1 }
        found && /^- \[/ && !/$task_id/ { exit }
        found && /^##/ { exit }
        found && /^---/ { exit }
        found { print }
    " "$TASKS_FILE"
}

# Check if task has unmet dependencies
check_dependencies() {
    local task_id="$1"
    # Get 10 lines after this task and find Depends line
    local deps=$(awk "/$task_id/,/^- \[/{if(/Depends:/)print}" "$TASKS_FILE" | head -1 | sed 's/.*Depends: *//')

    if [ -z "$deps" ] || [ "$deps" = "None" ] || [[ "$deps" =~ ^[[:space:]]*$ ]]; then
        return 0  # No dependencies
    fi

    # Check each dependency
    for dep in $(echo "$deps" | tr ',' ' '); do
        dep=$(echo "$dep" | xargs)  # Trim whitespace
        # Skip empty deps or self-reference
        [ -z "$dep" ] && continue
        [ "$dep" = "$task_id" ] && continue

        # Check if dependency is incomplete (marked with [ ])
        if grep -q "^- \[ \] $dep" "$TASKS_FILE"; then
            log WARN "Task $task_id blocked by incomplete dependency: $dep"
            return 1
        fi
    done
    return 0
}

# Mark task as in progress
mark_in_progress() {
    local task_id="$1"
    sed -i '' "s/^\- \[ \] $task_id/- [~] $task_id/" "$TASKS_FILE"
}

# Mark task as complete
mark_complete() {
    local task_id="$1"
    sed -i '' "s/^\- \[.\] $task_id/- [x] $task_id/" "$TASKS_FILE"
}

# Mark task as needs review
mark_needs_review() {
    local task_id="$1"
    sed -i '' "s/^\- \[.\] $task_id/- [?] $task_id/" "$TASKS_FILE"
}

# ════════════════════════════════════════════════════════════════════════════════
# CLAUDE EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

run_claude() {
    local prompt="$1"
    local log_file="$2"
    local timeout="${3:-600}"  # 10 minute default timeout

    if [ "$DRY_RUN" = true ]; then
        log INFO "[DRY-RUN] Would execute Claude with prompt (${#prompt} chars)"
        echo "$prompt" | head -c 500
        echo "..."
        return 0
    fi

    # Execute with timeout
    echo "$prompt" | timeout "$timeout" ttadk code --model "$MODEL" \
        -a "--dangerously-skip-permissions --print --output-format stream-json" \
        2>&1 | tee -a "$log_file"

    return ${PIPESTATUS[1]}
}

# ════════════════════════════════════════════════════════════════════════════════
# WORKER AGENT
# ════════════════════════════════════════════════════════════════════════════════

run_worker() {
    local task_id="$1"
    local task_details="$2"

    log WORK "Worker starting task: $task_id"

    local prompt=$(cat << WORKER_PROMPT
# 🔧 Worker Agent: Execute Research Task

You are the **Worker Agent** in an autonomous research loop for the **Two-Stage 3D Scene Understanding** project.

## Your Current Task
\`\`\`
$task_details
\`\`\`

## Project Context

This project implements a two-stage framework:
- **Stage 1**: Task-conditioned keyframe retrieval from 3D scene graphs
- **Stage 2**: VLM agentic reasoning over retrieved visual evidence

Academic innovation points:
1. Adaptive Evidence Acquisition (VLM decides when to request more evidence)
2. Symbolic-to-Visual Repair (Stage 2 validates/corrects Stage 1 hypotheses)
3. Evidence-Grounded Uncertainty (explicit uncertainty output)
4. Unified Multi-Task Policy (QA, grounding, navigation, manipulation)

## Execution Guidelines

1. **Read first**: Review relevant existing code before writing
2. **Small changes**: Make incremental, testable changes
3. **Test everything**: Run unit tests after each change
4. **Document**: Add docstrings and type hints
5. **Format**: Use \`.venv/bin/python -m black <file>\` to format

## Commands Reference
- Run tests: \`.venv/bin/python -m pytest <test_file> -v\`
- Format code: \`.venv/bin/python -m black <file>\`
- Stage 2 agent tests: \`.venv/bin/python -m pytest conceptgraph/agents/tests/test_stage2_deep_agent.py -q\`
- Benchmark tests: \`.venv/bin/python -m pytest conceptgraph/benchmarks/tests/ -v\`

## Important Files
- Research direction: \`memory/research_direction.md\`
- TODO overview: \`TODO.md\`
- Task tracking: \`TASKS.md\`
- Stage 2 design: \`docs/stage2_vlm_agent_design.md\`

## Output Requirements

When done, output a brief summary:
\`\`\`
## Task $task_id Summary
- Status: DONE | PARTIAL | BLOCKED
- Files changed: <list>
- Tests: <passed/total>
- Notes: <any issues or follow-ups>
\`\`\`

Now execute task $task_id. Focus on quality and correctness.
WORKER_PROMPT
)

    run_claude "$prompt" "$WORKER_LOG" 900
    return $?
}

# ════════════════════════════════════════════════════════════════════════════════
# REVIEWER AGENT
# ════════════════════════════════════════════════════════════════════════════════

run_reviewer() {
    local task_id="$1"

    log REV "Reviewer analyzing changes for: $task_id"

    # Get git diff
    local git_diff=$(git diff --cached --stat 2>/dev/null || git diff --stat 2>/dev/null || echo "No changes detected")
    local git_diff_full=$(git diff --cached 2>/dev/null || git diff 2>/dev/null | head -500)

    local prompt=$(cat << REVIEWER_PROMPT
# 🔍 Reviewer Agent: Code Review & Academic Alignment

You are the **Reviewer Agent** in an autonomous research loop. Your job is to:
1. Review code changes for quality and correctness
2. Check alignment with research goals
3. Suggest improvements

## Changes to Review (Task: $task_id)

### Git Diff Summary
\`\`\`
$git_diff
\`\`\`

### Detailed Changes (truncated)
\`\`\`diff
$git_diff_full
\`\`\`

## Review Criteria

### Code Quality (Score 1-10)
- [ ] Follows Python best practices
- [ ] Has proper type hints
- [ ] Has docstrings
- [ ] No hardcoded values
- [ ] Error handling present
- [ ] Tests included

### Research Alignment (Score 1-10)
- [ ] Supports academic innovation claims
- [ ] Follows two-stage paradigm correctly
- [ ] Consistent with Stage 2 agent design
- [ ] Enables benchmark evaluation

### Suggestions
- List specific improvements
- Flag any issues that need fixing
- Note if task should be marked complete or needs more work

## Output Format

\`\`\`json
{
  "task_id": "$task_id",
  "code_quality_score": <1-10>,
  "research_alignment_score": <1-10>,
  "overall_verdict": "APPROVE" | "REQUEST_CHANGES" | "NEEDS_DISCUSSION",
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "should_commit": true | false
}
\`\`\`

Be constructive but rigorous. Our goal is publishable research.
REVIEWER_PROMPT
)

    run_claude "$prompt" "$REVIEWER_LOG" 300
    return $?
}

# ════════════════════════════════════════════════════════════════════════════════
# RESEARCH EXPLORER AGENT
# ════════════════════════════════════════════════════════════════════════════════

run_research_explorer() {
    log INFO "🔬 Research Explorer: Searching for new academic insights..."

    local prompt=$(cat << RESEARCH_PROMPT
# 🎓 Research Explorer Agent: Academic Insight Discovery

You are the **Research Explorer** in an autonomous research loop. Your mission:
1. Search for latest papers relevant to our research
2. Identify potential academic contributions we're missing
3. Find comparison baselines and benchmarks

## Our Research Direction

**Title**: Two-Stage Framework for 3D Scene Understanding with Evidence-Seeking VLM Agents

**Core Claims**:
1. VLM agents that dynamically request evidence outperform one-shot baselines
2. Visual repair can correct scene graph detection failures
3. Explicit uncertainty reduces hallucination
4. Unified multi-task policy (QA, grounding, navigation, manipulation)

**Target Venues**: CVPR, ICCV, NeurIPS, ICLR, ECCV 2025-2026

## Search Tasks

Use the Agent tool with subagent_type="Explore" or WebSearch to:

1. **Competitor Analysis**: Search for recent papers on:
   - "3D scene graph + LLM reasoning 2024 2025"
   - "VLM agent embodied QA"
   - "iterative visual reasoning benchmark"

2. **Benchmark Updates**: Check for:
   - New EQA/VQA benchmarks released in 2025
   - Updated SOTA results on OpenEQA, SQA3D, ScanRefer
   - New evaluation metrics being adopted

3. **Method Gaps**: Look for:
   - What existing methods don't address
   - Common failure modes in recent papers
   - Under-explored research directions

## Output Format

\`\`\`markdown
## Research Insights Report

### New Related Work
- [Paper Title](url) - Venue 2025
  - Relevance: <how it relates to us>
  - Differentiation: <how we differ>

### Updated Benchmarks
- <benchmark name>: New SOTA is X% by <method>

### Identified Gaps
- Gap 1: <description> → Opportunity: <what we can do>

### Recommended Actions
1. <action item for TASKS.md>
\`\`\`

Search now and report findings.
RESEARCH_PROMPT
)

    run_claude "$prompt" "$RESEARCH_LOG" 600
    return $?
}

# ════════════════════════════════════════════════════════════════════════════════
# GIT OPERATIONS
# ════════════════════════════════════════════════════════════════════════════════

auto_commit() {
    local task_id="$1"
    local commit_type="${2:-feat}"

    if [ -z "$(git status --porcelain)" ]; then
        log INFO "No changes to commit"
        return 0
    fi

    git add -A

    # Generate commit message
    local files_changed=$(git diff --cached --name-only | wc -l | xargs)
    local commit_msg="$commit_type(research): $task_id - automated implementation

Files changed: $files_changed
Session: $TIMESTAMP

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
Co-Authored-By: TTADK <ttadk@bytedance.com>"

    git commit -m "$commit_msg"
    log OK "Committed changes for $task_id"
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════════

main() {
    log INFO "════════════════════════════════════════════════════════════"
    log INFO "  Auto-Claude Research Loop Started"
    log INFO "  Model: $MODEL | Max Iterations: $MAX_ITERATIONS"
    log INFO "  Session: $TIMESTAMP"
    log INFO "════════════════════════════════════════════════════════════"

    check_prerequisites

    local iteration=0
    local consecutive_failures=0
    local max_failures=3

    while [ $iteration -lt $MAX_ITERATIONS ]; do
        iteration=$((iteration + 1))

        echo ""
        log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log INFO "  Iteration $iteration / $MAX_ITERATIONS"
        log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Get next task
        # ─────────────────────────────────────────────────────────────────────
        local task_id=$(get_next_task)

        if [ -z "$task_id" ]; then
            log OK "🎉 All tasks completed! Research loop finished."
            break
        fi

        # Check dependencies
        if ! check_dependencies "$task_id"; then
            log WARN "Skipping $task_id due to unmet dependencies"
            # Mark as blocked and try next
            sed -i '' "s/^\- \[ \] $task_id/- [!] $task_id/" "$TASKS_FILE"
            continue
        fi

        local task_details=$(get_task_details "$task_id")
        log INFO "Selected task: $task_id"

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Run Worker Agent
        # ─────────────────────────────────────────────────────────────────────
        mark_in_progress "$task_id"

        if ! run_worker "$task_id" "$task_details"; then
            log ERROR "Worker failed for $task_id"
            consecutive_failures=$((consecutive_failures + 1))

            if [ $consecutive_failures -ge $max_failures ]; then
                log ERROR "Too many consecutive failures. Stopping."
                exit 1
            fi

            log WARN "Waiting 30s before retry..."
            sleep 30
            mark_needs_review "$task_id"
            continue
        fi

        consecutive_failures=0

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Run Reviewer Agent (if changes exist)
        # ─────────────────────────────────────────────────────────────────────
        if [ -n "$(git status --porcelain)" ]; then
            if ! run_reviewer "$task_id"; then
                log WARN "Reviewer had issues, but continuing..."
            fi

            # Auto-commit (reviewer feedback is informational for now)
            auto_commit "$task_id" "feat"
        fi

        # Mark task complete
        mark_complete "$task_id"
        log OK "✓ Task $task_id completed"

        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Periodic Research Explorer
        # ─────────────────────────────────────────────────────────────────────
        if [ $((iteration % RESEARCH_INTERVAL)) -eq 0 ]; then
            run_research_explorer || log WARN "Research explorer failed, continuing..."
        fi

        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Brief pause
        # ─────────────────────────────────────────────────────────────────────
        log INFO "Sleeping ${SLEEP_BETWEEN}s before next iteration..."
        sleep $SLEEP_BETWEEN

    done

    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    log INFO "════════════════════════════════════════════════════════════"
    log INFO "  Session Complete"
    log INFO "  Total iterations: $iteration"
    log INFO "  Logs: $LOG_DIR/"
    log INFO "════════════════════════════════════════════════════════════"

    # Show remaining tasks
    local remaining=$(grep -c '^\- \[ \]' "$TASKS_FILE" 2>/dev/null || echo "0")
    local completed=$(grep -c '^\- \[x\]' "$TASKS_FILE" 2>/dev/null || echo "0")
    log INFO "Tasks: $completed completed, $remaining remaining"
}

# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if [ "$DAEMON_MODE" = true ]; then
    log INFO "Starting in daemon mode..."
    nohup "$0" > "$LOG_DIR/daemon.log" 2>&1 &
    echo "Daemon started with PID $!"
    exit 0
fi

main "$@"
