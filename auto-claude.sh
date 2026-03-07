#!/bin/bash
# auto-claude.sh - Automated iterative improvement for Keyframe Selector
#
# This script runs Claude Code in a loop to iteratively improve the
# keyframe selector based on TODO.md tasks.
#
# Usage:
#   chmod +x auto-claude.sh
#   ./auto-claude.sh

set -e

# Configuration
MAX_ITERATIONS=20           # Maximum iterations
SLEEP_BETWEEN=5             # Seconds between iterations
LOG_FILE="results/auto-claude.log"
TODO_FILE="TODO.md"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize
mkdir -p results
echo "=== Auto-Claude Started at $(date) ===" | tee -a "$LOG_FILE"

# Check if TODO.md exists
if [ ! -f "$TODO_FILE" ]; then
    echo -e "${RED}Error: $TODO_FILE not found${NC}"
    exit 1
fi

# Check if there are pending tasks
has_pending_tasks() {
    grep -q "\- \[ \]" "$TODO_FILE"
}

# Get current pass rate from latest test results
get_pass_rate() {
    if [ -f "/Users/bytedance/Replica/room0/query_visualizations/test_results.json" ]; then
        .venv/bin/python -c "
import json
with open('/Users/bytedance/Replica/room0/query_visualizations/test_results.json') as f:
    data = json.load(f)
passed = sum(1 for r in data if r.get('matched_objects'))
total = len(data)
print(f'{passed}/{total}')
"
    else
        echo "N/A"
    fi
}

# Main loop
iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    echo ""
    echo -e "${YELLOW}=== Iteration $iteration / $MAX_ITERATIONS ===${NC}" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"

    # Check if there are remaining tasks
    if ! has_pending_tasks; then
        echo -e "${GREEN}All tasks completed!${NC}" | tee -a "$LOG_FILE"
        break
    fi

    # Show current pass rate
    current_rate=$(get_pass_rate)
    echo "Current pass rate: $current_rate" | tee -a "$LOG_FILE"

    # Execute Claude Code
    claude -p "
你正在执行 Keyframe Selector 改进任务的自动化迭代。

请查看 TODO.md，选择第一个未完成的任务（标记为 - [ ]）。

执行流程：
1. 阅读当前任务描述
2. 执行任务（分析/实现/测试）
3. 将结果记录到 results/ 目录下的对应文件
4. 如果是代码改动，运行 e2e 测试验证效果：
   REPLICA_ROOT=/Users/bytedance/Replica SCENE_NAME=room0 .venv/bin/python -m conceptgraph.query_scene.examples.e2e_query_test
5. 更新 TODO.md：
   - 将完成的任务标记为 - [x]
   - 移动到 Completed 区域
   - 更新 Results Log 表格
6. 如果发现新问题，添加新任务到 Pending

重要：
- 每次只完成一个任务
- 必须记录改动前后的 pass rate 对比
- 目标：通过率达到 98%+ (至少 58/59)
- 当前 baseline: 57/59 (96.6%)
- 使用 .venv/bin/python 运行 Python

当前 TODO.md 内容：
$(cat TODO.md)
" --dangerously-skip-permissions 2>&1 | tee -a "$LOG_FILE"

    exit_code=$?

    # Check Claude exit status
    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}Claude exited with error code $exit_code${NC}" | tee -a "$LOG_FILE"
        echo "Waiting 30 seconds before retry..." | tee -a "$LOG_FILE"
        sleep 30
        continue
    fi

    # Get new pass rate
    new_rate=$(get_pass_rate)
    echo "New pass rate: $new_rate" | tee -a "$LOG_FILE"

    # Auto-commit if there are changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "Committing changes..." | tee -a "$LOG_FILE"
        git add -A
        git commit -m "auto(selector): iteration $iteration - pass rate $new_rate

Changes from automated improvement iteration.
See results/auto-claude.log for details.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    fi

    # Check if we've reached the target
    passed=$(echo "$new_rate" | cut -d'/' -f1)
    total=$(echo "$new_rate" | cut -d'/' -f2)
    if [ "$passed" = "$total" ] || [ "$passed" -ge 58 ]; then
        echo -e "${GREEN}Target achieved! Pass rate: $new_rate${NC}" | tee -a "$LOG_FILE"
        break
    fi

    echo "Waiting ${SLEEP_BETWEEN} seconds..." | tee -a "$LOG_FILE"
    sleep $SLEEP_BETWEEN
done

echo ""
echo "=== Auto-Claude Finished at $(date) ===" | tee -a "$LOG_FILE"
echo "Final pass rate: $(get_pass_rate)" | tee -a "$LOG_FILE"
echo "Total iterations: $iteration" | tee -a "$LOG_FILE"
