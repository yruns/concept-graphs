#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

# 初始化 conda
# eval "$(conda shell.bash hook)"
conda activate conceptgraph

# 写入到log文件和stdout中
python conceptgraph/query_scene/examples/e2e_query_test.py > >(tee docs/e2e_query_test_run.log) 2>&1
