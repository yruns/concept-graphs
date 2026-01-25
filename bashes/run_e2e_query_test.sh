#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export http_proxy=socks5h://localhost:18080
export https_proxy=socks5h://localhost:18080
export ALL_PROXY=socks5h://localhost:18080
export all_proxy=socks5h://localhost:18080

if [ -f "${HOME}/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

# 写入到log文件和stdout中
python conceptgraph/query_scene/examples/e2e_query_test.py > >(tee docs/e2e_query_test_run.log) 2>&1
