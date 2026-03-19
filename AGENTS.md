# Repository Guidelines

## Memory Bootstrap (Required)
- 本仓库的长期背景知识存放在 `memory/`。
- 每次新对话/新任务开始时，必须先按顺序读取：
1. `memory/project_context.md`
2. `memory/room0_artifact_lineage.md`
3. `memory/query_scene_knowledge.md`
4. `memory/bash_scripts_index.md`
- 这些文件用于快速拉取项目背景；若与当前源码冲突，以源码为准。
- 当你修改了 query scene 行为或 pipeline 脚本后，必须同步更新对应 `memory/*.md`。

## Memory References
- 索引入口：`memory/README.md`
- 项目背景：`memory/project_context.md`
- `room0` 产物溯源：`memory/room0_artifact_lineage.md`
- Query Scene 知识：`memory/query_scene_knowledge.md`
- Bash 脚本索引：`memory/bash_scripts_index.md`

## Fast Commands
- 下列命令中的 `python`：Darwin 请替换为 `.venv/bin/python`；Linux 请先激活 `conceptgraph` conda 环境。
- Query Scene 单元回归：
  - `python -m pytest conceptgraph/query_scene/tests/test_keyframe_selector_hypothesis.py conceptgraph/query_scene/tests/test_query_parser_hypothesis.py conceptgraph/query_scene/tests/test_hypothesis_output_schema.py conceptgraph/query_scene/tests/test_open_world_sample_builder.py -q`
- Query Scene 单条查询：
  - `REPLICA_ROOT=/abs/path/to/Replica python -m conceptgraph.query_scene.examples.query_keyframes --scene_path "$REPLICA_ROOT/room0" --query "pillow on the sofa" --k 3 --llm_model gpt-5.2-2025-12-11`
- Query Scene 端到端：
  - `REPLICA_ROOT=/abs/path/to/Replica SCENE_NAME=room0 python -m conceptgraph.query_scene.examples.e2e_query_test`
- 查询关键帧预处理：
  - `REPLICA_ROOT=/abs/path/to/Replica bash bashes/6b_build_visibility_index.sh room0`
  - `REPLICA_ROOT=/abs/path/to/Replica bash bashes/7b_query_scene.sh room0 "pillow on the sofa" 3`

## Python Environment (Required)
- **检测系统**：先执行 `uname -s` 判断当前 OS
- **macOS (Darwin)**：使用项目根目录的 `.venv`
  ```bash
  .venv/bin/python -m conceptgraph.query_scene.examples.e2e_query_test
  ```
- **Linux**：使用 conda 环境 `conceptgraph`
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
  python -m conceptgraph.query_scene.examples.e2e_query_test
  ```

## Query Scene Notes
- 当前 query parser 的对外协议是 `HypothesisOutputV1`，主入口是 `KeyframeSelector.select_keyframes_v2()`；返回 metadata 中版本号仍写作 `v3`。
- `KeyframeSelector.parse_query_hypotheses()` 默认会生成 `scene_path/bev/scene_bev_<hash>.png` 作为多模态输入；该图使用 `ReplicaDefaultBEVConfig`，是 mesh-only、无 object marker/label 的 BEV。
- `conceptgraph/query_scene/examples/simple_parse_test.py` 以及 `conceptgraph/query_scene/examples/test_nested_query_parsing.py` 中的 `SimpleQueryParser` 分支已过时；当前源码里不存在 `SimpleQueryParser`，不要把它们当作冒烟命令。
- `bashes/6b_build_visibility_index.sh`、`bashes/7b_query_scene.sh`、`bashes/run_e2e_query_test.sh` 目前仍是 Linux/conda 风格脚本，没有统一适配 `.venv`。
- `bashes/7b_query_scene.sh` 的默认 `REPLICA_ROOT` 是 `$HOME/Datasets/Replica/Replica`，与 `6b`/`run_full_detect_pipeline_to_6b.sh` 的 `$HOME/Datasets/Replica` 不一致；运行 query scene bash 脚本前请显式导出 `REPLICA_ROOT`。

## Working Conventions
- Python 4 空格缩进，命名采用 `snake_case` / `PascalCase`。
- 运行配置统一走 `env_vars.bash`，避免硬编码路径。
- 提交优先按子模块拆分：`query_scene`、`slam`、`scenegraph`、`bashes`。
