# Repository Guidelines

## Memory Bootstrap (Required)
- 本仓库的长期背景知识存放在 `memory/`。
- 每次新对话/新任务开始时，必须先按顺序读取：
1. `memory/project_context.md`
2. `memory/query_scene_knowledge.md`
3. `memory/bash_scripts_index.md`
- 这些文件用于快速拉取项目背景；若与当前源码冲突，以源码为准。
- 当你修改了 query scene 行为或 pipeline 脚本后，必须同步更新对应 `memory/*.md`。

## Memory References
- 索引入口：`memory/README.md`
- 项目背景：`memory/project_context.md`
- Query Scene 知识：`memory/query_scene_knowledge.md`
- Bash 脚本索引：`memory/bash_scripts_index.md`

## Fast Commands
- Query Scene 解析冒烟：
  - `python -m conceptgraph.query_scene.examples.simple_parse_test`
  - `python -m conceptgraph.query_scene.examples.test_nested_query_parsing --llm_model gpt-5.2-2025-12-11`
- Query Scene 端到端：
  - `bash bashes/run_e2e_query_test.sh`
- 查询关键帧：
  - `bash bashes/6b_build_visibility_index.sh room0`
  - `bash bashes/7b_query_scene.sh room0 "pillow on the sofa" 3`

## Working Conventions
- Python 4 空格缩进，命名采用 `snake_case` / `PascalCase`。
- 运行配置统一走 `env_vars.bash`，避免硬编码路径。
- 提交优先按子模块拆分：`query_scene`、`slam`、`scenegraph`、`bashes`。
