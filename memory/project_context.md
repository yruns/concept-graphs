# Project Context

## Repository 结构（主干）
- `conceptgraph/`: 主代码。
- `bashes/`: 端到端流程脚本（分阶段）。
- `docs/`: 说明文档与执行记录。
- `Grounded-Segment-Anything/`, `gradslam/`, `chamferdist/`: 依赖代码树（修改需谨慎）。

## 环境与配置
- 推荐 Python 3.10 + conda 环境 `conceptgraph`。
- 本地路径/密钥通过 `env_vars.bash` 管理（由 `env_vars.bash.template` 复制）。
- 常见变量：`REPLICA_ROOT`, `GSA_PATH`, `LLM_BASE_URL`, `LLM_MODEL`。

## 贡献约定（简）
- 命名：`snake_case`（函数/文件）、`PascalCase`（类）。
- 代码风格：4 空格缩进，新增公共函数尽量加类型标注。
- 提交建议按模块拆分：`query_scene`、`slam`、`scenegraph`、`bashes`。
