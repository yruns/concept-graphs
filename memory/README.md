# Memory Index

本目录用于沉淀本仓库的稳定项目背景，供每次新对话启动时快速加载。

## 加载顺序（建议）
1. `memory/project_context.md`
2. `memory/room0_artifact_lineage.md`
3. `memory/query_scene_knowledge.md`
4. `memory/bash_scripts_index.md`
5. `memory/research_direction.md`

## 使用规则
- 这些文件是“背景知识”，不是最终真相。
- 若与源码、脚本参数或最新提交冲突，以源码和最新改动为准。
- 修改 pipeline 或 query 行为后，请同步更新对应 memory 文件。
- 若任务直接涉及 Stage 2 agent，请在完成上述 memory 加载后继续阅读 `docs/stage2_agent_handoff.md`。
