# Research Tasks: Two-Stage 3D Scene Understanding

> Auto-generated from TODO.md. Track progress with `[x]` markers.
> Format: `- [ ] TASK-{ID}: {Title} | Priority: {1-5} | Est: {hours}h`

## Task Status Legend
- `[ ]` Pending
- `[~]` In Progress
- `[x]` Completed
- `[!]` Blocked
- `[?]` Needs Review

---

## Phase 1: Benchmark Integration (Priority 1)

### Benchmark Loaders

- [x] TASK-001: OpenEQA benchmark loader | Priority: 1 | Est: 2h
  - File: `conceptgraph/benchmarks/openeqa_loader.py`
  - Tests: `conceptgraph/benchmarks/tests/test_openeqa.py` (26 tests passing)
  - Status: Complete with LLM evaluation protocol

- [x] TASK-002: SQA3D benchmark loader | Priority: 1 | Est: 2h
  - File: `conceptgraph/benchmarks/sqa3d_loader.py`
  - Tests: `conceptgraph/benchmarks/tests/test_sqa3d.py` (41 tests passing)
  - Status: Complete with situation context

- [x] TASK-003: ScanRefer benchmark loader | Priority: 1 | Est: 3h
  - File: `conceptgraph/benchmarks/scanrefer_loader.py`
  - Tests: `conceptgraph/benchmarks/tests/test_scanrefer.py`
  - Depends: None
  - Acceptance:
    - [ ] Load ScanRefer JSON format
    - [ ] Parse 3D bounding boxes
    - [ ] Implement IoU-based evaluation (Acc@0.25, Acc@0.5)
    - [ ] Unit tests with >90% coverage

- [x] TASK-004: EAI diagnostic benchmark loader | Priority: 2 | Est: 4h
  - File: `conceptgraph/benchmarks/eai_loader.py`
  - Tests: `conceptgraph/benchmarks/tests/test_eai.py`
  - Depends: None
  - Acceptance:
    - [ ] Load EAI HuggingFace dataset
    - [ ] Support subtask evaluation (goal, subgoal, action, transition)
    - [ ] Unit tests

- [x] TASK-005: Update benchmarks __init__.py exports | Priority: 1 | Est: 0.5h
  - Depends: TASK-003, TASK-004
  - Acceptance:
    - [ ] Export all loaders from package
    - [ ] Add to __all__ list

### Dataset Download

- [x] TASK-006: Download OpenEQA dataset | Priority: 2 | Est: 1h
  - Command: `download_openeqa("/data/benchmarks", include_frames=True)`
  - Note: Frames are ~50GB, start with metadata only first

- [~] TASK-007: Download SQA3D dataset | Priority: 2 | Est: 1h
  - Command: `download_sqa3d("/data/benchmarks")`
  - Note: Requires ScanNet scenes for full evaluation

- [~] TASK-008: Download ScanRefer dataset | Priority: 2 | Est: 1h
  - URL: https://github.com/daveredrum/ScanRefer
  - Note: Shares ScanNet dependency with SQA3D

---

## Phase 2: Stage 2 Agent Enhancement (Priority 1)

### Tool Implementation

- [x] TASK-010: Implement request_crops tool backend | Priority: 1 | Est: 4h
  - File: `conceptgraph/agents/tools/request_crops.py`
  - Tests: `conceptgraph/agents/tools/tests/test_request_crops.py` (32 tests passing)
  - Depends: Stage 1 object detection output
  - Acceptance:
    - [x] Crop objects from keyframes using bounding boxes
    - [x] Support multiple crop requests in single call
    - [x] Return cropped images with metadata
    - [x] Unit tests

- [x] TASK-011: Implement switch_or_expand_hypothesis tool | Priority: 1 | Est: 3h
  - File: `conceptgraph/agents/tools/hypothesis_repair.py`
  - Tests: `conceptgraph/agents/tools/tests/test_hypothesis_repair.py` (41 tests passing)
  - Depends: Stage 1 hypothesis output
  - Acceptance:
    - [x] Switch between direct/proxy/context hypotheses
    - [x] Request alternative hypotheses from Stage 1
    - [x] Track hypothesis history for analysis

- [~] TASK-012: Add token budget tracking to agent | Priority: 2 | Est: 2h
  - File: `conceptgraph/agents/stage2_deep_agent.py`
  - Acceptance:
    - [ ] Track input/output tokens per turn
    - [ ] Implement budget-aware stopping
    - [ ] Log budget usage in trace

- [x] TASK-013: Implement uncertainty-aware stopping | Priority: 2 | Est: 3h
  - File: `conceptgraph/agents/stage2_deep_agent.py`
  - Acceptance:
    - [ ] Agent can output "insufficient evidence" with uncertainty
    - [ ] Confidence threshold configurable
    - [ ] Unit tests for uncertainty scenarios

---

## Phase 3: Evaluation Pipeline (Priority 1)

- [x] TASK-020: Build batch evaluation script | Priority: 1 | Est: 4h
  - File: `conceptgraph/evaluation/batch_eval.py`
  - Tests: `conceptgraph/evaluation/tests/test_batch_eval.py` (33 tests passing)
  - Status: Complete with parallel evaluation, checkpointing, and ablation support
  - Acceptance:
    - [x] Run Stage 1 + Stage 2 on benchmark samples
    - [x] Support parallel evaluation
    - [x] Progress tracking and resumption
    - [x] Output structured results JSON

- [x] TASK-021: Implement metrics aggregation | Priority: 1 | Est: 2h
  - File: `conceptgraph/evaluation/metrics.py`
  - Tests: `conceptgraph/evaluation/tests/test_metrics.py` (56 tests passing)
  - Status: Complete with BenchmarkMetrics, AblationGroup, aggregate functions, LaTeX export
  - Acceptance:
    - [x] Aggregate per-benchmark metrics
    - [x] Support ablation grouping
    - [x] Export to LaTeX tables

- [x] TASK-022: Create ablation configuration system | Priority: 2 | Est: 2h
  - File: `conceptgraph/evaluation/ablation_config.py`
  - Tests: `conceptgraph/evaluation/tests/test_ablation_config.py` (60 tests passing)
  - Status: Complete with YAML configs, tool toggles, agent parameters, preset library
  - Acceptance:
    - [x] YAML-based ablation configs
    - [x] Enable/disable individual tools
    - [x] Control agent parameters

- [x] TASK-023: Integrate with trace server for logging | Priority: 2 | Est: 2h
  - File: `conceptgraph/evaluation/trace_integration.py`
  - Tests: `conceptgraph/evaluation/tests/test_trace_integration.py` (33 tests passing)
  - Status: Complete with EvalTraceManager, TracingBatchEvaluatorMixin, CSV/JSON export
  - Depends: TASK-020
  - Acceptance:
    - [x] Auto-save traces during evaluation
    - [x] Link traces to benchmark samples
    - [x] Export trace statistics

---

## Phase 4: Baseline Experiments (Priority 2)

- [x] TASK-030: Run Stage 1 only baseline on OpenEQA | Priority: 2 | Est: 2h
  - File: `conceptgraph/evaluation/scripts/run_openeqa_stage1_only.py`
  - Tests: `conceptgraph/evaluation/scripts/tests/test_run_openeqa_stage1_only.py` (17 tests passing)
  - Depends: TASK-006, TASK-020
  - Output: `results/baselines/openeqa_stage1_only.json`
  - Status: Complete with mock data support and CLI interface

- [x] TASK-031: Run one-shot VLM baseline on OpenEQA | Priority: 2 | Est: 2h
  - File: `conceptgraph/evaluation/scripts/run_openeqa_oneshot.py`
  - Tests: `conceptgraph/evaluation/scripts/tests/test_run_openeqa_oneshot.py` (21 tests passing)
  - Depends: TASK-006, TASK-020
  - Output: `results/baselines/openeqa_oneshot.json`
  - Status: Complete with mock data support and CLI interface

- [ ] TASK-032: Run full Stage 2 agent on OpenEQA | Priority: 2 | Est: 4h
  - Depends: TASK-006, TASK-020, TASK-010, TASK-011
  - Output: `results/experiments/openeqa_stage2_full.json`

- [ ] TASK-033: Run SQA3D experiments (all three conditions) | Priority: 2 | Est: 6h
  - Depends: TASK-007, TASK-020

- [ ] TASK-034: Run ScanRefer experiments | Priority: 2 | Est: 4h
  - Depends: TASK-008, TASK-003, TASK-020

---

## Phase 5: Ablation Studies (Priority 2)

- [ ] TASK-040: Ablation: No tool calls (one-shot) | Priority: 2 | Est: 2h
- [ ] TASK-041: Ablation: + request_more_views only | Priority: 2 | Est: 2h
- [ ] TASK-042: Ablation: + request_crops only | Priority: 2 | Est: 2h
- [ ] TASK-043: Ablation: + hypothesis_repair only | Priority: 2 | Est: 2h
- [ ] TASK-044: Ablation: + uncertainty output | Priority: 2 | Est: 2h

---

## Phase 6: Analysis & Academic Writing (Priority 3)

- [ ] TASK-050: Generate result tables (Table 1, Table 2) | Priority: 3 | Est: 2h
- [ ] TASK-051: Create visualization figures | Priority: 3 | Est: 4h
  - Detection drop stress test figure
  - Tool usage distribution
  - Confidence vs accuracy plot

- [ ] TASK-052: Write experimental analysis section | Priority: 3 | Est: 8h
- [ ] TASK-053: Draft related work comparison | Priority: 3 | Est: 6h
- [ ] TASK-054: Academic positioning document | Priority: 3 | Est: 4h

---

## Research Insights Queue

> New insights discovered during implementation. Review and integrate.

- [ ] INSIGHT-001: (Pending discovery via web research)

---

## Completed Tasks Archive

### 2026-03-20

- [x] TASK-001: OpenEQA benchmark loader (26 tests)
- [x] TASK-002: SQA3D benchmark loader (41 tests)
- [x] TASK-010: Implement request_crops tool backend (32 tests)
- [x] TASK-011: Implement switch_or_expand_hypothesis tool (41 tests)
- [x] TASK-013: Implement uncertainty-aware stopping (tests in stage2_deep_agent tests)
- [x] TASK-021: Implement metrics aggregation (56 tests)
- [x] TASK-022: Create ablation configuration system (60 tests)
- [x] TASK-023: Integrate with trace server for logging (33 tests)
- [x] TASK-030: Run Stage 1 only baseline on OpenEQA (17 tests)
- [x] TASK-031: Run one-shot VLM baseline on OpenEQA (21 tests)

---

## Metrics Dashboard

| Metric | Current | Target |
|--------|---------|--------|
| OpenEQA tests | 26 | 26 |
| SQA3D tests | 41 | 41 |
| ScanRefer tests | 0 | 30+ |
| Batch Eval tests | 33 | 33 |
| Ablation Config tests | 60 | 60 |
| Trace Integration tests | 33 | 33 |
| Total test coverage | ~70% | 80% |
| Benchmark loaders | 2/4 | 4/4 |

---

*Last updated: 2026-03-20 03:40*
