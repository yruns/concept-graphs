# 3DVLMReasoning Migration Tasks

> Migration tasks from concept-graphs to 3DVLMReasoning repository.
> Run via `./auto-claude.sh --execute` for one-click migration.

## Task Status Legend
- `[ ]` Pending
- `[~]` In Progress
- `[x]` Completed
- `[!]` Blocked

---

## Phase 1: Cleanup Unused Code (Est: 4h)

- [ ] TASK-100: Remove VLMClient from vlm_interface.py | Est: 1h
  - Target: `3DVLMReasoning/src/query_scene/vlm_interface.py`
  - Actions:
    - Delete VLMClient class (lines 430-494 in original)
    - Remove `requests` import
    - Keep: VLMInput, STRATEGY_MAP, VLMInputConstructor, VLMOutputParser
    - Update __init__.py exports
  - Acceptance:
    - [ ] No VLMClient class in file
    - [ ] All tests pass
    - [ ] No unused imports

- [ ] TASK-101: Remove deprecated query strategies | Est: 2h
  - Target: `3DVLMReasoning/src/query_scene/`
  - Actions:
    - Audit query_pipeline.py for deprecated VLM paths
    - Remove any llm_evaluator_v1.py if exists
    - Clean up unused strategy code
  - Acceptance:
    - [ ] No deprecated strategy code
    - [ ] Pipeline still functional

- [ ] TASK-102: Clean up test fixtures | Est: 1h
  - Target: `3DVLMReasoning/tests/`
  - Actions:
    - Remove fixtures referencing VLMClient
    - Update test imports
    - Verify test suite passes
  - Acceptance:
    - [ ] All tests pass
    - [ ] No orphaned fixtures

---

## Phase 2: Migrate Missing Modules (Est: 8h)

### Evaluation Module

- [ ] TASK-110: Migrate evaluation module core | Est: 2h
  - Source: `conceptgraph/evaluation/*.py` (core files)
  - Target: `3DVLMReasoning/src/evaluation/`
  - Files: batch_eval.py, metrics.py, ablation_config.py, trace_integration.py
  - Acceptance:
    - [ ] All core files migrated
    - [ ] Import paths updated (conceptgraph → relative)

- [ ] TASK-111: Migrate evaluation scripts | Est: 2h
  - Source: `conceptgraph/evaluation/scripts/`
  - Target: `3DVLMReasoning/src/evaluation/scripts/`
  - Files: run_openeqa_*.py, run_sqa3d_*.py, run_scanrefer_*.py
  - Acceptance:
    - [ ] All scripts migrated
    - [ ] CLI interfaces functional

- [ ] TASK-112: Migrate evaluation ablations | Est: 1h
  - Source: `conceptgraph/evaluation/ablations/`
  - Target: `3DVLMReasoning/src/evaluation/ablations/`
  - Files: run_*_ablation.py
  - Acceptance:
    - [ ] All ablation scripts migrated

- [ ] TASK-113: Migrate evaluation tests | Est: 1h
  - Source: `conceptgraph/evaluation/tests/`
  - Target: `3DVLMReasoning/tests/evaluation/`
  - Acceptance:
    - [ ] All tests migrated and passing

- [ ] TASK-114: Create evaluation __init__.py | Est: 0.5h
  - Target: `3DVLMReasoning/src/evaluation/__init__.py`
  - Acceptance:
    - [ ] All public APIs exported

### Dataset Module

- [ ] TASK-120: Migrate dataset module | Est: 1h
  - Source: `conceptgraph/dataset/`
  - Target: `3DVLMReasoning/src/dataset/`
  - Files: replica_constants.py, datasets_common.py, preprocess_r3d_file.py, save_record3d.py
  - Acceptance:
    - [ ] All files migrated
    - [ ] Import paths updated

- [ ] TASK-121: Create dataset __init__.py | Est: 0.5h
  - Target: `3DVLMReasoning/src/dataset/__init__.py`
  - Acceptance:
    - [ ] Module exports configured

### Scripts Module

- [ ] TASK-130: Migrate critical scripts | Est: 1h
  - Source: `conceptgraph/scripts/`
  - Target: `3DVLMReasoning/src/scripts/`
  - Files:
    - build_visibility_index.py
    - build_open_world_samples.py
    - build_multibenchmark_scene_manifest.py
    - prepare_openeqa_scannet_scene.py
    - validate_scene_graph.py
  - Acceptance:
    - [ ] Critical scripts migrated
    - [ ] CLI functional

- [ ] TASK-131: Migrate ScanNet processing scripts | Est: 1h
  - Source: `conceptgraph/scripts/scannet_process/`
  - Target: `3DVLMReasoning/src/scripts/scannet_process/`
  - Acceptance:
    - [ ] ScanNet processing pipeline available

---

## Phase 3: Architectural Rewrite (Est: 12h)

### Dataset Abstraction

- [ ] TASK-200: Create dataset base abstractions | Est: 2h
  - Target: `3DVLMReasoning/src/dataset/base.py`
  - Classes: SceneMetadata, FrameData, DatasetAdapter ABC
  - Acceptance:
    - [ ] ABC defines: get_scene_ids, load_scene_metadata, iter_frames, load_frame, get_coordinate_transform
    - [ ] Type hints complete
    - [ ] Docstrings complete

- [ ] TASK-201: Create dataset registry | Est: 1h
  - Target: `3DVLMReasoning/src/dataset/registry.py`
  - Functions: register_dataset decorator, get_dataset_adapter, list_datasets
  - Acceptance:
    - [ ] Registry pattern functional
    - [ ] Factory function works

- [ ] TASK-202: Implement ReplicaAdapter | Est: 2h
  - Target: `3DVLMReasoning/src/dataset/replica_adapter.py`
  - Acceptance:
    - [ ] Implements DatasetAdapter ABC
    - [ ] Registered via @register_dataset("replica")
    - [ ] All Replica scenes accessible

- [ ] TASK-203: Implement ScanNetAdapter | Est: 2h
  - Target: `3DVLMReasoning/src/dataset/scannet_adapter.py`
  - Acceptance:
    - [ ] Implements DatasetAdapter ABC
    - [ ] Registered via @register_dataset("scannet")
    - [ ] ScanNet scene loading works

- [ ] TASK-204: Add dataset configuration | Est: 1h
  - Target: `3DVLMReasoning/src/config/datasets.yaml`
  - Config: root paths, coordinate systems, frame strides
  - Acceptance:
    - [ ] YAML config loads correctly
    - [ ] Environment variable overrides work

### Query Scene Refactor

- [ ] TASK-210: Refactor query_scene core | Est: 2h
  - Target: `3DVLMReasoning/src/query_scene/core/`
  - Files: query_types.py, hypotheses.py, results.py
  - Acceptance:
    - [ ] Core abstractions in dedicated subpackage
    - [ ] Clean imports

- [ ] TASK-211: Refactor query parsing | Est: 1h
  - Target: `3DVLMReasoning/src/query_scene/parsing/`
  - Files: parser.py, structures.py
  - Acceptance:
    - [ ] Parser in dedicated subpackage
    - [ ] AST-like structures defined

- [ ] TASK-212: Refactor keyframe retrieval | Est: 1h
  - Target: `3DVLMReasoning/src/query_scene/retrieval/`
  - Files: keyframe_selector.py, visibility_index.py, spatial_index.py
  - Acceptance:
    - [ ] Retrieval logic separated
    - [ ] Clean interface

### Agents Refactor

- [ ] TASK-220: Refactor agents core | Est: 2h
  - Target: `3DVLMReasoning/src/agents/core/`
  - Files: agent_config.py, response_schema.py, task_types.py
  - Acceptance:
    - [ ] Core config in dedicated subpackage
    - [ ] Type definitions clean

- [ ] TASK-221: Create benchmark adapter base | Est: 1h
  - Target: `3DVLMReasoning/src/agents/adapters/base.py`
  - Class: BenchmarkAdapter ABC
  - Acceptance:
    - [ ] ABC defines benchmark interface
    - [ ] Adapter pattern ready

- [ ] TASK-222: Refactor agent runtime | Est: 1h
  - Target: `3DVLMReasoning/src/agents/runtime/`
  - Files: langchain_agent.py, deepagents_agent.py
  - Acceptance:
    - [ ] Runtime implementations separated
    - [ ] Clean inheritance

---

## Phase 4: Multi-Dataset Support (Est: 8h)

- [ ] TASK-300: Integrate adapters with pipeline | Est: 4h
  - Target: `3DVLMReasoning/src/query_scene/query_pipeline.py`
  - Acceptance:
    - [ ] Pipeline accepts dataset adapter
    - [ ] Coordinate transforms applied
    - [ ] Scene loading uses adapters

- [ ] TASK-301: Test Replica end-to-end | Est: 2h
  - Test: Run full pipeline on Replica room0
  - Acceptance:
    - [ ] Pipeline completes without errors
    - [ ] Results match pre-migration behavior

- [ ] TASK-302: Test ScanNet end-to-end | Est: 2h
  - Test: Run full pipeline on ScanNet scene
  - Acceptance:
    - [ ] Pipeline works with ScanNet data
    - [ ] Coordinate differences handled

---

## Phase 5: Equivalence Testing Framework (Est: 10h)

- [ ] TASK-400: Create ground truth generation script | Est: 2h
  - Target: `3DVLMReasoning/scripts/generate_migration_test_data.py`
  - Captures:
    - Stage 1 keyframe selection for 50 queries
    - Query parsing outputs for 100 queries
    - Hypothesis generation for various query types
  - Acceptance:
    - [ ] Script generates reproducible ground truth

- [ ] TASK-401: Generate ground truth data | Est: 1h
  - Run: `python scripts/generate_migration_test_data.py`
  - Output: `tests/migration/ground_truth/`
  - Acceptance:
    - [ ] Ground truth files created
    - [ ] Data validated

- [ ] TASK-410: Create keyframe equivalence tests | Est: 2h
  - Target: `3DVLMReasoning/tests/migration/test_stage1_equivalence.py`
  - Tests:
    - Simple query keyframes (exact match)
    - Spatial query keyframes (80% overlap)
  - Acceptance:
    - [ ] Tests cover major query types
    - [ ] Tolerance levels appropriate

- [ ] TASK-411: Create parsing equivalence tests | Est: 2h
  - Target: `3DVLMReasoning/tests/migration/test_parsing_equivalence.py`
  - Tests:
    - HypothesisOutputV1 structure match
    - Query type classification
  - Acceptance:
    - [ ] Parsing behavior preserved

- [ ] TASK-412: Create integration equivalence tests | Est: 2h
  - Target: `3DVLMReasoning/tests/migration/test_integration_equivalence.py`
  - Tests:
    - End-to-end pipeline results
    - Tool outputs
  - Acceptance:
    - [ ] Integration tests pass

- [ ] TASK-420: Run full equivalence suite | Est: 1h
  - Run: `pytest tests/migration/ -v`
  - Acceptance:
    - [ ] >= 95% match rate
    - [ ] All critical paths pass

---

## Phase 6: Integration Validation (Est: 4h)

- [ ] TASK-500: Run migration scorecard | Est: 2h
  - Target: `3DVLMReasoning/scripts/run_migration_scorecard.py`
  - Outputs:
    - Stage 1 retrieval recall@k
    - Query parsing accuracy
    - End-to-end success rate
    - Performance metrics
  - Acceptance:
    - [ ] All metrics within tolerance

- [ ] TASK-501: Generate final report | Est: 1h
  - Output: `3DVLMReasoning/docs/migration_report.md`
  - Contents:
    - Before/after comparison
    - Test results
    - Performance benchmarks
  - Acceptance:
    - [ ] Report generated
    - [ ] All sections complete

- [ ] TASK-502: Update documentation | Est: 1h
  - Files:
    - README.md
    - CONTRIBUTING.md
    - API documentation
  - Acceptance:
    - [ ] No conceptgraph references
    - [ ] Import paths updated
    - [ ] Examples work

---

## Success Criteria

### Functional
- [ ] All 128+ existing tests pass
- [ ] Equivalence tests >= 95% match rate
- [ ] End-to-end works for Replica
- [ ] End-to-end works for ScanNet
- [ ] OpenEQA, SQA3D, ScanRefer benchmarks run

### Code Quality
- [ ] 100% black formatted
- [ ] No conceptgraph imports remain
- [ ] Type hints on all public APIs
- [ ] Docstrings on all public functions
- [ ] Zero ruff linting errors

### Performance
- [ ] Pipeline latency within 10% of baseline
- [ ] Memory usage within 20% of baseline

---

## Metrics Dashboard

| Phase | Tasks | Completed |
|-------|-------|-----------|
| Phase 1: Cleanup | 3 | 0 |
| Phase 2: Migration | 8 | 0 |
| Phase 3: Architecture | 10 | 0 |
| Phase 4: Multi-dataset | 3 | 0 |
| Phase 5: Testing | 6 | 0 |
| Phase 6: Validation | 3 | 0 |
| **Total** | **33** | **0** |

---

*Generated: 2026-03-22*
