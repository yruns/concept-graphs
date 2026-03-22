# 3DVLMReasoning Migration Plan

> **Objective**: Complete, polished migration from concept-graphs with architectural improvements, multi-dataset support, and verified equivalence.

## Migration Overview

```
Source: /Users/bytedance/project/concept-graphs/conceptgraph/
Target: /Users/bytedance/project/3DVLMReasoning/src/

Migration Phases:
  Phase 1: Cleanup unused code (VLMClient, deprecated strategies)
  Phase 2: Migrate missing modules (evaluation, dataset, scripts)
  Phase 3: Architectural rewrite with elegance
  Phase 4: Multi-dataset abstraction layer
  Phase 5: Equivalence testing framework
  Phase 6: Integration validation
```

---

## Phase 1: Cleanup Unused Code (Est: 4h)

### 1.1 Remove VLMClient from vlm_interface.py

**File**: `src/query_scene/vlm_interface.py`

**Why Remove**: VLMClient is a legacy HTTP client for Ollama/local VLMs. The current architecture uses:
- `utils/llm_client.py` for Azure OpenAI / Gemini
- DeepAgents runtime for Stage 2 agents

**Actions**:
- [ ] Delete `VLMClient` class (lines 430-494)
- [ ] Keep: `VLMInput`, `STRATEGY_MAP`, `VLMInputConstructor`, `VLMOutputParser`
- [ ] Update `__init__.py` exports
- [ ] Run tests to verify no breakage

### 1.2 Remove Deprecated Query Strategies

**Files to examine**:
- `src/query_scene/query_pipeline.py` - check for deprecated paths
- `src/query_scene/llm_evaluator_v2.py` - if v1 exists, remove

### 1.3 Remove Dead Test Fixtures

- Scan `tests/` for fixtures referencing removed code
- Update test imports

---

## Phase 2: Migrate Missing Modules (Est: 8h)

### 2.1 Evaluation Module (Critical)

**Source**: `conceptgraph/evaluation/` (53 files)
**Target**: `src/evaluation/`

**Structure**:
```
src/evaluation/
├── __init__.py           # Export all public APIs
├── batch_eval.py         # BatchEvalConfig, BatchEvaluator
├── metrics.py            # BenchmarkMetrics, AggregatedResults
├── ablation_config.py    # AblationConfig, preset configs
├── trace_integration.py  # EvalTraceManager, TracingBatchEvaluatorMixin
├── result_tables.py      # MethodResult, PaperResults
├── visualizations.py     # Figure generators
├── experimental_analysis.py
├── related_work.py
├── academic_positioning.py
├── scripts/              # Benchmark evaluation scripts
│   ├── run_openeqa_*.py
│   ├── run_sqa3d_*.py
│   └── run_scanrefer_*.py
├── ablations/            # Ablation study scripts
│   ├── run_oneshot_ablation.py
│   └── run_*_ablation.py
└── tests/                # Comprehensive tests
```

**Import transforms**:
- `from conceptgraph.evaluation import X` → `from evaluation import X`
- `from conceptgraph.benchmarks import X` → `from benchmarks import X`
- `from conceptgraph.agents import X` → `from agents import X`

### 2.2 Dataset Module

**Source**: `conceptgraph/dataset/` (5 files)
**Target**: `src/dataset/`

**Files**:
- `replica_constants.py` - Replica scene configurations
- `datasets_common.py` - Shared dataset utilities
- `preprocess_r3d_file.py` - Record3D preprocessing
- `save_record3d.py` - Record3D saving utilities
- `__init__.py` - Module exports

### 2.3 Scripts Module (Partial)

**Source**: `conceptgraph/scripts/` (24 files)
**Target**: `src/scripts/` (selective migration)

**Critical scripts to migrate**:
- `build_visibility_index.py` - Required for Stage 1
- `build_open_world_samples.py` - Sample generation
- `build_multibenchmark_scene_manifest.py` - Multi-benchmark support
- `prepare_openeqa_scannet_scene.py` - OpenEQA preprocessing
- `prepare_scannet_replica_scene.py` - Cross-dataset preparation
- `validate_scene_graph.py` - Validation utilities
- `scannet_process/` - ScanNet data processing

**Skip** (visualization/legacy):
- `animate_mapping_*.py`
- `visualize_cfslam_*.py`
- AI2THOR generation (out of scope)

---

## Phase 3: Architectural Rewrite (Est: 12h)

### 3.1 Unified Dataset Abstraction

**Goal**: Single interface for Replica, ScanNet, and future datasets.

**New file**: `src/dataset/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import numpy as np

@dataclass
class SceneMetadata:
    """Dataset-agnostic scene metadata."""
    scene_id: str
    dataset_name: str
    num_frames: int
    has_depth: bool
    has_poses: bool
    has_mesh: bool
    coordinate_system: str  # "replica" | "scannet" | "custom"

@dataclass
class FrameData:
    """Single frame data with optional fields."""
    frame_id: int
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # 4x4 camera-to-world
    intrinsics: Optional[np.ndarray] = None  # 3x3

class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @abstractmethod
    def get_scene_ids(self) -> List[str]:
        """Return all available scene IDs."""
        pass

    @abstractmethod
    def load_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """Load metadata for a scene."""
        pass

    @abstractmethod
    def iter_frames(self, scene_id: str, stride: int = 1) -> Iterator[FrameData]:
        """Iterate over frames in a scene."""
        pass

    @abstractmethod
    def load_frame(self, scene_id: str, frame_id: int) -> FrameData:
        """Load a specific frame."""
        pass

    @abstractmethod
    def get_coordinate_transform(self) -> np.ndarray:
        """Return 4x4 transform to canonical coordinate system."""
        pass
```

### 3.2 Registry Pattern for Datasets

**New file**: `src/dataset/registry.py`

```python
from typing import Dict, Type
from .base import DatasetAdapter

_DATASET_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}

def register_dataset(name: str):
    """Decorator to register a dataset adapter."""
    def decorator(cls: Type[DatasetAdapter]):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset_adapter(name: str, root_path: str) -> DatasetAdapter:
    """Factory function to get dataset adapter by name."""
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name](root_path)

def list_datasets() -> List[str]:
    """List all registered datasets."""
    return list(_DATASET_REGISTRY.keys())
```

### 3.3 Concrete Adapters

**New files**:
- `src/dataset/replica_adapter.py` - Replica dataset adapter
- `src/dataset/scannet_adapter.py` - ScanNet dataset adapter

### 3.4 Clean Stage 1 Architecture

**Goal**: Separate concerns cleanly.

```
src/query_scene/
├── core/                    # Core abstractions
│   ├── __init__.py
│   ├── query_types.py       # QueryType, QueryInfo (from data_structures.py)
│   ├── hypotheses.py        # HypothesisKind, QueryHypothesis, HypothesisOutputV1
│   └── results.py           # GroundingResult, ExecutionResult
├── parsing/                 # Query parsing
│   ├── __init__.py
│   ├── parser.py            # QueryParser (clean version)
│   └── structures.py        # AST-like query structures
├── retrieval/               # Keyframe retrieval
│   ├── __init__.py
│   ├── keyframe_selector.py # KeyframeSelector (refactored)
│   ├── visibility_index.py  # VisibilityIndex
│   └── spatial_index.py     # SpatialIndex
├── execution/               # Query execution
│   ├── __init__.py
│   ├── executor.py          # QueryExecutor
│   └── spatial_ops.py       # SpatialRelationChecker
└── deprecated/              # Deprecated code (for reference)
    └── vlm_client.py        # Moved here, not imported
```

### 3.5 Clean Stage 2 Architecture

**Goal**: Simpler agent structure.

```
src/agents/
├── core/                    # Core agent abstractions
│   ├── __init__.py
│   ├── agent_config.py      # AgentConfig, ModelConfig
│   ├── response_schema.py   # Stage2StructuredResponse
│   └── task_types.py        # TaskType enum
├── tools/                   # Agent tools (existing, clean)
│   ├── __init__.py
│   ├── request_crops.py
│   ├── request_more_views.py
│   ├── hypothesis_repair.py
│   └── inspect_metadata.py
├── adapters/                # Benchmark adapters
│   ├── __init__.py
│   ├── base.py              # BenchmarkAdapter ABC
│   ├── openeqa_adapter.py
│   ├── sqa3d_adapter.py
│   └── scanrefer_adapter.py
├── runtime/                 # Agent runtime
│   ├── __init__.py
│   ├── langchain_agent.py   # LangChain-based agent
│   └── deepagents_agent.py  # DeepAgents-based agent
└── stage2_agent.py          # Main entry point (simplified)
```

---

## Phase 4: Multi-Dataset Support (Est: 8h)

### 4.1 Replica Adapter Implementation

```python
@register_dataset("replica")
class ReplicaAdapter(DatasetAdapter):
    """Adapter for Replica dataset."""

    SCENES = ["room0", "room1", "room2", "office0", ...]
    DEFAULT_STRIDE = 5

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self._validate_structure()
```

### 4.2 ScanNet Adapter Implementation

```python
@register_dataset("scannet")
class ScanNetAdapter(DatasetAdapter):
    """Adapter for ScanNet dataset."""

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self._sensor_data_cache = {}
```

### 4.3 Unified Configuration

**New file**: `src/config/datasets.yaml`

```yaml
datasets:
  replica:
    root_env: REPLICA_ROOT
    default_root: ~/Datasets/Replica
    coordinate_system: replica
    frame_stride: 5

  scannet:
    root_env: SCANNET_ROOT
    default_root: ~/Datasets/ScanNet
    coordinate_system: scannet
    frame_stride: 20

  openeqa:
    root_env: OPENEQA_ROOT
    default_root: ~/Datasets/OpenEQA
    coordinate_system: varies  # mixed sources
```

---

## Phase 5: Equivalence Testing Framework (Est: 10h)

### 5.1 Test Data Generation

**Pre-migration**: Generate ground truth outputs from concept-graphs.

```python
# scripts/generate_migration_test_data.py
"""
Run before migration to capture ground truth outputs.

Captures:
1. Stage 1 keyframe selection for 50 queries on room0
2. Query parsing outputs for 100 diverse queries
3. Hypothesis generation for spatial/counting/simple queries
4. Visibility index entries for 3 scenes
"""
```

### 5.2 Equivalence Test Suite

**New file**: `tests/migration/test_stage1_equivalence.py`

```python
"""
Stage 1 equivalence tests.

Compares 3DVLMReasoning outputs to pre-captured concept-graphs outputs.
Tolerance: Exact match for deterministic operations, top-k match for retrieval.
"""

import pytest
from pathlib import Path
import json

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"

class TestKeyframeEquivalence:
    """Test keyframe selection equivalence."""

    @pytest.fixture
    def ground_truth_keyframes(self):
        return json.loads((GROUND_TRUTH_DIR / "keyframes.json").read_text())

    def test_simple_query_keyframes(self, ground_truth_keyframes):
        """Simple queries should return identical keyframes."""
        for case in ground_truth_keyframes["simple_queries"]:
            result = select_keyframes(case["query"], case["scene"])
            assert result.frame_ids == case["expected_frame_ids"]

    def test_spatial_query_keyframes(self, ground_truth_keyframes):
        """Spatial queries should return same top-k with tolerance."""
        for case in ground_truth_keyframes["spatial_queries"]:
            result = select_keyframes(case["query"], case["scene"])
            overlap = len(set(result.frame_ids) & set(case["expected_frame_ids"]))
            assert overlap >= len(case["expected_frame_ids"]) * 0.8
```

### 5.3 Regression Test Suite

**New file**: `tests/migration/test_parsing_equivalence.py`

```python
"""
Query parsing equivalence tests.

Verifies HypothesisOutputV1 structure is identical between implementations.
"""

class TestQueryParsingEquivalence:
    """Test query parsing produces identical hypotheses."""

    def test_hypothesis_structure(self, ground_truth_hypotheses):
        """Hypothesis structure should be identical."""
        for case in ground_truth_hypotheses:
            result = parse_query(case["query"], case["categories"])
            assert result.target == case["expected"]["target"]
            assert result.hypothesis_kind == case["expected"]["hypothesis_kind"]
```

### 5.4 Numerical Comparison Utilities

```python
# tests/migration/utils.py

def compare_arrays(a, b, rtol=1e-5, atol=1e-8):
    """Compare numpy arrays with tolerance."""
    return np.allclose(a, b, rtol=rtol, atol=atol)

def compare_keyframe_sets(a, b, k_tolerance=2):
    """Compare keyframe sets allowing k-tolerance in ranking."""
    # Top-k frames should overlap significantly
    overlap = len(set(a[:k]) & set(b[:k+k_tolerance]))
    return overlap >= k * 0.8
```

---

## Phase 6: Integration Validation (Est: 4h)

### 6.1 End-to-End Test

```python
# tests/integration/test_e2e_pipeline.py

@pytest.mark.integration
class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_replica_room0_simple_query(self, replica_adapter):
        """Test simple query on room0."""
        result = run_pipeline(
            dataset="replica",
            scene="room0",
            query="pillow on the sofa",
            k=3
        )
        assert result.success
        assert len(result.keyframes) == 3

    def test_openeqa_scannet_query(self, openeqa_loader):
        """Test OpenEQA question on ScanNet scene."""
        sample = openeqa_loader.get_sample("...")
        result = run_pipeline(
            dataset="scannet",
            scene=sample.scene_id,
            query=sample.question,
            task_type="qa"
        )
        assert result.answer is not None
```

### 6.2 Benchmark Scorecard

```python
# scripts/run_migration_scorecard.py
"""
Generate migration scorecard comparing pre/post migration metrics.

Outputs:
- Stage 1 retrieval recall@k
- Query parsing accuracy
- End-to-end task success rate
- Performance benchmarks (latency, memory)
"""
```

---

## Execution Order for auto-claude.sh

```bash
# Phase 1: Cleanup (3 tasks)
TASK-100: Remove VLMClient from vlm_interface.py
TASK-101: Remove deprecated query strategies
TASK-102: Clean up test fixtures

# Phase 2: Migration (8 tasks)
TASK-110: Migrate evaluation module core
TASK-111: Migrate evaluation scripts
TASK-112: Migrate evaluation ablations
TASK-113: Migrate evaluation tests
TASK-114: Create evaluation __init__.py
TASK-120: Migrate dataset module
TASK-121: Create dataset __init__.py
TASK-130: Migrate critical scripts
TASK-131: Migrate ScanNet processing scripts

# Phase 3: Architecture (10 tasks)
TASK-200: Create dataset base abstractions
TASK-201: Create dataset registry
TASK-202: Implement ReplicaAdapter
TASK-203: Implement ScanNetAdapter
TASK-204: Add dataset configuration
TASK-210: Refactor query_scene core
TASK-211: Refactor query parsing
TASK-212: Refactor keyframe retrieval
TASK-220: Refactor agents core
TASK-221: Create benchmark adapter base
TASK-222: Refactor agent runtime

# Phase 4: Multi-dataset (3 tasks)
TASK-300: Integrate adapters with pipeline
TASK-301: Test Replica end-to-end
TASK-302: Test ScanNet end-to-end

# Phase 5: Equivalence Testing (6 tasks)
TASK-400: Create ground truth generation script
TASK-401: Generate ground truth data
TASK-410: Create keyframe equivalence tests
TASK-411: Create parsing equivalence tests
TASK-412: Create integration equivalence tests
TASK-420: Run full equivalence suite

# Phase 6: Validation (3 tasks)
TASK-500: Run migration scorecard
TASK-501: Generate final report
TASK-502: Update documentation
```

---

## Success Criteria

### Functional Requirements

- [ ] All 128+ existing tests pass
- [ ] Equivalence tests pass with >= 95% match rate
- [ ] End-to-end pipeline works for Replica
- [ ] End-to-end pipeline works for ScanNet
- [ ] OpenEQA, SQA3D, ScanRefer benchmarks run successfully

### Code Quality Requirements

- [ ] 100% black formatted
- [ ] No conceptgraph imports remain
- [ ] Type hints on all public APIs
- [ ] Docstrings on all public functions
- [ ] Zero ruff linting errors

### Performance Requirements

- [ ] Pipeline latency within 10% of concept-graphs
- [ ] Memory usage within 20% of concept-graphs

---

*Generated: 2026-03-22*
