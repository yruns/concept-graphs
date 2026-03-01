# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConceptGraphs is a research project for building open-vocabulary 3D scene graphs from RGB-D sequences. It uses Grounded-SAM for 2D detection/segmentation, CLIP for visual features, and LLM/VLM for scene understanding and querying.

## Environment Setup

```bash
# Create conda environment
conda create -n conceptgraph anaconda python=3.10
conda activate conceptgraph

# Install package in editable mode
pip install -e .

# Configure and load environment variables
source env_vars.bash
```

### Environment Variables

Configure in `env_vars.bash` (auto-generated from template):

| Variable | Description | Default |
|----------|-------------|---------|
| `REPLICA_ROOT` | Replica dataset root path | `/Users/bytedance/Replica` |
| `CG_FOLDER` | Project root (auto-detected) | - |
| `GSA_PATH` | Grounded-Segment-Anything path | `${CG_FOLDER}/Grounded-Segment-Anything` |
| `LLM_MODEL` | LLM for query parsing | `gpt-5.2-2025-12-11` |
| `SCENE_NAME` | Default scene name | `room0` |

For a new machine, just update `REPLICA_ROOT` in `env_vars.bash` and run `source env_vars.bash`.

## Pipeline Commands

The pipeline runs from the `conceptgraph/` directory. Shell scripts in `bashes/` wrap the Python commands.

### Full Pipeline (Replica dataset)

```bash
cd conceptgraph

# 1. Extract 2D segmentation with class-aware detection
bash ../bashes/1b_extract_2d_segmentation_detect.sh room0

# 2. Build 3D object map from segmentation
bash ../bashes/2b_build_3d_object_map_detect.sh room0

# 3. Visualize the object map
python scripts/visualize_cfslam_results.py --result_path $REPLICA_ROOT/room0/pcd_saves/*_post.pkl.gz

# 4-6. Scene graph generation (requires OpenAI API)
bash ../bashes/4b_extract_object_captions_detect.sh room0
bash ../bashes/5b_refine_object_captions_detect.sh room0
bash ../bashes/6_build_scene_graph.sh room0
```

### Query System Testing

```bash
# Run end-to-end query tests
bash bashes/run_e2e_query_test.sh

# Or directly
python conceptgraph/query_scene/examples/e2e_query_test.py
```

## Architecture

### Core Modules (`conceptgraph/`)

- **slam/**: 3D object mapping pipeline
  - `cfslam_pipeline_batch.py`: Main mapping script (Hydra config)
  - `slam_classes.py`: `DetectionList`, `MapObjectList` data structures
  - `mapping.py`: Spatial/visual similarity computation and object merging

- **query_scene/**: Natural language spatial query system
  - `query_parser.py`: LLM-based query parsing to `GroundingQuery` structures
  - `query_executor.py`: Bottom-up recursive query execution
  - `query_pipeline.py`: `QueryScenePipeline` orchestrates parsing/execution
  - `spatial_relations.py`: Geometric relation checking (on, near, above, etc.)
  - `scene_representation.py`: `QuerySceneRepresentation` wraps scene data

- **scenegraph/**: Scene graph construction
  - `build_scenegraph_cfslam.py`: Caption extraction and edge building with LLaVA/GPT

- **dataset/**: Data loading
  - `datasets_common.py`: `get_dataset()` returns dataset loaders (Replica, TUM, ScanNet)

- **llava/**: VLM integration
  - `unified_client.py`, `ollama_adapter.py`: Local VLM inference

- **utils/**: Shared utilities
  - `llm_client.py`: LangChain Azure OpenAI wrapper with model configs

### Key Data Flow

1. RGB-D images + poses → `generate_gsa_results.py` → 2D masks + CLIP features
2. 2D detections → `cfslam_pipeline_batch.py` → 3D objects (`.pkl.gz`)
3. 3D objects → `build_scenegraph_cfslam.py` → Scene graph with captions
4. Scene graph + query → `QueryScenePipeline` → Grounded objects

### Configuration

- Hydra configs in `conceptgraph/hydra_configs/`
- Default SLAM config: `hydra_configs/cfslam_pipeline_batch.yaml`

## Code Patterns

### Loading Scene Data

```python
import gzip, pickle
with gzip.open("path/to/scene_post.pkl.gz", "rb") as f:
    data = pickle.load(f)
objects = data.get("objects", [])  # List of dicts with 'pcd_np', 'clip_ft', 'class_name', etc.
```

### Query Pipeline Usage

```python
from conceptgraph.query_scene.query_pipeline import QueryScenePipeline

pipeline = QueryScenePipeline.from_scene(
    scene_path="/path/to/room0",
    llm_model="gpt-5.2-2025-12-11"
)
result = pipeline.query("the pillow on the sofa near the door")
```

### Spatial Relations

```python
from conceptgraph.query_scene.spatial_relations import SpatialRelationChecker

checker = SpatialRelationChecker(objects)
result = checker.check("on", obj_a, obj_b)  # Returns RelationResult with score
```

## Output Formats

- **3D object map**: `.pkl.gz` containing `{"objects": [...], "bg_objects": [...], ...}`
- **Scene graph**: JSON with node captions and edge relations
- **Visualization**: PLY point clouds in `pcd_saves/ply_export_detect/`
