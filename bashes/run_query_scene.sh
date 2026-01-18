#!/bin/bash
# =============================================================================
# QueryScene: Query-Driven Scene Understanding
# =============================================================================
# 基于查询的3D场景理解系统
#
# Usage:
#   ./run_query_scene.sh [SCENE_NAME] [QUERY]
#
# Examples:
#   ./run_query_scene.sh room0                    # 交互模式
#   ./run_query_scene.sh room0 "lamp"             # 单次查询
#   ./run_query_scene.sh room0 "桌子旁边的椅子"    # 空间关系查询
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate environment
if [ -f "/home/shyue/anaconda3/bin/activate" ]; then
    source /home/shyue/anaconda3/bin/activate conceptgraph
fi

# Load env vars
if [ -f "/home/shyue/codebase/concept-graphs/env_vars.bash" ]; then
    source /home/shyue/codebase/concept-graphs/env_vars.bash
fi

# Parameters
SCENE_NAME="${1:-room0}"
QUERY="${2:-}"
REPLICA_ROOT="${REPLICA_ROOT:-/home/shyue/Datasets/Replica/Replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# LLM settings
LLM_URL="${LLM_BASE_URL:-http://10.21.231.7:8006}"
LLM_MODEL="${LLM_MODEL:-gemini-3-flash-preview}"

# Find pcd file (prefer RAM-tagged version with semantic labels)
PCD_FILE=$(ls -t "${SCENE_PATH}/pcd_saves/"*ram*_post.pkl.gz 2>/dev/null | head -1)
if [ -z "$PCD_FILE" ]; then
    PCD_FILE=$(ls -t "${SCENE_PATH}/pcd_saves/"*_post.pkl.gz 2>/dev/null | head -1)
fi
if [ -z "$PCD_FILE" ]; then
    echo -e "${RED}Error: No pcd file found in ${SCENE_PATH}/pcd_saves/${NC}"
    echo "Please run the ConceptGraphs pipeline first:"
    echo "  bash bashes/1_extract_2d_segmentation.sh"
    echo "  bash bashes/2_build_3d_object_map.sh"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}QueryScene: Query-Driven Scene Understanding${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "Scene: ${SCENE_NAME}"
echo "PCD: $(basename ${PCD_FILE})"
echo "LLM: ${LLM_URL} / ${LLM_MODEL}"
echo -e "${GREEN}============================================================${NC}"
echo ""

cd /home/shyue/codebase/concept-graphs

# Output directory for visualizations
OUTPUT_DIR="${SCENE_PATH}/query_results"
mkdir -p "${OUTPUT_DIR}"

if [ -n "${QUERY}" ]; then
    # Single query mode with visualization
    python -c "
import sys
sys.path.insert(0, '.')

from conceptgraph.query_scene import QueryScenePipeline
from conceptgraph.query_scene.query_pipeline import visualize_result
from conceptgraph.query_scene.utils import format_result

print('Loading scene and building indices...')
pipeline = QueryScenePipeline.from_scene(
    '${SCENE_PATH}',
    '${PCD_FILE}',
    llm_url='${LLM_URL}',
    llm_model='${LLM_MODEL}',
)

print()
print('Objects in scene:', len(pipeline.scene.objects))
print()

query = '''${QUERY}'''
print(f'Query: {query}')
print('-' * 40)

result = pipeline.query(query)
print(format_result(result))

# Generate visualizations
if result.success:
    print()
    print('Generating visualizations...')
    outputs = visualize_result('${PCD_FILE}', result, '${OUTPUT_DIR}')
    print()
    print('Output files:')
    for name, path in outputs.items():
        print(f'  - {name}: {path}')
    print()
    print('Color legend:')
    print('  RED = Target object')
    print('  GREEN = Reference object')
    print('  GRAY = Background')
"
else
    # Interactive mode
    python -c "
import sys
sys.path.insert(0, '.')

from conceptgraph.query_scene import QueryScenePipeline
from conceptgraph.query_scene.utils import format_result
from collections import Counter

print('Loading scene and building indices...')
pipeline = QueryScenePipeline.from_scene(
    '${SCENE_PATH}',
    '${PCD_FILE}',
    llm_url='${LLM_URL}',
    llm_model='${LLM_MODEL}',
)

print()
print('=' * 60)
print(f'Scene loaded: {len(pipeline.scene.objects)} objects')
print()
print('Categories:')
cats = Counter(obj.category for obj in pipeline.scene.objects)
for cat, count in cats.most_common(10):
    print(f'  - {cat}: {count}')
print()
print('Commands:')
print('  <query>    - Query the scene (e.g., \"lamp\", \"桌子旁边的椅子\")')
print('  list       - List all objects')
print('  quit       - Exit')
print('=' * 60)
print()

while True:
    try:
        query = input('QueryScene> ').strip()
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print('Bye!')
            break
        elif query.lower() == 'list':
            for obj in pipeline.scene.objects[:20]:
                pos = obj.centroid
                pos_str = f'({pos[0]:.2f}, {pos[1]:.2f})' if pos is not None else 'N/A'
                print(f'  [{obj.obj_id:2d}] {obj.category:15s} at {pos_str}')
            if len(pipeline.scene.objects) > 20:
                print(f'  ... and {len(pipeline.scene.objects) - 20} more')
        else:
            result = pipeline.query(query)
            print(format_result(result))
    except KeyboardInterrupt:
        print()
        print('Bye!')
        break
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
"
fi
