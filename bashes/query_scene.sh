#!/bin/bash
# =============================================================================
# Query Scene Interface
# =============================================================================
# 交互式查询场景
#
# Usage:
#   ./query_scene.sh [SCENE_NAME] [QUERY]
#
# Examples:
#   ./query_scene.sh room0 "沙发旁边的台灯"
#   ./query_scene.sh room0 "房间里有几把椅子"
#   ./query_scene.sh room0  # 进入交互模式
# =============================================================================

set -e

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

# File paths
THRESHOLD=1.2
PCD_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

# LLM settings
LLM_URL="${LLM_BASE_URL:-http://10.21.231.7:8006}"
LLM_MODEL="${LLM_MODEL:-gemini-3-flash-preview}"

echo "============================================================"
echo "Scene Query Interface"
echo "============================================================"
echo "Scene: ${SCENE_NAME}"
echo "LLM: ${LLM_URL} / ${LLM_MODEL}"
echo "============================================================"

cd /home/shyue/codebase/concept-graphs

OUTPUT_DIR="${SCENE_PATH}/query_results"
mkdir -p "${OUTPUT_DIR}"

if [ -n "${QUERY}" ]; then
    # Single query mode with visualization
    python -c "
import sys
sys.path.insert(0, '/home/shyue/codebase/concept-graphs')

from implicit_scene.store.vector_store import SceneVectorStore
from implicit_scene.tasks.scene_interface import SceneInterface
from implicit_scene.visualize import visualize_query_result, visualize_topdown
import json

# Load scene
store = SceneVectorStore()
store.load_from_pcd('${PCD_FILE}')

# Create interface
interface = SceneInterface(store, llm_url='${LLM_URL}', llm_model='${LLM_MODEL}')

# Query
query = '''${QUERY}'''
print(f'\nQuery: {query}\n')

# Detect query type
if '?' in query or '几' in query or '什么' in query or '哪' in query:
    # QA mode
    result = interface.answer_question(query)
    print(f\"Answer: {result['answer']}\n\")
else:
    # Grounding mode
    result = interface.ground(query, return_all=True)
    print(f\"Found {len(result['objects'])} objects:\n\")
    for obj in result['objects'][:5]:
        print(f\"  - {obj['tag']} (id={obj['id']}, pos={obj['position'][:2]})\")
    print(f\"\nConfidence: {result['confidence']:.3f}\")
    print(f\"Parsed: target='{result['parsed']['target']}', constraint='{result['parsed']['spatial_constraint']}'\")
    
    # Visualize results
    target_ids = [obj['id'] for obj in result['objects']]
    ref_ids = []
    for ctx in result.get('context', []):
        if ctx['id'] not in target_ids:
            ref_ids.append(ctx['id'])
    
    # 3D点云可视化
    output_ply = '${OUTPUT_DIR}/query_result.ply'
    visualize_query_result(
        '${PCD_FILE}',
        target_ids=target_ids,
        reference_ids=ref_ids,
        output_path=output_ply,
        show_all=True,
    )
    
    # 2D俯视图可视化
    labels = {obj['id']: obj['tag'] for obj in result['objects']}
    labels.update({ctx['id']: ctx['tag'] for ctx in result.get('context', [])})
    
    output_png = '${OUTPUT_DIR}/query_result_topdown.png'
    visualize_topdown(
        '${PCD_FILE}',
        target_ids=target_ids,
        reference_ids=ref_ids,
        output_path=output_png,
        labels=labels,
    )
    
    print(f'\n可视化文件已保存:')
    print(f'  - 3D点云: {output_ply}')
    print(f'  - 2D俯视图: {output_png}')
    print('颜色说明: 红色=目标物体, 绿色=参照物, 灰色=背景')
"
else
    # Interactive mode
    python -c "
import sys
sys.path.insert(0, '/home/shyue/codebase/concept-graphs')

from implicit_scene.store.vector_store import SceneVectorStore
from implicit_scene.tasks.scene_interface import SceneInterface
import json

# Load scene
print('Loading scene...')
store = SceneVectorStore()
store.load_from_pcd('${PCD_FILE}')

# Create interface
interface = SceneInterface(store, llm_url='${LLM_URL}', llm_model='${LLM_MODEL}')

print(f\"\nScene loaded: {store.summary()['n_objects']} objects\")
print('\nCommands:')
print('  <query>      - Query the scene')
print('  list         - List all objects')
print('  summary      - Show scene summary')
print('  quit/exit    - Exit')
print('')

while True:
    try:
        query = input('Query> ').strip()
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        elif query.lower() == 'list':
            for obj in store.objects[:20]:
                print(f'  {obj.id}: {obj.tag} at {obj.position[:2].tolist()}')
            if len(store.objects) > 20:
                print(f'  ... and {len(store.objects)-20} more')
        elif query.lower() == 'summary':
            print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
        else:
            # Query
            if '?' in query or '几' in query or '什么' in query:
                result = interface.answer_question(query)
                print(f\"\nAnswer: {result['answer']}\n\")
            else:
                result = interface.ground(query, return_all=True)
                print(f\"\nFound {len(result['objects'])} objects:\")
                for obj in result['objects'][:5]:
                    print(f\"  - {obj['tag']} (id={obj['id']})\")
                print(f\"Confidence: {result['confidence']:.3f}\n\")
    except KeyboardInterrupt:
        print('\nBye!')
        break
    except Exception as e:
        print(f'Error: {e}')
"
fi
