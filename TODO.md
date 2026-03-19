# Research TODO: Task-Conditioned Keyframe Retrieval + Agentic Visual Reasoning

> **Research Narrative**: We propose a two-stage framework for 3D scene understanding that combines task-conditioned keyframe retrieval (Stage 1) with evidence-seeking VLM agents (Stage 2). The key innovation is treating the VLM not as a passive answer generator, but as an adaptive evidence acquisition agent that can validate, repair, and extend symbolic scene graph hypotheses through direct visual inspection.

## Academic Innovation Points

### Innovation A: Adaptive Evidence Acquisition
**Claim**: VLM agents that dynamically request additional visual evidence outperform one-shot top-k baselines under fixed token budgets.

**Mechanism**:
- Stage 2 agent decides when evidence is insufficient
- Can request more views, object crops, or alternative hypotheses
- Budget-aware stopping condition prevents over-acquisition

**Differentiation from Prior Work**:
| Work | Approach | Our Difference |
|------|----------|----------------|
| 3D-LLM (Hong et al. 2023) | One-shot point cloud + text | We use iterative keyframe refinement |
| LEO (Huang et al. 2024) | Embodied agent with egocentric views | We focus on retrieval-augmented reasoning |
| 3DGraphLLM (Zemskova et al. 2024) | Static scene graph to LLM | We allow visual repair of graph errors |
| VAGEN (NeurIPS 2025) | RL for multi-turn VLM agents | We focus on evidence-seeking, not world model |

### Innovation B: Symbolic-to-Visual Repair
**Claim**: Stage 2 agents can detect and correct Stage 1 hypotheses that are invalid in the actual visual evidence, recovering from scene graph detection failures.

**Mechanism**:
- Stage 1 outputs `direct/proxy/context` hypotheses as *soft priors*
- Stage 2 visually verifies each hypothesis in keyframes
- If direct hypothesis fails, agent can switch to proxy or request alternative views

**Research Question**: Does visual repair improve task success rate when object detection has 20-40% miss rate?

### Innovation C: Evidence-Grounded Uncertainty
**Claim**: Agents that output explicit uncertainty and cited evidence have lower hallucination rates than deterministic answerers.

**Mechanism**:
- Mandatory `uncertainties` field in structured response
- `cited_frame_indices` must support each claim
- Agent can output "insufficient evidence" rather than forced answer

### Innovation D: Unified Multi-Task Policy
**Claim**: A single evidence selection policy can support QA, visual grounding, navigation planning, and manipulation planning through task-conditioned payloads.

**Mechanism**:
- Shared agent core with common tools (`request_more_views`, `request_crops`, etc.)
- Task-specific `payload` in unified `Stage2StructuredResponse`
- Enables cross-task transfer evaluation

## Target Benchmarks (Prioritized)

### Tier 1: Primary Evaluation (Must Run)

#### 1. SQA3D - Situated Question Answering in 3D Scenes
- **Source**: CVPR 2023, https://sqa3d.github.io/
- **Tasks**: Embodied QA requiring spatial reasoning
- **Size**: 6.8K situations, 20.4K descriptions, 33.4K questions
- **Base Data**: ScanNet scenes
- **Why**: Direct QA evaluation with situated context; tests spatial reasoning
- **Download**: `git clone https://github.com/SilongYong/SQA3D`

#### 2. ScanRefer - 3D Visual Grounding
- **Source**: ECCV 2020, https://daveredrum.github.io/ScanRefer/
- **Tasks**: Localize objects from natural language in 3D scans
- **Size**: 51,583 descriptions of 11,046 objects from 800 ScanNet scenes
- **Why**: Tests visual grounding capability; has standard evaluation protocol
- **Download**: `git clone https://github.com/daveredrum/ScanRefer`

#### 3. OpenEQA - Embodied Question Answering
- **Source**: CVPR 2024, https://open-eqa.github.io/
- **Tasks**: Open-vocabulary EQA from episodic memory
- **Size**: 1,600+ questions from 180+ real environments
- **Why**: Modern benchmark; direct comparison with GPT-4V baselines
- **Download**: `git clone https://github.com/facebookresearch/open-eqa`

### Tier 2: Extended Evaluation (Should Run)

#### 4. Embodied Agent Interface (EAI)
- **Source**: NeurIPS 2024 (Oral), https://embodied-agent-interface.github.io/
- **Tasks**: Goal interpretation, subgoal decomposition, action sequencing, transition modeling
- **Why**: Diagnostic benchmark decomposing LLM embodied decision-making; enables fine-grained analysis
- **Download**: `pip install eai-eval` + HuggingFace dataset

#### 5. Nr3D/Sr3D - 3D Referring Expressions
- **Source**: ACL 2020
- **Tasks**: 3D object localization from referring expressions
- **Size**: 83.5K human utterances (Nr3D) + 78K synthetic (Sr3D)
- **Why**: Complements ScanRefer with different annotation styles

#### 6. SG-Nav Evaluation
- **Source**: NeurIPS 2024, https://github.com/bagh2178/SG-Nav
- **Tasks**: Zero-shot object navigation using 3D scene graphs + LLM
- **Why**: Direct competitor using scene graphs for navigation; good comparison target

### Tier 3: Stress Tests (Nice to Have)

#### 7. MMScan - Multi-Modal 3D Scene Dataset
- **Source**: NeurIPS 2024
- **Tasks**: 1.4M captions on 109K objects
- **Why**: Large-scale multi-modal test

#### 8. Custom Ablation: Detection Drop Test
- **Design**: Artificially remove 20-40% of object detections from Stage 1
- **Measure**: Stage 2 recovery rate vs. pure scene graph baseline
- **Why**: Directly tests our core claim about visual repair

## Experiment Design

### Main Experiments (Table 1 in paper)

| Method | SQA3D Acc | ScanRefer Acc@0.5 | OpenEQA Score |
|--------|-----------|-------------------|---------------|
| Stage 1 only (scene graph) | - | - | - |
| Stage 1 + one-shot VLM | - | - | - |
| **Stage 1 + Stage 2 Agent (Ours)** | - | - | - |

### Ablation Studies (Table 2)

| Ablation | ΔAccuracy |
|----------|-----------|
| No tool calls (one-shot) | baseline |
| + request_more_views | +? |
| + request_crops | +? |
| + hypothesis repair | +? |
| + uncertainty output | +? |

### Stress Test: Detection Drop (Figure 3)

- X-axis: Detection drop rate (0%, 20%, 40%)
- Y-axis: Task success rate
- Lines: Stage 1 only, Stage 1 + one-shot, Stage 1 + Stage 2 Agent

## Implementation Tasks

### Phase 1: Benchmark Integration (Week 1-2)
- [ ] Download SQA3D dataset
- [ ] Download ScanRefer dataset
- [ ] Download OpenEQA dataset
- [ ] Create unified `conceptgraph/benchmarks/` loader module
- [ ] Write data loading tests for each benchmark
- [ ] Implement evaluation metrics (Accuracy, Acc@0.5, LLM-eval for OpenEQA)

### Phase 2: Stage 2 Agent Enhancement (Week 2-3)
- [ ] Implement `request_crops` tool with real cropping backend
- [ ] Implement `switch_or_expand_hypothesis` tool
- [ ] Add budget tracking (token count, image count)
- [ ] Implement uncertainty-aware stopping
- [ ] Write unit tests for each tool

### Phase 3: Evaluation Pipeline (Week 3-4)
- [ ] Build batch evaluation script
- [ ] Implement result logging with trace server
- [ ] Create ablation configuration system
- [ ] Run baseline experiments
- [ ] Run ablation experiments

### Phase 4: Analysis & Writing (Week 4-5)
- [ ] Generate result tables
- [ ] Create visualization figures
- [ ] Write experimental analysis
- [ ] Draft related work comparison

## Comparison with Latest Related Work

### 3D Scene Understanding + LLM (2024-2025)

| Paper | Venue | Key Idea | Our Difference |
|-------|-------|----------|----------------|
| 3DGraphLLM | ICCV 2025 | Learnable scene graph repr for LLM | We use raw keyframes, not learned repr |
| VAGEN | NeurIPS 2025 | RL for multi-turn VLM world model | We focus on evidence acquisition, not RL |
| SG-Nav | NeurIPS 2024 | Scene graph prompting for navigation | We do multiple tasks, not just navigation |
| LEO | 2024 | Embodied multimodal generalist | We emphasize retrieval + repair |
| EAI | NeurIPS 2024 | Diagnostic benchmark for embodied LLM | We target this benchmark |
| ConceptGraphs | ICRA 2024 | Open-vocab 3D scene graphs | Our Stage 1 builds on this |

### Keyframe Selection (2024-2025)

| Paper | Venue | Key Idea | Our Difference |
|-------|-------|----------|----------------|
| K-frames | 2025 | Scene-driven any-k selection | We do task-conditioned selection |
| Logic-in-Frames | NeurIPS 2025 | Semantic-logical verification | We focus on agent-driven refinement |
| AKS | CVPR 2025 | Adaptive keyframe sampling | We integrate with agentic reasoning |

## Code Quality Requirements

- [ ] All code formatted with `black`
- [ ] All functions have unit tests (pytest)
- [ ] Type hints for all public APIs
- [ ] Docstrings following Google style
- [ ] No hardcoded paths (use environment variables)

## Files to Create/Modify

```
conceptgraph/
├── benchmarks/                    # NEW: Benchmark loaders
│   ├── __init__.py
│   ├── sqa3d_loader.py
│   ├── scanrefer_loader.py
│   ├── openeqa_loader.py
│   ├── eai_loader.py
│   └── tests/
│       ├── test_sqa3d.py
│       ├── test_scanrefer.py
│       └── test_openeqa.py
├── agents/
│   ├── stage2_deep_agent.py      # MODIFY: Add budget tracking, tool enhancements
│   ├── tools/                     # NEW: Modular tools
│   │   ├── __init__.py
│   │   ├── request_crops.py
│   │   ├── hypothesis_repair.py
│   │   └── tests/
│   └── evaluation/                # NEW: Evaluation pipeline
│       ├── __init__.py
│       ├── batch_eval.py
│       ├── metrics.py
│       └── ablation_config.py
└── experiments/                   # NEW: Experiment scripts
    ├── run_sqa3d_eval.py
    ├── run_scanrefer_eval.py
    ├── run_openeqa_eval.py
    └── run_ablation.py
```

## Success Criteria

1. **Quantitative**: Stage 2 agent outperforms one-shot VLM baseline on at least 2/3 primary benchmarks
2. **Ablation**: Each tool (more_views, crops, hypothesis_repair) shows positive delta
3. **Stress Test**: Stage 2 shows >20% relative improvement when detection drop is 30%
4. **Code Quality**: 80%+ test coverage, all tests passing, black formatted

---

*Last updated: 2026-03-20*
