"""Benchmark loaders for 3D scene understanding evaluation.

Supported benchmarks:
- OpenEQA: Embodied Question Answering (CVPR 2024)
- SQA3D: Situated Question Answering in 3D Scenes (CVPR 2023)
- ScanRefer: 3D Visual Grounding (ECCV 2020)
- EAI: Embodied Agent Interface (NeurIPS 2024)
"""

from .openeqa_loader import OpenEQADataset, OpenEQASample, download_openeqa

__all__ = [
    "OpenEQADataset",
    "OpenEQASample",
    "download_openeqa",
]
