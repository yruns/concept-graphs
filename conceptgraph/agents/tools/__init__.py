"""Stage 2 agent tool backends.

This package provides backend implementations for the Stage-2 evidence tools.
Each tool module implements callbacks that can be injected into the agent
to provide real evidence acquisition from the scene.
"""

from .request_crops import (
    CropRequest,
    CropResult,
    CropBackend,
    create_crop_callback,
)

__all__ = [
    "CropRequest",
    "CropResult",
    "CropBackend",
    "create_crop_callback",
]
