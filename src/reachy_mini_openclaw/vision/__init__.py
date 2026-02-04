"""Vision modules for face tracking, detection, and image understanding."""

from reachy_mini_openclaw.vision.head_tracker import get_head_tracker

__all__ = [
    "get_head_tracker",
]

# Lazy imports for optional heavy dependencies
def get_vision_processor():
    """Get the VisionProcessor class (requires torch, transformers)."""
    from reachy_mini_openclaw.vision.processors import VisionProcessor
    return VisionProcessor

def get_vision_manager():
    """Get the VisionManager class (requires torch, transformers)."""
    from reachy_mini_openclaw.vision.processors import VisionManager
    return VisionManager
