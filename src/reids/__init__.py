"""
Object detector interfaces and implementations.
"""

from .base import Detection, ReidDetector
from .factory import create_reid_detector

__all__ = [
    "Reid",
    "create_reid",
]