"""
Object detector interfaces and implementations.
"""

from .base import Detection, Detector
from .factory import create_detector

__all__ = [
    "Detection",
    "Detector",
    "create_detector",
]