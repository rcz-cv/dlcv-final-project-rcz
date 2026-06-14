"""
Object detector base class.
"""

from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int | None = None

class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Returns detections in MOT bbox format: x, y, w, h."""
