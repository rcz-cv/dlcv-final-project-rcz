"""
Object detector base class.
"""

from typing import Protocol
import numpy as np
from common.types import Detection

class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Returns detections in MOT bbox format: x, y, w, h."""
