"""
Object detector base class.
"""

from typing import Protocol
import numpy as np
from common.types import Detection

class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray, frame_idx) -> list[Detection]:
        """Returns detections in MOT bbox format: x, y, w, h."""

    def sequence(seq_info):
        """Hook for MOT16 detector"""
        None
