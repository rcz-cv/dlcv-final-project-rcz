"""
ReID detector base class.
"""

from typing import Protocol
import numpy as np
from common.types import Detection


class ReidDetector(Protocol):
    name: str

    def reid(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """Populate the feature field of each detection using ReID.

        Args:
            image: Source image containing the detections.
            detections: Detections in MOT (tlwh) format.

        Returns:
            The detections with feature embeddings attached.
        """
