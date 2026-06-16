from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Detection in MOT / DeepSORT TLWH format.

    x, y are top-left coordinates.
    w, h are width and height.
    feature is the optional ReID appearance embedding.
    """

    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    feature: np.ndarray | None = None

    @property
    def tlwh(self) -> np.ndarray:
        """Return box as (top-left x, top-left y, width, height)."""
        return np.array([self.x, self.y, self.w, self.h], dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        """Alias for this project's box format: top-left x/y + width/height."""
        return self.tlwh

    def to_xyah(self) -> np.ndarray:
        """Return box as (center x, center y, aspect ratio, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret

    def require_feature(self) -> np.ndarray:
        """Return feature, make obvious fail if not present."""
        if self.feature is None:
            raise ValueError("Detection.feature is required but has not been set")
        return self.feature
