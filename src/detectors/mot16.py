"""
mot16.py

Return original MOT16 baseline detections accompanying videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathlib import Path
from common import gather_sequence_info, mot_gt_detections_for_frame

from .base import Detection

@dataclass
class MOT16detector:
    """MOT16 "detector"

    Returns pedestrian detections as Detection(x, y, w, h, confidence),
    where x/y/w/h are in MOT format.
    """

    model_name: str | None = None
    min_confidence: float | None = None
    min_detection_height: int = 0

    def __post_init__(self) -> None:
        None

    @property
    def name(self) -> str:
        return "self.model_name"

    def sequence(self, seq_info):
        """Sequence directory and info"""
        self.seq_info = seq_info
        None

    def detect(self, frame, frame_idx) -> list[Detection]:

        detections = self.seq_info["detections"]
        frame_rows = detections[detections[:, 0].astype(int) == frame_idx]

        detections = []
        for row in frame_rows:
            _, track_id, x, y, w, h, confidence, _, _, _ = row[:10]

            if self.min_confidence and confidence < self.min_confidence:
                continue
            if h < self.min_detection_height:
                continue

            det = Detection(
                x=float(x),
                y=float(y),
                w=float(w),
                h=float(h),
                confidence=confidence,
                class_id=1
            )
            detections.append(det)

        return detections
