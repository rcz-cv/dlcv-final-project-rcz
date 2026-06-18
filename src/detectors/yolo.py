"""
yolo.py

Create detections based on Ultralytics Yolo models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
PERSON_ID = 0

from .base import Detection

@dataclass
class YOLOdetector:
    """YOLO detector.

    Returns pedestrian detections as Detection(x, y, w, h, confidence),
    where x/y/w/h are in MOT format.
    """

    model_name: str = "yolo26m.pt"
    min_confidence: float = 0.25
    iou_threshold: float = 0.45
    min_detection_height: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        model_path = Path(__file__).resolve().parents[2] / "resources" / "networks" / "yolo" / self.model_name
        self.model = YOLO(model_path)

    @property
    def name(self) -> str:
        return self.model_name

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        if bgr_image is None:
            raise ValueError("bgr_image is None")

        detections: list[Detection] = []

        results = self.model(
            source=bgr_image,
            conf = self.min_confidence,
            iou = self.iou_threshold,
            classes=[PERSON_ID],
            device = self.device,
            verbose=False
        )

        for result in results:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), confidence, class_id in zip(
                xyxy, confs, classes
            ):
                if class_id != PERSON_ID:                           # only people
                    continue
                if x2 <= x1 or y2 <= y1:                            # paranoid checking
                    continue
                if self.min_detection_height and y2-y1 < self.min_detection_height:
                    continue

                detections.append(
                    Detection(
                        x=float(x1),
                        y=float(y1),
                        w=float(x2 - x1),
                        h=float(y2 - y1),
                        confidence=float(confidence),
                        class_id=int(class_id),
                    )
                )

            return detections
