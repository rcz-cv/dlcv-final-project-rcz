"""
yolov5.py

Preprocessor to create new detections based on Yolo v5 and its variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from .base import Detection

COCO_PERSON_CLASS_ID = 0

@dataclass
class YOLOv5Detector:
    """YOLOv5 detector adapter using torch.hub.load.

    Returns pedestrian detections as Detection(x, y, w, h, confidence),
    where x/y/w/h are in MOT format.
    """

    model_name: str = "yolov5m"
    yolov5_dir: str = "external/yolov5"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str | None = None
    force_reload: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.model = torch.hub.load(
            self.yolov5_dir,
            self.model_name,
            source="local",
            pretrained=True,
            force_reload=self.force_reload,
        )

        self.model.to(self.device)
        self.model.eval()

        # thresholds from YOLOv5 AutoShape model
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
        self.model.classes = [COCO_PERSON_CLASS_ID]

    @property
    def name(self) -> str:
        return self.model_name

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        if bgr_image is None:
            raise ValueError("bgr_image is None")

        # torch.hub YOLOv5 expects RGB images.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = self.model(rgb_image)

        # results.xyxy[0]: tensor with columns:
        # x1, y1, x2, y2, confidence, class_id
        predictions = results.xyxy[0]

        if hasattr(predictions, "detach"):
            predictions = predictions.detach().cpu().numpy()

        detections: list[Detection] = []

        height, width = bgr_image.shape[:2]

        for x1, y1, x2, y2, confidence, class_id in predictions:
            if int(class_id) != COCO_PERSON_CLASS_ID:
                continue

            if float(confidence) < self.confidence_threshold:
                continue

            # Clamp to image boundaries.
            x1 = max(0.0, min(float(x1), width - 1.0))
            y1 = max(0.0, min(float(y1), height - 1.0))
            x2 = max(0.0, min(float(x2), width - 1.0))
            y2 = max(0.0, min(float(y2), height - 1.0))

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=box_w,
                    h=box_h,
                    confidence=float(confidence),
                    class_id=COCO_PERSON_CLASS_ID,
                )
            )

        return detections
