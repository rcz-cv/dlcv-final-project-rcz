"""
yoloseg.py

Create detections based on Ultralytics YOLO segmentation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PERSON_ID = 0

from .base import Detection


@dataclass
class YOLOSegDetector:
    """YOLO segmentation detector.

    Returns pedestrian detections as Detection(x, y, w, h, confidence, class_id, mask),
    where x/y/w/h are in MOT format.

    The mask is resized to full-frame image coordinates so downstream ReID can
    crop it using the detection bbox.
    """

    model_name: str | None = None
    min_confidence: float | None = None
    iou_threshold: float | None = None
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

        model_path = (
            Path(__file__).resolve().parents[2]
            / "resources"
            / "networks"
            / "yolo"
            / self.model_name
        )
        self.model = YOLO(model_path)

    @property
    def name(self) -> str:
        return self.model_name

    def detect(self, bgr_image: np.ndarray, frame_idx) -> list[Detection]:
        if bgr_image is None:
            raise ValueError("bgr_image is None")

        detections: list[Detection] = []

        results = self.model(
            source=bgr_image,
            conf=self.min_confidence,
            iou=self.iou_threshold,
            classes=[PERSON_ID],
            device=self.device,
            verbose=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            masks = self._extract_full_frame_masks(result, bgr_image.shape[:2])

            for idx, ((x1, y1, x2, y2), confidence, class_id) in enumerate(
                zip(xyxy, confs, classes)
            ):
                if class_id != PERSON_ID:                           # only people
                    continue
                if x2 <= x1 or y2 <= y1:                            # paranoid checking
                    continue
                if self.min_detection_height and y2 - y1 < self.min_detection_height:
                    continue

                mask = masks[idx] if masks is not None and idx < len(masks) else None

                detections.append(
                    Detection(
                        x=float(x1),
                        y=float(y1),
                        w=float(x2 - x1),
                        h=float(y2 - y1),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        mask=mask,
                    )
                )

        return detections

    def _extract_full_frame_masks(
        self,
        result,
        image_hw: tuple[int, int],
    ) -> list[np.ndarray] | None:
        if result.masks is None:
            return None

        mask_tensor = getattr(result.masks, "data", None)
        if mask_tensor is None:
            return None

        masks_np = mask_tensor.cpu().numpy()
        image_h, image_w = image_hw

        full_frame_masks: list[np.ndarray] = []

        for mask in masks_np:
            mask = mask.astype(np.float32)

            if mask.shape != (image_h, image_w):
                mask = cv2.resize(
                    mask,
                    (image_w, image_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            full_frame_masks.append(mask)

        return full_frame_masks
