"""
Object detector factory.
"""

def create_detector(name: str, **kwargs):
    if name in {"yolov5", "yolov5m"}:
        from .yolov5 import YOLOv5Detector
        return YOLOv5Detector(model_name="yolov5m", **kwargs)

    if name in {"nanodet-plus-m-416"}:
        from .nanodet import NanoDetDetector
        return NanoDetDetector(model_name=name, **kwargs)

    raise ValueError(f"Unknown detector: {name}")
