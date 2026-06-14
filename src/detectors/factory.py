"""
Object detector factory.
"""

def create_detector(name: str, **kwargs):
    if name in {"yolov5", "yolov5m"}:
        from .yolov5 import YOLOv5Detector
        return YOLOv5Detector(model_name="yolov5m", **kwargs)

    raise ValueError(f"Unknown detector: {name}")
