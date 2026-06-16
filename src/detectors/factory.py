"""
Object detector factory.
"""

def create_detector(name: str, **kwargs):
    if name.startswith("yolo"):
        from .yolo import YOLOdetector
        return YOLOdetector(model_name=name, **kwargs)

    raise ValueError(f"Unknown detector: {name}")
