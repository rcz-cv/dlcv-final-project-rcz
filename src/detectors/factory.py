"""
Object detector factory.
"""

def create_detector(name: str, **kwargs):
    if name.startswith("yolo"):
        if not name.endswith(".pt"):
            name += ".pt"

        if name.endswith("-seg.pt"):
            from .yoloseg import YOLOSegDetector
            return YOLOSegDetector(model_name=name, **kwargs)
        else:
            from .yolo import YOLOdetector
            return YOLOdetector(model_name=name, **kwargs)
        
    if name == "mot16":
        from .mot16 import MOT16detector
        return MOT16detector(model_name=name, **kwargs)
    raise ValueError(f"Unknown detector: {name}")
