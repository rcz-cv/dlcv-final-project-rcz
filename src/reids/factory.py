"""
ReID detector factory.
"""

def create_reid_detector(name: str, **kwargs):
    if name.startswith("mars"):
        from .mars import MarsReid
        return MarsReid(**kwargs)

    raise ValueError(f"Unknown detector: {name}")
