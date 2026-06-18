"""
ReID detector factory.
"""

def create_reid_detector(name: str, **kwargs):
    if name.startswith("mars"):
        from .mars import MarsReid
        return MarsReid(**kwargs)
    if name.startswith("osnet"):
        from .osnet import OsnetReid
        return OsnetReid(**kwargs, model_name=name)

    raise ValueError(f"Unknown detector: {name}")
