import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "external" / "deep-person-reid"))

from torchreid.utils import FeatureExtractor
from torch import cuda,backends

device = (
    "cuda" if cuda.is_available()
    else "mps" if backends.mps.is_available()
    else "cpu"
)

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="resources/networks/osnet/osnet_x1_0_market1501.pth",
    device=device,
)

print("ok")
