import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
print("project_root:", str(project_root))

torchreid_root = project_root / "external" / "deep-person-reid"

torchreid_root = Path(torchreid_root)
if str(torchreid_root) not in sys.path:
    sys.path.insert(0, str(torchreid_root))
print("torchreid_root:", str(torchreid_root))

import torchreid
print(torchreid.__file__)


from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    device="cpu",
)

print("ok")

