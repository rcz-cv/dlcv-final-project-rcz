# DLCV Week 6 - Final Project

**`README-dlcv-rcz.md`**

## Installation & Preparation

### Python Requirements

```
python3.12 -m venv .venv

# Change PS1 prompt on line ~70 to:
# PS1="(.venv) ${PS1:-}"
vi .venv/bin/activate

source .venv/bin/activate

# Confirm version:
% python --version
Python 3.12.12

pip install --upgrade pip 

pip install numpy opencv-python scipy tensorflow tf-slim tf-keras
```

### Install resources

As of 20260612 the resource files are available on Google Drive [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)

Move the unzipped folders into:

```
resources/
    detections/
    networks/
```

Also copy the MOT16 benchmark data from [Kaggle](https://www.kaggle.com/datasets/takshmandar/mot16-dataset) into:

```
data/
	MOT16/
```

### Smoke Test

The following example runs the tracker against one of the MOT16 sequences:

```
python deep_sort_app.py \
    --sequence_dir=./data/MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```
